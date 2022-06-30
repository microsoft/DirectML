#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "OnnxDispatchable.h"
#include "OnnxParsers.h"

using Microsoft::WRL::ComPtr;

#define THROW_IF_NOT_OK(status) {auto localStatus = (status); if (localStatus) throw E_FAIL;}

template<typename T>
using deleting_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

static Ort::Value CreateTensorFromResource(
    const OrtDmlApi* ortDmlApi,
    Ort::MemoryInfo const& memoryInformation,
    ID3D12Resource* d3dResource,
    gsl::span<const int64_t> tensorDimensions,
    ONNXTensorElementDataType elementDataType,
    void** dmlEpResourceWrapper)
{
    *dmlEpResourceWrapper = nullptr;

    void* dmlAllocatorResource;
    THROW_IF_NOT_OK(ortDmlApi->CreateGPUAllocationFromD3DResource(d3dResource, &dmlAllocatorResource));
    auto deleter = [&](void*) {ortDmlApi->FreeGPUAllocation(dmlAllocatorResource); };
    deleting_unique_ptr<void> dmlAllocatorResourceCleanup(dmlAllocatorResource, deleter);

    size_t tensorByteSize = static_cast<size_t>(d3dResource->GetDesc().Width);
    Ort::Value newValue(
        Ort::Value::CreateTensor(
            memoryInformation,
            dmlAllocatorResource,
            tensorByteSize,
            tensorDimensions.data(),
            tensorDimensions.size(),
            elementDataType
        )
    );
    *dmlEpResourceWrapper = dmlAllocatorResource;
    dmlAllocatorResourceCleanup.release();

    return newValue;
}

static ID3D12Resource* GetResourceFromModelBinding(
    const std::string& tensorName, 
    const Dispatchable::Bindings& bindings)
{
    auto binding = bindings.find(tensorName);
    if (binding == bindings.end())
    {
        throw std::runtime_error(fmt::format("Could not find binding for tensor '%s'", tensorName));
    }
    auto& bindingSources = binding->second;

    if (bindingSources.size() != 1)
    {
        throw std::invalid_argument("ONNX dispatchables' tensors must map to a single binding source.");
    }

    auto& bindingSource = bindingSources[0];

    if (bindingSource.counterResource != nullptr)
    {
        throw std::invalid_argument("ONNX dispatchables do not support counter resources in bindings.");
    }
    
    if (bindingSource.elementOffset != 0)
    {
        throw std::invalid_argument("ONNX dispatchables do not support binding offsets.");
    }

    return bindingSource.resource;
}

OnnxDispatchable::OnnxDispatchable(
    std::shared_ptr<Device> device, 
    const Model::OnnxDispatchableDesc& desc,
    const CommandLineArgs& args
    ) : m_device(device), m_desc(desc), m_args(args)
{
}

void OnnxDispatchable::Initialize()
{
    const OrtApi& ortApi = Ort::GetApi();
    THROW_IF_NOT_OK(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&m_ortDmlApi)));

    Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "DxDispatch"); // Note ORT_LOGGING_LEVEL_VERBOSE is useful too.
    
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.DisableMemPattern();
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); // Note ORT_ENABLE_BASIC is useful for debugging.
 
    for (auto& freeDimOverride : m_args.GetOnnxFreeDimensionOverrides())
    {
        ortApi.AddFreeDimensionOverrideByName(sessionOptions, freeDimOverride.first.c_str(), freeDimOverride.second);
    }

    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProviderEx_DML(sessionOptions, m_device->DML(), m_device->GetCommandQueue()));

    m_session = Ort::Session(ortEnvironment, m_desc.sourcePath.wstring().c_str(), sessionOptions);
    m_ioBindings = Ort::IoBinding::IoBinding(*m_session);
}

void OnnxDispatchable::Bind(const Bindings& bindings)
{
    m_ioBindings->ClearBoundInputs();
    m_ioBindings->ClearBoundOutputs();
    m_tensors.clear();
    m_tensorWrappers.clear();

    Ort::MemoryInfo memoryInformation("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    Ort::Allocator deviceAllocator(*m_session, memoryInformation);

    auto inputCount = m_session->GetInputCount();
    auto outputCount = m_session->GetOutputCount();
    m_tensors.resize(inputCount + outputCount);

    for (int bindingPass = 0; bindingPass < 2; ++bindingPass)
    {
        const bool isInputTensor = (bindingPass == 0);
        const size_t tensorCount = isInputTensor ? inputCount : outputCount;

        for (size_t tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex)
        {
            std::string tensorName = OnnxParsers::GetTensorName(tensorIndex, *m_session, isInputTensor);
            Ort::TypeInfo typeInfo = isInputTensor ? m_session->GetInputTypeInfo(tensorIndex) : m_session->GetOutputTypeInfo(tensorIndex);
            if (typeInfo.GetONNXType() != ONNXType::ONNX_TYPE_TENSOR)
            {
                throw std::runtime_error(fmt::format("Unknown binding type for '%s'", tensorName));
            }

            Ort::Unowned<Ort::TensorTypeAndShapeInfo> shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
            const ONNXTensorElementDataType tensorDataType = shapeInfo.GetElementType();
            if (!OnnxParsers::IsSupportedOnnxTensorElementDataType(tensorDataType))
            {
                throw std::runtime_error("Unsupported tensor data type");
            }

            // Convert free dimensions (-1) to their minimum positive size (1).
            std::vector<int64_t> tensorShape = shapeInfo.GetShape();
            for (auto& dim : tensorShape)
            {
                dim = std::abs(dim);
            }

            auto resource = GetResourceFromModelBinding(tensorName, bindings);

            std::optional<Ort::Value>& tensor = m_tensors[tensorIndex + bindingPass * inputCount];

            // Create an ORT tensor from the existing D3D resource.
            Microsoft::WRL::ComPtr<IUnknown> resourceWrapper;
            tensor = CreateTensorFromResource(
                m_ortDmlApi, 
                memoryInformation, 
                resource,
                tensorShape,
                tensorDataType,
                &resourceWrapper);

            m_tensorWrappers.push_back(std::move(resourceWrapper));

            // Bind the tensor.
            if (isInputTensor)
            {
                m_ioBindings->BindInput(tensorName.c_str(), *tensor);
            }
            else
            {
                m_ioBindings->BindOutput(tensorName.c_str(), *tensor);
            }
        }
    }
}

void OnnxDispatchable::Dispatch(const Model::DispatchCommand& args)
{
    Ort::RunOptions runOptions;
    m_session->Run(runOptions, *m_ioBindings);
    m_ioBindings->SynchronizeOutputs();
}