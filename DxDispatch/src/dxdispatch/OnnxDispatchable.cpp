#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "OnnxDispatchable.h"
#include "OnnxParsers.h"

using Microsoft::WRL::ComPtr;

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
    Ort::ThrowOnError(ortDmlApi->CreateGPUAllocationFromD3DResource(d3dResource, &dmlAllocatorResource));
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
        throw std::runtime_error(fmt::format("Could not find binding for tensor '{}'", tensorName));
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
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&m_ortDmlApi)));

    m_environment = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DxDispatch"); // Note ORT_LOGGING_LEVEL_VERBOSE is useful too.


    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.DisableMemPattern();
    sessionOptions.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(m_args.GetOnnxGraphOptimizationLevel()));
 
    for (auto& freeDimOverride : m_args.GetOnnxFreeDimensionNameOverrides())
    {
        Ort::ThrowOnError(ortApi.AddFreeDimensionOverrideByName(sessionOptions, freeDimOverride.first.c_str(), freeDimOverride.second));
    }

    for (auto& freeDimOverride : m_args.GetOnnxFreeDimensionDenotationOverrides())
    {
        Ort::ThrowOnError(ortApi.AddFreeDimensionOverride(sessionOptions, freeDimOverride.first.c_str(), freeDimOverride.second));
    }

    for (auto& configEntry : m_args.GetOnnxSessionOptionConfigEntries())
    {
        Ort::ThrowOnError(ortApi.AddSessionConfigEntry(sessionOptions, configEntry.first.c_str(), configEntry.second.c_str()));
    }

    const OrtDmlApi* ortDmlApi;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
    Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, m_device->DML(), m_device->GetCommandQueue()));

    m_session = Ort::Session(*m_environment, m_desc.sourcePath.wstring().c_str(), sessionOptions);
    m_ioBindings = Ort::IoBinding::IoBinding(*m_session);
}

void OnnxDispatchable::Bind(const Bindings& bindings)
{
    // This table summarizes the resource bindings provided to the ONNX Runtime session:
    //
    // Kind   | Type       | DXD Binding    | DML Supported Data Type | ORT binding
    // -------|------------|----------------|-------------------------|----------------------------------
    // input  | tensor     | true           | *                       | pre-allocated DX resource
    // input  | tensor     | false          | true                    | explicit DX resource (uninitialized values)
    // input  | tensor     | false          | false                   | explicit CPU resource (uninitialized values)
    // input  | non-tensor | *              | *                       | none
    // output | tensor     | true           | *                       | pre-allocated DX resource
    // output | tensor     | false          | true                    | implicit DX resource
    // output | tensor     | false          | false                   | implicit CPU resource
    // output | non-tensor | *              | *                       | implicit CPU resource
    //
    // - "DXD Binding" refers to the binding specified in a DxDispatch JSON model or created by OnnxParsers::ParseModel. 
    // - "ORT Binding" refers to the final binding passed to the ONNX Runtime session.
    // - A pre-allocated DX resource is a buffer that is created by DxDispatch (independently of the ONNX model/session).
    // - An explicit ORT binding means creating an Ort::Value and storing it in m_tensors.
    // - An implicit ORT binding means passing an Ort::MemoryInfo to Ort::IoBinding::BindOutput, which lets the underlying
    //   execution provider allocate as necessary. This is useful when outputs have dynamic shapes that can't be pre-allocated.

    m_ioBindings->ClearBoundInputs();
    m_ioBindings->ClearBoundOutputs();
    m_tensors.clear();
    m_tensorWrappers.clear();

    Ort::MemoryInfo cpuMemoryInformation = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo dmlMemoryInformation("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);

    auto inputCount = m_session->GetInputCount();
    auto outputCount = m_session->GetOutputCount();

    for (int bindingPass = 0; bindingPass < 2; ++bindingPass)
    {
        const bool isInputTensor = (bindingPass == 0);
        const size_t tensorCount = isInputTensor ? inputCount : outputCount;

        for (size_t tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex)
        {
            Ort::TypeInfo typeInfo = isInputTensor ? m_session->GetInputTypeInfo(tensorIndex) : m_session->GetOutputTypeInfo(tensorIndex);
            std::string tensorName = OnnxParsers::GetTensorName(tensorIndex, *m_session, isInputTensor);
            bool isDmlSupportedType = false;

            std::vector<int64_t> tensorShape;

            if (typeInfo.GetONNXType() == ONNXType::ONNX_TYPE_TENSOR)
            {
                auto shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();

                tensorShape = shapeInfo.GetShape();
                for (auto& dim : tensorShape)
                {
                    dim = std::abs(dim);
                }

                auto dataTypeInfo = OnnxParsers::GetDataTypeInfo(shapeInfo.GetElementType());
                isDmlSupportedType = dataTypeInfo.dmlDataType != DML_TENSOR_DATA_TYPE_UNKNOWN;

                if (bindings.find(tensorName) == bindings.end())
                {
                    // No DXD binding exists, so allocate input tensors on the CPU.
                    // Outputs are implicitly allocated using memInfo to handle dynamic shapes.
                    if (isInputTensor)
                    {
                        if (isDmlSupportedType)
                        {
                            // Allocate a DX resource.
                            std::vector<uint32_t> sizes;
                            for (auto& dimSize : tensorShape) { sizes.push_back(static_cast<uint32_t>(dimSize)); }
                            auto resource = m_device->CreateDefaultBuffer(DMLCalcBufferTensorSize(
                                dataTypeInfo.dmlDataType,
                                sizes.size(),
                                sizes.data(),
                                nullptr
                                ));

                            // Wrap the explicitly allocated DX resource.
                            Microsoft::WRL::ComPtr<IUnknown> resourceWrapper;
                            m_tensors.emplace_back(CreateTensorFromResource(
                                m_ortDmlApi,
                                dmlMemoryInformation,
                                resource.Get(),
                                tensorShape,
                                dataTypeInfo.onnxDataType,
                                &resourceWrapper));

                            m_tensorWrappers.push_back(std::move(resourceWrapper));
                        }
                        else
                        {
                            auto allocator = static_cast<OrtAllocator*>(Ort::AllocatorWithDefaultOptions());
                            m_tensors.emplace_back(Ort::Value::CreateTensor(
                                allocator, 
                                tensorShape.data(),
                                tensorShape.size(), 
                                dataTypeInfo.onnxDataType));
                        }
                    }
                }
                else
                {
                    // Wrap the pre-allocated DX resource.
                    Microsoft::WRL::ComPtr<IUnknown> resourceWrapper;
                    m_tensors.emplace_back(CreateTensorFromResource(
                        m_ortDmlApi,
                        dmlMemoryInformation,
                        GetResourceFromModelBinding(tensorName, bindings),
                        tensorShape,
                        dataTypeInfo.onnxDataType,
                        &resourceWrapper));

                    m_tensorWrappers.push_back(std::move(resourceWrapper));
                }
            }

            if (isInputTensor)
            {
                if (typeInfo.GetONNXType() != ONNXType::ONNX_TYPE_TENSOR)
                {
                    // Don't bind non-tensor inputs.
                    continue;
                }

                // Bind the input tensor.
                m_ioBindings->BindInput(tensorName.c_str(), m_tensors.back());
            }
            else
            {
                assert(!isInputTensor);

                if (bindings.find(tensorName) == bindings.end())
                {
                    // Let the execution provider allocate the output.
                    m_ioBindings->BindOutput(tensorName.c_str(), isDmlSupportedType ? dmlMemoryInformation : cpuMemoryInformation);
                }
                else
                {
                    // Bind the pre-allocated DX resource.
                    m_ioBindings->BindOutput(tensorName.c_str(), m_tensors.back());
                }
            }
        }
    }
}

void OnnxDispatchable::Dispatch(const Model::DispatchCommand& args)
{
    PIXBeginEvent(m_device->GetCommandList(), PIX_COLOR(255, 255, 0), "ONNX: '%s'", args.dispatchableName.c_str());
    m_device->RecordTimestamp();
    m_device->ExecuteCommandList();

    Ort::RunOptions runOptions;
    m_session->Run(runOptions, *m_ioBindings);

    m_device->RecordTimestamp();
    PIXEndEvent(m_device->GetCommandList());
    m_device->ExecuteCommandList();
}

void OnnxDispatchable::Wait()
{
    m_ioBindings->SynchronizeOutputs();
}