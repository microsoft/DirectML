#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "OnnxDispatchable.h"

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


static std::string GetTensorName(size_t index, Ort::Session const& session, bool isInput)
{
    Ort::AllocatorWithDefaultOptions allocator;
    auto name = isInput ? session.GetInputNameAllocated(index, allocator) : session.GetOutputNameAllocated(index, allocator);
    std::string returnName(name.get());
    return returnName;
}

struct DataTypeInfo
{
    ONNXTensorElementDataType onnxDataType;
    DML_TENSOR_DATA_TYPE dmlDataType;
    uint32_t sizeInBytes;
};

static DataTypeInfo GetDataTypeInfo(ONNXTensorElementDataType dataType)
{
    DataTypeInfo info = {};
    info.onnxDataType = dataType;
    info.dmlDataType = DML_TENSOR_DATA_TYPE_UNKNOWN;

    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_UINT8;
        info.sizeInBytes = 1;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_INT8;
        info.sizeInBytes = 1;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_UINT16;
        info.sizeInBytes = 2;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_INT16;
        info.sizeInBytes = 2;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_FLOAT16;
        info.sizeInBytes = 2;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_INT32;
        info.sizeInBytes = 4;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_UINT32;
        info.sizeInBytes = 4;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
        info.sizeInBytes = 4;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_UINT64;
        info.sizeInBytes = 8;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_INT64;
        info.sizeInBytes = 8;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_FLOAT64;
        info.sizeInBytes = 8;
        break;
    }

    return info;
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
}

void OnnxDispatchable::Bind(const Bindings& jsonBindings)
{
    // Type         | Static Shape | DML Data Type | JSON Binding | Result
    // -------------|--------------|---------------|--------------|---------------------------------------------------------
    // Input/Output | TRUE/FALSE   | TRUE/FALSE    | TRUE         | Bind DX resource allocated by executor
    // Input/Output | TRUE         | TRUE          | FALSE        | Bind DX resource lazily allocated by this dispatchable
    // Input/Output | TRUE         | FALSE         | FALSE        | Bind CPU resource lazily allocated by this dispatchable
    // Output       | FALSE        | TRUE          | FALSE        | Bind DML EP memory info (resource allocated each run)
    // Output       | FALSE        | FALSE         | FALSE        | Bind CPU EP memory info (resource allocated each run)

    if (m_ioBindings)
    {
        return;
    }

    m_ioBindings = Ort::IoBinding::IoBinding(*m_session);

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
            TensorBinding binding = {};
            auto tensorName = GetTensorName(tensorIndex, *m_session, isInputTensor);

            Ort::TypeInfo typeInfo = isInputTensor ? m_session->GetInputTypeInfo(tensorIndex) : m_session->GetOutputTypeInfo(tensorIndex);

            bool isDmlSupportedType = false;

            std::vector<int64_t> tensorShape;

            if (typeInfo.GetONNXType() == ONNXType::ONNX_TYPE_TENSOR)
            {
                auto shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
                auto dataTypeInfo = GetDataTypeInfo(shapeInfo.GetElementType());
                isDmlSupportedType = dataTypeInfo.dmlDataType != DML_TENSOR_DATA_TYPE_UNKNOWN;

                bool hasStaticShape = true;

                tensorShape = shapeInfo.GetShape();
                uint64_t elementCount = 1;
                std::vector<uint32_t> sizes;

                for (auto& dimSize : tensorShape)
                {
                    if (dimSize == -1) // Dimensions that aren't statically known/inferrable are "free dimensions" with size -1. 
                    {
                        if (isInputTensor)
                        {
                            // Try fixing any free dimensions that appear on *inputs* to size 1, which may make the graph valid
                            // for execution (e.g. dim represents batch size); however, this is not guaranteed to be valid
                            // in all models (e.g. dim represents a spatial size that will be downsampled). Forcing dims to 1 
                            // effectively gives all input tensors static shapes that can be preallocated. This trick cannot be
                            // done for outputs, since their free dimensions may correspond to symbolic dimensions that are
                            // only known at runtime (e.g. a shape node generates the dimensions based on input tensor values).
                            dimSize = 1;
                        }
                        else
                        {
                            // Tensors with one or more free dimensions have "dynamic shapes" and cannot be preallocated, 
                            // since their total size is unknown.
                            hasStaticShape = false;
                        }
                    }

                    sizes.push_back(static_cast<uint32_t>(std::abs(dimSize)));
                    elementCount *= sizes.back();
                }

                // Scalars have empty shapes.
                if (sizes.empty())
                {
                    sizes.push_back(1);
                }

                if (jsonBindings.find(tensorName) == jsonBindings.end())
                {
                    // Attempt to lazily create resources/bindings for tensors not bound in the JSON model (if any).
                    // Only tensors with static shapes can be preallocated.
                    if (hasStaticShape)
                    {
                        if (isDmlSupportedType)
                        {
                            // If the data type is supported by DML, then the resource is lazily allocated.
                            binding.resource = m_device->CreateDefaultBuffer(DMLCalcBufferTensorSize(
                                dataTypeInfo.dmlDataType,
                                sizes.size(),
                                sizes.data(),
                                nullptr
                            ));

                            binding.ortValue = CreateTensorFromResource(
                                m_ortDmlApi,
                                dmlMemoryInformation,
                                binding.resource.Get(),
                                tensorShape,
                                dataTypeInfo.onnxDataType,
                                &binding.wrapper
                            );
                        }
                        else
                        {
                            // Preallocate as a CPU resource.
                            binding.ortValue = Ort::Value::CreateTensor(
                                static_cast<OrtAllocator*>(Ort::AllocatorWithDefaultOptions()), 
                                tensorShape.data(),
                                tensorShape.size(), 
                                dataTypeInfo.onnxDataType
                            );
                        }
                    }
                }
                else
                {
                    // If a DX resource was explicitly bound in the JSON model, then it has already been allocated.
                    // Simply wrap the existing DX resource as an OrtValue. Note this path does NOT check if the
                    // the tensor has a static shape: it assume the user has allocated a large resource
                    // to accomodate the dynamic shape.
                    binding.resource = GetResourceFromModelBinding(tensorName, jsonBindings);
                    binding.ortValue = CreateTensorFromResource(
                        m_ortDmlApi,
                        dmlMemoryInformation,
                        binding.resource.Get(),
                        tensorShape,
                        dataTypeInfo.onnxDataType,
                        &binding.wrapper
                    );
                }
            }

            if (isInputTensor)
            {
                if (binding.ortValue)
                {
                    m_ioBindings->BindInput(tensorName.c_str(), *binding.ortValue);
                }
                else
                {
                    // Only non-tensor inputs should remain unbound.
                    assert(typeInfo.GetONNXType() != ONNXType::ONNX_TYPE_TENSOR);
                }
            }
            else
            {
                assert(!isInputTensor);

                if (binding.ortValue)
                {
                    m_ioBindings->BindOutput(tensorName.c_str(), *binding.ortValue);
                }
                else
                {
                    // Let the execution provider allocate the output.
                    m_ioBindings->BindOutput(tensorName.c_str(), isDmlSupportedType ? dmlMemoryInformation : cpuMemoryInformation);
                }
            }

            m_mergedBindings.emplace_back(std::move(binding));
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