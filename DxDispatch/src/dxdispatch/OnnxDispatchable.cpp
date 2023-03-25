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
    // ONNX dispatchables allow resources & bindings to be lazily instantiated. The final bindings are the union 
    // of JSON bindings and bindings to lazily-allocated DX resources from the first Bind(). Lazily-allocated DX resources
    // have a lifetime tied to the dispatchable instance and cannot be referenced by any other dispatchables. 
    //
    // Note that bindings and DX resources are only stored for tensors with static shapes: ONNX model inputs/outputs 
    // with free dimensions cannot be allocated ahead of time (unknown size until sometime during Session::Run), so 
    // they are implicitly allocated by ORT.

    if (m_ioBindings)
    {
        return;
    }

    m_ioBindings = Ort::IoBinding::IoBinding(*m_session);

    Ort::MemoryInfo cpuMemoryInformation = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo dmlMemoryInformation("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);

    auto inputCount = m_session->GetInputCount();
    auto outputCount = m_session->GetOutputCount();

    std::unordered_map<const char*, uint32_t> symbolicDimsForcedToSizeOne;

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

                tensorShape = shapeInfo.GetShape();
                uint64_t elementCount = 1;
                std::vector<uint32_t> sizes;

                // SymbolicDims will contain the names of any symbolic dimensions that haven't been overriden with -f/-F.
                bool hasFreeDimensions = false;
                std::vector<const char*> symbolicDims(shapeInfo.GetDimensionsCount());
                shapeInfo.GetSymbolicDimensions(symbolicDims.data(), shapeInfo.GetDimensionsCount());

                for (size_t dimIndex = 0; dimIndex < shapeInfo.GetDimensionsCount(); dimIndex++)
                {
                    auto& dimSize = tensorShape[dimIndex];

                    // Symbolic dimensions that have not been overriden (using -f/-F options) will have a size of -1.
                    if (dimSize == -1)
                    {
                        auto dimName = symbolicDims[dimIndex];

                        if (isInputTensor)
                        {
                            // We try fixing any symbolic dimensions that appear on inputs to size 1, which *may* make the graph valid
                            // for execution (e.g. symbolic dim represents batch size); however, this is not guaranteed to be valid
                            // in all models (e.g. symbolic dim represents a spatial size that will be downsampled). Forcing dims to 1 
                            // effectively gives input tensors static shapes that can be lazily allocated as DX resources. 
                            if (strlen(dimName) > 0)
                            {
                                symbolicDimsForcedToSizeOne[symbolicDims[dimIndex]] = 1;
                            }
                            dimSize = 1;
                        }
                        else
                        {
                            // Symbolic dimensions that appear on outputs cannot naively be forced to size 1 since they may be 
                            // computed during session run. However, it is safe if an input forced the same name to size 1 earlier.
                            if (strlen(dimName) > 0 && symbolicDimsForcedToSizeOne[symbolicDims[dimIndex]] == 1)
                            {
                                dimSize = 1;
                            }
                        }
                    }

                    hasFreeDimensions = hasFreeDimensions || dimSize == -1;

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
                    if (!hasFreeDimensions && isDmlSupportedType)
                    {
                        // Lazily allocate DX resource, wrap it as a DML tensor, and store a binding.
                        // This applies to both inputs and outputs so long as they have static shapes.
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
                    else if (isInputTensor && !isDmlSupportedType)
                    {
                        // Inputs with unsupported data types can be explicitly as CPU tensors.
                        binding.ortValue = Ort::Value::CreateTensor(
                            static_cast<OrtAllocator*>(Ort::AllocatorWithDefaultOptions()), 
                            tensorShape.data(),
                            tensorShape.size(), 
                            dataTypeInfo.onnxDataType
                        );
                    }
                }
                else
                {
                    // Wrap the pre-allocated DX resource from JSON model.
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

            m_tensors.emplace_back(std::move(binding));
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