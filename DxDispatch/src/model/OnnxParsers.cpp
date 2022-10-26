#include "pch.h"
#include "OnnxParsers.h"
#include "DirectMLX.h"
#include "dml_provider_factory.h"

std::string OnnxParsers::GetTensorName(size_t index, Ort::Session const& session, bool isInput)
{
    Ort::AllocatorWithDefaultOptions allocator;
    auto name = isInput ? session.GetInputNameAllocated(index, allocator) : session.GetOutputNameAllocated(index, allocator);
    std::string returnName(name.get());
    return returnName;
}

bool OnnxParsers::IsSupportedOnnxTensorElementDataType(ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:   return false;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:        return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:        return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:      return false;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:     return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:       return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   return false;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  return false;
    default: return false;
    }
}

DML_TENSOR_DATA_TYPE OnnxParsers::ConvertOnnxTensorDataType(ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return DML_TENSOR_DATA_TYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return DML_TENSOR_DATA_TYPE_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return DML_TENSOR_DATA_TYPE_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return DML_TENSOR_DATA_TYPE_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return DML_TENSOR_DATA_TYPE_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return DML_TENSOR_DATA_TYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return DML_TENSOR_DATA_TYPE_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return DML_TENSOR_DATA_TYPE_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return DML_TENSOR_DATA_TYPE_UINT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return DML_TENSOR_DATA_TYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return DML_TENSOR_DATA_TYPE_FLOAT64;
    default: throw std::invalid_argument("Unsupported tensor type");
    }
}

static uint32_t OnnxTensorDataTypeSize(ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return 8;
    default: throw std::invalid_argument("Unsupported tensor type");
    }
}

Model OnnxParsers::ParseModel(
    IDMLDevice* device,
    ID3D12CommandQueue* queue,
    const std::filesystem::path& filePath, 
    gsl::span<const std::pair<std::string, uint32_t>> freeDimNameOverrides,
    gsl::span<const std::pair<std::string, uint32_t>> freeDimDenotationOverrides)
{
    const OrtApi& ortApi = Ort::GetApi();

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.DisableMemPattern();
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    for (auto& freeDimOverride : freeDimNameOverrides)
    {
        Ort::ThrowOnError(ortApi.AddFreeDimensionOverrideByName(sessionOptions, freeDimOverride.first.c_str(), freeDimOverride.second));
    }

    for (auto& freeDimOverride : freeDimDenotationOverrides)
    {
        Ort::ThrowOnError(ortApi.AddFreeDimensionOverride(sessionOptions, freeDimOverride.first.c_str(), freeDimOverride.second));
    }

    const OrtDmlApi* ortDmlApi;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
    Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, device, queue));

    Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "DxDispatch");

    Ort::Session session = Ort::Session(ortEnvironment, filePath.wstring().c_str(), sessionOptions);

    auto inputCount = session.GetInputCount();
    auto outputCount = session.GetOutputCount();

    BucketAllocator allocator;

    // Create resources.
    std::vector<Model::ResourceDesc> resources;
    Model::Bindings bindings;

    for (int bindingPass = 0; bindingPass < 2; ++bindingPass)
    {
        const bool isInputTensor = (bindingPass == 0);
        const size_t tensorCount = isInputTensor ? inputCount : outputCount;

        for (size_t tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex)
        {
            Model::ResourceDesc resourceDesc = {};

            resourceDesc.name = OnnxParsers::GetTensorName(tensorIndex, session, isInputTensor);
            Ort::TypeInfo typeInfo = isInputTensor ? session.GetInputTypeInfo(tensorIndex) : session.GetOutputTypeInfo(tensorIndex);
            
            if (typeInfo.GetONNXType() != ONNXType::ONNX_TYPE_TENSOR)
            {
                continue;
            }

            Ort::Unowned<Ort::TensorTypeAndShapeInfo> shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
            const ONNXTensorElementDataType tensorDataType = shapeInfo.GetElementType();

            bool hasFreeDimensions = false;
            uint64_t elementCount = 1;
            std::vector<uint32_t> sizes;
            for (auto dim : shapeInfo.GetShape())
            {
                // ONNX models may have dynamic shapes where some dimensions are not statically defined in the model.
                // These dimensions may be specified at runtime (e.g., using -f option in dxdispatch.exe). Dimensions
                // that are neither statically defined nor provided at runtime are "free dimensions" with an invalid 
                // size of -1. It's safe to fix free dimensions to size 1 for inputs; however, it is NOT safe to do this
                // for outputs, which may have symbolic dimensions computed as a part of running the model.
                hasFreeDimensions = hasFreeDimensions || (dim == -1 && !isInputTensor);
                sizes.push_back(std::abs(dim));
                elementCount *= sizes.back();
            }

            // It's best to pre-allocate DX resources for efficiency: the resource can be allocated once and bound without 
            // incurring any copies or repeated allocations. It's safe to pre-allocate a resource so long as there are no 
            // remaining free dimensions and DML supports the tensor data type; otherwise, allocation will occur in the
            // OnnxDispatchable itself (either at binding or execution time). Unsupported tensor data types will be placed
            // on the CPU.
            if (!hasFreeDimensions && IsSupportedOnnxTensorElementDataType(tensorDataType))
            {
                Model::BufferDesc bufferDesc = {};
                bufferDesc.initialValuesDataType = ConvertOnnxTensorDataType(tensorDataType);
                bufferDesc.sizeInBytes = DMLCalcBufferTensorSize(
                    bufferDesc.initialValuesDataType,
                    sizes.size(),
                    sizes.data(),
                    nullptr
                );

                resourceDesc.value = bufferDesc;
                bindings[resourceDesc.name] = {Model::BufferBindingSource{
                    resourceDesc.name,
                    elementCount,
                    OnnxTensorDataTypeSize(tensorDataType)
                }};

                resources.emplace_back(std::move(resourceDesc));
            }
        }
    }

    // Create dispatchable.
    std::string dispatchableName = filePath.filename().string();
    std::vector<Model::DispatchableDesc> dispatchables = 
    {
        {dispatchableName, Model::OnnxDispatchableDesc{filePath}}
    };

    // Create dispatch command.
    std::vector<Model::Command> commands =
    {
        Model::DispatchCommand{dispatchableName, std::move(bindings), {}}
    };

    return {std::move(resources), std::move(dispatchables), std::move(commands), std::move(allocator)};
}