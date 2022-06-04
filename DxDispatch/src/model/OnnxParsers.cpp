#include "pch.h"
#include "OnnxParsers.h"
#include "DirectMLX.h"

std::string OnnxParsers::GetTensorName(size_t index, Ort::Session const& session, bool isInput)
{
    Ort::AllocatorWithDefaultOptions allocator;
    char* name = isInput ? session.GetInputName(index, allocator) : session.GetOutputName(index, allocator);
    std::string returnName(name);
    allocator.Free(name); // Don't leak memory.
    return returnName;
}

bool OnnxParsers::IsSupportedOnnxTensorElementDataType(ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:   return true;
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

Model OnnxParsers::ParseModel(const std::filesystem::path& filePath, gsl::span<const std::pair<std::string, uint32_t>> freeDimOverrides)
{
    const OrtApi& ortApi = Ort::GetApi();

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.DisableMemPattern();
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    for (auto& freeDimOverride : freeDimOverrides)
    {
        ortApi.AddFreeDimensionOverrideByName(sessionOptions, freeDimOverride.first.c_str(), freeDimOverride.second);
    }

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
                throw std::invalid_argument("Unsupported non-tensor input/output in ONNX model");
            }

            Ort::Unowned<Ort::TensorTypeAndShapeInfo> shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
            const ONNXTensorElementDataType tensorDataType = shapeInfo.GetElementType();
            if (!OnnxParsers::IsSupportedOnnxTensorElementDataType(tensorDataType))
            {
                throw std::invalid_argument("Unsupported tensor data type in ONNX model");
            }

            uint64_t elementCount = 1;
            std::vector<uint32_t> sizes;
            for (auto dim : shapeInfo.GetShape())
            {
                // std::abs to convert free dimensions (-1) to their minimum size of 1.
                sizes.push_back(std::abs(dim));
                elementCount *= sizes.back();
            }

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
                OnnxTensorDataTypeSize(tensorDataType)}};

            resources.emplace_back(std::move(resourceDesc));
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