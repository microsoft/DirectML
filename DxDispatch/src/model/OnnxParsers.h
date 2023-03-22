#pragma once

#include "Model.h"
#include <onnxruntime_cxx_api.h>

namespace OnnxParsers
{
    std::string GetTensorName(size_t index, Ort::Session const& session, bool isInput);

    struct DataTypeInfo
    {
        ONNXTensorElementDataType onnxDataType;
        DML_TENSOR_DATA_TYPE dmlDataType;
        uint32_t sizeInBytes;
    };

    DataTypeInfo GetDataTypeInfo(ONNXTensorElementDataType dataType);

    // Generates a DxDispatch model that has appropriate resources for an ONNX model's
    // inputs and outputs. The resources will be initialized with random values.
    Model ParseModel(
        IDMLDevice* device,
        ID3D12CommandQueue* queue,
        const std::filesystem::path& filePath, 
        gsl::span<const std::pair<std::string, uint32_t>> freeDimNameOverrides,
        gsl::span<const std::pair<std::string, uint32_t>> freeDimDenotationOverrides,
        uint32_t optimizationLevel);
}