#pragma once

#include "Model.h"
#include <onnxruntime_cxx_api.h>

namespace OnnxParsers
{
    std::string GetTensorName(size_t index, Ort::Session const& session, bool isInput);

    bool IsSupportedOnnxTensorElementDataType(ONNXTensorElementDataType dataType);

    DML_TENSOR_DATA_TYPE ConvertOnnxTensorDataType(ONNXTensorElementDataType dataType);

    // Generates a DxDispatch model that has appropriate resources for an ONNX model's
    // inputs and outputs. The resources will be initialized with random values.
    Model ParseModel(
        IDMLDevice* device,
        ID3D12CommandQueue* queue,
        const std::filesystem::path& filePath, 
        gsl::span<const std::pair<std::string, uint32_t>> freeDimOverrides);
}