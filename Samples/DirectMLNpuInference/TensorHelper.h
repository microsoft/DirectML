// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <onnxruntime_cxx_api.h>

std::pair<Ort::Value, std::unique_ptr<void, void (*)(void*)>> CreateDmlValue(
    const Ort::ConstTensorTypeAndShapeInfo& tensor_info,
    ID3D12CommandQueue* queue);