// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

union ActivationOperatorDescUnion
{
    DML_ACTIVATION_IDENTITY_OPERATOR_DESC identity;
    DML_ACTIVATION_ELU_OPERATOR_DESC elu;
    DML_ACTIVATION_CELU_OPERATOR_DESC celu;
    DML_ACTIVATION_HARDMAX_OPERATOR_DESC hardmax;
    DML_ACTIVATION_HARDMAX1_OPERATOR_DESC hardmax1;
    DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC hardSigmoid;
    DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC leakyRelu;
    DML_ACTIVATION_LINEAR_OPERATOR_DESC linear;
    DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC logSoftmax;
    DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC logSoftmax1;
    DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC parameterizedRelu;
    DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC parametricSoftplus;
    DML_ACTIVATION_RELU_OPERATOR_DESC relu;
    DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC scaledTanh;
    DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC scaledElu;
    DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoid;
    DML_ACTIVATION_SOFTMAX_OPERATOR_DESC softmax;
    DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC softmax1;
    DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC softplus;
    DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC softsign;
    DML_ACTIVATION_TANH_OPERATOR_DESC tanh;
    DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC thresholdedRelu;
    DML_ACTIVATION_SHRINK_OPERATOR_DESC shrink;
    DML_ACTIVATION_GELU_OPERATOR_DESC gelu;
};

struct ActivationOperatorDesc
{
    ActivationOperatorDescUnion params;
    DML_OPERATOR_TYPE activationType;

    #pragma warning(push)
    #pragma warning(disable:4702)
    DML_OPERATOR_DESC GetDmlDesc() const
    {
        switch (activationType)
        {
        case DML_OPERATOR_ACTIVATION_ELU: return { activationType, &params.elu };
        case DML_OPERATOR_ACTIVATION_CELU: return { activationType, &params.celu };
        case DML_OPERATOR_ACTIVATION_HARDMAX: return { activationType, &params.hardmax };
        case DML_OPERATOR_ACTIVATION_HARDMAX1: return { activationType, &params.hardmax1 };
        case DML_OPERATOR_ACTIVATION_HARD_SIGMOID: return { activationType, &params.sigmoid };
        case DML_OPERATOR_ACTIVATION_IDENTITY: return { activationType, &params.identity };
        case DML_OPERATOR_ACTIVATION_LEAKY_RELU: return { activationType, &params.leakyRelu };
        case DML_OPERATOR_ACTIVATION_LINEAR: return { activationType, &params.linear };
        case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX: return { activationType, &params.logSoftmax };
        case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1: return { activationType, &params.logSoftmax1 };
        case DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU: return { activationType, &params.parameterizedRelu };
        case DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS: return { activationType, &params.parametricSoftplus };
        case DML_OPERATOR_ACTIVATION_RELU: return { activationType, &params.relu };
        case DML_OPERATOR_ACTIVATION_SCALED_ELU: return { activationType, &params.scaledElu };
        case DML_OPERATOR_ACTIVATION_SCALED_TANH: return { activationType, &params.scaledTanh };
        case DML_OPERATOR_ACTIVATION_SIGMOID: return { activationType, &params.sigmoid };
        case DML_OPERATOR_ACTIVATION_SOFTMAX: return { activationType, &params.softmax };
        case DML_OPERATOR_ACTIVATION_SOFTMAX1: return { activationType, &params.softmax1 };
        case DML_OPERATOR_ACTIVATION_SOFTPLUS: return { activationType, &params.softplus };
        case DML_OPERATOR_ACTIVATION_SOFTSIGN: return { activationType, &params.softsign };
        case DML_OPERATOR_ACTIVATION_TANH: return { activationType, &params.tanh };
        case DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU: return { activationType, &params.thresholdedRelu };
        case DML_OPERATOR_ACTIVATION_SHRINK: return { activationType, &params.shrink };
        case DML_OPERATOR_ACTIVATION_GELU: return { activationType, &params.gelu };
        default:
            return { activationType, &params.relu };
        }
    }
    #pragma warning(pop)
};

// DML_BUFFER_TENSOR_DESC (DML_TENSOR_TYPE_BUFFER)
struct DmlBufferTensorDesc
{
    DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_UNKNOWN;
    DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;
    std::vector<uint32_t> sizes;
    std::optional<std::vector<uint32_t>> strides;
    uint64_t totalTensorSizeInBytes = 0;
    uint32_t guaranteedBaseOffsetAlignment = 0;

    DmlBufferTensorDesc() = default;

    /*implicit*/ DmlBufferTensorDesc(const DML_BUFFER_TENSOR_DESC& desc)
        : dataType(desc.DataType)
        , flags(desc.Flags)
        , sizes(desc.Sizes, desc.Sizes + desc.DimensionCount)
        , totalTensorSizeInBytes(desc.TotalTensorSizeInBytes)
        , guaranteedBaseOffsetAlignment(desc.GuaranteedBaseOffsetAlignment)
    {
        if (desc.Strides)
        {
            strides.emplace(desc.Strides, desc.Strides + desc.DimensionCount);
        }
    }

    // Constructs a DmlBufferTensorDesc from a generic DML_TENSOR_DESC. The type must be DML_TENSOR_TYPE_BUFFER.
    /*implicit*/ DmlBufferTensorDesc(const DML_TENSOR_DESC& desc)
        : DmlBufferTensorDesc(*static_cast<const DML_BUFFER_TENSOR_DESC*>(desc.Desc))
    {
        assert(desc.Type == DML_TENSOR_TYPE_BUFFER);
    }
};
