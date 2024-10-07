// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

class OperatorField;

struct AbstractOperatorDesc
{
    const DML_OPERATOR_SCHEMA* schema = nullptr;
    std::vector<OperatorField> fields;

    AbstractOperatorDesc() = default;
    AbstractOperatorDesc(const DML_OPERATOR_SCHEMA* schema, std::vector<OperatorField>&& fields)
        : schema(schema)
        , fields(std::move(fields))
    {}

    std::vector<DmlBufferTensorDesc*> GetInputTensors()
    {
        return GetTensors<DmlBufferTensorDesc, DML_SCHEMA_FIELD_KIND_INPUT_TENSOR>();
    }

    std::vector<const DmlBufferTensorDesc*> GetInputTensors() const
    {
        return GetTensors<const DmlBufferTensorDesc, DML_SCHEMA_FIELD_KIND_INPUT_TENSOR>();
    }

    std::vector<DmlBufferTensorDesc*> GetOutputTensors()
    {
        return GetTensors<DmlBufferTensorDesc, DML_SCHEMA_FIELD_KIND_OUTPUT_TENSOR>();
    }

    std::vector<const DmlBufferTensorDesc*> GetOutputTensors() const
    {
        return GetTensors<const DmlBufferTensorDesc, DML_SCHEMA_FIELD_KIND_OUTPUT_TENSOR>();
    }

private:
    // Implemented in AbstractOperatorDescImpl.h.
    template <typename TensorType, DML_SCHEMA_FIELD_KIND Kind>
    std::vector<TensorType*> GetTensors() const;
};
