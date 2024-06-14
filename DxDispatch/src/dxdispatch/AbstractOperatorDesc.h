//  Copyright (c) Microsoft Corporation.  All rights reserved.

#pragma once
#include <map>

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

    const char* GetInputTensorName(uint32_t tensorIndex) const;
    const char* GetOutputTensorName(uint32_t tensorIndex) const;

    std::vector<std::optional<DmlBufferTensorDesc>> CopyInputTensors() const
    {
        std::vector<std::optional<DmlBufferTensorDesc>> copiedTensors;
        std::vector<const DmlBufferTensorDesc*> inputTensors = GetInputTensors();

        copiedTensors.resize(inputTensors.size());

        for (uint32_t i = 0; i < inputTensors.size(); ++i)
        {
            if (inputTensors[i])
            {
                copiedTensors[i] = *inputTensors[i];
            }
        }

        return copiedTensors;
    }

    std::vector<std::optional<DmlBufferTensorDesc>> CopyOutputTensors() const
    {
        std::vector<std::optional<DmlBufferTensorDesc>> copiedTensors;
        std::vector<const DmlBufferTensorDesc*> outputTensors = GetOutputTensors();

        copiedTensors.resize(outputTensors.size());

        for (uint32_t i = 0; i < outputTensors.size(); ++i)
        {
            if (outputTensors[i])
            {
                copiedTensors[i] = *outputTensors[i];
            }
        }

        return copiedTensors;
    }

    static AbstractOperatorDesc GetAbstractDescFromWrapper(IDMLOperatorDescWrapperPrivate * wrapper);

private:
    const char* GetTensorFieldName(DML_SCHEMA_FIELD_KIND fieldKind, uint32_t instanceOfFieldKind) const;

    // Implemented in AbstractOperatorDescImpl.h.
    template <typename TensorType, DML_SCHEMA_FIELD_KIND Kind>
    std::vector<TensorType*> GetTensors() const;
};
