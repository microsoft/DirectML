//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

namespace pydml
{
    struct CompiledModel
    {
        CompiledModel(
            dml::Graph& graph, 
            DML_EXECUTION_FLAGS flags,
            std::vector<dml::Expression>& outputs
            ) : 
            op(graph.Compile(flags, outputs))
        {}

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> op;
    };

    struct TensorData
    {
        TensorData(py::buffer_info const& info) :
            itemSize(info.itemsize),
            format(info.format),
            dimensions(info.ndim),
            shape(info.shape),
            strides(info.strides)
        {
            auto sizeInBytes = Size();
            buffer.resize(sizeInBytes);
            memcpy(buffer.data(), info.ptr, sizeInBytes);

            // Numpy strides use bytes.
            std::for_each(strides.begin(), strides.end(), [=](auto& i) {i *= itemSize; });
        }

        TensorData(dml::TensorDesc* desc) :
            itemSize(sizeof(float)),
            format(py::format_descriptor<float>::format()),
            dimensions(desc->sizes.size())
        {
            for (auto size : desc->sizes)
            {
                shape.push_back(static_cast<ssize_t>(size));
            }

            if (desc->strides)
            {
                for (auto stride : *desc->strides)
                {
                    strides.push_back(static_cast<ssize_t>(stride));
                }
            }
            else
            {
                // Use default descending packed strides.
                strides.resize(shape.size());
                ssize_t stride = 1;
                for (size_t i = strides.size(); i-- > 0; )
                {
                    strides[i] = stride;
                    stride *= shape[i];
                }
            }
            // Numpy strides use bytes.
            std::for_each(strides.begin(), strides.end(), [=](auto& i) {i *= itemSize; });

            buffer.resize(static_cast<size_t>(desc->totalTensorSizeInBytes));
        }

        TensorData() {}

        void* Get() const { return static_cast<void*>(const_cast<byte*>(buffer.data())); }

        size_t Size() const
        {
            size_t size = 1;

            for (auto length : shape)
            {
                size *= length;
            }

            return size * itemSize;
        }

        std::vector<byte> buffer;
        size_t itemSize;
        std::string format;
        size_t dimensions;
        std::vector<ssize_t> shape;
        std::vector<ssize_t> strides;
    };

    struct Binding
    {
        explicit Binding(dml::Expression& expression, py::buffer_info const& info)
            :   desc(expression.GetOutputDesc()),
                data(info)
        {}

        Binding() = default;

        dml::TensorDesc desc;
        TensorData data;
    };
}
