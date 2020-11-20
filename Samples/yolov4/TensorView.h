//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

// Provides a non-owning typed view of an N-dimensional strided tensor over a linear array of elements.
template <typename T, size_t N>
class TensorViewND
{
public:
    using Extents = TensorExtents<N>;

    TensorViewND() = default;

    TensorViewND(dml::Span<T> data, Extents sizes, Extents strides)
        : m_data(data)
        , m_sizes(sizes)
        , m_strides(strides)
    {
    #if _DEBUG
        // Ensure the buffer is large enough
        uint32_t offsetOfLastElement = 0;
        for (size_t i = 0; i < N; ++i)
        {
            assert(m_sizes[i] > 0); // Zero size is invalid
            offsetOfLastElement += (m_sizes[i] - 1) * m_strides[i];
        }
        assert(static_cast<uint32_t>(data.size()) > offsetOfLastElement);
    #endif
    }

    TensorViewND(dml::Span<T> data, Extents sizes)
        : TensorViewND(data, sizes, TensorUtil::GetPackedStrides(sizes))
    {
    }

    const Extents& Sizes() const
    {
        return m_sizes;
    }

    const Extents& Strides() const
    {
        return m_strides;
    }

    uint32_t ElementCount() const
    {
        return TensorUtil::GetElementCount(m_sizes);
    }

    // Access an element by e.g. NCHW coordinate using an Extents of the appropriate dimension. Example:
    //   float x = view(NchwExtents(1, 2, 3, 4));
    T& operator()(Extents indices) const
    {
        return At(indices);
    }

    // Access an element by e.g. NCHW coordinate. The meaning of the indices depends on how many are provided. Examples:
    //    float a = view(1, 2);          // HW:    H=1, W=2
    //    float b = view(1, 2, 3);       // CHW:   C=1, H=2, W=3
    //    float c = view(1, 2, 3, 4);    // NCHW:  N=1, C=2, H=3, W=4
    //    float d = view(1, 2, 3, 4, 5); // NCDHW: N=1, C=2, D=3, H=4, W=5
    // 
    // If you provide fewer indices than the dimensionality of the view, the missing indices are treated as 0.
    template <typename... Ts>
    T& operator()(Ts&&... indices) const
    {
        static_assert(sizeof...(indices) >= 2, "Too few indices provided.");
        static_assert(sizeof...(indices) <= N, "Too many indices provided.");
        return At(Extents(std::forward<Ts>(indices)...));
    }

    // Access an element by linear index.
    T& operator[](uint32_t elementIndex) const
    {
        Extents indices = TensorUtil::GetElementIndices(elementIndex, m_sizes);
        return At(indices);
    }

protected:
    T& At(Extents indices) const
    {
        // Bounds check
        for (size_t i = 0; i < N; ++i)
        {
            assert(indices[i] < m_sizes[i]);
        }

        uint32_t elementOffset = TensorUtil::GetElementOffset(indices, m_strides);

        assert(elementOffset < static_cast<uint32_t>(m_data.size()));
        return m_data[elementOffset];
    }

private:
    dml::Span<T> m_data = {};
    Extents m_sizes = {};
    Extents m_strides = {};
};

// 4D tensor is the default
template <typename T>
using TensorView = TensorViewND<T, 4>;

template <typename T>
using TensorView5D = TensorViewND<T, 5>;
