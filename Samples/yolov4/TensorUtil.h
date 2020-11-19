//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

namespace TensorUtil
{
    template <size_t N>
    uint32_t GetElementCount(TensorExtents<N> sizes)
    {
        uint32_t elementCount = 1;
        for (size_t i = 0; i < N; ++i)
        {
            elementCount *= sizes[i];
        }
        return elementCount;
    }

    template <size_t N>
    TensorExtents<N> GetPackedStrides(TensorExtents<N> sizes)
    {
        TensorExtents<N> strides;

        strides[N - 1] = 1;
        for (ptrdiff_t i = static_cast<ptrdiff_t>(N) - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * sizes[i + 1];
        }
        return strides;
    }

    template <size_t N>
    uint32_t GetElementOffset(TensorExtents<N> indices, TensorExtents<N> strides)
    {
        uint32_t elementOffset = 0;
        for (size_t i = 0; i < N; ++i)
        {
            elementOffset += indices[i] * strides[i];
        }
        return elementOffset;
    }

    template <size_t N>
    TensorExtents<N> GetElementIndices(uint32_t elementIndex, TensorExtents<N> sizes)
    {
        TensorExtents<N> indices;

        for (ptrdiff_t i = static_cast<ptrdiff_t>(N) - 1; i >= 0; --i)
        {
            uint32_t size = sizes[i];
            indices[i] = elementIndex % size;
            elementIndex /= size;
        }

        // The element should have been reduced to zero by all dimensions by now.
        // If not, then the passed-in index is out of bounds.
        assert(elementIndex == 0);

        return indices;
    }
}