//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

namespace StringUtil
{
    struct NameAndIndex
    {
        const char* name; // Null terminated.
        uint32_t index;
    };

    struct WideNameAndIndex
    {
        const wchar_t* name; // Null terminated.
        uint32_t index;
    };

    inline std::optional<uint32_t> MapToIndex(std::string_view mode, gsl::span<const NameAndIndex> nameAndIndexList)
    {
        for (auto& nameAndIndex : nameAndIndexList)
        {
            if (strncmp(nameAndIndex.name, mode.data(), mode.size()) == 0)
            {
                return nameAndIndex.index;
            }
        }

        return {};
    }

    inline std::optional<uint32_t> MapToIndex(std::wstring_view mode, gsl::span<const WideNameAndIndex> nameAndIndexList)
    {
        for (auto& nameAndIndex : nameAndIndexList)
        {
            if (wcsncmp(nameAndIndex.name, mode.data(), mode.size()) == 0)
            {
                return nameAndIndex.index;
            }
        }

        return {};
    }
}

/*
inline constexpr uint32_t GetDataTypeBitPrecision(DML_TENSOR_DATA_TYPE dmlDataType)
{
    switch (dmlDataType)
    {
    case DML_TENSOR_DATA_TYPE_UINT4:
    case DML_TENSOR_DATA_TYPE_INT4:
        return 4;

    case DML_TENSOR_DATA_TYPE_UINT8:
    case DML_TENSOR_DATA_TYPE_INT8:
        return 8;

    case DML_TENSOR_DATA_TYPE_FLOAT16:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_INT16:
        return 16;

    case DML_TENSOR_DATA_TYPE_FLOAT32:
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_INT32:
        return 32;

    case DML_TENSOR_DATA_TYPE_FLOAT64:
    case DML_TENSOR_DATA_TYPE_UINT64:
    case DML_TENSOR_DATA_TYPE_INT64:
        return 64;

    default:
        assert(false);
        THROW_HR(E_UNEXPECTED);
    }
}

// divides and rounds up
inline uint64_t CeilDivide(uint64_t dividend, uint64_t divisor)
{
    uint64_t temp = dividend + divisor - 1;
    return temp / divisor;
}

// divides and rounds up
inline uint32_t CeilDivide(uint32_t dividend, uint32_t divisor)
{
    return static_cast<uint32_t>(CeilDivide(static_cast<uint64_t>(dividend), static_cast<uint64_t>(divisor)));
}

inline uint32_t GetDataTypeSize(DML_TENSOR_DATA_TYPE dmlDataType)
{
    return CeilDivide(GetDataTypeBitPrecision(dmlDataType), 8);
}

template <typename T>
T RoundUpToMultiple(T value, T multiple)
{
    static_assert(std::is_integral_v<T>);

    T remainder = value % multiple;
    if (remainder != 0)
    {
        value += multiple - remainder;
    }

    return value;
}

inline UINT64 CalcBufferTensorSize(
    DML_TENSOR_DATA_TYPE dataType,
    UINT dimensionCount,
    UINT64 precedingByteOffset, // For alignment calculation
    _In_reads_(dimensionCount) const UINT* sizes,
    _In_reads_opt_(dimensionCount) const UINT* strides,
    uint64_t sizeAlignment = 4ull)
{
    uint64_t elementSizeInBits = GetDataTypeBitPrecision(dataType);

    uint64_t minimumImpliedSizeInBits = 0;
    if (!strides)
    {
        minimumImpliedSizeInBits = sizes[0];
        for (UINT i = 1; i < dimensionCount; ++i)
        {
            minimumImpliedSizeInBits *= sizes[i];
        }
        minimumImpliedSizeInBits *= elementSizeInBits;
    }
    else
    {
        uint64_t indexOfLastElement = 0;
        for (UINT i = 0; i < dimensionCount; ++i)
        {
            indexOfLastElement += (sizes[i] - 1) * static_cast<uint64_t>(strides[i]);
        }

        minimumImpliedSizeInBits = (indexOfLastElement + 1) * elementSizeInBits;
    }

    uint64_t minimumImpliedSizeInBytes = CeilDivide(minimumImpliedSizeInBits, 8);

    // Round up to the nearest 4 bytes based on the bound location
    minimumImpliedSizeInBytes = RoundUpToMultiple(precedingByteOffset + minimumImpliedSizeInBytes, sizeAlignment) - precedingByteOffset;

    return minimumImpliedSizeInBytes;
}

uint64_t CalculateBufferSizeInBytes()
{
    // TODO #39698112: Implement DmlBufferTensorDesc's helper methods more rigorously for V1 descs and block layout
    // https://dev.azure.com/microsoft/OS/_workitems/edit/39698112

    if (v1 && v1->layoutType == DML_TENSOR_LAYOUT_TYPE_PRIVATE_BLOCK_PER_TILE)
    {
        assert(v1->dimensionBlockSizes.has_value());
        return CalcBufferTensorSize(
            dataType,
            GetDimensionCount(),
            v1->byteOffset,
            v1->dimensionBlockSizes->data(),
            nullptr
        );
    }

    return CalcBufferTensorSize(
        dataType,
        GetDimensionCount(),
        v1 ? v1->byteOffset : 0,
        sizes.data(),
        strides ? strides->data() : nullptr
    );
}
*/