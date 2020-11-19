//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

template <size_t DimensionCount> struct TensorExtents {}; // Purposefully undefined

template <>
struct TensorExtents<1> // nExtents for 1D arrays of n elements.
{
    static constexpr size_t DimensionCount = 1;

    union
    {
        struct
        {
            uint32_t n;
        };

        uint32_t asArray[DimensionCount];
    };

    // Constructors

    TensorExtents() = default;

    /*implicit*/ TensorExtents(dml::Span<const uint32_t> extents)
    {
        assert(extents.size() == DimensionCount);
        std::copy(extents.begin(), extents.end(), asArray);
    }

    TensorExtents(uint32_t n)
    {
        this->n = n;
    }

    // Accessors
    uint32_t& operator[](size_t i) { assert(i < DimensionCount); return asArray[i]; }
    const uint32_t& operator[](size_t i) const { assert(i < DimensionCount); return asArray[i]; }

    constexpr size_t size() const { return DimensionCount; }
};

//-----------------------------------------------------------------------------

template <>
struct TensorExtents<2> // HwExtents
{
    static constexpr size_t DimensionCount = 2;

    union
    {
        struct
        {
            uint32_t h;
            uint32_t w;
        };

        uint32_t asArray[DimensionCount];
    };

    // Constructors

    TensorExtents() = default;

    /*implicit*/ TensorExtents(dml::Span<const uint32_t> extents)
    {
        assert(extents.size() == DimensionCount);
        std::copy(extents.begin(), extents.end(), asArray);
    }

    TensorExtents(uint32_t h, uint32_t w)
    {
        this->h = h;
        this->w = w;
    }

    // Accessors
    uint32_t& operator[](size_t i) { assert(i < DimensionCount); return asArray[i]; }
    const uint32_t& operator[](size_t i) const { assert(i < DimensionCount); return asArray[i]; }

    constexpr size_t size() const { return DimensionCount; }
};

//-----------------------------------------------------------------------------

template <>
struct TensorExtents<3> // ChwExtents
{
    static constexpr size_t DimensionCount = 3;

    union
    {
        struct
        {
            uint32_t c;
            uint32_t h;
            uint32_t w;
        };

        uint32_t asArray[DimensionCount];
    };

    // Constructors

    TensorExtents() = default;

    /*implicit*/ TensorExtents(dml::Span<const uint32_t> extents)
    {
        assert(extents.size() == DimensionCount);
        std::copy(extents.begin(), extents.end(), asArray);
    }

    TensorExtents(uint32_t c, uint32_t h, uint32_t w)
    {
        this->c = c;
        this->h = h;
        this->w = w;
    }

    TensorExtents(uint32_t h, uint32_t w)
        : TensorExtents(0, h, w)
    {}

    // Accessors
    uint32_t& operator[](size_t i) { assert(i < DimensionCount); return asArray[i]; }
    const uint32_t& operator[](size_t i) const { assert(i < DimensionCount); return asArray[i]; }

    constexpr size_t size() const { return DimensionCount; }
};

//-----------------------------------------------------------------------------

template <>
struct TensorExtents<4> // NchwExtents
{
    static constexpr size_t DimensionCount = 4;

    union
    {
        struct
        {
            uint32_t n;
            uint32_t c;
            uint32_t h;
            uint32_t w;
        };

        uint32_t asArray[DimensionCount];
    };

    // Constructors

    TensorExtents() = default;

    /*implicit*/ TensorExtents(dml::Span<const uint32_t> extents)
    {
        assert(extents.size() == DimensionCount);
        std::copy(extents.begin(), extents.end(), asArray);
    }

    TensorExtents(uint32_t n, uint32_t c, uint32_t h, uint32_t w)
    {
        this->n = n;
        this->c = c;
        this->h = h;
        this->w = w;
    }

    TensorExtents(uint32_t c, uint32_t h, uint32_t w)
        : TensorExtents(0, c, h, w)
    {}

    TensorExtents(uint32_t h, uint32_t w)
        : TensorExtents(0, 0, h, w)
    {}

    // Accessors
    uint32_t& operator[](size_t i) { assert(i < DimensionCount); return asArray[i]; }
    const uint32_t& operator[](size_t i) const { assert(i < DimensionCount); return asArray[i]; }

    dml::Span<const uint32_t> AsSpan() const noexcept{ return dml::Span<const uint32_t>(data(), size()); }
    dml::Span<uint32_t> AsSpan() noexcept{ return dml::Span<uint32_t>(data(), size()); }

    constexpr size_t size() const { return DimensionCount; }
    const uint32_t* data() const noexcept { return &asArray[0]; }
    const uint32_t* begin() const noexcept { return &asArray[0]; }
    const uint32_t* end() const noexcept { return &asArray[DimensionCount]; }
    uint32_t* data() noexcept { return &asArray[0]; }
    uint32_t* begin() noexcept { return &asArray[0]; }
    uint32_t* end() noexcept { return &asArray[DimensionCount]; }
};

//-----------------------------------------------------------------------------

template <>
struct TensorExtents<5> // NcdhwExtents
{
    static constexpr size_t DimensionCount = 5;

    union
    {
        struct
        {
            uint32_t n;
            uint32_t c;
            uint32_t d;
            uint32_t h;
            uint32_t w;
        };

        uint32_t asArray[DimensionCount];
    };

    // Constructors

    TensorExtents() = default;

    /*implicit*/ TensorExtents(dml::Span<const uint32_t> extents)
    {
        assert(extents.size() == DimensionCount);
        std::copy(extents.begin(), extents.end(), asArray);
    }

    TensorExtents(uint32_t n, uint32_t c, uint32_t d, uint32_t h, uint32_t w)
    {
        this->n = n;
        this->c = c;
        this->d = d;
        this->h = h;
        this->w = w;
    }

    TensorExtents(uint32_t n, uint32_t c, uint32_t h, uint32_t w)
        : TensorExtents(n, c, 0, h, w)
    {}

    TensorExtents(uint32_t c, uint32_t h, uint32_t w)
        : TensorExtents(0, c, 0, h, w)
    {}

    TensorExtents(uint32_t h, uint32_t w)
        : TensorExtents(0, 0, 0, h, w)
    {}

    // Accessors
    uint32_t& operator[](size_t i) { assert(i < DimensionCount); return asArray[i]; }
    const uint32_t& operator[](size_t i) const { assert(i < DimensionCount); return asArray[i]; }

    dml::Span<const uint32_t> AsSpan() const noexcept{ return dml::Span<const uint32_t>(data(), size()); }
    dml::Span<uint32_t> AsSpan() noexcept{ return dml::Span<uint32_t>(data(), size()); }

    constexpr size_t size() const { return DimensionCount; }
    const uint32_t* data() const noexcept { return &asArray[0]; }
    const uint32_t* begin() const noexcept { return &asArray[0]; }
    const uint32_t* end() const noexcept { return &asArray[DimensionCount]; }
    uint32_t* data() noexcept { return &asArray[0]; }
    uint32_t* begin() noexcept { return &asArray[0]; }
    uint32_t* end() noexcept { return &asArray[DimensionCount]; }
};

//-----------------------------------------------------------------------------
// Operator overloads

template <size_t DimensionCount>
bool operator==(const TensorExtents<DimensionCount>& lhs, const TensorExtents<DimensionCount>& rhs)
{
    return std::equal(std::begin(lhs.asArray), std::end(lhs.asArray), std::begin(rhs.asArray), std::end(rhs.asArray));
}

template <size_t DimensionCount>
bool operator!=(const TensorExtents<DimensionCount>& lhs, const TensorExtents<DimensionCount>& rhs)
{
    return !(lhs == rhs);
}

template <size_t DimensionCount>
TensorExtents<DimensionCount>& operator+=(TensorExtents<DimensionCount>& lhs, const TensorExtents<DimensionCount>& rhs)
{
    for (size_t i = 0; i < DimensionCount; ++i)
    {
        lhs[i] += rhs[i];
    }
    return lhs;
}

template <size_t DimensionCount>
TensorExtents<DimensionCount> operator+(TensorExtents<DimensionCount> lhs, const TensorExtents<DimensionCount>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <size_t DimensionCount>
TensorExtents<DimensionCount>& operator-=(TensorExtents<DimensionCount>& lhs, const TensorExtents<DimensionCount>& rhs)
{
    for (size_t i = 0; i < DimensionCount; ++i)
    {
        lhs[i] -= rhs[i];
    }
    return lhs;
}

template <size_t DimensionCount>
TensorExtents<DimensionCount> operator-(TensorExtents<DimensionCount> lhs, const TensorExtents<DimensionCount>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <size_t DimensionCount>
TensorExtents<DimensionCount>& operator&=(TensorExtents<DimensionCount>& lhs, const TensorExtents<DimensionCount>& rhs)
{
    for (size_t i = 0; i < DimensionCount; ++i)
    {
        lhs[i] &= rhs[i];
    }
    return lhs;
}

template <size_t DimensionCount>
TensorExtents<DimensionCount> operator&(TensorExtents<DimensionCount> lhs, const TensorExtents<DimensionCount>& rhs)
{
    lhs &= rhs;
    return lhs;
}

//-----------------------------------------------------------------------------
// Helper typedefs
using HwExtents = TensorExtents<2>;
using ChwExtents = TensorExtents<3>;
using NchwExtents = TensorExtents<4>;
using NcdhwExtents = TensorExtents<5>;
