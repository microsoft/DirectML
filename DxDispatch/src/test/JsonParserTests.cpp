#define NOMINMAX

#ifndef WIN32
#include <wsl/winadapter.h>
#include "directml_guids.h"
#endif

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <wrl/client.h>
#include "JsonParsers.h"
#include "DirectMLX.h"

using namespace rapidjson;
using namespace JsonParsers;

// ----------------------------------------------------------------------------
// FLOAT16
// ----------------------------------------------------------------------------

TEST(ParseFloat16Test, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": 1.2, 
        "x1": 5,
        "x2": -1.2345
    })");
    ASSERT_FALSE(d.HasParseError());
    half_float::half expectedValues[] = { half_float::half(1.2f), half_float::half(5.0f), half_float::half(-1.2345f) };
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName = fmt::format("x{}", i);
        EXPECT_EQ(ParseFloat16(d[fieldName.data()]), expectedValues[i]);
        EXPECT_EQ(ParseFloat16Field(d, fieldName.data()), expectedValues[i]);
    }
}

TEST(ParseFloat16Test, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": [],
        "x3": "cat",
        "x4": true
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseFloat16(field->value), std::invalid_argument);
        EXPECT_THROW(ParseFloat16Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseFloat16Test, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseFloat16Field(d, "x0", false, half_float::half(1.2f)), half_float::half(1.2f));
    EXPECT_THROW(ParseFloat16Field(d, "x0"), std::invalid_argument);
}

TEST(ParseFloat16Test, NaN)
{
    Document d;
    d.Parse(R"({ 
        "x0": "NaN", 
        "x1": "nan"
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_TRUE(std::isnan(ParseFloat16(field->value)));
        EXPECT_TRUE(std::isnan(ParseFloat16Field(d, field->name.GetString())));
    }
}

TEST(ParseFloat16Test, Infinity) 
{
    Document d;
    d.Parse(R"({ 
        "x0": "Inf", 
        "x1": "inf", 
        "x2": "-Inf", 
        "x3": "-inf"
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_TRUE(std::isinf(ParseFloat16(field->value)));
        EXPECT_TRUE(std::isinf(ParseFloat16Field(d, field->name.GetString())));
    }
}

TEST(ParseFloat16ArrayTest, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": [1.2],
        "x1": [1.2, 3.4, 5.6, "NaN", "Inf"]
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    auto x0 = ParseFloat16Array(d["x0"], allocator);
    EXPECT_EQ(1, x0.size());
    EXPECT_EQ(x0[0], half_float::half(1.2f));

    auto x1 = ParseFloat16Array(d["x1"], allocator);
    EXPECT_EQ(5, x1.size());
    EXPECT_EQ(x1[0], half_float::half(1.2f));
    EXPECT_EQ(x1[1], half_float::half(3.4f));
    EXPECT_EQ(x1[2], half_float::half(5.6f));
    EXPECT_TRUE(std::isnan(x1[3]));
    EXPECT_TRUE(std::isinf(x1[4]));
}

// ----------------------------------------------------------------------------
// FLOAT32
// ----------------------------------------------------------------------------

TEST(ParseFloat32Test, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": 1.2, 
        "x1": 5,
        "x2": -1.2345
    })");
    ASSERT_FALSE(d.HasParseError());
    float expectedValues[] = { 1.2f, 5.0f, -1.2345f };
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName = fmt::format("x{}", i);
        EXPECT_EQ(ParseFloat32(d[fieldName.data()]), expectedValues[i]);
        EXPECT_EQ(ParseFloat32Field(d, fieldName.data()), expectedValues[i]);
    }
}

TEST(ParseFloat32Test, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": [],
        "x3": "cat",
        "x4": true
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseFloat32(field->value), std::invalid_argument);
        EXPECT_THROW(ParseFloat32Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseFloat32Test, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseFloat32Field(d, "x0", false, 1.2f), 1.2f);
    EXPECT_THROW(ParseFloat32Field(d, "x0"), std::invalid_argument);
}

TEST(ParseFloat32Test, NaN)
{
    Document d;
    d.Parse(R"({ 
        "x0": "NaN", 
        "x1": "nan"
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_TRUE(std::isnan(ParseFloat32(field->value)));
        EXPECT_TRUE(std::isnan(ParseFloat32Field(d, field->name.GetString())));
    }
}

TEST(ParseFloat32Test, Infinity) 
{
    Document d;
    d.Parse(R"({ 
        "x0": "Inf", 
        "x1": "inf", 
        "x2": "-Inf", 
        "x3": "-inf"
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_TRUE(std::isinf(ParseFloat32(field->value)));
        EXPECT_TRUE(std::isinf(ParseFloat32Field(d, field->name.GetString())));
    }
}

TEST(ParseFloat32ArrayTest, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": [1.2],
        "x1": [1.2, 3.4, 5.6, "NaN", "Inf"]
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    auto x0 = ParseFloat32Array(d["x0"], allocator);
    EXPECT_EQ(1, x0.size());
    EXPECT_EQ(x0[0], 1.2f);

    auto x1 = ParseFloat32Array(d["x1"], allocator);
    EXPECT_EQ(5, x1.size());
    EXPECT_EQ(x1[0], 1.2f);
    EXPECT_EQ(x1[1], 3.4f);
    EXPECT_EQ(x1[2], 5.6f);
    EXPECT_TRUE(std::isnan(x1[3]));
    EXPECT_TRUE(std::isinf(x1[4]));
}

TEST(ParseFloat32ArrayTest, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": "cat",
        "x3": true,
        "x4": 123,
        "x5": 2.4
    })");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseFloat32Array(field->value, allocator), std::invalid_argument);
        EXPECT_THROW(ParseFloat32ArrayField(d, field->name.GetString(), allocator), std::invalid_argument);
    }
}

TEST(ParseFloat32ArrayTest, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    EXPECT_TRUE(ParseFloat32ArrayField(d, "x0", allocator, false).empty());
    EXPECT_THROW(ParseFloat32ArrayField(d, "x0", allocator), std::invalid_argument);
}

TEST(ParseFloat32ArrayTest, EmptyArray) 
{
    Document d;
    d.Parse(R"({ "x0": [] })");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    auto x0 = ParseFloat32Array(d["x0"], allocator);
    EXPECT_TRUE(x0.empty());
}

// ----------------------------------------------------------------------------
// FLOAT64
// ----------------------------------------------------------------------------

TEST(ParseFloat64Test, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": 1.2, 
        "x1": 5,
        "x2": -1.2345
    })");
    ASSERT_FALSE(d.HasParseError());
    double expectedValues[] = { 1.2, 5.0, -1.2345 };
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName = fmt::format("x{}", i);
        EXPECT_EQ(ParseFloat64(d[fieldName.data()]), expectedValues[i]);
        EXPECT_EQ(ParseFloat64Field(d, fieldName.data()), expectedValues[i]);
    }
}

TEST(ParseFloat64Test, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": [],
        "x3": "cat",
        "x4": true
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseFloat64(field->value), std::invalid_argument);
        EXPECT_THROW(ParseFloat64Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseFloat64Test, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseFloat64Field(d, "x0", false, 1.2f), 1.2f);
    EXPECT_THROW(ParseFloat64Field(d, "x0"), std::invalid_argument);
}

TEST(ParseFloat64Test, NaN)
{
    Document d;
    d.Parse(R"({ 
        "x0": "NaN", 
        "x1": "nan"
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_TRUE(std::isnan(ParseFloat64(field->value)));
        EXPECT_TRUE(std::isnan(ParseFloat64Field(d, field->name.GetString())));
    }
}

TEST(ParseFloat64Test, Infinity) 
{
    Document d;
    d.Parse(R"({ 
        "x0": "Inf", 
        "x1": "inf", 
        "x2": "-Inf", 
        "x3": "-inf"
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_TRUE(std::isinf(ParseFloat64(field->value)));
        EXPECT_TRUE(std::isinf(ParseFloat64Field(d, field->name.GetString())));
    }
}

// ----------------------------------------------------------------------------
// INT8
// ----------------------------------------------------------------------------

TEST(ParseInt8Test, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": 1, 
        "x1": 5,
        "x2": -3,
        "x3": -128,
        "x4": 127
    })");
    ASSERT_FALSE(d.HasParseError());
    int8_t expectedValues[] = { 1, 5, -3, -128, 127 };
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName = fmt::format("x{}", i);
        EXPECT_EQ(ParseInt8(d[fieldName.data()]), expectedValues[i]);
        EXPECT_EQ(ParseInt8Field(d, fieldName.data()), expectedValues[i]);
    }
}

TEST(ParseInt8Test, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": [],
        "x3": "cat",
        "x4": true,
        "x5": 1.2
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseInt8(field->value), std::invalid_argument);
        EXPECT_THROW(ParseInt8Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseInt8Test, InvalidNarrowing) 
{
    Document d;
    d.Parse(R"({ 
        "x0": -129,
        "x1": 128
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseInt8(field->value), gsl::narrowing_error);
        EXPECT_THROW(ParseInt8Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseInt8Test, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseInt8Field(d, "x0", false, 13), 13);
    EXPECT_THROW(ParseInt8Field(d, "x0"), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// INT16
// ----------------------------------------------------------------------------

TEST(ParseInt16Test, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": 1, 
        "x1": 5,
        "x2": -3,
        "x3": -32768,
        "x4": 32767
    })");
    ASSERT_FALSE(d.HasParseError());
    int16_t expectedValues[] = { 1, 5, -3, -32768, 32767 };
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName = fmt::format("x{}", i);
        EXPECT_EQ(ParseInt16(d[fieldName.data()]), expectedValues[i]);
        EXPECT_EQ(ParseInt16Field(d, fieldName.data()), expectedValues[i]);
    }
}

TEST(ParseInt16Test, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": [],
        "x3": "cat",
        "x4": true,
        "x5": 1.2
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseInt16(field->value), std::invalid_argument);
        EXPECT_THROW(ParseInt16Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseInt16Test, InvalidNarrowing) 
{
    Document d;
    d.Parse(R"({ 
        "x0": -32769,
        "x1": 32768
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseInt16(field->value), gsl::narrowing_error);
        EXPECT_THROW(ParseInt16Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseInt16Test, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseInt16Field(d, "x0", false, 13), 13);
    EXPECT_THROW(ParseInt16Field(d, "x0"), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// INT32
// ----------------------------------------------------------------------------

TEST(ParseInt32Test, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": 1, 
        "x1": 5,
        "x2": -3,
        "x3": -2147483648,
        "x4": 2147483647
    })");
    ASSERT_FALSE(d.HasParseError());
    int32_t expectedValues[] = { 1, 5, -3, INT32_MIN, INT32_MAX };
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName = fmt::format("x{}", i);
        EXPECT_EQ(ParseInt32(d[fieldName.data()]), expectedValues[i]);
        EXPECT_EQ(ParseInt32Field(d, fieldName.data()), expectedValues[i]);
    }
}

TEST(ParseInt32Test, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": [],
        "x3": "cat",
        "x4": true,
        "x5": 1.2
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseInt32(field->value), std::invalid_argument);
        EXPECT_THROW(ParseInt32Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseInt32Test, InvalidNarrowing) 
{
    Document d;
    d.Parse(R"({ 
        "x0": -2147483649,
        "x1": 2147483648
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseInt16(field->value), gsl::narrowing_error);
        EXPECT_THROW(ParseInt16Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseInt32Test, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseInt32Field(d, "x0", false, 13), 13);
    EXPECT_THROW(ParseInt32Field(d, "x0"), std::invalid_argument);
}

TEST(ParseInt32ArrayTest, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": [2],
        "x1": [2, 4, -5]
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    auto x0 = ParseInt32Array(d["x0"], allocator);
    EXPECT_EQ(1, x0.size());
    EXPECT_EQ(x0[0], 2);

    auto x1 = ParseInt32Array(d["x1"], allocator);
    EXPECT_EQ(3, x1.size());
    EXPECT_EQ(x1[0], 2);
    EXPECT_EQ(x1[1], 4);
    EXPECT_EQ(x1[2], -5);
}

TEST(ParseInt32ArrayTest, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": "cat",
        "x3": true,
        "x4": 123,
        "x5": 2.4,
        "x6": [1.2],
        "x7": ["nan"]
    })");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseInt32Array(field->value, allocator), std::invalid_argument);
        EXPECT_THROW(ParseInt32ArrayField(d, field->name.GetString(), allocator), std::invalid_argument);
    }
}

TEST(ParseInt32ArrayTest, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    EXPECT_TRUE(ParseInt32ArrayField(d, "x0", allocator, false).empty());
    EXPECT_THROW(ParseInt32ArrayField(d, "x0", allocator), std::invalid_argument);
}

TEST(ParseInt32ArrayTest, EmptyArray) 
{
    Document d;
    d.Parse(R"({ "x0": [] })");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    auto x0 = ParseInt32Array(d["x0"], allocator);
    EXPECT_TRUE(x0.empty());
}

// ----------------------------------------------------------------------------
// INT64
// ----------------------------------------------------------------------------

TEST(ParseInt64Test, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": 1, 
        "x1": 5,
        "x2": -3,
        "x3": -3000000000,
        "x4": 3000000000
    })");
    ASSERT_FALSE(d.HasParseError());
    int64_t expectedValues[] = { 1, 5, -3, -3000000000LL, 3000000000LL };
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName = fmt::format("x{}", i);
        EXPECT_EQ(ParseInt64(d[fieldName.data()]), expectedValues[i]);
        EXPECT_EQ(ParseInt64Field(d, fieldName.data()), expectedValues[i]);
    }
}

TEST(ParseInt64Test, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": [],
        "x3": "cat",
        "x4": true,
        "x5": 1.2
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseInt64(field->value), std::invalid_argument);
        EXPECT_THROW(ParseInt64Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseInt64Test, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseInt64Field(d, "x0", false, 13), 13);
    EXPECT_THROW(ParseInt64Field(d, "x0"), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// UINT8
// ----------------------------------------------------------------------------

TEST(ParseUInt8Test, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": 1, 
        "x1": 5,
        "x2": 255
    })");
    ASSERT_FALSE(d.HasParseError());
    uint8_t expectedValues[] = { 1, 5, 255 };
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName = fmt::format("x{}", i);
        EXPECT_EQ(ParseUInt8(d[fieldName.data()]), expectedValues[i]);
        EXPECT_EQ(ParseUInt8Field(d, fieldName.data()), expectedValues[i]);
    }
}

TEST(ParseUInt8Test, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": [],
        "x3": "cat",
        "x4": true,
        "x5": 1.2
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseUInt8(field->value), std::invalid_argument);
        EXPECT_THROW(ParseUInt8Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseUInt8Test, InvalidNarrowing) 
{
    Document d;
    d.Parse(R"({ "x0": 256 })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseUInt8(field->value), gsl::narrowing_error);
        EXPECT_THROW(ParseUInt8Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseUInt8Test, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseUInt8Field(d, "x0", false, 13), 13);
    EXPECT_THROW(ParseUInt8Field(d, "x0"), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// UINT16
// ----------------------------------------------------------------------------

TEST(ParseUInt16Test, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": 1, 
        "x1": 5,
        "x2": 65535
    })");
    ASSERT_FALSE(d.HasParseError());
    uint16_t expectedValues[] = { 1, 5, 65535 };
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName = fmt::format("x{}", i);
        EXPECT_EQ(ParseUInt16(d[fieldName.data()]), expectedValues[i]);
        EXPECT_EQ(ParseUInt16Field(d, fieldName.data()), expectedValues[i]);
    }
}

TEST(ParseUInt16Test, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": [],
        "x3": "cat",
        "x4": true,
        "x5": 1.2
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseUInt16(field->value), std::invalid_argument);
        EXPECT_THROW(ParseUInt16Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseUInt16Test, InvalidNarrowing) 
{
    Document d;
    d.Parse(R"({ "x0": 65536 })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseUInt16(field->value), gsl::narrowing_error);
        EXPECT_THROW(ParseUInt16Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseUInt16Test, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseUInt16Field(d, "x0", false, 13), 13);
    EXPECT_THROW(ParseUInt16Field(d, "x0"), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// UINT32
// ----------------------------------------------------------------------------

TEST(ParseUInt32Test, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": 1, 
        "x1": 5,
        "x2": 4294967295
    })");
    ASSERT_FALSE(d.HasParseError());
    uint32_t expectedValues[] = { 1, 5, 4294967295 };
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName = fmt::format("x{}", i);
        EXPECT_EQ(ParseUInt32(d[fieldName.data()]), expectedValues[i]);
        EXPECT_EQ(ParseUInt32Field(d, fieldName.data()), expectedValues[i]);
    }
}

TEST(ParseUInt32Test, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": [],
        "x3": "cat",
        "x4": true,
        "x5": 1.2,
        "x6": -5
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseUInt32(field->value), std::invalid_argument);
        EXPECT_THROW(ParseUInt32Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseUInt32Test, InvalidNarrowing) 
{
    Document d;
    d.Parse(R"({ "x0": 4294967296 })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseUInt16(field->value), gsl::narrowing_error);
        EXPECT_THROW(ParseUInt16Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseUInt32Test, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseUInt32Field(d, "x0", false, 13), 13);
    EXPECT_THROW(ParseUInt32Field(d, "x0"), std::invalid_argument);
}

TEST(ParseUInt32ArrayTest, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": [2],
        "x1": [2, 4, 12]
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    auto x0 = ParseUInt32Array(d["x0"], allocator);
    EXPECT_EQ(1, x0.size());
    EXPECT_EQ(x0[0], 2);

    auto x1 = ParseUInt32Array(d["x1"], allocator);
    EXPECT_EQ(3, x1.size());
    EXPECT_EQ(x1[0], 2);
    EXPECT_EQ(x1[1], 4);
    EXPECT_EQ(x1[2], 12);
}

TEST(ParseUInt32ArrayTest, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": "cat",
        "x3": true,
        "x4": 123,
        "x5": 2.4,
        "x6": [1.2],
        "x7": ["nan"],
        "x8": [-15]
    })");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseUInt32Array(field->value, allocator), std::invalid_argument);
        EXPECT_THROW(ParseUInt32ArrayField(d, field->name.GetString(), allocator), std::invalid_argument);
    }
}

TEST(ParseUInt32ArrayTest, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    EXPECT_TRUE(ParseUInt32ArrayField(d, "x0", allocator, false).empty());
    EXPECT_THROW(ParseUInt32ArrayField(d, "x0", allocator), std::invalid_argument);
}

TEST(ParseUInt32ArrayTest, EmptyArray) 
{
    Document d;
    d.Parse(R"({ "x0": [] })");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    auto x0 = ParseUInt32Array(d["x0"], allocator);
    EXPECT_TRUE(x0.empty());
}

// ----------------------------------------------------------------------------
// UINT64
// ----------------------------------------------------------------------------

TEST(ParseUInt64Test, ValidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": 1, 
        "x1": 5,
        "x2": 4294967296
    })");
    ASSERT_FALSE(d.HasParseError());
    uint64_t expectedValues[] = { 1, 5, 4294967296 };
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName = fmt::format("x{}", i);
        EXPECT_EQ(ParseUInt64(d[fieldName.data()]), expectedValues[i]);
        EXPECT_EQ(ParseUInt64Field(d, fieldName.data()), expectedValues[i]);
    }
}

TEST(ParseUInt64Test, InvalidInput) 
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": [],
        "x3": "cat",
        "x4": true,
        "x5": 1.2,
        "x6": -5
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseUInt64(field->value), std::invalid_argument);
        EXPECT_THROW(ParseUInt64Field(d, field->name.GetString()), std::invalid_argument);
    }
}

TEST(ParseUInt64Test, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseUInt64Field(d, "x0", false, 13), 13);
    EXPECT_THROW(ParseUInt64Field(d, "x0"), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// Mixed Primitives
// ----------------------------------------------------------------------------

TEST(ParseMixedPrimitiveArrayTest, ValidInput) 
{
    Document d;
    d.Parse(R"([
        { "type": "FLOAT32", "value": 1.23 },
        { "type": "UINT8", "value": 55 },
        { "type": "INT16", "value": -12 },
        { "type": "FLOAT32", "value": 3.1415 },
        { "type": "FLOAT32", "value": 19 }
    ])");
    ASSERT_FALSE(d.HasParseError());

    auto result = ParseMixedPrimitiveArray(d);
    ASSERT_EQ(result.size(), 15);
    EXPECT_EQ(*reinterpret_cast<float*>(result.data()), 1.23f);
    EXPECT_EQ(*reinterpret_cast<uint8_t*>(result.data() + 4), 55);
    EXPECT_EQ(*reinterpret_cast<int16_t*>(result.data() + 5), -12);
    EXPECT_EQ(*reinterpret_cast<float*>(result.data() + 7), 3.1415f);
    EXPECT_EQ(*reinterpret_cast<float*>(result.data() + 11), 19.0f);
}

// ----------------------------------------------------------------------------
// DML_TENSOR_DATA_TYPE
// ----------------------------------------------------------------------------

TEST(ParseDmlTensorDataTypeTest, ValidInput)
{
    Document d;
    d.Parse(R"({
        "x0": "DML_TENSOR_DATA_TYPE_UNKNOWN",
        "x1": "UNKNOWN",
        "x2": "DML_TENSOR_DATA_TYPE_FLOAT32",
        "x3": "FLOAT32",
        "x4": "DML_TENSOR_DATA_TYPE_FLOAT16",
        "x5": "FLOAT16",
        "x6": "DML_TENSOR_DATA_TYPE_UINT32",
        "x7": "UINT32",
        "x8": "DML_TENSOR_DATA_TYPE_UINT16",
        "x9": "UINT16",
        "x10": "DML_TENSOR_DATA_TYPE_UINT8",
        "x11": "UINT8",
        "x12": "DML_TENSOR_DATA_TYPE_INT32",
        "x13": "INT32",
        "x14": "DML_TENSOR_DATA_TYPE_INT16",
        "x15": "INT16",
        "x16": "DML_TENSOR_DATA_TYPE_INT8",
        "x17": "INT8",
        "x18": "DML_TENSOR_DATA_TYPE_FLOAT64",
        "x19": "FLOAT64",
        "x20": "DML_TENSOR_DATA_TYPE_UINT64",
        "x21": "UINT64",
        "x22": "DML_TENSOR_DATA_TYPE_INT64",
        "x23": "INT64"
    })");
    ASSERT_FALSE(d.HasParseError());

    const DML_TENSOR_DATA_TYPE expectedValues[] = { 
        DML_TENSOR_DATA_TYPE_UNKNOWN,
        DML_TENSOR_DATA_TYPE_FLOAT32,
        DML_TENSOR_DATA_TYPE_FLOAT16,
        DML_TENSOR_DATA_TYPE_UINT32,
        DML_TENSOR_DATA_TYPE_UINT16,
        DML_TENSOR_DATA_TYPE_UINT8,
        DML_TENSOR_DATA_TYPE_INT32,
        DML_TENSOR_DATA_TYPE_INT16,
        DML_TENSOR_DATA_TYPE_INT8,
        DML_TENSOR_DATA_TYPE_FLOAT64,
        DML_TENSOR_DATA_TYPE_UINT64,
        DML_TENSOR_DATA_TYPE_INT64
    };

    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName1 = fmt::format("x{}", 2*i);
        auto fieldName2 = fmt::format("x{}", 2*i+1);
        EXPECT_EQ(ParseDmlTensorDataType(d[fieldName1.data()]), expectedValues[i]);
        EXPECT_EQ(ParseDmlTensorDataTypeField(d, fieldName1), expectedValues[i]);
        EXPECT_EQ(ParseDmlTensorDataType(d[fieldName2.data()]), expectedValues[i]);
        EXPECT_EQ(ParseDmlTensorDataTypeField(d, fieldName2), expectedValues[i]);
    }
}

TEST(ParseDmlTensorDataTypeTest, InvalidInput)
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": [],
        "x3": "cat",
        "x4": true,
        "x5": 1.2,
        "x6": -5
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseDmlTensorDataType(field->value), std::invalid_argument);
    }
}

TEST(ParseDmlTensorDataTypeTest, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseDmlTensorDataTypeField(d, "x0", false, DML_TENSOR_DATA_TYPE_FLOAT32), DML_TENSOR_DATA_TYPE_FLOAT32);
    EXPECT_THROW(ParseDmlTensorDataTypeField(d, "x0"), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// DML_EXECUTION_FLAGS
// ----------------------------------------------------------------------------

TEST(ParseDmlExecutionFlagsTest, ValidInputString)
{
    Document d;
    d.Parse(R"({
        "x0": "DML_EXECUTION_FLAG_NONE",
        "x1": "NONE",
        "x2": "DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION",
        "x3": "ALLOW_HALF_PRECISION_COMPUTATION",
        "x4": "DML_EXECUTION_FLAG_DISABLE_META_COMMANDS",
        "x5": "DISABLE_META_COMMANDS",
        "x6": "DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE",
        "x7": "DESCRIPTORS_VOLATILE"
    })");
    ASSERT_FALSE(d.HasParseError());

    const DML_EXECUTION_FLAGS expectedValues[] = { 
        DML_EXECUTION_FLAG_NONE,
        DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION,
        DML_EXECUTION_FLAG_DISABLE_META_COMMANDS,
        DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE,
    };

    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName1 = fmt::format("x{}", 2*i);
        auto fieldName2 = fmt::format("x{}", 2*i+1);
        EXPECT_EQ(ParseDmlExecutionFlags(d[fieldName1.data()]), expectedValues[i]);
        EXPECT_EQ(ParseDmlExecutionFlagsField(d, fieldName1), expectedValues[i]);
        EXPECT_EQ(ParseDmlExecutionFlags(d[fieldName2.data()]), expectedValues[i]);
        EXPECT_EQ(ParseDmlExecutionFlagsField(d, fieldName2), expectedValues[i]);
    }
}

TEST(ParseDmlExecutionFlagsTest, ValidInputArray)
{
    Document d;
    d.Parse(R"({
        "x0": [],
        "x1": ["NONE"],
        "x2": ["ALLOW_HALF_PRECISION_COMPUTATION", "DESCRIPTORS_VOLATILE"]
    })");
    ASSERT_FALSE(d.HasParseError());

    const DML_EXECUTION_FLAGS expectedValues[] = { 
        DML_EXECUTION_FLAG_NONE,
        DML_EXECUTION_FLAG_NONE,
        DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION | DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE,
    };

    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        auto fieldName = fmt::format("x{}", i);
        EXPECT_EQ(ParseDmlExecutionFlags(d[fieldName.data()]), expectedValues[i]);
        EXPECT_EQ(ParseDmlExecutionFlagsField(d, fieldName), expectedValues[i]);
    }
}

TEST(ParseDmlExecutionFlagsTest, InvalidInput)
{
    Document d;
    d.Parse(R"({ 
        "x0": null, 
        "x1": {},
        "x2": "cat",
        "x3": true,
        "x4": 1.2,
        "x5": -5
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseDmlExecutionFlags(field->value), std::invalid_argument);
    }
}

TEST(ParseDmlExecutionFlagsTest, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    EXPECT_EQ(ParseDmlExecutionFlagsField(d, "x0", false, DML_EXECUTION_FLAG_DISABLE_META_COMMANDS), DML_EXECUTION_FLAG_DISABLE_META_COMMANDS);
    EXPECT_THROW(ParseDmlExecutionFlagsField(d, "x0"), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// DML_SIZE_2D
// ----------------------------------------------------------------------------

TEST(ParseDmlSize2dTest, ValidInput) 
{
    Document d;
    d.Parse(R"({
        "x0": { "Width": 4, "Height": 15 }
    })");
    ASSERT_FALSE(d.HasParseError());

    auto result = ParseDmlSize2d(d["x0"]);
    EXPECT_EQ(result.Width, 4);
    EXPECT_EQ(result.Height, 15);

    BucketAllocator allocator;
    auto result2 = ParseDmlSize2dField(d, "x0", allocator);
    EXPECT_EQ(result2->Width, 4);
    EXPECT_EQ(result2->Height, 15);
}

TEST(ParseDmlSize2dTest, InvalidInput) 
{
    Document d;
    d.Parse(R"({
        "x0": { "Width": 4.2, "Height": 15.1 },
        "x1": null,
        "x2": "cat",
        "x3": [],
        "x4": 5
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseDmlTensorDataType(field->value), std::invalid_argument);
    }
}

TEST(ParseDmlSize2dTest, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    EXPECT_EQ(ParseDmlSize2dField(d, "x0", allocator, false, nullptr), nullptr);
    EXPECT_THROW(ParseDmlSize2dField(d, "x0", allocator), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// DML_SIZE_3D
// ----------------------------------------------------------------------------

TEST(ParseDmlSize3dTest, ValidInput) 
{
    Document d;
    d.Parse(R"({
        "x0": { "Width": 4, "Height": 15 }
    })");
    ASSERT_FALSE(d.HasParseError());

    auto result = ParseDmlSize3d(d["x0"]);
    EXPECT_EQ(result.Width, 4);
    EXPECT_EQ(result.Height, 15);

    BucketAllocator allocator;
    auto result2 = ParseDmlSize3dField(d, "x0", allocator);
    EXPECT_EQ(result2->Width, 4);
    EXPECT_EQ(result2->Height, 15);
}

TEST(ParseDmlSize3dTest, InvalidInput) 
{
    Document d;
    d.Parse(R"({
        "x0": { "Width": 4.2, "Height": 15.1 },
        "x1": null,
        "x2": "cat",
        "x3": [],
        "x4": 5
    })");
    ASSERT_FALSE(d.HasParseError());
    for (auto field = d.MemberBegin(); field < d.MemberEnd(); field++)
    {
        EXPECT_THROW(ParseDmlTensorDataType(field->value), std::invalid_argument);
    }
}

TEST(ParseDmlSize3dTest, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    EXPECT_EQ(ParseDmlSize3dField(d, "x0", allocator, false, nullptr), nullptr);
    EXPECT_THROW(ParseDmlSize3dField(d, "x0", allocator), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// DML_SCALAR_UNION
// ----------------------------------------------------------------------------

TEST(ParseDmlScalarUnionTest, ValidObjects) 
{
    Document d;
    d.Parse(R"({
        "x0": { "Int8": -100 },
        "x1": { "UInt8": 32 },
        "x2": { "Int16": 123 },
        "x3": { "UInt16": 15 },
        "x4": { "Int32": 33 },
        "x5": { "UInt32": 1000 },
        "x6": { "Int64": -12345 },
        "x7": { "UInt64": 423 },
        "x8": { "Float32": 1.2 },
        "x9": { "Float64": 3.4 }
    })");
    ASSERT_FALSE(d.HasParseError());

    EXPECT_EQ(ParseDmlScalarUnion(d["x0"], DML_TENSOR_DATA_TYPE_INT8).Int8, -100);
    EXPECT_EQ(ParseDmlScalarUnion(d["x1"], DML_TENSOR_DATA_TYPE_UINT8).UInt8, 32);
    EXPECT_EQ(ParseDmlScalarUnion(d["x2"], DML_TENSOR_DATA_TYPE_INT16).Int16, 123);
    EXPECT_EQ(ParseDmlScalarUnion(d["x3"], DML_TENSOR_DATA_TYPE_UINT16).UInt16, 15);
    EXPECT_EQ(ParseDmlScalarUnion(d["x4"], DML_TENSOR_DATA_TYPE_INT32).Int32, 33);
    EXPECT_EQ(ParseDmlScalarUnion(d["x5"], DML_TENSOR_DATA_TYPE_UINT32).UInt32, 1000);
    EXPECT_EQ(ParseDmlScalarUnion(d["x6"], DML_TENSOR_DATA_TYPE_INT64).Int64, -12345);
    EXPECT_EQ(ParseDmlScalarUnion(d["x7"], DML_TENSOR_DATA_TYPE_UINT64).UInt64, 423);
    EXPECT_EQ(ParseDmlScalarUnion(d["x8"], DML_TENSOR_DATA_TYPE_FLOAT32).Float32, 1.2f);
    EXPECT_EQ(ParseDmlScalarUnion(d["x9"], DML_TENSOR_DATA_TYPE_FLOAT64).Float64, 3.4);
}

TEST(ParseDmlScalarUnionTest, InvalidObjects) 
{
    Document d;
    d.Parse(R"({
        "x0": { "Int8": -100 },
        "x1": { "UInt32": 1.2 }
    })");
    ASSERT_FALSE(d.HasParseError());

    EXPECT_THROW(ParseDmlScalarUnion(d["x0"], DML_TENSOR_DATA_TYPE_INT16), std::invalid_argument);
    EXPECT_THROW(ParseDmlScalarUnion(d["x1"], DML_TENSOR_DATA_TYPE_UINT32), std::invalid_argument);
}

TEST(ParseDmlScalarUnionTest, ValidNumbers) 
{
    Document d;
    d.Parse(R"({
        "x0": -100,
        "x1": 32,
        "x2": 123,
        "x3": 15,
        "x4": 33,
        "x5": 1000,
        "x6": -12345,
        "x7": 423,
        "x8": 1.2,
        "x9": 3.4
    })");
    ASSERT_FALSE(d.HasParseError());

    EXPECT_EQ(ParseDmlScalarUnion(d["x0"], DML_TENSOR_DATA_TYPE_INT8).Int8, -100);
    EXPECT_EQ(ParseDmlScalarUnion(d["x1"], DML_TENSOR_DATA_TYPE_UINT8).UInt8, 32);
    EXPECT_EQ(ParseDmlScalarUnion(d["x2"], DML_TENSOR_DATA_TYPE_INT16).Int16, 123);
    EXPECT_EQ(ParseDmlScalarUnion(d["x3"], DML_TENSOR_DATA_TYPE_UINT16).UInt16, 15);
    EXPECT_EQ(ParseDmlScalarUnion(d["x4"], DML_TENSOR_DATA_TYPE_INT32).Int32, 33);
    EXPECT_EQ(ParseDmlScalarUnion(d["x5"], DML_TENSOR_DATA_TYPE_UINT32).UInt32, 1000);
    EXPECT_EQ(ParseDmlScalarUnion(d["x6"], DML_TENSOR_DATA_TYPE_INT64).Int64, -12345);
    EXPECT_EQ(ParseDmlScalarUnion(d["x7"], DML_TENSOR_DATA_TYPE_UINT64).UInt64, 423);
    EXPECT_EQ(ParseDmlScalarUnion(d["x8"], DML_TENSOR_DATA_TYPE_FLOAT32).Float32, 1.2f);
    EXPECT_EQ(ParseDmlScalarUnion(d["x9"], DML_TENSOR_DATA_TYPE_FLOAT64).Float64, 3.4);
}

TEST(ParseDmlScalarUnionTest, Fields) 
{
    Document d;
    d.Parse(R"({
        "DataType0": "DML_TENSOR_DATA_TYPE_FLOAT32",
        "DataType1": "DML_TENSOR_DATA_TYPE_INT16",
        "x0": { "Float32": -1.234 },
        "x1": { "Int16": 32 }
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    EXPECT_EQ(ParseDmlScalarUnionField(d, "x0", "DataType0", allocator)->Float32, -1.234f);
    EXPECT_EQ(ParseDmlScalarUnionField(d, "x1", "DataType1", allocator)->Int16, 32);

    EXPECT_THROW(ParseDmlScalarUnionField(d, "x0", "DataType1", allocator), std::invalid_argument);
    EXPECT_THROW(ParseDmlScalarUnionField(d, "x1", "DataType0", allocator), std::invalid_argument);
}

TEST(ParseDmlScalarUnionTest, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    EXPECT_EQ(ParseDmlScalarUnionField(d, "x0", "foo", allocator, false, nullptr), nullptr);
    EXPECT_THROW(ParseDmlScalarUnionField(d, "x0", "foo", allocator), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// DML_SCALE_BIAS
// ----------------------------------------------------------------------------

TEST(ParseDmlScaleBiasTest, ValidInput) 
{
    Document d;
    d.Parse(R"({
        "x0": { "Scale": 4.2, "Bias": 15.5 }
    })");
    ASSERT_FALSE(d.HasParseError());

    auto result = ParseDmlScaleBias(d["x0"]);
    EXPECT_EQ(result.Scale, 4.2f);
    EXPECT_EQ(result.Bias, 15.5f);

    BucketAllocator allocator;
    auto result2 = ParseDmlScaleBiasField(d, "x0", allocator);
    EXPECT_EQ(result2->Scale, 4.2f);
    EXPECT_EQ(result2->Bias, 15.5f);
}

TEST(ParseDmlScaleBiasTest, MissingField) 
{
    Document d;
    d.Parse(R"({})");
    ASSERT_FALSE(d.HasParseError());
    BucketAllocator allocator;
    EXPECT_EQ(ParseDmlScaleBiasField(d, "x0", allocator, false, nullptr), nullptr);
    EXPECT_THROW(ParseDmlScaleBiasField(d, "x0", allocator), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// DML_BUFFER_TENSOR_DESC
// ----------------------------------------------------------------------------

TEST(ParseDmlBufferTensorDescTest, ValidInput) 
{
    Document d;
    d.Parse(R"({
        "x0": 
        { 
            "DataType": "DML_TENSOR_DATA_TYPE_FLOAT16", 
            "Flags": "DML_TENSOR_FLAG_OWNED_BY_DML",
            "DimensionCount": 2,
            "Sizes": [5,55],
            "Strides": [9,3],
            "TotalTensorSizeInBytes": 123,
            "GuaranteedBaseOffsetAlignment": 5
        }
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    auto result = ParseDmlBufferTensorDesc(d["x0"], allocator);
    EXPECT_EQ(result->DataType, DML_TENSOR_DATA_TYPE_FLOAT16);
    EXPECT_EQ(result->Flags, DML_TENSOR_FLAG_OWNED_BY_DML);
    EXPECT_EQ(result->DimensionCount, 2);
    EXPECT_EQ(result->Sizes[0], 5);
    EXPECT_EQ(result->Sizes[1], 55);
    EXPECT_EQ(result->Strides[0], 9);
    EXPECT_EQ(result->Strides[1], 3);
    EXPECT_EQ(result->TotalTensorSizeInBytes, 123);
    EXPECT_EQ(result->GuaranteedBaseOffsetAlignment, 5);
}

TEST(ParseDmlBufferTensorDescTest, DefaultValues) 
{
    Document d;
    d.Parse(R"({
        "x0": 
        { 
            "DataType": "DML_TENSOR_DATA_TYPE_FLOAT32", 
            "Sizes": [3,2,5]
        },
        "x1": 
        { 
            "DataType": "DML_TENSOR_DATA_TYPE_FLOAT32", 
            "Sizes": [3,2,5],
            "Strides": [0,0,1]
        }
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    auto x0 = ParseDmlBufferTensorDesc(d["x0"], allocator);
    EXPECT_EQ(x0->DataType, DML_TENSOR_DATA_TYPE_FLOAT32);
    EXPECT_EQ(x0->Flags, DML_TENSOR_FLAG_NONE);
    EXPECT_EQ(x0->DimensionCount, 3);
    EXPECT_EQ(x0->Sizes[0], 3);
    EXPECT_EQ(x0->Sizes[1], 2);
    EXPECT_EQ(x0->Sizes[2], 5);
    EXPECT_EQ(x0->Strides, nullptr);
    EXPECT_EQ(x0->TotalTensorSizeInBytes, DMLCalcBufferTensorSize(x0->DataType, x0->DimensionCount, x0->Sizes, x0->Strides));
    EXPECT_EQ(x0->GuaranteedBaseOffsetAlignment, 0);

    auto x1 = ParseDmlBufferTensorDesc(d["x1"], allocator);
    EXPECT_EQ(x1->DataType, DML_TENSOR_DATA_TYPE_FLOAT32);
    EXPECT_EQ(x1->Flags, DML_TENSOR_FLAG_NONE);
    EXPECT_EQ(x1->DimensionCount, 3);
    EXPECT_EQ(x1->Sizes[0], 3);
    EXPECT_EQ(x1->Sizes[1], 2);
    EXPECT_EQ(x1->Sizes[2], 5);
    EXPECT_EQ(x1->Strides[0], 0);
    EXPECT_EQ(x1->Strides[1], 0);
    EXPECT_EQ(x1->Strides[2], 1);
    EXPECT_EQ(x1->TotalTensorSizeInBytes, DMLCalcBufferTensorSize(x1->DataType, x1->DimensionCount, x1->Sizes, x1->Strides));
    EXPECT_EQ(x1->GuaranteedBaseOffsetAlignment, 0);
}

// ----------------------------------------------------------------------------
// DML_TENSOR_DESC
// ----------------------------------------------------------------------------

TEST(ParseDmlTensorDescTest, FullDesc) 
{
    Document d;
    d.Parse(R"({
        "x0": 
        { 
            "Type": "DML_TENSOR_TYPE_BUFFER",
            "Desc":
            {
                "DataType": "DML_TENSOR_DATA_TYPE_FLOAT16",
                "Flags": "DML_TENSOR_FLAG_OWNED_BY_DML",
                "DimensionCount": 2,
                "Sizes": [1,4],
                "Strides": [4,1],
                "TotalTensorSizeInBytes": 16,
                "GuaranteedBaseOffsetAlignment": 0
            }
        }
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    auto result = ParseDmlTensorDesc(d["x0"], allocator);
    EXPECT_EQ(result->Type, DML_TENSOR_TYPE_BUFFER);
    ASSERT_NE(result->Desc, nullptr);
    auto bufferDesc = static_cast<const DML_BUFFER_TENSOR_DESC*>(result->Desc);
    EXPECT_EQ(bufferDesc->DataType, DML_TENSOR_DATA_TYPE_FLOAT16);
    EXPECT_EQ(bufferDesc->Flags, DML_TENSOR_FLAG_OWNED_BY_DML);
    ASSERT_EQ(bufferDesc->DimensionCount, 2);
    EXPECT_EQ(bufferDesc->Sizes[0], 1);
    EXPECT_EQ(bufferDesc->Sizes[1], 4);
    EXPECT_EQ(bufferDesc->Strides[0], 4);
    EXPECT_EQ(bufferDesc->Strides[1], 1);
    EXPECT_EQ(bufferDesc->TotalTensorSizeInBytes, 16);
    EXPECT_EQ(bufferDesc->GuaranteedBaseOffsetAlignment, 0);
}

TEST(ParseDmlTensorDescTest, CollapsedDesc) 
{
    Document d;
    d.Parse(R"({
        "x0": 
        { 
            "Type": "DML_TENSOR_TYPE_BUFFER",
            "DataType": "DML_TENSOR_DATA_TYPE_FLOAT16",
            "Flags": "DML_TENSOR_FLAG_OWNED_BY_DML",
            "DimensionCount": 2,
            "Sizes": [1,4],
            "Strides": [4,1],
            "TotalTensorSizeInBytes": 16,
            "GuaranteedBaseOffsetAlignment": 0
        }
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    auto result = ParseDmlTensorDesc(d["x0"], allocator);
    EXPECT_EQ(result->Type, DML_TENSOR_TYPE_BUFFER);
    ASSERT_NE(result->Desc, nullptr);
    auto bufferDesc = static_cast<const DML_BUFFER_TENSOR_DESC*>(result->Desc);
    EXPECT_EQ(bufferDesc->DataType, DML_TENSOR_DATA_TYPE_FLOAT16);
    EXPECT_EQ(bufferDesc->Flags, DML_TENSOR_FLAG_OWNED_BY_DML);
    ASSERT_EQ(bufferDesc->DimensionCount, 2);
    EXPECT_EQ(bufferDesc->Sizes[0], 1);
    EXPECT_EQ(bufferDesc->Sizes[1], 4);
    EXPECT_EQ(bufferDesc->Strides[0], 4);
    EXPECT_EQ(bufferDesc->Strides[1], 1);
    EXPECT_EQ(bufferDesc->TotalTensorSizeInBytes, 16);
    EXPECT_EQ(bufferDesc->GuaranteedBaseOffsetAlignment, 0);
}

// ----------------------------------------------------------------------------
// DML_OPERATOR_DESC
// ----------------------------------------------------------------------------

TEST(ParseDmlOperatorDescTest, FullDesc) 
{
    Document d;
    d.Parse(R"({
        "x0":
        {
            "Type": "ELEMENT_WISE_ADD",
            "Desc":
            {
                "ATensor": { "DataType": "FLOAT32", "Sizes": [3] },
                "BTensor": { "DataType": "FLOAT32", "Sizes": [3] },
                "OutputTensor": { "DataType": "FLOAT32", "Sizes": [3] }
            }
        }
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    auto result = ParseDmlOperatorDesc(d["x0"], false, allocator);
    EXPECT_EQ(result->Type, DML_OPERATOR_ELEMENT_WISE_ADD);
    
    ASSERT_NE(result->Desc, nullptr);
    auto addDesc = static_cast<const DML_ELEMENT_WISE_ADD_OPERATOR_DESC*>(result->Desc);
    
    ASSERT_NE(addDesc->ATensor, nullptr);
    ASSERT_EQ(addDesc->ATensor->Type, DML_TENSOR_TYPE_BUFFER);
    auto aTensorDesc = static_cast<const DML_BUFFER_TENSOR_DESC*>(addDesc->ATensor->Desc);
    EXPECT_EQ(aTensorDesc->DataType, DML_TENSOR_DATA_TYPE_FLOAT32);
    ASSERT_EQ(aTensorDesc->DimensionCount, 1);
    EXPECT_EQ(aTensorDesc->Sizes[0], 3);

    ASSERT_NE(addDesc->BTensor, nullptr);
    ASSERT_EQ(addDesc->BTensor->Type, DML_TENSOR_TYPE_BUFFER);
    auto bTensorDesc = static_cast<const DML_BUFFER_TENSOR_DESC*>(addDesc->BTensor->Desc);
    EXPECT_EQ(bTensorDesc->DataType, DML_TENSOR_DATA_TYPE_FLOAT32);
    ASSERT_EQ(bTensorDesc->DimensionCount, 1);
    EXPECT_EQ(bTensorDesc->Sizes[0], 3);

    ASSERT_NE(addDesc->OutputTensor, nullptr);
    ASSERT_EQ(addDesc->OutputTensor->Type, DML_TENSOR_TYPE_BUFFER);
    auto outputTensorDesc = static_cast<const DML_BUFFER_TENSOR_DESC*>(addDesc->OutputTensor->Desc);
    EXPECT_EQ(outputTensorDesc->DataType, DML_TENSOR_DATA_TYPE_FLOAT32);
    ASSERT_EQ(outputTensorDesc->DimensionCount, 1);
    EXPECT_EQ(outputTensorDesc->Sizes[0], 3);
}

TEST(ParseDmlOperatorDescTest, CollapsedDesc) 
{
    Document d;
    d.Parse(R"({
        "x0":
        {
            "Type": "ELEMENT_WISE_ADD",
            "ATensor": { "DataType": "FLOAT32", "Sizes": [3] },
            "BTensor": { "DataType": "FLOAT32", "Sizes": [3] },
            "OutputTensor": { "DataType": "FLOAT32", "Sizes": [3] }
        }
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    auto result = ParseDmlOperatorDesc(d["x0"], false, allocator);
    EXPECT_EQ(result->Type, DML_OPERATOR_ELEMENT_WISE_ADD);
    
    ASSERT_NE(result->Desc, nullptr);
    auto addDesc = static_cast<const DML_ELEMENT_WISE_ADD_OPERATOR_DESC*>(result->Desc);
    
    ASSERT_NE(addDesc->ATensor, nullptr);
    ASSERT_EQ(addDesc->ATensor->Type, DML_TENSOR_TYPE_BUFFER);
    auto aTensorDesc = static_cast<const DML_BUFFER_TENSOR_DESC*>(addDesc->ATensor->Desc);
    EXPECT_EQ(aTensorDesc->DataType, DML_TENSOR_DATA_TYPE_FLOAT32);
    ASSERT_EQ(aTensorDesc->DimensionCount, 1);
    EXPECT_EQ(aTensorDesc->Sizes[0], 3);

    ASSERT_NE(addDesc->BTensor, nullptr);
    ASSERT_EQ(addDesc->BTensor->Type, DML_TENSOR_TYPE_BUFFER);
    auto bTensorDesc = static_cast<const DML_BUFFER_TENSOR_DESC*>(addDesc->BTensor->Desc);
    EXPECT_EQ(bTensorDesc->DataType, DML_TENSOR_DATA_TYPE_FLOAT32);
    ASSERT_EQ(bTensorDesc->DimensionCount, 1);
    EXPECT_EQ(bTensorDesc->Sizes[0], 3);

    ASSERT_NE(addDesc->OutputTensor, nullptr);
    ASSERT_EQ(addDesc->OutputTensor->Type, DML_TENSOR_TYPE_BUFFER);
    auto outputTensorDesc = static_cast<const DML_BUFFER_TENSOR_DESC*>(addDesc->OutputTensor->Desc);
    EXPECT_EQ(outputTensorDesc->DataType, DML_TENSOR_DATA_TYPE_FLOAT32);
    ASSERT_EQ(outputTensorDesc->DimensionCount, 1);
    EXPECT_EQ(outputTensorDesc->Sizes[0], 3);
}

// ----------------------------------------------------------------------------
// Model::Resource
// ----------------------------------------------------------------------------

TEST(ParseModelResourceDesc, BufferArrayInitializer) 
{
    Document d;
    d.Parse(R"({
        "initialValuesDataType": "FLOAT32",
        "initialValues": [1,2,3,4]
    })");
    ASSERT_FALSE(d.HasParseError());

    auto result = ParseModelResourceDesc("testFloatArray", "", d);
    EXPECT_EQ(result.name, "testFloatArray");
    ASSERT_TRUE(std::holds_alternative<Model::BufferDesc>(result.value));
    auto& desc = std::get<Model::BufferDesc>(result.value);
    EXPECT_EQ(desc.initialValuesDataType, DML_TENSOR_DATA_TYPE_FLOAT32);
    EXPECT_EQ(desc.initialValuesOffsetInBytes, 0);
    EXPECT_EQ(desc.sizeInBytes, 16);

    constexpr float expectedValues[] = {1,2,3,4};
    ASSERT_EQ(desc.initialValues.size(), sizeof(expectedValues));
    float* floatData = reinterpret_cast<float*>(desc.initialValues.data());
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        EXPECT_EQ(floatData[i], expectedValues[i]);
    }
}

TEST(ParseModelResourceDesc, BufferArrayAtValidOffset) 
{
    Document d;
    d.Parse(R"({
        "initialValuesDataType": "FLOAT32",
        "initialValues": [1,2,3,4],
        "initialValuesOffsetInBytes": 8,
        "sizeInBytes": 100
    })");
    ASSERT_FALSE(d.HasParseError());

    auto result = ParseModelResourceDesc("test", "", d);
    EXPECT_EQ(result.name, "test");
    ASSERT_TRUE(std::holds_alternative<Model::BufferDesc>(result.value));
    auto& desc = std::get<Model::BufferDesc>(result.value);
    EXPECT_EQ(desc.initialValuesDataType, DML_TENSOR_DATA_TYPE_FLOAT32);
    EXPECT_EQ(desc.initialValuesOffsetInBytes, 8);
    EXPECT_EQ(desc.sizeInBytes, 100);

    constexpr float expectedValues[] = {1,2,3,4};
    ASSERT_EQ(desc.initialValues.size(), sizeof(expectedValues));
    float* floatData = reinterpret_cast<float*>(desc.initialValues.data());
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        EXPECT_EQ(floatData[i], expectedValues[i]);
    }
}

TEST(ParseModelResourceDesc, BufferArrayAtInvalidOffset) 
{
    Document d;
    d.Parse(R"({
        "initialValuesDataType": "FLOAT32",
        "initialValues": [1,2,3,4],
        "initialValuesOffsetInBytes": 9,
        "sizeInBytes": 24
    })");
    ASSERT_FALSE(d.HasParseError());

    EXPECT_THROW(ParseModelResourceDesc("test", "", d), std::invalid_argument);
}

TEST(ParseModelResourceDesc, BufferArrayMixed) 
{
    Document d;
    d.Parse(R"({
        "initialValuesDataType": "UNKNOWN",
        "initialValues":
        [
            { "type": "FLOAT32", "value": -4.1092 },
            { "type": "UINT8", "value": 15 },
            { "type": "INT16", "value": -3 },
            { "type": "FLOAT32", "value": 55.1 }
        ]
    })");
    ASSERT_FALSE(d.HasParseError());

    auto result = ParseModelResourceDesc("testMixedArray", "", d);
    EXPECT_EQ(result.name, "testMixedArray");
    ASSERT_TRUE(std::holds_alternative<Model::BufferDesc>(result.value));
    auto& desc = std::get<Model::BufferDesc>(result.value);
    EXPECT_EQ(desc.initialValuesDataType, DML_TENSOR_DATA_TYPE_UNKNOWN);
    EXPECT_EQ(desc.initialValuesOffsetInBytes, 0);
    EXPECT_EQ(desc.sizeInBytes, 12); // should be rounded up from 11 to 12 bytes for 4-byte alignment

    ASSERT_LE(desc.initialValues.size(), desc.sizeInBytes);
    EXPECT_EQ(*reinterpret_cast<float*>(desc.initialValues.data()), -4.1092f);
    EXPECT_EQ(*reinterpret_cast<uint8_t*>(desc.initialValues.data() + 4), 15);
    EXPECT_EQ(*reinterpret_cast<int16_t*>(desc.initialValues.data() + 5), -3);
    EXPECT_EQ(*reinterpret_cast<float*>(desc.initialValues.data() + 7), 55.1f);
}

TEST(ParseModelResourceDesc, BufferConstantInitializer) 
{
    Document d;
    d.Parse(R"({
        "initialValuesDataType": "DML_TENSOR_DATA_TYPE_UINT32", 
        "initialValues": { "valueCount": 6, "value": 2 }
    })");
    ASSERT_FALSE(d.HasParseError());

    auto result = ParseModelResourceDesc("testUintConstant", "", d);
    EXPECT_EQ(result.name, "testUintConstant");
    ASSERT_TRUE(std::holds_alternative<Model::BufferDesc>(result.value));
    auto& desc = std::get<Model::BufferDesc>(result.value);
    EXPECT_EQ(desc.initialValuesDataType, DML_TENSOR_DATA_TYPE_UINT32);
    EXPECT_EQ(desc.initialValuesOffsetInBytes, 0);
    EXPECT_EQ(desc.sizeInBytes, 24);

    constexpr uint32_t expectedValues[] = {2,2,2,2,2,2};
    ASSERT_EQ(desc.initialValues.size(), sizeof(expectedValues));
    uint32_t* floatData = reinterpret_cast<uint32_t*>(desc.initialValues.data());
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        EXPECT_EQ(floatData[i], expectedValues[i]);
    }
}

TEST(ParseModelResourceDesc, BufferSequenceInitializer) 
{
    Document d;
    d.Parse(R"({
        "initialValuesDataType": "DML_TENSOR_DATA_TYPE_FLOAT32", 
        "initialValues": { "valueCount": 5, "valueStart": 3, "valueDelta": 2 }
    })");
    ASSERT_FALSE(d.HasParseError());

    auto result = ParseModelResourceDesc("testFloatSequence", "", d);
    EXPECT_EQ(result.name, "testFloatSequence");
    ASSERT_TRUE(std::holds_alternative<Model::BufferDesc>(result.value));
    auto& desc = std::get<Model::BufferDesc>(result.value);
    EXPECT_EQ(desc.initialValuesDataType, DML_TENSOR_DATA_TYPE_FLOAT32);
    EXPECT_EQ(desc.initialValuesOffsetInBytes, 0);
    EXPECT_EQ(desc.sizeInBytes, 20);

    constexpr float expectedValues[] = {3,5,7,9,11};
    ASSERT_EQ(desc.initialValues.size(), sizeof(expectedValues));
    float* floatData = reinterpret_cast<float*>(desc.initialValues.data());
    for (size_t i = 0; i < _countof(expectedValues); i++)
    {
        EXPECT_EQ(floatData[i], expectedValues[i]);
    }
}

// ----------------------------------------------------------------------------
// Model::DmlDispatchableDesc
// ----------------------------------------------------------------------------

TEST(ParseModelDispatchableDesc, DmlAdd) 
{
    Document d;
    d.Parse(R"({
        "type": "DML_OPERATOR_ELEMENT_WISE_ADD",
        "desc": 
        {
            "ATensor": { "DataType": "FLOAT32", "Sizes": [4] },
            "BTensor": { "DataType": "FLOAT32", "Sizes": [4] },
            "OutputTensor": { "DataType": "FLOAT32", "Sizes": [4] }
        }
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    auto result = ParseModelDispatchableDesc("add", "", d, allocator);
    ASSERT_EQ(result.name, "add");
    ASSERT_TRUE(std::holds_alternative<Model::DmlDispatchableDesc>(result.value));
    auto modelDmlOpDesc = std::get<Model::DmlDispatchableDesc>(result.value);

    ASSERT_EQ(modelDmlOpDesc.bindPoints.inputs.size(), 2);
    EXPECT_EQ(modelDmlOpDesc.bindPoints.inputs[0].name, "ATensor");
    EXPECT_EQ(modelDmlOpDesc.bindPoints.inputs[0].resourceCount, 1);
    EXPECT_TRUE(modelDmlOpDesc.bindPoints.inputs[0].required);
    EXPECT_EQ(modelDmlOpDesc.bindPoints.inputs[1].name, "BTensor");
    EXPECT_EQ(modelDmlOpDesc.bindPoints.inputs[1].resourceCount, 1);
    EXPECT_TRUE(modelDmlOpDesc.bindPoints.inputs[1].required);
    ASSERT_EQ(modelDmlOpDesc.bindPoints.outputs.size(), 1);
    EXPECT_EQ(modelDmlOpDesc.bindPoints.outputs[0].name, "OutputTensor");
    EXPECT_EQ(modelDmlOpDesc.bindPoints.outputs[0].resourceCount, 1);
    EXPECT_TRUE(modelDmlOpDesc.bindPoints.outputs[0].required);

    ASSERT_NE(modelDmlOpDesc.desc, nullptr);
    ASSERT_EQ(modelDmlOpDesc.desc->Type, DML_OPERATOR_ELEMENT_WISE_ADD);

    ASSERT_NE(modelDmlOpDesc.desc->Desc, nullptr);
    auto addDesc = static_cast<const DML_ELEMENT_WISE_ADD_OPERATOR_DESC*>(modelDmlOpDesc.desc->Desc);

    auto VerifyTensorDesc = [](const DML_TENSOR_DESC* desc)
    {
        EXPECT_EQ(desc->Type, DML_TENSOR_TYPE_BUFFER);
        ASSERT_NE(desc->Desc, nullptr);
        auto bufferDesc = static_cast<const DML_BUFFER_TENSOR_DESC*>(desc->Desc);
        EXPECT_EQ(bufferDesc->DataType, DML_TENSOR_DATA_TYPE_FLOAT32);
        EXPECT_EQ(bufferDesc->DimensionCount, 1);
        EXPECT_EQ(bufferDesc->Flags, DML_TENSOR_FLAG_NONE);
        EXPECT_EQ(bufferDesc->GuaranteedBaseOffsetAlignment, 0);
        EXPECT_EQ(bufferDesc->TotalTensorSizeInBytes, 16);
        EXPECT_EQ(bufferDesc->Sizes[0], 4);
        EXPECT_EQ(bufferDesc->Strides, nullptr);
    };

    VerifyTensorDesc(addDesc->ATensor);
    VerifyTensorDesc(addDesc->BTensor);
    VerifyTensorDesc(addDesc->OutputTensor);
}

TEST(ParseModelDispatchableDesc, HlslAdd) 
{
    Document d;
    d.Parse(R"({
        "type": "hlsl",
        "sourcePath": "c:/foo/bar/test.hlsl",
        "compiler": "dxc",
        "compilerArgs": 
        [
            "-T", "cs_6_0",
            "-E", "CSMain",
            "-D", "NUM_THREADS=4"
        ]
    })");
    ASSERT_FALSE(d.HasParseError());

    BucketAllocator allocator;
    auto result = ParseModelDispatchableDesc("my test operator", "", d, allocator);
    ASSERT_EQ(result.name, "my test operator");
    ASSERT_TRUE(std::holds_alternative<Model::HlslDispatchableDesc>(result.value));
    auto modelHlslOpDesc = std::get<Model::HlslDispatchableDesc>(result.value);

    EXPECT_EQ(modelHlslOpDesc.compiler, Model::HlslDispatchableDesc::Compiler::DXC);
    EXPECT_EQ(modelHlslOpDesc.sourcePath, "c:/foo/bar/test.hlsl");
    ASSERT_EQ(modelHlslOpDesc.compilerArgs.size(), 6);
    EXPECT_EQ(modelHlslOpDesc.compilerArgs[0], "-T");
    EXPECT_EQ(modelHlslOpDesc.compilerArgs[1], "cs_6_0");
    EXPECT_EQ(modelHlslOpDesc.compilerArgs[2], "-E");
    EXPECT_EQ(modelHlslOpDesc.compilerArgs[3], "CSMain");
    EXPECT_EQ(modelHlslOpDesc.compilerArgs[4], "-D");
    EXPECT_EQ(modelHlslOpDesc.compilerArgs[5], "NUM_THREADS=4");
}

// ----------------------------------------------------------------------------
// ParseModelCommand
// ----------------------------------------------------------------------------

TEST(ParseExecuteCommandTest, StringBindings) 
{
    Document d;
    d.Parse(R"({
        "type": "dispatch",
        "dispatchable": "add",
        "bindings": 
        {
            "ATensor": "A",
            "BTensor": "B",
            "OutputTensor": "Out"
        }
    })");
    ASSERT_FALSE(d.HasParseError());

    auto command = ParseModelCommand(d);
    ASSERT_TRUE(std::holds_alternative<Model::DispatchCommand>(command));
    auto& cmd = std::get<Model::DispatchCommand>(command);
    EXPECT_EQ(cmd.dispatchableName, "add");
    ASSERT_EQ(cmd.bindings.size(), 3);

    {
        auto binding = cmd.bindings.find("ATensor");
        ASSERT_TRUE(binding != cmd.bindings.end());
        ASSERT_EQ(binding->second.size(), 1);
        EXPECT_EQ(binding->second[0].name, "A");
        EXPECT_EQ(binding->second[0].elementCount, 0);
        EXPECT_EQ(binding->second[0].elementOffset, 0);
        EXPECT_EQ(binding->second[0].elementSizeInBytes, 0);
        EXPECT_EQ(binding->second[0].format, std::nullopt);
    }

    {
        auto binding = cmd.bindings.find("BTensor");
        ASSERT_TRUE(binding != cmd.bindings.end());
        ASSERT_EQ(binding->second.size(), 1);
        EXPECT_EQ(binding->second[0].name, "B");
        EXPECT_EQ(binding->second[0].elementCount, 0);
        EXPECT_EQ(binding->second[0].elementOffset, 0);
        EXPECT_EQ(binding->second[0].elementSizeInBytes, 0);
        EXPECT_EQ(binding->second[0].format, std::nullopt);
    }

    {
        auto binding = cmd.bindings.find("OutputTensor");
        ASSERT_TRUE(binding != cmd.bindings.end());
        ASSERT_EQ(binding->second.size(), 1);
        EXPECT_EQ(binding->second[0].name, "Out");
        EXPECT_EQ(binding->second[0].elementCount, 0);
        EXPECT_EQ(binding->second[0].elementOffset, 0);
        EXPECT_EQ(binding->second[0].elementSizeInBytes, 0);
        EXPECT_EQ(binding->second[0].format, std::nullopt);
    }
}

TEST(ParseExecuteCommandTest, ObjectBindings) 
{
    Document d;
    d.Parse(R"({
        "type": "dispatch",
        "dispatchable": "test",
        "bindings": 
        {
            "InputTensor":  { "name": "A", "elementCount": 1, "elementOffset": 8, "elementSizeInBytes": 4, "format": "R32_FLOAT" },
            "OutputTensor": { "name": "B", "elementCount": 4, "elementOffset": 2, "elementSizeInBytes": 4 }
        }
    })");
    ASSERT_FALSE(d.HasParseError());

    auto command = ParseModelCommand(d);
    ASSERT_TRUE(std::holds_alternative<Model::DispatchCommand>(command));
    auto& cmd = std::get<Model::DispatchCommand>(command);
    EXPECT_EQ(cmd.dispatchableName, "test");
    ASSERT_EQ(cmd.bindings.size(), 2);

    {
        auto binding = cmd.bindings.find("InputTensor");
        ASSERT_TRUE(binding != cmd.bindings.end());
        ASSERT_EQ(binding->second.size(), 1);
        EXPECT_EQ(binding->second[0].name, "A");
        EXPECT_EQ(binding->second[0].elementCount, 1);
        EXPECT_EQ(binding->second[0].elementOffset, 8);
        EXPECT_EQ(binding->second[0].elementSizeInBytes, 4);
        ASSERT_TRUE(binding->second[0].format.has_value());
        EXPECT_EQ(*binding->second[0].format, DXGI_FORMAT_R32_FLOAT);
    }

    {
        auto binding = cmd.bindings.find("OutputTensor");
        ASSERT_TRUE(binding != cmd.bindings.end());
        ASSERT_EQ(binding->second.size(), 1);
        EXPECT_EQ(binding->second[0].name, "B");
        EXPECT_EQ(binding->second[0].elementCount, 4);
        EXPECT_EQ(binding->second[0].elementOffset, 2);
        EXPECT_EQ(binding->second[0].elementSizeInBytes, 4);
        EXPECT_EQ(binding->second[0].format, std::nullopt);
    }
}

TEST(ParseExecuteCommandTest, ArrayBindings) 
{
    Document d;
    d.Parse(R"({
        "type": "dispatch",
        "dispatchable": "join",
        "bindings": 
        {
            "InputTensors": 
            [
                { "name": "A", "elementCount": 1, "elementOffset": 8, "elementSizeInBytes": 2, "format": "R16_FLOAT" },
                "B",
                { "name": "C", "elementCount": 3 }
            ],
            "OutputTensor": "Out"
        }
    })");
    ASSERT_FALSE(d.HasParseError());

    auto command = ParseModelCommand(d);
    ASSERT_TRUE(std::holds_alternative<Model::DispatchCommand>(command));
    auto& cmd = std::get<Model::DispatchCommand>(command);
    EXPECT_EQ(cmd.dispatchableName, "join");
    ASSERT_EQ(cmd.bindings.size(), 2);

    {
        auto binding = cmd.bindings.find("InputTensors");
        ASSERT_TRUE(binding != cmd.bindings.end());
        ASSERT_EQ(binding->second.size(), 3);

        EXPECT_EQ(binding->second[0].name, "A");
        EXPECT_EQ(binding->second[0].elementCount, 1);
        EXPECT_EQ(binding->second[0].elementOffset, 8);
        EXPECT_EQ(binding->second[0].elementSizeInBytes, 2);
        ASSERT_TRUE(binding->second[0].format.has_value());
        EXPECT_EQ(*binding->second[0].format, DXGI_FORMAT_R16_FLOAT);

        EXPECT_EQ(binding->second[1].name, "B");
        EXPECT_EQ(binding->second[1].elementCount, 0);
        EXPECT_EQ(binding->second[1].elementOffset, 0);
        EXPECT_EQ(binding->second[1].elementSizeInBytes, 0);
        EXPECT_EQ(binding->second[1].format, std::nullopt);

        EXPECT_EQ(binding->second[2].name, "C");
        EXPECT_EQ(binding->second[2].elementCount, 3);
        EXPECT_EQ(binding->second[2].elementOffset, 0);
        EXPECT_EQ(binding->second[2].elementSizeInBytes, 0);
        EXPECT_EQ(binding->second[2].format, std::nullopt);
    }

    {
        auto binding = cmd.bindings.find("OutputTensor");
        ASSERT_TRUE(binding != cmd.bindings.end());
        ASSERT_EQ(binding->second.size(), 1);
        EXPECT_EQ(binding->second[0].name, "Out");
        EXPECT_EQ(binding->second[0].elementCount, 0);
        EXPECT_EQ(binding->second[0].elementOffset, 0);
        EXPECT_EQ(binding->second[0].elementSizeInBytes, 0);
        EXPECT_EQ(binding->second[0].format, std::nullopt);
    }
}