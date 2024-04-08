#include "pch.h"
#include "JsonParsers.h"
#include "StdSupport.h"
#include "NpyReaderWriter.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

#ifndef WIN32
#define _stricmp strcasecmp
#endif

using Microsoft::WRL::ComPtr;

std::string RapidJsonToString(const rapidjson::Value& value)
{
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    value.Accept(writer);
    return buffer.GetString();
}

static uint32_t GetSizeInBytes(DML_TENSOR_DATA_TYPE dataType)
{
    switch (dataType)
    {
        case DML_TENSOR_DATA_TYPE_INT8:
        case DML_TENSOR_DATA_TYPE_UINT8:
            return 1;

        case DML_TENSOR_DATA_TYPE_FLOAT16:
        case DML_TENSOR_DATA_TYPE_INT16:
        case DML_TENSOR_DATA_TYPE_UINT16:
            return 2;

        case DML_TENSOR_DATA_TYPE_FLOAT32:
        case DML_TENSOR_DATA_TYPE_INT32:
        case DML_TENSOR_DATA_TYPE_UINT32:
            return 4;
    
        case DML_TENSOR_DATA_TYPE_FLOAT64:
        case DML_TENSOR_DATA_TYPE_INT64:
        case DML_TENSOR_DATA_TYPE_UINT64:
            return 8;

        default:
            throw std::invalid_argument("Unexpected DML_TENSOR_DATA_TYPE");
    }
}

DXGI_FORMAT ParseDxgiFormat(const rapidjson::Value& value)
{
    if (value.GetType() != rapidjson::Type::kStringType)
    {
        throw std::invalid_argument("DML_OPERATOR_TYPE must be a string.");
    }
    auto valueString = value.GetString();

    if (!strcmp(valueString, "DXGI_FORMAT_UNKNOWN") || !strcmp(valueString, "UNKNOWN")) { return DXGI_FORMAT_UNKNOWN; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G32B32A32_TYPELESS") || !strcmp(valueString, "R32G32B32A32_TYPELESS")) { return DXGI_FORMAT_R32G32B32A32_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G32B32A32_FLOAT") || !strcmp(valueString, "R32G32B32A32_FLOAT")) { return DXGI_FORMAT_R32G32B32A32_FLOAT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G32B32A32_UINT") || !strcmp(valueString, "R32G32B32A32_UINT")) { return DXGI_FORMAT_R32G32B32A32_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G32B32A32_SINT") || !strcmp(valueString, "R32G32B32A32_SINT")) { return DXGI_FORMAT_R32G32B32A32_SINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G32B32_TYPELESS") || !strcmp(valueString, "R32G32B32_TYPELESS")) { return DXGI_FORMAT_R32G32B32_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G32B32_FLOAT") || !strcmp(valueString, "R32G32B32_FLOAT")) { return DXGI_FORMAT_R32G32B32_FLOAT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G32B32_UINT") || !strcmp(valueString, "R32G32B32_UINT")) { return DXGI_FORMAT_R32G32B32_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G32B32_SINT") || !strcmp(valueString, "R32G32B32_SINT")) { return DXGI_FORMAT_R32G32B32_SINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16G16B16A16_TYPELESS") || !strcmp(valueString, "R16G16B16A16_TYPELESS")) { return DXGI_FORMAT_R16G16B16A16_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16G16B16A16_FLOAT") || !strcmp(valueString, "R16G16B16A16_FLOAT")) { return DXGI_FORMAT_R16G16B16A16_FLOAT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16G16B16A16_UNORM") || !strcmp(valueString, "R16G16B16A16_UNORM")) { return DXGI_FORMAT_R16G16B16A16_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16G16B16A16_UINT") || !strcmp(valueString, "R16G16B16A16_UINT")) { return DXGI_FORMAT_R16G16B16A16_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16G16B16A16_SNORM") || !strcmp(valueString, "R16G16B16A16_SNORM")) { return DXGI_FORMAT_R16G16B16A16_SNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16G16B16A16_SINT") || !strcmp(valueString, "R16G16B16A16_SINT")) { return DXGI_FORMAT_R16G16B16A16_SINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G32_TYPELESS") || !strcmp(valueString, "R32G32_TYPELESS")) { return DXGI_FORMAT_R32G32_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G32_FLOAT") || !strcmp(valueString, "R32G32_FLOAT")) { return DXGI_FORMAT_R32G32_FLOAT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G32_UINT") || !strcmp(valueString, "R32G32_UINT")) { return DXGI_FORMAT_R32G32_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G32_SINT") || !strcmp(valueString, "R32G32_SINT")) { return DXGI_FORMAT_R32G32_SINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32G8X24_TYPELESS") || !strcmp(valueString, "R32G8X24_TYPELESS")) { return DXGI_FORMAT_R32G8X24_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_D32_FLOAT_S8X24_UINT") || !strcmp(valueString, "D32_FLOAT_S8X24_UINT")) { return DXGI_FORMAT_D32_FLOAT_S8X24_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS") || !strcmp(valueString, "R32_FLOAT_X8X24_TYPELESS")) { return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_X32_TYPELESS_G8X24_UINT") || !strcmp(valueString, "X32_TYPELESS_G8X24_UINT")) { return DXGI_FORMAT_X32_TYPELESS_G8X24_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R10G10B10A2_TYPELESS") || !strcmp(valueString, "R10G10B10A2_TYPELESS")) { return DXGI_FORMAT_R10G10B10A2_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_R10G10B10A2_UNORM") || !strcmp(valueString, "R10G10B10A2_UNORM")) { return DXGI_FORMAT_R10G10B10A2_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R10G10B10A2_UINT") || !strcmp(valueString, "R10G10B10A2_UINT")) { return DXGI_FORMAT_R10G10B10A2_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R11G11B10_FLOAT") || !strcmp(valueString, "R11G11B10_FLOAT")) { return DXGI_FORMAT_R11G11B10_FLOAT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8G8B8A8_TYPELESS") || !strcmp(valueString, "R8G8B8A8_TYPELESS")) { return DXGI_FORMAT_R8G8B8A8_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8G8B8A8_UNORM") || !strcmp(valueString, "R8G8B8A8_UNORM")) { return DXGI_FORMAT_R8G8B8A8_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB") || !strcmp(valueString, "R8G8B8A8_UNORM_SRGB")) { return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8G8B8A8_UINT") || !strcmp(valueString, "R8G8B8A8_UINT")) { return DXGI_FORMAT_R8G8B8A8_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8G8B8A8_SNORM") || !strcmp(valueString, "R8G8B8A8_SNORM")) { return DXGI_FORMAT_R8G8B8A8_SNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8G8B8A8_SINT") || !strcmp(valueString, "R8G8B8A8_SINT")) { return DXGI_FORMAT_R8G8B8A8_SINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16G16_TYPELESS") || !strcmp(valueString, "R16G16_TYPELESS")) { return DXGI_FORMAT_R16G16_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16G16_FLOAT") || !strcmp(valueString, "R16G16_FLOAT")) { return DXGI_FORMAT_R16G16_FLOAT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16G16_UNORM") || !strcmp(valueString, "R16G16_UNORM")) { return DXGI_FORMAT_R16G16_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16G16_UINT") || !strcmp(valueString, "R16G16_UINT")) { return DXGI_FORMAT_R16G16_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16G16_SNORM") || !strcmp(valueString, "R16G16_SNORM")) { return DXGI_FORMAT_R16G16_SNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16G16_SINT") || !strcmp(valueString, "R16G16_SINT")) { return DXGI_FORMAT_R16G16_SINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32_TYPELESS") || !strcmp(valueString, "R32_TYPELESS")) { return DXGI_FORMAT_R32_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_D32_FLOAT") || !strcmp(valueString, "D32_FLOAT")) { return DXGI_FORMAT_D32_FLOAT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32_FLOAT") || !strcmp(valueString, "R32_FLOAT")) { return DXGI_FORMAT_R32_FLOAT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32_UINT") || !strcmp(valueString, "R32_UINT")) { return DXGI_FORMAT_R32_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R32_SINT") || !strcmp(valueString, "R32_SINT")) { return DXGI_FORMAT_R32_SINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R24G8_TYPELESS") || !strcmp(valueString, "R24G8_TYPELESS")) { return DXGI_FORMAT_R24G8_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_D24_UNORM_S8_UINT") || !strcmp(valueString, "D24_UNORM_S8_UINT")) { return DXGI_FORMAT_D24_UNORM_S8_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R24_UNORM_X8_TYPELESS") || !strcmp(valueString, "R24_UNORM_X8_TYPELESS")) { return DXGI_FORMAT_R24_UNORM_X8_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_X24_TYPELESS_G8_UINT") || !strcmp(valueString, "X24_TYPELESS_G8_UINT")) { return DXGI_FORMAT_X24_TYPELESS_G8_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8G8_TYPELESS") || !strcmp(valueString, "R8G8_TYPELESS")) { return DXGI_FORMAT_R8G8_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8G8_UNORM") || !strcmp(valueString, "R8G8_UNORM")) { return DXGI_FORMAT_R8G8_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8G8_UINT") || !strcmp(valueString, "R8G8_UINT")) { return DXGI_FORMAT_R8G8_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8G8_SNORM") || !strcmp(valueString, "R8G8_SNORM")) { return DXGI_FORMAT_R8G8_SNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8G8_SINT") || !strcmp(valueString, "R8G8_SINT")) { return DXGI_FORMAT_R8G8_SINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16_TYPELESS") || !strcmp(valueString, "R16_TYPELESS")) { return DXGI_FORMAT_R16_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16_FLOAT") || !strcmp(valueString, "R16_FLOAT")) { return DXGI_FORMAT_R16_FLOAT; }
    if (!strcmp(valueString, "DXGI_FORMAT_D16_UNORM") || !strcmp(valueString, "D16_UNORM")) { return DXGI_FORMAT_D16_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16_UNORM") || !strcmp(valueString, "R16_UNORM")) { return DXGI_FORMAT_R16_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16_UINT") || !strcmp(valueString, "R16_UINT")) { return DXGI_FORMAT_R16_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16_SNORM") || !strcmp(valueString, "R16_SNORM")) { return DXGI_FORMAT_R16_SNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R16_SINT") || !strcmp(valueString, "R16_SINT")) { return DXGI_FORMAT_R16_SINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8_TYPELESS") || !strcmp(valueString, "R8_TYPELESS")) { return DXGI_FORMAT_R8_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8_UNORM") || !strcmp(valueString, "R8_UNORM")) { return DXGI_FORMAT_R8_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8_UINT") || !strcmp(valueString, "R8_UINT")) { return DXGI_FORMAT_R8_UINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8_SNORM") || !strcmp(valueString, "R8_SNORM")) { return DXGI_FORMAT_R8_SNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8_SINT") || !strcmp(valueString, "R8_SINT")) { return DXGI_FORMAT_R8_SINT; }
    if (!strcmp(valueString, "DXGI_FORMAT_A8_UNORM") || !strcmp(valueString, "A8_UNORM")) { return DXGI_FORMAT_A8_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R1_UNORM") || !strcmp(valueString, "R1_UNORM")) { return DXGI_FORMAT_R1_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R9G9B9E5_SHAREDEXP") || !strcmp(valueString, "R9G9B9E5_SHAREDEXP")) { return DXGI_FORMAT_R9G9B9E5_SHAREDEXP; }
    if (!strcmp(valueString, "DXGI_FORMAT_R8G8_B8G8_UNORM") || !strcmp(valueString, "R8G8_B8G8_UNORM")) { return DXGI_FORMAT_R8G8_B8G8_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_G8R8_G8B8_UNORM") || !strcmp(valueString, "G8R8_G8B8_UNORM")) { return DXGI_FORMAT_G8R8_G8B8_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC1_TYPELESS") || !strcmp(valueString, "BC1_TYPELESS")) { return DXGI_FORMAT_BC1_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC1_UNORM") || !strcmp(valueString, "BC1_UNORM")) { return DXGI_FORMAT_BC1_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC1_UNORM_SRGB") || !strcmp(valueString, "BC1_UNORM_SRGB")) { return DXGI_FORMAT_BC1_UNORM_SRGB; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC2_TYPELESS") || !strcmp(valueString, "BC2_TYPELESS")) { return DXGI_FORMAT_BC2_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC2_UNORM") || !strcmp(valueString, "BC2_UNORM")) { return DXGI_FORMAT_BC2_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC2_UNORM_SRGB") || !strcmp(valueString, "BC2_UNORM_SRGB")) { return DXGI_FORMAT_BC2_UNORM_SRGB; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC3_TYPELESS") || !strcmp(valueString, "BC3_TYPELESS")) { return DXGI_FORMAT_BC3_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC3_UNORM") || !strcmp(valueString, "BC3_UNORM")) { return DXGI_FORMAT_BC3_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC3_UNORM_SRGB") || !strcmp(valueString, "BC3_UNORM_SRGB")) { return DXGI_FORMAT_BC3_UNORM_SRGB; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC4_TYPELESS") || !strcmp(valueString, "BC4_TYPELESS")) { return DXGI_FORMAT_BC4_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC4_UNORM") || !strcmp(valueString, "BC4_UNORM")) { return DXGI_FORMAT_BC4_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC4_SNORM") || !strcmp(valueString, "BC4_SNORM")) { return DXGI_FORMAT_BC4_SNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC5_TYPELESS") || !strcmp(valueString, "BC5_TYPELESS")) { return DXGI_FORMAT_BC5_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC5_UNORM") || !strcmp(valueString, "BC5_UNORM")) { return DXGI_FORMAT_BC5_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC5_SNORM") || !strcmp(valueString, "BC5_SNORM")) { return DXGI_FORMAT_BC5_SNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_B5G6R5_UNORM") || !strcmp(valueString, "B5G6R5_UNORM")) { return DXGI_FORMAT_B5G6R5_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_B5G5R5A1_UNORM") || !strcmp(valueString, "B5G5R5A1_UNORM")) { return DXGI_FORMAT_B5G5R5A1_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_B8G8R8A8_UNORM") || !strcmp(valueString, "B8G8R8A8_UNORM")) { return DXGI_FORMAT_B8G8R8A8_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_B8G8R8X8_UNORM") || !strcmp(valueString, "B8G8R8X8_UNORM")) { return DXGI_FORMAT_B8G8R8X8_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM") || !strcmp(valueString, "R10G10B10_XR_BIAS_A2_UNORM")) { return DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_B8G8R8A8_TYPELESS") || !strcmp(valueString, "B8G8R8A8_TYPELESS")) { return DXGI_FORMAT_B8G8R8A8_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_B8G8R8A8_UNORM_SRGB") || !strcmp(valueString, "B8G8R8A8_UNORM_SRGB")) { return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB; }
    if (!strcmp(valueString, "DXGI_FORMAT_B8G8R8X8_TYPELESS") || !strcmp(valueString, "B8G8R8X8_TYPELESS")) { return DXGI_FORMAT_B8G8R8X8_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_B8G8R8X8_UNORM_SRGB") || !strcmp(valueString, "B8G8R8X8_UNORM_SRGB")) { return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC6H_TYPELESS") || !strcmp(valueString, "BC6H_TYPELESS")) { return DXGI_FORMAT_BC6H_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC6H_UF16") || !strcmp(valueString, "BC6H_UF16")) { return DXGI_FORMAT_BC6H_UF16; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC6H_SF16") || !strcmp(valueString, "BC6H_SF16")) { return DXGI_FORMAT_BC6H_SF16; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC7_TYPELESS") || !strcmp(valueString, "BC7_TYPELESS")) { return DXGI_FORMAT_BC7_TYPELESS; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC7_UNORM") || !strcmp(valueString, "BC7_UNORM")) { return DXGI_FORMAT_BC7_UNORM; }
    if (!strcmp(valueString, "DXGI_FORMAT_BC7_UNORM_SRGB") || !strcmp(valueString, "BC7_UNORM_SRGB")) { return DXGI_FORMAT_BC7_UNORM_SRGB; }
    if (!strcmp(valueString, "DXGI_FORMAT_AYUV") || !strcmp(valueString, "AYUV")) { return DXGI_FORMAT_AYUV; }
    if (!strcmp(valueString, "DXGI_FORMAT_Y410") || !strcmp(valueString, "Y410")) { return DXGI_FORMAT_Y410; }
    if (!strcmp(valueString, "DXGI_FORMAT_Y416") || !strcmp(valueString, "Y416")) { return DXGI_FORMAT_Y416; }
    if (!strcmp(valueString, "DXGI_FORMAT_NV12") || !strcmp(valueString, "NV12")) { return DXGI_FORMAT_NV12; }
    if (!strcmp(valueString, "DXGI_FORMAT_P010") || !strcmp(valueString, "P010")) { return DXGI_FORMAT_P010; }
    if (!strcmp(valueString, "DXGI_FORMAT_P016") || !strcmp(valueString, "P016")) { return DXGI_FORMAT_P016; }
    if (!strcmp(valueString, "DXGI_FORMAT_420_OPAQUE") || !strcmp(valueString, "420_OPAQUE")) { return DXGI_FORMAT_420_OPAQUE; }
    if (!strcmp(valueString, "DXGI_FORMAT_YUY2") || !strcmp(valueString, "YUY2")) { return DXGI_FORMAT_YUY2; }
    if (!strcmp(valueString, "DXGI_FORMAT_Y210") || !strcmp(valueString, "Y210")) { return DXGI_FORMAT_Y210; }
    if (!strcmp(valueString, "DXGI_FORMAT_Y216") || !strcmp(valueString, "Y216")) { return DXGI_FORMAT_Y216; }
    if (!strcmp(valueString, "DXGI_FORMAT_NV11") || !strcmp(valueString, "NV11")) { return DXGI_FORMAT_NV11; }
    if (!strcmp(valueString, "DXGI_FORMAT_AI44") || !strcmp(valueString, "AI44")) { return DXGI_FORMAT_AI44; }
    if (!strcmp(valueString, "DXGI_FORMAT_IA44") || !strcmp(valueString, "IA44")) { return DXGI_FORMAT_IA44; }
    if (!strcmp(valueString, "DXGI_FORMAT_P8") || !strcmp(valueString, "P8")) { return DXGI_FORMAT_P8; }
    if (!strcmp(valueString, "DXGI_FORMAT_A8P8") || !strcmp(valueString, "A8P8")) { return DXGI_FORMAT_A8P8; }
    if (!strcmp(valueString, "DXGI_FORMAT_B4G4R4A4_UNORM") || !strcmp(valueString, "B4G4R4A4_UNORM")) { return DXGI_FORMAT_B4G4R4A4_UNORM; }

    throw std::invalid_argument("Unrecognized DXGI format");
}

template <typename TReturn>
TReturn ParseFieldHelper(
    const rapidjson::Value& object, 
    std::string_view fieldName, 
    bool required,
    TReturn defaultValue,
    std::function<TReturn(const rapidjson::Value&)> func)
{
    auto fieldIterator = object.FindMember(fieldName.data());
    if (fieldIterator == object.MemberEnd())
    {
        if (required)
        { 
            throw std::invalid_argument(fmt::format("Field '{}' is required.", fieldName)); 
        }
        return defaultValue;
    }

    try
    {
        return func(fieldIterator->value);
    }
    catch (const std::exception& e)
    {
        throw std::invalid_argument(fmt::format("Error parsing field '{}': {}", fieldName, e.what()));
    }
}

template <typename T>
T ParseFloatingPointNumber(const rapidjson::Value& value)
{
    static_assert(std::is_floating_point_v<T> || std::is_same_v<T, half_float::half>);
    if (value.IsFloat() || value.IsDouble() || value.IsLosslessDouble())
    {
        return static_cast<T>(value.GetDouble());
    }
    else if (value.IsString())
    {
        auto strValue = value.GetString();
        if (!_stricmp(strValue, "inf")) { return std::numeric_limits<T>::infinity(); }
        if (!_stricmp(strValue, "-inf")) { return -std::numeric_limits<T>::infinity(); }
        if (!_stricmp(strValue, "nan")) { return std::numeric_limits<T>::quiet_NaN(); }
        throw std::invalid_argument("Expected 'NaN', 'Inf', or '-Inf'.");
    }
    else
    {
        throw std::invalid_argument("Expected a number or 'NaN', 'Inf', or '-Inf'.");
    }
}

template <typename T> 
gsl::span<T> ParseArray(
    const rapidjson::Value& value, 
    BucketAllocator& allocator, 
    std::function<T(const rapidjson::Value&)> elementParser)
{
    if (value.GetType() != rapidjson::Type::kArrayType)
    {
        throw std::invalid_argument("Expected an array.");
    }

    auto valueArray = value.GetArray();
    auto outputElements = allocator.Allocate<T>(valueArray.Size());
    for (uint32_t i = 0; i < valueArray.Size(); i++)
    {
        outputElements[i] = elementParser(valueArray[i]);
    }

    return gsl::make_span(outputElements, valueArray.Size());
}

template <typename T> 
std::vector<T> ParseArrayAsVector(
    const rapidjson::Value& value, 
    std::function<T(const rapidjson::Value&)> elementParser)
{
    if (value.GetType() != rapidjson::Type::kArrayType)
    {
        throw std::invalid_argument("Expected an array.");
    }

    auto valueArray = value.GetArray();
    std::vector<T> outputElements(valueArray.Size());
    for (uint32_t i = 0; i < valueArray.Size(); i++)
    {
        outputElements[i] = elementParser(valueArray[i]);
    }

    return outputElements;
}

template <typename T>
std::vector<std::byte> ParseArrayAsBytes(
    const rapidjson::Value& value,
    std::function<T(const rapidjson::Value&)> elementParser)
{
    if (value.GetType() != rapidjson::Type::kArrayType)
    {
        throw std::invalid_argument("Expected an array.");
    }

    std::vector<std::byte> output;
    for (auto& element : value.GetArray())
    {
        T elementValue = elementParser(element);
        for (auto byte : gsl::as_bytes(gsl::make_span<T>(&elementValue, 1)))
        {
            output.push_back(byte);
        }
    }

    return output;
}

// Helper for parsing flags from a JSON string or array. Flags are enums that may be bitwise-OR'd together
// to create a mask. This helper takes a function that parses a single flag value from a JSON string value.
// For example, if the flag enums are defined as "enum FOO {A=0x1,B=0x2,C=x4}" then the parser should be
// able to convert "A" to A, "B" to B, and "C" to C.
template <typename T>
T ParseFlags(const rapidjson::Value& value, std::function<T(const rapidjson::Value& value)> flagParser)
{
    if (value.GetType() == rapidjson::Type::kStringType)
    {
        return flagParser(value);
    }
    else if (value.GetType() == rapidjson::Type::kArrayType)
    {
        T flags = {};
        for (auto& elementValue : value.GetArray())
        {
            flags |= flagParser(elementValue);
        }
        return flags;
    }

    throw std::invalid_argument("Expected a string or an array of strings.");
}

template <typename T>
T* AsPointer(gsl::span<T> s) 
{ 
    return s.empty() ? nullptr : s.data();
}

namespace JsonParsers
{
// ----------------------------------------------------------------------------
// STRING
// ----------------------------------------------------------------------------

std::string ParseString(const rapidjson::Value& value)
{
    if (!value.IsString())
    {
        throw std::invalid_argument("Expected a string.");
    }
    return value.GetString();
}

std::string ParseStringField(const rapidjson::Value& object, std::string_view fieldName, bool required = true, std::string defaultValue = {})
{
    return ParseFieldHelper<std::string>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseString(value); 
    });
}

// ----------------------------------------------------------------------------
// BOOL
// ----------------------------------------------------------------------------

bool ParseBool(const rapidjson::Value& value)
{
    if (!value.IsBool())
    {
        throw std::invalid_argument("Expected a bool.");
    }
    return value.GetBool();
}

bool ParseBoolField(const rapidjson::Value& object, std::string_view fieldName, bool required, bool defaultValue)
{
    return ParseFieldHelper<bool>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseBool(value); 
    });
}

// ----------------------------------------------------------------------------
// FLOAT16
// ----------------------------------------------------------------------------

half_float::half ParseFloat16(const rapidjson::Value& value)
{
    auto parsedValue = ParseFloatingPointNumber<float>(value);
    return half_float::half(parsedValue);
}

half_float::half ParseFloat16Field(const rapidjson::Value& object, std::string_view fieldName, bool required, half_float::half defaultValue)
{
    return ParseFieldHelper<half_float::half>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseFloat16(value); 
    });
}

gsl::span<half_float::half> ParseFloat16Array(const rapidjson::Value& value, BucketAllocator& allocator)
{
    return ParseArray<half_float::half>(value, allocator, ParseFloat16);
}

gsl::span<half_float::half> ParseFloat16ArrayField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required, gsl::span<half_float::half> defaultValue)
{
    return ParseFieldHelper<gsl::span<half_float::half>>(object, fieldName, required, defaultValue, [&allocator](auto& value){ 
        return ParseFloat16Array(value, allocator); 
    });
}

// ----------------------------------------------------------------------------
// FLOAT32
// ----------------------------------------------------------------------------

float ParseFloat32(const rapidjson::Value& value)
{
    return ParseFloatingPointNumber<float>(value);
}

float ParseFloat32Field(const rapidjson::Value& object, std::string_view fieldName, bool required, float defaultValue)
{
    return ParseFieldHelper<float>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseFloat32(value); 
    });
}

gsl::span<float> ParseFloat32Array(const rapidjson::Value& value, BucketAllocator& allocator)
{
    return ParseArray<float>(value, allocator, ParseFloat32);
}

gsl::span<float> ParseFloat32ArrayField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required, gsl::span<float> defaultValue)
{
    return ParseFieldHelper<gsl::span<float>>(object, fieldName, required, defaultValue, [&allocator](auto& value){ 
        return ParseFloat32Array(value, allocator); 
    });
}

// ----------------------------------------------------------------------------
// FLOAT64
// ----------------------------------------------------------------------------

double ParseFloat64(const rapidjson::Value& value)
{
    return ParseFloatingPointNumber<double>(value);
}

double ParseFloat64Field(const rapidjson::Value& object, std::string_view fieldName, bool required, double defaultValue)
{
    return ParseFieldHelper<double>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseFloat64(value); 
    });
}

// ----------------------------------------------------------------------------
// INT8
// ----------------------------------------------------------------------------

int8_t ParseInt8(const rapidjson::Value& value)
{
    if (!value.IsInt64())
    {
        throw std::invalid_argument("Expected a signed integer.");
    }
    return gsl::narrow<int8_t>(value.GetInt64());
}

int8_t ParseInt8Field(const rapidjson::Value& object, std::string_view fieldName, bool required, int8_t defaultValue)
{
    return ParseFieldHelper<int8_t>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseInt8(value); 
    });
}

// ----------------------------------------------------------------------------
// INT16
// ----------------------------------------------------------------------------

int16_t ParseInt16(const rapidjson::Value& value)
{
    if (!value.IsInt64())
    {
        throw std::invalid_argument("Expected a signed integer.");
    }
    return gsl::narrow<int16_t>(value.GetInt64());
}

int16_t ParseInt16Field(const rapidjson::Value& object, std::string_view fieldName, bool required, int16_t defaultValue)
{
    return ParseFieldHelper<int16_t>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseInt16(value); 
    });
}

// ----------------------------------------------------------------------------
// INT32
// ----------------------------------------------------------------------------

int32_t ParseInt32(const rapidjson::Value& value)
{
    if (!value.IsInt64())
    {
        throw std::invalid_argument("Expected a signed integer.");
    }
    return gsl::narrow<int32_t>(value.GetInt64());
}

int32_t ParseInt32Field(const rapidjson::Value& object, std::string_view fieldName, bool required, int32_t defaultValue)
{
    return ParseFieldHelper<int32_t>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseInt32(value); 
    });
}

gsl::span<int32_t> ParseInt32Array(const rapidjson::Value& value, BucketAllocator& allocator)
{
    return ParseArray<int32_t>(value, allocator, ParseInt32);
}

gsl::span<int32_t> ParseInt32ArrayField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required, gsl::span<int32_t> defaultValue)
{
    return ParseFieldHelper<gsl::span<int32_t>>(object, fieldName, required, defaultValue, [&allocator](auto& value){ 
        return ParseInt32Array(value, allocator); 
    });
}

// ----------------------------------------------------------------------------
// INT64
// ----------------------------------------------------------------------------

int64_t ParseInt64(const rapidjson::Value& value)
{
    if (!value.IsInt64())
    {
        throw std::invalid_argument("Expected a signed integer.");
    }
    return value.GetInt64();
}

int64_t ParseInt64Field(const rapidjson::Value& object, std::string_view fieldName, bool required, int64_t defaultValue)
{
    return ParseFieldHelper<int64_t>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseInt64(value); 
    });
}

std::vector<int64_t> ParseInt64ArrayAsVector(const rapidjson::Value& object)
{
    return ParseArrayAsVector<int64_t>(object, ParseInt64);
}

std::vector<int64_t> ParseInt64ArrayAsVectorField(const rapidjson::Value& object, std::string_view fieldName, bool required, std::vector<int64_t> defaultValue)
{
    return ParseFieldHelper<std::vector<int64_t>>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseInt64ArrayAsVector(value); 
    });
}

// ----------------------------------------------------------------------------
// UINT8
// ----------------------------------------------------------------------------

uint8_t ParseUInt8(const rapidjson::Value& value)
{
    if (!value.IsUint64())
    {
        throw std::invalid_argument("Expected an unsigned integer.");
    }
    return gsl::narrow<uint8_t>(value.GetUint64());
}

uint8_t ParseUInt8Field(const rapidjson::Value& object, std::string_view fieldName, bool required, uint8_t defaultValue)
{
    return ParseFieldHelper<uint8_t>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseUInt8(value); 
    });
}

// ----------------------------------------------------------------------------
// UINT16
// ----------------------------------------------------------------------------

uint16_t ParseUInt16(const rapidjson::Value& value)
{
    if (!value.IsUint64())
    {
        throw std::invalid_argument("Expected an unsigned integer.");
    }
    return gsl::narrow<uint16_t>(value.GetUint64());
}

uint16_t ParseUInt16Field(const rapidjson::Value& object, std::string_view fieldName, bool required, uint16_t defaultValue)
{
    return ParseFieldHelper<uint16_t>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseUInt16(value); 
    });
}

// ----------------------------------------------------------------------------
// UINT32
// ----------------------------------------------------------------------------

uint32_t ParseUInt32(const rapidjson::Value& value)
{
    if (!value.IsUint64())
    {
        throw std::invalid_argument("Expected an unsigned integer.");
    }
    return gsl::narrow<uint32_t>(value.GetUint64());
}

uint32_t ParseUInt32Field(const rapidjson::Value& object, std::string_view fieldName, bool required, uint32_t defaultValue)
{
    return ParseFieldHelper<uint32_t>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseUInt32(value); 
    });
}

gsl::span<uint32_t> ParseUInt32Array(const rapidjson::Value& value, BucketAllocator& allocator)
{
    return ParseArray<uint32_t>(value, allocator, ParseUInt32);
}

gsl::span<uint32_t> ParseUInt32ArrayField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required, gsl::span<uint32_t> defaultValue)
{
    return ParseFieldHelper<gsl::span<uint32_t>>(object, fieldName, required, defaultValue, [&allocator](auto& value){ 
        return ParseUInt32Array(value, allocator); 
    });
}

// ----------------------------------------------------------------------------
// UINT64
// ----------------------------------------------------------------------------

uint64_t ParseUInt64(const rapidjson::Value& value)
{
    if (!value.IsUint64())
    {
        throw std::invalid_argument("Expected an unsigned integer.");
    }
    return value.GetUint64();
}

uint64_t ParseUInt64Field(const rapidjson::Value& object, std::string_view fieldName, bool required, uint64_t defaultValue)
{
    return ParseFieldHelper<uint64_t>(object, fieldName, required, defaultValue, [](auto& value){ 
        return ParseUInt64(value); 
    });
}

// ----------------------------------------------------------------------------
// Mixed Primitives
// ----------------------------------------------------------------------------

template <typename T>
void PushBytes(const T& value, std::vector<std::byte>& outputBuffer)
{
    for (auto& byte : gsl::as_bytes(gsl::make_span<const T>(&value, 1)))
    {
        outputBuffer.push_back(byte);
    }
}

std::vector<std::byte> ParseMixedPrimitiveArray(const rapidjson::Value& object)
{
    if (object.GetType() != rapidjson::Type::kArrayType)
    {
        throw std::invalid_argument("Expected an array.");
    }

    std::vector<std::byte> data;
    for (auto& element : object.GetArray())
    {
        if (element.GetType() != rapidjson::Type::kObjectType)
        {
            throw std::invalid_argument("Expected an object.");
        }

        auto elementType = ParseDmlTensorDataTypeField(element, "type");
        switch (elementType)
        {
        case DML_TENSOR_DATA_TYPE_FLOAT32: PushBytes(ParseFloat32Field(element, "value"), data); break;
        case DML_TENSOR_DATA_TYPE_FLOAT64: PushBytes(ParseFloat64Field(element, "value"), data); break;
        case DML_TENSOR_DATA_TYPE_UINT8: PushBytes(ParseUInt8Field(element, "value"), data); break;
        case DML_TENSOR_DATA_TYPE_UINT16: PushBytes(ParseUInt16Field(element, "value"), data); break;
        case DML_TENSOR_DATA_TYPE_UINT32: PushBytes(ParseUInt32Field(element, "value"), data); break;
        case DML_TENSOR_DATA_TYPE_UINT64: PushBytes(ParseUInt64Field(element, "value"), data); break;
        case DML_TENSOR_DATA_TYPE_INT8: PushBytes(ParseInt8Field(element, "value"), data); break;
        case DML_TENSOR_DATA_TYPE_INT16: PushBytes(ParseInt16Field(element, "value"), data); break;
        case DML_TENSOR_DATA_TYPE_INT32: PushBytes(ParseInt32Field(element, "value"), data); break;
        case DML_TENSOR_DATA_TYPE_INT64: PushBytes(ParseInt64Field(element, "value"), data); break;
        default: throw std::invalid_argument("Data type not supported.");
        }
    }

    return data;
}

// ----------------------------------------------------------------------------
// DML_SIZE_2D
// ----------------------------------------------------------------------------
static void ParseDmlSize2d(const rapidjson::Value& value, DML_SIZE_2D& returnValue)
{
    if (!value.IsObject())
    {
        throw std::invalid_argument("Expected a non-null JSON object.");
    }
    returnValue.Width = ParseUInt32Field(value, "Width");
    returnValue.Height = ParseUInt32Field(value, "Height");
}

DML_SIZE_2D ParseDmlSize2d(const rapidjson::Value& value)
{
    DML_SIZE_2D returnValue = {};
    ParseDmlSize2d(value, returnValue);
    return returnValue;
}

DML_SIZE_2D* ParseDmlSize2d(const rapidjson::Value& value, BucketAllocator& allocator)
{
    auto returnValue = allocator.Allocate<DML_SIZE_2D>();
    ParseDmlSize2d(value, *returnValue);
    return returnValue;
}

DML_SIZE_2D* ParseDmlSize2dField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required, DML_SIZE_2D* defaultValue)
{
    return ParseFieldHelper<DML_SIZE_2D*>(object, fieldName, required, defaultValue, [&allocator](auto& value){ 
        return ParseDmlSize2d(value, allocator); 
    });
}

// ----------------------------------------------------------------------------
// DML_SCALAR_UNION
// ----------------------------------------------------------------------------
static void ParseDmlScalarUnion(const rapidjson::Value& value, DML_TENSOR_DATA_TYPE dataType, DML_SCALAR_UNION& returnValue)
{
    if (value.IsObject())
    {
        switch (dataType)
        {
        case DML_TENSOR_DATA_TYPE_FLOAT16: returnValue.UInt16 = ParseUInt16Field(value, "UInt16"); break;
        case DML_TENSOR_DATA_TYPE_FLOAT32: returnValue.Float32 = ParseFloat32Field(value, "Float32"); break;
        case DML_TENSOR_DATA_TYPE_FLOAT64: returnValue.Float64 = ParseFloat64Field(value, "Float64"); break;
        case DML_TENSOR_DATA_TYPE_UINT8: returnValue.UInt8 = ParseUInt8Field(value, "UInt8"); break;
        case DML_TENSOR_DATA_TYPE_UINT16: returnValue.UInt16 = ParseUInt16Field(value, "UInt16"); break;
        case DML_TENSOR_DATA_TYPE_UINT32: returnValue.UInt32 = ParseUInt32Field(value, "UInt32"); break;
        case DML_TENSOR_DATA_TYPE_UINT64: returnValue.UInt64 = ParseUInt64Field(value, "UInt64"); break;
        case DML_TENSOR_DATA_TYPE_INT8: returnValue.Int8 = ParseInt8Field(value, "Int8"); break;
        case DML_TENSOR_DATA_TYPE_INT16: returnValue.Int16 = ParseInt16Field(value, "Int16"); break;
        case DML_TENSOR_DATA_TYPE_INT32: returnValue.Int32 = ParseInt32Field(value, "Int32"); break;
        case DML_TENSOR_DATA_TYPE_INT64: returnValue.Int64 = ParseInt64Field(value, "Int64"); break;
        default: throw std::invalid_argument("Data type not supported for DML_SCALAR_UNION.");
        }
    }
    else if (value.IsNumber())
    {
        switch (dataType)
        {
        case DML_TENSOR_DATA_TYPE_FLOAT16:
        {
            auto halfValue = ParseFloat16(value);
            returnValue.UInt16 = *reinterpret_cast<const uint16_t*>(&halfValue);
            break;
        }
        case DML_TENSOR_DATA_TYPE_FLOAT32: returnValue.Float32 = ParseFloat32(value); break;
        case DML_TENSOR_DATA_TYPE_FLOAT64: returnValue.Float64 = ParseFloat64(value); break;
        case DML_TENSOR_DATA_TYPE_UINT8: returnValue.UInt8 = ParseUInt8(value); break;
        case DML_TENSOR_DATA_TYPE_UINT16: returnValue.UInt16 = ParseUInt16(value); break;
        case DML_TENSOR_DATA_TYPE_UINT32: returnValue.UInt32 = ParseUInt32(value); break;
        case DML_TENSOR_DATA_TYPE_UINT64: returnValue.UInt64 = ParseUInt64(value); break;
        case DML_TENSOR_DATA_TYPE_INT8: returnValue.Int8 = ParseInt8(value); break;
        case DML_TENSOR_DATA_TYPE_INT16: returnValue.Int16 = ParseInt16(value); break;
        case DML_TENSOR_DATA_TYPE_INT32: returnValue.Int32 = ParseInt32(value); break;
        case DML_TENSOR_DATA_TYPE_INT64: returnValue.Int64 = ParseInt64(value); break;
        default: throw std::invalid_argument("Data type not supported for DML_SCALAR_UNION.");
        }
    }
    else
    {
        throw std::invalid_argument("Expected a non-null JSON object or number.");
    }
}

DML_SCALAR_UNION ParseDmlScalarUnion(const rapidjson::Value& value, DML_TENSOR_DATA_TYPE dataType)
{
    DML_SCALAR_UNION returnValue{};
    ParseDmlScalarUnion(value, dataType, returnValue);
    return returnValue;
}

DML_SCALAR_UNION* ParseDmlScalarUnion(const rapidjson::Value& value, DML_TENSOR_DATA_TYPE dataType, BucketAllocator& allocator)
{
    auto returnValue = allocator.Allocate<DML_SCALAR_UNION>();
    ParseDmlScalarUnion(value, dataType, *returnValue);
    return returnValue;
}

DML_SCALAR_UNION* ParseDmlScalarUnionField(
    const rapidjson::Value& object, 
    std::string_view scalarUnionFieldName, 
    std::string_view dataTypeFieldName,
    BucketAllocator& allocator, 
    bool required,
    DML_SCALAR_UNION* defaultValue)
{
    auto dataType = ParseDmlTensorDataTypeField(object, dataTypeFieldName, required);
    return ParseFieldHelper<DML_SCALAR_UNION*>(object, scalarUnionFieldName, required, defaultValue, [=, &allocator](auto& value){ 
        return ParseDmlScalarUnion(value, dataType, allocator); 
    });
}

// ----------------------------------------------------------------------------
// DML_SCALE_BIAS
// ----------------------------------------------------------------------------
static void ParseDmlScaleBias(const rapidjson::Value& value, DML_SCALE_BIAS& returnValue)
{
    if (!value.IsObject())
    {
        throw std::invalid_argument("Expected a non-null JSON object.");
    }
    returnValue.Scale = ParseFloat32Field(value, "Scale");
    returnValue.Bias = ParseFloat32Field(value, "Bias");
}

DML_SCALE_BIAS ParseDmlScaleBias(const rapidjson::Value& value)
{
    DML_SCALE_BIAS returnValue{};
    ParseDmlScaleBias(value, returnValue);
    return returnValue;
}

DML_SCALE_BIAS* ParseDmlScaleBias(const rapidjson::Value& value, BucketAllocator& allocator)
{
    auto returnValue = allocator.Allocate<DML_SCALE_BIAS>();
    ParseDmlScaleBias(value, *returnValue);
    return returnValue;
}

DML_SCALE_BIAS* ParseDmlScaleBiasField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required, DML_SCALE_BIAS* defaultValue)
{
    return ParseFieldHelper<DML_SCALE_BIAS*>(object, fieldName, required, defaultValue, [&allocator](auto& value){ 
        return ParseDmlScaleBias(value, allocator); 
    });
}

// ----------------------------------------------------------------------------
// DML_BUFFER_TENSOR_DESC
// ----------------------------------------------------------------------------

DML_BUFFER_TENSOR_DESC* ParseDmlBufferTensorDesc(const rapidjson::Value& value, BucketAllocator& allocator)
{
    if (!value.IsObject())
    {
        throw std::invalid_argument("Expected a non-null JSON object.");
    }

    auto sizes = ParseUInt32ArrayField(value, "Sizes", allocator);
    auto strides = ParseUInt32ArrayField(value, "Strides", allocator, false);

    auto desc = allocator.Allocate<DML_BUFFER_TENSOR_DESC>();
    desc->DimensionCount = ParseUInt32Field(value, "DimensionCount", false, static_cast<uint32_t>(sizes.size()));
    desc->DataType = ParseDmlTensorDataTypeField(value, "DataType");
    desc->Flags = ParseDmlTensorFlagsField(value, "Flags", false, DML_TENSOR_FLAG_NONE);
    desc->Sizes = sizes.data();
    desc->Strides = strides.empty() ? nullptr : strides.data();
    desc->TotalTensorSizeInBytes = ParseUInt64Field(value, "TotalTensorSizeInBytes", false, 0);
    if (!desc->TotalTensorSizeInBytes)
    {
        desc->TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
            desc->DataType,
            desc->DimensionCount,
            desc->Sizes,
            desc->Strides);
    }
    desc->GuaranteedBaseOffsetAlignment = ParseUInt32Field(value, "GuaranteedBaseOffsetAlignment", false, 0);

    return desc;
}

DML_BUFFER_TENSOR_DESC* ParseDmlBufferTensorDescField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required, DML_BUFFER_TENSOR_DESC* defaultValue)
{
    return ParseFieldHelper<DML_BUFFER_TENSOR_DESC*>(object, fieldName, required, defaultValue, [&allocator](auto& value){ 
        return ParseDmlBufferTensorDesc(value, allocator); 
    });
}

// ----------------------------------------------------------------------------
// DML_TENSOR_DESC
// ----------------------------------------------------------------------------

DML_TENSOR_DESC* ParseDmlTensorDesc(const rapidjson::Value& value, BucketAllocator& allocator)
{
    if (!value.IsObject())
    {
        throw std::invalid_argument("Expected a non-null JSON object.");
    }

    auto desc = allocator.Allocate<DML_TENSOR_DESC>();
    desc->Type = ParseDmlTensorTypeField(value, "Type", false, DML_TENSOR_TYPE_BUFFER);
    if (value.HasMember("Desc"))
    {
        desc->Desc = ParseDmlBufferTensorDesc(value["Desc"], allocator);
    }
    else
    {
        desc->Desc = ParseDmlBufferTensorDesc(value, allocator);
    }
    return desc;
}

DML_TENSOR_DESC* ParseDmlTensorDescField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required, DML_TENSOR_DESC* defaultValue)
{
    return ParseFieldHelper<DML_TENSOR_DESC*>(object, fieldName, required, defaultValue, [&allocator](auto& value){ 
        return ParseDmlTensorDesc(value, allocator); 
    });
}

gsl::span<DML_TENSOR_DESC> ParseDmlTensorDescArray(const rapidjson::Value& value, BucketAllocator& allocator)
{
    if (value.GetType() != rapidjson::Type::kArrayType)
    {
        throw std::invalid_argument("Expected an array.");
    }

    auto valueArray = value.GetArray();
    auto outputElements = allocator.Allocate<DML_TENSOR_DESC>(valueArray.Size());
    for (uint32_t i = 0; i < valueArray.Size(); i++)
    {
        outputElements[i] = *ParseDmlTensorDesc(valueArray[i], allocator);
    }

    return gsl::make_span(outputElements, valueArray.Size());
}

gsl::span<DML_TENSOR_DESC> ParseDmlTensorDescArrayField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required, gsl::span<DML_TENSOR_DESC> defaultValue)
{
    return ParseFieldHelper<gsl::span<DML_TENSOR_DESC>>(object, fieldName, required, defaultValue, [&allocator](auto& value){ 
        return ParseDmlTensorDescArray(value, allocator); 
    });
}

// ----------------------------------------------------------------------------
// DML_OPERATOR_DESC
// ----------------------------------------------------------------------------

DML_OPERATOR_DESC* ParseDmlOperatorDescField(const rapidjson::Value& object, std::string_view fieldName, bool fused, BucketAllocator& allocator, bool required, DML_OPERATOR_DESC* defaultValue)
{
    return ParseFieldHelper<DML_OPERATOR_DESC*>(object, fieldName, required, defaultValue, [=,&allocator](auto& value){ 
        return ParseDmlOperatorDesc(value, fused, allocator); 
    });
}

gsl::span<DML_OPERATOR_DESC> ParseDmlOperatorDescArray(const rapidjson::Value& value, bool fused, BucketAllocator& allocator)
{
    if (value.GetType() != rapidjson::Type::kArrayType)
    {
        throw std::invalid_argument("Expected an array.");
    }

    auto valueArray = value.GetArray();
    auto outputElements = allocator.Allocate<DML_OPERATOR_DESC>(valueArray.Size());
    for (uint32_t i = 0; i < valueArray.Size(); i++)
    {
        outputElements[i] = *ParseDmlOperatorDesc(valueArray[i], fused, allocator);
    }

    return gsl::make_span(outputElements, valueArray.Size());
}

gsl::span<DML_OPERATOR_DESC> ParseDmlOperatorDescArrayField(const rapidjson::Value& object, std::string_view fieldName, bool fused, BucketAllocator& allocator, bool required, gsl::span<DML_OPERATOR_DESC> defaultValue)
{
    return ParseFieldHelper<gsl::span<DML_OPERATOR_DESC>>(object, fieldName, required, defaultValue, [=,&allocator](auto& value){ 
        return ParseDmlOperatorDescArray(value, fused, allocator); 
    });
}

// ----------------------------------------------------------------------------
// OTHER
// ----------------------------------------------------------------------------

uint64_t GetTensorSize(const DML_TENSOR_DESC& desc)
{
    if (desc.Type == DML_TENSOR_TYPE_BUFFER)
    {
        return static_cast<const DML_BUFFER_TENSOR_DESC*>(desc.Desc)->TotalTensorSizeInBytes;
    }
    throw std::invalid_argument("Cannot determine size of invalid tensor desc.");
}

#include "JsonParsersGenerated.cpp"

std::vector<std::byte> GenerateInitialValuesFromList(DML_TENSOR_DATA_TYPE dataType, const rapidjson::Value& object)
{
    switch (dataType)
    {
    case DML_TENSOR_DATA_TYPE_FLOAT16: return ParseArrayAsBytes<half_float::half>(object, ParseFloat16);
    case DML_TENSOR_DATA_TYPE_FLOAT32: return ParseArrayAsBytes<float>(object, ParseFloat32);
    case DML_TENSOR_DATA_TYPE_FLOAT64: return ParseArrayAsBytes<double>(object, ParseFloat64);
    case DML_TENSOR_DATA_TYPE_UINT8: return ParseArrayAsBytes<uint8_t>(object, ParseUInt8);
    case DML_TENSOR_DATA_TYPE_UINT16: return ParseArrayAsBytes<uint16_t>(object, ParseUInt16);
    case DML_TENSOR_DATA_TYPE_UINT32: return ParseArrayAsBytes<uint32_t>(object, ParseUInt32);
    case DML_TENSOR_DATA_TYPE_UINT64: return ParseArrayAsBytes<uint64_t>(object, ParseUInt64);
    case DML_TENSOR_DATA_TYPE_INT8: return ParseArrayAsBytes<int8_t>(object, ParseInt8);
    case DML_TENSOR_DATA_TYPE_INT16: return ParseArrayAsBytes<int16_t>(object, ParseInt16);
    case DML_TENSOR_DATA_TYPE_INT32: return ParseArrayAsBytes<int32_t>(object, ParseInt32);
    case DML_TENSOR_DATA_TYPE_INT64: return ParseArrayAsBytes<int64_t>(object, ParseInt64);
    default: throw std::invalid_argument(fmt::format("Invalid tensor data type."));
    }
}

std::vector<std::byte> GenerateInitialValuesFromConstant(DML_TENSOR_DATA_TYPE dataType, const rapidjson::Value& object)
{
    auto valueCount = ParseUInt32Field(object, "valueCount");

    auto AsBytes = [=](auto value)->std::vector<std::byte>
    {
        std::vector<std::byte> valueBytes;
        for (auto& byte : gsl::as_bytes(gsl::make_span(&value, 1)))
        {
            valueBytes.push_back(byte);
        }

        std::vector<std::byte> allBytes(valueBytes.size() * valueCount);
        for (size_t i = 0; i < valueCount; i++)
        {
            std::copy(valueBytes.begin(), valueBytes.end(), allBytes.begin() + i * valueBytes.size());
        }
        return allBytes;
    };

    switch (dataType)
    {
    case DML_TENSOR_DATA_TYPE_FLOAT16: return AsBytes(ParseFloat16Field(object, "value"));
    case DML_TENSOR_DATA_TYPE_FLOAT32: return AsBytes(ParseFloat32Field(object, "value"));
    case DML_TENSOR_DATA_TYPE_FLOAT64: return AsBytes(ParseFloat64Field(object, "value"));
    case DML_TENSOR_DATA_TYPE_UINT8: return AsBytes(ParseUInt8Field(object, "value"));
    case DML_TENSOR_DATA_TYPE_UINT16: return AsBytes(ParseUInt16Field(object, "value"));
    case DML_TENSOR_DATA_TYPE_UINT32: return AsBytes(ParseUInt32Field(object, "value"));
    case DML_TENSOR_DATA_TYPE_UINT64: return AsBytes(ParseUInt64Field(object, "value"));
    case DML_TENSOR_DATA_TYPE_INT8: return AsBytes(ParseInt8Field(object, "value"));
    case DML_TENSOR_DATA_TYPE_INT16: return AsBytes(ParseInt16Field(object, "value"));
    case DML_TENSOR_DATA_TYPE_INT32: return AsBytes(ParseInt32Field(object, "value"));
    case DML_TENSOR_DATA_TYPE_INT64: return AsBytes(ParseInt64Field(object, "value"));
    default: throw std::invalid_argument(fmt::format("Invalid tensor data type."));
    }
}

std::vector<std::byte> GenerateInitialValuesFromSequence(DML_TENSOR_DATA_TYPE dataType, const rapidjson::Value& object)
{
    auto valueCount = ParseUInt32Field(object, "valueCount");

    auto AsBytes = [=,&object](auto& parser, auto defaultValue)->std::vector<std::byte>
    {
        auto value = parser(object, "valueStart", true, defaultValue);
        auto valueDelta = parser(object, "valueDelta", true, defaultValue);

        std::vector<std::byte> allBytes;
        allBytes.reserve(sizeof(value) * valueCount);
        for (size_t i = 0; i < valueCount; i++)
        {
            for (auto byte : gsl::as_bytes(gsl::make_span(&value, 1)))
            {
                allBytes.push_back(byte);
            }
            value += valueDelta;
        }
        return allBytes;
    };

    switch (dataType)
    {
    case DML_TENSOR_DATA_TYPE_FLOAT16: return AsBytes(ParseFloat16Field, half_float::half(0));
    case DML_TENSOR_DATA_TYPE_FLOAT32: return AsBytes(ParseFloat32Field, 0.0f);
    case DML_TENSOR_DATA_TYPE_FLOAT64: return AsBytes(ParseFloat64Field, 0.0);
    case DML_TENSOR_DATA_TYPE_UINT8: return AsBytes(ParseUInt8Field, static_cast<uint8_t>(0));
    case DML_TENSOR_DATA_TYPE_UINT16: return AsBytes(ParseUInt16Field, static_cast<uint16_t>(0));
    case DML_TENSOR_DATA_TYPE_UINT32: return AsBytes(ParseUInt32Field, static_cast<uint32_t>(0));
    case DML_TENSOR_DATA_TYPE_UINT64: return AsBytes(ParseUInt64Field, static_cast<uint64_t>(0));
    case DML_TENSOR_DATA_TYPE_INT8: return AsBytes(ParseInt8Field, static_cast<int8_t>(0));
    case DML_TENSOR_DATA_TYPE_INT16: return AsBytes(ParseInt16Field, static_cast<int16_t>(0));
    case DML_TENSOR_DATA_TYPE_INT32: return AsBytes(ParseInt32Field, static_cast<int32_t>(0));
    case DML_TENSOR_DATA_TYPE_INT64: return AsBytes(ParseInt64Field, static_cast<int64_t>(0));
    default: throw std::invalid_argument(fmt::format("Invalid tensor data type."));
    }
}

std::filesystem::path ResolveInputFilePath(const std::filesystem::path& parentPath, std::string_view sourcePath)
{
    auto filePathRelativeToParent = std::filesystem::absolute(parentPath / sourcePath);
    if (std::filesystem::exists(filePathRelativeToParent))
    {
        return filePathRelativeToParent;
    }

    auto filePathRelativeToCurrentDirectory = std::filesystem::absolute(sourcePath);
    return filePathRelativeToCurrentDirectory;
}

std::filesystem::path ResolveOutputFilePath(const std::filesystem::path& parentPath, std::string_view targetPath)
{
    auto filePathRelativeToParent = std::filesystem::absolute(parentPath / targetPath);
    return filePathRelativeToParent;
}

std::vector<std::byte> ReadFileContent(const std::string& fileName)
{
    std::ifstream file(fileName.c_str(), std::ifstream::ate | std::ifstream::binary);
    if (!file.is_open())
    {
        throw std::ios::failure(fmt::format("Given filename '{}' could not be opened.", fileName));
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<std::byte> allBytes(fileSize);
    file.read(reinterpret_cast<char*>(allBytes.data()), fileSize);

    return allBytes;
}

std::tuple<std::vector<std::byte>, DML_TENSOR_DATA_TYPE, std::filesystem::path> GenerateInitialValuesFromFile(
    const std::filesystem::path& parentPath,
    const rapidjson::Value& object)
{
    auto sourcePath = ParseStringField(object, "sourcePath");
    auto filePath = ResolveInputFilePath(parentPath, sourcePath);

    std::vector<std::byte> allBytes = ReadFileContent(filePath.string());

    DML_TENSOR_DATA_TYPE tensorDataType = DML_TENSOR_DATA_TYPE_UNKNOWN;

    // Check for NumPy array files. Otherwise read it as raw file data, such as a .dat/.bin file.
    if (IsNpyFilenameExtension(sourcePath))
    {
        std::vector<uint32_t> dimensions;
        std::vector<std::byte> arrayByteData;
        ReadNpy(allBytes, /*out*/ tensorDataType, /*out*/ dimensions, /*out*/ arrayByteData);
        allBytes = std::move(arrayByteData);
    }

    return {std::move(allBytes), tensorDataType, filePath};
}

Model::BufferDesc ParseModelBufferDesc(const std::filesystem::path& parentPath, const rapidjson::Value& object)
{
    if (!object.IsObject())
    {
        throw std::invalid_argument("Expected a non-null JSON object.");
    }

    Model::BufferDesc buffer = {};
    buffer.initialValuesDataType = ParseDmlTensorDataTypeField(object, "initialValuesDataType", /*required*/ false);

    auto ensureInitialValuesDataType = [&]()
    {
        if (buffer.initialValuesDataType == DML_TENSOR_DATA_TYPE_UNKNOWN)
        {
            throw std::invalid_argument("Field 'initialValuesDataType' is required."); 
        }
    };

    auto initialValuesField = object.FindMember("initialValues");
    if (initialValuesField == object.MemberEnd())
    {
        throw std::invalid_argument("Field 'initialValues' is required."); 
    }

    if (initialValuesField->value.IsString())
    {
        if (initialValuesField->value != "deferred")
        {
            throw std::invalid_argument("The 'initialValuesDataType' only supports deferred");
        }
        buffer.initialValues.clear();
        buffer.initialValuesOffsetInBytes = 0;
        buffer.sizeInBytes = 0;
        buffer.useDeferredBinding = true;
        return buffer;
    }
    else if (initialValuesField->value.IsArray())
    {
        // e.g. "initialValues": [{"type": "UINT32", "value": 42}, {"type": "FLOAT32", "value": 3.14159}]
        if (buffer.initialValuesDataType == DML_TENSOR_DATA_TYPE_UNKNOWN)
        {
            buffer.initialValues = ParseMixedPrimitiveArray(initialValuesField->value);
        }
        // e.g. "initialValues": [1,2,3]
        else
        {
            buffer.initialValues = GenerateInitialValuesFromList(buffer.initialValuesDataType, initialValuesField->value);
        }
    } 
    else if (initialValuesField->value.IsObject())
    {
        // e.g. "initialValues": { "value": 0, "valueCount": 3 }
        if (initialValuesField->value.HasMember("value"))
        {
            if (initialValuesField->value.HasMember("valueStart") || initialValuesField->value.HasMember("sourcePath"))
            {
                throw std::invalid_argument("The 'initialValuesDataType' may contain a value, valueStart, or sourcePath, but they are mutually exclusive.");
            }

            ensureInitialValuesDataType();
            buffer.initialValues = GenerateInitialValuesFromConstant(buffer.initialValuesDataType, initialValuesField->value);
        }
        // e.g. "initialValues": { "valueStart": 0, "valueDelta": 2, "valueCount": 10 }
        else if (initialValuesField->value.HasMember("valueStart"))
        {
            ensureInitialValuesDataType();
            buffer.initialValues = GenerateInitialValuesFromSequence(buffer.initialValuesDataType, initialValuesField->value);
        }
        // e.g. "initialValues": { "sourcePath": "inputFile.npy" }
        else if (initialValuesField->value.HasMember("sourcePath"))
        {
            auto [initialValues, fileBufferDataType, fileName] = GenerateInitialValuesFromFile(parentPath, initialValuesField->value);

            // Depending on the file type (.npy vs .dat), the file may have an explict data type.
            // Use the data type if present, else require initialValuesDataType if not.
            if (buffer.initialValuesDataType == DML_TENSOR_DATA_TYPE_UNKNOWN)
            {
                buffer.initialValuesDataType = fileBufferDataType;
            }

            if ((fileBufferDataType != DML_TENSOR_DATA_TYPE_UNKNOWN) && (fileBufferDataType != buffer.initialValuesDataType))
            {
                throw std::invalid_argument(fmt::format("Data type from file '{}' does not match field 'initialValuesDataType'.", fileName.string()));
            }

            ensureInitialValuesDataType(); // Raw data requires 'initialValuesDataType'. Typed data (e.g. .npy) already had a type.
            buffer.initialValues = std::move(initialValues);
        }
        else
        {
            throw std::invalid_argument("Error parsing 'initialValues' object: unknown generator type."); 
        }
    }
    else
    {
        throw std::invalid_argument("Field 'initialValues' must be an array of numbers, an object, or deferred.");
    }

    if (buffer.initialValues.empty())
    {
        throw std::invalid_argument("'initialValues' must be non-empty.");
    }

    buffer.sizeInBytes = ParseUInt64Field(object, "sizeInBytes", false, buffer.initialValues.size());
    if (!object.HasMember("sizeInBytes"))
    {
        // Unless the size was explicitly set, round up to the nearest 4 bytes.
        buffer.sizeInBytes = (buffer.sizeInBytes + 3) & ~3ull;
    }

    buffer.initialValuesOffsetInBytes = ParseUInt64Field(object, "initialValuesOffsetInBytes", false, 0);

    if (buffer.initialValues.size() + buffer.initialValuesOffsetInBytes > buffer.sizeInBytes)
    {
        throw std::invalid_argument(fmt::format(
            "The buffer size ({} bytes) is too small for the initialValues ({} bytes) at offset {} bytes.", 
            buffer.sizeInBytes, 
            buffer.initialValues.size(),
            buffer.initialValuesOffsetInBytes));
    }

    return buffer;
}

Model::ResourceDesc ParseModelResourceDesc(
    std::string_view name,
    const std::filesystem::path& parentPath,
    const rapidjson::Value& object)
{
    Model::ResourceDesc desc;
    desc.name = name;
    desc.value = ParseModelBufferDesc(parentPath, object);
    return desc;
}

Model::HlslDispatchableDesc ParseModelHlslDispatchableDesc(const std::filesystem::path& parentPath, const rapidjson::Value& object)
{
    Model::HlslDispatchableDesc desc = {};
    auto sourcePath = ParseStringField(object, "sourcePath");
    desc.sourcePath = ResolveInputFilePath(parentPath, sourcePath);

    auto compilerStr = ParseStringField(object, "compiler", false, "dxc");
    if (!_stricmp(compilerStr.data(), "dxc"))
    {
        desc.compiler = Model::HlslDispatchableDesc::Compiler::DXC;
    }
    else
    {
        throw std::invalid_argument("Unrecognized compiler");
    }

    auto compilerArgsField = object.FindMember("compilerArgs");
    if (compilerArgsField == object.MemberEnd() || !compilerArgsField->value.IsArray())
    {
        throw std::invalid_argument("Field 'compilerArgs' is required and must be an array.");
    }

    desc.compilerArgs.reserve(compilerArgsField->value.GetArray().Size());
    for (auto& compilerArg : compilerArgsField->value.GetArray())
    {
        desc.compilerArgs.push_back(compilerArg.GetString());
    }

    return desc;
}

Model::OnnxDispatchableDesc ParseModelOnnxDispatchableDesc(const std::filesystem::path& parentPath, const rapidjson::Value& object)
{
    Model::OnnxDispatchableDesc desc = {};
    auto sourcePath = ParseStringField(object, "sourcePath");
    desc.sourcePath = ResolveInputFilePath(parentPath, sourcePath);

    desc.freeDimNameOverrides = ParseFieldHelper<std::vector<std::pair<std::string, uint32_t>>>(
        object, "freeDimensionNameOverrides", false, {}, [](auto& value)
    { 
        std::vector<std::pair<std::string, uint32_t>> overrides;

        if (!value.IsObject())
        {
            throw std::invalid_argument("Expected a non-null JSON object.");
        }

        for (auto member = value.MemberBegin(); member != value.MemberEnd(); member++)
        {
            overrides.emplace_back(member->name.GetString(), ParseUInt32(member->value));
        }

        return overrides;
    });

    desc.freeDimDenotationOverrides = ParseFieldHelper<std::vector<std::pair<std::string, uint32_t>>>(
        object, "freeDimensionDenotationOverrides", false, {}, [](auto& value)
    { 
        std::vector<std::pair<std::string, uint32_t>> overrides;
        
        if (!value.IsObject())
        {
            throw std::invalid_argument("Expected a non-null JSON object.");
        }

        for (auto member = value.MemberBegin(); member != value.MemberEnd(); member++)
        {
            overrides.emplace_back(member->name.GetString(), ParseUInt32(member->value));
        }

        return overrides;
    });

    desc.sessionOptionsConfigEntries = ParseFieldHelper<std::vector<std::pair<std::string, std::string>>>(
        object, "sessionOptionsConfigEntries", false, {}, [](auto& value)
    { 
        std::vector<std::pair<std::string, std::string>> overrides;
        
        if (!value.IsObject())
        {
            throw std::invalid_argument("Expected a non-null JSON object.");
        }

        for (auto member = value.MemberBegin(); member != value.MemberEnd(); member++)
        {
            overrides.emplace_back(member->name.GetString(), ParseString(member->value));
        }

        return overrides;
    });

    desc.graphOptimizationLevel = ParseUInt32Field(object, "graphOptimizationLevel", false, 99);
    desc.loggingLevel = ParseUInt32Field(object, "loggingLevel", false, 2);

    return desc;
}

Model::BufferBindingSource ParseBufferBindingSource(const rapidjson::Value& value)
{
    Model::BufferBindingSource bindingSource = {};

    if (value.IsString())
    {
        bindingSource.name = value.GetString();
    }
    else if (value.IsObject())
    {
        bindingSource.name = ParseStringField(value, "name");
        bindingSource.elementCount = ParseUInt64Field(value, "elementCount", false, 0);
        bindingSource.elementSizeInBytes = ParseUInt64Field(value, "elementSizeInBytes", false, 0);
        bindingSource.elementOffset = ParseUInt64Field(value, "elementOffset", false, 0);
        if (value.HasMember("format"))
        {
            bindingSource.format = ParseDxgiFormat(value["format"]);
        }
        if (value.HasMember("counter"))
        {
            bindingSource.counterName = ParseStringField(value, "counter");
            bindingSource.counterOffsetBytes = ParseUInt64Field(value, "counterOffsetBytes", false);
        }
        bindingSource.shape = ParseInt64ArrayAsVectorField(value, "shape", false);
    }

    return bindingSource;
}

std::vector<Model::BufferBindingSource> ParseBindingSource(const rapidjson::Value& object)
{
    std::vector<Model::BufferBindingSource> sourceResources;
    if (object.IsArray())
    {
        for (auto& bindingValue : object.GetArray())
        {
            sourceResources.push_back(ParseBufferBindingSource(bindingValue));
        }
    }
    else
    {
        sourceResources.push_back(ParseBufferBindingSource(object));
    }
    return sourceResources;
}

Model::DmlDispatchableDesc ParseModelDmlDispatchableDesc(const rapidjson::Value& object, BucketAllocator& allocator)
{
    Model::DmlDispatchableDesc desc;
    desc.desc = ParseDmlOperatorDesc(object, false, allocator);
    desc.bindPoints = GetBindPoints(*desc.desc);
    desc.executionFlags = ParseDmlExecutionFlagsField(object, "executionFlags", false, DML_EXECUTION_FLAG_NONE);

    auto bindingsField = object.FindMember("bindings");
    if (bindingsField != object.MemberEnd() && bindingsField->value.IsObject())
    {
        for (auto bindingMember = bindingsField->value.MemberBegin(); bindingMember != bindingsField->value.MemberEnd(); bindingMember++)
        {
            desc.initBindings[bindingMember->name.GetString()] = ParseBindingSource(bindingMember->value);
        }
    }

    return desc;
}

Model::DispatchableDesc ParseModelDispatchableDesc(
    std::string_view name,
    const std::filesystem::path& parentPath,
    const rapidjson::Value& object,
    BucketAllocator& allocator)
{
    if (!object.IsObject())
    {
        throw std::invalid_argument("Expected a non-null JSON object.");
    }

    Model::DispatchableDesc desc;
    desc.name = name;
    auto type = ParseStringField(object, "type");
    if (!_stricmp(type.data(), "hlsl")) 
    { 
        desc.value = ParseModelHlslDispatchableDesc(parentPath, object);
    }
    else if (!_stricmp(type.data(), "onnx"))
    {
        desc.value = ParseModelOnnxDispatchableDesc(parentPath, object);
    }
    else
    {
        desc.value = ParseModelDmlDispatchableDesc(object, allocator);
    }

    return desc;
}

Model::DispatchCommand ParseDispatchCommand(const rapidjson::Value& object)
{
    Model::DispatchCommand command = {};
    
    command.dispatchableName = ParseStringField(object, "dispatchable");

    auto threadGroupCountField = object.FindMember("threadGroupCount");
    if (threadGroupCountField != object.MemberEnd())
    {
        if (!threadGroupCountField->value.IsArray())
        {
            throw std::invalid_argument("If 'threadGroupCount' is present it must be an array with 3 integers larger than 1");
        }
        auto threadGroupCountArray = threadGroupCountField->value.GetArray();
        if (threadGroupCountArray.Size() != 3)
        {
            throw std::invalid_argument("If 'threadGroupCount' is present it must be an array with 3 integers larger than 1");
        }
        uint32_t x = threadGroupCountArray[0].GetUint();
        uint32_t y = threadGroupCountArray[1].GetUint();
        uint32_t z = threadGroupCountArray[2].GetUint();
        command.threadGroupCount = {x, y, z};
    }
    else
    {
        command.threadGroupCount = {1, 1, 1};
    }

    auto bindingsField = object.FindMember("bindings");
    if (bindingsField == object.MemberEnd() || !bindingsField->value.IsObject())
    {
        throw std::invalid_argument("Expected an object field named 'bindings'.");
    }

    for (auto bindingMember = bindingsField->value.MemberBegin(); bindingMember != bindingsField->value.MemberEnd(); bindingMember++)
    {
        command.bindings[bindingMember->name.GetString()] = ParseBindingSource(bindingMember->value);
    }

    return command;
}

Model::PrintCommand ParsePrintCommand(const rapidjson::Value& object)
{
    Model::PrintCommand command = {};
    command.resourceName = ParseStringField(object, "resource");
    return command;
}

Model::WriteFileCommand ParseWriteFileCommand(const rapidjson::Value& object, const std::filesystem::path& outputPath)
{
    Model::WriteFileCommand command = {};
    command.resourceName = ParseStringField(object, "resource");
    command.targetPath = ResolveOutputFilePath(outputPath, ParseStringField(object, "targetPath")).string();
    BucketAllocator allocator;
    auto dimensions = ParseUInt32ArrayField(object, "dimensions", allocator, false);
    command.dimensions.assign(dimensions.begin(), dimensions.end());

    return command;
}
Model::Command ParseModelCommand(const rapidjson::Value& object, const std::filesystem::path& outputPath)
{
    return ParseModelCommandDesc(object, outputPath).command;
}

Model::CommandDesc ParseModelCommandDesc(const rapidjson::Value& object, const std::filesystem::path& outputPath)
{
    Model::CommandDesc commandDesc = {};

    commandDesc.type = ParseStringField(object, "type");
    commandDesc.parameters = RapidJsonToString(object);

    if (!_stricmp(commandDesc.type.data(), "dispatch"))
    { 
        commandDesc.command = ParseDispatchCommand(object);
    }
    else if (!_stricmp(commandDesc.type.data(), "print"))
    {
        commandDesc.command = ParsePrintCommand(object);
    }
    else if (!_stricmp(commandDesc.type.data(), "writeFile"))
    {
        commandDesc.command = ParseWriteFileCommand(object, outputPath);
    }
    else
    {
        throw std::invalid_argument("Unrecognized command");
    }

    return commandDesc;
}

// Determine the line and column in text by counting the line breaking characters
// up to the given offset. Note the line and column counts are zero-based, and so the
// caller may want to add 1 when displaying it, as most text editors show one-based
// values to the user.
void MapCharacterOffsetToLineColumn(
    std::string_view documentText,
    size_t errorOffset,
    _Out_ uint32_t& line,
    _Out_ uint32_t& column
    )
{
    uint32_t lineCount = 0;
    uint32_t columnCount = 0;

    bool precededByCr = false;
    for (size_t i = 0; i < errorOffset; ++i)
    {
        wchar_t ch = documentText[i];
        bool foundNewLine = false;

        switch (ch)
        {
        case 0x2028: // U+2028 LINE SEPARATOR
        case 0x2029: // U+2029 PARAGRAPH SEPARATOR
        case 0x000B: // U+000B VERTICAL TABULATION
        case 0x000C: // U+000C FORM FEED
        case 0x000D: // U+000D CARRIAGE RETURN
            foundNewLine = true;
            break;

        case 0x000A: // U+000A LINE FEED
            // Count CR LF pair as one line break.
            foundNewLine = !precededByCr;
            break;

        default: // Any other character.
            ++columnCount;
            break;
        }
        precededByCr = (ch == 0x000D);

        if (foundNewLine)
        {
            ++lineCount;
            columnCount = 0;
        }
    }

    line = lineCount;
    column = columnCount;
}

std::string GetJsonParseErrorMessage(
    const rapidjson::Document& jsonDocument,
    std::string_view jsonDocumentText
    )
{
    // Gather a snippet of preview text at the error, stripping any new lines for preview sake.
    // Note RapidJSON doesn't include the line number, just document offset.
    std::string_view applicableText = jsonDocumentText.substr(jsonDocument.GetErrorOffset(), 40);
    std::string newLineStrippedText(applicableText);

    for (auto& ch : newLineStrippedText)
    {
        if (ch == '\r' || ch == '\n')
            ch = ' ';
    }

    uint32_t line = 0, column = 0;
    MapCharacterOffsetToLineColumn(jsonDocumentText, jsonDocument.GetErrorOffset(), /*out*/ line, /*out*/ column);

    std::string formattedErrorMessage = fmt::format(
        "JSON parse error at char offset:{}, line:{}, column:{}, error:{} {}\nSnippet: >>>{}<<<",
        int(jsonDocument.GetErrorOffset()),
        line + 1,
        column + 1,
        int(jsonDocument.GetParseError()),
        rapidjson::GetParseError_En(jsonDocument.GetParseError()),
        newLineStrippedText.c_str()
    );

    return formattedErrorMessage;
}

Model ParseModel(
    const rapidjson::Document& doc,
    const std::string_view& jsonDocumentText,
    const std::filesystem::path& inputPath,
    const std::filesystem::path& outputPath)
{
    if (doc.HasParseError())
    {
        std::string errorMessage = GetJsonParseErrorMessage(doc, jsonDocumentText);
        throw std::invalid_argument(errorMessage);
    }

    BucketAllocator allocator;

    std::vector<Model::ResourceDesc> resources;
    auto resourcesField = doc.FindMember("resources");
    if (resourcesField == doc.MemberEnd() || !resourcesField->value.IsObject())
    {
        throw std::invalid_argument("Expected an object named 'resources'");
    }
    for (auto field = resourcesField->value.MemberBegin(); field != resourcesField->value.MemberEnd(); field++)
    {
        try
        {
            resources.emplace_back(std::move(ParseModelResourceDesc(field->name.GetString(), inputPath, field->value)));
        }
        catch (std::exception& e)
        {
            throw std::invalid_argument(fmt::format("Failed to parse resource {}: {}", field->name.GetString(), e.what()));
        }
    }

    std::vector<Model::DispatchableDesc> operators;
    auto dispatchablesField = doc.FindMember("dispatchables");
    if (dispatchablesField == doc.MemberEnd() || !dispatchablesField->value.IsObject())
    {
        throw std::invalid_argument("Expected an object named 'dispatchables'");
    }
    for (auto field = dispatchablesField->value.MemberBegin(); field != dispatchablesField->value.MemberEnd(); field++)
    {
        try
        {
            operators.emplace_back(std::move(ParseModelDispatchableDesc(field->name.GetString(), inputPath, field->value, allocator)));
        }
        catch (std::exception& e)
        {
            throw std::invalid_argument(fmt::format("Failed to parse dispatchable {}: {}", field->name.GetString(), e.what()));
        }
    }

    std::vector<Model::CommandDesc> commands;
    auto commandsField = doc.FindMember("commands");
    if (commandsField == doc.MemberEnd() || !commandsField->value.IsArray())
    {
        throw std::invalid_argument("Expected an array field named 'commands'");
    }
    auto commandsArray = commandsField->value.GetArray();
    for (uint32_t i = 0; i < commandsArray.Size(); i++)
    {
        try
        {
            commands.emplace_back(std::move(ParseModelCommandDesc(commandsArray[i], outputPath)));
        }
        catch (std::exception& e)
        {
            throw std::invalid_argument(fmt::format("Failed to parse command at index {}: {}", i, e.what()));
        }
    }

    return {std::move(resources), std::move(operators), std::move(commands), std::move(allocator)};
}

Model ParseModel(
    const std::filesystem::path& filePath,
    std::filesystem::path inputPath,
    std::filesystem::path outputPath)
{

    std::filesystem::path modelPath = filePath;
    if (!std::filesystem::exists(filePath))
    {
        modelPath = inputPath / filePath;
        if (!std::filesystem::exists(modelPath))
        {
            throw std::invalid_argument(fmt::format("Model does not exist. Path given: '{}'.", filePath.string()));
        }
    }
    if (std::filesystem::is_directory(modelPath))
    {
        throw std::invalid_argument(fmt::format("Model must be a JSON file, not a directory. Path given: '{}'", modelPath.string()));
    }

    std::vector<std::byte> allBytes = ReadFileContent(modelPath.string());
    allBytes.push_back(std::byte(0)); // Ensure null terminated for parser.
    char* fileContentBegin = reinterpret_cast<char*>(allBytes.data());
    std::string_view fileContent{fileContentBegin, allBytes.size()};

    rapidjson::Document doc;

    constexpr rapidjson::ParseFlag parseFlags = rapidjson::ParseFlag(
        rapidjson::kParseFullPrecisionFlag | 
        rapidjson::kParseCommentsFlag |
        rapidjson::kParseTrailingCommasFlag |
        rapidjson::kParseStopWhenDoneFlag);

    doc.ParseInsitu<parseFlags>(fileContentBegin);

    return ParseModel(doc, fileContent, inputPath, outputPath);
}

} // namespace JsonParsers