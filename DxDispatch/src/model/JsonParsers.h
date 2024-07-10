#pragma once

#include <DirectML.h>
#include <rapidjson/document.h>
#include <half.hpp>
#include "Model.h"

namespace JsonParsers
{
    // ------------------------------------------------------------------------
    // PRIMITIVE TYPES
    // ------------------------------------------------------------------------

    // BOOL
    bool ParseBool(const rapidjson::Value& object);
    bool ParseBoolField(const rapidjson::Value& object, std::string_view fieldName, bool required = true, bool defaultValue = false);

    // FLOAT16
    half_float::half ParseFloat16(const rapidjson::Value& object);
    half_float::half ParseFloat16Field(const rapidjson::Value& object, std::string_view fieldName, bool required = true, half_float::half defaultValue = half_float::half(0.0f));
    gsl::span<half_float::half> ParseFloat16Array(const rapidjson::Value& object, BucketAllocator& allocator);
    gsl::span<half_float::half> ParseFloat16ArrayField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required = true, gsl::span<half_float::half> defaultValue = {});

    // FLOAT32
    float ParseFloat32(const rapidjson::Value& object);
    float ParseFloat32Field(const rapidjson::Value& object, std::string_view fieldName, bool required = true, float defaultValue = 0.0f);
    gsl::span<float> ParseFloat32Array(const rapidjson::Value& object, BucketAllocator& allocator);
    gsl::span<float> ParseFloat32ArrayField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required = true, gsl::span<float> defaultValue = {});

    // FLOAT64
    double ParseFloat64(const rapidjson::Value& object);
    double ParseFloat64Field(const rapidjson::Value& object, std::string_view fieldName, bool required = true, double defaultValue = 0.0);

    // INT8
    int8_t ParseInt8(const rapidjson::Value& object);
    int8_t ParseInt8Field(const rapidjson::Value& object, std::string_view fieldName, bool required = true, int8_t defaultValue = 0);

    // INT16
    int16_t ParseInt16(const rapidjson::Value& object);
    int16_t ParseInt16Field(const rapidjson::Value& object, std::string_view fieldName, bool required = true, int16_t defaultValue = 0);

    // INT32
    int32_t ParseInt32(const rapidjson::Value& object);
    int32_t ParseInt32Field(const rapidjson::Value& object, std::string_view fieldName, bool required = true, int32_t defaultValue = 0);
    gsl::span<int32_t> ParseInt32Array(const rapidjson::Value& object, BucketAllocator& allocator);
    gsl::span<int32_t> ParseInt32ArrayField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required = true, gsl::span<int32_t> defaultValue = {});

    // INT64
    int64_t ParseInt64(const rapidjson::Value& object);
    int64_t ParseInt64Field(const rapidjson::Value& object, std::string_view fieldName, bool required = true, int64_t defaultValue = 0);
    std::vector<int64_t> ParseInt64ArrayAsVector(const rapidjson::Value& object);
    std::vector<int64_t> ParseInt64ArrayAsVectorField(const rapidjson::Value& object, std::string_view fieldName, bool required = true, std::vector<int64_t> defaultValue = {});

    // UINT8
    uint8_t ParseUInt8(const rapidjson::Value& object);
    uint8_t ParseUInt8Field(const rapidjson::Value& object, std::string_view fieldName, bool required = true, uint8_t defaultValue = 0);

    // UINT16
    uint16_t ParseUInt16(const rapidjson::Value& object);
    uint16_t ParseUInt16Field(const rapidjson::Value& object, std::string_view fieldName, bool required = true, uint16_t defaultValue = 0);

    // UINT32
    uint32_t ParseUInt32(const rapidjson::Value& object);
    uint32_t ParseUInt32Field(const rapidjson::Value& object, std::string_view fieldName, bool required = true, uint32_t defaultValue = 0);
    gsl::span<uint32_t> ParseUInt32Array(const rapidjson::Value& object, BucketAllocator& allocator);
    gsl::span<uint32_t> ParseUInt32ArrayField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required = true, gsl::span<uint32_t> defaultValue = {});

    // UINT64
    uint64_t ParseUInt64(const rapidjson::Value& object);
    uint64_t ParseUInt64Field(const rapidjson::Value& object, std::string_view fieldName, bool required = true, uint64_t defaultValue = 0);

    // Mixed primitives array
    std::vector<std::byte> ParseMixedPrimitiveArray(const rapidjson::Value& object);

    // ------------------------------------------------------------------------
    // DIRECTML ENUMS
    // ------------------------------------------------------------------------

    // DML_TENSOR_DATA_TYPE
    DML_TENSOR_DATA_TYPE ParseDmlTensorDataType(const rapidjson::Value& value);
    DML_TENSOR_DATA_TYPE ParseDmlTensorDataTypeField(const rapidjson::Value& object, std::string_view fieldName, bool required = true, DML_TENSOR_DATA_TYPE defaultValue = DML_TENSOR_DATA_TYPE_UNKNOWN);

    // DML_TENSOR_FLAGS
    DML_TENSOR_FLAGS ParseDmlTensorFlags(const rapidjson::Value& value);
    DML_TENSOR_FLAGS ParseDmlTensorFlagsField(const rapidjson::Value& object, std::string_view fieldName, bool required = true, DML_TENSOR_FLAGS defaultValue = DML_TENSOR_FLAG_NONE);

    // DML_EXECUTION_FLAGS
    DML_EXECUTION_FLAGS ParseDmlExecutionFlags(const rapidjson::Value& value);
    DML_EXECUTION_FLAGS ParseDmlExecutionFlagsField(const rapidjson::Value& object, std::string_view fieldName, bool required = true, DML_EXECUTION_FLAGS defaultValue = DML_EXECUTION_FLAG_NONE);

    // DmlCompileType
    DmlCompileType ParseDmlCompileType(const rapidjson::Value& value);
    DmlCompileType ParseDmlCompileTypeField(const rapidjson::Value& object, std::string_view fieldName, bool required = true, DmlCompileType defaultValue = DmlCompileType::DmlCompileOp);

    // DML_TENSOR_TYPE
    DML_TENSOR_TYPE ParseDmlTensorType(const rapidjson::Value& value);
    DML_TENSOR_TYPE ParseDmlTensorTypeField(const rapidjson::Value& object, std::string_view fieldName, bool required = true, DML_TENSOR_TYPE defaultValue = DML_TENSOR_TYPE_INVALID);

    // ------------------------------------------------------------------------
    // DIRECTML STRUCTS
    // ------------------------------------------------------------------------
    
    // DML_SIZE_2D
    DML_SIZE_2D ParseDmlSize2d(const rapidjson::Value& value);
    DML_SIZE_2D* ParseDmlSize2d(const rapidjson::Value& value, BucketAllocator& allocator);
    DML_SIZE_2D* ParseDmlSize2dField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required = true, DML_SIZE_2D* defaultValue = nullptr);

    // DML_SCALAR_UNION
    DML_SCALAR_UNION ParseDmlScalarUnion(const rapidjson::Value& value, DML_TENSOR_DATA_TYPE dataType);
    DML_SCALAR_UNION* ParseDmlScalarUnion(const rapidjson::Value& value, DML_TENSOR_DATA_TYPE dataType, BucketAllocator& allocator);
    DML_SCALAR_UNION* ParseDmlScalarUnionField(const rapidjson::Value& object, std::string_view scalarUnionFieldName, std::string_view dataTypeFieldName, BucketAllocator& allocator, bool required = true, DML_SCALAR_UNION* defaultValue = nullptr);

    // DML_SCALE_BIAS
    DML_SCALE_BIAS ParseDmlScaleBias(const rapidjson::Value& value);
    DML_SCALE_BIAS* ParseDmlScaleBias(const rapidjson::Value& value, BucketAllocator& allocator);
    DML_SCALE_BIAS* ParseDmlScaleBiasField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required = true, DML_SCALE_BIAS* defaultValue = nullptr);

    // DML_BUFFER_TENSOR_DESC
    DML_BUFFER_TENSOR_DESC* ParseDmlBufferTensorDesc(const rapidjson::Value& value, BucketAllocator& allocator);
    DML_BUFFER_TENSOR_DESC* ParseDmlBufferTensorDescField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required = true, DML_BUFFER_TENSOR_DESC* defaultValue = nullptr);

    // DML_TENSOR_DESC
    DML_TENSOR_DESC* ParseDmlTensorDesc(const rapidjson::Value& value, BucketAllocator& allocator);
    DML_TENSOR_DESC* ParseDmlTensorDescField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required = true, DML_TENSOR_DESC* defaultValue = nullptr);
    gsl::span<DML_TENSOR_DESC> ParseDmlTensorDescArray(const rapidjson::Value& value, BucketAllocator& allocator);
    gsl::span<DML_TENSOR_DESC> ParseDmlTensorDescArrayField(const rapidjson::Value& object, std::string_view fieldName, BucketAllocator& allocator, bool required = true, gsl::span<DML_TENSOR_DESC> defaultValue = {});

    // DML_OPERATOR_DESC
    DML_OPERATOR_DESC* ParseDmlOperatorDesc(const rapidjson::Value& value, bool fused, BucketAllocator& allocator);
    DML_OPERATOR_DESC* ParseDmlOperatorDescField(const rapidjson::Value& object, std::string_view fieldName, bool fused, BucketAllocator& allocator, bool required = true, DML_OPERATOR_DESC* defaultValue = nullptr);
    gsl::span<DML_OPERATOR_DESC> ParseDmlOperatorDescArray(const rapidjson::Value& value, bool fused, BucketAllocator& allocator);
    gsl::span<DML_OPERATOR_DESC> ParseDmlOperatorDescArrayField(const rapidjson::Value& object, std::string_view fieldName, bool fused, BucketAllocator& allocator, bool required = true, gsl::span<DML_OPERATOR_DESC> defaultValue = {});

    // ------------------------------------------------------------------------
    // MODEL STRUCTS
    // ------------------------------------------------------------------------

    Model::ResourceDesc ParseModelResourceDesc(std::string_view name, const std::filesystem::path& parentPath, const rapidjson::Value& object);
    Model::DispatchableDesc ParseModelDispatchableDesc(std::string_view name, const std::filesystem::path& parentPath, const rapidjson::Value& object, BucketAllocator& allocator);
    Model::Command ParseModelCommand(const rapidjson::Value& object, const std::filesystem::path& outputPath);
    Model::CommandDesc ParseModelCommandDesc(const rapidjson::Value& object, const std::filesystem::path& outputPath);

    Model ParseModel(
        const rapidjson::Document& doc,
        const std::string_view &jsonDocumentText,
        const std::filesystem::path& inputPath,
        const std::filesystem::path& outputPath);

    Model ParseModel(
        const std::filesystem::path& filePath, 
        std::filesystem::path inputPath,
        std::filesystem::path outputPath);
}