param
(
    [string]$SchemaFilePath = "$PSScriptRoot\DmlSchema.json",
    [string]$MaxFeatureLevel = "5.1"
)

function ConvertSnakeToCamelCase($SnakeCaseName)
{
    return [regex]::replace(
        $SnakeCaseName.ToLower(), 
        '^(.)|_(.)', 
        {$args[0].Groups[1].Value.ToUpper() + $args[0].Groups[2].Value.ToUpper()})
}

function WriteEnumParser($Enum)
{
    $EnumNameCamelCase = ConvertSnakeToCamelCase($Enum.Name)

    $Cpp = @()
    $Cpp += "$($Enum.Name) Parse${EnumNameCamelCase}(const rapidjson::Value& value)"
    $Cpp += "{"
    $Cpp += "    if (value.GetType() != rapidjson::Type::kStringType)"
    $Cpp += "    {"
    $Cpp += "        throw std::invalid_argument(`"$($Enum.Name) must be a string.`");"
    $Cpp += "    }"
    $Cpp += "    auto valueString = value.GetString();"
    foreach ($Value in $Enum.Values)
    {
        if ($Value.StartsWith("$($Enum.Name)_"))
        {
            # Most enum values follow a "DML_<ENUM_NAME>_<VALUE>" convention. E.g. DML_TENSOR_DATA_TYPE_FLOAT32 is a value in
            # the DML_TENSOR_DATA_TYPE enum. In these cases allow the JSON to only provide the ending (e.g. "FLOAT32").
            $ShortName = $Value -replace "$($Enum.Name)_"
            $Cpp += "    if (!strcmp(valueString, `"$Value`") || !strcmp(valueString, `"$ShortName`")) { return $Value; }"
        }
        elseif ($Enum.Name -eq "DML_OPERATOR_TYPE")
        {
            # DML_OPERATOR_TYPE is unique (at the moment) in that its values don't follow the above convention and omit
            # the '_TYPE' portion of the enum name. E.g. DML_OPERATOR_ELEMENT_WISE_ABS instead of DML_OPERATOR_TYPE_ELEMENT_WISE_ABS.
            $ShortName = $Value -replace "DML_OPERATOR_"
            $Cpp += "    if (!strcmp(valueString, `"$Value`") || !strcmp(valueString, `"$ShortName`")) { return $Value; }"
        }
        else
        {
            $Cpp += "    if (!strcmp(valueString, `"$Value`")) { return $Value; }"
        }
    }
    $Cpp += "    throw std::invalid_argument(fmt::format(`"'{}' is not a recognized value for $($Enum.Name).`", valueString));"
    $Cpp += "}"
    $Cpp += ""
    $Cpp += "$($Enum.Name) Parse${EnumNameCamelCase}Field(const rapidjson::Value& object, std::string_view fieldName, bool required, $($Enum.Name) defaultValue)"
    $Cpp += "{"
    $Cpp += "    return ParseFieldHelper<$($Enum.Name)>(object, fieldName, required, defaultValue, [](auto& value){"
    $Cpp += "        return Parse${EnumNameCamelCase}(value); "
    $Cpp += "    });"
    $Cpp += "}"
    $Cpp += ""
    return $Cpp
}

function WriteFlagsParser($Flags)
{
    $FlagsNameCamelCase = ConvertSnakeToCamelCase($Flags.Name)
    $FlagsNameSingular = $Flags.Name -replace "FLAGS", "FLAG"

    $Cpp = @()
    $Cpp += "$($Flags.Name) ParseSingleFlagFrom${FlagsNameCamelCase}(const rapidjson::Value& value)"
    $Cpp += "{"
    $Cpp += "    if (value.GetType() != rapidjson::Type::kStringType)"
    $Cpp += "    {"
    $Cpp += "        throw std::invalid_argument(`"Expected a string.`");"
    $Cpp += "    }"
    $Cpp += "    auto valueString = value.GetString();"
    $Cpp += "    if (!strcmp(valueString, `"${FlagsNameSingular}_NONE`") || !strcmp(valueString, `"NONE`")) { return ${FlagsNameSingular}_NONE; }"
    foreach ($Value in $Flags.Values)
    {
        $ShortName = $Value -replace "${FlagsNameSingular}_"
        $Cpp += "    if (!strcmp(valueString, `"$Value`") || !strcmp(valueString, `"$ShortName`")) { return $Value; }"
    }
    $Cpp += "    throw std::invalid_argument(fmt::format(`"'{}' is not a recognized value for $($Flags.Name).`", valueString));"    
    $Cpp += "}"
    $Cpp += ""
    $Cpp += "$($Flags.Name) Parse${FlagsNameCamelCase}(const rapidjson::Value& value)"
    $Cpp += "{"
    $Cpp += "    return ParseFlags<$($Flags.Name)>(value, ParseSingleFlagFrom${FlagsNameCamelCase});"
    $Cpp += "}"
    $Cpp += ""
    $Cpp += "$($Flags.Name) Parse${FlagsNameCamelCase}Field(const rapidjson::Value& object, std::string_view fieldName, bool required, $($Flags.Name) defaultValue)"
    $Cpp += "{"
    $Cpp += "    return ParseFieldHelper<$($Flags.Name)>(object, fieldName, required, defaultValue, [](auto& value){"
    $Cpp += "        return Parse${FlagsNameCamelCase}(value); "
    $Cpp += "    });"
    $Cpp += "}"
    $Cpp += ""
    return $Cpp
}

function WriteOperatorFunction($Operator)
{
    $OpFunctionName = "ParseDml$(ConvertSnakeToCamelCase $Operator.Name)OperatorDesc"

    $Cpp = @()
    $Cpp += "DML_OPERATOR_DESC* $OpFunctionName(const rapidjson::Value& value, bool fused, BucketAllocator& allocator)"
    $Cpp += "{"
    $Cpp += "    if (!value.IsObject()) { throw std::invalid_argument(`"Expected a valid JSON object.`"); }"
    $Cpp += "    auto desc = allocator.Allocate<DML_$($Operator.Name)_OPERATOR_DESC>();"

    foreach ($Field in $Operator.Fields)
    {
        $Required = if ($Field.Optional) { 'false' } else { 'true' }
        $Deref = if ($Field.Optional) { '' } else { '*' }

        if ($Field.Type -eq "tensorDesc")
        {
            $Cpp += "    desc->$($Field.Name) = fused ? nullptr : ParseDmlTensorDescField(value, `"$($Field.Name)`", allocator, $Required);"
        }
        elseif ($Field.Type -eq "tensorDescArray")
        {
            $Cpp += "    desc->$($Field.Name) = fused ? nullptr : AsPointer(ParseDmlTensorDescArrayField(value, `"$($Field.Name)`", allocator, $Required));"
        }
        elseif ($Field.Type -eq "bool" -or $Field.Type -eq "bool_uint32")
        {
            $Cpp += "    desc->$($Field.Name) = ParseBoolField(value, `"$($Field.Name)`", $Required) ? 1 : 0;"
        }
        elseif ($Field.Type -eq "float32")
        {
            $Cpp += "    desc->$($Field.Name) = ParseFloat32Field(value, `"$($Field.Name)`", $Required);"
        }
        elseif ($Field.Type -eq "float32Array")
        {
            $Cpp += "    desc->$($Field.Name) = AsPointer(ParseFloat32ArrayField(value, `"$($Field.Name)`", allocator, $Required));"
        }
        elseif ($Field.Type -eq "int32")
        {
            $Cpp += "    desc->$($Field.Name) = ParseInt32Field(value, `"$($Field.Name)`", $Required);"
        }
        elseif ($Field.Type -eq "int32Array")
        {
            $Cpp += "    desc->$($Field.Name) = AsPointer(ParseInt32ArrayField(value, `"$($Field.Name)`", allocator, $Required));"
        }
        elseif ($Field.Type -eq "uint32")
        {
            if ($Field.Enum)
            {
                $EnumNameCamelCase = ConvertSnakeToCamelCase $Field.Enum
                $Cpp += "    desc->$($Field.Name) = Parse${EnumNameCamelCase}Field(value, `"$($Field.Name)`", $Required, {});"
            }
            else
            {
                $Cpp += "    desc->$($Field.Name) = ParseUInt32Field(value, `"$($Field.Name)`", $Required);"
            }
        }
        elseif ($Field.Type -eq "uint32Array")
        {
            $Cpp += "    desc->$($Field.Name) = AsPointer(ParseUInt32ArrayField(value, `"$($Field.Name)`", allocator, $Required));"
        }
        elseif ($Field.Type -eq "operatorDesc")
        {
            $Fused = if ($Field.Name -eq 'FusedActivation') { 'true' } else { 'false' }
            $Cpp += "    desc->$($Field.Name) = ParseDmlOperatorDescField(value, `"$($Field.Name)`", $Fused, allocator, $Required);"
        }
        elseif ($Field.Type -eq "operatorDescArray")
        {
            $Cpp += "    desc->$($Field.Name) = AsPointer(ParseDmlOperatorDescArrayField(value, `"$($Field.Name)`", true, allocator, $Required));"
        }
        elseif ($Field.Type -eq "scalarUnion")
        {
            $Cpp += "    desc->$($Field.Name) = ${Deref}ParseDmlScalarUnionField(value, `"$($Field.Name)`", `"ValueDataType`", allocator, $Required);"
        }
        elseif ($Field.Type -eq "scaleBias")
        {
            $Cpp += "    desc->$($Field.Name) = ParseDmlScaleBiasField(value, `"$($Field.Name)`", allocator, $Required);"
        }
        elseif ($Field.Type -eq "size2D")
        {
            $Cpp += "    desc->$($Field.Name) = ${Deref}ParseDmlSize2dField(value, `"$($Field.Name)`", allocator, $Required);"
        }
        else
        {
            throw "Serializing fields of type '$($Field.Type)' is not implemented!"
        }
    }

    $Cpp += "    auto opDesc = allocator.Allocate<DML_OPERATOR_DESC>();"
    $Cpp += "    opDesc->Type = DML_OPERATOR_$($Operator.Name);"
    $Cpp += "    opDesc->Desc = desc;"
    $Cpp += "    return opDesc;"
    $Cpp += "}"
    $Cpp += " "

    $Cpp += "Model::DmlDispatchableDesc::BindPoints GetBindPoints(const DML_$($Operator.Name)_OPERATOR_DESC& desc)"
    $Cpp += "{"
    $Cpp += "    Model::DmlDispatchableDesc::BindPoints bindPoints = {};"
    foreach ($Field in $Operator.Fields)
    {
        $Required = if ($Field.Optional) { 'false' } else { 'true' }

        if ($Field.Type -eq "tensorDesc")
        {
            if ($Field.Kind -eq "input")
            {
                $Cpp += "    bindPoints.inputs.push_back({`"$($Field.Name)`", 1, $Required});"
            }
            else
            {
                $Cpp += "    bindPoints.outputs.push_back({`"$($Field.Name)`", 1, $Required});"
            }
        }
        elseif ($Field.Type -eq "tensorDescArray")
        {
            if ($Field.Kind -eq "input")
            {
                $Cpp += "    bindPoints.inputs.push_back({`"$($Field.Name)`", desc.$($Field.SizeAttribute), $Required});"
            }
            else
            {
                $Cpp += "    bindPoints.outputs.push_back({`"$($Field.Name)`", desc.$($Field.SizeAttribute), $Required});"
            }
        }
    }
    $Cpp += "    return bindPoints;"
    $Cpp += "}"
    $Cpp += " "

    return $Cpp
}

$Schema = (Get-Content $SchemaFilePath) | ConvertFrom-Json

$SuccessfulOps = @()

$Cpp = @("// This file is generated by GenerateHelpers.ps1. Do not edit manually or your changes will be lost!")
$Cpp += ""
$Cpp += "// $('='*100)"
$Cpp += "// DIRECTML ENUMS"
$Cpp += "// $('='*100)"
$Cpp += ""
foreach ($Enum in $Schema.ApiEnums)
{
    $Cpp += WriteEnumParser $Enum
}

$Cpp += "// $('='*100)"
$Cpp += "// DIRECTML FLAGS"
$Cpp += "// $('='*100)"
$Cpp += ""
foreach ($Flags in $Schema.ApiFlags)
{
    $Cpp += WriteFlagsParser $Flags
}

$Cpp += "// $('='*100)"
$Cpp += "// DIRECTML OPERATORS"
$Cpp += "// $('='*100)"
$Cpp += ""
$AllOperators = $Schema.Operators + $Schema.ActivationOperators
foreach ($Operator in $AllOperators)
{
    if (!($Operator.Capabilities.FeatureLevel -le $MaxFeatureLevel))
    {
        Write-Host "Skipping '$($Operator.Name)': not supported until later feature levels"
        continue 
    }

    try
    {
        $Cpp += WriteOperatorFunction $Operator
        $SuccessfulOps += $Operator.Name
    }
    catch
    {
        Write-Warning "Could not generate code for '$($Operator.Name)' : $_"
    }
}

# --------------------------
# ParseDmlOperatorDesc
# --------------------------
$Cpp += "DML_OPERATOR_DESC* ParseDmlOperatorDesc(const rapidjson::Value& value, bool fused, BucketAllocator& allocator)"
$Cpp += "{"
$Cpp += "    if (!value.IsObject())"
$Cpp += "    {"
$Cpp += "        throw std::invalid_argument(`"Expected a non-null JSON object.`");"
$Cpp += "    }"
$Cpp += "    auto typeMember = value.FindMember(`"Type`");"
$Cpp += "    if (typeMember == value.MemberEnd())"
$Cpp += "    {"
$Cpp += "        typeMember = value.FindMember(`"type`");"
$Cpp += "    }"
$Cpp += "    if (typeMember == value.MemberEnd())"
$Cpp += "    {"
$Cpp += "        throw std::invalid_argument(`"Expected a member 'Type' with the operator type.`");"
$Cpp += "    }"
$Cpp += "    if (typeMember->value.GetType() != rapidjson::Type::kStringType)"
$Cpp += "    {"
$Cpp += "        throw std::invalid_argument(`"The member 'Type' must be a string.`");"
$Cpp += "    }"
$Cpp += "    auto type = typeMember->value.GetString();"
$Cpp += "    auto descMember = value.FindMember(`"Desc`");"
$Cpp += "    if (descMember == value.MemberEnd())"
$Cpp += "    {"
$Cpp += "        descMember = value.FindMember(`"desc`");"
$Cpp += "    }"
$Cpp += "    const rapidjson::Value& descValue = descMember != value.MemberEnd() ? descMember->value : value;"
foreach ($OperatorName in $SuccessfulOps)
{
    $OpFunctionName = "ParseDml$(ConvertSnakeToCamelCase $OperatorName)OperatorDesc"
    $Cpp += "    if (!strcmp(type, `"DML_OPERATOR_$OperatorName`") || !strcmp(type, `"$OperatorName`")) return $OpFunctionName(descValue, fused, allocator);"
}
$Cpp += "    throw std::invalid_argument(`"Unknown operator type.`");"
$Cpp += "}"
$Cpp += ""

# --------------------------
# GetBindPoints
# --------------------------
$Cpp += "Model::DmlDispatchableDesc::BindPoints GetBindPoints(const DML_OPERATOR_DESC& desc)"
$Cpp += "{"
$Cpp += "    switch (desc.Type)"
$Cpp += "    {"
foreach ($OperatorName in $SuccessfulOps)
{
    $OpFunctionName = "ParseDml$(ConvertSnakeToCamelCase $OperatorName)OperatorDesc"
    $Cpp += "    case DML_OPERATOR_${OperatorName}: return GetBindPoints(*reinterpret_cast<const DML_${OperatorName}_OPERATOR_DESC*>(desc.Desc));"
}
$Cpp += "    default: throw std::invalid_argument(`"Unknown operator type.`");"
$Cpp += "    }"
$Cpp += "}"

$Cpp | Out-File "$PSScriptRoot/../src/model/JsonParsersGenerated.cpp" -Encoding utf8