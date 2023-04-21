param(
    [int]$n = 1,
    [int]$c = 32,
    [int]$h = 360,
    [int]$w = 640,
    $layout = 'nhwc',
    $dataType = 'FLOAT32',
    [switch]$useDml,
    $numThreadsPerGroup = 128,
    $elementsPerThread = 1,
    [parameter(ValueFromRemainingArguments=$true, Position=0)][string[]]$ExtraArgs
)

$elementCount = $n * $c * $h * $w

if ($layout -eq 'nchw')
{
    $stridesList = (($c*$h*$w), ($h*$w), $w, 1)
    $strides = "[$($c*$h*$w), $($h*$w), $($w), 1]"
}
else
{
    $stridesList = (($h*$w*$c), 1, ($w*$c), $c)
    $strides = "[$($h*$w*$c), 1, $($w*$c), $c]"
}

$cumulativeShape = (($c * $h * $w), ($h * $w), $w, 1)

if ($useDml)
{
    $template = @"
    {
        "resources": 
        {
            "A": { "initialValuesDataType": "$dataType", "initialValues": { "valueCount": $elementCount, "value": 0 } },
            "B": { "initialValuesDataType": "$dataType", "initialValues": { "valueCount": $elementCount, "value": 0 } },
            "Out": { "initialValuesDataType": "$dataType", "initialValues": { "valueCount": $elementCount, "value": 0 } }
        },
    
        "dispatchables": 
        {
            "add": 
            {
                "type": "DML_OPERATOR_ELEMENT_WISE_ADD",
                "desc": 
                {
                    "ATensor": { "DataType": "$dataType", "Sizes": [$n,$c,$h,$w], "Strides": $strides },
                    "BTensor": { "DataType": "$dataType", "Sizes": [$n,$c,$h,$w], "Strides": $strides },
                    "OutputTensor": { "DataType": "$dataType", "Sizes": [$n,$c,$h,$w], "Strides": $strides }
                }
            }
        },
    
        "commands": [ { "type": "dispatch", "dispatchable": "add", "bindings": { "ATensor": "A", "BTensor": "B", "OutputTensor": "Out" } } ]
    }
"@
}
else
{
    $hlslType = if ($dataType -eq "FLOAT32") { "float" } else { "float16_t" }
    $groupsX = [Math]::Ceiling($elementCount / ($numThreadsPerGroup * $elementsPerThread))
    $threadGroupCount = "[$groupsX, 1, 1]"

    $hlslPath = (Resolve-Path "$PSScriptRoot/../models/hlsl_add.hlsl") -replace '\\','/'

    $template = @"
    {
        "resources": 
        {
            "A": { "initialValuesDataType": "$dataType", "initialValues": { "valueCount": $elementCount, "value": 0 } },
            "B": { "initialValuesDataType": "$dataType", "initialValues": { "valueCount": $elementCount, "value": 0 } },
            "Out": { "initialValuesDataType": "$dataType", "initialValues": { "valueCount": $elementCount, "value": 0 } },
            "Constants": 
            {
                "initialValuesDataType": "UNKNOWN",
                "initialValues": 
                [
                    { "name": "elementCount", "type": "UINT32",  "value": $elementCount },
                    { "name": "elementsPerThread", "type": "UINT32",  "value": $elementsPerThread },
                    { "name": "shapeN", "type": "UINT32",  "value": $n },
                    { "name": "shapeC", "type": "UINT32",  "value": $c },
                    { "name": "shapeH", "type": "UINT32",  "value": $h },
                    { "name": "shapeW", "type": "UINT32",  "value": $w },
                    { "name": "aStridesN", "type": "UINT32",  "value": $($stridesList[0]) },
                    { "name": "aStridesC", "type": "UINT32",  "value": $($stridesList[1]) },
                    { "name": "aStridesH", "type": "UINT32",  "value": $($stridesList[2]) },
                    { "name": "aStridesW", "type": "UINT32",  "value": $($stridesList[3]) },
                    { "name": "bStridesN", "type": "UINT32",  "value": $($stridesList[0]) },
                    { "name": "bStridesC", "type": "UINT32",  "value": $($stridesList[1]) },
                    { "name": "bStridesH", "type": "UINT32",  "value": $($stridesList[2]) },
                    { "name": "bStridesW", "type": "UINT32",  "value": $($stridesList[3]) },
                    { "name": "outStridesN", "type": "UINT32",  "value": $($stridesList[0]) },
                    { "name": "outStridesC", "type": "UINT32",  "value": $($stridesList[1]) },
                    { "name": "outStridesH", "type": "UINT32",  "value": $($stridesList[2]) },
                    { "name": "outStridesW", "type": "UINT32",  "value": $($stridesList[3]) },
                    { "name": "cumulativeShapeN", "type": "UINT32",  "value": $($cumulativeShape[0]) },
                    { "name": "cumulativeShapeC", "type": "UINT32",  "value": $($cumulativeShape[1]) },
                    { "name": "cumulativeShapeH", "type": "UINT32",  "value": $($cumulativeShape[2]) },
                    { "name": "cumulativeShapeW", "type": "UINT32",  "value": $($cumulativeShape[3]) },
                ],
                "sizeInBytes": 256
            }
        },
    
        "dispatchables": 
        {
            "add": 
            {
                "type": "hlsl",
                "sourcePath": "$hlslPath",
                "compiler": "dxc",
                "compilerArgs": 
                [
                    "-T", "cs_6_2",
                    "-E", "CSMain",
                    "-D", "NUM_THREADS=$numThreadsPerGroup",
                    "-D", "T=$hlslType",
                    "-enable-16bit-types"
                ]
            }
        },
    
        "commands": 
        [
            {
                "type": "dispatch",
                "dispatchable": "add",
                "threadGroupCount": $threadGroupCount,
                "bindings": { "inputA": "A", "inputB": "B", "output": "Out", "constants": "Constants" }
            }
        ]
    }
"@
}


$template | out-file "temp.json"
iex "..\build\win-x64\Release\dxdispatch.exe temp.json $ExtraArgs"
# rm temp.json