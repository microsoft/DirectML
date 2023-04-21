param(
    $n = 1,
    $c = 32,
    $h = 360,
    $w = 640,
    $layout = 'nhwc',
    $dataType = 'FLOAT32',
    [switch]$useDml,
    [parameter(ValueFromRemainingArguments=$true, Position=0)][string[]]$ExtraArgs
)

$elementCount = $n * $c * $h * $w

if ($layout -eq 'nchw')
{
    $strides = "[$($c*$h*$w), $($h*$w), $($w), 1]"
}
else
{
    $strides = "[$($h*$w*$c), 1, $($w*$c), $c]"
}

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
    $numThreads = 128
    $hlslType = if ($dataType -eq "FLOAT32") { "float" } else { "float16_t" }
    $groupsX = [Math]::Ceiling($elementCount / $numThreads)
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
                    { "name": "elementCount", "type": "UINT32",  "value": $elementCount }
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
                    "-D", "NUM_THREADS=$numThreads",
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