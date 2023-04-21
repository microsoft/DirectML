param(
    $n = 1,
    $c = 32,
    $h = 360,
    $w = 640,
    $layout = 'nhwc',
    $dataType = 'FLOAT32',
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

$template | out-file "temp.json"
iex "..\build\win-x64\Release\dxdispatch.exe temp.json $ExtraArgs"
rm temp.json