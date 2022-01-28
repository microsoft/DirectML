# DxDispatch Guide <!-- omit in toc -->

- [Overview](#overview)
- [Running the Program](#running-the-program)
- [Execution Model](#execution-model)
- [Models](#models)
  - [Resources](#resources)
    - [Buffers](#buffers)
    - [Buffer: Constant Initializer](#buffer-constant-initializer)
    - [Buffer: Array Initializer](#buffer-array-initializer)
    - [Buffer: Sequence Initializer](#buffer-sequence-initializer)
    - [Buffer: List Initializer](#buffer-list-initializer)
  - [Dispatchables](#dispatchables)
    - [DirectML Operator](#directml-operator)
    - [HLSL Compute Shader](#hlsl-compute-shader)
  - [Commands](#commands)
    - [Dispatch](#dispatch)
    - [Print](#print)
  - [Special Parsing Rules](#special-parsing-rules)
    - [Desc Structs (type and void*)](#desc-structs-type-and-void)
    - [Abbreviated Enum Values](#abbreviated-enum-values)
    - [DML_TENSOR_DESC](#dml_tensor_desc)
  - [Advanced Binding](#advanced-binding)
- [Scenarios](#scenarios)
  - [Debugging DirectX API Usage](#debugging-directx-api-usage)
  - [Benchmarking](#benchmarking)
  - [GPU Captures in PIX](#gpu-captures-in-pix)
  - [Shader Debugging in PIX](#shader-debugging-in-pix)

# Overview

DxDispatch is simple command-line executable for launching DirectX 12 compute programs without writing all the C++ boilerplate. The input to the tool is a JSON model that defines resources, dispatchables (compute shaders or DirectML operators), and commands to execute. 

This guide is organized as follows:
- The first section (**Running the Program**) shows basic command-line usage.
- The second section (**Execution Model**) explains the steps involves in executing a model.
- The third section (**Models**) covers the model schema with examples.
- The final section (**Scenarios**) goes through some of the ways you might use the tool.

# Running the Program

The most basic usage is to simply pass in the path to a JSON model. For example, this will run on of the examples:

```
> dxdispatch.exe .\models\dml_reduce.json

Running on 'NVIDIA GeForce RTX 2070 SUPER'
Resource 'input': 1, 2, 3, 4, 5, 6, 7, 8, 9
Resource 'output': 6, 15, 24
```

There are more options available to you. If you run the executable with no arguments (or `--help`) it will display the available options:

```
dxdispatch version 0.1.0
Usage:
  dxdispatch [OPTION...] <PATH_TO_MODEL_JSON>

  -d, --debug                Enable D3D and DML debug layers
  -b, --benchmark            Show benchmarking information
  -i, --dispatch_repeat arg  [Benchmarking only] The number of times to      
                             repeat each dispatch (default: 128)
  -h, --help                 Print command-line usage help
  -a, --adapter arg          Substring to match a desired DirectX adapter    
                             (default: )
  -s, --show_adapters        Show all available DirectX adapters
  -q, --direct_queue         Use a direct queue/lists (default is compute
                             queue/lists)
```



If your machine has multiple adapters you can display them:

```
> .\dxdispatch.exe -s    

NVIDIA GeForce RTX 2070 SUPER
-Version: 27.21.14.5671
-Hardware: true
-Integrated: false
-Dedicated Adapter Memory: 7.82 GB
-Dedicated System Memory: 0 bytes
-Shared System Memory: 15.92 GB

Intel(R) UHD Graphics 630
-Version: 27.20.100.8681
-Hardware: true
-Integrated: true
-Dedicated Adapter Memory: 128.00 MB
-Dedicated System Memory: 0 bytes
-Shared System Memory: 15.92 GB

Microsoft Basic Render Driver
-Version: 10.0.19041.546
-Hardware: false
-Integrated: false
-Dedicated Adapter Memory: 0 bytes
-Dedicated System Memory: 0 bytes
-Shared System Memory: 15.92 GB
```

Discrete hardware adapters are preferred by default. You can specify a desired adapter by passing a part of its name (will match by substring):

```
> dxdispatch.exe .\models\dml_reduce.json -a Intel

Running on 'Intel(R) UHD Graphics 630'
Resource 'input': 1, 2, 3, 4, 5, 6, 7, 8, 9
Resource 'output': 6, 15, 24
```

# Execution Model

Before going into the model schema, it's important to understand how models are executed: the model abstraction makes it easy to experiment, but it also preserves low-level control and flexibility. The only way to preserve this flexibility is to keep the abstraction close to how D3D12 programs are written. This doc assumes that you're familiar with D3D12 concepts like resources (buffers/textures), command lists, command queues, and barriers.

DxDispatch is an extremely basic command-line-only tool that executes a single model and then exits. The steps involved in executing a model are below. If an error occurs at any point the process will (hopefully) print a helpful message and immediately terminate.

1. **Load & parse JSON model**: the JSON model is converted into a C++ object representation
2. **Create device**: an appropriate DX adapter is selected to perform the work (you have some control over this). The necessary D3D/DML interfaces (devices, command list, queues, fences, etc.) are created.
3. **Allocate resources**: all resources defined in the model are allocated, initialized, and uploaded to GPU-visible memory. Completion of this step includes CPU/GPU synchronization.
4. **Initialize dispatchables**: all dispatchables (i.e. DML ops or HLSL shaders) defined in the model are created, compiled, and initialized. Completion of this step includes CPU/GPU synchronization.
5. **Execute commands**: all commands (e.g. dispatch, print resource) are processed in the order they're defined in the model. Each command is recorded into its own command list, and completion of each command includes CPU/GPU synchronization. All D3D work is done using a single D3D command list and command queue on a single thread. It is currently inefficient to issue multiple dispatch commands back-to-back since CPU/GPU synchronization will occur between each dispatch.

The execution model is imperative, so the order of commands matters and any side effects are permanent for the lifetime of the program. In particular, resource state will not be reinitialized for each dispatch command.

# Models

Each model is a collection of resources, dispatchables, and commands to execute. A model file has a root object with three members:

- **Resources**: a dictionary that maps names to GPU resources that will be initialized with specified values. The `Buffers` member is a dictionary that maps each buffer's user-specified name to its definition.
- **Dispatchables**: a dictionary that maps names to dispatchable objects. A dispatchable object is either a DirectML operator or HLSL compute shader.
- **Commands**: an array of instructions to run, such as executing dispatchables or printing resources.

Below is a very simple model that adds two buffer resources (using the DirectML API) and prints the result. This example has three resources named 'A', 'B', and 'Out' and a single dispatchable named 'simple_add'. The resource and dispatchable names can be any valid JSON string you like, but they must be unique with their respective dictionaries. You use these names to refer to the appropriate objects in model commands (e.g. binding resources).

```json
{
    "resources": 
    {
        "A": { "initialValuesDataType": "FLOAT32", "initialValues": [1,2,3] },
        "B": { "initialValuesDataType": "FLOAT32", "initialValues": [5,8,-5] },
        "Out": { "initialValuesDataType": "FLOAT32", "initialValues": { "valueCount": 3, "value": 0 } }
    },

    "dispatchables": 
    {
        "simple_add": 
        {
            "type": "DML_OPERATOR_ELEMENT_WISE_ADD",
            "desc": 
            {
                "ATensor": { "DataType": "FLOAT32", "Sizes": [3] },
                "BTensor": { "DataType": "FLOAT32", "Sizes": [3] },
                "OutputTensor": { "DataType": "FLOAT32", "Sizes": [3] }
            }
        }
    },

    "commands": 
    [
        {
            "type": "dispatch",
            "dispatchable": "simple_add",
            "bindings": { "ATensor": "A", "BTensor": "B", "OutputTensor": "Out" }
        },
        { "type": "print", "resource": "Out" }
    ]
}
```

## Resources

Each resource in the resources dictionary maps to a separate D3D resource (specifically a *committed* resource in a *default* heap). Textures are not yet implemented, so for now all resources must be buffers. Each resource declaration includes an initializer that is used to set the initial contents of the resource. **Keep in mind that initializers don't restrict the usage of the resource in any way: you can initialize a resource with floats, for example, and later bind it as a buffer of UINT32.**

### Buffers

All buffer resources have the following fields, though only the first two are required:

| Member                  | JSON Type                     | Description                                                |
| ----------------------- | ----------------------------- | ---------------------------------------------------------- |
| `initialValues`         | Object or array               | Determines initial contents of the buffer.                 |
| `initialValuesDataType` | String (DML_TENSOR_DATA_TYPE) | The data type associated with the buffer's initial values. |
| `sizeInBytes`           | Number (UINT64)               | **Optional**. Will be calculated if omitted.               |

The following rules apply:

- The `initialValuesDataType` field must be a string form of [DML_TENSOR_DATA_TYPE](https://docs.microsoft.com/en-us/windows/win32/api/directml/ne-directml-dml_tensor_data_type). You can use a shortened version that omits the `DML_TENSOR_DATA_TYPE_` prefix.
- JSON numbers will be converted to the appropriate data type in C++. You can use strings as well for "nan", "inf", and "-inf", but only when the data type is floating-point.
- The buffer's `sizeInBytes` will, if not supplied, be calculated as as number of elements in `initialValues` multiplied by the size of `initialValuesDataType`. If supplied, the `sizeInBytes` must be *at least* as large as the size calculated from the initial values. The total size will be inflated, if necessary, to meet DirectML's 4-byte alignment requirement.
- The `initialValues` are always written to the start of the buffer; if the `sizeInBytes` is larger than the implied size of the initial data then the end will be padded.
- The initial values are written into an upload-heap resource that is then copied to the default-heap buffer resource at startup; any commands that write into a buffer after startup will have a permanent effect on that buffer's contents for the duration of model execution.

### Buffer: Constant Initializer

The simplest way to initialize a buffer is filling it with a single repeated value of a specific data type. The example below will write `[0,0,0,0]` into the buffer.

- The `initialValuesDataType` must not be `"UNKNOWN"`.
- The `valueCount` must be larger than 0.

```json
{
    "initialValuesDataType": "FLOAT32",
    "initialValues": { "valueCount": 4, "value": 0 }
}
```

### Buffer: Array Initializer

You can initialize a buffer is using an array of values all with the same type:

- The `initialValuesDataType` must not be `"UNKNOWN"`.
- The `initialValues` size must be larger than 0.

```json
{
    "initialValuesDataType": "FLOAT16",
    "initialValues": [ 65504, -65504, 4, 3, -4, -6, "nan", "inf", "-inf" ]
}
```

### Buffer: Sequence Initializer

You can initialize a buffer is using a sequence. The example below will write `[1, 3.5, 6, 8.5]` into the buffer.

- The `initialValuesDataType` must not be `"UNKNOWN"`.
- The `valueCount` must be larger than 0.

```json
{
    "initialValuesDataType": "FLOAT32",
    "initialValues": { "valueCount": 4, "valueStart": 1, "valueDelta": 2.5 }
}
```

### Buffer: List Initializer

You can initialize a buffer is using an array of elements with different types and sizes. The primary use for this initializer is recording values for a constant buffer used in an HLSL dispatchable.

- The `initialValuesDataType` must be `"UNKNOWN"`.
- Elements in `initialValues` must be JSON objects with the `type` and `value` fields set. The `name` field is optional and purely for the user (like a comment).
- The `initialValues` size must be larger than 0.
- You'll want an alignment of 256 bytes if viewing this resource with a CBV, so take advantage of the `sizeInBytes` field!

```json
{
    "initialValuesDataType": "UNKNOWN",
    "initialValues": 
    [
        { "name": "elementCount", "type": "UINT32", "value": 6 },
        { "name": "alpha", "type": "FLOAT32", "value": 6 },
        { "name": "beta", "type": "FLOAT32", "value": 2.3 },
    ],
    "sizeInBytes": 256
}
```

## Dispatchables

Dispatchables are objects that can be written into a D3D command list. The model supports two types of dispatchables: DirectML operators and custom HLSL compute shaders.

### DirectML Operator

The JSON format for defining operators closely mirrors the DirectML API for creating operators: you are filling out a `DML_OPERATOR_DESC` struct that will be used to instantiate an `IDMLOperator` object. Below is an example that creates a dispatchable using `DML_CONVOLUTION_OPERATOR_DESC`:

```json
{
    "type": "DML_OPERATOR_CONVOLUTION",
    "desc": 
    {
        "InputTensor": { "DataType": "FLOAT32", "Sizes": [1,1,3,3] },
        "FilterTensor": { "DataType": "FLOAT32", "Sizes": [1,1,2,2] },
        "OutputTensor": { "DataType": "FLOAT32", "Sizes": [1,1,2,2] },
        "Mode": "DML_CONVOLUTION_MODE_CROSS_CORRELATION",
        "Direction": "DML_CONVOLUTION_DIRECTION_FORWARD",
        "DimensionCount": 2,
        "Strides": [1,1],
        "Dilations": [1,1],
        "StartPadding": [0,0],
        "EndPadding": [0,0],
        "OutputPadding": [0,0],
        "GroupCount": 1
    }
}
```

You must define **all** fields of the appropriate operator desc *unless* a field is marked as optional in the API. Optional fields in DirectML operator descs are generally pointers and annotated with `_Maybenull_` or `_Field_size_opt_`. If you do not define an optional field in the JSON dispatchable then it will be a nullptr in the C++ definition. Additionally, there are some [special parsing rules](#special-parsing-rules) that make it more convenient to define DML dispatchables: take note of the abbreviated enum values and defaults for `DML_TENSOR_DESC` fields.

**NOTE**: take care to use the same casing when setting the fields. Most of the JSON field names in the model start with a lowercase letter, but DML structs generally start with an uppercase letter.

### HLSL Compute Shader

You can execute custom compute shaders using an HLSL dispatchable. These objects will result in loading and compiling HLSL source at runtime, which can be very useful for prototyping. 

The example below shows how to reference the shader in the JSON model:

```json
{
    "type": "hlsl",
    "sourcePath": "models/add_fp16.hlsl",
    "compiler": "dxc",
    "compilerArgs": 
    [
        "-T", "cs_6_2",
        "-E", "CSMain",
        "-D", "NUM_THREADS=6",
        "-enable-16bit-types"
    ]
}
```

The contents of `models/add_fp16.hlsl` could, for example, point at the following contents:

```c
StructuredBuffer<float16_t> inputA;
StructuredBuffer<float16_t> inputB;
RWStructuredBuffer<float16_t> output;
cbuffer constants { uint elementCount; };

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x < elementCount)
    {
        output[dtid.x] = inputA[dtid.x] + inputB[dtid.x];
    }
}
```
**Notes**:
- Only the modern DXC compiler is supported at this time. I may add FXC support later.
- For command-line arguments refer to [this page](https://github.com/microsoft/DirectXShaderCompiler/wiki/Using-dxc.exe-and-dxcompiler.dll#using-the-compiler-interface). Pay special attention to how the arguments should be split in the array.
- A compatible root signature will be generated automatically. You do not have control over the root signature and should not declare one inline in the HLSL.
- A single descriptor table will reference all shader resources (buffers). SRVs, UAVs, and CBVs will be created automatically by reflecting the HLSL source and using appropriate views. You have some control over these views when binding (discussed later).
- You may declare shader resources using any type of buffer view, but textures are not supported. Arrays of resources (e.g. `Buffer<float> inputs[2];`), including unbounded arrays, are not yet supported. This is on the backlog though!
- If you declare a resource in HLSL but do not reference it in the shader program then it will likely be optimized away! Binding failures will result if you try to bind a buffer in the model to an unused shader input.

## Commands
### Dispatch

A dispatch command records a dispatchable into a D3D command list and executes it. Each dispatch command has the following fields:

| Member           | Type                     | DML      | HLSL     | Description                                        |
| ---------------- | ------------------------ | -------- | -------- | -------------------------------------------------- |
| type             | String                   | Required | Required | Must be `"dispatch"`.                              |
| dispatchable     | String                   | Required | Required | Name of the dispatchable object.                   |
| bindings         | String, Object, or Array | Required | Required | Size of each element (or structure) in the buffer. |
| threadGroupCount | Array                    | -        | Optional | Number of thread groups in X, Y, and Z dimensions. |

The only difference between DML and HLSL dispatches is that DML ops ignore the `threadGroupCount` field (defaults to `[1,1,1]` for HLSL if omitted).

Below is an example of a dispatch command for a DML operator:

```json
{
    "type": "dispatch",
    "dispatchable": "join",
    "bindings": 
    {
        "InputTensors": [ "A", "B", "C" ],
        "OutputTensor": "Out"
    }
}
```

Below is an example of a dispatch command for an HLSL operator:

```json
{
    "type": "dispatch",
    "dispatchable": "add",
    "threadGroupCount": [ 2, 1, 1 ],
    "bindings": 
    {
        "inputA": "A",
        "inputB": "B",
        "output": "Out",
        "constants": "Constants"
    }
}
```


The most important property to discuss in this command is the `bindings`, which ties the resources in the model to the *bind points* in the dispatchable. The `bindings` field is a dictionary that maps the name of a bind point to an array of resource bindings. Bind points in DML operators are the names of `DML_TENSOR_DESC` fields in the operator desc. For example, the join operator has **two** bind points: `InputTensors` and `OutputTensor`. Each bind point may have 1 or more resources bound to it. In the case of join, `InputTensors` has a variable number of resources bound to it, which is determined using the value of of `InputCount`. The `OutputTensor` bind point must have 1 and only 1 resource bound to it.

```cpp
struct DML_JOIN_OPERATOR_DESC
{
    UINT InputCount;
    _Field_size_(InputCount) const DML_TENSOR_DESC* InputTensors;
    const DML_TENSOR_DESC* OutputTensor;
    UINT Axis;
};
```

Bind points in compute shaders are the shader input names. The following HLSL has 4 bind points: `inputA`, `inputB`, `output`, and `constants`:

```c
StructuredBuffer<float16_t> inputA;
StructuredBuffer<float16_t> inputB;
RWStructuredBuffer<float16_t> output;
cbuffer constants { uint elementCount; };
```



### Print

This command is used to print the contents of a resource to stdout. If the resource lives in a GPU-visible-only heap then it will first be downloaded into a CPU-visible readback heap. Buffers are always printed as a flat 1D view of elements: the data type and number of elements display will be derived using the resource's initializer. More control over printing may be added in the future.

```json
{ 
    "type": "Print", 
    "resource": "Out" 
}
```

## Special Parsing Rules

### Desc Structs (type and void*)

DirectML "desc" structs use a type enumeration and `void*` pair to achieve polymorphism: 

- `DML_OPERATOR_DESC` is used to describe any type of operator (e.g. element-wise identity).
- `DML_TENSOR_DESC` is used to describe any type of tensor (currently only buffers). 

These high-level desc structs have no fields except for `<enum> Type` and `const void* Desc`; their only purpose is to give the API a way to communicate a polymorphic type to DirectML without using `IUnknown` or rev'ing structs and APIs for each new subtype that's added. As such, it's *usually* not useful to retain the exact layout of these structs when representing objects in JSON.

For example, instead of writing the `FusedActivation` member like this:

```json
{
    "type": "DML_OPERATOR_ELEMENT_WISE_ADD1",
    "desc": 
    {
        "ATensor": { "DataType": "FLOAT32", "Sizes": [3] },
        "BTensor": { "DataType": "FLOAT32", "Sizes": [3] },
        "OutputTensor": { "DataType": "FLOAT32", "Sizes": [3] },
        "FusedActivation": { "Type": "ACTIVATION_ELU", "Desc": { "Alpha": 0.5 } }
    }
}
```

... it's easier to write the field `Alpha` inline without the extra `Desc` level:

```json
{
    "type": "DML_OPERATOR_ELEMENT_WISE_ADD1",
    "desc": 
    {
        "ATensor": { "DataType": "FLOAT32", "Sizes": [3] },
        "BTensor": { "DataType": "FLOAT32", "Sizes": [3] },
        "OutputTensor": { "DataType": "FLOAT32", "Sizes": [3] },
        "FusedActivation": { "Type": "ACTIVATION_ELU", "Alpha": 0.5 }
    }
}
```

When defining DirectML desc structs in JSON, the parser will recognize either of the above formats. However, if the subtype itself has a field named `Type` then it will collide with the `Type` field of the higher-level struct; in this case you **must** use the longer format and explicitly list fields using the `Desc` member:

```json
{
    "Type": "DML_OPERATOR_RANDOM_GENERATOR",
    "Desc":
    {
        "InputTensor": { ... },
        "OutputTensor": { ... },
        "OutputStateTensor": { ... },
        "Type": "DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10"
    }  
}
```

### Abbreviated Enum Values

DirectML enumerations follow a naming convention where values are prefixed by their respective type name. The JSON parser allows initializing enum values by either their full name or their suffix. Below are some examples:

| Type                               | Example Value                                         | Example Value Suffix |
| ---------------------------------- | ----------------------------------------------------- | -------------------- |
| <b><i>DML_TENSOR_TYPE</b></i>      | <b><i>DML_TENSOR_TYPE_</b></i>BUFFER                  | BUFFER               |
| <b><i>DML_TENSOR_DATA_TYPE</b></i> | <b><i>DML_TENSOR_DATA_TYPE_</b></i>FLOAT32            | FLOAT32              |
| <b><i>DML_REDUCE_FUNCTION</b></i>  | <b><i>DML_REDUCE_FUNCTION</b></i>_SUM                 | SUM                  |
| <b><i>DML_OPERATOR_</b></i>TYPE    | <b><i>DML_OPERATOR_</b></i>GEMM                       | GEMM                 |
| <b><i>DML_EXECUTION_FLAG</b></i>S  | <b><i>DML_EXECUTION_FLAG</b></i>_DESCRIPTORS_VOLATILE | DESCRIPTORS_VOLATILE |

Take note of the few odd cases that don't follow the usual rule exactly:

- Enum values of type `DML_OPERATOR_TYPE` omit `_TYPE` from their prefix. It's `DML_OPERATOR_GEMM`, not `DML_OPERATOR_TYPE_GEMM`.
- Flag values are singular and omit the "S". It's `DML_EXECUTION_FLAG_NONE`, not `DML_EXECUTION_FLAGS_NONE`. 

### DML_TENSOR_DESC

Since tensor descs are so common, the JSON parser provides default values for most fields.

| Field                           | DML Type             | JSON Type | Required | Default                                                   |
| ------------------------------- | -------------------- | --------- | -------- | --------------------------------------------------------- |
| `Type`                          | DML_TENSOR_TYPE      | String    | No       | DML_TENSOR_TYPE_BUFFER                                    |
| `DataType`                      | DML_TENSOR_DATA_TYPE | String    | Yes      | -                                                         |
| `Flags`                         | DML_TENSOR_FLAGS     | String    | No       | DML_TENSOR_FLAG_NONE                                      |
| `DimensionCount`                | UINT                 | Integer   | No       | inferred from size of `Sizes` field                       |
| `Sizes`                         | UINT*                | List      | Yes      | -                                                         |
| `Strides`                       | UINT*                | List      | No       | null                                                      |
| `TotalTensorSizeInBytes`        | UINT                 | Integer   | No       | inferred (packed) from `Sizes`, `Strides`, and `DataType` |
| `GuaranteedBaseOffsetAlignment` | UINT                 | Integer   | No       | 0                                                         |

Below is an example of a fully defined `DML_TENSOR_DESC`:

```json
{ 
    "Type": "DML_TENSOR_TYPE_BUFFER",
    "Desc":
    {
        "DataType": "DML_TENSOR_DATA_TYPE_FLOAT32",
        "Flags": "DML_TENSOR_FLAG_NONE",
        "DimensionCount": 4,
        "Sizes": [1,1,2,3],
        "Strides": [6,6,3,1],
        "TotalTensorSizeInBytes": 24,
        "GuaranteedBaseOffsetAlignment": 0
    }
}
```

The following `DML_TENSOR_DESC` in JSON is equivalent to the definition above:

```json
{ 
    "DataType": "FLOAT32",
    "Sizes": [1,1,2,3],
}
```

## Advanced Binding

In this simplest case you provide a resource binding by its name only (e.g. `"inputA": "A"`). However, you also have the option of providing additional information to view a subrange of the resource or reinterpret its type. Below is an example that fills out a binding object with these additional properties:

Binding object:
```json
{
    "name": "A",
    "format": "R32_FLOAT",
    "elementCount": 12,
    "elementStride": 4,
    "elementOffset": 16
}
```

| Member          | Type                 | DML      | HLSL     | Description                                         |
| --------------- | -------------------- | -------- | -------- | --------------------------------------------------- |
| `name`          | String               | Required | Required | Name of the model buffer to bind.                   |
| `format`        | String (DXGI_FORMAT) | -        | Optional | Format to use for the buffer view.                  |
| `elementCount`  | Number (UINT32)      | -        | Optional | Number of elements in the buffer.                   |
| `elementStride` | Number (UINT32)      | -        | Optional | Size of each element (or structure) in the buffer.  |
| `elementOffset` | Number (UINT32)      | Optional | Optional | Offset (measured in elements) to the first element. |

When binding with only a name the above values default to viewing the portion of the resource that is initialized when declared in the model. So, for example, if you have a 64KB buffer named "A" that is initialized with 3x FLOAT32 values the default view will have `elementCount=3`, `elementStride=4`, `elementOffset=0`, `format=R32_FLOAT`. This is generally what you want, but the extra parameters can be useful for experimentation.

Finally, recall that you may bind *multiple* resources to a bind point. This means a bind point in the model `bindings` may map to a string, an object, or an array of strings/objects. The example below shows advanced binding that mixes objects and strings, including binding different regions of the same resource "A":
```json
{
    "type": "dispatch",
    "dispatchable": "join",
    "bindings": 
    {
        "InputTensors": 
        [ 
            { "name": "A", "elementCount": 3, "elementOffset": 0 },
            "B",
            { "name": "A", "elementCount": 5, "elementOffset": 4 } 
        ],
        "OutputTensor": "Out"
    }
}
```

# Scenarios

## Debugging DirectX API Usage

It's quite easy to make a mistake when creating, compiling, and binding DirectML operators. The error codes returned from an invalid API call are rarely useful enough to diagnose the problem, so these libraries have dedicated debug layers that can be turned on during development and turned off (for efficiency) in deployment. You can enable these debug layers using the `--debug` option when running the program. For example, let's say you call the reduce operator incorrectly with the following dispatchable:

```json
"sum":
{
    "type": "DML_OPERATOR_REDUCE",
    "desc": 
    {
        "InputTensor": { "DataType": "FLOAT32", "Sizes": [1,1,3,3] },
        "OutputTensor": { "DataType": "FLOAT32", "Sizes": [1,1,1,1] },
        "Function": "DML_REDUCE_FUNCTION_SUM",
        "AxisCount": 1,
        "Axes": [3]
    }
}
```

The above output shape in the above code is wrong: only axis 3 is indicated as being reduced. If you run the model normally it will simply tell you that creating the operator failed:

```
> dxdispatch.exe .\models\dml_reduce.json   

Running on 'NVIDIA GeForce RTX 2070 SUPER'
ERROR: failed to execute the model.
ERROR creating dispatchable 'sum'
```

With the debug layers enabled you'll get a helpful message from the DirectML debug layers:

```
> dxdispatch.exe .\models\dml_reduce.json -d 

Running on 'NVIDIA GeForce RTX 2070 SUPER'
DML_OPERATOR_REDUCE: expected OutputTensor.Sizes[2] to be InputTensor.Sizes[2] since dimension 2 is not reduced, but 1 and 3 were provided
ERROR: failed to execute the model.
ERROR creating dispatchable 'sum'
```

Likewise, you will get output from the D3D debug layers (including GPU-based validation). If you bind a buffer that's too small:

```
Running on 'NVIDIA GeForce RTX 2070 SUPER'
ERROR while binding resources: dxdispatch.exe!00007FF788C0F7AD: (caller: 00007FF788C95E2D) Exception(1) tid(50dc) 887A0005 The GPU device instance has been suspended. Use GetDeviceRemovedReason to determine the appropriate action.
    [DmlDispatchable::Bind(m_device->DML()->GetDeviceRemovedReason())]

Error during binding for dispatchable 'sum': The value of SizeInBytes is too small. 4 was provided, but 36 was expected.
Resource 'input': 1
Resource 'output': 0, 0, 0
```

The debug layers won't catch everything, but they can be extremely helpful. Keep in mind running with the `--debug` option will greatly reduce performance, so don't use it with benchmarking!

## Benchmarking

The `--benchmark` option gives you a simple way to measure the elapsed time of a dispatch. It's usually better to profile using a dedicated tool like PIX, but this option can be useful for quickly scripting multiple executions with different arguments (e.g. problem size).

```
> dxdispatch.exe .\models\dml_reduce.json -b

Running on 'NVIDIA GeForce RTX 2070 SUPER'
Dispatch 'sum': 0.00671445 ms
Resource 'input': 1, 2, 3, 4, 5, 6, 7, 8, 9
Resource 'output': 6, 15, 24
```

When `--benchmark` mode is enabled each dispatch command will record the multiple instances of the same dispatchable into the same command list and divide the total execution duration by the number of instances. The default is 256 times, but you can control this with `--dispatch_repeat <i>`. For example, the following invocation will record the DML reduce operator 1000 times:

```
> dxdispatch.exe .\models\dml_reduce.json -b -i 1000

Running on 'NVIDIA GeForce RTX 2070 SUPER'
Dispatch 'sum': 0.00562345 ms
Resource 'input': 1, 2, 3, 4, 5, 6, 7, 8, 9
Resource 'output': 6, 15, 24
```

A larger number of repeats may give a more accurate time. This is a very simplistic benchmarking technique and may be improved in the future.

## GPU Captures in PIX

For a deeper look into performance you'll want to use a dedicated profiling tool like [PIX](https://devblogs.microsoft.com/pix/introduction/). Hardware vendors also provide their own profiling tools that should also be compatible. Using these tools is outside the scope of this guide, but as an example here's a GPU capture with PIX:

![](images/pix_gpu_capture0.png)

You'll notice that resources and dispatches are labeled with the names of objects in the model. This makes it easy to map D3D objects in the capture back to your model objects:

![](images/pix_gpu_capture1.png)

The image below shows some of the occupancy and timing data captured (the dml_reduce.json model was modified to have ~4M elements, since reducing 9 elements is not interesting enough for profiling). As you might expect, reduction has low arithmetic complexity and thus the ALU/FMA pipes are not too busy; however, there is good occupancy on the cores and some degree of L1 cache hits. Reduction is memory-bandwidth bound.

![](images/pix_gpu_capture2.png)

## Shader Debugging in PIX

The profiling examples in the previous example used a DML operator, but of course you can profile your custom shaders using the same approach. HLSL dispatchables can also be debugged and modified on the fly in PIX. Make sure to add `-Zi` (or `-Zs`) and `-Od` to your HLSL dispatchable arguments, which will disable optimizations produce a PDB for your compiled shader. You can then launch your HLSL model, step through it, and even make changes all within PIX:

![](images/pix_shader_debug.png)

Below is a modified version of `add_fp16.json` used in the example above. Within PIX you may need to resolve the PDB path (should be generated in the working directory from which dxdispatch.exe was launched).

```json
"add": 
{
    "type": "hlsl",
    "sourcePath": "models/add_fp16.hlsl",
    "compiler": "dxc",
    "compilerArgs": 
    [
        "-T", "cs_6_2",
        "-E", "CSMain",
        "-D", "NUM_THREADS=6",
        "-enable-16bit-types",
        "-Zs", "-Od"
    ]
}
```