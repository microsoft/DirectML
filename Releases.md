# DirectML Release History <!-- omit in toc -->

See [DirectML version history on MSDN](https://docs.microsoft.com/windows/win32/direct3d12/dml-version-history) for more detailed notes.

| Version                  | Feature level                                                                                                            | First available in OS                                             | Redistributable                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [1.15.1](#directml-1151) | [DML_FEATURE_LEVEL_6_4](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_6_4) | TBD                                                               | [Microsoft.AI.DirectML.1.15.1](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.1) |
| [1.15.0](#directml-1150) | [DML_FEATURE_LEVEL_6_4](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_6_4) | TBD                                                               | [Microsoft.AI.DirectML.1.15.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.0) |
| [1.13.1](#directml-1131) | [DML_FEATURE_LEVEL_6_2](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_6_2) | TBD                                                               | [Microsoft.AI.DirectML.1.13.1](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.13.1) |
| [1.13.0](#directml-1130) | [DML_FEATURE_LEVEL_6_2](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_6_2) | TBD                                                               | [Microsoft.AI.DirectML.1.13.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.13.0) |
| [1.12.1](#directml-1121) | [DML_FEATURE_LEVEL_6_1](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_6_1) | TBD                                                               | [Microsoft.AI.DirectML.1.12.1](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.12.1) |
| [1.12.0](#directml-1120) | [DML_FEATURE_LEVEL_6_1](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_6_1) | TBD                                                               | [Microsoft.AI.DirectML.1.12.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.12.0) |
| [1.11.0](#directml-1110) | [DML_FEATURE_LEVEL_6_0](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_6_0) | TBD                                                               | [Microsoft.AI.DirectML.1.11.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.11.0) |
| [1.10.1](#directml-1101) | [DML_FEATURE_LEVEL_5_2](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_5_2) | TBD                                                               | [Microsoft.AI.DirectML.1.10.1](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.10.1) |
| [1.10.0](#directml-1100) | [DML_FEATURE_LEVEL_5_2](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_5_2) | TBD                                                               | [Microsoft.AI.DirectML.1.10.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.10.0) |
| [1.9.1](#directml-191)   | [DML_FEATURE_LEVEL_5_1](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_5_1) | TBD                                                               | [Microsoft.AI.DirectML.1.9.1](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.9.1)   |
| [1.9.0](#directml-190)   | [DML_FEATURE_LEVEL_5_1](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_5_1) | TBD                                                               | [Microsoft.AI.DirectML.1.9.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.9.0)   |
| [1.8.2](#directml-182)   | [DML_FEATURE_LEVEL_5_0](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_5_0) | TBD                                                               | [Microsoft.AI.DirectML.1.8.2](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.8.2)   |
| [1.8.1](#directml-181)   | [DML_FEATURE_LEVEL_5_0](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_5_0) | TBD                                                               | [Microsoft.AI.DirectML.1.8.1](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.8.1)   |
| [1.8.0](#directml-180)   | [DML_FEATURE_LEVEL_5_0](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_5_0) | Windows 11 2022 Update (Build 22621, 22H2)                        | [Microsoft.AI.DirectML.1.8.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.8.0)   |
| [1.7.0](#directml-170)   | [DML_FEATURE_LEVEL_4_1](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_4_1) | Redistributable only                                              | [Microsoft.AI.DirectML.1.7.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.7.0)   |
| [1.6.0](#directml-160)   | [DML_FEATURE_LEVEL_4_0](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_4_0) | Windows 11 (Build 10.0.22000; 21H2)                               | [Microsoft.AI.DirectML.1.6.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.6.0)   |
| [1.5.1](#directml-151)   | [DML_FEATURE_LEVEL_3_1](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_3_1) | Redistributable only                                              | [Microsoft.AI.DirectML.1.5.1](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.5.1)   |
| [1.5.0](#directml-150)   | [DML_FEATURE_LEVEL_3_1](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_3_1) | Redistributable only                                              | [Microsoft.AI.DirectML.1.5.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.5.0)   |
| [1.4.3](#directml-143)   | [DML_FEATURE_LEVEL_3_0](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_3_0) | Redistributable only                                              | [Microsoft.AI.DirectML.1.4.3](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.4.3)   |
| [1.4.2](#directml-142)   | [DML_FEATURE_LEVEL_3_0](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_3_0) | Redistributable only                                              | [Microsoft.AI.DirectML.1.4.2](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.4.2)   |
| [1.4.1](#directml-141)   | [DML_FEATURE_LEVEL_3_0](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_3_0) | Redistributable only                                              | [Microsoft.AI.DirectML.1.4.1](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.4.1)   |
| [1.4.0](#directml-140)   | [DML_FEATURE_LEVEL_3_0](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_3_0) | Redistributable only                                              | [Microsoft.AI.DirectML.1.4.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.4.0)   |
| [1.1.0](#directml-110)   | [DML_FEATURE_LEVEL_2_0](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_2_0) | Windows 10 May 2020 Update, Version 2004 (Build 10.0.19041, 20H1) | -                                                                                           |
| [1.0.0](#directml-100)   | [DML_FEATURE_LEVEL_1_0](https://learn.microsoft.com/windows/ai/directml/dml-feature-level-history#dml_feature_level_1_0) | Windows 10 May 2019 Update, Version 1903 (Build 10.0.18362; 19H1) | -                                                                                           |

# DirectML 1.15.1
- Improved performance across many operators on certain NPU hardware
  
# DirectML 1.15.0
- Introduced DML_FEATURE_LEVEL 6.4:
  - Added 3 new operators:
    - DML_OPERATOR_RESAMPLE3
    - DML_OPERATOR_FOLD
    - DML_OPERATOR_UNFOLD
  - Updated 3 operators:
    - DML_OPERATOR_PADDING (updated to accept DML_PADDING_MODE::DML_PADDING_MODE_WRAP)
    - DML_OPERATOR_PADDING1 (updated to accept DML_PADDING_MODE::DML_PADDING_MODE_WRAP)
    - DML_OPERATOR_ACTIVATION_SOFTPLUS (updated to allow Steepness < 1)

- Introduced DML_FEATURE_LEVEL 6.3:
  - Added data type Support:
    - DML_TENSOR_DATA_TYPE_UINT4
    - DML_TENSOR_DATA_TYPE_INT4
  - Added 4 new operators:
    - DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2
    - DML_OPERATOR_MULTIHEAD_ATTENTION1
    - DML_OPERATOR_QUANTIZE (added with INT4/UINT4 support)
    - DML_OPERATOR_DEQUANTIZE (added with INT4/UINT4 support)

- Optimizations:
  - (LLM) Added INT4 Dequantize + GEMM fusion metacommand and DXIL lowerings.
  - (LLM) Added Multihead Attention fusion
  - Added Gemm fusion optimizations
  - (Intel ARC GPU) Fix pooling metacommand calls by driver version
 
Bug fixes:
  - Swish now produces correct output when invoked with strided input tensors
  - Intel
    - (Precision) FP16 GemmWave emulated on FP32
 
# DirectML 1.13.1

- Fixed performance issues and bugs introduced in the 1.13 release for models that use the Join operator.
- Fixed performance issues bugs for models that contain GEMM followed by Add.
- Fixed bug with constant nodes for large data sizes.

# DirectML 1.13.0

- Enabled Intel Core Ultra NPU support with performance focused on select deep learning models. 
- Introduced DML_FEATURE_LEVEL 6.2:
  - Added 6 new operators:
    - DML_OPERATOR_LP_POOLING1
    - DML_OPERATOR_AVERAGE_POOLING1
    - DML_OPERATOR_ACTIVATION_SWISH
    - DML_OPERATOR_ACTIVATION_HARD_SWISH
    - DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING
    - DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT
- Made ZeroPointTensor optional for DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR and DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR.
- Added a new graph node type DML_GRAPH_NODE_TYPE_CONSTANT to enable compile-time optimizations that require content of small tensors.
- Fixed bug in DML_OPERATOR_LSTM when used with reduction-based activation function like DML_OPERATOR_ACTIVATION_HARDMAX. 
- Fixed bug in DML_OPERATOR_GEMM when there is additional padding and channel axis is a multiple of 16 bytes.
- Added UINT8/INT8 data type support for DML_OPERATOR_RESAMPLE2.
- Added ARM64EC variant for DirectML redistributable package in NuGet.
- Updated Xbox Series X/S binaries to support the 230307, 230603, and 231000 GDK editions.
- Bug fixes.

**Known Issue**: models containing Join operations may exhibit a slight performance regression relative to 1.12.1, particularly if the model is low-precision or memory-bandwidth limited. This issue will be fixed in the 1.13.1 release.

# DirectML 1.12.1

- Fixed bug in compiled graphs with elided cast nodes that could result in some models producing invalid output values ([issue](https://github.com/microsoft/DirectML/issues/483)).

# DirectML 1.12.0

- Introduced DML_FEATURE_LEVEL 6.1
  - Added DML_OPERATOR_MULTIHEAD_ATTENTION.
  - Added DML_OPERATOR_ACTIVATION_SOFTMAX and DML_OPERATOR_ACTIVATION_SOFTMAX1 to the list of fusable activations for DML_OPERATOR_GEMM.

# DirectML 1.11.0

- Introduced DML_FEATURE_LEVEL 6.0
  - Added UINT64 and INT64 data type support for DML_OPERATOR_ELEMENT_WISE_DIVIDE, DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR, and DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE.
  - Added FLOAT16 data type support in ScaleTensor for DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR
  - Added FLOAT16 data type support in ScaleTensor and OutputTensor for DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR
  - Added DML_OPERATOR_ELEMENT_WISE_CLIP operator to the supported fused activation list.
- Improved performance of DML_OPERATOR_ACTIVATION_SOFTMAX/1.
- Improved performance of DML_OPERATOR_GEMM on certain GPU hardware.
- Improved performance of DML_OPERATOR_CONVOLUTION_INTEGER and DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION on INT8 tensors.
- Fixed bug in DML_OPERATOR_CONVOLUTION_INTEGER and DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION on certain GPU hardware.
- Fixed crash in DML graph compilation when DML_OPERATOR_ROI_ALIGN_GRAD is used.
- Fixed Windows App Certification Kit failure in windows app using DirectML.

# DirectML 1.10.1

- Fixed bug in DMLCreateDevice1 that would cause it to incorrectly fail when a minimumFeatureLevel greater than DML_FEATURE_LEVEL_5_0 is supplied.

# DirectML 1.10.0

- Introduced DML_FEATURE_LEVEL 5.2:
  - Expanded supported rank count across certain DML operators.
  - Enabled DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION/1 to allow optional ScaleTensor regardless of the value of the BiasTensor and vice-versa.
- Enabled DMLCreateDevice/1 to use DML_CREATE_DEVICE_FLAG_DEBUG flag even if Direct3D 12 debug layer is not enabled.
- Improved DML_OPERATOR_RESAMPLE/1/2 APIs performance significantly.
- Improved graph-level layout transformation logic performance significantly.
- Fixed DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE/FLOOR precision issue with non-power of 2 on specific GPUs.
- Fixed DML_OPERATOR_CAST casting between 16-bit and 64-bit data types on specific GPUs.
- Fixed DML_OPERATOR_ACTIVATION_HARDMAX/1 result for certain OutputTensor stride.
- Added Xbox Series X/S binaries to support the 220301, 220604, and 221000 GDK editions.

# DirectML 1.9.1

- Fixed bug in DML_OPERATOR_ONE_HOT operator when using large uint64 indices. 
- Improve FP32 convolution performance.
- Improve DML_OPERATOR_JOIN operator performance.
- Fix bug with unconnected split nodes when executing in DML graph.
- Fix identity node optimization when near end of DML graph.

# DirectML 1.9.0

- Introduces [DML_FEATURE_LEVEL_5_1](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_5_1)
  - Adds 7 new operators:
    - DML_OPERATOR_ACTIVATION_GELU
    - DML_OPERATOR_ACTIVATION_SOFTMAX1
    - DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1
    - DML_OPERATOR_ACTIVATION_HARDMAX1
    - DML_OPERATOR_RESAMPLE2
    - DML_OPERATOR_RESAMPLE_GRAD1
    - DML_OPERATOR_DIAGONAL_MATRIX1
- GRU significant performance boost.
- INT8 convolution performance improvement using DP4A HLSL intrinsics.

# DirectML 1.8.2

- Fix Linux-specific execution failure in a TensorFlow graph due to bad alignment related to bitscan forward instruction.
- Fix incorrect results in 2D convolution with certain combinations of parameters where group count > 1 ([issue](https://github.com/microsoft/DirectML/issues/234)).

# DirectML 1.8.1

- Fix telemetry bug that caused slower CPU execution over time with repeated operator creation.

# DirectML 1.8.0

- Introduces [DML_FEATURE_LEVEL_5_0](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_5_0)
  - Adds 4 new operators:
    - DML_OPERATOR_ELEMENT_WISE_CLIP1
    - DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD1
    - DML_OPERATOR_PADDING1
    - DML_OPERATOR_ELEMENT_WISE_NEGATE
  - Supports 64-bit data type for operators: CLIP, CLIP_GRAD, CUMULATIVE_SUMMATION, CUMULATIVE_PRODUCT, ELEMENT_WISE_MAX, ELEMENT_WISE_MIN, REDUCE+REDUCE_FUNCTION_MAX, REDUCE+REDUCE_FUNCTION_MAX, REDUCE+REDUCE_FUNCTION_SUM, REDUCE+REDUCE_FUNCTION_MULTIPLY, REDUCE+REDUCE_FUNCTION_SUM_SQUARE, REDUCE+REDUCE_FUNCTION_L1, PADDING, SPACE_TO_DEPTH, DEPTH_TO_SPACE, TOP_K, ELEMENT_WISE_NEGATE, ELEMENT_WISE_IF, MAX_POOLING, MAX_UNPOOLING, FILL_VALUE_SEQUENCE, REVERSE_SUBSEQUENCES, ROI_ALIGN BatchIndicesTensor.
- Bug fixes.

# DirectML 1.7.0

- Introduces [DML_FEATURE_LEVEL_4_1](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_4_1)
  - Adds 3 new operators:
    - DML_OPERATOR_ROI_ALIGN_GRAD
    - DML_OPERATOR_BATCH_NORMALIZATION_TRAINING
    - DML_OPERATOR_BATCH_NORMALIZATION_TRAINING_GRAD
  - Supports 64-bit data type for operators: ELEMENT_WISE_IDENTITY, ELEMENT_WISE_ADD, ELEMENT_WISE_SUBTRACT, ELEMENT_WISE_MULTIPLY, ELEMENT_WISE_ABS, ELEMENT_WISE_SIGN, ELEMENT_WISE_LOGICAL_EQUALS, ELEMENT_WISE_LOGICAL_GREATER_THAN, ELEMENT_WISE_LOGICAL_LESS_THAN, ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL, ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL, ELEMENT_WISE_BIT_SHIFT_LEFT, ELEMENT_WISE_BIT_SHIFT_RIGHT, ELEMENT_WISE_BIT_AND, ELEMENT_WISE_BIT_OR, ELEMENT_WISE_BIT_NOT, ELEMENT_WISE_BIT_XOR, ELEMENT_WISE_BIT_COUNT, ARGMIN, ARGMAX, CAST, SLICE, SLICE1, SLICE_GRAD, SPLIT, JOIN, GATHER, GATHER_ELEMENTS, GATHER_ND, GATHER_ND1, SCATTER, SCATTER_ND, FILL_VALUE_CONSTANT, TILE, ONE_HOT
- Substantial performance improvements for several operators (especially in training scenarios).
- Bug fixes.

# DirectML 1.6.0

- Introduces [DML_FEATURE_LEVEL_4_0](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_4_0)
  - Adds 3 new operators:
    - DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD
    - DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR
    - DML_OPERATOR_ROI_ALIGN1
  - Supports 8D tensors for operators: FILL_VALUE_CONSTANT, FILL_VALUE_SEQUENCE, CUMULATIVE_SUMMATION, CUMULATIVE_PRODUCT, REVERSE_SUBSEQUENCES, ACTIVATION_RELU_GRAD, RANDOM_GENERATOR, NONZERO_COORDINATES, ADAM_OPTIMIZER, DYNAMIC_QUANTIZE_LINEAR, ELEMENT_WISE_QUANTIZED_LINEAR_ADD
- Substantial performance improvements for several operators.
- Bug fixes.

# DirectML 1.5.1

- Adds a workaround for a driver issue that affects some Intel devices. For the best performance it is recommended to use the latest drivers.

# DirectML 1.5.0

- Introduces a new feature level: [DML_FEATURE_LEVEL_3_1](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_3_1)
  - Adds 6 new operators:
    - DML_OPERATOR_ELEMENT_WISE_ATAN_YX,
    - DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD,
    - DML_OPERATOR_ELEMENT_WISE_DIFFERENCE_SQUARE,
    - DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION_GRAD,
    - DML_OPERATOR_CUMULATIVE_PRODUCT,
    - DML_OPERATOR_BATCH_NORMALIZATION_GRAD,
  - Supports 8D tensors for operators: ELEMENT_WISE_CLIP_GRAD, ELEMENT_WISE_DIFFERENCE_SQUARE, ELEMENT_WISE_ATAN_YX, CAST, JOIN, PADDING, TILE, TOP_K, BATCH_NORMALIZATION, BATCH_NORMALIZATION_GRAD, LP_NORMALIZATION, TOP_K1, MEAN_VARIANCE_NORMALIZATION1, SLICE_GRAD
- Initial support ARM/ARM64 builds of DirectML.
- Substantial performance improvements for several operators.
- Bug fixes.

# DirectML 1.4.3

- Fix perf issue for NHWC layouts of fused activation with Convolution/GEMM/Normalization.

# DirectML 1.4.2

- Add PIX markers support to redist to enable profiling graph at operator level.

# DirectML 1.4.1

- Bug fixes related to metacomands:
  - Fix DML_OPERATOR_BATCH_NORMALIZATION crash when the operator is created with DimensionCount > 5.
  - Fix DML_OPERATOR_MAX_POOLING1/2 binding order for optional output indices tensor. This did not affect the output, but when running with GPU validation enabled, an error would happen "Supplied parameters size doesn't match enumerated size".

# DirectML 1.4.0

- First release of DirectML as a redistributable NuGet package, [Microsoft.AI.DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML).
- Introduces two new feature levels since DirectML 1.1.0: [DML_FEATURE_LEVEL_3_0](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_3_0) and [DML_FEATURE_LEVEL_2_1](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_2_1).
  - Adds 44 new operators.
  - The maximum number of tensor dimensions has been increased from 5 to 8 for operators: ELEMENT_WISE_IDENTITY, ELEMENT_WISE_ABS, ELEMENT_WISE_ACOS, ELEMENT_WISE_ADD, ELEMENT_WISE_ASIN, ELEMENT_WISE_ATAN, ELEMENT_WISE_CEIL, ELEMENT_WISE_CLIP, ELEMENT_WISE_COS, ELEMENT_WISE_DIVIDE, ELEMENT_WISE_EXP, ELEMENT_WISE_FLOOR, ELEMENT_WISE_LOG, ELEMENT_WISE_LOGICAL_AND, ELEMENT_WISE_LOGICAL_EQUALS, ELEMENT_WISE_LOGICAL_GREATER_THAN, ELEMENT_WISE_LOGICAL_LESS_THAN, ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL, ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL, ELEMENT_WISE_LOGICAL_NOT, ELEMENT_WISE_LOGICAL_OR, ELEMENT_WISE_LOGICAL_XOR, ELEMENT_WISE_MAX, ELEMENT_WISE_MEAN, ELEMENT_WISE_MIN, ELEMENT_WISE_MULTIPLY, ELEMENT_WISE_POW, ELEMENT_WISE_CONSTANT_POW, ELEMENT_WISE_RECIP, ELEMENT_WISE_SIN, ELEMENT_WISE_SQRT, ELEMENT_WISE_SUBTRACT, ELEMENT_WISE_TAN, ELEMENT_WISE_THRESHOLD, ELEMENT_WISE_QUANTIZE_LINEAR, ELEMENT_WISE_DEQUANTIZE_LINEAR, ARGMIN, ARGMAX, SLICE, SPLIT, GATHER, ELEMENT_WISE_SIGN, ELEMENT_WISE_IS_NAN, ELEMENT_WISE_ERF, ELEMENT_WISE_SINH, ELEMENT_WISE_COSH, ELEMENT_WISE_TANH, ELEMENT_WISE_ASINH, ELEMENT_WISE_ACOSH, ELEMENT_WISE_ATANH, ELEMENT_WISE_IF, ELEMENT_WISE_ADD1, SCATTER, ONE_HOT, ELEMENT_WISE_BIT_SHIFT_LEFT, ELEMENT_WISE_BIT_SHIFT_RIGHT, ELEMENT_WISE_ROUND, ELEMENT_WISE_IS_INFINITY, ELEMENT_WISE_MODULUS_TRUNCATE, ELEMENT_WISE_MODULUS_FLOOR, GATHER_ELEMENTS, GATHER_ND, SCATTER_ND, SLICE1, ELEMENT_WISE_BIT_AND, ELEMENT_WISE_BIT_OR, ELEMENT_WISE_BIT_XOR, ELEMENT_WISE_BIT_NOT, ELEMENT_WISE_BIT_COUNT, GATHER_ND1
  - Select operators support additional tensor data types.
- Substantial performance improvements for several operators.
- Bug fixes.

# DirectML 1.1.0

- Introduces a new feature level: [DML_FEATURE_LEVEL_2_0](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_2_0).
  - Adds 19 new operators.
  - When binding an input resource for dispatch of an IDMLOperatorInitializer, it is now legal to provide a resource with D3D12_HEAP_TYPE_CUSTOM (in addition to D3D12_HEAP_TYPE_DEFAULT), as long as appropriate heap properties are also set.
  - Select operators support 8-bit integer tensors.
  - 5D activation functions now support the use of strides on their input and output tensors.
- Substantial performance improvements for several operators.
- Bug fixes.

# DirectML 1.0.0

- First release of DirectML
