# DirectML Release History <!-- omit in toc -->

See [DirectML version history on MSDN](https://docs.microsoft.com/windows/win32/direct3d12/dml-version-history) for more detailed notes.

| Version                | Feature level                                                                                                                      | First available in (Windows 10)                          | Redistributable                                                                           |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [1.4.1](#directml-140) | [DML_FEATURE_LEVEL_3_0](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_3_0) | Not yet released                                         | [Microsoft.AI.DirectML.1.4.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.4.1) |
| [1.4.0](#directml-140) | [DML_FEATURE_LEVEL_3_0](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_3_0) | Not yet released                                         | [Microsoft.AI.DirectML.1.4.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.4.0) |
| [1.1.0](#directml-110) | [DML_FEATURE_LEVEL_2_0](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_2_0) | Version 2004 (Build 10.0.19041; 20H1, "May 2020 Update") | -                                                                                         |
| [1.0.0](#directml-100) | [DML_FEATURE_LEVEL_1_0](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_1_0) | Version 1903 (Build 10.0.18362; 19H1, "May 2019 Update") | -                                                                                         |

# DirectML 1.4.1

- Bug fixes related to metaccomands:
  - Fix BATCH_NORMALIZATION crash when the operator is created with DimensionCount > 5.
  - Fix DML_OPERATOR_MAX_POOLING1/2 binding order for optional output indices tensor. This did not affect the output, but when running GPU validation enabled, an error would happen "Supplied parameters size doesn't match enumerated size".

- First release of DirectML as a redistributable NuGet package, [Microsoft.AI.DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML).
- Introduces two new feature levels since DirectML 1.1.0: [DML_FEATURE_LEVEL_3_0](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_3_0) and [DML_FEATURE_LEVEL_2_1](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_2_1).
  - Adds 44 new operators.
  - The maximum number of tensor dimensions has been increased from 5 to 8.
  - Select operators support additional tensor data types.
- Substantial performance improvements for several operators.

# DirectML 1.4.0

- First release of DirectML as a redistributable NuGet package, [Microsoft.AI.DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML).
- Introduces two new feature levels since DirectML 1.1.0: [DML_FEATURE_LEVEL_3_0](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_3_0) and [DML_FEATURE_LEVEL_2_1](https://docs.microsoft.com/windows/win32/direct3d12/dml-feature-level-history#dml_feature_level_2_1).
  - Adds 44 new operators.
  - The maximum number of tensor dimensions has been increased from 5 to 8.
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