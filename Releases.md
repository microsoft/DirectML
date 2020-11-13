# DirectML Release History <!-- omit in toc -->

DirectML is distributed as a system component of Windows 10, and is available as part of the Windows 10 OS in all versions 1903 and newer. Starting with DirectML version 1.4.0, DirectML is also available as a [standalone redistributable package](https://www.nuget.org/packages/Microsoft.AI.DirectML), which is useful for applications that wish to use a fixed version of DirectML or when running on older versions of Windows 10. DirectML follows the [semantic versioning](https://semver.org/) conventions. That is, version numbers follow the form `major.minor.patch`. The first release of DirectML has a version of 1.0.0.

# History

| Version                            | Feature level                  | First available in (Windows 10)                          | Redistributable                                                                           |
| ---------------------------------- | ------------------------------ | -------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [1.4.0](#directml-140)<sup>1</sup> | [DML_FEATURE_LEVEL_3_0](#todo) | Not yet released                                         | [Microsoft.AI.DirectML.1.4.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.4.0) |
| [1.1.0](#directml-110)             | [DML_FEATURE_LEVEL_2_0](#todo) | Version 2004 (Build 10.0.19041; 20H1, "May 2020 Update") | -                                                                                         |
| [1.0.0](#directml-100)             | [DML_FEATURE_LEVEL_1_0](#todo) | Version 1903 (Build 10.0.18362; 19H1, "May 2019 Update") | -                                                                                         |

<sup>1</sup> DirectML versions 1.2.0 and 1.3.0 were not made widely available.

# DirectML 1.4.0

- First release of DirectML as a redistributable NuGet package, [Microsoft.AI.DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML).
- Introduces two new feature levels since DirectML 1.1.0: [DML_FEATURE_LEVEL_3_0](#todo) and [DML_FEATURE_LEVEL_2_1](#todo).
  - Adds 44 new operators
  - The maximum number of tensor dimensions has been increased from 5 to 8.
  - Select operators support additional tensor data types.
- Substantial performance improvements for several operators
- Bug fixes

# DirectML 1.1.0

- Introduces a new feature level: [DML_FEATURE_LEVEL_2_0](#todo)
  - Adds 19 new operators
  - When binding an input resource for dispatch of an IDMLOperatorInitializer, it is now legal to provide a resource with D3D12_HEAP_TYPE_CUSTOM (in addition to D3D12_HEAP_TYPE_DEFAULT), as long as appropriate heap properties are also set. See also: TODO: LINK ME
  - Select operators support 8-bit integer tensors.
  - 5D activation functions now support the use of strides on their input and output tensors.
- Substantial performance improvements for several operators
- Bug fixes

# DirectML 1.0.0

- First release of DirectML