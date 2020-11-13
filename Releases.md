# DirectML Release History

DirectML is distributed as a system component of Windows 10, and is available as part of the Windows 10 OS in all versions 1903 and newer.

Starting with DirectML version 1.4.0, DirectML is also available as a [standalone redistributable package](https://www.nuget.org/packages/Microsoft.AI.DirectML), which is useful for applications that wish to use a fixed version of DirectML or when running on older versions of Windows 10.

DirectML follows the [semantic versioning](https://semver.org/) conventions. That is, version numbers follow the form `major.minor.patch`. The first release of DirectML has a version of 1.0.0.

| Version           | Feature level         | DML_TARGET_VERSION | First available in (Windows 10)                          | Redistributable                                                                           |
| ----------------- | --------------------- | ------------------ | -------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 1.4.0<sup>1</sup> | DML_FEATURE_LEVEL_3_0 | `0x3000`           | Not yet released                                         | [Microsoft.AI.DirectML.1.4.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.4.0) |
| 1.1.0             | DML_FEATURE_LEVEL_2_0 | `0x2000`           | Version 2004 (Build 10.0.19041; 20H1, "May 2020 Update") | -                                                                                         |
| 1.0.0             | DML_FEATURE_LEVEL_1_0 | `0x1000`           | Version 1903 (Build 10.0.18362; 19H1, "May 2019 Update") | -                                                                                         |

<sup>1</sup> DirectML versions 1.2.* and 1.3.* were not made widely available.

More information on the DirectML Version History may be found 