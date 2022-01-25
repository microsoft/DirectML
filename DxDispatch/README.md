# DxDispatch

DxDispatch is simple command-line executable for launching DirectX 12 compute programs without writing all the C++ boilerplate. The input to the tool is a JSON model that defines resources, dispatchables (compute shaders or DirectML operators), and commands to execute. The model abstraction makes it easy to experiment, but it also preserves low-level control and flexibility.

Some of the things you can do with this tool:
- Launch [DirectML](https://github.com/Microsoft/directml) operators to understand how they work with different inputs.
- Run custom HLSL compute shaders that are compiled at runtime using the [DirectX Shader Compiler](https://github.com/Microsoft/DirectXShaderCompiler).
- Debug binding and API usage issues. DxDispatch hooks into the Direct3D and DirectML debug layers and prints errors and warnings directly to the console; no need to attach a debugger.
- Experiment with performance by benchmarking dispatches.
- Take GPU or timing captures using [PIX on Windows](https://devblogs.microsoft.com/pix/download/). Labeled events and resources make it easy to correlate model objects to D3D objects.

This tool is *not* designed to be a general-purpose framework for building large computational models or running in production scenarios. The focus is on experimentation and learning!

# DirectX Component Versions

DxDispatch currently targets the following versions of DirectX components:

| Component               | Version                                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| DirectML                | [1.8.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.8.0)                                    |
| Direct3D 12             | [1.4.10 (Agility SDK)](https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.4.10)                 |
| DirectX Shader Compiler | [December 2021 (v1.6.2106)](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.6.2112) |

# System Requirements

DxDispatch is currently only available for Windows 10/11, but support for WSL may be added in the future. DxDispatch depends on the [DirectX 12 Agility SDK](https://devblogs.microsoft.com/directx/announcing-dx12agility/), which is only available on Version 1909+.

- A DirectX 12 capable hardware device.
- Windows 10 November 2019 Update (Version 1909; Build 18363) or newer.
- If testing shaders that use Shader Model 6.6:
  - AMD: [Adrenalin 21.4.1 preview driver](https://www.amd.com/en/support/kb/release-notes/rn-rad-win-21-4-1-dx12-agility-sdk)
  - NVIDIA: drivers with version 466.11 or higher

Each Windows release of DxDispatch bundles a recent redistributable copy of DirectML (plus debug layers), Direct3D 12 (plus debug layers), and the DX Shader Compiler. You do not need to install any additional software to run the dxdispatch executable, but a recent driver for your DX12 device is recommended.

# Getting Started

See the [guide](doc/Guide.md) for detailed usage instructions. The [models](./models) directory contains some simple examples to get started. For example, here's an example that invokes DML's reduction operator:

```
> dxdispatch.exe models/dml_reduce.json

Running on 'NVIDIA GeForce RTX 2070 SUPER'
Resource 'input': 1, 2, 3, 4, 5, 6, 7, 8, 9
Resource 'output': 6, 15, 24
```

# Building, Testing, and Installing

DxDispatch relies on several external dependencies that are downloaded when the project is configured. See [ThirdPartyNotices.txt](./ThirdPartyNotices.txt) for relevant license info.

This project uses CMake so you may generate a build system of your choosing. However, some of the CMake scripts to fetch external dependencies currently assume a Visual Studio generator. Until this is resolved you'll want to stick with VS. Tested with the following configuration:
- Microsoft Visual Studio 2019/2022
- Windows SDK 10.0.19041.0 or newer

Example from a terminal in the clone directory (change install location as desired):

**Generate, Build, and Install:**
```
cmake . -B build -DCMAKE_INSTALL_PREFIX=c:/dxdispatch
cmake --build build --target INSTALL
```

**Test**:
```
cd build
ctest .
```