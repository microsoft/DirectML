# DxDispatch

DxDispatch is simple command-line executable for launching DirectX 12 compute programs without writing all the C++ boilerplate. The input to the tool is a JSON model that defines resources, dispatchables (compute shaders, DirectML operators, ONNX models), and commands to execute. The model abstraction makes it easy to experiment, but it also preserves low-level control and flexibility.

Some of the things you can do with this tool:
- Launch [DirectML](https://github.com/Microsoft/directml) operators to understand how they work with different inputs.
- Run custom HLSL compute shaders that are compiled at runtime using the [DirectX Shader Compiler](https://github.com/Microsoft/DirectXShaderCompiler).
- Run ONNX models using ONNX Runtime with the DirectML execution provider.
- Debug binding and API usage issues. DxDispatch hooks into the Direct3D and DirectML debug layers and prints errors and warnings directly to the console; no need to attach a debugger.
- Experiment with performance by benchmarking dispatches.
- Take GPU or timing captures using [PIX on Windows](https://devblogs.microsoft.com/pix/download/). Labeled events and resources make it easy to correlate model objects to D3D objects.

This tool is *not* designed to be a general-purpose framework for building large computational models or running in production scenarios. The focus is on experimentation and learning!

# Getting Started

See the [guide](doc/Guide.md) for detailed usage instructions. The [models](./models) directory contains some simple examples to get started. For example, here's an example that invokes DML's reduction operator:

```
> dxdispatch.exe models/dml_reduce.json

Running on 'NVIDIA GeForce RTX 2070 SUPER'
Resource 'input': 1, 2, 3, 4, 5, 6, 7, 8, 9
Resource 'output': 6, 15, 24
```

# System Requirements

The exact system requirements vary depending on how you configure and run DxDispatch. The default builds rely on redistributable versions of DirectX components when possible, which provides the latest features to the widest range of systems. For default builds of DxDispatch you should consider the following as the minimum system requirements across the range of platforms:

- A DirectX 12 capable hardware device.
- Windows 10 November 2019 Update (Version 1909; Build 18363) or newer.
- If testing shaders that use Shader Model 6.6:
  - AMD: [Adrenalin 21.4.1 preview driver](https://www.amd.com/en/support/kb/release-notes/rn-rad-win-21-4-1-dx12-agility-sdk)
  - NVIDIA: drivers with version 466.11 or higher

# Features and DirectX Components

DxDispatch tries to depend on pre-built redistributable versions of its external dependencies. However, the build can be configured to use alternative sources when desired or necessary. Each component can use one of the available (✅) sources in the table below, with the <b><u>default</u></b> selection for each platform listed first. Not all configurations are tested, and some platforms don't include the optional<sup>+</sup> components.

<table>
  <tr>
    <th>Platform</th>
    <th><a href="https://docs.microsoft.com/windows/ai/directml/dml-intro">DirectML</a></th>
    <th><a href="https://docs.microsoft.com/windows/win32/direct3d12/what-is-directx-12-">Direct3D 12</a></th>
    <th><a href="https://github.com/microsoft/DirectXShaderCompiler">DX Compiler</a><sup>+</sup></th>
    <th><a href="https://devblogs.microsoft.com/pix/winpixeventruntime/">PIX Event Runtime</a><sup>+</sup></th>
    <th><a href="https://onnxruntime.ai/">ONNX Runtime</a><sup>+</sup></th>
  </tr>
  <tr>
    <td>Windows - x64</td>
    <td><b>✅ <u>nuget</u></b><br>✅ winsdk<br>✅ local</td>
    <td><b>✅ <u>nuget</u></b><br>✅ winsdk</td>
    <td><b>✅ <u>archive</u></b></td>
    <td><b>✅ <u>nuget</u></b></td>
    <td><b>✅ <u>nuget</u></b></td>
  </tr>
  <tr>
    <td>Windows - x86</td>
    <td><b>✅ <u>nuget</u></b><br>✅ winsdk<br>✅ local</td>
    <td><b>✅ <u>nuget</u></b><br>✅ winsdk</td>
    <td>❌ none</td>
    <td>❌ none</td>
    <td><b>✅ <u>nuget</u></b></td>
  </tr>
  <tr>
    <td>Windows - ARM64</td>
    <td><b>✅ <u>nuget</u></b><br>✅ winsdk<br>✅ local</td>
    <td><b>✅ <u>nuget</u></b><br>✅ winsdk</td>
    <td><b>✅ <u>archive</u></b></td>
    <td><b>✅ <u>nuget</u></b></td>
    <td><b>✅ <u>nuget</u></b></td>
  </tr>
  <tr>
    <td>Linux - x64 (WSL)</td>
    <td><b>✅ <u>nuget</u></b><br>✅ local</td>
    <td><b>✅ <u>wsl</u></b></td>
    <td>❌ none</td>
    <td>❌ none</td>
    <td>❌ none</td>
  </tr>
  <!-- <tr>
    <td>Xbox Scarlett</td>
    <td><b>✅ <u>local</u></b></td>
    <td><b>✅ <u>gdk</u></b></td>
    <td><b>✅ <u>gdk</u></b></td>
    <td><b>✅ <u>gdk</u></b></td>
    <td><b>✅ <u>local</u></b></td>
  </tr> -->
</table>

The default redistributable versions of components (e.g. nuget, archives):
- **DirectML (nuget)**: [Microsoft.AI.DirectML (1.10.1)](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.10.1) - 2023/01/26
- **Direct3D 12 (nuget)**: [Microsoft.Direct3D.D3D12 (1.608.2)](https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.608.2) -  2023/01/03
- **DX Compiler (archive)**: [December 2022 (v1.7.2212)](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.7.2212) - 2022/12/16
- **PIX Event Runtime (nuget)**: [WinPixEventRuntime (1.0.220810001)](https://www.nuget.org/packages/WinPixEventRuntime/1.0.220810001) - 2022/08/10
- **ONNX Runtime (nuget)**: [Microsoft.ML.OnnxRuntime.DirectML (1.14.1)](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/1.14.1) - 2023/02/27

Configuration is done using CMake cache variables. For example, Direct3D can be switched to a system dependency by adding `-DDXD_DIRECT3D_TYPE=winsdk` to the command line when first configuring the project. Use `cmake-gui` or `ccmake` to view the available variables.

# Building, Testing, and Installing

DxDispatch relies on several external dependencies that are downloaded when the project is configured. See [ThirdPartyNotices.txt](./ThirdPartyNotices.txt) for relevant license info.

Configure presets are listed configuration in [CMakePresets.json](CMakePresets.json):
```
> cmake --list-presets
Available configure presets:

  "win-x64"       - Windows x64
  "win-x86"       - Windows x86
  "win-arm64"     - Windows ARM64
  "xbox-scarlett" - Xbox Scarlett
  "linux-x64"     - Linux x64
```

To generate the project, provide one of the above names (e.g. `win-x64`) to cmake:
```
> cmake --preset <configure_preset_name>
```

You can build from the generated VS solution under `build\<configure_preset_name>\dxdispatch.sln`. 

Alternatively, build from the command line by using `--build` option and appending the build configuration to the preset name (e.g. the `win-x64` configure preset has the build presets named `win-x64-release` and `win-x64-debug`).

```
> cmake --build --preset <configure_preset_name>-(release|debug)
```

To run tests, change your working directory to the build folder and execute `ctest` (only supported on some platforms). You need to specify the build configuration (release or debug) since the presets use VS, which is a multi-configuration generator:
```
> cd build\<configure_preset_name>
> ctest -C Release .
```