---
page_type: sample
languages:
- cpp
products:
- windows
- xbox
name: DirectMLNpuInference
urlFragment: DirectMLNpuInference
---

# HelloDirectML sample

A minimal but complete DirectML sample that demonstrates how to perform OnnxRuntime inference via D3D12 and DirectML on a NPU. Select a NPU device, create a OnnxRuntime session, execute the model on the NPU, and retrieve the results.

This sample executes the mobilenet model.

When built using the "Debug" configuration, the sample enables the D3D12 and DirectML debug layers, which require the Graphics Tools feature-on-demand (FOD) to be installed. For more information, see [Using the DirectML debug layer](https://docs.microsoft.com/windows/desktop/direct3d12/dml-debug-layer).
