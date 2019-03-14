# HelloDirectML sample

A minimal but complete DirectML sample that demonstrates how to initialize D3D12 and DirectML, create and compile an operator, execute the operator on the GPU, and retrieve the results.

This sample executes the DirectML "identity" operator over a 1x2x3x4 tensor. The identity operator is the simplest DirectML operator and performs the expression `f(x) = x`, which is functionally equivalent to a copy of the tensor contents. The expected output is:

```
input tensor: 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62
output tensor: 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.62
```

When built using the "Debug" configuration, the sample enables the D3D12 and DirectML debug layers, which require the Graphics Tools feature-on-demand (FOD) to be installed. For more information, see [Using the DirectML debug layer](https://docs.microsoft.com/windows/desktop/direct3d12/dml-debug-layer).
