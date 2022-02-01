# Potential Improvements

**Resources**
- Ability to configure heaps (currently everything is a default heap)
- Support textures
- More buffer initializers: file binary data, random data

**Dispatchables**
- HLSL resource arrays, including unbounded
- Support fxc compiler
- Improved shader PDB output path

**Binding**
- Support flags (i.e. D3D12_DESCRIPTOR_RANGE_FLAGS)
- Option to bind using root constants/descriptors (perhaps a bool) instead of descriptor tables

**Other**
- Better printing: reinterpret shape (default 1D) and data type
- Documentation
- Python helper layer to build JSON and HLSL in a single file and configures thread groups, etc. using code.