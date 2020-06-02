-----------------------------------------------
DirectXTK - the DirectX Tool Kit for DirectX 12
-----------------------------------------------

Copyright (c) Microsoft Corporation. All rights reserved.

August 21, 2019

This package contains the "DirectX Tool Kit", a collection of helper classes for
writing Direct3D 12 C++ code for Universal Windows Platform (UWP) apps, Win32
desktop applications for Windows 10, and Xbox One.

This code is designed to build with Visual Studio 2015 Update 3, Visual Studio 2017,
or Visual Studio 2019. It is recommended that you make use of VS 2015 Update 3,
Windows Tools 1.4.1, and the Windows 10 Anniversary Update SDK (14393) -or-
VS 2017 (15.9 update) / VS 2019 with the Windows 10 October 2018 Update SDK (17763).

Inc\
    Public Header Files (in the DirectX C++ namespace):

    Audio.h - low-level audio API using XAudio2 (DirectXTK for Audio public header)
    CommonStates.h - common D3D state combinations
    DDSTextureLoader.h - light-weight DDS file texture loader
    DescriptorHeap.h - helper for managing DX12 descriptor heaps
    DirectXHelpers.h - misc C++ helpers for D3D programming
    EffectPipelineStateDescription.h - helper for creating PSOs
    Effects.h - set of built-in shaders for common rendering tasks
    GamePad.h - gamepad controller helper using XInput
    GeometricPrimitive.h - draws basic shapes such as cubes and spheres
    GraphicsMemory.h - helper for managing dynamic graphics memory allocation
    Keyboard.h - keyboard state tracking helper
    Model.h - draws meshes loaded from .SDKMESH or .VBO files
    Mouse.h - mouse helper
    PostProcess.h - set of built-in shaders for common post-processing operations
    PrimitiveBatch.h - simple and efficient way to draw user primitives
    RenderTargetState.h - helper for communicating render target requirements when creating PSOs
    ResourceUploadBatch.h - helper for managing texture resource upload to the GPU
    ScreenGrab.h - light-weight screen shot saver
    SimpleMath.h - simplified C++ wrapper for DirectXMath
    SpriteBatch.h - simple & efficient 2D sprite rendering
    SpriteFont.h - bitmap based text rendering
    VertexTypes.h - structures for commonly used vertex data formats
    WICTextureLoader.h - WIC-based image file texture loader
    XboxDDSTextureLoader.h - Xbox One exclusive apps variant of DDSTextureLoader

Src\
    DirectXTK source files and internal implementation headers

Audio\
    DirectXTK for Audio source files and internal implementation headers

NOTE: MakeSpriteFont and XWBTool can be found in the DirectX Tool Kit for
      DirectX 11 package.

All content and source code for this package are subject to the terms of the
MIT License. <http://opensource.org/licenses/MIT>.

Documentation is available at <https://github.com/Microsoft/DirectXTK12/wiki>.

For the latest version of DirectX Tool Kit, bug reports, etc. please visit the
project site. <http://go.microsoft.com/fwlink/?LinkID=615561>

This project has adopted the Microsoft Open Source Code of Conduct. For more
information see the Code of Conduct FAQ or contact opencode@microsoft.com with
any additional questions or comments.

https://opensource.microsoft.com/codeofconduct/


--------
XBOX ONE
--------

* Developers using the Xbox One XDK need to generate the
  Src\Shaders\Compiled\XboxOne*.inc files to build the library as they are not
  included in the distribution package. They are built by running the script
  in Src\Shaders - "CompileShaders xbox" from the "Xbox One XDK Command Prompt".
  They are XDK version-specific. While they will continue to work if outdated,
  a mismatch will cause runtime compilation overhead that would otherwise be
  avoided.


---------------------------------
COMPARISONS TO DIRECTX 11 VERSION
---------------------------------

* No support for loading .CMO models or DGSL effect shaders (i.e. DGSLEffect)

* VertexTypes does not include VertexPositionNormalTangentColorTexture or
  VertexPositionNormalTangentColorTextureSkinning which were intended for use
  with the DGSL pipeline.

* DirectX Tool Kit for DirectX 11 supports Feature Level 9.x, while DirectX 12
  requires Direct3D Feature Level 11.0. There are no expected DirectX 12 drivers
  for any lower feature level devices.

* The library assumes it is building for Windows 10 (aka _WIN32_WINNT=0x0A00)
  so it makes use of XAudio 2.9 and WIC2 as well as DirectX 12.

* DirectX Tool Kit for Audio, GamePad, Keyboard, Mouse, and SimpleMath are
  identical to the DirectX 11 version.


-------------
RELEASE NOTES
-------------

* The VS 2017/2019 projects make use of /permissive- for improved C++ standard
  conformance. Use of a Windows 10 SDK prior to the Fall Creators Update (16299)
  or an Xbox One XDK prior to June 2017 QFE 4 may result in failures due to
  problems with the system headers. You can work around these by disabling this
  switch in the project files which is found in the <ConformanceMode> elements.

* The VS 2017 projects require the 15.5 update or later. For UWP and Win32
  classic desktop projects with the 15.5 - 15.7 updates, you need to install the
  standalone Windows 10 SDK (17763) which is otherwise included in the 15.8.6 or
  later update. Older VS 2017 updates will fail to load the projects due to use
  of the <ConformanceMode> element. If using the 15.5 or 15.6 updates, you will
  see "warning D9002: ignoring unknown option '/Zc:__cplusplus'" because this
  switch isn't supported until 15.7. It is safe to ignore this warning, or you
  can edit the project files <AdditionalOptions> elements.

* The UWP projects include configurations for the ARM64 platform. These require
  VS 2017 (15.9 update) or VS 2019 to build.


---------------
RELEASE HISTORY
---------------

August 21, 2019
    Updated D3DX12 internal copy to latest version
    Code cleanup

June 30, 2019
    Clang/LLVM warning cleanup
    Renamed DirectXTK_Windows10.vcxproj to _Windows10_2017.vcxproj
    Added VS 2019 UWP project

May 30, 2019
    PBREffect updated with additional set methods
    Additional debugging output for GraphicsMemory in error cases
    Added CMake project files
    Code cleanup

April 26, 2019
    Updated auto-generated mipmaps support to make it more robust
    Added optional LoadStaticBuffers method for GeometricPrimitive
    Added VS 2019 desktop projects
    Fixed guards w.r.t. to windows.h usage in Keyboard/Mouse headers
    Added C++/WinRT SetWindow helper to Keyboard/Mouse
    Update HLSL script to use Shader Model 5.1 instead of 5.0
    Code cleanup

February 7, 2019
    Model now supports loading SDKMESH v2 models
    PBREffectFactory added to support PBR materials
    PBREffect and NormalMapEffect shaders updated to support BC5_UNORM compressed normal maps
    SpriteFont: DrawString overloads for UTF-8 chars in addition to UTF-16LE wide chars
    Fixed bug with GraphicsMemory dtor introduced with mGPU handling
    Made library agonstic to legacy Windows SDK pix.h vs. latest pix3.h from NuGet

November 16, 2018
    VS 2017 updated for Windows 10 October 2018 Update SDK (17763)
    ARM64 platform configurations added to UWP projects
    Minor code review

October 31, 2018
    Model loader for SDKMESH now attempts to use legacy DE3CN compressed normals
    - This is an approximation only and emits a warning in debug builds
    IEffectTextureFactory's CreateTexture interface method now returns the 'slot'
    - This is for use with GetResource method
    Minor code review

October 25, 2018
    Use UTF-8 instead of ANSI for narrow strings
    Updated D3DX12 internal copy to latest version
    Improved debug diagnostics
    Minor code review

September 13, 2018
    Broke DescriptorHeap header dependency on D3DX12.H

August 17, 2018
    Improved validation for 16k textures and other large resources
    Improved debug output for failed texture loads and screengrabs
    Updated for VS 2017 15.8
    Code cleanup

July 3, 2018
    Model LoadStaticBuffers method to use static vs. dynamic VB/IB
    *breaking change* Custom Model loaders and renderers should be updated for changes to ModelMeshPart
    ModelMeshPart DrawInstanced method added
    Code and project cleanup

May 31, 2018
    VS 2017 updated for Windows 10 April 2018 Update SDK (17134)
    Regenerated shaders using Windows 10 April 2018 Update SDK (17134)

May 14, 2018
    EffectPipelineStateDescription updated with GetDesc method
    Updated for VS 2017 15.7 update warnings
    Code and project cleanup

April 23, 2018
    AlignUp, AlignDown template functions in DirectXHelpers.h
    ScopedBarrier added to DirectXHelpers.h
    Mouse support for cursor visibility
    SimpleMath and VertexTypes updated with default copy and move ctors
    SimpleMath updates to use constexpr
    Basic multi-GPU support added
    More debug object naming for PIX
    PostProcess updated with 'big triangle' optimization
    Code and project file cleanup

February 7, 2018
    Mouse fix for cursor behavior when using Remote Desktop for Win32
    Updated for a few more VS 2017 warnings

December 13, 2017
    PBREffect and DebugEffect added
    NormalMapEffect no longer requires or uses explicit vertex tangents
    Updated for VS 2017 15.5 update warnings
    Code cleanup

November 1, 2017
    VS 2017 updated for Windows 10 Fall Creators Update SDK (16299)
    Regenerated shaders using Windows 10 Fall Creators Update SDK (16299)
    Updated D3DX12 internal copy to latest version

September 22, 2017
    Updated for VS 2017 15.3 update /permissive- changes
    ScreenGrab updated to use non-sRGB metadata for PNG
    Mouse use of WM_INPUT updated for Remote Desktop scenarios

July 28, 2017
    Fix for WIC writer when codec target format requires a palette
    Fix for error detection in ResourceUploadBatch::End method
    Code cleanup

June 21, 2017
    Post-processing support
    Added DescriptorPile utility
    SDKMESH loader fix when loading legacy files with all zero materials
    DirectXTK for Audio: Minor fixes for environmental audio
    Optimized root signatures for Effects shaders
    Minor code cleanup

April 24, 2017
    Regenerated shaders using Windows 10 Creators Update SDK (15063)
    Fixed NormalMapEffect shader selection for specular texture usage
    Fixed Direct3D validation layer issues when using Creators Update
    Fixed AudioEngine enumeration when using Single Threaded Apartment (STA)
    Fixed bug with GamePad (Windows.Gaming.Input) when no user bound
    Updated D3DX12 internal copy to latest version

April 7, 2017
    VS 2017 updated for Windows Creators Update SDK (15063)
    XboxDDSTextureLoader updates

February 10, 2017
    SpriteBatch default rasterizer state now matches DirectX 11 version
    DDSTextureLoader now supports loading planar video format textures
    GamePad now supports special value of -1 for 'most recently connected controller'
    WIC format 40bppCMYKAlpha should be converted to RGBA8 rather than RGBA16
    DDS support for L8A8 with bitcount 8 rather than 16
    Updated D3DX12 internal copy to latest version
    Minor code cleanup

December 5, 2016
    Mouse and Keyboard classes updated with IsConnected method
    Windows10 project /ZW switch removed to support use in C++/WinRT projection apps
    VS 2017 RC projects added
    Updated D3DX12 internal copy to latest version
    Minor code cleanup

October 6, 2016
    SDKMESH loader and BasicEffects support for compressed vertex normals with biasing
    *breaking change*
      DDSTextureLoader Ex bool forceSRGB and generateMipsIfMissing parmeters are now a DDS_LOADER flag
      WICTextureLoader Ex bool forceSRGB and generateMips parameters are now a WIC_LOADER flag
    Add vertexCount member to ModelMeshPart
    Minor code cleanup

September 15, 2016
    Rebuild shaders using 1.0 Root Signature for improved compatibility
    Minor code cleanup

September 1, 2016
    EffectPipelineStateDescription is now in it's own header
    Additional debug object naming
    Fixed Tier 1 hardware support issues with BasicEffect and generating mipmaps
    Fixed default graphics memory alignment to resolve rendering problems on some hardware
    Added forceSRGB optional parameter to SpriteFont ctor
    EffectFactory method EnableForceSRGB added
    Removed problematic ABI::Windows::Foundation::Rect interop for SimpleMath
    Updated D3DX12 internal copy for the Windows 10 Anniversary Update SDK (14393)
    Minor code cleanup

August 4, 2016
    GraphicsMemory fix for robustness during cleanup
    Regenerated shaders using Windows 10 Anniversary Update SDK (14393)

August 2, 2016
    Updated for VS 2015 Update 3 and Windows 10 SDK (14393)

August 1, 2016
    Model effects array is now indexed by part rather than by material
    GamePad capabilities information updated for Universal Windows and Xbox One platforms
    Specular falloff lighting computation fix in shaders

July 18, 2016
    *breaking changes* to CommonStates, DescriptorHeap, Effects, Model,
        EffectPipelineStateDescription, and SpriteBatchPipelineStateDescription
    - added texture sampler control to Effects and SpriteBatch
    - fixed Model control of blend and rasterizer state
    - fixed problems with PerPixelLighting control (EffectFactory defaults to per-pixel lighting)
    - fixed control of weights-per-vertex optimization for SkinnedEffect
    - removed unnecesary "one-light" shader permutations
    - fixed bug in AlphaTestEfect implementation
    - improved debug messages for misconfigured effects
    NormalMapEffect for normal-map with optional specular map rendering
    EnvironmentMapEffect now supports per-pixel lighting
    Effects updated with SetMatrices and SetColorAndAlpha methods
    GraphicsMemory support for SharedGraphicsResource shared_ptr style smart-pointer
    PrimitiveBatch fix for DrawQuad
    ScreenGrab handles resource state transition
    SimpleMath: improved interop with DirectXMath constants
    WICTextureLoader module LoadWICTexture* methods
    Fixed bugs with GenerateMips for sRGB and BGRA formats
    Code cleanup

June 30, 2016
    Original release
