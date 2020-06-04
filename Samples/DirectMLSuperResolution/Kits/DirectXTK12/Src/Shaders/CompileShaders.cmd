@echo off
rem Copyright (c) Microsoft Corporation. All rights reserved.
rem Licensed under the MIT License.

setlocal
set error=0

set FXCOPTS=/nologo /WX /Ges /Zi /Zpc /Qstrip_reflect /Qstrip_debug

if %1.==xbox. goto continuexbox
if %1.==dxil. goto continuedxil
if %1.==. goto continuepc
echo usage: CompileShaders [xbox]
exit /b

:continuexbox
set XBOXPREFIX=XboxOne
set XBOXOPTS=/D__XBOX_DISABLE_SHADER_NAME_EMPLACEMENT
if NOT %2.==noprecompile. goto skipnoprecompile
set XBOXOPTS=%XBOXOPTS% /D__XBOX_DISABLE_PRECOMPILE=1
:skipnoprecompile

set XBOXFXC="%XboxOneXDKLatest%\xdk\FXC\amd64\FXC.exe"
if exist %XBOXFXC% goto continue
set XBOXFXC="%XboxOneXDKLatest%xdk\FXC\amd64\FXC.exe"
if exist %XBOXFXC% goto continue
set XBOXFXC="%XboxOneXDKBuild%xdk\FXC\amd64\FXC.exe"
if exist %XBOXFXC% goto continue
set XBOXFXC="%DurangoXDK%xdk\FXC\amd64\FXC.exe"
if not exist %XBOXFXC% goto needxdk
goto continue

:continuedxil
set PCDXC="%WindowsSdkBinPath%%WindowsSDKVersion%\x86\dxc.exe"
if exist %PCDXC% goto continue
goto needdxil

:continuepc
set PCOPTS=/force_rootsig_ver rootsig_1_0

set PCFXC="%WindowsSdkBinPath%%WindowsSDKVersion%\x86\fxc.exe"
if exist %PCFXC% goto continue
set PCFXC="%WindowsSdkDir%bin\%WindowsSDKVersion%\x86\fxc.exe"
if exist %PCFXC% goto continue

set PCFXC=fxc.exe

:continue
@if not exist Compiled mkdir Compiled
call :CompileShader%1 AlphaTestEffect vs VSAlphaTest
call :CompileShader%1 AlphaTestEffect vs VSAlphaTestNoFog
call :CompileShader%1 AlphaTestEffect vs VSAlphaTestVc
call :CompileShader%1 AlphaTestEffect vs VSAlphaTestVcNoFog

call :CompileShader%1 AlphaTestEffect ps PSAlphaTestLtGt
call :CompileShader%1 AlphaTestEffect ps PSAlphaTestLtGtNoFog
call :CompileShader%1 AlphaTestEffect ps PSAlphaTestEqNe
call :CompileShader%1 AlphaTestEffect ps PSAlphaTestEqNeNoFog

call :CompileShader%1 BasicEffect vs VSBasic
call :CompileShader%1 BasicEffect vs VSBasicNoFog
call :CompileShader%1 BasicEffect vs VSBasicVc
call :CompileShader%1 BasicEffect vs VSBasicVcNoFog
call :CompileShader%1 BasicEffect vs VSBasicTx
call :CompileShader%1 BasicEffect vs VSBasicTxNoFog
call :CompileShader%1 BasicEffect vs VSBasicTxVc
call :CompileShader%1 BasicEffect vs VSBasicTxVcNoFog

call :CompileShader%1 BasicEffect vs VSBasicVertexLighting
call :CompileShader%1 BasicEffect vs VSBasicVertexLightingBn
call :CompileShader%1 BasicEffect vs VSBasicVertexLightingVc
call :CompileShader%1 BasicEffect vs VSBasicVertexLightingVcBn
call :CompileShader%1 BasicEffect vs VSBasicVertexLightingTx
call :CompileShader%1 BasicEffect vs VSBasicVertexLightingTxBn
call :CompileShader%1 BasicEffect vs VSBasicVertexLightingTxVc
call :CompileShader%1 BasicEffect vs VSBasicVertexLightingTxVcBn

call :CompileShader%1 BasicEffect vs VSBasicPixelLighting
call :CompileShader%1 BasicEffect vs VSBasicPixelLightingBn
call :CompileShader%1 BasicEffect vs VSBasicPixelLightingVc
call :CompileShader%1 BasicEffect vs VSBasicPixelLightingVcBn
call :CompileShader%1 BasicEffect vs VSBasicPixelLightingTx
call :CompileShader%1 BasicEffect vs VSBasicPixelLightingTxBn
call :CompileShader%1 BasicEffect vs VSBasicPixelLightingTxVc
call :CompileShader%1 BasicEffect vs VSBasicPixelLightingTxVcBn

call :CompileShader%1 BasicEffect ps PSBasic
call :CompileShader%1 BasicEffect ps PSBasicNoFog
call :CompileShader%1 BasicEffect ps PSBasicTx
call :CompileShader%1 BasicEffect ps PSBasicTxNoFog

call :CompileShader%1 BasicEffect ps PSBasicVertexLighting
call :CompileShader%1 BasicEffect ps PSBasicVertexLightingNoFog
call :CompileShader%1 BasicEffect ps PSBasicVertexLightingTx
call :CompileShader%1 BasicEffect ps PSBasicVertexLightingTxNoFog

call :CompileShader%1 BasicEffect ps PSBasicPixelLighting
call :CompileShader%1 BasicEffect ps PSBasicPixelLightingTx

call :CompileShader%1 DualTextureEffect vs VSDualTexture
call :CompileShader%1 DualTextureEffect vs VSDualTextureNoFog
call :CompileShader%1 DualTextureEffect vs VSDualTextureVc
call :CompileShader%1 DualTextureEffect vs VSDualTextureVcNoFog

call :CompileShader%1 DualTextureEffect ps PSDualTexture
call :CompileShader%1 DualTextureEffect ps PSDualTextureNoFog

call :CompileShader%1 EnvironmentMapEffect vs VSEnvMap
call :CompileShader%1 EnvironmentMapEffect vs VSEnvMapBn
call :CompileShader%1 EnvironmentMapEffect vs VSEnvMapFresnel
call :CompileShader%1 EnvironmentMapEffect vs VSEnvMapFresnelBn
call :CompileShader%1 EnvironmentMapEffect vs VSEnvMapPixelLighting
call :CompileShader%1 EnvironmentMapEffect vs VSEnvMapPixelLightingBn

call :CompileShader%1 EnvironmentMapEffect ps PSEnvMap
call :CompileShader%1 EnvironmentMapEffect ps PSEnvMapNoFog
call :CompileShader%1 EnvironmentMapEffect ps PSEnvMapSpecular
call :CompileShader%1 EnvironmentMapEffect ps PSEnvMapSpecularNoFog
call :CompileShader%1 EnvironmentMapEffect ps PSEnvMapPixelLighting
call :CompileShader%1 EnvironmentMapEffect ps PSEnvMapPixelLightingNoFog
call :CompileShader%1 EnvironmentMapEffect ps PSEnvMapPixelLightingFresnel
call :CompileShader%1 EnvironmentMapEffect ps PSEnvMapPixelLightingFresnelNoFog

call :CompileShader%1 SkinnedEffect vs VSSkinnedVertexLightingOneBone
call :CompileShader%1 SkinnedEffect vs VSSkinnedVertexLightingOneBoneBn
call :CompileShader%1 SkinnedEffect vs VSSkinnedVertexLightingTwoBones
call :CompileShader%1 SkinnedEffect vs VSSkinnedVertexLightingTwoBonesBn
call :CompileShader%1 SkinnedEffect vs VSSkinnedVertexLightingFourBones
call :CompileShader%1 SkinnedEffect vs VSSkinnedVertexLightingFourBonesBn

call :CompileShader%1 SkinnedEffect vs VSSkinnedPixelLightingOneBone
call :CompileShader%1 SkinnedEffect vs VSSkinnedPixelLightingOneBoneBn
call :CompileShader%1 SkinnedEffect vs VSSkinnedPixelLightingTwoBones
call :CompileShader%1 SkinnedEffect vs VSSkinnedPixelLightingTwoBonesBn
call :CompileShader%1 SkinnedEffect vs VSSkinnedPixelLightingFourBones
call :CompileShader%1 SkinnedEffect vs VSSkinnedPixelLightingFourBonesBn

call :CompileShader%1 SkinnedEffect ps PSSkinnedVertexLighting
call :CompileShader%1 SkinnedEffect ps PSSkinnedVertexLightingNoFog
call :CompileShader%1 SkinnedEffect ps PSSkinnedPixelLighting

call :CompileShader%1 NormalMapEffect vs VSNormalPixelLightingTx
call :CompileShader%1 NormalMapEffect vs VSNormalPixelLightingTxBn
call :CompileShader%1 NormalMapEffect vs VSNormalPixelLightingTxVc
call :CompileShader%1 NormalMapEffect vs VSNormalPixelLightingTxVcBn

call :CompileShader%1 NormalMapEffect vs VSNormalPixelLightingTxNoSpec
call :CompileShader%1 NormalMapEffect vs VSNormalPixelLightingTxNoSpecBn
call :CompileShader%1 NormalMapEffect vs VSNormalPixelLightingTxVcNoSpec
call :CompileShader%1 NormalMapEffect vs VSNormalPixelLightingTxVcNoSpecBn

call :CompileShader%1 NormalMapEffect ps PSNormalPixelLightingTx
call :CompileShader%1 NormalMapEffect ps PSNormalPixelLightingTxNoFog
call :CompileShader%1 NormalMapEffect ps PSNormalPixelLightingTxNoSpec
call :CompileShader%1 NormalMapEffect ps PSNormalPixelLightingTxNoFogSpec

call :CompileShader%1 PBREffect vs VSConstant
call :CompileShader%1 PBREffect vs VSConstantVelocity
call :CompileShader%1 PBREffect vs VSConstantBn
call :CompileShader%1 PBREffect vs VSConstantVelocityBn

call :CompileShader%1 PBREffect ps PSConstant
call :CompileShader%1 PBREffect ps PSTextured
call :CompileShader%1 PBREffect ps PSTexturedEmissive
call :CompileShader%1 PBREffect ps PSTexturedVelocity
call :CompileShader%1 PBREffect ps PSTexturedEmissiveVelocity

call :CompileShader%1 DebugEffect vs VSDebug
call :CompileShader%1 DebugEffect vs VSDebugBn
call :CompileShader%1 DebugEffect vs VSDebugVc
call :CompileShader%1 DebugEffect vs VSDebugVcBn

call :CompileShader%1 DebugEffect ps PSHemiAmbient
call :CompileShader%1 DebugEffect ps PSRGBNormals
call :CompileShader%1 DebugEffect ps PSRGBTangents
call :CompileShader%1 DebugEffect ps PSRGBBiTangents

call :CompileShader%1 SpriteEffect vs SpriteVertexShader
call :CompileShader%1 SpriteEffect ps SpritePixelShader

call :CompileShader%1 SpriteEffect vs SpriteVertexShaderHeap
call :CompileShader%1 SpriteEffect ps SpritePixelShaderHeap

call :CompileShader%1 PostProcess vs VSQuad
call :CompileShader%1 PostProcess vs VSQuadNoCB
call :CompileShader%1 PostProcess vs VSQuadDual
call :CompileShader%1 PostProcess ps PSCopy
call :CompileShader%1 PostProcess ps PSMonochrome
call :CompileShader%1 PostProcess ps PSSepia
call :CompileShader%1 PostProcess ps PSDownScale2x2
call :CompileShader%1 PostProcess ps PSDownScale4x4
call :CompileShader%1 PostProcess ps PSGaussianBlur5x5
call :CompileShader%1 PostProcess ps PSBloomExtract
call :CompileShader%1 PostProcess ps PSBloomBlur
call :CompileShader%1 PostProcess ps PSMerge
call :CompileShader%1 PostProcess ps PSBloomCombine

call :CompileComputeShader%1 GenerateMips main

call :CompileShader%1 ToneMap vs VSQuad
call :CompileShader%1 ToneMap ps PSCopy
call :CompileShader%1 ToneMap ps PSSaturate
call :CompileShader%1 ToneMap ps PSReinhard
call :CompileShader%1 ToneMap ps PSACESFilmic
call :CompileShader%1 ToneMap ps PS_SRGB
call :CompileShader%1 ToneMap ps PSSaturate_SRGB
call :CompileShader%1 ToneMap ps PSReinhard_SRGB
call :CompileShader%1 ToneMap ps PSACESFilmic_SRGB
call :CompileShader%1 ToneMap ps PSHDR10

if NOT %1.==xbox. goto skipxboxonly

call :CompileShaderxbox ToneMap ps PSHDR10_Saturate
call :CompileShaderxbox ToneMap ps PSHDR10_Reinhard
call :CompileShaderxbox ToneMap ps PSHDR10_ACESFilmic
call :CompileShaderxbox ToneMap ps PSHDR10_Saturate_SRGB
call :CompileShaderxbox ToneMap ps PSHDR10_Reinhard_SRGB
call :CompileShaderxbox ToneMap ps PSHDR10_ACESFilmic_SRGB

:skipxboxonly

echo.

if %error% == 0 (
    echo Shaders compiled ok
) else (
    echo There were shader compilation errors!
)

endlocal
exit /b

:CompileShader
set fxc=%PCFXC% %1.fx %FXCOPTS% /T%2_5_1 %PCOPTS% /E%3 /FhCompiled\%1_%3.inc /FdCompiled\%1_%3.pdb /Vn%1_%3
echo.
echo %fxc%
%fxc% || set error=1
exit /b

:CompileComputeShader
set fxc=%PCFXC% %1.hlsl %FXCOPTS% /Tcs_5_1 %PCOPTS% /E%2 /FhCompiled\%1_%2.inc /FdCompiled\%1_%2.pdb /Vn%1_%2
echo.
echo %fxc%
%fxc% || set error=1
exit /b

:CompileShaderdxil
set dxc=%PCDXC% %1.fx %FXCOPTS% /T%2_6_0 /E%3 /FhCompiled\%1_%3.inc /FdCompiled\%1_%3.pdb /Vn%1_%3
echo.
echo %dxc%
%dxc% || set error=1
exit /b

:CompileComputeShaderdxil
set dxc=%PCDXC% %1.hlsl %FXCOPTS% /Tcs_6_0 /E%2 /FhCompiled\%1_%2.inc /FdCompiled\%1_%2.pdb /Vn%1_%2
echo.
echo %dxc%
%dxc% || set error=1
exit /b

:CompileShaderxbox
set fxc=%XBOXFXC% %1.fx %FXCOPTS% /T%2_5_1 %XBOXOPTS% /E%3 /FhCompiled\%XBOXPREFIX%%1_%3.inc /FdCompiled\%XBOXPREFIX%%1_%3.pdb /Vn%1_%3
echo.
echo %fxc%
%fxc% || set error=1
exit /b

:CompileComputeShaderxbox
set fxc==%XBOXFXC% %1.hlsl %FXCOPTS% /Tcs_5_1 %XBOXOPTS% /E%2 /FhCompiled\%XBOXPREFIX%%1_%2.inc /FdCompiled\%XBOXPREFIX%%1_%2.pdb /Vn%1_%2
echo.
echo %fxc%
%fxc% || set error=1
exit /b

:needxdk
echo ERROR: CompileShaders xbox requires the Microsoft Xbox One XDK
echo        (try re-running from the XDK Command Prompt)
exit /b

:needdxil
echo ERROR: CompileShaders dxil requires the Microsoft Windows 10 SDK (16299 or later)
exit /b
