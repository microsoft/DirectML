//--------------------------------------------------------------------------------------
// File: Effects.h
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#pragma once

#if defined(_XBOX_ONE) && defined(_TITLE)
#include <d3d12_x.h>
#else
#include <d3d12.h>
#endif

#include <DirectXMath.h>
#include <memory>
#include <string>

#include "RenderTargetState.h"
#include "EffectPipelineStateDescription.h"


namespace DirectX
{
    class DescriptorHeap;
    class ResourceUploadBatch;

    //----------------------------------------------------------------------------------
    // Abstract interface representing any effect which can be applied onto a D3D device context.
    class IEffect
    {
    public:
        virtual ~IEffect() = default;

        IEffect(const IEffect&) = delete;
        IEffect& operator=(const IEffect&) = delete;

        IEffect(IEffect&&) = delete;
        IEffect& operator=(IEffect&&) = delete;

        virtual void __cdecl Apply(_In_ ID3D12GraphicsCommandList* commandList) = 0;

    protected:
        IEffect() = default;
    };


    // Abstract interface for effects with world, view, and projection matrices.
    class IEffectMatrices
    {
    public:
        virtual ~IEffectMatrices() = default;

        IEffectMatrices(const IEffectMatrices&) = delete;
        IEffectMatrices& operator=(const IEffectMatrices&) = delete;

        IEffectMatrices(IEffectMatrices&&) = delete;
        IEffectMatrices& operator=(IEffectMatrices&&) = delete;

        virtual void XM_CALLCONV SetWorld(FXMMATRIX value) = 0;
        virtual void XM_CALLCONV SetView(FXMMATRIX value) = 0;
        virtual void XM_CALLCONV SetProjection(FXMMATRIX value) = 0;
        virtual void XM_CALLCONV SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection);

    protected:
        IEffectMatrices() = default;
    };


    // Abstract interface for effects which support directional lighting.
    class IEffectLights
    {
    public:
        virtual ~IEffectLights() = default;

        IEffectLights(const IEffectLights&) = delete;
        IEffectLights& operator=(const IEffectLights&) = delete;

        IEffectLights(IEffectLights&&) = delete;
        IEffectLights& operator=(IEffectLights&&) = delete;

        virtual void XM_CALLCONV SetAmbientLightColor(FXMVECTOR value) = 0;

        virtual void __cdecl SetLightEnabled(int whichLight, bool value) = 0;
        virtual void XM_CALLCONV SetLightDirection(int whichLight, FXMVECTOR value) = 0;
        virtual void XM_CALLCONV SetLightDiffuseColor(int whichLight, FXMVECTOR value) = 0;
        virtual void XM_CALLCONV SetLightSpecularColor(int whichLight, FXMVECTOR value) = 0;

        virtual void __cdecl EnableDefaultLighting() = 0;

        static const int MaxDirectionalLights = 3;

    protected:
        IEffectLights() = default;
    };


    // Abstract interface for effects which support fog.
    class IEffectFog
    {
    public:
        virtual ~IEffectFog() = default;

        IEffectFog(const IEffectFog&) = delete;
        IEffectFog& operator=(const IEffectFog&) = delete;

        IEffectFog(IEffectFog&&) = delete;
        IEffectFog& operator=(IEffectFog&&) = delete;

        virtual void __cdecl SetFogStart(float value) = 0;
        virtual void __cdecl SetFogEnd(float value) = 0;
        virtual void XM_CALLCONV SetFogColor(FXMVECTOR value) = 0;

    protected:
        IEffectFog() = default;
    };


    // Abstract interface for effects which support skinning
    class IEffectSkinning
    {
    public:
        virtual ~IEffectSkinning() = default;

        IEffectSkinning(const IEffectSkinning&) = delete;
        IEffectSkinning& operator=(const IEffectSkinning&) = delete;

        IEffectSkinning(IEffectSkinning&&) = delete;
        IEffectSkinning& operator=(IEffectSkinning&&) = delete;

        virtual void __cdecl SetBoneTransforms(_In_reads_(count) XMMATRIX const* value, size_t count) = 0;
        virtual void __cdecl ResetBoneTransforms() = 0;

        static const int MaxBones = 72;

    protected:
        IEffectSkinning() = default;
    };


    //----------------------------------------------------------------------------------
    namespace EffectFlags
    {
        const int None                = 0x00;
        const int Fog                 = 0x01;
        const int Lighting            = 0x02;
        const int PerPixelLighting    = 0x04 | Lighting; // per pixel lighting implies lighting enabled
        const int VertexColor         = 0x08;
        const int Texture             = 0x10;

        const int BiasedVertexNormals = 0x10000; // compressed vertex normals need x2 bias
    }


    //----------------------------------------------------------------------------------
    // Built-in shader supports optional texture mapping, vertex coloring, directional lighting, and fog.
    class BasicEffect : public IEffect, public IEffectMatrices, public IEffectLights, public IEffectFog
    {
    public:
        BasicEffect(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription);
        BasicEffect(BasicEffect&& moveFrom) noexcept;
        BasicEffect& operator= (BasicEffect&& moveFrom) noexcept;

        BasicEffect(BasicEffect const&) = delete;
        BasicEffect& operator= (BasicEffect const&) = delete;

        virtual ~BasicEffect() override;

        // IEffect methods.
        void __cdecl Apply(_In_ ID3D12GraphicsCommandList* commandList) override;

        // Camera settings.
        void XM_CALLCONV SetWorld(FXMMATRIX value) override;
        void XM_CALLCONV SetView(FXMMATRIX value) override;
        void XM_CALLCONV SetProjection(FXMMATRIX value) override;
        void XM_CALLCONV SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection) override;

        // Material settings.
        void XM_CALLCONV SetDiffuseColor(FXMVECTOR value);
        void XM_CALLCONV SetEmissiveColor(FXMVECTOR value);
        void XM_CALLCONV SetSpecularColor(FXMVECTOR value);
        void __cdecl SetSpecularPower(float value);
        void __cdecl DisableSpecular();
        void __cdecl SetAlpha(float value);
        void XM_CALLCONV SetColorAndAlpha(FXMVECTOR value);

        // Light settings.
        void XM_CALLCONV SetAmbientLightColor(FXMVECTOR value) override;

        void __cdecl SetLightEnabled(int whichLight, bool value) override;
        void XM_CALLCONV SetLightDirection(int whichLight, FXMVECTOR value) override;
        void XM_CALLCONV SetLightDiffuseColor(int whichLight, FXMVECTOR value) override;
        void XM_CALLCONV SetLightSpecularColor(int whichLight, FXMVECTOR value) override;

        void __cdecl EnableDefaultLighting() override;

        // Fog settings.
        void __cdecl SetFogStart(float value) override;
        void __cdecl SetFogEnd(float value) override;
        void XM_CALLCONV SetFogColor(FXMVECTOR value) override;

        // Texture setting.
        void __cdecl SetTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor, D3D12_GPU_DESCRIPTOR_HANDLE samplerDescriptor);

    private:
        // Private implementation.
        class Impl;

        std::unique_ptr<Impl> pImpl;
    };


    // Built-in shader supports per-pixel alpha testing.
    class AlphaTestEffect : public IEffect, public IEffectMatrices, public IEffectFog
    {
    public:
        AlphaTestEffect(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription,
                        D3D12_COMPARISON_FUNC alphaFunction = D3D12_COMPARISON_FUNC_GREATER);
        AlphaTestEffect(AlphaTestEffect&& moveFrom) noexcept;
        AlphaTestEffect& operator= (AlphaTestEffect&& moveFrom) noexcept;

        AlphaTestEffect(AlphaTestEffect const&) = delete;
        AlphaTestEffect& operator= (AlphaTestEffect const&) = delete;

        virtual ~AlphaTestEffect() override;

        // IEffect methods.
        void __cdecl Apply(_In_ ID3D12GraphicsCommandList* commandList) override;

        // Camera settings.
        void XM_CALLCONV SetWorld(FXMMATRIX value) override;
        void XM_CALLCONV SetView(FXMMATRIX value) override;
        void XM_CALLCONV SetProjection(FXMMATRIX value) override;
        void XM_CALLCONV SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection) override;

        // Material settings.
        void XM_CALLCONV SetDiffuseColor(FXMVECTOR value);
        void __cdecl SetAlpha(float value);
        void XM_CALLCONV SetColorAndAlpha(FXMVECTOR value);
        
        // Fog settings.
        void __cdecl SetFogStart(float value) override;
        void __cdecl SetFogEnd(float value) override;
        void XM_CALLCONV SetFogColor(FXMVECTOR value) override;

        // Texture setting.
        void __cdecl SetTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor, D3D12_GPU_DESCRIPTOR_HANDLE samplerDescriptor);

        // Alpha test settings.
        void __cdecl SetReferenceAlpha(int value);

    private:
        // Private implementation.
        class Impl;

        std::unique_ptr<Impl> pImpl;
    };


    // Built-in shader supports two layer multitexturing (eg. for lightmaps or detail textures).
    class DualTextureEffect : public IEffect, public IEffectMatrices, public IEffectFog
    {
    public:
        DualTextureEffect(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription);
        DualTextureEffect(DualTextureEffect&& moveFrom) noexcept;
        DualTextureEffect& operator= (DualTextureEffect&& moveFrom) noexcept;

        DualTextureEffect(DualTextureEffect const&) = delete;
        DualTextureEffect& operator= (DualTextureEffect const&) = delete;

        virtual ~DualTextureEffect() override;

        // IEffect methods.
        void __cdecl Apply(_In_ ID3D12GraphicsCommandList* commandList) override;

        // Camera settings.
        void XM_CALLCONV SetWorld(FXMMATRIX value) override;
        void XM_CALLCONV SetView(FXMMATRIX value) override;
        void XM_CALLCONV SetProjection(FXMMATRIX value) override;
        void XM_CALLCONV SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection) override;

        // Material settings.
        void XM_CALLCONV SetDiffuseColor(FXMVECTOR value);
        void __cdecl SetAlpha(float value);
        void XM_CALLCONV SetColorAndAlpha(FXMVECTOR value);
        
        // Fog settings.
        void __cdecl SetFogStart(float value) override;
        void __cdecl SetFogEnd(float value) override;
        void XM_CALLCONV SetFogColor(FXMVECTOR value) override;

        // Texture settings.
        void __cdecl SetTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor, D3D12_GPU_DESCRIPTOR_HANDLE samplerDescriptor);
        void __cdecl SetTexture2(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor, D3D12_GPU_DESCRIPTOR_HANDLE samplerDescriptor);
        
    private:
        // Private implementation.
        class Impl;

        std::unique_ptr<Impl> pImpl;
    };


    // Built-in shader supports cubic environment mapping.
    class EnvironmentMapEffect : public IEffect, public IEffectMatrices, public IEffectLights, public IEffectFog
    {
    public:
        EnvironmentMapEffect(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, bool fresnelEnabled = true, bool specularEnabled = false);
        EnvironmentMapEffect(EnvironmentMapEffect&& moveFrom) noexcept;
        EnvironmentMapEffect& operator= (EnvironmentMapEffect&& moveFrom) noexcept;

        EnvironmentMapEffect(EnvironmentMapEffect const&) = delete;
        EnvironmentMapEffect& operator= (EnvironmentMapEffect const&) = delete;

        virtual ~EnvironmentMapEffect() override;

        // IEffect methods.
        void __cdecl Apply(_In_ ID3D12GraphicsCommandList* commandList) override;

        // Camera settings.
        void XM_CALLCONV SetWorld(FXMMATRIX value) override;
        void XM_CALLCONV SetView(FXMMATRIX value) override;
        void XM_CALLCONV SetProjection(FXMMATRIX value) override;
        void XM_CALLCONV SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection) override;

        // Material settings.
        void XM_CALLCONV SetDiffuseColor(FXMVECTOR value);
        void XM_CALLCONV SetEmissiveColor(FXMVECTOR value);
        void __cdecl SetAlpha(float value);
        void XM_CALLCONV SetColorAndAlpha(FXMVECTOR value);
        
        // Light settings.
        void XM_CALLCONV SetAmbientLightColor(FXMVECTOR value) override;

        void __cdecl SetLightEnabled(int whichLight, bool value) override;
        void XM_CALLCONV SetLightDirection(int whichLight, FXMVECTOR value) override;
        void XM_CALLCONV SetLightDiffuseColor(int whichLight, FXMVECTOR value) override;

        void __cdecl EnableDefaultLighting() override;

        // Fog settings.
        void __cdecl SetFogStart(float value) override;
        void __cdecl SetFogEnd(float value) override;
        void XM_CALLCONV SetFogColor(FXMVECTOR value) override;

        // Texture setting.
        void __cdecl SetTexture(D3D12_GPU_DESCRIPTOR_HANDLE texture, D3D12_GPU_DESCRIPTOR_HANDLE sampler);

        // Environment map settings.
        void __cdecl SetEnvironmentMap(D3D12_GPU_DESCRIPTOR_HANDLE texture, D3D12_GPU_DESCRIPTOR_HANDLE sampler);
        void __cdecl SetEnvironmentMapAmount(float value);
        void XM_CALLCONV SetEnvironmentMapSpecular(FXMVECTOR value);
        void __cdecl SetFresnelFactor(float value);
        
    private:
        // Private implementation.
        class Impl;

        std::unique_ptr<Impl> pImpl;

        // Unsupported interface methods.
        void XM_CALLCONV SetLightSpecularColor(int whichLight, FXMVECTOR value) override;
    };


    // Built-in shader supports skinned animation.
    class SkinnedEffect : public IEffect, public IEffectMatrices, public IEffectLights, public IEffectFog, public IEffectSkinning
    {
    public:
        SkinnedEffect(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, int weightsPerVertex = 4);
        SkinnedEffect(SkinnedEffect&& moveFrom) noexcept;
        SkinnedEffect& operator= (SkinnedEffect&& moveFrom) noexcept;

        SkinnedEffect(SkinnedEffect const&) = delete;
        SkinnedEffect& operator= (SkinnedEffect const&) = delete;

        virtual ~SkinnedEffect() override;

        // IEffect methods.
        void __cdecl Apply(_In_ ID3D12GraphicsCommandList* commandList) override;

        // Camera settings.
        void XM_CALLCONV SetWorld(FXMMATRIX value) override;
        void XM_CALLCONV SetView(FXMMATRIX value) override;
        void XM_CALLCONV SetProjection(FXMMATRIX value) override;
        void XM_CALLCONV SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection) override;

        // Material settings.
        void XM_CALLCONV SetDiffuseColor(FXMVECTOR value);
        void XM_CALLCONV SetEmissiveColor(FXMVECTOR value);
        void XM_CALLCONV SetSpecularColor(FXMVECTOR value);
        void __cdecl SetSpecularPower(float value);
        void __cdecl DisableSpecular();
        void __cdecl SetAlpha(float value);
        void XM_CALLCONV SetColorAndAlpha(FXMVECTOR value);
        
        // Light settings.
        void XM_CALLCONV SetAmbientLightColor(FXMVECTOR value) override;

        void __cdecl SetLightEnabled(int whichLight, bool value) override;
        void XM_CALLCONV SetLightDirection(int whichLight, FXMVECTOR value) override;
        void XM_CALLCONV SetLightDiffuseColor(int whichLight, FXMVECTOR value) override;
        void XM_CALLCONV SetLightSpecularColor(int whichLight, FXMVECTOR value) override;

        void __cdecl EnableDefaultLighting() override;

        // Fog settings.
        void __cdecl SetFogStart(float value) override;
        void __cdecl SetFogEnd(float value) override;
        void XM_CALLCONV SetFogColor(FXMVECTOR value) override;

        // Texture setting.
        void __cdecl SetTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor, D3D12_GPU_DESCRIPTOR_HANDLE samplerDescriptor);
        
        // Animation settings.
        void __cdecl SetBoneTransforms(_In_reads_(count) XMMATRIX const* value, size_t count) override;
        void __cdecl ResetBoneTransforms() override;

    private:
        // Private implementation.
        class Impl;

        std::unique_ptr<Impl> pImpl;
    };


    //----------------------------------------------------------------------------------
    // Built-in shader extends BasicEffect with normal map and optional specular map
    class NormalMapEffect : public IEffect, public IEffectMatrices, public IEffectLights, public IEffectFog
    {
    public:
        NormalMapEffect(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, bool specularMap = true);
        NormalMapEffect(NormalMapEffect&& moveFrom) noexcept;
        NormalMapEffect& operator= (NormalMapEffect&& moveFrom) noexcept;

        NormalMapEffect(NormalMapEffect const&) = delete;
        NormalMapEffect& operator= (NormalMapEffect const&) = delete;

        virtual ~NormalMapEffect() override;

        // IEffect methods.
        void __cdecl Apply(_In_ ID3D12GraphicsCommandList* commandList) override;

        // Camera settings.
        void XM_CALLCONV SetWorld(FXMMATRIX value) override;
        void XM_CALLCONV SetView(FXMMATRIX value) override;
        void XM_CALLCONV SetProjection(FXMMATRIX value) override;
        void XM_CALLCONV SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection) override;

        // Material settings.
        void XM_CALLCONV SetDiffuseColor(FXMVECTOR value);
        void XM_CALLCONV SetEmissiveColor(FXMVECTOR value);
        void XM_CALLCONV SetSpecularColor(FXMVECTOR value);
        void __cdecl SetSpecularPower(float value);
        void __cdecl DisableSpecular();
        void __cdecl SetAlpha(float value);
        void XM_CALLCONV SetColorAndAlpha(FXMVECTOR value);

        // Light settings.
        void XM_CALLCONV SetAmbientLightColor(FXMVECTOR value) override;

        void __cdecl SetLightEnabled(int whichLight, bool value) override;
        void XM_CALLCONV SetLightDirection(int whichLight, FXMVECTOR value) override;
        void XM_CALLCONV SetLightDiffuseColor(int whichLight, FXMVECTOR value) override;
        void XM_CALLCONV SetLightSpecularColor(int whichLight, FXMVECTOR value) override;

        void __cdecl EnableDefaultLighting() override;

        // Fog settings.
        void __cdecl SetFogStart(float value) override;
        void __cdecl SetFogEnd(float value) override;
        void XM_CALLCONV SetFogColor(FXMVECTOR value) override;

        // Texture setting - albedo, normal and specular intensity
        void __cdecl SetTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor, D3D12_GPU_DESCRIPTOR_HANDLE samplerDescriptor);
        void __cdecl SetNormalTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor);
        void __cdecl SetSpecularTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor);

    private:
        // Private implementation.
        class Impl;

        std::unique_ptr<Impl> pImpl;
    };


    //----------------------------------------------------------------------------------
    // Built-in shader for Physically-Based Rendering (Roughness/Metalness) with Image-based lighting
    class PBREffect : public IEffect, public IEffectMatrices, public IEffectLights
    {
    public:
        explicit PBREffect(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, bool emissive = false, bool generateVelocity = false);
        PBREffect(PBREffect&& moveFrom) noexcept;
        PBREffect& operator= (PBREffect&& moveFrom) noexcept;

        PBREffect(PBREffect const&) = delete;
        PBREffect& operator= (PBREffect const&) = delete;

        virtual ~PBREffect() override;

        // IEffect methods.
        void __cdecl Apply(_In_ ID3D12GraphicsCommandList* commandList) override;

        // Camera settings.
        void XM_CALLCONV SetWorld(FXMMATRIX value) override;
        void XM_CALLCONV SetView(FXMMATRIX value) override;
        void XM_CALLCONV SetProjection(FXMMATRIX value) override;
        void XM_CALLCONV SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection) override;

        // Light settings.
        void __cdecl SetLightEnabled(int whichLight, bool value) override;
        void XM_CALLCONV SetLightDirection(int whichLight, FXMVECTOR value) override;
        void XM_CALLCONV SetLightDiffuseColor(int whichLight, FXMVECTOR value) override;

        void __cdecl EnableDefaultLighting() override;

        // PBR Settings.
        void __cdecl SetAlpha(float value);
        void XM_CALLCONV SetConstantAlbedo(FXMVECTOR value);
        void __cdecl SetConstantMetallic(float value);
        void __cdecl SetConstantRoughness(float value);

        // Texture settings.
        void __cdecl SetAlbedoTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor, D3D12_GPU_DESCRIPTOR_HANDLE samplerDescriptor);
        void __cdecl SetNormalTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor);
        void __cdecl SetRMATexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor);

        void __cdecl SetEmissiveTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor);

        void __cdecl SetSurfaceTextures(
            D3D12_GPU_DESCRIPTOR_HANDLE albedo,
            D3D12_GPU_DESCRIPTOR_HANDLE normal,
            D3D12_GPU_DESCRIPTOR_HANDLE roughnessMetallicAmbientOcclusion,
            D3D12_GPU_DESCRIPTOR_HANDLE sampler);

        void __cdecl SetIBLTextures(
            D3D12_GPU_DESCRIPTOR_HANDLE radiance,
            int numRadianceMips,
            D3D12_GPU_DESCRIPTOR_HANDLE irradiance,
            D3D12_GPU_DESCRIPTOR_HANDLE sampler);

        // Render target size, required for velocity buffer output.
        void __cdecl SetRenderTargetSizeInPixels(int width, int height);

    private:
        // Private implementation.
        class Impl;

        std::unique_ptr<Impl> pImpl;

        // Unsupported interface methods.
        void XM_CALLCONV SetAmbientLightColor(FXMVECTOR value) override;
        void XM_CALLCONV SetLightSpecularColor(int whichLight, FXMVECTOR value) override;
    };


    //----------------------------------------------------------------------------------
    // Built-in shader for debug visualization of normals, tangents, etc.
    class DebugEffect : public IEffect, public IEffectMatrices
    {
    public:
        enum Mode
        {
            Mode_Default = 0,   // Hemispherical ambient lighting
            Mode_Normals,       // RGB normals
            Mode_Tangents,      // RGB tangents
            Mode_BiTangents,    // RGB bi-tangents
        };

        explicit DebugEffect(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, Mode debugMode = Mode_Default);
        DebugEffect(DebugEffect&& moveFrom) noexcept;
        DebugEffect& operator= (DebugEffect&& moveFrom) noexcept;

        DebugEffect(DebugEffect const&) = delete;
        DebugEffect& operator= (DebugEffect const&) = delete;

        virtual ~DebugEffect() override;

        // IEffect methods.
        void __cdecl Apply(_In_ ID3D12GraphicsCommandList* commandList) override;

        // Camera settings.
        void XM_CALLCONV SetWorld(FXMMATRIX value) override;
        void XM_CALLCONV SetView(FXMMATRIX value) override;
        void XM_CALLCONV SetProjection(FXMMATRIX value) override;
        void XM_CALLCONV SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection) override;

        // Debug Settings.
        void XM_CALLCONV SetHemisphericalAmbientColor(FXMVECTOR upper, FXMVECTOR lower);
        void __cdecl SetAlpha(float value);

    private:
        // Private implementation.
        class Impl;

        std::unique_ptr<Impl> pImpl;
    };


    //----------------------------------------------------------------------------------
    // Abstract interface to factory texture resources
    class IEffectTextureFactory
    {
    public:
        virtual ~IEffectTextureFactory() = default;

        IEffectTextureFactory(const IEffectTextureFactory&) = delete;
        IEffectTextureFactory& operator=(const IEffectTextureFactory&) = delete;

        IEffectTextureFactory(IEffectTextureFactory&&) = delete;
        IEffectTextureFactory& operator=(IEffectTextureFactory&&) = delete;

        virtual size_t __cdecl CreateTexture(_In_z_ const wchar_t* name, int descriptorIndex) = 0;

    protected:
        IEffectTextureFactory() = default;
    };


    // Factory for sharing texture resources
    class EffectTextureFactory : public IEffectTextureFactory
    {
    public:
        EffectTextureFactory(
            _In_ ID3D12Device* device,
            ResourceUploadBatch& resourceUploadBatch,
            _In_ ID3D12DescriptorHeap* descriptorHeap);

        EffectTextureFactory(
            _In_ ID3D12Device* device,
            ResourceUploadBatch& resourceUploadBatch,
            _In_ size_t numDescriptors,
            _In_ D3D12_DESCRIPTOR_HEAP_FLAGS descriptorHeapFlags);

        EffectTextureFactory(EffectTextureFactory&& moveFrom) noexcept;
        EffectTextureFactory& operator= (EffectTextureFactory&& moveFrom) noexcept;

        EffectTextureFactory(EffectTextureFactory const&) = delete;
        EffectTextureFactory& operator= (EffectTextureFactory const&) = delete;

        virtual ~EffectTextureFactory() override;

        virtual size_t __cdecl CreateTexture(_In_z_ const wchar_t* name, int descriptorIndex) override;

        ID3D12DescriptorHeap* __cdecl Heap() const;

        // Shorthand accessors for the descriptor heap
        D3D12_CPU_DESCRIPTOR_HANDLE __cdecl GetCpuDescriptorHandle(size_t index) const;
        D3D12_GPU_DESCRIPTOR_HANDLE __cdecl GetGpuDescriptorHandle(size_t index) const;

        // How many textures are there in this factory?
        size_t __cdecl ResourceCount() const;

        // Get a resource in a specific slot (note: increases reference count on resource)
        void __cdecl GetResource(size_t slot, _Out_ ID3D12Resource** resource, _Out_opt_ bool* isCubeMap = nullptr);

        // Settings.
        void __cdecl ReleaseCache();

        void __cdecl SetSharing(bool enabled);

        void __cdecl EnableForceSRGB(bool forceSRGB);
        void __cdecl EnableAutoGenMips(bool generateMips);

        void __cdecl SetDirectory(_In_opt_z_ const wchar_t* path);

    private:
        // Private implementation
        class Impl;

        std::unique_ptr<Impl> pImpl;
    };


    //----------------------------------------------------------------------------------
    // Abstract interface to factory for sharing effects
    class IEffectFactory
    {
    public:
        virtual ~IEffectFactory() = default;

        IEffectFactory(const IEffectFactory&) = delete;
        IEffectFactory& operator=(const IEffectFactory&) = delete;

        IEffectFactory(IEffectFactory&&) = delete;
        IEffectFactory& operator=(IEffectFactory&&) = delete;

        struct EffectInfo
        {
            std::wstring        name;
            bool                perVertexColor;
            bool                enableSkinning;
            bool                enableDualTexture;
            bool                enableNormalMaps;
            bool                biasedVertexNormals;
            float               specularPower;
            float               alphaValue;
            XMFLOAT3            ambientColor;
            XMFLOAT3            diffuseColor;
            XMFLOAT3            specularColor;
            XMFLOAT3            emissiveColor;
            int                 diffuseTextureIndex;
            int                 specularTextureIndex;
            int                 normalTextureIndex;
            int                 emissiveTextureIndex;
            int                 samplerIndex;
            int                 samplerIndex2;

            EffectInfo() noexcept
                : perVertexColor(false)
                , enableSkinning(false)
                , enableDualTexture(false)
                , enableNormalMaps(false)
                , biasedVertexNormals(false)
                , specularPower(0)
                , alphaValue(0)
                , ambientColor(0, 0, 0)
                , diffuseColor(0, 0, 0)
                , specularColor(0, 0, 0)
                , emissiveColor(0, 0, 0)
                , diffuseTextureIndex(-1)
                , specularTextureIndex(-1)
                , normalTextureIndex(-1)
                , emissiveTextureIndex(-1)
                , samplerIndex(-1)
                , samplerIndex2(-1)
            {
            }
        };

        virtual std::shared_ptr<IEffect> __cdecl CreateEffect(
            const EffectInfo& info, 
            const EffectPipelineStateDescription& opaquePipelineState,
            const EffectPipelineStateDescription& alphaPipelineState,
            const D3D12_INPUT_LAYOUT_DESC& inputLayout, 
            int textureDescriptorOffset = 0,
            int samplerDescriptorOffset = 0) = 0;

    protected:
        IEffectFactory() = default;
    };


    // Factory for sharing effects
    class EffectFactory : public IEffectFactory
    {
    public:
        EffectFactory(_In_ ID3D12Device* device);
        EffectFactory(
            _In_ ID3D12DescriptorHeap* textureDescriptors,
            _In_ ID3D12DescriptorHeap* samplerDescriptors);

        EffectFactory(EffectFactory&& moveFrom) noexcept;
        EffectFactory& operator= (EffectFactory&& moveFrom) noexcept;

        EffectFactory(EffectFactory const&) = delete;
        EffectFactory& operator= (EffectFactory const&) = delete;

        virtual ~EffectFactory() override;

        // IEffectFactory methods.
        virtual std::shared_ptr<IEffect> __cdecl CreateEffect(
            const EffectInfo& info,
            const EffectPipelineStateDescription& opaquePipelineState,
            const EffectPipelineStateDescription& alphaPipelineState,
            const D3D12_INPUT_LAYOUT_DESC& inputLayout,
            int textureDescriptorOffset = 0,
            int samplerDescriptorOffset = 0) override;

        // Settings.
        void __cdecl ReleaseCache();

        void __cdecl SetSharing(bool enabled);

        void __cdecl EnablePerPixelLighting(bool enabled);

        void __cdecl EnableNormalMapEffect(bool enabled);

        void __cdecl EnableFogging(bool enabled);

    private:
        // Private implementation.
        class Impl;

        std::shared_ptr<Impl> pImpl;
    };


    // Factory for Physically Based Rendering (PBR)
    class PBREffectFactory : public IEffectFactory
    {
    public:
        PBREffectFactory(_In_ ID3D12Device* device);
        PBREffectFactory(
            _In_ ID3D12DescriptorHeap* textureDescriptors,
            _In_ ID3D12DescriptorHeap* samplerDescriptors);

        PBREffectFactory(PBREffectFactory&& moveFrom) noexcept;
        PBREffectFactory& operator= (PBREffectFactory&& moveFrom) noexcept;

        PBREffectFactory(PBREffectFactory const&) = delete;
        PBREffectFactory& operator= (PBREffectFactory const&) = delete;

        virtual ~PBREffectFactory() override;

        // IEffectFactory methods.
        virtual std::shared_ptr<IEffect> __cdecl CreateEffect(
            const EffectInfo& info,
            const EffectPipelineStateDescription& opaquePipelineState,
            const EffectPipelineStateDescription& alphaPipelineState,
            const D3D12_INPUT_LAYOUT_DESC& inputLayout,
            int textureDescriptorOffset = 0,
            int samplerDescriptorOffset = 0) override;

        // Settings.
        void __cdecl ReleaseCache();

        void __cdecl SetSharing(bool enabled);

    private:
        // Private implementation.
        class Impl;

        std::shared_ptr<Impl> pImpl;
    };
}
