//--------------------------------------------------------------------------------------
// File: EffectCommon.h
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#pragma once

#include <memory>

#include "Effects.h"
#include "PlatformHelpers.h"
#include "SharedResourcePool.h"
#include "AlignedNew.h"
#include "DescriptorHeap.h"
#include "GraphicsMemory.h"
#include "DirectXHelpers.h"
#include "RenderTargetState.h"

// BasicEffect, SkinnedEffect, et al, have many things in common, but also significant
// differences (for instance, not all the effects support lighting). This header breaks
// out common functionality into a set of helpers which can be assembled in different
// combinations to build up whatever subset is needed by each effect.


namespace DirectX
{
    // Internal effect flags
    namespace EffectFlags
    {
        const int PerPixelLightingBit = 0x04;
    }

    static_assert(((EffectFlags::PerPixelLighting) & EffectFlags::PerPixelLightingBit) != 0, "PerPixelLighting enum flags mismatch");

    // Bitfield tracks which derived parameter values need to be recomputed.
    namespace EffectDirtyFlags
    {
        const int ConstantBuffer        = 0x01;
        const int WorldViewProj         = 0x02;
        const int WorldInverseTranspose = 0x04;
        const int EyePosition           = 0x08;
        const int MaterialColor         = 0x10;
        const int FogVector             = 0x20;
        const int FogEnable             = 0x40;
        const int AlphaTest             = 0x80;
    }

    // Helper stores matrix parameter values, and computes derived matrices.
    struct EffectMatrices
    {
        EffectMatrices() noexcept;

        XMMATRIX world;
        XMMATRIX view;
        XMMATRIX projection;
        XMMATRIX worldView;

        void SetConstants(_Inout_ int& dirtyFlags, _Inout_ XMMATRIX& worldViewProjConstant);
    };


    // Helper stores the current fog settings, and computes derived shader parameters.
    struct EffectFog
    {
        EffectFog() noexcept;

        bool enabled;
        float start;
        float end;

        void XM_CALLCONV SetConstants(_Inout_ int& dirtyFlags, _In_ FXMMATRIX worldView, _Inout_ XMVECTOR& fogVectorConstant);
    };


    // Helper stores material color settings, and computes derived parameters for shaders that do not support realtime lighting.
    struct EffectColor
    {
        EffectColor() noexcept;

        XMVECTOR diffuseColor;
        float alpha;

        void SetConstants(_Inout_ int& dirtyFlags, _Inout_ XMVECTOR& diffuseColorConstant);
    };


    // Helper stores the current light settings, and computes derived shader parameters.
    struct EffectLights : public EffectColor
    {
        EffectLights() noexcept;

        static const int MaxDirectionalLights = IEffectLights::MaxDirectionalLights;


        // Fields.
        XMVECTOR emissiveColor;
        XMVECTOR ambientLightColor;

        bool lightEnabled[MaxDirectionalLights];
        XMVECTOR lightDiffuseColor[MaxDirectionalLights];
        XMVECTOR lightSpecularColor[MaxDirectionalLights];


        // Methods.
        void InitializeConstants(_Out_ XMVECTOR& specularColorAndPowerConstant, _Out_writes_all_(MaxDirectionalLights) XMVECTOR* lightDirectionConstant, _Out_writes_all_(MaxDirectionalLights) XMVECTOR* lightDiffuseConstant, _Out_writes_all_(MaxDirectionalLights) XMVECTOR* lightSpecularConstant) const;
        void SetConstants(_Inout_ int& dirtyFlags, _In_ EffectMatrices const& matrices, _Inout_ XMMATRIX& worldConstant, _Inout_updates_(3) XMVECTOR worldInverseTransposeConstant[3], _Inout_ XMVECTOR& eyePositionConstant, _Inout_ XMVECTOR& diffuseColorConstant, _Inout_ XMVECTOR& emissiveColorConstant, bool lightingEnabled);

        int SetLightEnabled(int whichLight, bool value, _Inout_updates_(MaxDirectionalLights) XMVECTOR* lightDiffuseConstant, _Inout_updates_(MaxDirectionalLights) XMVECTOR* lightSpecularConstant);
        int XM_CALLCONV SetLightDiffuseColor(int whichLight, FXMVECTOR value, _Inout_updates_(MaxDirectionalLights) XMVECTOR* lightDiffuseConstant);
        int XM_CALLCONV SetLightSpecularColor(int whichLight, FXMVECTOR value, _Inout_updates_(MaxDirectionalLights) XMVECTOR* lightSpecularConstant);

        static void ValidateLightIndex(int whichLight);
        static void EnableDefaultLighting(_In_ IEffectLights* effect);
    };

    // Factory for lazily instantiating shared root signatures.
    class EffectDeviceResources
    {
    public:
        EffectDeviceResources(_In_ ID3D12Device* device) noexcept
            : mDevice(device)
        { }

        ID3D12RootSignature* DemandCreateRootSig(_Inout_ Microsoft::WRL::ComPtr<ID3D12RootSignature>& rootSig, D3D12_ROOT_SIGNATURE_DESC const& desc);

    protected:
        Microsoft::WRL::ComPtr<ID3D12Device> mDevice;

        std::mutex mMutex;
    };

    // Templated base class provides functionality common to all the built-in effects.
    template<typename Traits>
    class EffectBase : public AlignedNew<typename Traits::ConstantBufferType>
    {
    public:
        typename Traits::ConstantBufferType constants;

       // Constructor.
        EffectBase(_In_ ID3D12Device* device)
            : constants{},
            dirtyFlags(INT_MAX),
            mRootSignature(nullptr),
            mDeviceResources(deviceResourcesPool.DemandCreate(device))
        {
            // Initialize the constant buffer data
            mConstantBuffer = GraphicsMemory::Get(device).AllocateConstant(constants);
        }

        // Commits constants to the constant buffer memory
        void UpdateConstants()
        {
            // Make sure the constant buffer is up to date.
            if (dirtyFlags & EffectDirtyFlags::ConstantBuffer)
            {
                mConstantBuffer = GraphicsMemory::Get(mDeviceResources->GetDevice()).AllocateConstant(constants);

                dirtyFlags &= ~EffectDirtyFlags::ConstantBuffer;
            }
        }

        D3D12_GPU_VIRTUAL_ADDRESS GetConstantBufferGpuAddress()
        {
            return mConstantBuffer.GpuAddress();
        }

        ID3D12RootSignature* GetRootSignature(int slot, CD3DX12_ROOT_SIGNATURE_DESC const& rootSig)
        {
            return mDeviceResources->GetRootSignature(slot, rootSig);
        }

        // Fields.
        EffectMatrices matrices;
        EffectFog fog;
        int dirtyFlags;

    protected:
        // Static arrays hold all the precompiled shader permutations.
        static const D3D12_SHADER_BYTECODE VertexShaderBytecode[Traits::VertexShaderCount];
        static const D3D12_SHADER_BYTECODE PixelShaderBytecode[Traits::PixelShaderCount];
        // .. and shader entry points
        static const int VertexShaderIndices[Traits::ShaderPermutationCount];
        static const int PixelShaderIndices[Traits::ShaderPermutationCount];
        // ... and vertex layout tables
        static const D3D12_INPUT_LAYOUT_DESC VertexShaderInputLayouts[Traits::ShaderPermutationCount];

        // Per instance cache of PSOs, populated with variants for each shader & layout
        Microsoft::WRL::ComPtr<ID3D12PipelineState> mPipelineState;

        // Per instance root signature
        ID3D12RootSignature* mRootSignature;

    private:
        // D3D constant buffer holds a copy of the same data as the public 'constants' field.
        GraphicsResource mConstantBuffer;

        // Only one of these helpers is allocated per D3D device, even if there are multiple effect instances.
        class DeviceResources : public EffectDeviceResources
        {
        public:
            DeviceResources(_In_ ID3D12Device* device) noexcept
                : EffectDeviceResources(device),
                mRootSignature{}
            { }

            // Gets or lazily creates the specified root signature
            ID3D12RootSignature* GetRootSignature(int slot, D3D12_ROOT_SIGNATURE_DESC const& desc)
            {
                assert(slot >= 0 && slot < Traits::RootSignatureCount);
                _Analysis_assume_(slot >= 0 && slot < Traits::RootSignatureCount);

                return DemandCreateRootSig(mRootSignature[slot], desc);
            }

            ID3D12Device* GetDevice() const { return mDevice.Get(); }

        private:
            Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature[Traits::RootSignatureCount];
        };

        // Per-device resources.
        std::shared_ptr<DeviceResources> mDeviceResources;

        static SharedResourcePool<ID3D12Device*, DeviceResources> deviceResourcesPool;
    };
}
