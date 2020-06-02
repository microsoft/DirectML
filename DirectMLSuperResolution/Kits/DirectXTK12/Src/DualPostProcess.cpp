//--------------------------------------------------------------------------------------
// File: DualPostProcess.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "PostProcess.h"

#include "AlignedNew.h"
#include "CommonStates.h"
#include "DemandCreate.h"
#include "DirectXHelpers.h"
#include "EffectPipelineStateDescription.h"
#include "GraphicsMemory.h"
#include "SharedResourcePool.h"

using namespace DirectX;

using Microsoft::WRL::ComPtr;

namespace
{
    const int c_MaxSamples = 16;

    const int Dirty_ConstantBuffer  = 0x01;
    const int Dirty_Parameters      = 0x02;

    // Constant buffer layout. Must match the shader!
    __declspec(align(16)) struct PostProcessConstants
    {
        XMVECTOR sampleOffsets[c_MaxSamples];
        XMVECTOR sampleWeights[c_MaxSamples];
    };

    static_assert((sizeof(PostProcessConstants) % 16) == 0, "CB size not padded correctly");
}

// Include the precompiled shader code.
namespace
{
#if defined(_XBOX_ONE) && defined(_TITLE)
    #include "Shaders/Compiled/XboxOnePostProcess_VSQuadDual.inc"

    #include "Shaders/Compiled/XboxOnePostProcess_PSMerge.inc"
    #include "Shaders/Compiled/XboxOnePostProcess_PSBloomCombine.inc"
#else
    #include "Shaders/Compiled/PostProcess_VSQuadDual.inc"

    #include "Shaders/Compiled/PostProcess_PSMerge.inc"
    #include "Shaders/Compiled/PostProcess_PSBloomCombine.inc"
#endif
}

namespace
{
    const D3D12_SHADER_BYTECODE vertexShader =
        { PostProcess_VSQuadDual,       sizeof(PostProcess_VSQuadDual) };

    const D3D12_SHADER_BYTECODE pixelShaders[] =
    {
        { PostProcess_PSMerge,          sizeof(PostProcess_PSMerge) },
        { PostProcess_PSBloomCombine,   sizeof(PostProcess_PSBloomCombine) },
    };

    static_assert(_countof(pixelShaders) == DualPostProcess::Effect_Max, "array/max mismatch");

    // Factory for lazily instantiating shared root signatures.
    class DeviceResources
    {
    public:
        DeviceResources(_In_ ID3D12Device* device) noexcept
            : mDevice(device)
        { }

        ID3D12RootSignature* GetRootSignature(const D3D12_ROOT_SIGNATURE_DESC& desc)
        {
            return DemandCreate(mRootSignature, mMutex, [&](ID3D12RootSignature** pResult) -> HRESULT
            {
                HRESULT hr = CreateRootSignature(mDevice.Get(), &desc, pResult);

                if (SUCCEEDED(hr))
                    SetDebugObjectName(*pResult, L"DualPostProcess");

                return hr;
            });
        }

        ID3D12Device* GetDevice() const { return mDevice.Get(); }

    protected:
        ComPtr<ID3D12Device>                        mDevice;
        Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature;
        std::mutex                                  mMutex;
    };
}

class DualPostProcess::Impl : public AlignedNew<PostProcessConstants>
{
public:
    Impl(_In_ ID3D12Device* device, const RenderTargetState& rtState, Effect ifx);

    void Process(_In_ ID3D12GraphicsCommandList* commandList);

    void SetDirtyFlag() { mDirtyFlags = INT_MAX; }

    enum RootParameterIndex
    {
        TextureSRV,
        TextureSRV2,
        ConstantBuffer,
        RootParameterCount
    };

    // Fields.
    DualPostProcess::Effect                 fx;
    PostProcessConstants                    constants;
    D3D12_GPU_DESCRIPTOR_HANDLE             texture;
    D3D12_GPU_DESCRIPTOR_HANDLE             texture2;
    float                                   mergeWeight1;
    float                                   mergeWeight2;
    float                                   bloomIntensity;
    float                                   bloomBaseIntensity;
    float                                   bloomSaturation;
    float                                   bloomBaseSaturation;

private:
    int                                     mDirtyFlags;

   // D3D constant buffer holds a copy of the same data as the public 'constants' field.
    GraphicsResource mConstantBuffer;

    // Per instance cache of PSOs, populated with variants for each shader & layout
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPipelineState;

    // Per instance root signature
    ID3D12RootSignature* mRootSignature;

    // Per-device resources.
    std::shared_ptr<DeviceResources> mDeviceResources;

    static SharedResourcePool<ID3D12Device*, DeviceResources> deviceResourcesPool;
};


// Global pool of per-device DualPostProcess resources.
SharedResourcePool<ID3D12Device*, DeviceResources> DualPostProcess::Impl::deviceResourcesPool;


// Constructor.
DualPostProcess::Impl::Impl(_In_ ID3D12Device* device, const RenderTargetState& rtState, Effect ifx)
    : fx(ifx),
    constants{},
    texture{},
    texture2{},
    mergeWeight1(0.5f),
    mergeWeight2(0.5f),
    bloomIntensity(1.25f),
    bloomBaseIntensity(1.f),
    bloomSaturation(1.f),
    bloomBaseSaturation(1.f),
    mDirtyFlags(INT_MAX),
    mDeviceResources(deviceResourcesPool.DemandCreate(device))
{
    if (ifx >= Effect_Max)
        throw std::out_of_range("Effect not defined");
   
    // Create root signature.
    {
        D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
            D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;

        // Same as CommonStates::StaticLinearClamp
        CD3DX12_STATIC_SAMPLER_DESC sampler(
            0, // register
            D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            0.f,
            16,
            D3D12_COMPARISON_FUNC_LESS_EQUAL,
            D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
            0.f,
            D3D12_FLOAT32_MAX,
            D3D12_SHADER_VISIBILITY_PIXEL);

        CD3DX12_ROOT_PARAMETER rootParameters[RootParameterIndex::RootParameterCount] = {};

        CD3DX12_DESCRIPTOR_RANGE texture1Range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        rootParameters[RootParameterIndex::TextureSRV].InitAsDescriptorTable(1, &texture1Range, D3D12_SHADER_VISIBILITY_PIXEL);

        CD3DX12_DESCRIPTOR_RANGE texture2Range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);
        rootParameters[RootParameterIndex::TextureSRV2].InitAsDescriptorTable(1, &texture2Range, D3D12_SHADER_VISIBILITY_PIXEL);

        // Root parameter descriptor
        CD3DX12_ROOT_SIGNATURE_DESC rsigDesc = {};

        // Constant buffer
        rootParameters[RootParameterIndex::ConstantBuffer].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL);

        rsigDesc.Init(_countof(rootParameters), rootParameters, 1, &sampler, rootSignatureFlags);

        mRootSignature = mDeviceResources->GetRootSignature(rsigDesc);
    }

    assert(mRootSignature != nullptr);

    // Create pipeline state.
    EffectPipelineStateDescription psd(nullptr,
        CommonStates::Opaque,
        CommonStates::DepthNone,
        CommonStates::CullNone,
        rtState,
        D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);

    psd.CreatePipelineState(
        device,
        mRootSignature,
        vertexShader,
        pixelShaders[ifx],
        mPipelineState.GetAddressOf());

    SetDebugObjectName(mPipelineState.Get(), L"DualPostProcess");
}


// Sets our state onto the D3D device.
void DualPostProcess::Impl::Process(_In_ ID3D12GraphicsCommandList* commandList)
{
    // Set the root signature.
    commandList->SetGraphicsRootSignature(mRootSignature);

    // Set the texture.
    if (!texture.ptr || !texture2.ptr)
    {
        DebugTrace("ERROR: Missing texture(s) for DualPostProcess (%llu, %llu)\n", texture.ptr, texture2.ptr);
        throw std::exception("DualPostProcess");
    }
    commandList->SetGraphicsRootDescriptorTable(RootParameterIndex::TextureSRV, texture);
    commandList->SetGraphicsRootDescriptorTable(RootParameterIndex::TextureSRV2, texture2);

    // Set constants.
    if (mDirtyFlags & Dirty_Parameters)
    {
        mDirtyFlags &= ~Dirty_Parameters;
        mDirtyFlags |= Dirty_ConstantBuffer;

        switch (fx)
        {
        case Merge:
            constants.sampleWeights[0] = XMVectorReplicate(mergeWeight1);
            constants.sampleWeights[1] = XMVectorReplicate(mergeWeight2);
            break;

        case BloomCombine:
            constants.sampleWeights[0] = XMVectorSet(bloomBaseSaturation, bloomSaturation, 0.f, 0.f);
            constants.sampleWeights[1] = XMVectorReplicate(bloomBaseIntensity);
            constants.sampleWeights[2] = XMVectorReplicate(bloomIntensity);
            break;

        default:
            break;
        }
    }

    if (mDirtyFlags & Dirty_ConstantBuffer)
    {
        mDirtyFlags &= ~Dirty_ConstantBuffer;
        mConstantBuffer = GraphicsMemory::Get(mDeviceResources->GetDevice()).AllocateConstant(constants);
    }

    commandList->SetGraphicsRootConstantBufferView(RootParameterIndex::ConstantBuffer, mConstantBuffer.GpuAddress());

    // Set the pipeline state.
    commandList->SetPipelineState(mPipelineState.Get());

    // Draw quad.
    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    commandList->DrawInstanced(3, 1, 0, 0);
}


// Public constructor.
DualPostProcess::DualPostProcess(_In_ ID3D12Device* device, const RenderTargetState& rtState, Effect fx)
  : pImpl(std::make_unique<Impl>(device, rtState, fx))
{
}


// Move constructor.
DualPostProcess::DualPostProcess(DualPostProcess&& moveFrom) noexcept
  : pImpl(std::move(moveFrom.pImpl))
{
}


// Move assignment.
DualPostProcess& DualPostProcess::operator= (DualPostProcess&& moveFrom) noexcept
{
    pImpl = std::move(moveFrom.pImpl);
    return *this;
}


// Public destructor.
DualPostProcess::~DualPostProcess()
{
}


// IPostProcess methods.
void DualPostProcess::Process(_In_ ID3D12GraphicsCommandList* commandList)
{
    pImpl->Process(commandList);
}


// Properties
void DualPostProcess::SetSourceTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor)
{
    pImpl->texture = srvDescriptor;
}


void DualPostProcess::SetSourceTexture2(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor)
{
    pImpl->texture2 = srvDescriptor;
}


void DualPostProcess::SetMergeParameters(float weight1, float weight2)
{
    pImpl->mergeWeight1 = weight1;
    pImpl->mergeWeight2 = weight2;
    pImpl->SetDirtyFlag();
}


void DualPostProcess::SetBloomCombineParameters(float bloom, float base, float bloomSaturation, float baseSaturation)
{
    pImpl->bloomIntensity = bloom;
    pImpl->bloomBaseIntensity = base;
    pImpl->bloomSaturation = bloomSaturation;
    pImpl->bloomBaseSaturation = baseSaturation;
    pImpl->SetDirtyFlag();
}
