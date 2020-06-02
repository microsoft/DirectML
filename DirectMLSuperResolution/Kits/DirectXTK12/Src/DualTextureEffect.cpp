//--------------------------------------------------------------------------------------
// File: DualTextureEffect.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "EffectCommon.h"

using namespace DirectX;
using Microsoft::WRL::ComPtr;


// Constant buffer layout. Must match the shader!
struct DualTextureEffectConstants
{
    XMVECTOR diffuseColor;
    XMVECTOR fogColor;
    XMVECTOR fogVector;
    XMMATRIX worldViewProj;
};

static_assert((sizeof(DualTextureEffectConstants) % 16) == 0, "CB size not padded correctly");


// Traits type describes our characteristics to the EffectBase template.
struct DualTextureEffectTraits
{
    using ConstantBufferType = DualTextureEffectConstants;

    static const int VertexShaderCount = 4;
    static const int PixelShaderCount = 2;
    static const int ShaderPermutationCount = 4;
    static const int RootSignatureCount = 1;
};


// Internal DualTextureEffect implementation class.
class DualTextureEffect::Impl : public EffectBase<DualTextureEffectTraits>
{
public:
    Impl(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription);
    
    enum RootParameterIndex
    {
        Texture1SRV,
        Texture1Sampler,
        Texture2SRV,
        Texture2Sampler,
        ConstantBuffer,
        RootParameterCount
    };

    EffectColor color;

    D3D12_GPU_DESCRIPTOR_HANDLE texture1;
    D3D12_GPU_DESCRIPTOR_HANDLE texture1Sampler;
    D3D12_GPU_DESCRIPTOR_HANDLE texture2;
    D3D12_GPU_DESCRIPTOR_HANDLE texture2Sampler;

    int GetPipelineStatePermutation(bool vertexColorEnabled) const;

    void Apply(_In_ ID3D12GraphicsCommandList* commandList);
};


// Include the precompiled shader code.
namespace
{
#if defined(_XBOX_ONE) && defined(_TITLE)
    #include "Shaders/Compiled/XboxOneDualTextureEffect_VSDualTexture.inc"
    #include "Shaders/Compiled/XboxOneDualTextureEffect_VSDualTextureNoFog.inc"
    #include "Shaders/Compiled/XboxOneDualTextureEffect_VSDualTextureVc.inc"
    #include "Shaders/Compiled/XboxOneDualTextureEffect_VSDualTextureVcNoFog.inc"

    #include "Shaders/Compiled/XboxOneDualTextureEffect_PSDualTexture.inc"
    #include "Shaders/Compiled/XboxOneDualTextureEffect_PSDualTextureNoFog.inc"
#else
    #include "Shaders/Compiled/DualTextureEffect_VSDualTexture.inc"
    #include "Shaders/Compiled/DualTextureEffect_VSDualTextureNoFog.inc"
    #include "Shaders/Compiled/DualTextureEffect_VSDualTextureVc.inc"
    #include "Shaders/Compiled/DualTextureEffect_VSDualTextureVcNoFog.inc"

    #include "Shaders/Compiled/DualTextureEffect_PSDualTexture.inc"
    #include "Shaders/Compiled/DualTextureEffect_PSDualTextureNoFog.inc"
#endif
}


template<>
const D3D12_SHADER_BYTECODE EffectBase<DualTextureEffectTraits>::VertexShaderBytecode[] =
{
    { DualTextureEffect_VSDualTexture,        sizeof(DualTextureEffect_VSDualTexture)        },
    { DualTextureEffect_VSDualTextureNoFog,   sizeof(DualTextureEffect_VSDualTextureNoFog)   },
    { DualTextureEffect_VSDualTextureVc,      sizeof(DualTextureEffect_VSDualTextureVc)      },
    { DualTextureEffect_VSDualTextureVcNoFog, sizeof(DualTextureEffect_VSDualTextureVcNoFog) },

};


template<>
const int EffectBase<DualTextureEffectTraits>::VertexShaderIndices[] =
{
    0,      // basic
    1,      // no fog
    2,      // vertex color
    3,      // vertex color, no fog
};


template<>
const D3D12_SHADER_BYTECODE EffectBase<DualTextureEffectTraits>::PixelShaderBytecode[] =
{
    { DualTextureEffect_PSDualTexture,        sizeof(DualTextureEffect_PSDualTexture)        },
    { DualTextureEffect_PSDualTextureNoFog,   sizeof(DualTextureEffect_PSDualTextureNoFog)   },

};


template<>
const int EffectBase<DualTextureEffectTraits>::PixelShaderIndices[] =
{
    0,      // basic
    1,      // no fog
    0,      // vertex color
    1,      // vertex color, no fog
};


// Global pool of per-device DualTextureEffect resources.
template<>
SharedResourcePool<ID3D12Device*, EffectBase<DualTextureEffectTraits>::DeviceResources> EffectBase<DualTextureEffectTraits>::deviceResourcesPool = {};


// Constructor.
DualTextureEffect::Impl::Impl(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription)
    : EffectBase(device),
    texture1{},
    texture1Sampler{},
    texture2{},
    texture2Sampler{}
{
    static_assert(_countof(EffectBase<DualTextureEffectTraits>::VertexShaderIndices) == DualTextureEffectTraits::ShaderPermutationCount, "array/max mismatch");
    static_assert(_countof(EffectBase<DualTextureEffectTraits>::VertexShaderBytecode) == DualTextureEffectTraits::VertexShaderCount, "array/max mismatch");
    static_assert(_countof(EffectBase<DualTextureEffectTraits>::PixelShaderBytecode) == DualTextureEffectTraits::PixelShaderCount, "array/max mismatch");
    static_assert(_countof(EffectBase<DualTextureEffectTraits>::PixelShaderIndices) == DualTextureEffectTraits::ShaderPermutationCount, "array/max mismatch");
    
    // Create root signature.
    {
        D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;

        CD3DX12_ROOT_PARAMETER rootParameters[RootParameterIndex::RootParameterCount] = {};
        rootParameters[RootParameterIndex::ConstantBuffer].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);

        // Texture 1
        CD3DX12_DESCRIPTOR_RANGE texture1Range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        CD3DX12_DESCRIPTOR_RANGE texture1SamplerRange(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0);
        rootParameters[RootParameterIndex::Texture1SRV].InitAsDescriptorTable(1, &texture1Range, D3D12_SHADER_VISIBILITY_PIXEL);
        rootParameters[RootParameterIndex::Texture1Sampler].InitAsDescriptorTable(1, &texture1SamplerRange, D3D12_SHADER_VISIBILITY_PIXEL);

        // Texture 2
        CD3DX12_DESCRIPTOR_RANGE texture2Range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);
        CD3DX12_DESCRIPTOR_RANGE texture2SamplerRange(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 1);
        rootParameters[RootParameterIndex::Texture2SRV].InitAsDescriptorTable(1, &texture2Range, D3D12_SHADER_VISIBILITY_PIXEL);
        rootParameters[RootParameterIndex::Texture2Sampler].InitAsDescriptorTable(1, &texture2SamplerRange, D3D12_SHADER_VISIBILITY_PIXEL);

        // Create the root signature
        CD3DX12_ROOT_SIGNATURE_DESC rsigDesc = {};
        rsigDesc.Init(_countof(rootParameters), rootParameters, 0, nullptr, rootSignatureFlags);

        mRootSignature = GetRootSignature(0, rsigDesc);
    }

    assert(mRootSignature != nullptr);

    // Validate flags & state.
    fog.enabled = (effectFlags & EffectFlags::Fog) != 0;

    if (effectFlags & EffectFlags::PerPixelLightingBit)
    {
        DebugTrace("ERROR: DualTextureEffect does not implement EffectFlags::PerPixelLighting\n");
        throw std::invalid_argument("DualTextureEffect");
    }
    else if (effectFlags & EffectFlags::Lighting)
    {
        DebugTrace("ERROR: DualTextureEffect does not implement EffectFlags::Lighting\n");
        throw std::invalid_argument("DualTextureEffect");
    }

    // Create pipeline state.
    int sp = GetPipelineStatePermutation(
        (effectFlags & EffectFlags::VertexColor) != 0);
    assert(sp >= 0 && sp < DualTextureEffectTraits::ShaderPermutationCount);
    _Analysis_assume_(sp >= 0 && sp < DualTextureEffectTraits::ShaderPermutationCount);

    int vi = EffectBase<DualTextureEffectTraits>::VertexShaderIndices[sp];
    assert(vi >= 0 && vi < DualTextureEffectTraits::VertexShaderCount);
    _Analysis_assume_(vi >= 0 && vi < DualTextureEffectTraits::VertexShaderCount);
    int pi = EffectBase<DualTextureEffectTraits>::PixelShaderIndices[sp];
    assert(pi >= 0 && pi < DualTextureEffectTraits::PixelShaderCount);
    _Analysis_assume_(pi >= 0 && pi < DualTextureEffectTraits::PixelShaderCount);

    pipelineDescription.CreatePipelineState(
        device,
        mRootSignature,
        EffectBase<DualTextureEffectTraits>::VertexShaderBytecode[vi],
        EffectBase<DualTextureEffectTraits>::PixelShaderBytecode[pi],
        mPipelineState.GetAddressOf());

    SetDebugObjectName(mPipelineState.Get(), L"DualTextureEffect");
}


int DualTextureEffect::Impl::GetPipelineStatePermutation(bool vertexColorEnabled) const
{
    int permutation = 0;

    // Use optimized shaders if fog is disabled.
    if (!fog.enabled)
    {
        permutation += 1;
    }

    // Support vertex coloring?
    if (vertexColorEnabled)
    {
        permutation += 2;
    }

    return permutation;
}


// Sets our state onto the D3D device.
void DualTextureEffect::Impl::Apply(_In_ ID3D12GraphicsCommandList* commandList)
{
    // Compute derived parameter values.
    matrices.SetConstants(dirtyFlags, constants.worldViewProj);

    fog.SetConstants(dirtyFlags, matrices.worldView, constants.fogVector);

    color.SetConstants(dirtyFlags, constants.diffuseColor);

    UpdateConstants();

    // Set the root signature
    commandList->SetGraphicsRootSignature(mRootSignature);

    // Set the textures
    if (!texture1.ptr || !texture2.ptr)
    {
        DebugTrace("ERROR: Missing texture(s) for DualTextureEffect (texture1 %llu, texture2 %llu)\n", texture1.ptr, texture2.ptr);
        throw std::exception("DualTextureEffect");
    }
    if (!texture1Sampler.ptr || !texture2Sampler.ptr)
    {
        DebugTrace("ERROR: Missing sampler(s) for DualTextureEffect (samplers1 %llu, samplers2 %llu)\n", texture2Sampler.ptr, texture2Sampler.ptr);
        throw std::exception("DualTextureEffect");
    }

    // **NOTE** If D3D asserts or crashes here, you probably need to call commandList->SetDescriptorHeaps() with the required descriptor heaps.
    commandList->SetGraphicsRootDescriptorTable(RootParameterIndex::Texture1SRV, texture1);
    commandList->SetGraphicsRootDescriptorTable(RootParameterIndex::Texture1Sampler, texture1Sampler);
    commandList->SetGraphicsRootDescriptorTable(RootParameterIndex::Texture2SRV, texture2);
    commandList->SetGraphicsRootDescriptorTable(RootParameterIndex::Texture2Sampler, texture2Sampler);

    // Set constants
    commandList->SetGraphicsRootConstantBufferView(RootParameterIndex::ConstantBuffer, GetConstantBufferGpuAddress());

    // Set the pipeline state
    commandList->SetPipelineState(EffectBase::mPipelineState.Get());
}


// Public constructor.
DualTextureEffect::DualTextureEffect(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription)
    : pImpl(std::make_unique<Impl>(device, effectFlags, pipelineDescription))
{
}


// Move constructor.
DualTextureEffect::DualTextureEffect(DualTextureEffect&& moveFrom) noexcept
  : pImpl(std::move(moveFrom.pImpl))
{
}


// Move assignment.
DualTextureEffect& DualTextureEffect::operator= (DualTextureEffect&& moveFrom) noexcept
{
    pImpl = std::move(moveFrom.pImpl);
    return *this;
}


// Public destructor.
DualTextureEffect::~DualTextureEffect()
{
}


// IEffect methods
void DualTextureEffect::Apply(_In_ ID3D12GraphicsCommandList* commandList)
{
    pImpl->Apply(commandList);
}


// Camera settings
void XM_CALLCONV DualTextureEffect::SetWorld(FXMMATRIX value)
{
    pImpl->matrices.world = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::WorldInverseTranspose | EffectDirtyFlags::FogVector;
}


void XM_CALLCONV DualTextureEffect::SetView(FXMMATRIX value)
{
    pImpl->matrices.view = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::EyePosition | EffectDirtyFlags::FogVector;
}


void XM_CALLCONV DualTextureEffect::SetProjection(FXMMATRIX value)
{
    pImpl->matrices.projection = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj;
}


void XM_CALLCONV DualTextureEffect::SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection)
{
    pImpl->matrices.world = world;
    pImpl->matrices.view = view;
    pImpl->matrices.projection = projection;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::WorldInverseTranspose | EffectDirtyFlags::EyePosition | EffectDirtyFlags::FogVector;
}


// Material settings
void XM_CALLCONV DualTextureEffect::SetDiffuseColor(FXMVECTOR value)
{
    pImpl->color.diffuseColor = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::MaterialColor;
}


void DualTextureEffect::SetAlpha(float value)
{
    pImpl->color.alpha = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::MaterialColor;
}


void XM_CALLCONV DualTextureEffect::SetColorAndAlpha(FXMVECTOR value)
{
    pImpl->color.diffuseColor = value;
    pImpl->color.alpha = XMVectorGetW(value);

    pImpl->dirtyFlags |= EffectDirtyFlags::MaterialColor;
}


// Fog settings.
void DualTextureEffect::SetFogStart(float value)
{
    pImpl->fog.start = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::FogVector;
}


void DualTextureEffect::SetFogEnd(float value)
{
    pImpl->fog.end = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::FogVector;
}


void XM_CALLCONV DualTextureEffect::SetFogColor(FXMVECTOR value)
{
    pImpl->constants.fogColor = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}


// Texture settings.
void DualTextureEffect::SetTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor, D3D12_GPU_DESCRIPTOR_HANDLE samplerDescriptor)
{
    pImpl->texture1 = srvDescriptor;
    pImpl->texture1Sampler = samplerDescriptor;
}


void DualTextureEffect::SetTexture2(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor, D3D12_GPU_DESCRIPTOR_HANDLE samplerDescriptor)
{
    pImpl->texture2 = srvDescriptor;
    pImpl->texture2Sampler = samplerDescriptor;
}
