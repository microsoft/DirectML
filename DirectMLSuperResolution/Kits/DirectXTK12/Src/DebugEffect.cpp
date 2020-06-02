//--------------------------------------------------------------------------------------
// File: DebugEffect.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "EffectCommon.h"

using namespace DirectX;


// Constant buffer layout. Must match the shader!
struct DebugEffectConstants
{
    XMVECTOR ambientDownAndAlpha;
    XMVECTOR ambientRange;
    
    XMMATRIX world;
    XMVECTOR worldInverseTranspose[3];
    XMMATRIX worldViewProj;
};

static_assert((sizeof(DebugEffectConstants) % 16) == 0, "CB size not padded correctly");


// Traits type describes our characteristics to the EffectBase template.
struct DebugEffectTraits
{
    using ConstantBufferType = DebugEffectConstants;

    static const int VertexShaderCount = 4;
    static const int PixelShaderCount = 4;
    static const int ShaderPermutationCount = 16;
    static const int RootSignatureCount = 1;
};


// Internal DebugEffect implementation class.
class DebugEffect::Impl : public EffectBase<DebugEffectTraits>
{
public:
    Impl(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, DebugEffect::Mode debugMode);

    enum RootParameterIndex
    {
        ConstantBuffer,
        RootParameterCount
    };

    int GetPipelineStatePermutation(bool vertexColorEnabled, DebugEffect::Mode debugMode, bool biasedVertexNormals) const;

    void Apply(_In_ ID3D12GraphicsCommandList* commandList);
};


// Include the precompiled shader code.
namespace
{
#if defined(_XBOX_ONE) && defined(_TITLE)
    #include "Shaders/Compiled/XboxOneDebugEffect_VSDebug.inc"
    #include "Shaders/Compiled/XboxOneDebugEffect_VSDebugVc.inc"

    #include "Shaders/Compiled/XboxOneDebugEffect_VSDebugBn.inc"
    #include "Shaders/Compiled/XboxOneDebugEffect_VSDebugVcBn.inc"

    #include "Shaders/Compiled/XboxOneDebugEffect_PSHemiAmbient.inc"
    #include "Shaders/Compiled/XboxOneDebugEffect_PSRGBNormals.inc"
    #include "Shaders/Compiled/XboxOneDebugEffect_PSRGBTangents.inc"
    #include "Shaders/Compiled/XboxOneDebugEffect_PSRGBBiTangents.inc"
#else    
    #include "Shaders/Compiled/DebugEffect_VSDebug.inc"
    #include "Shaders/Compiled/DebugEffect_VSDebugVc.inc"

    #include "Shaders/Compiled/DebugEffect_VSDebugBn.inc"
    #include "Shaders/Compiled/DebugEffect_VSDebugVcBn.inc"

    #include "Shaders/Compiled/DebugEffect_PSHemiAmbient.inc"
    #include "Shaders/Compiled/DebugEffect_PSRGBNormals.inc"
    #include "Shaders/Compiled/DebugEffect_PSRGBTangents.inc"
    #include "Shaders/Compiled/DebugEffect_PSRGBBiTangents.inc"
#endif
}


template<>
const D3D12_SHADER_BYTECODE EffectBase<DebugEffectTraits>::VertexShaderBytecode[] =
{    
    { DebugEffect_VSDebug,      sizeof(DebugEffect_VSDebug)     },
    { DebugEffect_VSDebugVc,    sizeof(DebugEffect_VSDebugVc)   },

    { DebugEffect_VSDebugBn,    sizeof(DebugEffect_VSDebugBn)   },
    { DebugEffect_VSDebugVcBn,  sizeof(DebugEffect_VSDebugVcBn) },
};


template<>
const int EffectBase<DebugEffectTraits>::VertexShaderIndices[] =
{    
    0,      // default
    0,      // normals
    0,      // tangents
    0,      // bitangents

    1,      // vertex color + default
    1,      // vertex color + normals
    1,      // vertex color + tangents
    1,      // vertex color + bitangents

    2,      // default (biased vertex normal)
    2,      // normals (biased vertex normal)
    2,      // tangents (biased vertex normal)
    2,      // bitangents (biased vertex normal)

    3,      // vertex color (biased vertex normal)
    3,      // vertex color (biased vertex normal) + normals
    3,      // vertex color (biased vertex normal) + tangents
    3,      // vertex color (biased vertex normal) + bitangents
};


template<>
const D3D12_SHADER_BYTECODE EffectBase<DebugEffectTraits>::PixelShaderBytecode[] =
{
    { DebugEffect_PSHemiAmbient,    sizeof(DebugEffect_PSHemiAmbient)          },
    { DebugEffect_PSRGBNormals,     sizeof(DebugEffect_PSRGBNormals)     },
    { DebugEffect_PSRGBTangents,    sizeof(DebugEffect_PSRGBTangents)    },
    { DebugEffect_PSRGBBiTangents,  sizeof(DebugEffect_PSRGBBiTangents) },
};


template<>
const int EffectBase<DebugEffectTraits>::PixelShaderIndices[] =
{    
    0,      // default
    1,      // normals
    2,      // tangents
    3,      // bitangents

    0,      // vertex color + default
    1,      // vertex color + normals
    2,      // vertex color + tangents
    3,      // vertex color + bitangents

    0,      // default (biased vertex normal)
    1,      // normals (biased vertex normal)
    2,      // tangents (biased vertex normal)
    3,      // bitangents (biased vertex normal)

    0,      // vertex color (biased vertex normal)
    1,      // vertex color (biased vertex normal) + normals
    2,      // vertex color (biased vertex normal) + tangents
    3,      // vertex color (biased vertex normal) + bitangents
};


// Global pool of per-deviceDebugEffect resources.
template<>
SharedResourcePool<ID3D12Device*, EffectBase<DebugEffectTraits>::DeviceResources> EffectBase<DebugEffectTraits>::deviceResourcesPool = {};


// Constructor.
DebugEffect::Impl::Impl(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, DebugEffect::Mode debugMode)
    : EffectBase(device)
{
    static_assert(_countof(EffectBase<DebugEffectTraits>::VertexShaderIndices) == DebugEffectTraits::ShaderPermutationCount, "array/max mismatch");
    static_assert(_countof(EffectBase<DebugEffectTraits>::VertexShaderBytecode) == DebugEffectTraits::VertexShaderCount, "array/max mismatch");
    static_assert(_countof(EffectBase<DebugEffectTraits>::PixelShaderBytecode) == DebugEffectTraits::PixelShaderCount, "array/max mismatch");
    static_assert(_countof(EffectBase<DebugEffectTraits>::PixelShaderIndices) == DebugEffectTraits::ShaderPermutationCount, "array/max mismatch");

    static const XMVECTORF32 s_lower = { { { 0.f, 0.f, 0.f, 1.f } } };

    constants.ambientDownAndAlpha = s_lower;
    constants.ambientRange = g_XMOne;

    // Create root signature.
    {
        D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;

        // Create root parameters and initialize first (constants)
        CD3DX12_ROOT_PARAMETER rootParameters[RootParameterIndex::RootParameterCount] = {};
        rootParameters[RootParameterIndex::ConstantBuffer].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);

        // Root parameter descriptor - conditionally initialized
        CD3DX12_ROOT_SIGNATURE_DESC rsigDesc = {};

        rsigDesc.Init(1, rootParameters, 0, nullptr, rootSignatureFlags);

        mRootSignature = GetRootSignature(0, rsigDesc);
    }

    assert(mRootSignature != nullptr);

    // Create pipeline state.
    int sp = GetPipelineStatePermutation(
        (effectFlags & EffectFlags::VertexColor) != 0,
        debugMode,
        (effectFlags & EffectFlags::BiasedVertexNormals) != 0);
    assert(sp >= 0 && sp < DebugEffectTraits::ShaderPermutationCount);
    _Analysis_assume_(sp >= 0 && sp < DebugEffectTraits::ShaderPermutationCount);

    int vi = EffectBase<DebugEffectTraits>::VertexShaderIndices[sp];
    assert(vi >= 0 && vi < DebugEffectTraits::VertexShaderCount);
    _Analysis_assume_(vi >= 0 && vi < DebugEffectTraits::VertexShaderCount);
    int pi = EffectBase<DebugEffectTraits>::PixelShaderIndices[sp];
    assert(pi >= 0 && pi < DebugEffectTraits::PixelShaderCount);
    _Analysis_assume_(pi >= 0 && pi < DebugEffectTraits::PixelShaderCount);

    pipelineDescription.CreatePipelineState(
        device,
        mRootSignature,
        EffectBase<DebugEffectTraits>::VertexShaderBytecode[vi],
        EffectBase<DebugEffectTraits>::PixelShaderBytecode[pi],
        mPipelineState.GetAddressOf());

    SetDebugObjectName(mPipelineState.Get(), L"DebugEffect");
}


int DebugEffect::Impl::GetPipelineStatePermutation(bool vertexColorEnabled, DebugEffect::Mode debugMode, bool biasedVertexNormals) const
{
    int permutation = static_cast<int>(debugMode);

    // Support vertex coloring?
    if (vertexColorEnabled)
    {
        permutation += 4;
    }

    if (biasedVertexNormals)
    {
        // Compressed normals need to be scaled and biased in the vertex shader.
        permutation += 8;
    }

    return permutation;
}


// Sets our state onto the D3D device.
void DebugEffect::Impl::Apply(_In_ ID3D12GraphicsCommandList* commandList)
{
    // Compute derived parameter values.
    matrices.SetConstants(dirtyFlags, constants.worldViewProj);

    // World inverse transpose matrix.
    if (dirtyFlags & EffectDirtyFlags::WorldInverseTranspose)
    {
        constants.world = XMMatrixTranspose(matrices.world);

        XMMATRIX worldInverse = XMMatrixInverse(nullptr, matrices.world);

        constants.worldInverseTranspose[0] = worldInverse.r[0];
        constants.worldInverseTranspose[1] = worldInverse.r[1];
        constants.worldInverseTranspose[2] = worldInverse.r[2];

        dirtyFlags &= ~EffectDirtyFlags::WorldInverseTranspose;
        dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
    }

    UpdateConstants();

    // Set the root signature
    commandList->SetGraphicsRootSignature(mRootSignature);

    // Set constants
    commandList->SetGraphicsRootConstantBufferView(RootParameterIndex::ConstantBuffer, GetConstantBufferGpuAddress());

    // Set the pipeline state
    commandList->SetPipelineState(EffectBase::mPipelineState.Get());
}


// Public constructor.
DebugEffect::DebugEffect(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, Mode debugMode)
  : pImpl(std::make_unique<Impl>(device, effectFlags, pipelineDescription, debugMode))
{
}


// Move constructor.
DebugEffect::DebugEffect(DebugEffect&& moveFrom) noexcept
  : pImpl(std::move(moveFrom.pImpl))
{
}


// Move assignment.
DebugEffect& DebugEffect::operator= (DebugEffect&& moveFrom) noexcept
{
    pImpl = std::move(moveFrom.pImpl);
    return *this;
}


// Public destructor.
DebugEffect::~DebugEffect()
{
}


// IEffect methods.
void DebugEffect::Apply(_In_ ID3D12GraphicsCommandList* commandList)
{
    pImpl->Apply(commandList);
}


// Camera settings.
void XM_CALLCONV DebugEffect::SetWorld(FXMMATRIX value)
{
    pImpl->matrices.world = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::WorldInverseTranspose;
}


void XM_CALLCONV DebugEffect::SetView(FXMMATRIX value)
{
    pImpl->matrices.view = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj;
}


void XM_CALLCONV DebugEffect::SetProjection(FXMMATRIX value)
{
    pImpl->matrices.projection = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj;
}


void XM_CALLCONV DebugEffect::SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection)
{
    pImpl->matrices.world = world;
    pImpl->matrices.view = view;
    pImpl->matrices.projection = projection;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::WorldInverseTranspose;
}


// Material settings.
void XM_CALLCONV DebugEffect::SetHemisphericalAmbientColor(FXMVECTOR upper, FXMVECTOR lower)
{
    // Set xyz to new value, but preserve existing w (alpha).
    pImpl->constants.ambientDownAndAlpha = XMVectorSelect(pImpl->constants.ambientDownAndAlpha, lower, g_XMSelect1110);

    pImpl->constants.ambientRange = XMVectorSubtract(upper, lower);

    pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}

void DebugEffect::SetAlpha(float value)
{
    // Set w to new value, but preserve existing xyz (ambient down).
    pImpl->constants.ambientDownAndAlpha = XMVectorSetW(pImpl->constants.ambientDownAndAlpha, value);

    pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}
