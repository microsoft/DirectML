//--------------------------------------------------------------------------------------
// File: AlphaTestEffect.cpp
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
struct AlphaTestEffectConstants
{
    XMVECTOR diffuseColor;
    XMVECTOR alphaTest;
    XMVECTOR fogColor;
    XMVECTOR fogVector;
    XMMATRIX worldViewProj;
};

static_assert((sizeof(AlphaTestEffectConstants) % 16) == 0, "CB size not padded correctly");


// Traits type describes our characteristics to the EffectBase template.
struct AlphaTestEffectTraits
{
    using ConstantBufferType = AlphaTestEffectConstants;

    static const int VertexShaderCount = 4;
    static const int PixelShaderCount = 4;
    static const int ShaderPermutationCount = 8;
    static const int RootSignatureCount = 1;
};


// Internal AlphaTestEffect implementation class.
class AlphaTestEffect::Impl : public EffectBase<AlphaTestEffectTraits>
{
public:
    Impl(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, D3D12_COMPARISON_FUNC alphaFunction);

    enum RootParameterIndex
    {
        ConstantBuffer,
        TextureSRV,
        TextureSampler,
        RootParameterCount
    };

    D3D12_COMPARISON_FUNC mAlphaFunction;
    int referenceAlpha;

    EffectColor color;

    D3D12_GPU_DESCRIPTOR_HANDLE texture;
    D3D12_GPU_DESCRIPTOR_HANDLE textureSampler;
    
    int GetPipelineStatePermutation(bool vertexColorEnabled) const;

    void Apply(_In_ ID3D12GraphicsCommandList* commandList);
};


// Include the precompiled shader code.
namespace
{
#if defined(_XBOX_ONE) && defined(_TITLE)
    #include "Shaders/Compiled/XboxOneAlphaTestEffect_VSAlphaTest.inc"
    #include "Shaders/Compiled/XboxOneAlphaTestEffect_VSAlphaTestNoFog.inc"
    #include "Shaders/Compiled/XboxOneAlphaTestEffect_VSAlphaTestVc.inc"
    #include "Shaders/Compiled/XboxOneAlphaTestEffect_VSAlphaTestVcNoFog.inc"

    #include "Shaders/Compiled/XboxOneAlphaTestEffect_PSAlphaTestLtGt.inc"
    #include "Shaders/Compiled/XboxOneAlphaTestEffect_PSAlphaTestLtGtNoFog.inc"
    #include "Shaders/Compiled/XboxOneAlphaTestEffect_PSAlphaTestEqNe.inc"
    #include "Shaders/Compiled/XboxOneAlphaTestEffect_PSAlphaTestEqNeNoFog.inc"
#else
    #include "Shaders/Compiled/AlphaTestEffect_VSAlphaTest.inc"
    #include "Shaders/Compiled/AlphaTestEffect_VSAlphaTestNoFog.inc"
    #include "Shaders/Compiled/AlphaTestEffect_VSAlphaTestVc.inc"
    #include "Shaders/Compiled/AlphaTestEffect_VSAlphaTestVcNoFog.inc"

    #include "Shaders/Compiled/AlphaTestEffect_PSAlphaTestLtGt.inc"
    #include "Shaders/Compiled/AlphaTestEffect_PSAlphaTestLtGtNoFog.inc"
    #include "Shaders/Compiled/AlphaTestEffect_PSAlphaTestEqNe.inc"
    #include "Shaders/Compiled/AlphaTestEffect_PSAlphaTestEqNeNoFog.inc"
#endif
}


template<>
const D3D12_SHADER_BYTECODE EffectBase<AlphaTestEffectTraits>::VertexShaderBytecode[] =
{
    { AlphaTestEffect_VSAlphaTest,        sizeof(AlphaTestEffect_VSAlphaTest)        },
    { AlphaTestEffect_VSAlphaTestNoFog,   sizeof(AlphaTestEffect_VSAlphaTestNoFog)   },
    { AlphaTestEffect_VSAlphaTestVc,      sizeof(AlphaTestEffect_VSAlphaTestVc)      },
    { AlphaTestEffect_VSAlphaTestVcNoFog, sizeof(AlphaTestEffect_VSAlphaTestVcNoFog) },
};


template<>
const int EffectBase<AlphaTestEffectTraits>::VertexShaderIndices[] =
{
    0,      // lt/gt
    1,      // lt/gt, no fog
    2,      // lt/gt, vertex color
    3,      // lt/gt, vertex color, no fog
    
    0,      // eq/ne
    1,      // eq/ne, no fog
    2,      // eq/ne, vertex color
    3,      // eq/ne, vertex color, no fog
};


template<>
const D3D12_SHADER_BYTECODE EffectBase<AlphaTestEffectTraits>::PixelShaderBytecode[] =
{
    { AlphaTestEffect_PSAlphaTestLtGt,      sizeof(AlphaTestEffect_PSAlphaTestLtGt)      },
    { AlphaTestEffect_PSAlphaTestLtGtNoFog, sizeof(AlphaTestEffect_PSAlphaTestLtGtNoFog) },
    { AlphaTestEffect_PSAlphaTestEqNe,      sizeof(AlphaTestEffect_PSAlphaTestEqNe)      },
    { AlphaTestEffect_PSAlphaTestEqNeNoFog, sizeof(AlphaTestEffect_PSAlphaTestEqNeNoFog) },
};


template<>
const int EffectBase<AlphaTestEffectTraits>::PixelShaderIndices[] =
{
    0,      // lt/gt
    1,      // lt/gt, no fog
    0,      // lt/gt, vertex color
    1,      // lt/gt, vertex color, no fog
    
    2,      // eq/ne
    3,      // eq/ne, no fog
    2,      // eq/ne, vertex color
    3,      // eq/ne, vertex color, no fog
};


// Global pool of per-device AlphaTestEffect resources.
template<>
SharedResourcePool<ID3D12Device*, EffectBase<AlphaTestEffectTraits>::DeviceResources> EffectBase<AlphaTestEffectTraits>::deviceResourcesPool = {};

// Constructor.
AlphaTestEffect::Impl::Impl(_In_ ID3D12Device* device,
    int effectFlags, const EffectPipelineStateDescription& pipelineDescription, D3D12_COMPARISON_FUNC alphaFunction)
    : EffectBase(device),
    mAlphaFunction(alphaFunction),
    referenceAlpha(0),
    texture{},
    textureSampler{}
{
    static_assert(_countof(EffectBase<AlphaTestEffectTraits>::VertexShaderIndices) == AlphaTestEffectTraits::ShaderPermutationCount, "array/max mismatch");
    static_assert(_countof(EffectBase<AlphaTestEffectTraits>::VertexShaderBytecode) == AlphaTestEffectTraits::VertexShaderCount, "array/max mismatch");
    static_assert(_countof(EffectBase<AlphaTestEffectTraits>::PixelShaderBytecode) == AlphaTestEffectTraits::PixelShaderCount, "array/max mismatch");
    static_assert(_countof(EffectBase<AlphaTestEffectTraits>::PixelShaderIndices) == AlphaTestEffectTraits::ShaderPermutationCount, "array/max mismatch");

    // Create root signature.
    {
        D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;

        CD3DX12_DESCRIPTOR_RANGE textureRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        CD3DX12_DESCRIPTOR_RANGE textureSamplerRange(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0);

        CD3DX12_ROOT_PARAMETER rootParameters[RootParameterIndex::RootParameterCount] = {};
        rootParameters[RootParameterIndex::TextureSRV].InitAsDescriptorTable(1, &textureRange, D3D12_SHADER_VISIBILITY_PIXEL);
        rootParameters[RootParameterIndex::TextureSampler].InitAsDescriptorTable(1, &textureSamplerRange, D3D12_SHADER_VISIBILITY_PIXEL);
        rootParameters[RootParameterIndex::ConstantBuffer].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);

        CD3DX12_ROOT_SIGNATURE_DESC rsigDesc = {};
        rsigDesc.Init(_countof(rootParameters), rootParameters, 0, nullptr, rootSignatureFlags);

        mRootSignature = GetRootSignature(0, rsigDesc);
    }

    assert(mRootSignature != nullptr);

    fog.enabled = (effectFlags & EffectFlags::Fog) != 0;

    if (effectFlags & EffectFlags::PerPixelLightingBit)
    {
        DebugTrace("ERROR: AlphaTestEffect does not implement EffectFlags::PerPixelLighting\n");
        throw std::invalid_argument("AlphaTestEffect");
    }
    else if (effectFlags & EffectFlags::Lighting)
    {
        DebugTrace("ERROR: DualTextureEffect does not implement EffectFlags::Lighting\n");
        throw std::invalid_argument("AlphaTestEffect");
    }

    // Create pipeline state.
    int sp = GetPipelineStatePermutation(
        (effectFlags & EffectFlags::VertexColor) != 0);
    assert(sp >= 0 && sp < AlphaTestEffectTraits::ShaderPermutationCount);
    _Analysis_assume_(sp >= 0 && sp < AlphaTestEffectTraits::ShaderPermutationCount);

    int vi = EffectBase<AlphaTestEffectTraits>::VertexShaderIndices[sp];
    assert(vi >= 0 && vi < AlphaTestEffectTraits::VertexShaderCount);
    _Analysis_assume_(vi >= 0 && vi < AlphaTestEffectTraits::VertexShaderCount);
    int pi = EffectBase<AlphaTestEffectTraits>::PixelShaderIndices[sp];
    assert(pi >= 0 && pi < AlphaTestEffectTraits::PixelShaderCount);
    _Analysis_assume_(pi >= 0 && pi < AlphaTestEffectTraits::PixelShaderCount);

    pipelineDescription.CreatePipelineState(
        device,
        mRootSignature,
        EffectBase<AlphaTestEffectTraits>::VertexShaderBytecode[vi],
        EffectBase<AlphaTestEffectTraits>::PixelShaderBytecode[pi],
        mPipelineState.GetAddressOf());

    SetDebugObjectName(mPipelineState.Get(), L"AlphaTestEffect");
}


int AlphaTestEffect::Impl::GetPipelineStatePermutation(bool vertexColorEnabled) const
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

    // Which alpha compare mode?
    if (mAlphaFunction == D3D12_COMPARISON_FUNC_EQUAL ||
        mAlphaFunction == D3D12_COMPARISON_FUNC_NOT_EQUAL)
    {
        permutation += 4;
    }

    return permutation;
}


// Sets our state onto the D3D device.
void AlphaTestEffect::Impl::Apply(_In_ ID3D12GraphicsCommandList* commandList)
{
    // Compute derived parameter values.
    matrices.SetConstants(dirtyFlags, constants.worldViewProj);
    fog.SetConstants(dirtyFlags, matrices.worldView, constants.fogVector);            
    color.SetConstants(dirtyFlags, constants.diffuseColor);

    UpdateConstants();

    // Recompute the alpha test settings?
    if (dirtyFlags & EffectDirtyFlags::AlphaTest)
    {
        // Convert reference alpha from 8 bit integer to 0-1 float format.
        auto reference = static_cast<float>(referenceAlpha) / 255.0f;
                
        // Comparison tolerance of half the 8 bit integer precision.
        const float threshold = 0.5f / 255.0f;

        // What to do if the alpha comparison passes or fails. Positive accepts the pixel, negative clips it.
        static const XMVECTORF32 selectIfTrue  = { { {  1, -1 } } };
        static const XMVECTORF32 selectIfFalse = { { { -1,  1 } } };
        static const XMVECTORF32 selectNever   = { { { -1, -1 } } };
        static const XMVECTORF32 selectAlways  = { { {  1,  1 } } };

        float compareTo;
        XMVECTOR resultSelector;

        switch (mAlphaFunction)
        {
            case D3D12_COMPARISON_FUNC_LESS:
                // Shader will evaluate: clip((a < x) ? z : w)
                compareTo = reference - threshold;
                resultSelector = selectIfTrue;
                break;

            case D3D12_COMPARISON_FUNC_LESS_EQUAL:
                // Shader will evaluate: clip((a < x) ? z : w)
                compareTo = reference + threshold;
                resultSelector = selectIfTrue;
                break;

            case D3D12_COMPARISON_FUNC_GREATER_EQUAL:
                // Shader will evaluate: clip((a < x) ? z : w)
                compareTo = reference - threshold;
                resultSelector = selectIfFalse;
                break;

            case D3D12_COMPARISON_FUNC_GREATER:
                // Shader will evaluate: clip((a < x) ? z : w)
                compareTo = reference + threshold;
                resultSelector = selectIfFalse;
                break;

            case D3D12_COMPARISON_FUNC_EQUAL:
                // Shader will evaluate: clip((abs(a - x) < y) ? z : w)
                compareTo = reference;
                resultSelector = selectIfTrue;
                break;

            case D3D12_COMPARISON_FUNC_NOT_EQUAL:
                // Shader will evaluate: clip((abs(a - x) < y) ? z : w)
                compareTo = reference;
                resultSelector = selectIfFalse;
                break;

            case D3D12_COMPARISON_FUNC_NEVER:
                // Shader will evaluate: clip((a < x) ? z : w)
                compareTo = 0;
                resultSelector = selectNever;
                break;

            case D3D12_COMPARISON_FUNC_ALWAYS:
                // Shader will evaluate: clip((a < x) ? z : w)
                compareTo = 0;
                resultSelector = selectAlways;
                break;

            default:
                throw std::exception("Unknown alpha test function");
        }

        // x = compareTo, y = threshold, zw = resultSelector.
        constants.alphaTest = XMVectorPermute<0, 1, 4, 5>(XMVectorSet(compareTo, threshold, 0, 0), resultSelector);
                
        dirtyFlags &= ~EffectDirtyFlags::AlphaTest;
        dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
    }

    // Set the root signature
    commandList->SetGraphicsRootSignature(mRootSignature);

    // Set the texture
    if (!texture.ptr || !textureSampler.ptr)
    {
        DebugTrace("ERROR: Missing texture or sampler for AlphaTestEffect (texture %llu, sampler %llu)\n", texture.ptr, textureSampler.ptr);
        throw std::exception("AlphaTestEffect");
    }

    // **NOTE** If D3D asserts or crashes here, you probably need to call commandList->SetDescriptorHeaps() with the required descriptor heaps.
    commandList->SetGraphicsRootDescriptorTable(RootParameterIndex::TextureSRV, texture);
    commandList->SetGraphicsRootDescriptorTable(RootParameterIndex::TextureSampler, textureSampler);

    // Set constants
    commandList->SetGraphicsRootConstantBufferView(RootParameterIndex::ConstantBuffer, GetConstantBufferGpuAddress());

    // Set the pipeline state
    commandList->SetPipelineState(EffectBase::mPipelineState.Get());
}

// Public constructor.
AlphaTestEffect::AlphaTestEffect(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, D3D12_COMPARISON_FUNC alphaFunction)
    : pImpl(std::make_unique<Impl>(device, effectFlags, pipelineDescription, alphaFunction))
{
}


// Move constructor.
AlphaTestEffect::AlphaTestEffect(AlphaTestEffect&& moveFrom) noexcept
  : pImpl(std::move(moveFrom.pImpl))
{
}


// Move assignment.
AlphaTestEffect& AlphaTestEffect::operator= (AlphaTestEffect&& moveFrom) noexcept
{
    pImpl = std::move(moveFrom.pImpl);
    return *this;
}


// Public destructor.
AlphaTestEffect::~AlphaTestEffect()
{
}


// IEffect methods
void AlphaTestEffect::Apply(_In_ ID3D12GraphicsCommandList* commandList)
{
    pImpl->Apply(commandList);
}


// Camera settings
void XM_CALLCONV AlphaTestEffect::SetWorld(FXMMATRIX value)
{
    pImpl->matrices.world = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::WorldInverseTranspose | EffectDirtyFlags::FogVector;
}


void XM_CALLCONV AlphaTestEffect::SetView(FXMMATRIX value)
{
    pImpl->matrices.view = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::EyePosition | EffectDirtyFlags::FogVector;
}


void XM_CALLCONV AlphaTestEffect::SetProjection(FXMMATRIX value)
{
    pImpl->matrices.projection = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj;
}


void XM_CALLCONV AlphaTestEffect::SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection)
{
    pImpl->matrices.world = world;
    pImpl->matrices.view = view;
    pImpl->matrices.projection = projection;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::WorldInverseTranspose | EffectDirtyFlags::EyePosition | EffectDirtyFlags::FogVector;
}


// Material settings
void XM_CALLCONV AlphaTestEffect::SetDiffuseColor(FXMVECTOR value)
{
    pImpl->color.diffuseColor = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::MaterialColor;
}


void AlphaTestEffect::SetAlpha(float value)
{
    pImpl->color.alpha = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::MaterialColor;
}


void XM_CALLCONV AlphaTestEffect::SetColorAndAlpha(FXMVECTOR value)
{
    pImpl->color.diffuseColor = value;
    pImpl->color.alpha = XMVectorGetW(value);

    pImpl->dirtyFlags |= EffectDirtyFlags::MaterialColor;
}


// Fog settings.
void AlphaTestEffect::SetFogStart(float value)
{
    pImpl->fog.start = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::FogVector;
}


void AlphaTestEffect::SetFogEnd(float value)
{
    pImpl->fog.end = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::FogVector;
}


void XM_CALLCONV AlphaTestEffect::SetFogColor(FXMVECTOR value)
{
    pImpl->constants.fogColor = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}


// Texture settings.
void AlphaTestEffect::SetTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor, D3D12_GPU_DESCRIPTOR_HANDLE samplerDescriptor)
{
    pImpl->texture = srvDescriptor;
    pImpl->textureSampler = samplerDescriptor;
}


void AlphaTestEffect::SetReferenceAlpha(int value)
{
    pImpl->referenceAlpha = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::AlphaTest;
}
