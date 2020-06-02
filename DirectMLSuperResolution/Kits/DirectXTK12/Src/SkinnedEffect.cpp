//--------------------------------------------------------------------------------------
// File: SkinnedEffect.cpp
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
struct SkinnedEffectConstants
{
    XMVECTOR diffuseColor;
    XMVECTOR emissiveColor;
    XMVECTOR specularColorAndPower;
    
    XMVECTOR lightDirection[IEffectLights::MaxDirectionalLights];
    XMVECTOR lightDiffuseColor[IEffectLights::MaxDirectionalLights];
    XMVECTOR lightSpecularColor[IEffectLights::MaxDirectionalLights];

    XMVECTOR eyePosition;

    XMVECTOR fogColor;
    XMVECTOR fogVector;

    XMMATRIX world;
    XMVECTOR worldInverseTranspose[3];
    XMMATRIX worldViewProj;

    XMVECTOR bones[SkinnedEffect::MaxBones][3];
};

static_assert((sizeof(SkinnedEffectConstants) % 16) == 0, "CB size not padded correctly");


// Traits type describes our characteristics to the EffectBase template.
struct SkinnedEffectTraits
{
    using ConstantBufferType = SkinnedEffectConstants;

    static const int VertexShaderCount = 12;
    static const int PixelShaderCount = 3;
    static const int ShaderPermutationCount = 24;
    static const int RootSignatureCount = 1;
};


// Internal SkinnedEffect implementation class.
class SkinnedEffect::Impl : public EffectBase<SkinnedEffectTraits>
{
public:
    Impl(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, int weightsPerVertex);

    enum RootParameterIndex
    {
        ConstantBuffer,
        TextureSRV,
        TextureSampler,
        RootParameterCount
    };

    D3D12_GPU_DESCRIPTOR_HANDLE texture;
    D3D12_GPU_DESCRIPTOR_HANDLE sampler;

    EffectLights lights;

    int GetPipelineStatePermutation(bool preferPerPixelLighting, int weightsPerVertex, bool biasedVertexNormals) const;

    void Apply(_In_ ID3D12GraphicsCommandList* commandList);
};


// Include the precompiled shader code.
namespace
{
#if defined(_XBOX_ONE) && defined(_TITLE)
    #include "Shaders/Compiled/XboxOneSkinnedEffect_VSSkinnedVertexLightingOneBone.inc"
    #include "Shaders/Compiled/XboxOneSkinnedEffect_VSSkinnedVertexLightingTwoBones.inc"
    #include "Shaders/Compiled/XboxOneSkinnedEffect_VSSkinnedVertexLightingFourBones.inc"

    #include "Shaders/Compiled/XboxOneSkinnedEffect_VSSkinnedPixelLightingOneBone.inc"
    #include "Shaders/Compiled/XboxOneSkinnedEffect_VSSkinnedPixelLightingTwoBones.inc"
    #include "Shaders/Compiled/XboxOneSkinnedEffect_VSSkinnedPixelLightingFourBones.inc"

    #include "Shaders/Compiled/XboxOneSkinnedEffect_VSSkinnedVertexLightingOneBoneBn.inc"
    #include "Shaders/Compiled/XboxOneSkinnedEffect_VSSkinnedVertexLightingTwoBonesBn.inc"
    #include "Shaders/Compiled/XboxOneSkinnedEffect_VSSkinnedVertexLightingFourBonesBn.inc"

    #include "Shaders/Compiled/XboxOneSkinnedEffect_VSSkinnedPixelLightingOneBoneBn.inc"
    #include "Shaders/Compiled/XboxOneSkinnedEffect_VSSkinnedPixelLightingTwoBonesBn.inc"
    #include "Shaders/Compiled/XboxOneSkinnedEffect_VSSkinnedPixelLightingFourBonesBn.inc"

    #include "Shaders/Compiled/XboxOneSkinnedEffect_PSSkinnedVertexLighting.inc"
    #include "Shaders/Compiled/XboxOneSkinnedEffect_PSSkinnedVertexLightingNoFog.inc"
    #include "Shaders/Compiled/XboxOneSkinnedEffect_PSSkinnedPixelLighting.inc"
#else
    #include "Shaders/Compiled/SkinnedEffect_VSSkinnedVertexLightingOneBone.inc"
    #include "Shaders/Compiled/SkinnedEffect_VSSkinnedVertexLightingTwoBones.inc"
    #include "Shaders/Compiled/SkinnedEffect_VSSkinnedVertexLightingFourBones.inc"

    #include "Shaders/Compiled/SkinnedEffect_VSSkinnedPixelLightingOneBone.inc"
    #include "Shaders/Compiled/SkinnedEffect_VSSkinnedPixelLightingTwoBones.inc"
    #include "Shaders/Compiled/SkinnedEffect_VSSkinnedPixelLightingFourBones.inc"

    #include "Shaders/Compiled/SkinnedEffect_VSSkinnedVertexLightingOneBoneBn.inc"
    #include "Shaders/Compiled/SkinnedEffect_VSSkinnedVertexLightingTwoBonesBn.inc"
    #include "Shaders/Compiled/SkinnedEffect_VSSkinnedVertexLightingFourBonesBn.inc"

    #include "Shaders/Compiled/SkinnedEffect_VSSkinnedPixelLightingOneBoneBn.inc"
    #include "Shaders/Compiled/SkinnedEffect_VSSkinnedPixelLightingTwoBonesBn.inc"
    #include "Shaders/Compiled/SkinnedEffect_VSSkinnedPixelLightingFourBonesBn.inc"

    #include "Shaders/Compiled/SkinnedEffect_PSSkinnedVertexLighting.inc"
    #include "Shaders/Compiled/SkinnedEffect_PSSkinnedVertexLightingNoFog.inc"
    #include "Shaders/Compiled/SkinnedEffect_PSSkinnedPixelLighting.inc"
#endif
}


template<>
const D3D12_SHADER_BYTECODE EffectBase<SkinnedEffectTraits>::VertexShaderBytecode[] =
{
    { SkinnedEffect_VSSkinnedVertexLightingOneBone,     sizeof(SkinnedEffect_VSSkinnedVertexLightingOneBone)     },
    { SkinnedEffect_VSSkinnedVertexLightingTwoBones,    sizeof(SkinnedEffect_VSSkinnedVertexLightingTwoBones)    },
    { SkinnedEffect_VSSkinnedVertexLightingFourBones,   sizeof(SkinnedEffect_VSSkinnedVertexLightingFourBones)   },

    { SkinnedEffect_VSSkinnedPixelLightingOneBone,      sizeof(SkinnedEffect_VSSkinnedPixelLightingOneBone)      },
    { SkinnedEffect_VSSkinnedPixelLightingTwoBones,     sizeof(SkinnedEffect_VSSkinnedPixelLightingTwoBones)     },
    { SkinnedEffect_VSSkinnedPixelLightingFourBones,    sizeof(SkinnedEffect_VSSkinnedPixelLightingFourBones)    },

    { SkinnedEffect_VSSkinnedVertexLightingOneBoneBn,   sizeof(SkinnedEffect_VSSkinnedVertexLightingOneBoneBn)   },
    { SkinnedEffect_VSSkinnedVertexLightingTwoBonesBn,  sizeof(SkinnedEffect_VSSkinnedVertexLightingTwoBonesBn)  },
    { SkinnedEffect_VSSkinnedVertexLightingFourBonesBn, sizeof(SkinnedEffect_VSSkinnedVertexLightingFourBonesBn) },

    { SkinnedEffect_VSSkinnedPixelLightingOneBoneBn,    sizeof(SkinnedEffect_VSSkinnedPixelLightingOneBoneBn)    },
    { SkinnedEffect_VSSkinnedPixelLightingTwoBonesBn,   sizeof(SkinnedEffect_VSSkinnedPixelLightingTwoBonesBn)   },
    { SkinnedEffect_VSSkinnedPixelLightingFourBonesBn,  sizeof(SkinnedEffect_VSSkinnedPixelLightingFourBonesBn)  },
};


template<>
const int EffectBase<SkinnedEffectTraits>::VertexShaderIndices[] =
{
    0,      // vertex lighting, one bone
    0,      // vertex lighting, one bone, no fog
    1,      // vertex lighting, two bones
    1,      // vertex lighting, two bones, no fog
    2,      // vertex lighting, four bones
    2,      // vertex lighting, four bones, no fog

    3,      // pixel lighting, one bone
    3,      // pixel lighting, one bone, no fog
    4,      // pixel lighting, two bones
    4,      // pixel lighting, two bones, no fog
    5,      // pixel lighting, four bones
    5,      // pixel lighting, four bones, no fog

    6,      // vertex lighting (biased vertex normals), one bone
    6,      // vertex lighting (biased vertex normals), one bone, no fog
    7,      // vertex lighting (biased vertex normals), two bones
    7,      // vertex lighting (biased vertex normals), two bones, no fog
    8,      // vertex lighting (biased vertex normals), four bones
    8,      // vertex lighting (biased vertex normals), four bones, no fog

    9,      // pixel lighting (biased vertex normals), one bone
    9,      // pixel lighting (biased vertex normals), one bone, no fog
    10,     // pixel lighting (biased vertex normals), two bones
    10,     // pixel lighting (biased vertex normals), two bones, no fog
    11,     // pixel lighting (biased vertex normals), four bones
    11,     // pixel lighting (biased vertex normals), four bones, no fog
};


template<>
const D3D12_SHADER_BYTECODE EffectBase<SkinnedEffectTraits>::PixelShaderBytecode[] =
{
    { SkinnedEffect_PSSkinnedVertexLighting,      sizeof(SkinnedEffect_PSSkinnedVertexLighting)      },
    { SkinnedEffect_PSSkinnedVertexLightingNoFog, sizeof(SkinnedEffect_PSSkinnedVertexLightingNoFog) },
    { SkinnedEffect_PSSkinnedPixelLighting,       sizeof(SkinnedEffect_PSSkinnedPixelLighting)       },
};


template<>
const int EffectBase<SkinnedEffectTraits>::PixelShaderIndices[] =
{
    0,      // vertex lighting, one bone
    1,      // vertex lighting, one bone, no fog
    0,      // vertex lighting, two bones
    1,      // vertex lighting, two bones, no fog
    0,      // vertex lighting, four bones
    1,      // vertex lighting, four bones, no fog

    2,      // pixel lighting, one bone
    2,      // pixel lighting, one bone, no fog
    2,      // pixel lighting, two bones
    2,      // pixel lighting, two bones, no fog
    2,      // pixel lighting, four bones
    2,      // pixel lighting, four bones, no fog

    0,      // vertex lighting (biased vertex normals), one bone
    1,      // vertex lighting (biased vertex normals), one bone, no fog
    0,      // vertex lighting (biased vertex normals), two bones
    1,      // vertex lighting (biased vertex normals), two bones, no fog
    0,      // vertex lighting (biased vertex normals), four bones
    1,      // vertex lighting (biased vertex normals), four bones, no fog

    2,      // pixel lighting (biased vertex normals), one bone
    2,      // pixel lighting (biased vertex normals), one bone, no fog
    2,      // pixel lighting (biased vertex normals), two bones
    2,      // pixel lighting (biased vertex normals), two bones, no fog
    2,      // pixel lighting (biased vertex normals), four bones
    2,      // pixel lighting (biased vertex normals), four bones, no fog
};


// Global pool of per-device SkinnedEffect resources.
template<>
SharedResourcePool<ID3D12Device*, EffectBase<SkinnedEffectTraits>::DeviceResources> EffectBase<SkinnedEffectTraits>::deviceResourcesPool = {};


// Constructor.
SkinnedEffect::Impl::Impl(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, int weightsPerVertex)
    : EffectBase(device),
    texture{},
    sampler{}
{
    static_assert(_countof(EffectBase<SkinnedEffectTraits>::VertexShaderIndices) == SkinnedEffectTraits::ShaderPermutationCount, "array/max mismatch");
    static_assert(_countof(EffectBase<SkinnedEffectTraits>::VertexShaderBytecode) == SkinnedEffectTraits::VertexShaderCount, "array/max mismatch");
    static_assert(_countof(EffectBase<SkinnedEffectTraits>::PixelShaderBytecode) == SkinnedEffectTraits::PixelShaderCount, "array/max mismatch");
    static_assert(_countof(EffectBase<SkinnedEffectTraits>::PixelShaderIndices) == SkinnedEffectTraits::ShaderPermutationCount, "array/max mismatch");

    if ((weightsPerVertex != 1) &&
        (weightsPerVertex != 2) &&
        (weightsPerVertex != 4))
    {
        DebugTrace("ERROR: SkinnedEffect's weightsPerVertex parameter must be 1, 2, or 4");
        throw std::out_of_range("weightsPerVertex must be 1, 2, or 4");
    }

    lights.InitializeConstants(constants.specularColorAndPower, constants.lightDirection, constants.lightDiffuseColor, constants.lightSpecularColor);

    for (int i = 0; i < MaxBones; i++)
    {
        constants.bones[i][0] = g_XMIdentityR0;
        constants.bones[i][1] = g_XMIdentityR1;
        constants.bones[i][2] = g_XMIdentityR2;
    }

    // Create root signature.
    {
        D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;

        CD3DX12_DESCRIPTOR_RANGE textureSrvDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        CD3DX12_DESCRIPTOR_RANGE textureSamplerDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0);

        CD3DX12_ROOT_PARAMETER rootParameters[RootParameterIndex::RootParameterCount] = {};
        rootParameters[RootParameterIndex::TextureSRV].InitAsDescriptorTable(1, &textureSrvDescriptorRange, D3D12_SHADER_VISIBILITY_PIXEL);
        rootParameters[RootParameterIndex::TextureSampler].InitAsDescriptorTable(1, &textureSamplerDescriptorRange, D3D12_SHADER_VISIBILITY_PIXEL);
        rootParameters[RootParameterIndex::ConstantBuffer].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);

        CD3DX12_ROOT_SIGNATURE_DESC rsigDesc = {};
        rsigDesc.Init(_countof(rootParameters), rootParameters, 0, nullptr, rootSignatureFlags);

        mRootSignature = GetRootSignature(0, rsigDesc);
    }

    assert(mRootSignature != nullptr);

    fog.enabled = (effectFlags & EffectFlags::Fog) != 0;

    if (effectFlags & EffectFlags::VertexColor)
    {
        DebugTrace("ERROR: SkinnedEffect does not implement EffectFlags::VertexColor\n");
        throw std::invalid_argument("SkinnedEffect");
    }

    // Create pipeline state.
    int sp = GetPipelineStatePermutation(
        (effectFlags & EffectFlags::PerPixelLightingBit) != 0,
        weightsPerVertex,
        (effectFlags & EffectFlags::BiasedVertexNormals) != 0);
    assert(sp >= 0 && sp < SkinnedEffectTraits::ShaderPermutationCount);
    _Analysis_assume_(sp >= 0 && sp < SkinnedEffectTraits::ShaderPermutationCount);

    int vi = EffectBase<SkinnedEffectTraits>::VertexShaderIndices[sp];
    assert(vi >= 0 && vi < SkinnedEffectTraits::VertexShaderCount);
    _Analysis_assume_(vi >= 0 && vi < SkinnedEffectTraits::VertexShaderCount);
    int pi = EffectBase<SkinnedEffectTraits>::PixelShaderIndices[sp];
    assert(pi >= 0 && pi < SkinnedEffectTraits::PixelShaderCount);
    _Analysis_assume_(pi >= 0 && pi < SkinnedEffectTraits::PixelShaderCount);

    pipelineDescription.CreatePipelineState(
        device,
        mRootSignature,
        EffectBase<SkinnedEffectTraits>::VertexShaderBytecode[vi],
        EffectBase<SkinnedEffectTraits>::PixelShaderBytecode[pi],
        mPipelineState.GetAddressOf());

    SetDebugObjectName(mPipelineState.Get(), L"SkinnedEffect");
}


int SkinnedEffect::Impl::GetPipelineStatePermutation(bool preferPerPixelLighting, int weightsPerVertex, bool biasedVertexNormals) const
{
    int permutation = 0;

    // Use optimized shaders if fog is disabled.
    if (!fog.enabled)
    {
        permutation += 1;
    }

    // Evaluate 1, 2, or 4 weights per vertex?
    if (weightsPerVertex == 2)
    {
        permutation += 2;
    }
    else if (weightsPerVertex == 4)
    {
        permutation += 4;
    }

    if (preferPerPixelLighting)
    {
        // Do lighting in the pixel shader.
        permutation += 6;
    }

    if (biasedVertexNormals)
    {
        // Compressed normals need to be scaled and biased in the vertex shader.
        permutation += 12;
    }

    return permutation;
}


// Sets our state onto the D3D device.
void SkinnedEffect::Impl::Apply(_In_ ID3D12GraphicsCommandList* commandList)
{
    // Compute derived parameter values.
    matrices.SetConstants(dirtyFlags, constants.worldViewProj);
    fog.SetConstants(dirtyFlags, matrices.worldView, constants.fogVector);
    lights.SetConstants(dirtyFlags, matrices, constants.world, constants.worldInverseTranspose, constants.eyePosition, constants.diffuseColor, constants.emissiveColor, true);

    UpdateConstants();

    // Set the root signature
    commandList->SetGraphicsRootSignature(mRootSignature);

    // Set the texture
    if (!texture.ptr || !sampler.ptr)
    {
        DebugTrace("ERROR: Missing texture or sampler for SkinnedEffect (texture %llu, sampler %llu)\n", texture.ptr, sampler.ptr);
        throw std::exception("SkinnedEffect");
    }

    // **NOTE** If D3D asserts or crashes here, you probably need to call commandList->SetDescriptorHeaps() with the required descriptor heaps.
    commandList->SetGraphicsRootDescriptorTable(RootParameterIndex::TextureSRV, texture);
    commandList->SetGraphicsRootDescriptorTable(RootParameterIndex::TextureSampler, sampler);
    
    // Set constants
    commandList->SetGraphicsRootConstantBufferView(RootParameterIndex::ConstantBuffer, GetConstantBufferGpuAddress());

    // Set the pipeline state
    commandList->SetPipelineState(EffectBase::mPipelineState.Get());
}


// Public constructor.
SkinnedEffect::SkinnedEffect(_In_ ID3D12Device* device, int effectFlags, const EffectPipelineStateDescription& pipelineDescription, int weightsPerVertex)
  : pImpl(std::make_unique<Impl>(device, effectFlags, pipelineDescription, weightsPerVertex))
{
}


// Move constructor.
SkinnedEffect::SkinnedEffect(SkinnedEffect&& moveFrom) noexcept
  : pImpl(std::move(moveFrom.pImpl))
{
}


// Move assignment.
SkinnedEffect& SkinnedEffect::operator= (SkinnedEffect&& moveFrom) noexcept
{
    pImpl = std::move(moveFrom.pImpl);
    return *this;
}


// Public destructor.
SkinnedEffect::~SkinnedEffect()
{
}


// IEffect methods.
void SkinnedEffect::Apply(_In_ ID3D12GraphicsCommandList* commandList)
{
    pImpl->Apply(commandList);
}


// Camera settings.
void XM_CALLCONV SkinnedEffect::SetWorld(FXMMATRIX value)
{
    pImpl->matrices.world = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::WorldInverseTranspose | EffectDirtyFlags::FogVector;
}


void XM_CALLCONV SkinnedEffect::SetView(FXMMATRIX value)
{
    pImpl->matrices.view = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::EyePosition | EffectDirtyFlags::FogVector;
}


void XM_CALLCONV SkinnedEffect::SetProjection(FXMMATRIX value)
{
    pImpl->matrices.projection = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj;
}


void XM_CALLCONV SkinnedEffect::SetMatrices(FXMMATRIX world, CXMMATRIX view, CXMMATRIX projection)
{
    pImpl->matrices.world = world;
    pImpl->matrices.view = view;
    pImpl->matrices.projection = projection;

    pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::WorldInverseTranspose | EffectDirtyFlags::EyePosition | EffectDirtyFlags::FogVector;
}


// Material settings.
void XM_CALLCONV SkinnedEffect::SetDiffuseColor(FXMVECTOR value)
{
    pImpl->lights.diffuseColor = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::MaterialColor;
}


void XM_CALLCONV SkinnedEffect::SetEmissiveColor(FXMVECTOR value)
{
    pImpl->lights.emissiveColor = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::MaterialColor;
}


void XM_CALLCONV SkinnedEffect::SetSpecularColor(FXMVECTOR value)
{
    // Set xyz to new value, but preserve existing w (specular power).
    pImpl->constants.specularColorAndPower = XMVectorSelect(pImpl->constants.specularColorAndPower, value, g_XMSelect1110);

    pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}


void SkinnedEffect::SetSpecularPower(float value)
{
    // Set w to new value, but preserve existing xyz (specular color).
    pImpl->constants.specularColorAndPower = XMVectorSetW(pImpl->constants.specularColorAndPower, value);

    pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}


void SkinnedEffect::DisableSpecular()
{
    // Set specular color to black, power to 1
    // Note: Don't use a power of 0 or the shader will generate strange highlights on non-specular materials

    pImpl->constants.specularColorAndPower = g_XMIdentityR3; 

    pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}


void SkinnedEffect::SetAlpha(float value)
{
    pImpl->lights.alpha = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::MaterialColor;
}


void XM_CALLCONV SkinnedEffect::SetColorAndAlpha(FXMVECTOR value)
{
    pImpl->lights.diffuseColor = value;
    pImpl->lights.alpha = XMVectorGetW(value);

    pImpl->dirtyFlags |= EffectDirtyFlags::MaterialColor;
}


// Light settings.
void XM_CALLCONV SkinnedEffect::SetAmbientLightColor(FXMVECTOR value)
{
    pImpl->lights.ambientLightColor = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::MaterialColor;
}


void SkinnedEffect::SetLightEnabled(int whichLight, bool value)
{
    pImpl->dirtyFlags |= pImpl->lights.SetLightEnabled(whichLight, value, pImpl->constants.lightDiffuseColor, pImpl->constants.lightSpecularColor);
}


void XM_CALLCONV SkinnedEffect::SetLightDirection(int whichLight, FXMVECTOR value)
{
    EffectLights::ValidateLightIndex(whichLight);

    pImpl->constants.lightDirection[whichLight] = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}


void XM_CALLCONV SkinnedEffect::SetLightDiffuseColor(int whichLight, FXMVECTOR value)
{
    pImpl->dirtyFlags |= pImpl->lights.SetLightDiffuseColor(whichLight, value, pImpl->constants.lightDiffuseColor);
}


void XM_CALLCONV SkinnedEffect::SetLightSpecularColor(int whichLight, FXMVECTOR value)
{
    pImpl->dirtyFlags |= pImpl->lights.SetLightSpecularColor(whichLight, value, pImpl->constants.lightSpecularColor);
}


void SkinnedEffect::EnableDefaultLighting()
{
    EffectLights::EnableDefaultLighting(this);
}


// Fog settings.
void SkinnedEffect::SetFogStart(float value)
{
    pImpl->fog.start = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::FogVector;
}


void SkinnedEffect::SetFogEnd(float value)
{
    pImpl->fog.end = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::FogVector;
}


void XM_CALLCONV SkinnedEffect::SetFogColor(FXMVECTOR value)
{
    pImpl->constants.fogColor = value;

    pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}


// Texture settings.
void SkinnedEffect::SetTexture(D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor, D3D12_GPU_DESCRIPTOR_HANDLE samplerDescriptor)
{
    pImpl->texture = srvDescriptor;
    pImpl->sampler = samplerDescriptor;
}


// Animation settings.
void SkinnedEffect::SetBoneTransforms(_In_reads_(count) XMMATRIX const* value, size_t count)
{
    if (count > MaxBones)
        throw std::out_of_range("count parameter out of range");

    auto boneConstant = pImpl->constants.bones;

    for (size_t i = 0; i < count; i++)
    {
        XMMATRIX boneMatrix = XMMatrixTranspose(value[i]);

        boneConstant[i][0] = boneMatrix.r[0];
        boneConstant[i][1] = boneMatrix.r[1];
        boneConstant[i][2] = boneMatrix.r[2];
    }

    pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}


void SkinnedEffect::ResetBoneTransforms()
{
    auto boneConstant = pImpl->constants.bones;

    for(size_t i = 0; i < MaxBones; ++i)
    {
        boneConstant[i][0] = g_XMIdentityR0;
        boneConstant[i][1] = g_XMIdentityR1;
        boneConstant[i][2] = g_XMIdentityR2;
    }

    pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}
