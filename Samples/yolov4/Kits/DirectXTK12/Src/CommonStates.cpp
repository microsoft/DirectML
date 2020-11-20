//--------------------------------------------------------------------------------------
// File: CommonStates.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "CommonStates.h"
#include "DirectXHelpers.h"
#include "DescriptorHeap.h"

using namespace DirectX;

// --------------------------------------------------------------------------
// Blend States
// --------------------------------------------------------------------------

const D3D12_BLEND_DESC CommonStates::Opaque =
{
    FALSE, // AlphaToCoverageEnable
    FALSE, // IndependentBlendEnable
    { {
        FALSE, // BlendEnable
        FALSE, // LogicOpEnable
        D3D12_BLEND_ONE, // SrcBlend
        D3D12_BLEND_ZERO, // DestBlend
        D3D12_BLEND_OP_ADD, // BlendOp
        D3D12_BLEND_ONE, // SrcBlendAlpha
        D3D12_BLEND_ZERO, // DestBlendAlpha
        D3D12_BLEND_OP_ADD, // BlendOpAlpha
        D3D12_LOGIC_OP_NOOP,
        D3D12_COLOR_WRITE_ENABLE_ALL
    } }
};

const D3D12_BLEND_DESC CommonStates::AlphaBlend =
{
    FALSE, // AlphaToCoverageEnable
    FALSE, // IndependentBlendEnable
    { {
        TRUE, // BlendEnable
        FALSE, // LogicOpEnable
        D3D12_BLEND_ONE, // SrcBlend
        D3D12_BLEND_INV_SRC_ALPHA, // DestBlend
        D3D12_BLEND_OP_ADD, // BlendOp
        D3D12_BLEND_ONE, // SrcBlendAlpha
        D3D12_BLEND_INV_SRC_ALPHA, // DestBlendAlpha
        D3D12_BLEND_OP_ADD, // BlendOpAlpha
        D3D12_LOGIC_OP_NOOP,
        D3D12_COLOR_WRITE_ENABLE_ALL
    } }
};

const D3D12_BLEND_DESC CommonStates::Additive =
{
    FALSE, // AlphaToCoverageEnable
    FALSE, // IndependentBlendEnable
    { {
        TRUE, // BlendEnable
        FALSE, // LogicOpEnable
        D3D12_BLEND_SRC_ALPHA, // SrcBlend
        D3D12_BLEND_ONE, // DestBlend
        D3D12_BLEND_OP_ADD, // BlendOp
        D3D12_BLEND_SRC_ALPHA, // SrcBlendAlpha
        D3D12_BLEND_ONE, // DestBlendAlpha
        D3D12_BLEND_OP_ADD, // BlendOpAlpha
        D3D12_LOGIC_OP_NOOP,
        D3D12_COLOR_WRITE_ENABLE_ALL
    } }
};

const D3D12_BLEND_DESC CommonStates::NonPremultiplied =
{
    FALSE, // AlphaToCoverageEnable
    FALSE, // IndependentBlendEnable
    { {
        TRUE, // BlendEnable
        FALSE, // LogicOpEnable
        D3D12_BLEND_SRC_ALPHA, // SrcBlend
        D3D12_BLEND_INV_SRC_ALPHA, // DestBlend
        D3D12_BLEND_OP_ADD, // BlendOp
        D3D12_BLEND_SRC_ALPHA, // SrcBlendAlpha
        D3D12_BLEND_INV_SRC_ALPHA, // DestBlendAlpha
        D3D12_BLEND_OP_ADD, // BlendOpAlpha
        D3D12_LOGIC_OP_NOOP,
        D3D12_COLOR_WRITE_ENABLE_ALL
    } }
};


// --------------------------------------------------------------------------
// Depth-Stencil States
// --------------------------------------------------------------------------

const D3D12_DEPTH_STENCIL_DESC CommonStates::DepthNone =
{
    FALSE, // DepthEnable
    D3D12_DEPTH_WRITE_MASK_ZERO,
    D3D12_COMPARISON_FUNC_LESS_EQUAL, // DepthFunc
    FALSE, // StencilEnable
    D3D12_DEFAULT_STENCIL_READ_MASK,
    D3D12_DEFAULT_STENCIL_WRITE_MASK,
    {
        D3D12_STENCIL_OP_KEEP, // StencilFailOp
        D3D12_STENCIL_OP_KEEP, // StencilDepthFailOp
        D3D12_STENCIL_OP_KEEP, // StencilPassOp
        D3D12_COMPARISON_FUNC_ALWAYS // StencilFunc
    }, // FrontFace
    {
        D3D12_STENCIL_OP_KEEP, // StencilFailOp
        D3D12_STENCIL_OP_KEEP, // StencilDepthFailOp
        D3D12_STENCIL_OP_KEEP, // StencilPassOp
        D3D12_COMPARISON_FUNC_ALWAYS // StencilFunc
    } // BackFace
};

const D3D12_DEPTH_STENCIL_DESC CommonStates::DepthDefault =
{
    TRUE, // DepthEnable
    D3D12_DEPTH_WRITE_MASK_ALL,
    D3D12_COMPARISON_FUNC_LESS_EQUAL, // DepthFunc
    FALSE, // StencilEnable
    D3D12_DEFAULT_STENCIL_READ_MASK,
    D3D12_DEFAULT_STENCIL_WRITE_MASK,
    {
        D3D12_STENCIL_OP_KEEP, // StencilFailOp
        D3D12_STENCIL_OP_KEEP, // StencilDepthFailOp
        D3D12_STENCIL_OP_KEEP, // StencilPassOp
        D3D12_COMPARISON_FUNC_ALWAYS // StencilFunc
    }, // FrontFace
    {
        D3D12_STENCIL_OP_KEEP, // StencilFailOp
        D3D12_STENCIL_OP_KEEP, // StencilDepthFailOp
        D3D12_STENCIL_OP_KEEP, // StencilPassOp
        D3D12_COMPARISON_FUNC_ALWAYS // StencilFunc
    } // BackFace
};

const D3D12_DEPTH_STENCIL_DESC CommonStates::DepthRead =
{
    TRUE, // DepthEnable
    D3D12_DEPTH_WRITE_MASK_ZERO,
    D3D12_COMPARISON_FUNC_LESS_EQUAL, // DepthFunc
    FALSE, // StencilEnable
    D3D12_DEFAULT_STENCIL_READ_MASK,
    D3D12_DEFAULT_STENCIL_WRITE_MASK,
    {
        D3D12_STENCIL_OP_KEEP, // StencilFailOp
        D3D12_STENCIL_OP_KEEP, // StencilDepthFailOp
        D3D12_STENCIL_OP_KEEP, // StencilPassOp
        D3D12_COMPARISON_FUNC_ALWAYS // StencilFunc
    }, // FrontFace
    {
        D3D12_STENCIL_OP_KEEP, // StencilFailOp
        D3D12_STENCIL_OP_KEEP, // StencilDepthFailOp
        D3D12_STENCIL_OP_KEEP, // StencilPassOp
        D3D12_COMPARISON_FUNC_ALWAYS // StencilFunc
    } // BackFace
};


// --------------------------------------------------------------------------
// Rasterizer States
// --------------------------------------------------------------------------

const D3D12_RASTERIZER_DESC CommonStates::CullNone =
{
    D3D12_FILL_MODE_SOLID,
    D3D12_CULL_MODE_NONE,
    FALSE, // FrontCounterClockwise
    D3D12_DEFAULT_DEPTH_BIAS,
    D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
    D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
    TRUE, // DepthClipEnable
    TRUE, // MultisampleEnable
    FALSE, // AntialiasedLineEnable
    0, // ForcedSampleCount
    D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
};

const D3D12_RASTERIZER_DESC CommonStates::CullClockwise =
{
    D3D12_FILL_MODE_SOLID,
    D3D12_CULL_MODE_FRONT,
    FALSE, // FrontCounterClockwise
    D3D12_DEFAULT_DEPTH_BIAS,
    D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
    D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
    TRUE, // DepthClipEnable
    TRUE, // MultisampleEnable
    FALSE, // AntialiasedLineEnable
    0, // ForcedSampleCount
    D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
};

const D3D12_RASTERIZER_DESC CommonStates::CullCounterClockwise =
{
    D3D12_FILL_MODE_SOLID,
    D3D12_CULL_MODE_BACK,
    FALSE, // FrontCounterClockwise
    D3D12_DEFAULT_DEPTH_BIAS,
    D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
    D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
    TRUE, // DepthClipEnable
    TRUE, // MultisampleEnable
    FALSE, // AntialiasedLineEnable
    0, // ForcedSampleCount
    D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
};

const D3D12_RASTERIZER_DESC CommonStates::Wireframe =
{
    D3D12_FILL_MODE_WIREFRAME,
    D3D12_CULL_MODE_NONE,
    FALSE, // FrontCounterClockwise
    D3D12_DEFAULT_DEPTH_BIAS,
    D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
    D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
    TRUE, // DepthClipEnable
    TRUE, // MultisampleEnable
    FALSE, // AntialiasedLineEnable
    0, // ForcedSampleCount
    D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
};


// --------------------------------------------------------------------------
// Static sampler States
// --------------------------------------------------------------------------

const D3D12_STATIC_SAMPLER_DESC CommonStates::StaticPointWrap(unsigned int shaderRegister, D3D12_SHADER_VISIBILITY shaderVisibility, unsigned int registerSpace)
{
    static const D3D12_STATIC_SAMPLER_DESC s_desc = {
        D3D12_FILTER_MIN_MAG_MIP_POINT,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressU
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressV
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressW
        0, // MipLODBias
        D3D12_MAX_MAXANISOTROPY,
        D3D12_COMPARISON_FUNC_NEVER,
        D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        0, // MinLOD
        FLT_MAX, // MaxLOD
        0, // ShaderRegister
        0, // RegisterSpace
        D3D12_SHADER_VISIBILITY_ALL,
    };

    D3D12_STATIC_SAMPLER_DESC desc = s_desc;
    desc.ShaderRegister = shaderRegister;
    desc.ShaderVisibility = shaderVisibility;
    desc.RegisterSpace = registerSpace;
    return desc;
}

const D3D12_STATIC_SAMPLER_DESC CommonStates::StaticPointClamp(unsigned int shaderRegister, D3D12_SHADER_VISIBILITY shaderVisibility, unsigned int registerSpace)
{
    static const D3D12_STATIC_SAMPLER_DESC s_desc = {
        D3D12_FILTER_MIN_MAG_MIP_POINT,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressW
        0, // MipLODBias
        D3D12_MAX_MAXANISOTROPY,
        D3D12_COMPARISON_FUNC_NEVER,
        D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        0, // MinLOD
        FLT_MAX, // MaxLOD
        0, // ShaderRegister
        0, // RegisterSpace
        D3D12_SHADER_VISIBILITY_ALL,
    };

    D3D12_STATIC_SAMPLER_DESC desc = s_desc;
    desc.ShaderRegister = shaderRegister;
    desc.ShaderVisibility = shaderVisibility;
    desc.RegisterSpace = registerSpace;
    return desc;
};
const D3D12_STATIC_SAMPLER_DESC CommonStates::StaticLinearWrap(unsigned int shaderRegister, D3D12_SHADER_VISIBILITY shaderVisibility, unsigned int registerSpace)
{
    static const D3D12_STATIC_SAMPLER_DESC s_desc = {
        D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressU
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressV
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressW
        0, // MipLODBias
        D3D12_MAX_MAXANISOTROPY,
        D3D12_COMPARISON_FUNC_NEVER,
        D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        0, // MinLOD
        FLT_MAX, // MaxLOD
        0, // ShaderRegister
        0, // RegisterSpace
        D3D12_SHADER_VISIBILITY_ALL,
    };

    D3D12_STATIC_SAMPLER_DESC desc = s_desc;
    desc.ShaderRegister = shaderRegister;
    desc.ShaderVisibility = shaderVisibility;
    desc.RegisterSpace = registerSpace;
    return desc;
};

const D3D12_STATIC_SAMPLER_DESC CommonStates::StaticLinearClamp(unsigned int shaderRegister, D3D12_SHADER_VISIBILITY shaderVisibility, unsigned int registerSpace)
{
    static const D3D12_STATIC_SAMPLER_DESC s_desc = {
        D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressW
        0, // MipLODBias
        D3D12_MAX_MAXANISOTROPY,
        D3D12_COMPARISON_FUNC_NEVER,
        D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        0, // MinLOD
        FLT_MAX, // MaxLOD
        0, // ShaderRegister
        0, // RegisterSpace
        D3D12_SHADER_VISIBILITY_ALL,
    };

    D3D12_STATIC_SAMPLER_DESC desc = s_desc;
    desc.ShaderRegister = shaderRegister;
    desc.ShaderVisibility = shaderVisibility;
    desc.RegisterSpace = registerSpace;
    return desc;
};

const D3D12_STATIC_SAMPLER_DESC CommonStates::StaticAnisotropicWrap(unsigned int shaderRegister, D3D12_SHADER_VISIBILITY shaderVisibility, unsigned int registerSpace)
{
    static const D3D12_STATIC_SAMPLER_DESC s_desc = {
        D3D12_FILTER_ANISOTROPIC,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressU
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressV
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressW
        0, // MipLODBias
        D3D12_MAX_MAXANISOTROPY,
        D3D12_COMPARISON_FUNC_NEVER,
        D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        0, // MinLOD
        FLT_MAX, // MaxLOD
        0, // ShaderRegister
        0, // RegisterSpace
        D3D12_SHADER_VISIBILITY_ALL,
    };

    D3D12_STATIC_SAMPLER_DESC desc = s_desc;
    desc.ShaderRegister = shaderRegister;
    desc.ShaderVisibility = shaderVisibility;
    desc.RegisterSpace = registerSpace;
    return desc;
};

const D3D12_STATIC_SAMPLER_DESC CommonStates::StaticAnisotropicClamp(unsigned int shaderRegister, D3D12_SHADER_VISIBILITY shaderVisibility, unsigned int registerSpace)
{
    static const D3D12_STATIC_SAMPLER_DESC s_desc = {
        D3D12_FILTER_ANISOTROPIC,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressW
        0, // MipLODBias
        D3D12_MAX_MAXANISOTROPY,
        D3D12_COMPARISON_FUNC_NEVER,
        D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        0, // MinLOD
        FLT_MAX, // MaxLOD
        0, // ShaderRegister
        0, // RegisterSpace
        D3D12_SHADER_VISIBILITY_ALL,
    };

    D3D12_STATIC_SAMPLER_DESC desc = s_desc;
    desc.ShaderRegister = shaderRegister;
    desc.ShaderVisibility = shaderVisibility;
    desc.RegisterSpace = registerSpace;
    return desc;
};

// --------------------------------------------------------------------------
// Samplers
// --------------------------------------------------------------------------

class CommonStates::Impl
{
public:

    static const D3D12_SAMPLER_DESC SamplerDescs[static_cast<int>(SamplerIndex::Count)];

    Impl(_In_ ID3D12Device* device)
        : mDescriptors(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, static_cast<size_t>(SamplerIndex::Count))
    {
        SetDebugObjectName(mDescriptors.Heap(), L"CommonStates");

        for (size_t i = 0; i < static_cast<size_t>(SamplerIndex::Count); ++i)
        {
            device->CreateSampler(&SamplerDescs[i], mDescriptors.GetCpuHandle(i));
        }
    }

    D3D12_GPU_DESCRIPTOR_HANDLE Get(SamplerIndex i) const
    {
        return mDescriptors.GetGpuHandle(static_cast<size_t>(i));
    }

    ID3D12DescriptorHeap* Heap() const
    {
        return mDescriptors.Heap();
    }

private:
    DescriptorHeap mDescriptors;
};

const D3D12_SAMPLER_DESC CommonStates::Impl::SamplerDescs[] =
{
    // PointWrap
    {
        D3D12_FILTER_MIN_MAG_MIP_POINT,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressU
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressV
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressW
        0, // MipLODBias
        D3D12_MAX_MAXANISOTROPY,
        D3D12_COMPARISON_FUNC_NEVER,
        { 0, 0, 0, 0 }, // BorderColor
        0, // MinLOD
        FLT_MAX // MaxLOD
    },
    // PointClamp
    {
        D3D12_FILTER_MIN_MAG_MIP_POINT,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressW
        0, // MipLODBias
        D3D12_MAX_MAXANISOTROPY,
        D3D12_COMPARISON_FUNC_NEVER,
        { 0, 0, 0, 0 }, // BorderColor
        0, // MinLOD
        FLT_MAX // MaxLOD
    },
    // LinearWrap
    {
        D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressU
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressV
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressW
        0, // MipLODBias
        D3D12_MAX_MAXANISOTROPY,
        D3D12_COMPARISON_FUNC_NEVER,
        { 0, 0, 0, 0 }, // BorderColor
        0, // MinLOD
        FLT_MAX // MaxLOD
    },
    // LinearClamp
    {
        D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressW
        0, // MipLODBias
        D3D12_MAX_MAXANISOTROPY,
        D3D12_COMPARISON_FUNC_NEVER,
        { 0, 0, 0, 0 }, // BorderColor
        0, // MinLOD
        FLT_MAX // MaxLOD
    },
    // AnisotropicWrap
    {
        D3D12_FILTER_ANISOTROPIC,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressU
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressV
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, // AddressW
        0, // MipLODBias
        D3D12_MAX_MAXANISOTROPY,
        D3D12_COMPARISON_FUNC_NEVER,
        { 0, 0, 0, 0 }, // BorderColor
        0, // MinLOD
        FLT_MAX // MaxLOD
    },
    // AnisotropicClamp
    {
        D3D12_FILTER_ANISOTROPIC,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // AddressW
        0, // MipLODBias
        D3D12_MAX_MAXANISOTROPY,
        D3D12_COMPARISON_FUNC_NEVER,
        { 0, 0, 0, 0 }, // BorderColor
        0, // MinLOD
        FLT_MAX // MaxLOD
    }
};


_Use_decl_annotations_
CommonStates::CommonStates(ID3D12Device* device)
{
    pImpl = std::make_unique<Impl>(device);
}

CommonStates::CommonStates(CommonStates&& moveFrom) noexcept
    : pImpl(std::move(moveFrom.pImpl))
{
}

CommonStates::~CommonStates() {}

CommonStates& CommonStates::operator = (CommonStates&& moveFrom) noexcept
{
    pImpl = std::move(moveFrom.pImpl);
    return *this;
}

D3D12_GPU_DESCRIPTOR_HANDLE CommonStates::PointWrap() const { return pImpl->Get(SamplerIndex::PointWrap); }
D3D12_GPU_DESCRIPTOR_HANDLE CommonStates::PointClamp() const { return pImpl->Get(SamplerIndex::PointClamp); }
D3D12_GPU_DESCRIPTOR_HANDLE CommonStates::LinearWrap() const { return pImpl->Get(SamplerIndex::LinearWrap); }
D3D12_GPU_DESCRIPTOR_HANDLE CommonStates::LinearClamp() const { return pImpl->Get(SamplerIndex::LinearClamp); }
D3D12_GPU_DESCRIPTOR_HANDLE CommonStates::AnisotropicWrap() const { return pImpl->Get(SamplerIndex::AnisotropicWrap); }
D3D12_GPU_DESCRIPTOR_HANDLE CommonStates::AnisotropicClamp() const { return pImpl->Get(SamplerIndex::AnisotropicClamp); }

ID3D12DescriptorHeap* CommonStates::Heap() const { return pImpl->Heap(); }
