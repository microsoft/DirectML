//--------------------------------------------------------------------------------------
// File: PBREffectFactory.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "Effects.h"
#include "CommonStates.h"
#include "DirectXHelpers.h"
#include "PlatformHelpers.h"
#include "DescriptorHeap.h"

#include <mutex>


using namespace DirectX;
using Microsoft::WRL::ComPtr;

// Internal PBREffectFactory implementation class. Only one of these helpers is allocated
// per D3D device, even if there are multiple public facing PBREffectFactory instances.
class PBREffectFactory::Impl
{
public:
    Impl(_In_ ID3D12Device* device, _In_ ID3D12DescriptorHeap* textureDescriptors, _In_ ID3D12DescriptorHeap* samplerDescriptors)
        : mTextureDescriptors(nullptr)
        , mSamplerDescriptors(nullptr)
        , mDevice(device)
        , mSharing(true)
    { 
        if (textureDescriptors)
            mTextureDescriptors = std::make_unique<DescriptorHeap>(textureDescriptors);
        if (samplerDescriptors)
            mSamplerDescriptors = std::make_unique<DescriptorHeap>(samplerDescriptors);
    }

    std::shared_ptr<IEffect> CreateEffect(
        const EffectInfo& info,
        const EffectPipelineStateDescription& opaquePipelineState,
        const EffectPipelineStateDescription& alphaPipelineState,
        const D3D12_INPUT_LAYOUT_DESC& inputLayout,
        int textureDescriptorOffset,
        int samplerDescriptorOffset);

    void ReleaseCache();
    void SetSharing(bool enabled) { mSharing = enabled; }

    std::unique_ptr<DescriptorHeap> mTextureDescriptors;
    std::unique_ptr<DescriptorHeap> mSamplerDescriptors;

private:
    ComPtr<ID3D12Device> mDevice;

    typedef std::map< std::wstring, std::shared_ptr<IEffect> > EffectCache;

    EffectCache  mEffectCache;

    bool mSharing;

    std::mutex mutex;
};


std::shared_ptr<IEffect> PBREffectFactory::Impl::CreateEffect(
    const EffectInfo& info,
    const EffectPipelineStateDescription& opaquePipelineState,
    const EffectPipelineStateDescription& alphaPipelineState,
    const D3D12_INPUT_LAYOUT_DESC& inputLayoutDesc,
    int textureDescriptorOffset,
    int samplerDescriptorOffset)
{
    if (!mTextureDescriptors)
    {
        DebugTrace("ERROR: PBREffectFactory created without texture descriptor heap!\n");
        throw std::exception("PBREffectFactory");
    }
    if (!mSamplerDescriptors)
    {
        DebugTrace("ERROR: PBREffectFactory created without sampler descriptor heap!\n");
        throw std::exception("PBREffectFactory");
    }

    int albetoTextureIndex = (info.diffuseTextureIndex != -1) ? info.diffuseTextureIndex + textureDescriptorOffset : -1;
    int rmaTextureIndex = (info.specularTextureIndex != -1) ? info.specularTextureIndex + textureDescriptorOffset : -1;
    int normalTextureIndex = (info.normalTextureIndex != -1) ? info.normalTextureIndex + textureDescriptorOffset : -1;
    int emissiveTextureIndex = (info.emissiveTextureIndex != -1) ? info.emissiveTextureIndex + textureDescriptorOffset : -1;
    int samplerIndex = (info.samplerIndex != -1) ? info.samplerIndex + samplerDescriptorOffset : -1;

    // Modify base pipeline state
    EffectPipelineStateDescription derivedPSD = (info.alphaValue < 1.0f) ? alphaPipelineState : opaquePipelineState;
    derivedPSD.inputLayout = inputLayoutDesc;

    // set effect flags for creation
    int effectflags = EffectFlags::Texture;

    if (info.biasedVertexNormals)
    {
        effectflags |= EffectFlags::BiasedVertexNormals;
    }

    std::wstring cacheName;
    if (mSharing && !info.name.empty())
    {
        uint32_t hash = derivedPSD.ComputeHash();
        cacheName = std::to_wstring(effectflags) + info.name + std::to_wstring(hash);

        auto it = mEffectCache.find(cacheName);
        if (mSharing && it != mEffectCache.end())
        {
            return it->second;
        }
    }

    auto effect = std::make_shared<PBREffect>(mDevice.Get(), effectflags, derivedPSD, (emissiveTextureIndex != -1));

    // We don't use EnableDefaultLighting generally for PBR as it uses Image-Based Lighting instead.

    effect->SetAlpha(info.alphaValue);

    effect->SetSurfaceTextures(
        mTextureDescriptors->GetGpuHandle(static_cast<size_t>(albetoTextureIndex)),
        mTextureDescriptors->GetGpuHandle(static_cast<size_t>(normalTextureIndex)),
        mTextureDescriptors->GetGpuHandle(static_cast<size_t>(rmaTextureIndex)),
        mSamplerDescriptors->GetGpuHandle(static_cast<size_t>(samplerIndex)));

    if (emissiveTextureIndex != -1)
    {
        effect->SetEmissiveTexture(mTextureDescriptors->GetGpuHandle(static_cast<size_t>(emissiveTextureIndex)));
    }

    if (mSharing && !info.name.empty())
    {
        std::lock_guard<std::mutex> lock(mutex);
        EffectCache::value_type v(cacheName, effect);
        mEffectCache.insert(v);
    }

    return std::move(effect);
}

void PBREffectFactory::Impl::ReleaseCache()
{
    std::lock_guard<std::mutex> lock(mutex);
    mEffectCache.clear();
}



//--------------------------------------------------------------------------------------
// PBREffectFactory
//--------------------------------------------------------------------------------------

PBREffectFactory::PBREffectFactory(_In_ ID3D12Device* device)
{
    pImpl = std::make_shared<Impl>(device, nullptr, nullptr);
}

PBREffectFactory::PBREffectFactory(_In_ ID3D12DescriptorHeap* textureDescriptors, _In_ ID3D12DescriptorHeap* samplerDescriptors)
{
    if (!textureDescriptors)
    {
        throw std::exception("Texture descriptor heap cannot be null if no device is provided. Use the alternative PBREffectFactory constructor instead.");
    }
    if (!samplerDescriptors)
    {
        throw std::exception("Descriptor heap cannot be null if no device is provided. Use the alternative PBREffectFactory constructor instead.");
    }

    if (textureDescriptors->GetDesc().Type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    {
        throw std::exception("PBREffectFactory::CreateEffect requires a CBV_SRV_UAV descriptor heap for textureDescriptors.");
    }
    if (samplerDescriptors->GetDesc().Type != D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)
    {
        throw std::exception("PBREffectFactory::CreateEffect requires a SAMPLER descriptor heap for samplerDescriptors.");
    }

    ComPtr<ID3D12Device> device;
#if defined(_XBOX_ONE) && defined(_TITLE)
    textureDescriptors->GetDevice(IID_GRAPHICS_PPV_ARGS(device.GetAddressOf()));
#else
    HRESULT hresult = textureDescriptors->GetDevice(IID_PPV_ARGS(device.GetAddressOf()));
    if (FAILED(hresult))
    {
        throw com_exception(hresult);
    }
#endif

    pImpl = std::make_shared<Impl>(device.Get(), textureDescriptors, samplerDescriptors);
}

PBREffectFactory::~PBREffectFactory()
{
}


PBREffectFactory::PBREffectFactory(PBREffectFactory&& moveFrom) noexcept
    : pImpl(std::move(moveFrom.pImpl))
{
}

PBREffectFactory& PBREffectFactory::operator= (PBREffectFactory&& moveFrom) noexcept
{
    pImpl = std::move(moveFrom.pImpl);
    return *this;
}

std::shared_ptr<IEffect> PBREffectFactory::CreateEffect(
    const EffectInfo& info, 
    const EffectPipelineStateDescription& opaquePipelineState,
    const EffectPipelineStateDescription& alphaPipelineState,
    const D3D12_INPUT_LAYOUT_DESC& inputLayout, 
    int textureDescriptorOffset,
    int samplerDescriptorOffset)
{
    return pImpl->CreateEffect(info, opaquePipelineState, alphaPipelineState, inputLayout, textureDescriptorOffset, samplerDescriptorOffset);
}

void PBREffectFactory::ReleaseCache()
{
    pImpl->ReleaseCache();
}

void PBREffectFactory::SetSharing(bool enabled)
{
    pImpl->SetSharing(enabled);
}
