//--------------------------------------------------------------------------------------
// File: EffectTextureFactory.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#include "pch.h"

#include "Effects.h"
#include "DirectXHelpers.h"
#include "DDSTextureLoader.h"
#include "DescriptorHeap.h"
#include "PlatformHelpers.h"
#include "ResourceUploadBatch.h"
#include "WICTextureLoader.h"

#include <mutex>


using namespace DirectX;
using Microsoft::WRL::ComPtr;


class EffectTextureFactory::Impl
{
public:
    struct TextureCacheEntry
    {
        ComPtr<ID3D12Resource> mResource;
        bool mIsCubeMap;
        size_t slot;

        TextureCacheEntry() noexcept : mIsCubeMap(false), slot(0) {}
    };

    typedef std::map< std::wstring, TextureCacheEntry > TextureCache;

    Impl(
        _In_ ID3D12Device* device,
        ResourceUploadBatch& resourceUploadBatch,
        _In_ ID3D12DescriptorHeap* descriptorHeap)
        : mPath{}
        , mTextureDescriptorHeap(descriptorHeap)
        , mDevice(device)
        , mResourceUploadBatch(resourceUploadBatch)
        , mSharing(true)
        , mForceSRGB(false)
        , mAutoGenMips(false)
    { 
        *mPath = 0; 
    }

    Impl(
        _In_ ID3D12Device* device,
        ResourceUploadBatch& resourceUploadBatch,
        _In_ size_t numDescriptors,
        _In_ D3D12_DESCRIPTOR_HEAP_FLAGS descriptorHeapFlags)
        : mPath{}
        , mTextureDescriptorHeap(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, descriptorHeapFlags, numDescriptors)
        , mDevice(device)
        , mResourceUploadBatch(resourceUploadBatch)
        , mSharing(true)
        , mForceSRGB(false)
        , mAutoGenMips(false)
    {
        SetDebugObjectName(mTextureDescriptorHeap.Heap(), L"EffectTextureFactory");
    }

    size_t CreateTexture(_In_z_ const wchar_t* name, int descriptorSlot);

    void ReleaseCache();
    void SetSharing(bool enabled) { mSharing = enabled; }
    void EnableForceSRGB(bool forceSRGB) { mForceSRGB = forceSRGB; }
    void EnableAutoGenMips(bool generateMips) { mAutoGenMips = generateMips; }

    wchar_t mPath[MAX_PATH];

    ::DescriptorHeap               mTextureDescriptorHeap;
    std::vector<TextureCacheEntry> mResources; // flat list of unique resources so we can index into it

private:
    ComPtr<ID3D12Device>           mDevice;
    ResourceUploadBatch&           mResourceUploadBatch;

    TextureCache                   mTextureCache;

    bool                           mSharing;
    bool                           mForceSRGB;
    bool                           mAutoGenMips;

    std::mutex                     mutex;
};


_Use_decl_annotations_
size_t EffectTextureFactory::Impl::CreateTexture(_In_z_ const wchar_t* name, int descriptorSlot)
{
    if (!name)
        throw std::exception("invalid arguments");

    auto it = mTextureCache.find(name);

    TextureCacheEntry textureEntry = {};

    if (mSharing && it != mTextureCache.end())
    {
        textureEntry = it->second;
    }
    else
    {
        wchar_t fullName[MAX_PATH] = {};
        wcscpy_s(fullName, mPath);
        wcscat_s(fullName, name);

        WIN32_FILE_ATTRIBUTE_DATA fileAttr = {};
        if (!GetFileAttributesExW(fullName, GetFileExInfoStandard, &fileAttr))
        {
            // Try Current Working Directory (CWD)
            wcscpy_s(fullName, name);
            if (!GetFileAttributesExW(fullName, GetFileExInfoStandard, &fileAttr))
            {
                DebugTrace("ERROR: EffectTextureFactory could not find texture file '%ls'\n", name);
                throw std::exception("CreateTexture");
            }
        }

        wchar_t ext[_MAX_EXT];
        _wsplitpath_s(name, nullptr, 0, nullptr, 0, nullptr, 0, ext, _MAX_EXT);

        unsigned int loadFlags = DDS_LOADER_DEFAULT;
        if (mForceSRGB)
            loadFlags |= DDS_LOADER_FORCE_SRGB;
        if (mAutoGenMips)
            loadFlags |= DDS_LOADER_MIP_AUTOGEN;

        static_assert(static_cast<int>(DDS_LOADER_DEFAULT) == static_cast<int>(WIC_LOADER_DEFAULT), "DDS/WIC Load flags mismatch");
        static_assert(static_cast<int>(DDS_LOADER_FORCE_SRGB) == static_cast<int>(WIC_LOADER_FORCE_SRGB), "DDS/WIC Load flags mismatch");
        static_assert(static_cast<int>(DDS_LOADER_MIP_AUTOGEN) == static_cast<int>(WIC_LOADER_MIP_AUTOGEN), "DDS/WIC Load flags mismatch");
        static_assert(static_cast<int>(DDS_LOADER_MIP_RESERVE) == static_cast<int>(WIC_LOADER_MIP_RESERVE), "DDS/WIC Load flags mismatch");

        if (_wcsicmp(ext, L".dds") == 0)
        {
            HRESULT hr = CreateDDSTextureFromFileEx(
                mDevice.Get(),
                mResourceUploadBatch,
                fullName,
                0u,
                D3D12_RESOURCE_FLAG_NONE,
                loadFlags,
                textureEntry.mResource.ReleaseAndGetAddressOf(),
                nullptr,
                &textureEntry.mIsCubeMap);
            if (FAILED(hr))
            {
                DebugTrace("ERROR: CreateDDSTextureFromFile failed (%08X) for '%ls'\n", hr, fullName);
                throw std::exception("CreateDDSTextureFromFile");
            }
        }
        else
        {
            textureEntry.mIsCubeMap = false;

            HRESULT hr = CreateWICTextureFromFileEx(
                mDevice.Get(),
                mResourceUploadBatch,
                fullName,
                0u,
                D3D12_RESOURCE_FLAG_NONE,
                loadFlags,
                textureEntry.mResource.ReleaseAndGetAddressOf());
            if (FAILED(hr))
            {
                DebugTrace("ERROR: CreateWICTextureFromFile failed (%08X) for '%ls'\n", hr, fullName);
                throw std::exception("CreateWICTextureFromFile");
            }
        }

        std::lock_guard<std::mutex> lock(mutex);
        textureEntry.slot = mResources.size();
        if (mSharing)
        {
            TextureCache::value_type v(name, textureEntry);
            mTextureCache.insert(v);
        }
        mResources.push_back(textureEntry);
    }

    assert(textureEntry.mResource != nullptr);

    // bind a new descriptor in slot 
    auto textureDescriptor = mTextureDescriptorHeap.GetCpuHandle(static_cast<size_t>(descriptorSlot));
    DirectX::CreateShaderResourceView(mDevice.Get(), textureEntry.mResource.Get(), textureDescriptor, textureEntry.mIsCubeMap);

    return textureEntry.slot;
}

void EffectTextureFactory::Impl::ReleaseCache()
{
    std::lock_guard<std::mutex> lock(mutex);
    mTextureCache.clear();
}



//--------------------------------------------------------------------------------------
// EffectTextureFactory
//--------------------------------------------------------------------------------------

_Use_decl_annotations_
EffectTextureFactory::EffectTextureFactory(
    ID3D12Device* device,
    ResourceUploadBatch& resourceUploadBatch,
    ID3D12DescriptorHeap* descriptorHeap)
{
    pImpl = std::make_unique<Impl>(device, resourceUploadBatch, descriptorHeap);
}

_Use_decl_annotations_
EffectTextureFactory::EffectTextureFactory(
    ID3D12Device* device,
    ResourceUploadBatch& resourceUploadBatch,
    size_t numDescriptors,
    D3D12_DESCRIPTOR_HEAP_FLAGS descriptorHeapFlags)
{
    pImpl = std::make_unique<Impl>(device, resourceUploadBatch, numDescriptors, descriptorHeapFlags);
}

EffectTextureFactory::~EffectTextureFactory()
{
}


EffectTextureFactory::EffectTextureFactory(EffectTextureFactory&& moveFrom) noexcept
    : pImpl(std::move(moveFrom.pImpl))
{
}

EffectTextureFactory& EffectTextureFactory::operator= (EffectTextureFactory&& moveFrom) noexcept
{
    pImpl = std::move(moveFrom.pImpl);
    return *this;
}

_Use_decl_annotations_
size_t EffectTextureFactory::CreateTexture(_In_z_ const wchar_t* name, int descriptorIndex)
{
    return pImpl->CreateTexture(name, descriptorIndex);
}

void EffectTextureFactory::ReleaseCache()
{
    pImpl->ReleaseCache();
}

void EffectTextureFactory::SetSharing(bool enabled)
{
    pImpl->SetSharing(enabled);
}

void EffectTextureFactory::EnableForceSRGB(bool forceSRGB)
{
    pImpl->EnableForceSRGB(forceSRGB);
}

void EffectTextureFactory::EnableAutoGenMips(bool generateMips)
{
    pImpl->EnableAutoGenMips(generateMips);
}

void EffectTextureFactory::SetDirectory(_In_opt_z_ const wchar_t* path)
{
    if (path && *path != 0)
    {
        wcscpy_s(pImpl->mPath, path);
        size_t len = wcsnlen(pImpl->mPath, MAX_PATH);
        if (len > 0 && len < (MAX_PATH - 1))
        {
            // Ensure it has a trailing slash
            if (pImpl->mPath[len - 1] != L'\\')
            {
                pImpl->mPath[len] = L'\\';
                pImpl->mPath[len + 1] = 0;
            }
        }
    }
    else
        *pImpl->mPath = 0;
}

ID3D12DescriptorHeap* EffectTextureFactory::Heap() const
{
    return pImpl->mTextureDescriptorHeap.Heap();
}

// Shorthand accessors for the descriptor heap
D3D12_CPU_DESCRIPTOR_HANDLE EffectTextureFactory::GetCpuDescriptorHandle(size_t index) const
{
    return pImpl->mTextureDescriptorHeap.GetCpuHandle(index);
}

D3D12_GPU_DESCRIPTOR_HANDLE EffectTextureFactory::GetGpuDescriptorHandle(size_t index) const
{
    return pImpl->mTextureDescriptorHeap.GetGpuHandle(index);
}

size_t EffectTextureFactory::ResourceCount() const
{
    return pImpl->mResources.size();
}

_Use_decl_annotations_
void EffectTextureFactory::GetResource(size_t slot, ID3D12Resource** resource, bool* isCubeMap)
{
    if (slot >= pImpl->mResources.size())
        throw std::exception("Accessing resource out of range.");

    const auto& textureEntry = pImpl->mResources[slot];

    textureEntry.mResource.CopyTo(resource);

    if (isCubeMap)
    {
        *isCubeMap = textureEntry.mIsCubeMap;
    }
}
