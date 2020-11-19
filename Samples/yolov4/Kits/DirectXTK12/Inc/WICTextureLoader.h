//--------------------------------------------------------------------------------------
// File: WICTextureLoader.h
//
// Function for loading a WIC image and creating a Direct3D runtime texture for it
// (auto-generating mipmaps if possible)
//
// Note: Assumes application has already called CoInitializeEx
//
// Note these functions are useful for images created as simple 2D textures. For
// more complex resources, DDSTextureLoader is an excellent light-weight runtime loader.
// For a full-featured DDS file reader, writer, and texture processing pipeline see
// the 'Texconv' sample and the 'DirectXTex' library.
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

#include <stdint.h>
#include <memory>


namespace DirectX
{
    enum WIC_LOADER_FLAGS : uint32_t
    {
        WIC_LOADER_DEFAULT      = 0,
        WIC_LOADER_FORCE_SRGB   = 0x1,
        WIC_LOADER_IGNORE_SRGB  = 0x2,
        WIC_LOADER_MIP_AUTOGEN  = 0x4,
        WIC_LOADER_MIP_RESERVE  = 0x8,
    };

    class ResourceUploadBatch;

    // Standard version
    HRESULT __cdecl LoadWICTextureFromMemory(
        _In_ ID3D12Device* d3dDevice,
        _In_reads_bytes_(wicDataSize) const uint8_t* wicData,
        size_t wicDataSize,
        _Outptr_ ID3D12Resource** texture,
        std::unique_ptr<uint8_t[]>& decodedData,
        D3D12_SUBRESOURCE_DATA& subresource,
        size_t maxsize = 0);

    HRESULT __cdecl LoadWICTextureFromFile(
        _In_ ID3D12Device* d3dDevice,
        _In_z_ const wchar_t* szFileName,
        _Outptr_ ID3D12Resource** texture,
        std::unique_ptr<uint8_t[]>& decodedData,
        D3D12_SUBRESOURCE_DATA& subresource,
        size_t maxsize = 0);

    // Standard version with resource upload
    HRESULT __cdecl CreateWICTextureFromMemory(
        _In_ ID3D12Device* d3dDevice,
         ResourceUploadBatch& resourceUpload,
        _In_reads_bytes_(wicDataSize) const uint8_t* wicData,
        size_t wicDataSize,
        _Outptr_ ID3D12Resource** texture,
        bool generateMips = false,
        size_t maxsize = 0);

    HRESULT __cdecl CreateWICTextureFromFile(
        _In_ ID3D12Device* d3dDevice,
        ResourceUploadBatch& resourceUpload,
        _In_z_ const wchar_t* szFileName,
        _Outptr_ ID3D12Resource** texture,
        bool generateMips = false,
        size_t maxsize = 0);

    // Extended version
    HRESULT __cdecl LoadWICTextureFromMemoryEx(
        _In_ ID3D12Device* d3dDevice,
        _In_reads_bytes_(wicDataSize) const uint8_t* wicData,
        size_t wicDataSize,
        size_t maxsize,
        D3D12_RESOURCE_FLAGS resFlags,
        unsigned int loadFlags,
        _Outptr_ ID3D12Resource** texture,
        std::unique_ptr<uint8_t[]>& decodedData,
        D3D12_SUBRESOURCE_DATA& subresource);

    HRESULT __cdecl LoadWICTextureFromFileEx(
        _In_ ID3D12Device* d3dDevice,
        _In_z_ const wchar_t* szFileName,
        size_t maxsize,
        D3D12_RESOURCE_FLAGS resFlags,
        unsigned int loadFlags,
        _Outptr_ ID3D12Resource** texture,
        std::unique_ptr<uint8_t[]>& decodedData,
        D3D12_SUBRESOURCE_DATA& subresource);

    // Extended version with resource upload
    HRESULT __cdecl CreateWICTextureFromMemoryEx(
        _In_ ID3D12Device* d3dDevice,
        ResourceUploadBatch& resourceUpload,
        _In_reads_bytes_(wicDataSize) const uint8_t* wicData,
        size_t wicDataSize,
        size_t maxsize,
        D3D12_RESOURCE_FLAGS resFlags,
        unsigned int loadFlags,
        _Outptr_ ID3D12Resource** texture);

    HRESULT __cdecl CreateWICTextureFromFileEx(
        _In_ ID3D12Device* d3dDevice,
        ResourceUploadBatch& resourceUpload,
        _In_z_ const wchar_t* szFileName,
        size_t maxsize,
        D3D12_RESOURCE_FLAGS resFlags,
        unsigned int loadFlags,
        _Outptr_ ID3D12Resource** texture);
}
