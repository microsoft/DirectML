//--------------------------------------------------------------------------------------
// File: DirectXHelpers.h
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

#include <DirectXMath.h>

#include <initializer_list>
#include <utility>
#include <vector>

#include <wrl/client.h>

#ifndef IID_GRAPHICS_PPV_ARGS
#define IID_GRAPHICS_PPV_ARGS(x) IID_PPV_ARGS(x)
#endif

//
// The d3dx12.h header includes the following helper C++ classes and functions
//  CD3DX12_RECT
//  CD3DX12_BOX
//  CD3DX12_DEPTH_STENCIL_DESC
//  CD3DX12_BLEND_DESC
//  CD3DX12_RASTERIZER_DESC
//  CD3DX12_RESOURCE_ALLOCATION_INFO
//  CD3DX12_HEAP_PROPERTIES
//  CD3DX12_HEAP_DESC
//  CD3DX12_CLEAR_VALUE
//  CD3DX12_RANGE
//  CD3DX12_SHADER_BYTECODE
//  CD3DX12_TILED_RESOURCE_COORDINATE
//  CD3DX12_TILE_REGION_SIZE
//  CD3DX12_SUBRESOURCE_TILING
//  CD3DX12_TILE_SHAPE
//  CD3DX12_RESOURCE_BARRIER
//  CD3DX12_PACKED_MIP_INFO
//  CD3DX12_SUBRESOURCE_FOOTPRINT
//  CD3DX12_TEXTURE_COPY_LOCATION
//  CD3DX12_DESCRIPTOR_RANGE
//  CD3DX12_ROOT_DESCRIPTOR_TABLE
//  CD3DX12_ROOT_CONSTANTS
//  CD3DX12_ROOT_DESCRIPTOR
//  CD3DX12_ROOT_PARAMETER
//  CD3DX12_STATIC_SAMPLER_DESC
//  CD3DX12_ROOT_SIGNATURE_DESC
//  CD3DX12_CPU_DESCRIPTOR_HANDLE
//  CD3DX12_GPU_DESCRIPTOR_HANDLE
//  CD3DX12_RESOURCE_DESC
//  D3D12CalcSubresource
//  D3D12DecomposeSubresource
//  D3D12GetFormatPlaneCount
//  MemcpySubresource
//  GetRequiredIntermediateSize
//  UpdateSubresources
//  D3D12IsLayoutOpaque
//  CommandListCast
// 


namespace DirectX
{
    constexpr D3D12_CPU_DESCRIPTOR_HANDLE D3D12_CPU_DESCRIPTOR_HANDLE_ZERO = {};

    // Creates a shader resource view from an arbitrary resource
    void __cdecl CreateShaderResourceView(
        _In_ ID3D12Device* device,
        _In_ ID3D12Resource* tex,
        D3D12_CPU_DESCRIPTOR_HANDLE srvDescriptor,
        bool isCubeMap = false);

    // Shorthand for creating a root signature
    inline HRESULT CreateRootSignature(
        _In_ ID3D12Device* device,
        _In_ const D3D12_ROOT_SIGNATURE_DESC* rootSignatureDesc,
        _Out_ ID3D12RootSignature** rootSignature)
    {
        Microsoft::WRL::ComPtr<ID3DBlob> pSignature;
        Microsoft::WRL::ComPtr<ID3DBlob> pError;
        HRESULT hr = D3D12SerializeRootSignature(rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, pSignature.GetAddressOf(), pError.GetAddressOf());
        if (SUCCEEDED(hr))
        {
            hr = device->CreateRootSignature(0, pSignature->GetBufferPointer(), pSignature->GetBufferSize(),
                IID_GRAPHICS_PPV_ARGS(rootSignature)
                );
        }
        return hr;
    }

    // Helper for obtaining texture size
    inline XMUINT2 GetTextureSize(_In_ ID3D12Resource* tex)
    {
        const auto desc = tex->GetDesc();
        return XMUINT2(static_cast<uint32_t>(desc.Width), static_cast<uint32_t>(desc.Height));
    }

#if defined(_PIX_H_) || defined(_PIX3_H_)
    // Scoped PIX event.
    class ScopedPixEvent
    {
    public:
        ScopedPixEvent(_In_ ID3D12GraphicsCommandList* pCommandList, UINT64 /*metadata*/, PCWSTR pFormat)
            : mCommandList(pCommandList)
        {
            PIXBeginEvent(pCommandList, 0, pFormat);
        }
        ~ScopedPixEvent()
        {
            PIXEndEvent(mCommandList);
        }

    private:
        ID3D12GraphicsCommandList* mCommandList;
    };
#endif

    // Helper sets a D3D resource name string (used by PIX and debug layer leak reporting).
    template<UINT TNameLength>
    inline void SetDebugObjectName(_In_ ID3D12DeviceChild* resource, _In_z_ const char(&name)[TNameLength])
    {
    #if !defined(NO_D3D12_DEBUG_NAME) && (defined(_DEBUG) || defined(PROFILE))
        wchar_t wname[MAX_PATH];
        int result = MultiByteToWideChar(CP_UTF8, 0, name, TNameLength, wname, MAX_PATH);
        if (result > 0)
        {
            resource->SetName(wname);
        }
    #else
        UNREFERENCED_PARAMETER(resource);
        UNREFERENCED_PARAMETER(name);
    #endif
    }

    template<UINT TNameLength>
    inline void SetDebugObjectName(_In_ ID3D12DeviceChild* resource, _In_z_ const wchar_t(&name)[TNameLength])
    {
    #if !defined(NO_D3D12_DEBUG_NAME) && (defined(_DEBUG) || defined(PROFILE))
        resource->SetName(name);
    #else
        UNREFERENCED_PARAMETER(resource);
        UNREFERENCED_PARAMETER(name);
    #endif
    }

    // Helper for resource barrier.
    inline void TransitionResource(
        _In_ ID3D12GraphicsCommandList* commandList,
        _In_ ID3D12Resource* resource,
        D3D12_RESOURCE_STATES stateBefore,
        D3D12_RESOURCE_STATES stateAfter)
    {
        assert(commandList != nullptr);
        assert(resource != nullptr);

        if (stateBefore == stateAfter)
            return;
        
        D3D12_RESOURCE_BARRIER desc = {};
        desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        desc.Transition.pResource = resource;
        desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        desc.Transition.StateBefore = stateBefore;
        desc.Transition.StateAfter = stateAfter;

        commandList->ResourceBarrier(1, &desc);
    }

    // Helper which applies one or more resources barriers and then reverses them on destruction.
    class ScopedBarrier
    {
    public:
        ScopedBarrier(
            _In_ ID3D12GraphicsCommandList* commandList,
            std::initializer_list<D3D12_RESOURCE_BARRIER> barriers)
            : mCommandList(commandList),
            mBarriers(barriers)
        {
            assert(mBarriers.size() <= UINT32_MAX);

            // Set barriers
            mCommandList->ResourceBarrier(static_cast<UINT>(mBarriers.size()), mBarriers.data());
        }

        ~ScopedBarrier()
        {
            // reverse barrier inputs and outputs
            for (auto& b : mBarriers)
            {
                std::swap(b.Transition.StateAfter, b.Transition.StateBefore);
            }

            // Set barriers
            mCommandList->ResourceBarrier(static_cast<UINT>(mBarriers.size()), mBarriers.data());
        }

    private:
        ID3D12GraphicsCommandList* mCommandList;
        std::vector<D3D12_RESOURCE_BARRIER> mBarriers;
    };

    // Helpers for aligning values by a power of 2
    template<typename T>
    inline T AlignDown(T size, size_t alignment)
    {
        if (alignment > 0)
        {
            assert(((alignment - 1) & alignment) == 0);
            T mask = static_cast<T>(alignment - 1);
            return size & ~mask;
        }
        return size;
    }

    template<typename T>
    inline T AlignUp(T size, size_t alignment)
    {
        if (alignment > 0)
        {
            assert(((alignment - 1) & alignment) == 0);
            T mask = static_cast<T>(alignment - 1);
            return (size + mask) & ~mask;
        }
        return size;
    }
}
