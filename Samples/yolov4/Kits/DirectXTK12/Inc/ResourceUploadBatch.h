//--------------------------------------------------------------------------------------
// File: ResourceUploadBatch.h
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

#include <future>
#include <memory>

#include "GraphicsMemory.h"


namespace DirectX
{
    // Has a command list of it's own so it can upload at any time.
    class ResourceUploadBatch
    {
    public:
        explicit ResourceUploadBatch(_In_ ID3D12Device* device);
        ResourceUploadBatch(ResourceUploadBatch&& moveFrom) noexcept;
        ResourceUploadBatch& operator= (ResourceUploadBatch&& moveFrom) noexcept;

        ResourceUploadBatch(ResourceUploadBatch const&) = delete;
        ResourceUploadBatch& operator= (ResourceUploadBatch const&) = delete;

        virtual ~ResourceUploadBatch();

        // Call this before your multiple calls to Upload.
        void __cdecl Begin();

        // Asynchronously uploads a resource. The memory in subRes is copied.
        // The resource must be in the COPY_DEST state.
        void __cdecl Upload(
            _In_ ID3D12Resource* resource,
            uint32_t subresourceIndexStart,
            _In_reads_(numSubresources) D3D12_SUBRESOURCE_DATA* subRes,
            uint32_t numSubresources);

        void __cdecl Upload(
            _In_ ID3D12Resource* resource,
            const SharedGraphicsResource& buffer
            );

        // Asynchronously generate mips from a resource.
        // Resource must be in the PIXEL_SHADER_RESOURCE state
        void __cdecl GenerateMips(_In_ ID3D12Resource* resource);

        // Transition a resource once you're done with it
        void __cdecl Transition(
            _In_ ID3D12Resource* resource,
            D3D12_RESOURCE_STATES stateBefore,
            D3D12_RESOURCE_STATES stateAfter);

        // Submits all the uploads to the driver.
        // No more uploads can happen after this call until Begin is called again.
        // This returns a handle to an event that can be waited on.
        std::future<void> __cdecl End(
            _In_ ID3D12CommandQueue* commandQueue);

        // Validates if the given DXGI format is supported for autogen mipmaps
        bool __cdecl IsSupportedForGenerateMips(DXGI_FORMAT format);

    private:
        // Private implementation.
        class Impl;

        std::unique_ptr<Impl> pImpl;
    };
}
