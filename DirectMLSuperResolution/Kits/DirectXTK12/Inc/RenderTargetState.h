//--------------------------------------------------------------------------------------
// File: RenderTargetState.h
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


namespace DirectX
{
    // Encapsulates all render target state when creating pipeline state objects
    class RenderTargetState
    {
    public:
        RenderTargetState() noexcept
            : sampleMask(~0U)
            , numRenderTargets(0)
            , rtvFormats{}
            , dsvFormat(DXGI_FORMAT_UNKNOWN)
            , sampleDesc{}
            , nodeMask(0)
        {
        }

        RenderTargetState(const RenderTargetState&) = default;
        RenderTargetState& operator=(const RenderTargetState&) = default;

        RenderTargetState(RenderTargetState&&) = default;
        RenderTargetState& operator=(RenderTargetState&&) = default;

        // Single render target convenience constructor
        RenderTargetState(
            _In_ DXGI_FORMAT rtFormat,
            _In_ DXGI_FORMAT dsFormat)
            : sampleMask(UINT_MAX)
            , numRenderTargets(1)
            , rtvFormats{}
            , dsvFormat(dsFormat)
            , sampleDesc{}
            , nodeMask(0)
        {
            sampleDesc.Count = 1;
            rtvFormats[0] = rtFormat;
        }

        // Convenience constructor converting from DXGI_SWAPCHAIN_DESC
        RenderTargetState(
            _In_ const DXGI_SWAP_CHAIN_DESC* desc,
            _In_ DXGI_FORMAT dsFormat)
            : sampleMask(UINT_MAX)
            , numRenderTargets(1)
            , rtvFormats{}
            , dsvFormat(dsFormat)
            , sampleDesc{}
            , nodeMask(0)
        {
            rtvFormats[0] = desc->BufferDesc.Format;
            sampleDesc = desc->SampleDesc;
        }

        uint32_t            sampleMask;
        uint32_t            numRenderTargets;
        DXGI_FORMAT         rtvFormats[D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT];
        DXGI_FORMAT         dsvFormat;
        DXGI_SAMPLE_DESC    sampleDesc;
        uint32_t            nodeMask;
    };
}
