//--------------------------------------------------------------------------------------
// File: EffectPipelineStateDescription.h
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

#include "RenderTargetState.h"


namespace DirectX
{
    // Pipeline state information for creating effects.
    struct EffectPipelineStateDescription
    {
        EffectPipelineStateDescription(
            _In_opt_ const D3D12_INPUT_LAYOUT_DESC* iinputLayout,
            const D3D12_BLEND_DESC& blend,
            const D3D12_DEPTH_STENCIL_DESC& depthStencil,
            const D3D12_RASTERIZER_DESC& rasterizer,
            const RenderTargetState& renderTarget,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE iprimitiveTopology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            D3D12_INDEX_BUFFER_STRIP_CUT_VALUE istripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED)
            :
            inputLayout{},
            blendDesc(blend),
            depthStencilDesc(depthStencil),
            rasterizerDesc(rasterizer),
            renderTargetState(renderTarget),
            primitiveTopology(iprimitiveTopology),
            stripCutValue(istripCutValue)
        {
            if (iinputLayout)
                this->inputLayout = *iinputLayout;
        }

        EffectPipelineStateDescription(const EffectPipelineStateDescription&) = default;
        EffectPipelineStateDescription& operator=(const EffectPipelineStateDescription&) = default;

        EffectPipelineStateDescription(EffectPipelineStateDescription&&) = default;
        EffectPipelineStateDescription& operator=(EffectPipelineStateDescription&&) = default;

        void CreatePipelineState(
            _In_ ID3D12Device* device,
            _In_ ID3D12RootSignature* rootSignature,
            const D3D12_SHADER_BYTECODE& vertexShader,
            const D3D12_SHADER_BYTECODE& pixelShader,
            _Outptr_ ID3D12PipelineState** pPipelineState) const;

        D3D12_GRAPHICS_PIPELINE_STATE_DESC GetDesc() const
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
            psoDesc.BlendState = blendDesc;
            psoDesc.SampleMask = renderTargetState.sampleMask;
            psoDesc.RasterizerState = rasterizerDesc;
            psoDesc.DepthStencilState = depthStencilDesc;
            psoDesc.InputLayout = inputLayout;
            psoDesc.IBStripCutValue = stripCutValue;
            psoDesc.PrimitiveTopologyType = primitiveTopology;
            psoDesc.NumRenderTargets = renderTargetState.numRenderTargets;
            memcpy_s(psoDesc.RTVFormats, sizeof(psoDesc.RTVFormats), renderTargetState.rtvFormats, sizeof(DXGI_FORMAT) * D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT);
            psoDesc.DSVFormat = renderTargetState.dsvFormat;
            psoDesc.SampleDesc = renderTargetState.sampleDesc;
            psoDesc.NodeMask = renderTargetState.nodeMask;
            return psoDesc;
        }

        uint32_t ComputeHash() const;

        D3D12_INPUT_LAYOUT_DESC             inputLayout;
        D3D12_BLEND_DESC                    blendDesc;
        D3D12_DEPTH_STENCIL_DESC            depthStencilDesc;
        D3D12_RASTERIZER_DESC               rasterizerDesc;
        RenderTargetState                   renderTargetState;
        D3D12_PRIMITIVE_TOPOLOGY_TYPE       primitiveTopology;
        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE  stripCutValue;
    };
}
