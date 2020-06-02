//--------------------------------------------------------------------------------------
// File: VertexTypes.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "VertexTypes.h"

using namespace DirectX;

//--------------------------------------------------------------------------------------
// Vertex struct holding position information.
const D3D12_INPUT_ELEMENT_DESC VertexPosition::InputElements[] =
{
    { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

static_assert(sizeof(VertexPosition) == 12, "Vertex struct/layout mismatch");

const D3D12_INPUT_LAYOUT_DESC VertexPosition::InputLayout =
{
    VertexPosition::InputElements,
    VertexPosition::InputElementCount
};

//--------------------------------------------------------------------------------------
// Vertex struct holding position and color information.
const D3D12_INPUT_ELEMENT_DESC VertexPositionColor::InputElements[] =
{
    { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "COLOR",       0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

static_assert(sizeof(VertexPositionColor) == 28, "Vertex struct/layout mismatch");

const D3D12_INPUT_LAYOUT_DESC VertexPositionColor::InputLayout =
{
    VertexPositionColor::InputElements,
    VertexPositionColor::InputElementCount
};

//--------------------------------------------------------------------------------------
// Vertex struct holding position and texture mapping information.
const D3D12_INPUT_ELEMENT_DESC VertexPositionTexture::InputElements[] =
{
    { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "TEXCOORD",    0, DXGI_FORMAT_R32G32_FLOAT,       0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

static_assert(sizeof(VertexPositionTexture) == 20, "Vertex struct/layout mismatch");

const D3D12_INPUT_LAYOUT_DESC VertexPositionTexture::InputLayout =
{
    VertexPositionTexture::InputElements,
    VertexPositionTexture::InputElementCount
};

//--------------------------------------------------------------------------------------
// Vertex struct holding position and dual texture mapping information.
const D3D12_INPUT_ELEMENT_DESC VertexPositionDualTexture::InputElements[] =
{
    { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "TEXCOORD",    0, DXGI_FORMAT_R32G32_FLOAT,       0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "TEXCOORD",    1, DXGI_FORMAT_R32G32_FLOAT,       0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

static_assert(sizeof(VertexPositionDualTexture) == 28, "Vertex struct/layout mismatch");

const D3D12_INPUT_LAYOUT_DESC VertexPositionDualTexture::InputLayout =
{
    VertexPositionDualTexture::InputElements,
    VertexPositionDualTexture::InputElementCount
};

//--------------------------------------------------------------------------------------
// Vertex struct holding position and normal vector.
const D3D12_INPUT_ELEMENT_DESC VertexPositionNormal::InputElements[] =
{
    { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "NORMAL",      0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

static_assert(sizeof(VertexPositionNormal) == 24, "Vertex struct/layout mismatch");


//--------------------------------------------------------------------------------------
// Vertex struct holding position, color, and texture mapping information.
const D3D12_INPUT_ELEMENT_DESC VertexPositionColorTexture::InputElements[] =
{
    { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "COLOR",       0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "TEXCOORD",    0, DXGI_FORMAT_R32G32_FLOAT,       0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

static_assert(sizeof(VertexPositionColorTexture) == 36, "Vertex struct/layout mismatch");

const D3D12_INPUT_LAYOUT_DESC VertexPositionColorTexture::InputLayout =
{
    VertexPositionColorTexture::InputElements,
    VertexPositionColorTexture::InputElementCount
};


//--------------------------------------------------------------------------------------
// Vertex struct holding position, normal vector, and color information.
const D3D12_INPUT_ELEMENT_DESC VertexPositionNormalColor::InputElements[] =
{
    { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "NORMAL",      0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "COLOR",       0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

static_assert(sizeof(VertexPositionNormalColor) == 40, "Vertex struct/layout mismatch");

const D3D12_INPUT_LAYOUT_DESC VertexPositionNormalColor::InputLayout =
{
    VertexPositionNormalColor::InputElements,
    VertexPositionNormalColor::InputElementCount
};

//--------------------------------------------------------------------------------------
// Vertex struct holding position, normal vector, and texture mapping information.
const D3D12_INPUT_ELEMENT_DESC VertexPositionNormalTexture::InputElements[] =
{
    { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "NORMAL",      0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "TEXCOORD",    0, DXGI_FORMAT_R32G32_FLOAT,       0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

static_assert(sizeof(VertexPositionNormalTexture) == 32, "Vertex struct/layout mismatch");

const D3D12_INPUT_LAYOUT_DESC VertexPositionNormalTexture::InputLayout =
{
    VertexPositionNormalTexture::InputElements,
    VertexPositionNormalTexture::InputElementCount
};

//--------------------------------------------------------------------------------------
// Vertex struct holding position, normal vector, color, and texture mapping information.
const D3D12_INPUT_ELEMENT_DESC VertexPositionNormalColorTexture::InputElements[] =
{
    { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "NORMAL",      0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "COLOR",       0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "TEXCOORD",    0, DXGI_FORMAT_R32G32_FLOAT,       0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

static_assert(sizeof(VertexPositionNormalColorTexture) == 48, "Vertex struct/layout mismatch");

const D3D12_INPUT_LAYOUT_DESC VertexPositionNormalColorTexture::InputLayout =
{
    VertexPositionNormalColorTexture::InputElements,
    VertexPositionNormalColorTexture::InputElementCount
};

//--------------------------------------------------------------------------------------
// VertexPositionNormalTangentColorTexture, VertexPositionNormalTangentColorTextureSkinning are not
// supported for DirectX 12 since they were only present for DGSL
