//--------------------------------------------------------------------------------------
// File: VertexTypes.h
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


namespace DirectX
{
    // Vertex struct holding position information.
    struct VertexPosition
    {
        VertexPosition() = default;

        VertexPosition(const VertexPosition&) = default;
        VertexPosition& operator=(const VertexPosition&) = default;

        VertexPosition(VertexPosition&&) = default;
        VertexPosition& operator=(VertexPosition&&) = default;

        VertexPosition(XMFLOAT3 const& iposition)
            : position(iposition)
        { }

        VertexPosition(FXMVECTOR iposition)
        {
            XMStoreFloat3(&this->position, iposition);
        }

        XMFLOAT3 position;

        static const D3D12_INPUT_LAYOUT_DESC InputLayout;

    private:
        static const int InputElementCount = 1;
        static const D3D12_INPUT_ELEMENT_DESC InputElements[InputElementCount];
    };


    // Vertex struct holding position and color information.
    struct VertexPositionColor
    {
        VertexPositionColor() = default;

        VertexPositionColor(const VertexPositionColor&) = default;
        VertexPositionColor& operator=(const VertexPositionColor&) = default;

        VertexPositionColor(VertexPositionColor&&) = default;
        VertexPositionColor& operator=(VertexPositionColor&&) = default;

        VertexPositionColor(XMFLOAT3 const& iposition, XMFLOAT4 const& icolor)
            : position(iposition),
            color(icolor)
        { }

        VertexPositionColor(FXMVECTOR iposition, FXMVECTOR icolor)
        {
            XMStoreFloat3(&this->position, iposition);
            XMStoreFloat4(&this->color, icolor);
        }

        XMFLOAT3 position;
        XMFLOAT4 color;

        static const D3D12_INPUT_LAYOUT_DESC InputLayout;

    private:
        static const int InputElementCount = 2;
        static const D3D12_INPUT_ELEMENT_DESC InputElements[InputElementCount];
    };


    // Vertex struct holding position and texture mapping information.
    struct VertexPositionTexture
    {
        VertexPositionTexture() = default;

        VertexPositionTexture(const VertexPositionTexture&) = default;
        VertexPositionTexture& operator=(const VertexPositionTexture&) = default;

        VertexPositionTexture(VertexPositionTexture&&) = default;
        VertexPositionTexture& operator=(VertexPositionTexture&&) = default;

        VertexPositionTexture(XMFLOAT3 const& iposition, XMFLOAT2 const& itextureCoordinate)
            : position(iposition),
            textureCoordinate(itextureCoordinate)
        { }

        VertexPositionTexture(FXMVECTOR iposition, FXMVECTOR itextureCoordinate)
        {
            XMStoreFloat3(&this->position, iposition);
            XMStoreFloat2(&this->textureCoordinate, itextureCoordinate);
        }

        XMFLOAT3 position;
        XMFLOAT2 textureCoordinate;

        static const D3D12_INPUT_LAYOUT_DESC InputLayout;

    private:
        static const int InputElementCount = 2;
        static const D3D12_INPUT_ELEMENT_DESC InputElements[InputElementCount];
    };


    // Vertex struct holding position and dual texture mapping information.
    struct VertexPositionDualTexture
    {
        VertexPositionDualTexture() = default;

        VertexPositionDualTexture(const VertexPositionDualTexture&) = default;
        VertexPositionDualTexture& operator=(const VertexPositionDualTexture&) = default;

        VertexPositionDualTexture(VertexPositionDualTexture&&) = default;
        VertexPositionDualTexture& operator=(VertexPositionDualTexture&&) = default;

        VertexPositionDualTexture(XMFLOAT3 const& iposition, XMFLOAT2 const& itextureCoordinate0, XMFLOAT2 const& itextureCoordinate1)
            : position(iposition),
            textureCoordinate0(itextureCoordinate0),
            textureCoordinate1(itextureCoordinate1)
        { }

        VertexPositionDualTexture(FXMVECTOR iposition,
            FXMVECTOR itextureCoordinate0,
            FXMVECTOR itextureCoordinate1)
        {
            XMStoreFloat3(&this->position, iposition);
            XMStoreFloat2(&this->textureCoordinate0, itextureCoordinate0);
            XMStoreFloat2(&this->textureCoordinate1, itextureCoordinate1);
        }

        XMFLOAT3 position;
        XMFLOAT2 textureCoordinate0;
        XMFLOAT2 textureCoordinate1;

        static const D3D12_INPUT_LAYOUT_DESC InputLayout;

    private:
        static const int InputElementCount = 3;
        static const D3D12_INPUT_ELEMENT_DESC InputElements[InputElementCount];
    };


    // Vertex struct holding position and normal vector.
    struct VertexPositionNormal
    {
        VertexPositionNormal() = default;

        VertexPositionNormal(const VertexPositionNormal&) = default;
        VertexPositionNormal& operator=(const VertexPositionNormal&) = default;

        VertexPositionNormal(VertexPositionNormal&&) = default;
        VertexPositionNormal& operator=(VertexPositionNormal&&) = default;

        VertexPositionNormal(XMFLOAT3 const& iposition, XMFLOAT3 const& inormal)
            : position(iposition),
            normal(inormal)
        { }

        VertexPositionNormal(FXMVECTOR iposition, FXMVECTOR inormal)
        {
            XMStoreFloat3(&this->position, iposition);
            XMStoreFloat3(&this->normal, inormal);
        }

        XMFLOAT3 position;
        XMFLOAT3 normal;

        static const D3D12_INPUT_LAYOUT_DESC InputLayout;

    private:
        static const int InputElementCount = 2;
        static const D3D12_INPUT_ELEMENT_DESC InputElements[InputElementCount];
    };


    // Vertex struct holding position, color, and texture mapping information.
    struct VertexPositionColorTexture
    {
        VertexPositionColorTexture() = default;

        VertexPositionColorTexture(const VertexPositionColorTexture&) = default;
        VertexPositionColorTexture& operator=(const VertexPositionColorTexture&) = default;

        VertexPositionColorTexture(VertexPositionColorTexture&&) = default;
        VertexPositionColorTexture& operator=(VertexPositionColorTexture&&) = default;

        VertexPositionColorTexture(XMFLOAT3 const& iposition, XMFLOAT4 const& icolor, XMFLOAT2 const& itextureCoordinate)
            : position(iposition),
            color(icolor),
            textureCoordinate(itextureCoordinate)
        { }

        VertexPositionColorTexture(FXMVECTOR iposition, FXMVECTOR icolor, FXMVECTOR itextureCoordinate)
        {
            XMStoreFloat3(&this->position, iposition);
            XMStoreFloat4(&this->color, icolor);
            XMStoreFloat2(&this->textureCoordinate, itextureCoordinate);
        }

        XMFLOAT3 position;
        XMFLOAT4 color;
        XMFLOAT2 textureCoordinate;

        static const D3D12_INPUT_LAYOUT_DESC InputLayout;

    private:
        static const int InputElementCount = 3;
        static const D3D12_INPUT_ELEMENT_DESC InputElements[InputElementCount];
    };


    // Vertex struct holding position, normal vector, and color information.
    struct VertexPositionNormalColor
    {
        VertexPositionNormalColor() = default;

        VertexPositionNormalColor(const VertexPositionNormalColor&) = default;
        VertexPositionNormalColor& operator=(const VertexPositionNormalColor&) = default;

        VertexPositionNormalColor(VertexPositionNormalColor&&) = default;
        VertexPositionNormalColor& operator=(VertexPositionNormalColor&&) = default;
    
        VertexPositionNormalColor(XMFLOAT3 const& iposition, XMFLOAT3 const& inormal, XMFLOAT4 const& icolor)
            : position(iposition),
            normal(inormal),
            color(icolor)
        { }

        VertexPositionNormalColor(FXMVECTOR iposition, FXMVECTOR inormal, FXMVECTOR icolor)
        {
            XMStoreFloat3(&this->position, iposition);
            XMStoreFloat3(&this->normal, inormal);
            XMStoreFloat4(&this->color, icolor);
        }

        XMFLOAT3 position;
        XMFLOAT3 normal;
        XMFLOAT4 color;

        static const D3D12_INPUT_LAYOUT_DESC InputLayout;

    private:
        static const int InputElementCount = 3;
        static const D3D12_INPUT_ELEMENT_DESC InputElements[InputElementCount];
    };


    // Vertex struct holding position, normal vector, and texture mapping information.
    struct VertexPositionNormalTexture
    {
        VertexPositionNormalTexture() = default;

        VertexPositionNormalTexture(const VertexPositionNormalTexture&) = default;
        VertexPositionNormalTexture& operator=(const VertexPositionNormalTexture&) = default;

        VertexPositionNormalTexture(VertexPositionNormalTexture&&) = default;
        VertexPositionNormalTexture& operator=(VertexPositionNormalTexture&&) = default;

        VertexPositionNormalTexture(XMFLOAT3 const& iposition, XMFLOAT3 const& inormal, XMFLOAT2 const& itextureCoordinate)
            : position(iposition),
            normal(inormal),
            textureCoordinate(itextureCoordinate)
        { }

        VertexPositionNormalTexture(FXMVECTOR iposition, FXMVECTOR inormal, FXMVECTOR itextureCoordinate)
        {
            XMStoreFloat3(&this->position, iposition);
            XMStoreFloat3(&this->normal, inormal);
            XMStoreFloat2(&this->textureCoordinate, itextureCoordinate);
        }

        XMFLOAT3 position;
        XMFLOAT3 normal;
        XMFLOAT2 textureCoordinate;

        static const D3D12_INPUT_LAYOUT_DESC InputLayout;

    private:
        static const int InputElementCount = 3;
        static const D3D12_INPUT_ELEMENT_DESC InputElements[InputElementCount];
    };


    // Vertex struct holding position, normal vector, color, and texture mapping information.
    struct VertexPositionNormalColorTexture
    {
        VertexPositionNormalColorTexture() = default;

        VertexPositionNormalColorTexture(const VertexPositionNormalColorTexture&) = default;
        VertexPositionNormalColorTexture& operator=(const VertexPositionNormalColorTexture&) = default;

        VertexPositionNormalColorTexture(VertexPositionNormalColorTexture&&) = default;
        VertexPositionNormalColorTexture& operator=(VertexPositionNormalColorTexture&&) = default;

        VertexPositionNormalColorTexture(XMFLOAT3 const& iposition, XMFLOAT3 const& inormal, XMFLOAT4 const& icolor, XMFLOAT2 const& itextureCoordinate)
            : position(iposition),
            normal(inormal),
            color(icolor),
            textureCoordinate(itextureCoordinate)
        { }

        VertexPositionNormalColorTexture(FXMVECTOR iposition, FXMVECTOR inormal, FXMVECTOR icolor, CXMVECTOR itextureCoordinate)
        {
            XMStoreFloat3(&this->position, iposition);
            XMStoreFloat3(&this->normal, inormal);
            XMStoreFloat4(&this->color, icolor);
            XMStoreFloat2(&this->textureCoordinate, itextureCoordinate);
        }

        XMFLOAT3 position;
        XMFLOAT3 normal;
        XMFLOAT4 color;
        XMFLOAT2 textureCoordinate;

        static const D3D12_INPUT_LAYOUT_DESC InputLayout;

    private:
        static const int InputElementCount = 4;
        static const D3D12_INPUT_ELEMENT_DESC InputElements[InputElementCount];
    };
}
