//--------------------------------------------------------------------------------------
// File: GeometricPrimitive.h
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#pragma once

#include "VertexTypes.h"

#include <memory>
#include <vector>


namespace DirectX
{
    class IEffect;
    class ResourceUploadBatch;

    class GeometricPrimitive
    {
    public:
        GeometricPrimitive(GeometricPrimitive const&) = delete;
        GeometricPrimitive& operator= (GeometricPrimitive const&) = delete;

        virtual ~GeometricPrimitive();

        using VertexType = VertexPositionNormalTexture;

        // Factory methods.
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateCube(float size = 1, bool rhcoords = true, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateBox(const XMFLOAT3& size, bool rhcoords = true, bool invertn = false, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateSphere(float diameter = 1, size_t tessellation = 16, bool rhcoords = true, bool invertn = false, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateGeoSphere(float diameter = 1, size_t tessellation = 3, bool rhcoords = true, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateCylinder(float height = 1, float diameter = 1, size_t tessellation = 32, bool rhcoords = true, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateCone(float diameter = 1, float height = 1, size_t tessellation = 32, bool rhcoords = true, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateTorus(float diameter = 1, float thickness = 0.333f, size_t tessellation = 32, bool rhcoords = true, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateTetrahedron(float size = 1, bool rhcoords = true, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateOctahedron(float size = 1, bool rhcoords = true, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateDodecahedron(float size = 1, bool rhcoords = true, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateIcosahedron(float size = 1, bool rhcoords = true, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateTeapot(float size = 1, size_t tessellation = 8, bool rhcoords = true, _In_opt_ ID3D12Device* device = nullptr);
        static std::unique_ptr<GeometricPrimitive> __cdecl CreateCustom(const std::vector<VertexType>& vertices, const std::vector<uint16_t>& indices, _In_opt_ ID3D12Device* device = nullptr);

        static void __cdecl CreateCube(std::vector<VertexType>& vertices, std::vector<uint16_t>& indices, float size = 1, bool rhcoords = true);
        static void __cdecl CreateBox(std::vector<VertexType>& vertices, std::vector<uint16_t>& indices, const XMFLOAT3& size, bool rhcoords = true, bool invertn = false);
        static void __cdecl CreateSphere(std::vector<VertexType>& vertices, std::vector<uint16_t>& indices, float diameter = 1, size_t tessellation = 16, bool rhcoords = true, bool invertn = false);
        static void __cdecl CreateGeoSphere(std::vector<VertexType>& vertices, std::vector<uint16_t>& indices, float diameter = 1, size_t tessellation = 3, bool rhcoords = true);
        static void __cdecl CreateCylinder(std::vector<VertexType>& vertices, std::vector<uint16_t>& indices, float height = 1, float diameter = 1, size_t tessellation = 32, bool rhcoords = true);
        static void __cdecl CreateCone(std::vector<VertexType>& vertices, std::vector<uint16_t>& indices, float diameter = 1, float height = 1, size_t tessellation = 32, bool rhcoords = true);
        static void __cdecl CreateTorus(std::vector<VertexType>& vertices, std::vector<uint16_t>& indices, float diameter = 1, float thickness = 0.333f, size_t tessellation = 32, bool rhcoords = true);
        static void __cdecl CreateTetrahedron(std::vector<VertexType>& vertices, std::vector<uint16_t>& indices, float size = 1, bool rhcoords = true);
        static void __cdecl CreateOctahedron(std::vector<VertexType>& vertices, std::vector<uint16_t>& indices, float size = 1, bool rhcoords = true);
        static void __cdecl CreateDodecahedron(std::vector<VertexType>& vertices, std::vector<uint16_t>& indices, float size = 1, bool rhcoords = true);
        static void __cdecl CreateIcosahedron(std::vector<VertexType>& vertices, std::vector<uint16_t>& indices, float size = 1, bool rhcoords = true);
        static void __cdecl CreateTeapot(std::vector<VertexType>& vertices, std::vector<uint16_t>& indices, float size = 1, size_t tessellation = 8, bool rhcoords = true);

        // Load VB/IB resources for static geometry.
        void __cdecl LoadStaticBuffers(_In_ ID3D12Device* device, ResourceUploadBatch& resourceUploadBatch);

        // Draw the primitive.
        void __cdecl Draw(_In_ ID3D12GraphicsCommandList* commandList) const;

    private:
        GeometricPrimitive() noexcept(false);

        // Private implementation.
        class Impl;

        std::unique_ptr<Impl> pImpl;
    };
}
