//--------------------------------------------------------------------------------------
// File: GeometricPrimitive.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "GeometricPrimitive.h"

#include "CommonStates.h"
#include "DirectXHelpers.h"
#include "Effects.h"
#include "Geometry.h"
#include "GraphicsMemory.h"
#include "PlatformHelpers.h"
#include "ResourceUploadBatch.h"

using namespace DirectX;
using Microsoft::WRL::ComPtr;

// Internal GeometricPrimitive implementation class.
class GeometricPrimitive::Impl
{
public:
    Impl() noexcept : mIndexCount(0), mVertexBufferView{}, mIndexBufferView{} {}

    void Initialize(const VertexCollection& vertices, const IndexCollection& indices, _In_opt_ ID3D12Device* device);

    void LoadStaticBuffers(_In_ ID3D12Device* device, ResourceUploadBatch& resourceUploadBatch);

    void Draw(_In_ ID3D12GraphicsCommandList* commandList) const;
    
    UINT                        mIndexCount;
    SharedGraphicsResource      mIndexBuffer;
    SharedGraphicsResource      mVertexBuffer;
    ComPtr<ID3D12Resource>      mStaticIndexBuffer;
    ComPtr<ID3D12Resource>      mStaticVertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW    mVertexBufferView;
    D3D12_INDEX_BUFFER_VIEW     mIndexBufferView;
};


// Initializes a geometric primitive instance that will draw the specified vertex and index data.
void GeometricPrimitive::Impl::Initialize(
    const VertexCollection& vertices,
    const IndexCollection& indices,
    _In_opt_ ID3D12Device* device)
{
    if (vertices.size() >= USHRT_MAX)
        throw std::exception("Too many vertices for 16-bit index buffer");

    if (indices.size() > UINT32_MAX)
        throw std::exception("Too many indices");

    // Vertex data
    uint64_t sizeInBytes = uint64_t(vertices.size()) * sizeof(vertices[0]);
    if (sizeInBytes > uint64_t(D3D12_REQ_RESOURCE_SIZE_IN_MEGABYTES_EXPRESSION_A_TERM * 1024u * 1024u))
        throw std::exception("VB too large for DirectX 12");

    auto vertSizeBytes = static_cast<size_t>(sizeInBytes);

    mVertexBuffer = GraphicsMemory::Get(device).Allocate(vertSizeBytes);

    auto verts = reinterpret_cast<const uint8_t*>(vertices.data());
    memcpy(mVertexBuffer.Memory(), verts, vertSizeBytes);

    // Index data
    sizeInBytes = uint64_t(indices.size()) * sizeof(indices[0]);
    if (sizeInBytes > uint64_t(D3D12_REQ_RESOURCE_SIZE_IN_MEGABYTES_EXPRESSION_A_TERM * 1024u * 1024u))
        throw std::exception("IB too large for DirectX 12");

    auto indSizeBytes = static_cast<size_t>(sizeInBytes);

    mIndexBuffer = GraphicsMemory::Get(device).Allocate(indSizeBytes);

    auto ind = reinterpret_cast<const uint8_t*>(indices.data());
    memcpy(mIndexBuffer.Memory(), ind, indSizeBytes);

    // Record index count for draw
    mIndexCount = static_cast<UINT>(indices.size());

    // Create views
    mVertexBufferView.BufferLocation = mVertexBuffer.GpuAddress();
    mVertexBufferView.StrideInBytes = static_cast<UINT>(sizeof(VertexCollection::value_type));
    mVertexBufferView.SizeInBytes = static_cast<UINT>(mVertexBuffer.Size());

    mIndexBufferView.BufferLocation = mIndexBuffer.GpuAddress();
    mIndexBufferView.SizeInBytes = static_cast<UINT>(mIndexBuffer.Size());
    mIndexBufferView.Format = DXGI_FORMAT_R16_UINT;
}


// Load VB/IB resources for static geometry.
_Use_decl_annotations_
void GeometricPrimitive::Impl::LoadStaticBuffers(
    ID3D12Device* device,
    ResourceUploadBatch& resourceUploadBatch)
{
    CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_DEFAULT);

    // Convert dynamic VB to static VB
    if (!mStaticVertexBuffer)
    {
        assert(mVertexBuffer);

        auto desc = CD3DX12_RESOURCE_DESC::Buffer(mVertexBuffer.Size());

        ThrowIfFailed(device->CreateCommittedResource(
            &heapProperties,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(mStaticVertexBuffer.GetAddressOf())
        ));

        SetDebugObjectName(mStaticVertexBuffer.Get(), L"GeometricPrimitive");

        resourceUploadBatch.Upload(mStaticVertexBuffer.Get(), mVertexBuffer);

        resourceUploadBatch.Transition(mStaticVertexBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

        // Update view
        mVertexBufferView.BufferLocation = mStaticVertexBuffer->GetGPUVirtualAddress();

        mVertexBuffer.Reset();
    }

    // Convert dynamic IB to static IB
    if (!mStaticIndexBuffer)
    {
        assert(mIndexBuffer);

        auto desc = CD3DX12_RESOURCE_DESC::Buffer(mIndexBuffer.Size());

        ThrowIfFailed(device->CreateCommittedResource(
            &heapProperties,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(mStaticIndexBuffer.GetAddressOf())
        ));

        SetDebugObjectName(mStaticIndexBuffer.Get(), L"GeometricPrimitive");

        resourceUploadBatch.Upload(mStaticIndexBuffer.Get(), mIndexBuffer);

        resourceUploadBatch.Transition(mStaticIndexBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER);

        // Update view
        mIndexBufferView.BufferLocation = mStaticIndexBuffer->GetGPUVirtualAddress();

        mIndexBuffer.Reset();
    }
}


// Draws the primitive.
_Use_decl_annotations_
void GeometricPrimitive::Impl::Draw(ID3D12GraphicsCommandList* commandList) const
{
    commandList->IASetVertexBuffers(0, 1, &mVertexBufferView);
    commandList->IASetIndexBuffer(&mIndexBufferView);
    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    commandList->DrawIndexedInstanced(mIndexCount, 1, 0, 0, 0);
}

//--------------------------------------------------------------------------------------
// GeometricPrimitive
//--------------------------------------------------------------------------------------

// Constructor.
GeometricPrimitive::GeometricPrimitive() noexcept(false)
    : pImpl(std::make_unique<Impl>())
{
}


// Destructor.
GeometricPrimitive::~GeometricPrimitive()
{
}


// Public entrypoints.
_Use_decl_annotations_
void GeometricPrimitive::LoadStaticBuffers(ID3D12Device* device, ResourceUploadBatch& resourceUploadBatch)
{
    pImpl->LoadStaticBuffers(device, resourceUploadBatch);
}


_Use_decl_annotations_
void GeometricPrimitive::Draw(ID3D12GraphicsCommandList* commandList) const
{
    pImpl->Draw(commandList);
}


//--------------------------------------------------------------------------------------
// Cube (aka a Hexahedron) or Box
//--------------------------------------------------------------------------------------

// Creates a cube primitive.
std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateCube(
    float size,
    bool rhcoords,
    _In_opt_ ID3D12Device* device)
{
    VertexCollection vertices;
    IndexCollection indices;
    ComputeBox(vertices, indices, XMFLOAT3(size, size, size), rhcoords, false);

    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}

void GeometricPrimitive::CreateCube(
    std::vector<VertexType>& vertices,
    std::vector<uint16_t>& indices,
    float size,
    bool rhcoords)
{
    ComputeBox(vertices, indices, XMFLOAT3(size, size, size), rhcoords, false);
}


// Creates a box primitive.
std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateBox(
    const XMFLOAT3& size,
    bool rhcoords,
    bool invertn,
    _In_opt_ ID3D12Device* device)
{
    VertexCollection vertices;
    IndexCollection indices;
    ComputeBox(vertices, indices, size, rhcoords, invertn);

    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}

void GeometricPrimitive::CreateBox(
    std::vector<VertexType>& vertices,
    std::vector<uint16_t>& indices,
    const XMFLOAT3& size,
    bool rhcoords,
    bool invertn)
{
    ComputeBox(vertices, indices, size, rhcoords, invertn);
}


//--------------------------------------------------------------------------------------
// Sphere
//--------------------------------------------------------------------------------------

// Creates a sphere primitive.
std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateSphere(
    float diameter,
    size_t tessellation,
    bool rhcoords,
    bool invertn,
    _In_opt_ ID3D12Device* device)
{
    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    VertexCollection vertices;
    IndexCollection indices;

    ComputeSphere(vertices, indices, diameter, tessellation, rhcoords, invertn);

    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}

void GeometricPrimitive::CreateSphere(
    std::vector<VertexType>& vertices,
    std::vector<uint16_t>& indices, 
    float diameter,
    size_t tessellation,
    bool rhcoords,
    bool invertn)
{
    ComputeSphere(vertices, indices, diameter, tessellation, rhcoords, invertn);
}


//--------------------------------------------------------------------------------------
// Geodesic sphere
//--------------------------------------------------------------------------------------

// Creates a geosphere primitive.
std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateGeoSphere(
    float diameter,
    size_t tessellation,
    bool rhcoords,
    _In_opt_ ID3D12Device* device)
{
    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    VertexCollection vertices;
    IndexCollection indices;
    ComputeGeoSphere(vertices, indices, diameter, tessellation, rhcoords);

    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}

void GeometricPrimitive::CreateGeoSphere(
    std::vector<VertexType>& vertices,
    std::vector<uint16_t>& indices,
    float diameter,
    size_t tessellation,
    bool rhcoords)
{
    ComputeGeoSphere(vertices, indices, diameter, tessellation, rhcoords);
}


//--------------------------------------------------------------------------------------
// Cylinder / Cone
//--------------------------------------------------------------------------------------

// Creates a cylinder primitive.
std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateCylinder(
    float height,
    float diameter,
    size_t tessellation,
    bool rhcoords,
    _In_opt_ ID3D12Device* device)
{
    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    VertexCollection vertices;
    IndexCollection indices;
    ComputeCylinder(vertices, indices, height, diameter, tessellation, rhcoords);

    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}

void GeometricPrimitive::CreateCylinder(
    std::vector<VertexType>& vertices,
    std::vector<uint16_t>& indices,
    float height, float diameter,
    size_t tessellation,
    bool rhcoords)
{
    ComputeCylinder(vertices, indices, height, diameter, tessellation, rhcoords);
}


// Creates a cone primitive.
std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateCone(
    float diameter,
    float height,
    size_t tessellation,
    bool rhcoords,
    _In_opt_ ID3D12Device* device)
{
    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    VertexCollection vertices;
    IndexCollection indices;
    ComputeCone(vertices, indices, diameter, height, tessellation, rhcoords);

    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}

void GeometricPrimitive::CreateCone(
    std::vector<VertexType>& vertices,
    std::vector<uint16_t>& indices,
    float diameter,
    float height, 
    size_t tessellation,
    bool rhcoords)
{
    ComputeCone(vertices, indices, diameter, height, tessellation, rhcoords);
}


//--------------------------------------------------------------------------------------
// Torus
//--------------------------------------------------------------------------------------

// Creates a torus primitive.
std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateTorus(
    float diameter,
    float thickness,
    size_t tessellation,
    bool rhcoords,
    _In_opt_ ID3D12Device* device)
{	
    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    VertexCollection vertices;
    IndexCollection indices;
    ComputeTorus(vertices, indices, diameter, thickness, tessellation, rhcoords);

    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}

void GeometricPrimitive::CreateTorus(
    std::vector<VertexType>& vertices,
    std::vector<uint16_t>& indices,
    float diameter,
    float thickness,
    size_t tessellation,
    bool rhcoords)
{
    ComputeTorus(vertices, indices, diameter, thickness, tessellation, rhcoords);
}


//--------------------------------------------------------------------------------------
// Tetrahedron
//--------------------------------------------------------------------------------------

std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateTetrahedron(
    float size,
    bool rhcoords,
    _In_opt_ ID3D12Device* device)
{
    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    VertexCollection vertices;
    IndexCollection indices;
    ComputeTetrahedron(vertices, indices, size, rhcoords);

    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}

void GeometricPrimitive::CreateTetrahedron(
    std::vector<VertexType>& vertices,
    std::vector<uint16_t>& indices,
    float size,
    bool rhcoords)
{
    ComputeTetrahedron(vertices, indices, size, rhcoords);
}


//--------------------------------------------------------------------------------------
// Octahedron
//--------------------------------------------------------------------------------------

std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateOctahedron(
    float size,
    bool rhcoords,
    _In_opt_ ID3D12Device* device)
{
    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    VertexCollection vertices;
    IndexCollection indices;
    ComputeOctahedron(vertices, indices, size, rhcoords);

    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}

void GeometricPrimitive::CreateOctahedron(
    std::vector<VertexType>& vertices,
    std::vector<uint16_t>& indices,
    float size,
    bool rhcoords)
{
    ComputeOctahedron(vertices, indices, size, rhcoords);
}


//--------------------------------------------------------------------------------------
// Dodecahedron
//--------------------------------------------------------------------------------------

std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateDodecahedron(
    float size,
    bool rhcoords,
    _In_opt_ ID3D12Device* device)
{
    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    VertexCollection vertices;
    IndexCollection indices;
    ComputeDodecahedron(vertices, indices, size, rhcoords);

    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}

void GeometricPrimitive::CreateDodecahedron(
    std::vector<VertexType>& vertices,
    std::vector<uint16_t>& indices,
    float size,
    bool rhcoords)
{
    ComputeDodecahedron(vertices, indices, size, rhcoords);
}


//--------------------------------------------------------------------------------------
// Icosahedron
//--------------------------------------------------------------------------------------

std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateIcosahedron(
    float size,
    bool rhcoords,
    _In_opt_ ID3D12Device* device)
{
    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    VertexCollection vertices;
    IndexCollection indices;
    ComputeIcosahedron(vertices, indices, size, rhcoords);

    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}

void GeometricPrimitive::CreateIcosahedron(
    std::vector<VertexType>& vertices,
    std::vector<uint16_t>& indices,
    float size,
    bool rhcoords)
{
    ComputeIcosahedron(vertices, indices, size, rhcoords);
}


//--------------------------------------------------------------------------------------
// Teapot
//--------------------------------------------------------------------------------------

// Creates a teapot primitive.
std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateTeapot(
    float size,
    size_t tessellation,
    bool rhcoords,
    _In_opt_ ID3D12Device* device)
{
    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    VertexCollection vertices;
    IndexCollection indices;
    ComputeTeapot(vertices, indices, size, tessellation, rhcoords);

    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}

void GeometricPrimitive::CreateTeapot(
    std::vector<VertexType>& vertices,
    std::vector<uint16_t>& indices,
    float size, size_t tessellation,
    bool rhcoords)
{
    ComputeTeapot(vertices, indices, size, tessellation, rhcoords);
}


//--------------------------------------------------------------------------------------
// Custom
//--------------------------------------------------------------------------------------

std::unique_ptr<GeometricPrimitive> GeometricPrimitive::CreateCustom(
    const std::vector<VertexType>& vertices,
    const std::vector<uint16_t>& indices,
    _In_opt_ ID3D12Device* device)
{
    // Extra validation
    if (vertices.empty() || indices.empty())
        throw std::exception("Requires both vertices and indices");

    if (indices.size() % 3)
        throw std::exception("Expected triangular faces");

    size_t nVerts = vertices.size();
    if (nVerts >= USHRT_MAX)
        throw std::exception("Too many vertices for 16-bit index buffer");

    for (auto it = indices.cbegin(); it != indices.cend(); ++it)
    {
        if (*it >= nVerts)
        {
            throw std::exception("Index not in vertices list");
        }
    }
    // Create the primitive object.
    std::unique_ptr<GeometricPrimitive> primitive(new GeometricPrimitive());

    // copy geometry
    primitive->pImpl->Initialize(vertices, indices, device);

    return primitive;
}
