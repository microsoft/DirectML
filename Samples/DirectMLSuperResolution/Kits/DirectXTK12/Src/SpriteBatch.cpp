//--------------------------------------------------------------------------------------
// File: SpriteBatch.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#include "pch.h"

#include "SpriteBatch.h"
#include "CommonStates.h"
#include "VertexTypes.h"
#include "SharedResourcePool.h"
#include "AlignedNew.h"
#include "ResourceUploadBatch.h"
#include "GraphicsMemory.h"
#include "DirectXHelpers.h"

using namespace DirectX;
using Microsoft::WRL::ComPtr;

namespace
{
    // Include the precompiled shader code.
    #if defined(_XBOX_ONE) && defined(_TITLE)
    #include "Shaders/Compiled/XboxOneSpriteEffect_SpriteVertexShader.inc"
    #include "Shaders/Compiled/XboxOneSpriteEffect_SpritePixelShader.inc"
    #include "Shaders/Compiled/XboxOneSpriteEffect_SpriteVertexShaderHeap.inc"
    #include "Shaders/Compiled/XboxOneSpriteEffect_SpritePixelShaderHeap.inc"
    #else
    #include "Shaders/Compiled/SpriteEffect_SpriteVertexShader.inc"
    #include "Shaders/Compiled/SpriteEffect_SpritePixelShader.inc"
    #include "Shaders/Compiled/SpriteEffect_SpriteVertexShaderHeap.inc"
    #include "Shaders/Compiled/SpriteEffect_SpritePixelShaderHeap.inc"
    #endif

    inline bool operator != (D3D12_GPU_DESCRIPTOR_HANDLE a, D3D12_GPU_DESCRIPTOR_HANDLE b)
    {
        return a.ptr != b.ptr;
    }
    inline bool operator < (D3D12_GPU_DESCRIPTOR_HANDLE a, D3D12_GPU_DESCRIPTOR_HANDLE b)
    {
        return a.ptr < b.ptr;
    }

    // Helper converts a RECT to XMVECTOR.
    inline XMVECTOR LoadRect(_In_ RECT const* rect)
    {
        XMVECTOR v = XMLoadInt4(reinterpret_cast<uint32_t const*>(rect));

        v = XMConvertVectorIntToFloat(v, 0);

        // Convert right/bottom to width/height.
        v = XMVectorSubtract(v, XMVectorPermute<0, 1, 4, 5>(g_XMZero, v));

        return v;
    }
}

// Internal SpriteBatch implementation class.
__declspec(align(16)) class SpriteBatch::Impl : public AlignedNew<SpriteBatch::Impl>
{
public:
    Impl(_In_ ID3D12Device* device,
         ResourceUploadBatch& upload,
         const SpriteBatchPipelineStateDescription& psoDesc,
         const D3D12_VIEWPORT* viewport);

    void XM_CALLCONV Begin(
        _In_ ID3D12GraphicsCommandList* commandList,
        SpriteSortMode sortMode = SpriteSortMode_Deferred,
        FXMMATRIX transformMatrix = MatrixIdentity);
    void End();

    void XM_CALLCONV Draw(
        D3D12_GPU_DESCRIPTOR_HANDLE texture,
        XMUINT2 const& textureSize,
        FXMVECTOR destination,
        _In_opt_ RECT const* sourceRectangle,
        FXMVECTOR color,
        FXMVECTOR originRotationDepth,
        unsigned int flags);

    // Info about a single sprite that is waiting to be drawn.
    __declspec(align(16)) struct SpriteInfo : public AlignedNew<SpriteInfo>
    {
        XMFLOAT4A source;
        XMFLOAT4A destination;
        XMFLOAT4A color;
        XMFLOAT4A originRotationDepth;
        D3D12_GPU_DESCRIPTOR_HANDLE texture;
        XMVECTOR textureSize;
        unsigned int flags;

        // Combine values from the public SpriteEffects enum with these internal-only flags.
        static const unsigned int SourceInTexels = 4;
        static const unsigned int DestSizeInPixels = 8;

        static_assert((SpriteEffects_FlipBoth & (SourceInTexels | DestSizeInPixels)) == 0, "Flag bits must not overlap");
    };

    DXGI_MODE_ROTATION mRotation;

    bool mSetViewport;
    D3D12_VIEWPORT mViewPort;

private:
    // Implementation helper methods.
    void GrowSpriteQueue();
    void PrepareForRendering();
    void FlushBatch();
    void SortSprites();
    void GrowSortedSprites();

    void RenderBatch(D3D12_GPU_DESCRIPTOR_HANDLE texture, XMVECTOR textureSize, _In_reads_(count) SpriteInfo const* const* sprites, size_t count);

    static void XM_CALLCONV RenderSprite(_In_ SpriteInfo const* sprite,
        _Out_writes_(VerticesPerSprite) VertexPositionColorTexture* vertices,
        FXMVECTOR textureSize,
        FXMVECTOR inverseTextureSize);

    XMMATRIX GetViewportTransform(_In_ DXGI_MODE_ROTATION rotation);

    // Constants.
    static const size_t MaxBatchSize = 2048;
    static const size_t MinBatchSize = 128;
    static const size_t InitialQueueSize = 64;
    static const size_t VerticesPerSprite = 4;
    static const size_t IndicesPerSprite = 6;

    //
    // The following functions and members are used to create the default pipeline state objects.
    //
    static const D3D12_SHADER_BYTECODE s_DefaultVertexShaderByteCodeStatic;
    static const D3D12_SHADER_BYTECODE s_DefaultPixelShaderByteCodeStatic;
    static const D3D12_SHADER_BYTECODE s_DefaultVertexShaderByteCodeHeap;
    static const D3D12_SHADER_BYTECODE s_DefaultPixelShaderByteCodeHeap;
    static const D3D12_INPUT_LAYOUT_DESC s_DefaultInputLayoutDesc;


    // Queue of sprites waiting to be drawn.
    std::unique_ptr<SpriteInfo[]> mSpriteQueue;

    size_t mSpriteQueueCount;
    size_t mSpriteQueueArraySize;


    // To avoid needlessly copying around bulky SpriteInfo structures, we leave that
    // actual data alone and just sort this array of pointers instead. But we want contiguous
    // memory for cache efficiency, so these pointers are just shortcuts into the single
    // mSpriteQueue array, and we take care to keep them in order when sorting is disabled.
    std::vector<SpriteInfo const*> mSortedSprites;


    // Mode settings from the last Begin call.
    bool mInBeginEndPair;

    SpriteSortMode mSortMode;
    ComPtr<ID3D12PipelineState> mPSO;
    ComPtr<ID3D12RootSignature> mRootSignature;
    D3D12_GPU_DESCRIPTOR_HANDLE mSampler;
    XMMATRIX mTransformMatrix;
    ComPtr<ID3D12GraphicsCommandList> mCommandList;

    // Batched data
    GraphicsResource mVertexSegment;
    size_t mVertexPageSize;
    size_t mSpriteCount;
    GraphicsResource mConstantBuffer;

    enum RootParameterIndex
    {
        TextureSRV,
        ConstantBuffer,
        TextureSampler,
        RootParameterCount
    };

    // Only one of these helpers is allocated per D3D device, even if there are multiple SpriteBatch instances.
    struct DeviceResources
    {
        DeviceResources(_In_ ID3D12Device* device, ResourceUploadBatch& upload);

        ComPtr<ID3D12Resource> indexBuffer;
        D3D12_INDEX_BUFFER_VIEW indexBufferView;
        ComPtr<ID3D12RootSignature> rootSignatureStatic;
        ComPtr<ID3D12RootSignature> rootSignatureHeap; 
        ID3D12Device* mDevice;

    private:
        void CreateIndexBuffer(_In_ ID3D12Device* device, ResourceUploadBatch& upload);
        void CreateRootSignatures(_In_ ID3D12Device* device);

        static std::vector<short> CreateIndexValues();
    };

    // Per-device data.
    std::shared_ptr<DeviceResources> mDeviceResources;
    static SharedResourcePool<ID3D12Device*, DeviceResources, ResourceUploadBatch&> deviceResourcesPool;
};


// Global pools of per-device and per-context SpriteBatch resources.
SharedResourcePool<ID3D12Device*, SpriteBatch::Impl::DeviceResources, ResourceUploadBatch&> SpriteBatch::Impl::deviceResourcesPool;


// Constants.
const XMMATRIX SpriteBatch::MatrixIdentity = XMMatrixIdentity();
const XMFLOAT2 SpriteBatch::Float2Zero(0, 0);

const D3D12_SHADER_BYTECODE SpriteBatch::Impl::s_DefaultVertexShaderByteCodeStatic = {SpriteEffect_SpriteVertexShader, sizeof(SpriteEffect_SpriteVertexShader)};
const D3D12_SHADER_BYTECODE SpriteBatch::Impl::s_DefaultPixelShaderByteCodeStatic = {SpriteEffect_SpritePixelShader, sizeof(SpriteEffect_SpritePixelShader)};

const D3D12_SHADER_BYTECODE SpriteBatch::Impl::s_DefaultVertexShaderByteCodeHeap = { SpriteEffect_SpriteVertexShaderHeap, sizeof(SpriteEffect_SpriteVertexShaderHeap) };
const D3D12_SHADER_BYTECODE SpriteBatch::Impl::s_DefaultPixelShaderByteCodeHeap = { SpriteEffect_SpritePixelShaderHeap, sizeof(SpriteEffect_SpritePixelShaderHeap) };

const D3D12_INPUT_LAYOUT_DESC SpriteBatch::Impl::s_DefaultInputLayoutDesc = VertexPositionColorTexture::InputLayout;

// Matches CommonStates::AlphaBlend
const D3D12_BLEND_DESC SpriteBatchPipelineStateDescription::s_DefaultBlendDesc =
{
    FALSE, // AlphaToCoverageEnable
    FALSE, // IndependentBlendEnable
    { {
        TRUE, // BlendEnable
        FALSE, // LogicOpEnable
        D3D12_BLEND_ONE, // SrcBlend
        D3D12_BLEND_INV_SRC_ALPHA, // DestBlend
        D3D12_BLEND_OP_ADD, // BlendOp
        D3D12_BLEND_ONE, // SrcBlendAlpha
        D3D12_BLEND_INV_SRC_ALPHA, // DestBlendAlpha
        D3D12_BLEND_OP_ADD, // BlendOpAlpha
        D3D12_LOGIC_OP_NOOP,
        D3D12_COLOR_WRITE_ENABLE_ALL
    } }
};

// Same to CommonStates::CullCounterClockwise
const D3D12_RASTERIZER_DESC SpriteBatchPipelineStateDescription::s_DefaultRasterizerDesc =
{
    D3D12_FILL_MODE_SOLID,
    D3D12_CULL_MODE_BACK,
    FALSE, // FrontCounterClockwise
    D3D12_DEFAULT_DEPTH_BIAS,
    D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
    D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
    TRUE, // DepthClipEnable
    TRUE, // MultisampleEnable
    FALSE, // AntialiasedLineEnable
    0, // ForcedSampleCount
    D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
};

// Same as CommonStates::DepthNone
const D3D12_DEPTH_STENCIL_DESC SpriteBatchPipelineStateDescription::s_DefaultDepthStencilDesc =
{
    FALSE, // DepthEnable
    D3D12_DEPTH_WRITE_MASK_ZERO,
    D3D12_COMPARISON_FUNC_LESS_EQUAL, // DepthFunc
    FALSE, // StencilEnable
    D3D12_DEFAULT_STENCIL_READ_MASK,
    D3D12_DEFAULT_STENCIL_WRITE_MASK,
    {
        D3D12_STENCIL_OP_KEEP, // StencilFailOp
        D3D12_STENCIL_OP_KEEP, // StencilDepthFailOp
        D3D12_STENCIL_OP_KEEP, // StencilPassOp
        D3D12_COMPARISON_FUNC_ALWAYS // StencilFunc
    }, // FrontFace
    {
        D3D12_STENCIL_OP_KEEP, // StencilFailOp
        D3D12_STENCIL_OP_KEEP, // StencilDepthFailOp
        D3D12_STENCIL_OP_KEEP, // StencilPassOp
        D3D12_COMPARISON_FUNC_ALWAYS // StencilFunc
    } // BackFace
};

// Per-device constructor.
SpriteBatch::Impl::DeviceResources::DeviceResources(_In_ ID3D12Device* device, ResourceUploadBatch& upload) :
    indexBufferView{},
    mDevice(device)
{
    CreateIndexBuffer(device, upload);
    CreateRootSignatures(device);
}

// Creates the SpriteBatch index buffer.
void SpriteBatch::Impl::DeviceResources::CreateIndexBuffer(_In_ ID3D12Device* device, ResourceUploadBatch& upload)
{
    static_assert((MaxBatchSize * VerticesPerSprite) < USHRT_MAX, "MaxBatchSize too large for 16-bit indices");

    CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(short) * MaxBatchSize * IndicesPerSprite);

    // Create the constant buffer.
    ThrowIfFailed(device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(indexBuffer.ReleaseAndGetAddressOf())));

    SetDebugObjectName(indexBuffer.Get(), L"SpriteBatch");

    auto indexValues = CreateIndexValues();

    D3D12_SUBRESOURCE_DATA indexDataDesc = {};
    indexDataDesc.pData = indexValues.data();
    indexDataDesc.RowPitch = static_cast<LONG_PTR>(bufferDesc.Width);
    indexDataDesc.SlicePitch = indexDataDesc.RowPitch;

    // Upload the resource
    upload.Upload(indexBuffer.Get(), 0, &indexDataDesc, 1);
    upload.Transition(indexBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER);
    SetDebugObjectName(indexBuffer.Get(), L"DirectXTK:SpriteBatch Index Buffer");

    // Create the index buffer view
    indexBufferView.BufferLocation = indexBuffer->GetGPUVirtualAddress();
    indexBufferView.Format = DXGI_FORMAT_R16_UINT;
    indexBufferView.SizeInBytes = static_cast<UINT>(bufferDesc.Width);
}

void SpriteBatch::Impl::DeviceResources::CreateRootSignatures(_In_ ID3D12Device* device)
{
    D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;

    CD3DX12_DESCRIPTOR_RANGE textureSRV(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);

    {
        // Same as CommonStates::StaticLinearClamp
        CD3DX12_STATIC_SAMPLER_DESC sampler(
            0, // register
            D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            0.f,
            16,
            D3D12_COMPARISON_FUNC_LESS_EQUAL,
            D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
            0.f,
            D3D12_FLOAT32_MAX,
            D3D12_SHADER_VISIBILITY_PIXEL);

        CD3DX12_ROOT_PARAMETER rootParameters[RootParameterIndex::RootParameterCount - 1] = {};
        rootParameters[RootParameterIndex::TextureSRV].InitAsDescriptorTable(1, &textureSRV, D3D12_SHADER_VISIBILITY_PIXEL);
        rootParameters[RootParameterIndex::ConstantBuffer].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);

        CD3DX12_ROOT_SIGNATURE_DESC rsigDesc;
        rsigDesc.Init(_countof(rootParameters), rootParameters, 1, &sampler, rootSignatureFlags);

        ThrowIfFailed(::CreateRootSignature(device, &rsigDesc, rootSignatureStatic.ReleaseAndGetAddressOf()));

        SetDebugObjectName(rootSignatureStatic.Get(), L"SpriteBatch");
    }

    {
        CD3DX12_DESCRIPTOR_RANGE textureSampler(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0);

        CD3DX12_ROOT_PARAMETER rootParameters[RootParameterIndex::RootParameterCount] = {};
        rootParameters[RootParameterIndex::TextureSRV].InitAsDescriptorTable(1, &textureSRV, D3D12_SHADER_VISIBILITY_PIXEL);
        rootParameters[RootParameterIndex::ConstantBuffer].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);
        rootParameters[RootParameterIndex::TextureSampler].InitAsDescriptorTable(1, &textureSampler, D3D12_SHADER_VISIBILITY_PIXEL);

        CD3DX12_ROOT_SIGNATURE_DESC rsigDesc;
        rsigDesc.Init(_countof(rootParameters), rootParameters, 0, nullptr, rootSignatureFlags);

        ThrowIfFailed(::CreateRootSignature(device, &rsigDesc, rootSignatureHeap.ReleaseAndGetAddressOf()));

        SetDebugObjectName(rootSignatureHeap.Get(), L"SpriteBatch");
    }
}

// Helper for populating the SpriteBatch index buffer.
std::vector<short> SpriteBatch::Impl::DeviceResources::CreateIndexValues()
{
    std::vector<short> indices;

    indices.reserve(MaxBatchSize * IndicesPerSprite);

    for (size_t j = 0; j < MaxBatchSize * VerticesPerSprite; j += VerticesPerSprite)
    {
        short i = static_cast<short>(j);

        indices.push_back(i);
        indices.push_back(i + 1);
        indices.push_back(i + 2);

        indices.push_back(i + 1);
        indices.push_back(i + 3);
        indices.push_back(i + 2);
    }

    return indices;
}

// Per-SpriteBatch constructor.
_Use_decl_annotations_
SpriteBatch::Impl::Impl(ID3D12Device* device, ResourceUploadBatch& upload, const SpriteBatchPipelineStateDescription& psoDesc, const D3D12_VIEWPORT* viewport)
    : mRotation(DXGI_MODE_ROTATION_IDENTITY),
    mSetViewport(false),
    mViewPort{},
    mSpriteQueueCount(0),
    mSpriteQueueArraySize(0),
    mInBeginEndPair(false),
    mSortMode(SpriteSortMode_Deferred),
    mSampler{},
    mTransformMatrix(MatrixIdentity),
    mVertexSegment{},
    mVertexPageSize(sizeof(VertexPositionColorTexture) * MaxBatchSize * VerticesPerSprite),
    mSpriteCount(0),
    mDeviceResources(deviceResourcesPool.DemandCreate(device, upload))
{
    if (viewport != nullptr)
    {
        mViewPort = *viewport;
        mSetViewport = true;
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC d3dDesc = {};
    d3dDesc.InputLayout = s_DefaultInputLayoutDesc;
    d3dDesc.BlendState = psoDesc.blendDesc;
    d3dDesc.DepthStencilState = psoDesc.depthStencilDesc;
    d3dDesc.RasterizerState = psoDesc.rasterizerDesc;
    d3dDesc.DSVFormat = psoDesc.renderTargetState.dsvFormat;
    d3dDesc.NodeMask = psoDesc.renderTargetState.nodeMask;
    d3dDesc.NumRenderTargets = psoDesc.renderTargetState.numRenderTargets;
    memcpy_s(d3dDesc.RTVFormats, sizeof(d3dDesc.RTVFormats), psoDesc.renderTargetState.rtvFormats, sizeof(DXGI_FORMAT) * D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT);
    d3dDesc.SampleDesc = psoDesc.renderTargetState.sampleDesc;
    d3dDesc.SampleMask = psoDesc.renderTargetState.sampleMask;
    d3dDesc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
    d3dDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

    // Three choices: (1) static sampler, (2) heap sampler, or (3) custom signature & shaders
    if (psoDesc.customRootSignature)
    {
        mRootSignature = psoDesc.customRootSignature;
    }
    else
    {
        mRootSignature = (psoDesc.samplerDescriptor.ptr) ? mDeviceResources->rootSignatureHeap.Get() : mDeviceResources->rootSignatureStatic.Get();
    }
    d3dDesc.pRootSignature = mRootSignature.Get();

    if (psoDesc.customVertexShader.pShaderBytecode)
    {
        d3dDesc.VS = psoDesc.customVertexShader;
    }
    else
    {
        d3dDesc.VS = (psoDesc.samplerDescriptor.ptr) ? s_DefaultVertexShaderByteCodeHeap : s_DefaultVertexShaderByteCodeStatic;
    }

    if (psoDesc.customPixelShader.pShaderBytecode)
    {
        d3dDesc.PS = psoDesc.customPixelShader;
    }
    else
    {
        d3dDesc.PS = (psoDesc.samplerDescriptor.ptr) ? s_DefaultPixelShaderByteCodeHeap : s_DefaultPixelShaderByteCodeStatic;
    }

    if (psoDesc.samplerDescriptor.ptr)
    {
        mSampler = psoDesc.samplerDescriptor;
    }

    ThrowIfFailed(device->CreateGraphicsPipelineState(
        &d3dDesc,
        IID_GRAPHICS_PPV_ARGS(mPSO.GetAddressOf())));

    SetDebugObjectName(mPSO.Get(), L"SpriteBatch");
}

// Begins a batch of sprite drawing operations.
_Use_decl_annotations_
void XM_CALLCONV SpriteBatch::Impl::Begin(ID3D12GraphicsCommandList* commandList, SpriteSortMode sortMode, FXMMATRIX transformMatrix)
{
    if (mInBeginEndPair)
        throw std::exception("Cannot nest Begin calls on a single SpriteBatch");

    mSortMode = sortMode;
    mTransformMatrix = transformMatrix;
    mCommandList = commandList;
    mSpriteCount = 0;

    if (sortMode == SpriteSortMode_Immediate)
    {
        PrepareForRendering();
    }

    mInBeginEndPair = true;
}


// Ends a batch of sprite drawing operations.
void SpriteBatch::Impl::End()
{
    if (!mInBeginEndPair)
        throw std::exception("Begin must be called before End");

    if (mSortMode != SpriteSortMode_Immediate)
    {
        PrepareForRendering();
        FlushBatch();
    }

    // Release this memory
    mVertexSegment.Reset();

    // Break circular reference chains, in case the state lambda closed
    // over an object that holds a reference to this SpriteBatch.
    mCommandList = nullptr;
    mInBeginEndPair = false;
}


// Adds a single sprite to the queue.
_Use_decl_annotations_
void XM_CALLCONV SpriteBatch::Impl::Draw(D3D12_GPU_DESCRIPTOR_HANDLE texture,
    XMUINT2 const& textureSize,
    FXMVECTOR destination,
    RECT const* sourceRectangle,
    FXMVECTOR color,
    FXMVECTOR originRotationDepth,
    unsigned int flags)
{
    if (!mInBeginEndPair)
        throw std::exception("Begin must be called before Draw");

    if (!texture.ptr)
        throw std::exception("Invalid texture for Draw");

    // Get a pointer to the output sprite.
    if (mSpriteQueueCount >= mSpriteQueueArraySize)
    {
        GrowSpriteQueue();
    }

    SpriteInfo* sprite = &mSpriteQueue[mSpriteQueueCount];

    XMVECTOR dest = destination;

    if (sourceRectangle)
    {
        // User specified an explicit source region.
        XMVECTOR source = LoadRect(sourceRectangle);

        XMStoreFloat4A(&sprite->source, source);

        // If the destination size is relative to the source region, convert it to pixels.
        if (!(flags & SpriteInfo::DestSizeInPixels))
        {
            dest = XMVectorPermute<0, 1, 6, 7>(dest, XMVectorMultiply(dest, source)); // dest.zw *= source.zw
        }

        flags |= SpriteInfo::SourceInTexels | SpriteInfo::DestSizeInPixels;
    }
    else
    {
        // No explicit source region, so use the entire texture.
        static const XMVECTORF32 wholeTexture = { { {0, 0, 1, 1} } };

        XMStoreFloat4A(&sprite->source, wholeTexture);
    }

    // Convert texture size
    XMVECTOR textureSizeV = XMLoadUInt2(&textureSize);

    // Store sprite parameters.
    XMStoreFloat4A(&sprite->destination, dest);
    XMStoreFloat4A(&sprite->color, color);
    XMStoreFloat4A(&sprite->originRotationDepth, originRotationDepth);

    sprite->texture = texture;
    sprite->textureSize = textureSizeV;
    sprite->flags = flags;

    if (mSortMode == SpriteSortMode_Immediate)
    {
        // If we are in immediate mode, draw this sprite straight away.
        RenderBatch(texture, textureSizeV, &sprite, 1);
    }
    else
    {
        // Queue this sprite for later sorting and batched rendering.
        mSpriteQueueCount++;
    }
}


// Dynamically expands the array used to store pending sprite information.
void SpriteBatch::Impl::GrowSpriteQueue()
{
    // Grow by a factor of 2.
    size_t newSize = std::max(InitialQueueSize, mSpriteQueueArraySize * 2);

    // Allocate the new array.
    std::unique_ptr<SpriteInfo[]> newArray(new SpriteInfo[newSize]);

    // Copy over any existing sprites.
    for (size_t i = 0; i < mSpriteQueueCount; i++)
    {
        newArray[i] = mSpriteQueue[i];
    }

    // Replace the previous array with the new one.
    mSpriteQueue = std::move(newArray);
    mSpriteQueueArraySize = newSize;

    // Clear any dangling SpriteInfo pointers left over from previous rendering.
    mSortedSprites.clear();
}


// Sets up D3D device state ready for drawing sprites.
void SpriteBatch::Impl::PrepareForRendering()
{
    auto commandList = mCommandList.Get();

    // Set root signature
    commandList->SetGraphicsRootSignature(mRootSignature.Get());

    // Set render state
    commandList->SetPipelineState(mPSO.Get());
    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // Set the index buffer.
    commandList->IASetIndexBuffer(&mDeviceResources->indexBufferView);

    // Set the transform matrix.
    XMMATRIX transformMatrix = (mRotation == DXGI_MODE_ROTATION_UNSPECIFIED)
        ? mTransformMatrix
        : (mTransformMatrix * GetViewportTransform(mRotation));

    mConstantBuffer = GraphicsMemory::Get(mDeviceResources->mDevice).AllocateConstant(transformMatrix);
    commandList->SetGraphicsRootConstantBufferView(RootParameterIndex::ConstantBuffer, mConstantBuffer.GpuAddress());
}


// Sends queued sprites to the graphics device.
void SpriteBatch::Impl::FlushBatch()
{
    if (!mSpriteQueueCount)
        return;

    SortSprites();

    // Walk through the sorted sprite list, looking for adjacent entries that share a texture.
    D3D12_GPU_DESCRIPTOR_HANDLE batchTexture = {};
    XMVECTOR batchTextureSize = {};
    size_t batchStart = 0;

    for (size_t pos = 0; pos < mSpriteQueueCount; pos++)
    {
        D3D12_GPU_DESCRIPTOR_HANDLE texture = mSortedSprites[pos]->texture;
        assert(texture.ptr != 0);
        XMVECTOR textureSize = mSortedSprites[pos]->textureSize;

        // Flush whenever the texture changes.
        if (texture != batchTexture)
        {
            if (pos > batchStart)
            {
                RenderBatch(batchTexture, batchTextureSize, &mSortedSprites[batchStart], pos - batchStart);
            }

            batchTexture = texture;
            batchTextureSize = textureSize;
            batchStart = pos;
        }
    }

    // Flush the final batch.
    RenderBatch(batchTexture, batchTextureSize, &mSortedSprites[batchStart], mSpriteQueueCount - batchStart);

    // Reset the queue.
    mSpriteQueueCount = 0;

    // When sorting is disabled, we persist mSortedSprites data from one batch to the next, to avoid
    // uneccessary work in GrowSortedSprites. But we never reuse these when sorting, because re-sorting
    // previously sorted items gives unstable ordering if some sprites have identical sort keys.
    if (mSortMode != SpriteSortMode_Deferred)
    {
        mSortedSprites.clear();
    }
}


// Sorts the array of queued sprites.
void SpriteBatch::Impl::SortSprites()
{
    // Fill the mSortedSprites vector.
    if (mSortedSprites.size() < mSpriteQueueCount)
    {
        GrowSortedSprites();
    }

    switch (mSortMode)
    {
    case SpriteSortMode_Texture:
        // Sort by texture.
        std::sort(mSortedSprites.begin(),
            mSortedSprites.begin() + static_cast<int>(mSpriteQueueCount),
            [](SpriteInfo const* x, SpriteInfo const* y) -> bool
            {
                return x->texture < y->texture;
            });
        break;

    case SpriteSortMode_BackToFront:
        // Sort back to front.
        std::sort(mSortedSprites.begin(),
            mSortedSprites.begin() + static_cast<int>(mSpriteQueueCount),
            [](SpriteInfo const* x, SpriteInfo const* y) -> bool
            {
                return x->originRotationDepth.w > y->originRotationDepth.w;
            });
        break;

    case SpriteSortMode_FrontToBack:
        // Sort front to back.
        std::sort(mSortedSprites.begin(),
            mSortedSprites.begin() + static_cast<int>(mSpriteQueueCount),
            [](SpriteInfo const* x, SpriteInfo const* y) -> bool
            {
                return x->originRotationDepth.w < y->originRotationDepth.w;
            });
        break;

    default:
        break;
    }
}


// Populates the mSortedSprites vector with pointers to individual elements of the mSpriteQueue array.
void SpriteBatch::Impl::GrowSortedSprites()
{
    size_t previousSize = mSortedSprites.size();

    mSortedSprites.resize(mSpriteQueueCount);

    for (size_t i = previousSize; i < mSpriteQueueCount; i++)
    {
        mSortedSprites[i] = &mSpriteQueue[i];
    }
}


// Submits a batch of sprites to the GPU.
_Use_decl_annotations_
void SpriteBatch::Impl::RenderBatch(D3D12_GPU_DESCRIPTOR_HANDLE texture, XMVECTOR textureSize, SpriteInfo const* const* sprites, size_t count)
{
    auto commandList = mCommandList.Get();

    // Draw using the specified texture.
    // **NOTE** If D3D asserts or crashes here, you probably need to call commandList->SetDescriptorHeaps() with the required descriptor heap(s)
    commandList->SetGraphicsRootDescriptorTable(RootParameterIndex::TextureSRV, texture);

    if (mSampler.ptr)
    {
        commandList->SetGraphicsRootDescriptorTable(RootParameterIndex::TextureSampler, mSampler);
    }

    // Convert to vector format.
    XMVECTOR inverseTextureSize = XMVectorReciprocal(textureSize);

    while (count > 0)
    {
        // How many sprites do we want to draw?
        size_t batchSize = count;

        // How many sprites does the D3D vertex buffer have room for?
        size_t remainingSpace = MaxBatchSize - mSpriteCount;

        if (batchSize > remainingSpace)
        {
            if (remainingSpace < MinBatchSize)
            {
                // If we are out of room, or about to submit an excessively small batch, wrap back to the start of the vertex buffer.
                mSpriteCount = 0;

                batchSize = std::min(count, MaxBatchSize);
            }
            else
            {
                // Take however many sprites fit in what's left of the vertex buffer.
                batchSize = remainingSpace;
            }
        }

        // Allocate a new page of vertex memory if we're starting the batch
        if (mSpriteCount == 0)
        {
            mVertexSegment = GraphicsMemory::Get(mDeviceResources->mDevice).Allocate(mVertexPageSize);
        }

        auto vertices = static_cast<VertexPositionColorTexture*>(mVertexSegment.Memory()) + mSpriteCount * VerticesPerSprite;

        // Generate sprite vertex data.
        for (size_t i = 0; i < batchSize; i++)
        {
            assert(i < count);
            _Analysis_assume_(i < count);
            RenderSprite(sprites[i], vertices, textureSize, inverseTextureSize);

            vertices += VerticesPerSprite;
        }

        // Set the vertex buffer view
        D3D12_VERTEX_BUFFER_VIEW vbv;
        size_t spriteVertexTotalSize = sizeof(VertexPositionColorTexture) * VerticesPerSprite;
        vbv.BufferLocation = mVertexSegment.GpuAddress() + (UINT64(mSpriteCount) * UINT64(spriteVertexTotalSize));
        vbv.StrideInBytes = sizeof(VertexPositionColorTexture);
        vbv.SizeInBytes = static_cast<UINT>(batchSize * spriteVertexTotalSize);
        commandList->IASetVertexBuffers(0, 1, &vbv);

        // Ok lads, the time has come for us draw ourselves some sprites!
        UINT indexCount = static_cast<UINT>(batchSize * IndicesPerSprite);

        commandList->DrawIndexedInstanced(indexCount, 1, 0, 0, 0);

        // Advance the buffer position.
        mSpriteCount += batchSize;

        sprites += batchSize;
        count -= batchSize;
    }
}


// Generates vertex data for drawing a single sprite.
_Use_decl_annotations_
void XM_CALLCONV SpriteBatch::Impl::RenderSprite(SpriteInfo const* sprite, VertexPositionColorTexture* vertices, FXMVECTOR textureSize, FXMVECTOR inverseTextureSize)
{
    // Load sprite parameters into SIMD registers.
    XMVECTOR source = XMLoadFloat4A(&sprite->source);
    XMVECTOR destination = XMLoadFloat4A(&sprite->destination);
    XMVECTOR color = XMLoadFloat4A(&sprite->color);
    XMVECTOR originRotationDepth = XMLoadFloat4A(&sprite->originRotationDepth);

    float rotation = sprite->originRotationDepth.z;
    unsigned int flags = sprite->flags;

    // Extract the source and destination sizes into separate vectors.
    XMVECTOR sourceSize = XMVectorSwizzle<2, 3, 2, 3>(source);
    XMVECTOR destinationSize = XMVectorSwizzle<2, 3, 2, 3>(destination);

    // Scale the origin offset by source size, taking care to avoid overflow if the source region is zero.
    XMVECTOR isZeroMask = XMVectorEqual(sourceSize, XMVectorZero());
    XMVECTOR nonZeroSourceSize = XMVectorSelect(sourceSize, g_XMEpsilon, isZeroMask);

    XMVECTOR origin = XMVectorDivide(originRotationDepth, nonZeroSourceSize);

    // Convert the source region from texels to mod-1 texture coordinate format.
    if (flags & SpriteInfo::SourceInTexels)
    {
        source = XMVectorMultiply(source, inverseTextureSize);
        sourceSize = XMVectorMultiply(sourceSize, inverseTextureSize);
    }
    else
    {
        origin = XMVectorMultiply(origin, inverseTextureSize);
    }

    // If the destination size is relative to the source region, convert it to pixels.
    if (!(flags & SpriteInfo::DestSizeInPixels))
    {
        destinationSize = XMVectorMultiply(destinationSize, textureSize);
    }

    // Compute a 2x2 rotation matrix.
    XMVECTOR rotationMatrix1;
    XMVECTOR rotationMatrix2;

    if (rotation != 0)
    {
        float sin, cos;

        XMScalarSinCos(&sin, &cos, rotation);

        XMVECTOR sinV = XMLoadFloat(&sin);
        XMVECTOR cosV = XMLoadFloat(&cos);

        rotationMatrix1 = XMVectorMergeXY(cosV, sinV);
        rotationMatrix2 = XMVectorMergeXY(XMVectorNegate(sinV), cosV);
    }
    else
    {
        rotationMatrix1 = g_XMIdentityR0;
        rotationMatrix2 = g_XMIdentityR1;
    }

    // The four corner vertices are computed by transforming these unit-square positions.
    static XMVECTORF32 cornerOffsets[VerticesPerSprite] =
    {
        { { { 0, 0, 0, 0 } } },
        { { { 1, 0, 0, 0 } } },
        { { { 0, 1, 0, 0 } } },
        { { { 1, 1, 0, 0 } } },
    };

    // Tricksy alert! Texture coordinates are computed from the same cornerOffsets
    // table as vertex positions, but if the sprite is mirrored, this table
    // must be indexed in a different order. This is done as follows:
    //
    //    position = cornerOffsets[i]
    //    texcoord = cornerOffsets[i ^ SpriteEffects]

    static_assert(SpriteEffects_FlipHorizontally == 1 &&
        SpriteEffects_FlipVertically == 2, "If you change these enum values, the mirroring implementation must be updated to match");

    const unsigned int mirrorBits = flags & 3u;

    // Generate the four output vertices.
    for (size_t i = 0; i < VerticesPerSprite; i++)
    {
        // Calculate position.
        XMVECTOR cornerOffset = XMVectorMultiply(XMVectorSubtract(cornerOffsets[i], origin), destinationSize);

        // Apply 2x2 rotation matrix.
        XMVECTOR position1 = XMVectorMultiplyAdd(XMVectorSplatX(cornerOffset), rotationMatrix1, destination);
        XMVECTOR position2 = XMVectorMultiplyAdd(XMVectorSplatY(cornerOffset), rotationMatrix2, position1);

        // Set z = depth.
        XMVECTOR position = XMVectorPermute<0, 1, 7, 6>(position2, originRotationDepth);

        // Write position as a Float4, even though VertexPositionColor::position is an XMFLOAT3.
        // This is faster, and harmless as we are just clobbering the first element of the
        // following color field, which will immediately be overwritten with its correct value.
        XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(&vertices[i].position), position);

        // Write the color.
        XMStoreFloat4(&vertices[i].color, color);

        // Compute and write the texture coordinate.
        XMVECTOR textureCoordinate = XMVectorMultiplyAdd(cornerOffsets[static_cast<unsigned int>(i) ^ mirrorBits], sourceSize, source);

        XMStoreFloat2(&vertices[i].textureCoordinate, textureCoordinate);
    }
}


// Generates a viewport transform matrix for rendering sprites using x-right y-down screen pixel coordinates.
XMMATRIX SpriteBatch::Impl::GetViewportTransform(_In_ DXGI_MODE_ROTATION rotation)
{
    if (!mSetViewport)
        throw std::exception("Viewport not set.");

    // Compute the matrix.
    float xScale = (mViewPort.Width > 0) ? 2.0f / mViewPort.Width : 0.0f;
    float yScale = (mViewPort.Height > 0) ? 2.0f / mViewPort.Height : 0.0f;

    switch (rotation)
    {
    case DXGI_MODE_ROTATION_ROTATE90:
        return XMMATRIX
            (
                0, -yScale, 0, 0,
                -xScale, 0, 0, 0,
                0, 0, 1, 0,
                1, 1, 0, 1
                );

    case DXGI_MODE_ROTATION_ROTATE270:
        return XMMATRIX
            (
                0, yScale, 0, 0,
                xScale, 0, 0, 0,
                0, 0, 1, 0,
                -1, -1, 0, 1
                );

    case DXGI_MODE_ROTATION_ROTATE180:
        return XMMATRIX
            (
                -xScale, 0, 0, 0,
                0, yScale, 0, 0,
                0, 0, 1, 0,
                1, -1, 0, 1
                );

    default:
        return XMMATRIX
            (
                xScale, 0, 0, 0,
                0, -yScale, 0, 0,
                0, 0, 1, 0,
                -1, 1, 0, 1
                );
    }
}


// Public constructor.
_Use_decl_annotations_
SpriteBatch::SpriteBatch(ID3D12Device* device,
    ResourceUploadBatch& upload,
    const SpriteBatchPipelineStateDescription& psoDesc,
    const D3D12_VIEWPORT* viewport)
    : pImpl(std::make_unique<Impl>(device, upload, psoDesc, viewport))
{
}


// Move constructor.
SpriteBatch::SpriteBatch(SpriteBatch&& moveFrom) noexcept
    : pImpl(std::move(moveFrom.pImpl))
{
}


// Move assignment.
SpriteBatch& SpriteBatch::operator= (SpriteBatch&& moveFrom) noexcept
{
    pImpl = std::move(moveFrom.pImpl);
    return *this;
}


// Public destructor.
SpriteBatch::~SpriteBatch()
{
}

_Use_decl_annotations_
void XM_CALLCONV SpriteBatch::Begin(
    ID3D12GraphicsCommandList* commandList,
    SpriteSortMode sortMode,
    FXMMATRIX transformMatrix)
{
    pImpl->Begin(
        commandList,
        sortMode,
        transformMatrix);
}


void SpriteBatch::End()
{
    pImpl->End();
}


void XM_CALLCONV SpriteBatch::Draw(D3D12_GPU_DESCRIPTOR_HANDLE texture,
    XMUINT2 const& textureSize,
    XMFLOAT2 const& position,
    FXMVECTOR color)
{
    XMVECTOR destination = XMVectorPermute<0, 1, 4, 5>(XMLoadFloat2(&position), g_XMOne); // x, y, 1, 1

    pImpl->Draw(texture, textureSize, destination, nullptr, color, g_XMZero, 0);
}


_Use_decl_annotations_
void XM_CALLCONV SpriteBatch::Draw(D3D12_GPU_DESCRIPTOR_HANDLE texture,
    XMUINT2 const& textureSize,
    XMFLOAT2 const& position, 
    RECT const* sourceRectangle,
    FXMVECTOR color,
    float rotation,
    XMFLOAT2 const& origin,
    float scale,
    SpriteEffects effects,
    float layerDepth)
{
    XMVECTOR destination = XMVectorPermute<0, 1, 4, 4>(XMLoadFloat2(&position), XMLoadFloat(&scale)); // x, y, scale, scale

    XMVECTOR originRotationDepth = XMVectorSet(origin.x, origin.y, rotation, layerDepth);

    pImpl->Draw(texture, textureSize, destination, sourceRectangle, color, originRotationDepth, static_cast<unsigned int>(effects));
}


_Use_decl_annotations_
void XM_CALLCONV SpriteBatch::Draw(D3D12_GPU_DESCRIPTOR_HANDLE texture, 
    XMUINT2 const& textureSize,
    XMFLOAT2 const& position,
    RECT const* sourceRectangle,
    FXMVECTOR color,
    float rotation,
    XMFLOAT2 const& origin,
    XMFLOAT2 const& scale,
    SpriteEffects effects,
    float layerDepth)
{
    XMVECTOR destination = XMVectorPermute<0, 1, 4, 5>(XMLoadFloat2(&position), XMLoadFloat2(&scale)); // x, y, scale.x, scale.y

    XMVECTOR originRotationDepth = XMVectorSet(origin.x, origin.y, rotation, layerDepth);

    pImpl->Draw(texture, textureSize, destination, sourceRectangle, color, originRotationDepth, static_cast<unsigned int>(effects));
}


void XM_CALLCONV SpriteBatch::Draw(D3D12_GPU_DESCRIPTOR_HANDLE texture, XMUINT2 const& textureSize, FXMVECTOR position, FXMVECTOR color)
{
    XMVECTOR destination = XMVectorPermute<0, 1, 4, 5>(position, g_XMOne); // x, y, 1, 1

    pImpl->Draw(texture, textureSize, destination, nullptr, color, g_XMZero, 0);
}


_Use_decl_annotations_
void XM_CALLCONV SpriteBatch::Draw(D3D12_GPU_DESCRIPTOR_HANDLE texture,
    XMUINT2 const& textureSize,
    FXMVECTOR position, 
    RECT const* sourceRectangle,
    FXMVECTOR color,
    float rotation,
    FXMVECTOR origin,
    float scale,
    SpriteEffects effects,
    float layerDepth)
{
    XMVECTOR destination = XMVectorPermute<0, 1, 4, 4>(position, XMLoadFloat(&scale)); // x, y, scale, scale

    XMVECTOR rotationDepth = XMVectorMergeXY(XMVectorReplicate(rotation), XMVectorReplicate(layerDepth));

    XMVECTOR originRotationDepth = XMVectorPermute<0, 1, 4, 5>(origin, rotationDepth);

    pImpl->Draw(texture, textureSize, destination, sourceRectangle, color, originRotationDepth, static_cast<unsigned int>(effects));
}


_Use_decl_annotations_
void XM_CALLCONV SpriteBatch::Draw(D3D12_GPU_DESCRIPTOR_HANDLE texture,
    XMUINT2 const& textureSize,
    FXMVECTOR position,
    RECT const* sourceRectangle,
    FXMVECTOR color,
    float rotation,
    FXMVECTOR origin,
    GXMVECTOR scale,
    SpriteEffects effects,
    float layerDepth)
{
    XMVECTOR destination = XMVectorPermute<0, 1, 4, 5>(position, scale); // x, y, scale.x, scale.y

    XMVECTOR rotationDepth = XMVectorMergeXY(XMVectorReplicate(rotation), XMVectorReplicate(layerDepth));

    XMVECTOR originRotationDepth = XMVectorPermute<0, 1, 4, 5>(origin, rotationDepth);

    pImpl->Draw(texture, textureSize, destination, sourceRectangle, color, originRotationDepth, static_cast<unsigned int>(effects));
}


void XM_CALLCONV SpriteBatch::Draw(D3D12_GPU_DESCRIPTOR_HANDLE texture, 
    XMUINT2 const& textureSize,
    RECT const& destinationRectangle,
    FXMVECTOR color)
{
    XMVECTOR destination = LoadRect(&destinationRectangle); // x, y, w, h

    pImpl->Draw(texture, textureSize, destination, nullptr, color, g_XMZero, Impl::SpriteInfo::DestSizeInPixels);
}


_Use_decl_annotations_
void XM_CALLCONV SpriteBatch::Draw(D3D12_GPU_DESCRIPTOR_HANDLE texture,
    XMUINT2 const& textureSize,
    RECT const& destinationRectangle,
    RECT const* sourceRectangle,
    FXMVECTOR color,
    float rotation,
    XMFLOAT2 const& origin, 
    SpriteEffects effects, 
    float layerDepth)
{
    XMVECTOR destination = LoadRect(&destinationRectangle); // x, y, w, h

    XMVECTOR originRotationDepth = XMVectorSet(origin.x, origin.y, rotation, layerDepth);

    pImpl->Draw(texture, textureSize, destination, sourceRectangle, color, originRotationDepth, static_cast<unsigned int>(effects) | Impl::SpriteInfo::DestSizeInPixels);
}


void SpriteBatch::SetRotation(DXGI_MODE_ROTATION mode)
{
    pImpl->mRotation = mode;
}


DXGI_MODE_ROTATION SpriteBatch::GetRotation() const
{
    return pImpl->mRotation;
}


void SpriteBatch::SetViewport(const D3D12_VIEWPORT& viewPort)
{
    pImpl->mSetViewport = true;
    pImpl->mViewPort = viewPort;
}
