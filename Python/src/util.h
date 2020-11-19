//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

inline void ThrowIfFailed(HRESULT hr)
{
    if (FAILED(hr))
        throw std::exception();
}

inline void ThrowIfNull(void* p)
{
    if (!p)
        throw std::exception();
}

// DML_BUFFER_TENSOR_DESC (DML_TENSOR_TYPE_BUFFER)
struct DmlBufferTensorDesc
{
    DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_UNKNOWN;
    DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;
    std::vector<uint32_t> sizes;
    std::optional<std::vector<uint32_t>> strides;
    uint64_t totalTensorSizeInBytes = 0;
    uint32_t guaranteedBaseOffsetAlignment = 0;

    DmlBufferTensorDesc() = default;

    /*implicit*/ DmlBufferTensorDesc(const DML_BUFFER_TENSOR_DESC& desc)
        : dataType(desc.DataType),
        flags(desc.Flags),
        sizes(desc.Sizes, desc.Sizes + desc.DimensionCount),
        totalTensorSizeInBytes(desc.TotalTensorSizeInBytes),
        guaranteedBaseOffsetAlignment(desc.GuaranteedBaseOffsetAlignment)
    {
        if (desc.Strides)
        {
            strides.emplace(desc.Strides, desc.Strides + desc.DimensionCount);
        }
    }

    // Constructs a DmlBufferTensorDesc from a generic DML_TENSOR_DESC. The type must be DML_TENSOR_TYPE_BUFFER.
    /*implicit*/ DmlBufferTensorDesc(const DML_TENSOR_DESC& desc)
        : DmlBufferTensorDesc(*static_cast<const DML_BUFFER_TENSOR_DESC*>(desc.Desc))
    {
        assert(desc.Type == DML_TENSOR_TYPE_BUFFER);
    }

    uint32_t GetDimensionCount() const
    {
        assert(!strides || strides->size() == sizes.size());
        return static_cast<uint32_t>(sizes.size());
    }

    operator DML_BUFFER_TENSOR_DESC() const
    {
        DML_BUFFER_TENSOR_DESC bufferTensorDesc;
        bufferTensorDesc.DataType = dataType;
        bufferTensorDesc.DimensionCount = GetDimensionCount();
        bufferTensorDesc.Flags = flags;
        bufferTensorDesc.GuaranteedBaseOffsetAlignment = guaranteedBaseOffsetAlignment;
        bufferTensorDesc.Sizes = sizes.data();
        bufferTensorDesc.Strides = strides ? strides->data() : nullptr;
        bufferTensorDesc.TotalTensorSizeInBytes = totalTensorSizeInBytes;

        return bufferTensorDesc;
    }
};

// (DML_BINDING_TYPE_NONE)
struct DmlNoneBinding
{
};

// DML_BUFFER_BINDING (DML_BINDING_TYPE_BUFFER)
struct DmlBufferBinding
{
    ID3D12Resource* buffer;
    uint64_t offset;
    uint64_t sizeInBytes;

    DmlBufferBinding() = default;

    /*implicit*/ DmlBufferBinding(const DML_BUFFER_BINDING& desc)
        : buffer(desc.Buffer),
        offset(desc.Offset),
        sizeInBytes(desc.SizeInBytes)
    {
    }
};

// DML_BUFFER_ARRAY_BINDING (DML_BINDING_TYPE_BUFFER_ARRAY)
struct DmlBufferArrayBinding
{
    std::vector<DmlBufferBinding> bindings;

    DmlBufferArrayBinding() = default;

    /*implicit*/ DmlBufferArrayBinding(const DML_BUFFER_ARRAY_BINDING& desc)
        : bindings(desc.Bindings, desc.Bindings + desc.BindingCount)
    {
    }
};

inline Microsoft::WRL::ComPtr<ID3D12Resource> CreateCommittedResource(
    ID3D12Device* device,
    const D3D12_RESOURCE_DESC& resourceDesc,
    const D3D12_HEAP_PROPERTIES& heapProperties,
    D3D12_RESOURCE_STATES initialState
    )
{
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    ThrowIfFailed(device->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        initialState,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(resource.GetAddressOf())
        ));

    return resource;
}

inline Microsoft::WRL::ComPtr<ID3D12Resource> CreateCpuCustomBuffer(
    ID3D12Device* device,
    UINT64 sizeInBytes,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    )
{
    D3D12_HEAP_PROPERTIES heapProperties = {
        D3D12_HEAP_TYPE_CUSTOM,
        D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
        D3D12_MEMORY_POOL_L0,
        0,
        0
    };

    return CreateCommittedResource(
        device,
        CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, flags),
        heapProperties,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
}

inline Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultBuffer(
    ID3D12Device* device,
    UINT64 sizeInBytes,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    )
{
    return CreateCommittedResource(
        device,
        CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, flags),
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
}

inline Microsoft::WRL::ComPtr<ID3D12Resource> CreateReadBackBuffer(ID3D12Device* device, UINT64 sizeInBytes)
{
    return CreateCommittedResource(
        device,
        CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes),
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_RESOURCE_STATE_COPY_DEST
        );
}

void FillGpuBuffer(
    ID3D12GraphicsCommandList* commandList,
    ID3D12DescriptorHeap* descriptorHeapCpuVisible,
    ID3D12DescriptorHeap* descriptorHeapGpuVisible,
    uint32_t descriptorOffset,
    ID3D12Resource* buffer,
    uint32_t value
    );

void WaitForQueueToComplete(ID3D12CommandQueue* queue);

inline std::string UintVectorToString(std::vector<uint32_t> const& v)
{
    if (v.empty())
        return std::string();

    return std::accumulate(v.begin() + 1, v.end(), std::to_string(v[0]),
        [](std::string const& a, int b) {
            return a + ',' + std::to_string(b);
        });
}

template <typename T>
T RoundUpToMultiple(T value, T multiple)
{
    static_assert(std::is_integral_v<T>);

    T remainder = value % multiple;
    if (remainder != 0)
    {
        value += multiple - remainder;
    }

    return value;
}

// Rounds up a value to the nearest power of two
template <typename T>
T RoundUpToPow2(T value)
{
    static_assert(std::is_integral_v<T>);

    if (value >= std::numeric_limits<T>::max() / 2)
    {
        ThrowIfFailed(E_INVALIDARG); // overflow
    }

    T pow2 = 1;
    while (pow2 < value)
    {
        pow2 *= 2;
    }

    return pow2;
}
