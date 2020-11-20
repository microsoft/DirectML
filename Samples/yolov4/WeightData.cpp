#pragma once

#include "pch.h"
#include "WeightData.h"

using Microsoft::WRL::ComPtr;

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

WeightData::WeightData(dml::Span<const ConvWeightData> weights, DX::DeviceResources* deviceResources)
{
    // We round up all bindings to this alignment, to ensure that every binding has an offset that meets the minimum
    // alignment requirement.
    const size_t requiredAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;

    size_t offsetInBytes = 0;

    // Generate binding offsets for each set of weights
    for (const ConvWeightData& weight : weights)
    {
        size_t filterSizeInBytes = RoundUpToMultiple(weight.filterData.size() * sizeof(float), requiredAlignment);
        m_bindings.push_back(DML_BUFFER_BINDING{ nullptr, offsetInBytes, filterSizeInBytes });
        offsetInBytes += filterSizeInBytes;

        size_t biasSizeInBytes = RoundUpToMultiple(weight.biasData.size() * sizeof(float), requiredAlignment);
        m_bindings.push_back(DML_BUFFER_BINDING{ nullptr, offsetInBytes, biasSizeInBytes });
        offsetInBytes += biasSizeInBytes;
    }

    // Create our weight buffer
    uint64_t resourceSizeInBytes = offsetInBytes;
    DX::ThrowIfFailed(deviceResources->GetD3DDevice()->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(resourceSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&m_weightBuffer)));

    // Create an upload heap
    ComPtr<ID3D12Resource> uploadHeap;
    DX::ThrowIfFailed(deviceResources->GetD3DDevice()->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(resourceSizeInBytes),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&uploadHeap)));

    // Copy all the weights into the upload heap
    byte* uploadHeapData = nullptr;
    DX::ThrowIfFailed(uploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&uploadHeapData)));
    assert(m_bindings.size() == weights.size() * 2);
    for (size_t i = 0; i < weights.size(); ++i)
    {
        const ConvWeightData& weightData = weights[i];
        DML_BUFFER_BINDING* filterBinding = &m_bindings[i * 2];
        DML_BUFFER_BINDING* biasBinding = &m_bindings[i * 2 + 1];

        // Copy filter weights
        size_t filterSizeInBytes = weightData.filterData.size() * sizeof(float); // excludes padding
        memcpy(uploadHeapData + filterBinding->Offset, weightData.filterData.data(), filterSizeInBytes);

        // Copy bias weights
        size_t biasSizeInBytes = weightData.biasData.size() * sizeof(float); // excludes padding
        memcpy(uploadHeapData + biasBinding->Offset, weightData.biasData.data(), biasSizeInBytes);

        // While we're here, also fill in the resource pointer for our bindings
        filterBinding->Buffer = m_weightBuffer.Get();
        biasBinding->Buffer = m_weightBuffer.Get();
    }
    uploadHeap->Unmap(0, nullptr);

    // Record the upload into the command list
    ID3D12GraphicsCommandList* commandList = deviceResources->GetCommandList();
    commandList->Reset(deviceResources->GetCommandAllocator(), nullptr);
    commandList->CopyResource(m_weightBuffer.Get(), uploadHeap.Get());
    commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            m_weightBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    DX::ThrowIfFailed(commandList->Close());

    // Kick off the upload
    deviceResources->GetCommandQueue()->ExecuteCommandLists(1, CommandListCast(&commandList));

    // Wait for the upload to complete
    deviceResources->WaitForGpu();
}
