//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#include "precomp.h"

using Microsoft::WRL::ComPtr;

void FillGpuBuffer(
    ID3D12GraphicsCommandList* commandList,
    ID3D12DescriptorHeap* descriptorHeapCpuVisible,
    ID3D12DescriptorHeap* descriptorHeapGpuVisible,
    uint32_t descriptorOffset,
    ID3D12Resource* buffer,
    uint32_t value)
{
    ComPtr<ID3D12Device> device;
    ThrowIfFailed(commandList->GetDevice(IID_GRAPHICS_PPV_ARGS(device.GetAddressOf())));

    uint32_t incrementSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Create a RAW buffer UAV over our resource
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.Buffer.NumElements = static_cast<uint32_t>(buffer->GetDesc().Width / sizeof(uint32_t));
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;

    CD3DX12_CPU_DESCRIPTOR_HANDLE cpuVisibleHandle(descriptorHeapCpuVisible->GetCPUDescriptorHandleForHeapStart(), descriptorOffset, incrementSize);
    CD3DX12_CPU_DESCRIPTOR_HANDLE gpuVisibleHandle(descriptorHeapGpuVisible->GetCPUDescriptorHandleForHeapStart(), descriptorOffset, incrementSize);

    device->CreateUnorderedAccessView(buffer, nullptr, &uav, cpuVisibleHandle);
    device->CreateUnorderedAccessView(buffer, nullptr, &uav, gpuVisibleHandle);

    commandList->SetDescriptorHeaps(1, &descriptorHeapGpuVisible);

    // Record a ClearUAV onto the command list, filling it with garbage
    uint32_t values[] = { value, value, value, value };
    commandList->ClearUnorderedAccessViewUint(
        CD3DX12_GPU_DESCRIPTOR_HANDLE(descriptorHeapGpuVisible->GetGPUDescriptorHandleForHeapStart(), descriptorOffset, incrementSize),
        cpuVisibleHandle,
        buffer,
        values,
        0,
        nullptr);
}

void WaitForQueueToComplete(ID3D12CommandQueue* queue)
{
    ComPtr<ID3D12Device> device;
    ThrowIfFailed(queue->GetDevice(IID_GRAPHICS_PPV_ARGS(device.GetAddressOf())));
    ComPtr<ID3D12Fence> fence;
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_GRAPHICS_PPV_ARGS(fence.GetAddressOf())));
    ThrowIfFailed(queue->Signal(fence.Get(), 1));
    ThrowIfFailed(fence->SetEventOnCompletion(1, nullptr));
}
