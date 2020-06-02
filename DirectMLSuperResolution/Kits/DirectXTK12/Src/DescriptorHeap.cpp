//--------------------------------------------------------------------------------------
// File: DescriptorHeap.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "PlatformHelpers.h"
#include "DirectXHelpers.h"
#include "DescriptorHeap.h"

using namespace DirectX;
using Microsoft::WRL::ComPtr;

namespace
{
    struct DescriptorHeapDesc
    {
        D3D12_DESCRIPTOR_HEAP_TYPE Type;
        D3D12_DESCRIPTOR_HEAP_FLAGS Flags;
    };

    static const DescriptorHeapDesc c_DescriptorHeapDescs[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES] =
    {
        { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,	D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE },
        { D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,		D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE },
        { D3D12_DESCRIPTOR_HEAP_TYPE_RTV,			D3D12_DESCRIPTOR_HEAP_FLAG_NONE },
        { D3D12_DESCRIPTOR_HEAP_TYPE_DSV,			D3D12_DESCRIPTOR_HEAP_FLAG_NONE }
    };
}

_Use_decl_annotations_
DescriptorHeap::DescriptorHeap(
    ID3D12DescriptorHeap* pExistingHeap)
    : m_pHeap(pExistingHeap)
{
    m_hCPU = pExistingHeap->GetCPUDescriptorHandleForHeapStart();
    m_hGPU = pExistingHeap->GetGPUDescriptorHandleForHeapStart();
    m_desc = pExistingHeap->GetDesc();

    ComPtr<ID3D12Device> device;
    pExistingHeap->GetDevice(IID_GRAPHICS_PPV_ARGS(device.GetAddressOf()));

    m_increment = device->GetDescriptorHandleIncrementSize(m_desc.Type);
}

_Use_decl_annotations_
DescriptorHeap::DescriptorHeap(
    ID3D12Device* device,
    const D3D12_DESCRIPTOR_HEAP_DESC* pDesc) :
    m_desc{},
    m_hCPU{},
    m_hGPU{},
    m_increment(0)
{
    Create(device, pDesc);
}

_Use_decl_annotations_
DescriptorHeap::DescriptorHeap(
    ID3D12Device* device,
    D3D12_DESCRIPTOR_HEAP_TYPE type,
    D3D12_DESCRIPTOR_HEAP_FLAGS flags,
    size_t count) :
    m_desc{},
    m_hCPU{},
    m_hGPU{},
    m_increment(0)
{
    if (count > UINT32_MAX)
        throw std::exception("Too many descriptors");

    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Flags = flags;
    desc.NumDescriptors = static_cast<UINT>(count);
    desc.Type = type;
    Create(device, &desc);
}

_Use_decl_annotations_
D3D12_GPU_DESCRIPTOR_HANDLE DescriptorHeap::WriteDescriptors(
    ID3D12Device* device,
    uint32_t offsetIntoHeap,
    uint32_t totalDescriptorCount,
    const D3D12_CPU_DESCRIPTOR_HANDLE* pDescriptorRangeStarts,
    const uint32_t* pDescriptorRangeSizes,
    uint32_t descriptorRangeCount)
{
    assert((size_t(offsetIntoHeap) + size_t(totalDescriptorCount)) <= size_t(m_desc.NumDescriptors));

    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = GetCpuHandle(offsetIntoHeap);

    device->CopyDescriptors(
        1,
        &cpuHandle,
        &totalDescriptorCount,
        descriptorRangeCount,
        pDescriptorRangeStarts,
        pDescriptorRangeSizes,
        m_desc.Type);

    auto gpuHandle = GetGpuHandle(offsetIntoHeap);

    return gpuHandle;
}

_Use_decl_annotations_
D3D12_GPU_DESCRIPTOR_HANDLE DescriptorHeap::WriteDescriptors(
    ID3D12Device* device,
    uint32_t offsetIntoHeap,
    const D3D12_CPU_DESCRIPTOR_HANDLE* pDescriptorRangeStarts,
    const uint32_t* pDescriptorRangeSizes,
    uint32_t descriptorRangeCount)
{
    uint32_t totalDescriptorCount = 0;
    for (uint32_t i = 0; i < descriptorRangeCount; ++i)
        totalDescriptorCount += pDescriptorRangeSizes[i];

    return WriteDescriptors(
        device,
        offsetIntoHeap,
        totalDescriptorCount,
        pDescriptorRangeStarts,
        pDescriptorRangeSizes,
        descriptorRangeCount);
}

_Use_decl_annotations_
D3D12_GPU_DESCRIPTOR_HANDLE DescriptorHeap::WriteDescriptors(
    ID3D12Device* device,
    uint32_t offsetIntoHeap,
    const D3D12_CPU_DESCRIPTOR_HANDLE* pDescriptors,
    uint32_t descriptorCount)
{
    return WriteDescriptors(
        device,
        offsetIntoHeap,
        descriptorCount,
        pDescriptors,
        &descriptorCount,
        1);
}

_Use_decl_annotations_
void DescriptorHeap::Create(
    ID3D12Device* pDevice,
    const D3D12_DESCRIPTOR_HEAP_DESC* pDesc)
{
    assert(pDesc != nullptr);

    m_desc = *pDesc;
    m_increment = pDevice->GetDescriptorHandleIncrementSize(pDesc->Type);

    if (pDesc->NumDescriptors == 0)
    {
        m_pHeap.Reset();
        m_hCPU.ptr = 0;
        m_hGPU.ptr = 0;
    }
    else
    {
        ThrowIfFailed(pDevice->CreateDescriptorHeap(
            pDesc,
            IID_GRAPHICS_PPV_ARGS(m_pHeap.ReleaseAndGetAddressOf())));

        SetDebugObjectName(m_pHeap.Get(), L"DescriptorHeap");

        m_hCPU = m_pHeap->GetCPUDescriptorHandleForHeapStart();

        if (pDesc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)
            m_hGPU = m_pHeap->GetGPUDescriptorHandleForHeapStart();

    }
}

_Use_decl_annotations_
void DescriptorHeap::DefaultDesc(
    D3D12_DESCRIPTOR_HEAP_TYPE type,
    D3D12_DESCRIPTOR_HEAP_DESC* pDesc)
{
    assert(c_DescriptorHeapDescs[type].Type == type);
    pDesc->Flags = c_DescriptorHeapDescs[type].Flags;
    pDesc->NumDescriptors = 0;
    pDesc->Type = type;
}


//======================================================================================
// DescriptorPile
//======================================================================================

void DescriptorPile::AllocateRange(size_t numDescriptors, _Out_ IndexType& start, _Out_ IndexType& end)
{
    // make sure we didn't allocate zero
    if (numDescriptors == 0)
    {
        throw std::out_of_range("Can't allocate zero descriptors");
    }

    // get the current top
    start = m_top;

    // increment top with new request
    m_top += numDescriptors;
    end = m_top;

    // make sure we have enough room
    if (m_top > Count())
    {
        DebugTrace("DescriptorPile has %zu of %zu descriptors; failed request for %zu more\n", start, Count(), numDescriptors);
        throw std::exception("Can't allocate more descriptors");
    }
}
