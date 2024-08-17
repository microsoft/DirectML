#include "pch.h"
#include "SwapChain.h"
#include <stdexcept>

using Microsoft::WRL::ComPtr;

SwapChain::SwapChain(
    ID3D12Device* device, 
    ID3D12CommandQueue* commandQueue, 
    HWND hwnd, 
    uint32_t bufferCount, 
    uint32_t width, 
    uint32_t height, 
    DXGI_FORMAT format) : 
    m_device(device), 
    m_commandQueue(commandQueue), 
    m_format(format)
{
    UINT dxgiFactoryFlags = 0;

#if defined(_DEBUG)
    dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif

    ComPtr<IDXGIFactory4> factory;
    THROW_IF_FAILED(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = bufferCount;
    swapChainDesc.Width = width;
    swapChainDesc.Height = height;
    swapChainDesc.Format = format;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

    ComPtr<IDXGISwapChain1> swapChain;
    THROW_IF_FAILED(factory->CreateSwapChainForHwnd(
        commandQueue,
        hwnd,
        &swapChainDesc,
        nullptr,
        nullptr,
        &swapChain
    ));

    THROW_IF_FAILED(swapChain.As(&m_swapChain));

    THROW_IF_FAILED(factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER));

    THROW_IF_FAILED(m_swapChain->SetMaximumFrameLatency(bufferCount));

    m_swapChainWaitableObject.reset(m_swapChain->GetFrameLatencyWaitableObject());

    D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc = {};
    rtvDescriptorHeapDesc.NumDescriptors = bufferCount;
    rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    THROW_IF_FAILED(m_device->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(&m_rtvDescriptorHeap)));

    m_fenceEvent.reset(CreateEvent(nullptr, FALSE, FALSE, nullptr));
    if (!m_fenceEvent)
    {
        THROW_IF_FAILED(HRESULT_FROM_WIN32(GetLastError()));
    }

    THROW_IF_FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
    m_fenceValue = 0;

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
    auto rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    m_frameBuffers.resize(bufferCount);

    uint32_t i = 0;
    for (auto& frameBuffer : m_frameBuffers)
    {
        THROW_IF_FAILED(m_swapChain->GetBuffer(i++, IID_PPV_ARGS(&frameBuffer.renderTarget)));
        
        THROW_IF_FAILED(m_device->CreateCommandAllocator(
            commandQueue->GetDesc().Type, 
            IID_PPV_ARGS(&frameBuffer.commandAllocator)
        ));

        frameBuffer.fenceValue = 0;

        frameBuffer.rtvHandle = rtvHandle;
        rtvHandle.Offset(1, rtvDescriptorSize);

        m_device->CreateRenderTargetView(frameBuffer.renderTarget.Get(), nullptr, frameBuffer.rtvHandle);
    }
}

void SwapChain::Resize(uint32_t width, uint32_t height)
{
    WaitForLastFrame();

    // Release existing references to the swap chain.
    for (auto& frameBuffer : m_frameBuffers)
    {
        frameBuffer.renderTarget = nullptr;
    }

    THROW_IF_FAILED(m_swapChain->ResizeBuffers(
        0,
        width,
        height,
        DXGI_FORMAT_UNKNOWN,
        DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT
    ));

    uint32_t i = 0;
    for (auto& frameBuffer : m_frameBuffers)
    {
        THROW_IF_FAILED(m_swapChain->GetBuffer(i++, IID_PPV_ARGS(&frameBuffer.renderTarget)));
        m_device->CreateRenderTargetView(frameBuffer.renderTarget.Get(), nullptr, frameBuffer.rtvHandle);
    }
}

void SwapChain::WaitForLastFrame()
{
    if (m_frameStarted)
    {
        throw std::logic_error("Cannot wait in the middle of a frame");
    }

    if (m_fence->GetCompletedValue() < m_fenceValue)
    {
        m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent.get());
        WaitForSingleObject(m_fenceEvent.get(), INFINITE);
    }
}

SwapChain::FrameBuffer& SwapChain::StartFrame(ID3D12GraphicsCommandList* commandList)
{
    auto& backBuffer = m_frameBuffers[m_swapChain->GetCurrentBackBufferIndex()];

    // Wait on the swap chain to wake up as soon as the system is ready to start a frame.
    HANDLE waitableObjects[2] = { m_swapChainWaitableObject.get(), nullptr };
    uint32_t waitableObjectsSize = 1;

    if (m_fence->GetCompletedValue() < backBuffer.fenceValue)
    {
        // We've run out of buffered command allocators, so we also need to wait
        // for outstanding GPU work to finish.
        m_fence->SetEventOnCompletion(backBuffer.fenceValue, m_fenceEvent.get());
        waitableObjects[1] = m_fenceEvent.get();
        waitableObjectsSize++;
    }

    WaitForMultipleObjects(waitableObjectsSize, waitableObjects, true, INFINITE);

    THROW_IF_FAILED(backBuffer.commandAllocator->Reset());

    m_frameStarted = true;

    if (commandList)
    {
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            backBuffer.renderTarget.Get(), 
            D3D12_RESOURCE_STATE_PRESENT, 
            D3D12_RESOURCE_STATE_RENDER_TARGET);

        commandList->Reset(backBuffer.commandAllocator.Get(), nullptr);
        commandList->ResourceBarrier(1, &barrier);
    }

    return backBuffer;
}

void SwapChain::EndFrame(ID3D12GraphicsCommandList* commandList)
{
    if (!m_frameStarted)
    {
        throw std::logic_error("EndFrame cannot be called without first calling StartFrame");
    }

    auto& backBuffer = m_frameBuffers[m_swapChain->GetCurrentBackBufferIndex()];

    if (commandList)
    {
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            backBuffer.renderTarget.Get(), 
            D3D12_RESOURCE_STATE_RENDER_TARGET, 
            D3D12_RESOURCE_STATE_PRESENT);

        commandList->ResourceBarrier(1, &barrier);
        THROW_IF_FAILED(commandList->Close());

        ID3D12CommandList* commandLists[] = { commandList };
        m_commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
    }

    m_swapChain->Present(1, 0); // Present with vsync

    m_commandQueue->Signal(m_fence.Get(), ++m_fenceValue);
    backBuffer.fenceValue = m_fenceValue;

    m_frameStarted = false;
}