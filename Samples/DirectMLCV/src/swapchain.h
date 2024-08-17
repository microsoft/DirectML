#pragma once

#include <vector>

class SwapChain
{
public:
    struct FrameBuffer
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> renderTarget;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocator;
        uint64_t fenceValue;
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
    };

    SwapChain(
        ID3D12Device* device, 
        ID3D12CommandQueue* commandQueue, 
        HWND hwnd, 
        uint32_t bufferCount, 
        uint32_t width, 
        uint32_t height, 
        DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM
    );

    void Resize(uint32_t width, uint32_t height);

    // Moves to the next available back buffer. Blocks until the buffer is ready 
    // and any previous work submitted to the buffer has finished. If a command list
    // is provided, it will be reset and a transition barrier added for the render target.
    FrameBuffer& StartFrame(ID3D12GraphicsCommandList* commandList = nullptr);

    // Presents the current back buffer. If a command list is provided, a transition barrier
    // is added for the render target before closing and executing it.
    void EndFrame(ID3D12GraphicsCommandList* commandList = nullptr);

    void WaitForLastFrame();

    size_t GetBufferCount() const { return m_frameBuffers.size(); }
    DXGI_FORMAT GetFormat() const { return m_format; }
    FrameBuffer& operator[](size_t index) { return m_frameBuffers[index]; }

private:
    Microsoft::WRL::ComPtr<ID3D12Device> m_device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_rtvDescriptorHeap;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> m_swapChain;

    std::vector<FrameBuffer> m_frameBuffers;
    wil::unique_handle m_swapChainWaitableObject;

    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    wil::unique_handle m_fenceEvent;
    uint64_t m_fenceValue = 0;

    DXGI_FORMAT m_format;

    bool m_frameStarted = false;
};