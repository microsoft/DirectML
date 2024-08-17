#pragma once

#include "pch.h"
#include <optional>
#include <string>
#include "SwapChain.h"

class App
{
public:
    App(uint32_t windowWidth, uint32_t windowHeight, std::wstring name);

    int Start(HINSTANCE hInstance, int nCmdShow);
    void Stop();
    void Render();
    void Resize(UINT width, UINT height);

private:
    uint32_t m_windowWidth = 0;
    uint32_t m_windowHeight = 0;
    std::wstring m_name;
    float m_aspectRatio = 1.0f;
    HWND m_hwnd = 0;
    bool m_running = true;

    Microsoft::WRL::ComPtr<ID3D12Device5> m_device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList5> m_commandList;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_srvHeap;
    std::optional<SwapChain> m_swapChain;
};