#include "pch.h"
#include "app.h"
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"

using Microsoft::WRL::ComPtr;

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
    HWND hWnd, 
    UINT msg, 
    WPARAM wParam, 
    LPARAM lParam
);

static LRESULT WINAPI WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
    {
        return true;
    }

    App* app = reinterpret_cast<App*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

    switch (msg)
    {
        case WM_CREATE:
            {
                LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
                SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
                return DefWindowProc(hWnd, msg, wParam, lParam);
            }
        
        case WM_SIZE:
            if (app && wParam != SIZE_MINIMIZED)
            {
                app->Resize((UINT)LOWORD(lParam), (UINT)HIWORD(lParam));
            }
            return 0;

        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            {
                return 0;
            }
            break;

        case WM_DESTROY:
            if (app)
            {
                app->Stop();
            }
            ::PostQuitMessage(0);
            return 0;
    }

    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

App::App(uint32_t windowWidth, uint32_t windowHeight, std::wstring name) : 
    m_windowWidth(windowWidth),
    m_windowHeight(windowHeight),
    m_name(name),
    m_aspectRatio(static_cast<float>(windowWidth) / windowHeight)
{
}

int App::Start(HINSTANCE hInstance, int nCmdShow)
{
    WNDCLASSEX windowClass = {};
    windowClass.cbSize = sizeof(WNDCLASSEX);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = WindowProc;
    windowClass.hInstance = hInstance;
    windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    windowClass.lpszClassName = L"OnnxEditorAppClass";
    RegisterClassEx(&windowClass);

    RECT windowRect = {0, 0, static_cast<LONG>(m_windowWidth), static_cast<LONG>(m_windowHeight)};
    AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);

    m_hwnd = CreateWindow(
        windowClass.lpszClassName,
        m_name.c_str(),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        windowRect.right - windowRect.left,
        windowRect.bottom - windowRect.top,
        nullptr, // parent window
        nullptr, // menu
        hInstance,
        this
    );

#if defined(_DEBUG)
    ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
    {
        debugController->EnableDebugLayer();
        // debugController->SetEnableGPUBasedValidation(true);
    }
#endif

    THROW_IF_FAILED(D3D12CreateDevice(
        nullptr,
        D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(&m_device)
    ));

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    THROW_IF_FAILED(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

    m_swapChain.emplace(m_device.Get(), m_commandQueue.Get(), m_hwnd, 2, m_windowWidth, m_windowHeight);

    THROW_IF_FAILED(m_device->CreateCommandList(
        0, 
        queueDesc.Type, 
        (*m_swapChain)[0].commandAllocator.Get(), 
        nullptr, 
        IID_PPV_ARGS(&m_commandList)
    ));
    THROW_IF_FAILED(m_commandList->Close());

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = 1;
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    THROW_IF_FAILED(m_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_srvHeap)));

    ShowWindow(m_hwnd, nCmdShow);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();

    ImGui_ImplWin32_Init(m_hwnd);
    ImGui_ImplDX12_Init(
        m_device.Get(), 
        m_swapChain->GetBufferCount(),
        m_swapChain->GetFormat(),
        m_srvHeap.Get(),
        m_srvHeap->GetCPUDescriptorHandleForHeapStart(),
        m_srvHeap->GetGPUDescriptorHandleForHeapStart()
    );

    MSG msg = {};
    while (m_running)
    {
        Render();

        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    m_swapChain->WaitForLastFrame();

    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    DestroyWindow(m_hwnd);
    UnregisterClassW(windowClass.lpszClassName, windowClass.hInstance);

    return static_cast<char>(msg.wParam);
}

void App::Stop()
{
    m_running = false;
}

void App::Resize(UINT width, UINT height)
{
    m_swapChain->Resize(width, height);
    m_windowWidth = width;
    m_windowHeight = height;
}

void App::Render()
{
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    // m_userInterface.RenderFrame(m_windowWidth, m_windowHeight);
    ImGui::Render();

    SwapChain::FrameBuffer& backBuffer = m_swapChain->StartFrame(m_commandList.Get());

    const float clearColor[4] = { 1, 0, 1, 1 };
    m_commandList->ClearRenderTargetView(backBuffer.rtvHandle, clearColor, 0, nullptr);
    m_commandList->OMSetRenderTargets(1, &backBuffer.rtvHandle, false, nullptr);
    m_commandList->SetDescriptorHeaps(1, m_srvHeap.GetAddressOf());
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), m_commandList.Get());

    m_swapChain->EndFrame(m_commandList.Get());
}