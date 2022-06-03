#pragma once

enum class PixCaptureType
{
    // (Begin|End)CapturableWork will always record a timing capture.
    ProgrammaticTiming,

    // (Begin|End)CapturableWork will always record a GPU capture.
    ProgrammaticGpu,

    // (Begin|End)CapturableWork will record a timing or GPU capture if launched 
    // within PIX and the user clicks the button to take a capture.
    Manual,

    // (Begin|End)CapturableWork will be a noop.
    None
};

class PixCaptureHelper
{
public:
    // Must be constructed *before* the D3D device: Windows PIX gpu capture library needs to load first.
    PixCaptureHelper(PixCaptureType captureType);

    // Must be called *after* the D3D device when a command queue is available.
    void Initialize(ID3D12CommandQueue* commandQueue);

    HRESULT BeginCapturableWork(std::string commandName);
    HRESULT EndCapturableWork();

private:
    PixCaptureType m_captureType;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
#ifndef _GAMING_XBOX
    wil::unique_hmodule m_gpuCaptureLibrary;
    Microsoft::WRL::ComPtr<ID3D12SharingContract> m_sharingContract;
#endif
};