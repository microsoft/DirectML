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
    // Must be constructed *before* the D3D device: Windows PIX GPU capture library needs to load first.
    PixCaptureHelper(PixCaptureType captureType, std::string_view captureName);

    // Must be called *after* the D3D device when a command queue is available.
    void Initialize(ID3D12CommandQueue* commandQueue);

    HRESULT BeginCapturableWork();
    HRESULT EndCapturableWork();
    PixCaptureType GetPixCaptureType() const { return m_captureType; }

private:
    PixCaptureType m_captureType;
    std::wstring m_captureName;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
#if !defined(_GAMING_XBOX) && !defined(PIX_NONE)
    wil::unique_hmodule m_gpuCaptureLibrary;
    Microsoft::WRL::ComPtr<ID3D12SharingContract> m_sharingContract;
#endif
};