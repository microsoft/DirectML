#include "pch.h"
#include "PixCaptureHelper.h"

#ifndef PIX_NONE
static std::wstring Utf8ToWideString(std::string_view utf8String)
{
    if (utf8String.empty())
    {
        return std::wstring();
    }

    int requiredSize = MultiByteToWideChar(CP_UTF8, 0, utf8String.data(), utf8String.size(), nullptr, 0);
    std::wstring result(requiredSize, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8String.data(), utf8String.size(), result.data(), result.size());
    return result;
}

std::wstring GetCaptureName(const std::string& commandName, std::wstring extension)
{
    return L"dxdispatch_" + Utf8ToWideString(commandName) + extension;
}
#endif


#ifdef PIX_NONE
// Stubs when PIX isn't built into DxDispatch
PixCaptureHelper::PixCaptureHelper(PixCaptureType captureType) { }
void PixCaptureHelper::Initialize(ID3D12CommandQueue* commandQueue) { }
HRESULT PixCaptureHelper::BeginCapturableWork(std::string commandName) { return S_OK; }
HRESULT PixCaptureHelper::EndCapturableWork() { return S_OK; }

#elif _GAMING_XBOX

PixCaptureHelper::PixCaptureHelper(PixCaptureType captureType)
{
    m_captureType = captureType;
}

void PixCaptureHelper::Initialize(ID3D12CommandQueue* commandQueue)
{
    m_commandQueue = commandQueue;
}

HRESULT PixCaptureHelper::BeginCapturableWork(std::string commandName)
{
    switch (m_captureType)
    {
        case PixCaptureType::ProgrammaticTiming:
        {
            auto captureName = GetCaptureName(commandName, L".pevt");

            PIXCaptureParameters captureParams = {};
            captureParams.TimingCaptureParameters.CaptureGpuTiming = TRUE;
            captureParams.TimingCaptureParameters.CaptureCallstacks = TRUE;
            captureParams.TimingCaptureParameters.CaptureCpuSamples = TRUE;
            captureParams.TimingCaptureParameters.CpuSamplesPerSecond = 4000;
            captureParams.TimingCaptureParameters.CaptureStorage = PIXCaptureParameters::Memory;
            captureParams.TimingCaptureParameters.FileName = captureName.c_str();
            captureParams.TimingCaptureParameters.MaximumToolingMemorySizeMb = 4096;
            return PIXBeginCapture(PIX_CAPTURE_TIMING, &captureParams);
        }

        case PixCaptureType::ProgrammaticGpu:
        {
            auto captureName = GetCaptureName(commandName, L".xpix");
            return m_commandQueue->PIXGpuBeginCapture(0, captureName.c_str());
        }

        case PixCaptureType::Manual:
        {
            return E_NOTIMPL;
        }

        case PixCaptureType::None:
        {
            return S_OK;
        }
    }

    return E_UNEXPECTED;
}

HRESULT PixCaptureHelper::EndCapturableWork()
{
    switch (m_captureType)
    {
        case PixCaptureType::ProgrammaticTiming:
        {
            HRESULT hr;
            do
            {
                hr = PIXEndCapture(/*discard*/FALSE);
                if (hr == E_PENDING)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            } while (hr == E_PENDING);

            return hr;
        }

        case PixCaptureType::ProgrammaticGpu:
        {
            return m_commandQueue->PIXGpuEndCapture();
        }

        case PixCaptureType::Manual:
        {
            return E_NOTIMPL;
        }

        case PixCaptureType::None:
        {
            return S_OK;
        }
    }

    return E_UNEXPECTED;
}

#else // Windows with PIX enabled

static constexpr GUID PIX_EVAL_CAPTURABLE_WORK_GUID = {
    0xd113b493,
    0xbba2,
    0x4993,
    {0x86, 0x08, 0xd7, 0x06, 0xa7, 0x3b, 0x91, 0xce} };

PixCaptureHelper::PixCaptureHelper(PixCaptureType captureType)
{
    m_captureType = captureType;
    if (captureType == PixCaptureType::ProgrammaticGpu)
    {
        m_gpuCaptureLibrary.reset(PIXLoadLatestWinPixGpuCapturerLibrary());
    }
}

void PixCaptureHelper::Initialize(ID3D12CommandQueue* commandQueue)
{
    m_commandQueue = commandQueue;
    if (m_captureType == PixCaptureType::Manual)
    {
        // Ignore HRESULT since this is only expected to succeed when running under PIX.
        m_commandQueue->QueryInterface(IID_PPV_ARGS(&m_sharingContract));
    }
}

HRESULT PixCaptureHelper::BeginCapturableWork(std::string commandName)
{
    switch (m_captureType)
    {
        case PixCaptureType::ProgrammaticTiming:
        {
            // There is currently no programmatic API for timing captures on Windows.
            return E_NOTIMPL;
        }

        case PixCaptureType::ProgrammaticGpu:
        {
            auto captureName = GetCaptureName(commandName, L".wpix");

            // PIXBeginCapture can only be used for GPU captures on Windows.
            PIXCaptureParameters captureParams = {};
            captureParams.TimingCaptureParameters.FileName = captureName.c_str();
            return PIXBeginCapture(PIX_CAPTURE_GPU, &captureParams);
        }

        case PixCaptureType::Manual:
        {
            if (m_sharingContract)
            {
                m_sharingContract->BeginCapturableWork(PIX_EVAL_CAPTURABLE_WORK_GUID);
            }

            return S_OK;
        }

        case PixCaptureType::None:
        {
            return S_OK;
        }
    }

    return E_UNEXPECTED;
}

HRESULT PixCaptureHelper::EndCapturableWork()
{
    switch (m_captureType)
    {
        case PixCaptureType::ProgrammaticTiming:
        {
            // There is currently no programmatic API for timing captures on Windows.
            return E_NOTIMPL;
        }

        case PixCaptureType::ProgrammaticGpu:
        {
            // PIX on Windows ignores the discard parameter.
            return PIXEndCapture(/*discard*/FALSE);
        }

        case PixCaptureType::Manual:
        {
            if (m_sharingContract)
            {
                m_sharingContract->EndCapturableWork(PIX_EVAL_CAPTURABLE_WORK_GUID);
            }

            return S_OK;
        }

        case PixCaptureType::None:
        {
            return S_OK;
        }
    }

    return E_UNEXPECTED;
}

#endif