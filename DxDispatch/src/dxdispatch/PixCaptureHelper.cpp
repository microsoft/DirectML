#include "pch.h"
#include "PixCaptureHelper.h"

static constexpr GUID PIX_EVAL_CAPTURABLE_WORK_GUID = {
    0xd113b493,
    0xbba2,
    0x4993,
    {0x86, 0x08, 0xd7, 0x06, 0xa7, 0x3b, 0x91, 0xce} };

#if _WIN32
static std::wstring Utf8ToWideString(std::string_view utf8String)
{
    if (utf8String.empty())
    {
        return std::wstring();
    }
    else if (uint64_t(utf8String.size()) > std::numeric_limits<int32_t>::max())
    {
        throw std::runtime_error("String too long to convert to wide string");
    }

    int requiredSize = MultiByteToWideChar(CP_UTF8, 0, utf8String.data(), static_cast<int32_t>(utf8String.size()), nullptr, 0);
    std::wstring result(requiredSize, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8String.data(), static_cast<int32_t>(utf8String.size()), result.data(), static_cast<int32_t>(result.size()));
    return result;
}
#endif

PixCaptureHelper::PixCaptureHelper(PixCaptureType captureType, std::string_view captureName)
{
    m_captureType = captureType;

#ifdef _WIN32
    m_captureName = Utf8ToWideString(captureName);
#endif

#if !defined(_GAMING_XBOX) && !defined(PIX_NONE)
    if (captureType == PixCaptureType::ProgrammaticGpu)
    {
        m_gpuCaptureLibrary.reset(PIXLoadLatestWinPixGpuCapturerLibrary());
    }
#endif
}

void PixCaptureHelper::Initialize(ID3D12CommandQueue* commandQueue)
{
    m_commandQueue = commandQueue;

#if !defined(_GAMING_XBOX) && !defined(PIX_NONE)
    if (m_captureType == PixCaptureType::Manual)
    {
        // Ignore HRESULT since this is only expected to succeed when running under PIX.
        m_commandQueue->QueryInterface(IID_PPV_ARGS(&m_sharingContract));
    }
#endif
}

HRESULT PixCaptureHelper::BeginCapturableWork()
{
    switch (m_captureType)
    {
        case PixCaptureType::ProgrammaticTiming:
        {
#ifdef _GAMING_XBOX
            auto captureName = m_captureName + L".pevt";
            PIXCaptureParameters captureParams = {};
            captureParams.TimingCaptureParameters.CaptureGpuTiming = TRUE;
            captureParams.TimingCaptureParameters.CaptureCallstacks = TRUE;
            captureParams.TimingCaptureParameters.CaptureCpuSamples = TRUE;
            captureParams.TimingCaptureParameters.CpuSamplesPerSecond = 4000;
            captureParams.TimingCaptureParameters.CaptureStorage = PIXCaptureParameters::Memory;
            captureParams.TimingCaptureParameters.FileName = captureName.data();
            captureParams.TimingCaptureParameters.MaximumToolingMemorySizeMb = 4096;
            return PIXBeginCapture(PIX_CAPTURE_TIMING, &captureParams);
#else
            // There is currently no programmatic API for timing captures on Windows.
            return E_NOTIMPL;
#endif
        }

        case PixCaptureType::ProgrammaticGpu:
        {
#if defined(PIX_NONE)
            return E_NOTIMPL;
#elif defined(_GAMING_XBOX)
            auto captureName = m_captureName + L".xpix";
            return m_commandQueue->PIXGpuBeginCapture(0, captureName.c_str());
#else
            if (!m_gpuCaptureLibrary)
            {
                throw std::runtime_error("The WinPix GPU capturer library was not found. Ensure PIX is installed.");
            }

            auto captureName = m_captureName + L".wpix";

            // PIXBeginCapture can only be used for GPU captures on Windows.
            PIXCaptureParameters captureParams = {};
            captureParams.TimingCaptureParameters.FileName = captureName.c_str();
            return PIXBeginCapture(PIX_CAPTURE_GPU, &captureParams);
#endif
        }

        case PixCaptureType::Manual:
        {
#if !defined(_GAMING_XBOX) && !defined(PIX_NONE)
            if (m_sharingContract)
            {
                m_sharingContract->BeginCapturableWork(PIX_EVAL_CAPTURABLE_WORK_GUID);
            }
#endif

            return S_OK;
        }
        default:
            break;
    }

    return E_UNEXPECTED;
}

HRESULT PixCaptureHelper::EndCapturableWork()
{
    switch (m_captureType)
    {
        case PixCaptureType::ProgrammaticTiming:
        {
#if defined(_GAMING_XBOX)
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
#else
            // There is currently no programmatic API for timing captures on Windows.
            return E_NOTIMPL;
#endif
        }

        case PixCaptureType::ProgrammaticGpu:
        {
#if defined(PIX_NONE)
            return E_NOTIMPL;
#elif defined(_GAMING_XBOX)
            return m_commandQueue->PIXGpuEndCapture();
#else
            // PIX on Windows ignores the discard parameter.
            return PIXEndCapture(/*discard*/FALSE);
#endif
        }

        case PixCaptureType::Manual:
        {
#if !defined(_GAMING_XBOX) && !defined(PIX_NONE)
            if (m_sharingContract)
            {
                m_sharingContract->EndCapturableWork(PIX_EVAL_CAPTURABLE_WORK_GUID);
            }
#endif
            return S_OK;
        }
        default:
            break;
    }

    return E_UNEXPECTED;
}