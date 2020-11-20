//--------------------------------------------------------------------------------------
// File: MediaEnginePlayer.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//-------------------------------------------------------------------------------------

#include "pch.h"

#include "MediaEnginePlayer.h"

using Microsoft::WRL::ComPtr;

namespace
{
    class MediaEngineNotify : public IMFMediaEngineNotify
    {
        long m_cRef;
        IMFNotify* m_pCB;

    public:
        MediaEngineNotify() noexcept :
            m_cRef(1),
            m_pCB(nullptr)
        {
        }

        STDMETHODIMP QueryInterface(REFIID riid, void** ppv)
        {
            if (__uuidof(IMFMediaEngineNotify) == riid)
            {
                *ppv = static_cast<IMFMediaEngineNotify*>(this);
            }
            else
            {
                *ppv = nullptr;
                return E_NOINTERFACE;
            }

            AddRef();

            return S_OK;
        }

        STDMETHODIMP_(ULONG) AddRef()
        {
            return InterlockedIncrement(&m_cRef);
        }

        STDMETHODIMP_(ULONG) Release()
        {
            LONG cRef = InterlockedDecrement(&m_cRef);
            if (cRef == 0)
            {
                delete this;
            }
            return cRef;
        }

        void SetCallback(IMFNotify* pCB)
        {
            m_pCB = pCB;
        }

        // EventNotify is called when the Media Engine sends an event.
        STDMETHODIMP EventNotify(DWORD meEvent, DWORD_PTR param1, DWORD)
        {
            if (meEvent == MF_MEDIA_ENGINE_EVENT_NOTIFYSTABLESTATE)
            {
                SetEvent(reinterpret_cast<HANDLE>(param1));
            }
            else
            {
                m_pCB->OnMediaEngineEvent(meEvent);
            }

            return S_OK;
        }
    };
}

MediaEnginePlayer::MediaEnginePlayer() noexcept :
    m_bkgColor{},
    m_isPlaying(false),
    m_isInfoReady(false),
    m_isFinished(false)
{
}

MediaEnginePlayer::~MediaEnginePlayer()
{
    Shutdown();

    MFShutdown();
}

void MediaEnginePlayer::Initialize(IDXGIFactory4* dxgiFactory, ID3D12Device* device, DXGI_FORMAT format)
{
    // Initialize Media Foundation
    HRESULT hr = MFStartup(MF_VERSION);
    if (FAILED(hr))
    {
        if (hr == E_NOTIMPL)
        {
            // See https://blogs.msdn.microsoft.com/chuckw/2010/08/13/who-moved-my-windows-media-cheese/
            OutputDebugStringA("ERROR: Media Foundation components not installed");
        }
        DX::ThrowIfFailed(hr);
    }

    // Create our own device to avoid threading issues
    auto adapterLuid = device->GetAdapterLuid();

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT adapterIndex = 0;
        DXGI_ERROR_NOT_FOUND != dxgiFactory->EnumAdapters1(
            adapterIndex,
            adapter.ReleaseAndGetAddressOf());
        ++adapterIndex)
    {
        DXGI_ADAPTER_DESC1 desc;
        DX::ThrowIfFailed(adapter->GetDesc1(&desc));

        if (desc.AdapterLuid.LowPart == adapterLuid.LowPart
            && desc.AdapterLuid.HighPart == adapterLuid.HighPart)
        {
            // Found the same adapter as our DX12 device
            break;
        }
    };

#if defined(NDEBUG)
    if (!adapter)
    {
        throw std::exception("No matching device for DirectX 12 found");
    }
#else
    if (!adapter)
    {
        if (FAILED(dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf()))))
        {
            throw std::exception("WARP12 not available. Enable the 'Graphics Tools' optional feature");
        }

        DXGI_ADAPTER_DESC1 desc;
        DX::ThrowIfFailed(adapter->GetDesc1(&desc));

        if (desc.AdapterLuid.LowPart != adapterLuid.LowPart
            || desc.AdapterLuid.HighPart != adapterLuid.HighPart)
        {
            throw std::exception("No matching device for DirectX 12 found");
        }
    }
#endif

    static const D3D_FEATURE_LEVEL s_featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    };

    D3D12_FEATURE_DATA_FEATURE_LEVELS featLevels =
    {
        _countof(s_featureLevels), s_featureLevels, D3D_FEATURE_LEVEL_11_0
    };

    hr = device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &featLevels, sizeof(featLevels));
    D3D_FEATURE_LEVEL fLevel = D3D_FEATURE_LEVEL_11_0;
    if (SUCCEEDED(hr))
    {
        fLevel = featLevels.MaxSupportedFeatureLevel;
    }

    ComPtr<ID3D11Device> baseDevice;
    DX::ThrowIfFailed(
        D3D11CreateDevice(
            adapter.Get(),
#if defined(NDEBUG)
            D3D_DRIVER_TYPE_UNKNOWN,
#else
            adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_WARP,
#endif
            nullptr,
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            &fLevel,
            1,
            D3D11_SDK_VERSION,
            baseDevice.GetAddressOf(),
            nullptr,
            nullptr
        ));

    ComPtr<ID3D10Multithread> multithreaded;
    DX::ThrowIfFailed(baseDevice.As(&multithreaded));
    multithreaded->SetMultithreadProtected(TRUE);

    DX::ThrowIfFailed(baseDevice.As(&m_device));

    // Setup Media Engine
    Microsoft::WRL::ComPtr<IMFDXGIDeviceManager> dxgiManager;
    UINT resetToken;
    DX::ThrowIfFailed(MFCreateDXGIDeviceManager(&resetToken, dxgiManager.GetAddressOf()));
    DX::ThrowIfFailed(dxgiManager->ResetDevice(m_device.Get(), resetToken));

    // Create our event callback object.
    ComPtr<MediaEngineNotify> spNotify = new MediaEngineNotify();
    if (spNotify == nullptr)
    {
        DX::ThrowIfFailed(E_OUTOFMEMORY);
    }

    spNotify->SetCallback(this);

    // Set configuration attribiutes.
    ComPtr<IMFAttributes> attributes;
    DX::ThrowIfFailed(MFCreateAttributes(attributes.GetAddressOf(), 1));
    DX::ThrowIfFailed(attributes->SetUnknown(MF_MEDIA_ENGINE_DXGI_MANAGER, reinterpret_cast<IUnknown*>(dxgiManager.Get())));
    DX::ThrowIfFailed(attributes->SetUnknown(MF_MEDIA_ENGINE_CALLBACK, reinterpret_cast<IUnknown*>(spNotify.Get())));
    DX::ThrowIfFailed(attributes->SetUINT32(MF_MEDIA_ENGINE_VIDEO_OUTPUT_FORMAT, format));

    // Create MediaEngine.
    ComPtr<IMFMediaEngineClassFactory> mfFactory;
    DX::ThrowIfFailed(
        CoCreateInstance(CLSID_MFMediaEngineClassFactory,
            nullptr,
            CLSCTX_ALL,
            IID_PPV_ARGS(mfFactory.GetAddressOf())));

    DX::ThrowIfFailed(
        mfFactory->CreateInstance(0,
            attributes.Get(),
            m_mediaEngine.ReleaseAndGetAddressOf()));

    // Create MediaEngineEx
    DX::ThrowIfFailed(m_mediaEngine.As(&m_engineEx));
}

void MediaEnginePlayer::Shutdown()
{
    if (m_mediaEngine)
    {
        m_mediaEngine->Shutdown();
    }
}

void MediaEnginePlayer::Play()
{
    if (m_isPlaying)
        return;

    if (m_mediaEngine)
    {
        DX::ThrowIfFailed(m_mediaEngine->Play());
        m_isPlaying = true;
        m_isFinished = false;
    }
}

void MediaEnginePlayer::SetLoop(bool loop)
{
    if (m_mediaEngine)
    {
        DX::ThrowIfFailed(m_mediaEngine->SetLoop(loop));
    }
}

void MediaEnginePlayer::Pause()
{
    if (!m_isPlaying || m_isFinished)
        return;

    if (m_mediaEngine)
    {
        DX::ThrowIfFailed(m_mediaEngine->Pause());
        m_isPlaying = false;
    }
}

void MediaEnginePlayer::SetMuted(bool muted)
{
    if (m_mediaEngine)
    {
        DX::ThrowIfFailed(m_mediaEngine->SetMuted(muted));
    }
}

void MediaEnginePlayer::SetSource(_In_z_ const wchar_t* sourceUri)
{
    static BSTR bstrURL = nullptr;

    if (bstrURL != nullptr)
    {
        ::CoTaskMemFree(bstrURL);
        bstrURL = nullptr;
    }

    size_t cchAllocationSize = 1 + ::wcslen(sourceUri);
    bstrURL = reinterpret_cast<BSTR>(::CoTaskMemAlloc(sizeof(wchar_t)*(cchAllocationSize)));

    if (bstrURL == 0)
    {
        DX::ThrowIfFailed(E_OUTOFMEMORY);
    }

    wcscpy_s(bstrURL, cchAllocationSize, sourceUri);

    m_isPlaying = false;
    m_isInfoReady = false;
    m_isFinished = false;

    if (m_mediaEngine)
    {
        DX::ThrowIfFailed(m_mediaEngine->SetSource(bstrURL));
    }
}

bool MediaEnginePlayer::TransferFrame(HANDLE textureHandle, MFVideoNormalizedRect rect, RECT rcTarget)
{
    if (m_mediaEngine != nullptr && m_isPlaying)
    {
        LONGLONG pts;
        if (m_mediaEngine->OnVideoStreamTick(&pts) == S_OK)
        {
            ComPtr<ID3D11Texture2D> mediaTexture;
            if (SUCCEEDED(m_device->OpenSharedResource1(textureHandle, IID_PPV_ARGS(mediaTexture.GetAddressOf()))))
            {
                if (m_mediaEngine->TransferVideoFrame(mediaTexture.Get(), &rect, &rcTarget, &m_bkgColor) == S_OK)
                    return true;
            }
        }
    }

    return false;
}

void MediaEnginePlayer::OnMediaEngineEvent(uint32_t meEvent)
{
    switch (meEvent)
    {
    case MF_MEDIA_ENGINE_EVENT_LOADEDMETADATA:
        m_isInfoReady = true;
        break;

    case MF_MEDIA_ENGINE_EVENT_CANPLAY:

        // Here we auto-play when ready...
        Play();

        break;

    case MF_MEDIA_ENGINE_EVENT_PLAY:
        break;

    case MF_MEDIA_ENGINE_EVENT_PAUSE:
        break;

    case MF_MEDIA_ENGINE_EVENT_ENDED:
        m_isFinished = true;
        break;

    case MF_MEDIA_ENGINE_EVENT_TIMEUPDATE:
        break;

        case MF_MEDIA_ENGINE_EVENT_ERROR:
            #ifdef _DEBUG
            if (m_mediaEngine)
            {
                ComPtr<IMFMediaError> error;
                m_mediaEngine->GetError(&error);
                USHORT errorCode = error->GetErrorCode();
                char buff[128] = {};
                sprintf_s(buff, "ERROR: Media Foundation Event Error %u", errorCode);
                OutputDebugStringA(buff);
            }
            #endif
            break;
    }
}

void MediaEnginePlayer::GetNativeVideoSize(uint32_t& cx, uint32_t& cy)
{
    cx = cy = 0;
    if (m_mediaEngine && m_isInfoReady)
    {
        DWORD x, y;
        DX::ThrowIfFailed(m_mediaEngine->GetNativeVideoSize(&x, &y));

        cx = x;
        cy = y;
    }
}
