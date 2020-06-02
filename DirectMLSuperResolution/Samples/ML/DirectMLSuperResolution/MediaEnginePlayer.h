//--------------------------------------------------------------------------------------
// File: MediaEnginePlayer.h
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//-------------------------------------------------------------------------------------

#pragma once

// Media Foundation needs this, but it seems to be compiled out (of wingdi.h) on Desktop.
// Not sure why; MF should support Desktop fine.
typedef struct tagBITMAPINFOHEADER {
    DWORD      biSize;
    LONG       biWidth;
    LONG       biHeight;
    WORD       biPlanes;
    WORD       biBitCount;
    DWORD      biCompression;
    DWORD      biSizeImage;
    LONG       biXPelsPerMeter;
    LONG       biYPelsPerMeter;
    DWORD      biClrUsed;
    DWORD      biClrImportant;
} BITMAPINFOHEADER, FAR *LPBITMAPINFOHEADER, *PBITMAPINFOHEADER;

#include <mfapi.h>
#include <mfmediaengine.h>


//-------------------------------------------------------------------------------------
class IMFNotify
{
public:
    virtual ~IMFNotify() = default;

    IMFNotify(const IMFNotify&) = delete;
    IMFNotify& operator=(const IMFNotify&) = delete;

    IMFNotify(IMFNotify&&) = delete;
    IMFNotify& operator=(IMFNotify&&) = delete;

    virtual void OnMediaEngineEvent(uint32_t meEvent) = 0;

protected:
    IMFNotify() = default;
};


//-------------------------------------------------------------------------------------
class MediaEnginePlayer : public IMFNotify
{
public:
    MediaEnginePlayer() noexcept;
    ~MediaEnginePlayer();

    MediaEnginePlayer(const MediaEnginePlayer&) = delete;
    MediaEnginePlayer& operator=(const MediaEnginePlayer&) = delete;

    MediaEnginePlayer(MediaEnginePlayer&&) = default;
    MediaEnginePlayer& operator=(MediaEnginePlayer&&) = default;

    void Initialize(IDXGIFactory4* dxgiFactory, ID3D12Device* device, DXGI_FORMAT format);
    void Shutdown();

    void Play();
    void Pause();
    void SetLoop(bool loop);
    void SetMuted(bool muted);

    void SetSource(_In_z_ const wchar_t* sourceUri);

    bool TransferFrame(HANDLE textureHandle, MFVideoNormalizedRect rect, RECT rcTarget);

    // Callbacks
    void OnMediaEngineEvent(uint32_t meEvent) override;

    // Properties
    void GetNativeVideoSize(uint32_t& cx, uint32_t& cy);
    bool IsPlaying() const { return m_isPlaying; }
    bool IsInfoReady() const { return m_isInfoReady; }
    bool IsFinished() const { return m_isFinished; }

private:
    Microsoft::WRL::ComPtr<ID3D11Device1>       m_device;
    Microsoft::WRL::ComPtr<IMFMediaEngine>      m_mediaEngine;
    Microsoft::WRL::ComPtr<IMFMediaEngineEx>    m_engineEx;

    MFARGB  m_bkgColor;

    bool m_isPlaying;
    bool m_isInfoReady;
    bool m_isFinished;
};
