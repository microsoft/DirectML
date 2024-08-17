#pragma once

#define UNICODE
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NODRAWTEXT
#define NOGDI
#define NOBITMAP
#define NOMCX
#define NOSERVICE
#define NOHELP

#include <Windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <wil/result.h>
#include <wil/resource.h>

#include <d3d12.h>
#include <dxgi1_6.h>
// #include "d3dx12.h"

#include <optional>
#include <string>

using Microsoft::WRL::ComPtr;

void CopyPixelDataFromImageFilename(std::wstring_view filename)
{
    ComPtr<IWICImagingFactory> wicFactory;
    THROW_IF_FAILED(CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&wicFactory)
    ));

    ComPtr<IWICBitmapDecoder> decoder;
    THROW_IF_FAILED(wicFactory->CreateDecoderFromFilename(
        filename.data(),
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnLoad,
        &decoder
    ));

    UINT frameCount;
    THROW_IF_FAILED(decoder->GetFrameCount(&frameCount));

    ComPtr<IWICBitmapFrameDecode> frame;
    THROW_IF_FAILED(decoder->GetFrame(0, &frame));

    UINT width, height;
    THROW_IF_FAILED(frame->GetSize(&width, &height));


    WICPixelFormatGUID pixelFormat;
    THROW_IF_FAILED(frame->GetPixelFormat(&pixelFormat));

    ComPtr<IWICBitmapSource> bitmapSource = frame;

    // convert to 24bppRGB (most ML models expect 3 channels, not 4)
    constexpr bool modelExpectsRGB = true;
    WICPixelFormatGUID desiredFormat = modelExpectsRGB ? GUID_WICPixelFormat24bppRGB : GUID_WICPixelFormat32bppBGR;
    if (pixelFormat != desiredFormat)
    {
        Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
        THROW_IF_FAILED(wicFactory->CreateFormatConverter(&converter));

        THROW_IF_FAILED(converter->Initialize(
            frame.Get(),
            GUID_WICPixelFormat24bppRGB,
            WICBitmapDitherTypeNone,
            nullptr,
            0.0f,
            WICBitmapPaletteTypeCustom
        ));

        Microsoft::WRL::ComPtr<IWICBitmap> bitmap;
        THROW_IF_FAILED(wicFactory->CreateBitmapFromSource(
            converter.Get(), 
            WICBitmapCacheOnLoad, 
            &bitmap
        ));

        bitmapSource = bitmap;
    } 
}

int main(int argc, char** argv)
{
    THROW_IF_FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));

    // load ort model session


    // ImageTensorFromFilename(LR"(C:\src\ort_sr_demo\zebra.jpg)");

    CoUninitialize();

    return 0;
}