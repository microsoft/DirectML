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

#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"

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

std::tuple<ComPtr<IDMLDevice>, ComPtr<ID3D12CommandQueue>> CreateDmlDeviceAndCommandQueue()
{
    ComPtr<ID3D12Device> d3d12Device;
    THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d12Device)));

    ComPtr<IDMLDevice> dmlDevice;
    THROW_IF_FAILED(DMLCreateDevice(d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dmlDevice)));

    D3D12_COMMAND_QUEUE_DESC queueDesc = 
    {
        .Type = D3D12_COMMAND_LIST_TYPE_COMPUTE,
        .Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
        .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
        .NodeMask = 0
    };

    ComPtr<ID3D12CommandQueue> commandQueue;
    THROW_IF_FAILED(d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));

    return { dmlDevice, commandQueue };
}

Ort::Session CreateOnnxRuntimeSession(IDMLDevice* dmlDevice, ID3D12CommandQueue* commandQueue, std::wstring_view modelPath)
{
    const OrtApi& ortApi = Ort::GetApi();
    // Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&m_ortDmlApi)));

    Ort::SessionOptions sessionOptions;
    sessionOptions.DisablePerSessionThreads();
    sessionOptions.DisableMemPattern();
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    const OrtDmlApi* ortDmlApi = nullptr;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
    Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, dmlDevice, commandQueue));

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "DirectML_CV");

    return Ort::Session(env, modelPath.data(), sessionOptions);
}

int main(int argc, char** argv)
{
    THROW_IF_FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));

    auto [dmlDevice, commandQueue] = CreateDmlDeviceAndCommandQueue();

    auto ortSession = CreateOnnxRuntimeSession(dmlDevice.Get(), commandQueue.Get(), LR"(C:\src\ort_sr_demo\xlsr.onnx)");

    // load input image

    CopyPixelDataFromImageFilename(LR"(C:\src\ort_sr_demo\zebra.jpg)");

    CoUninitialize();

    return 0;
}