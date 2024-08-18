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
#include <span>
#include <string>

#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"

using Microsoft::WRL::ComPtr;

struct ImageTensorData
{
    std::vector<std::byte> buffer;
    std::vector<int64_t> shape;

    const int64_t Channels() const { return shape[1]; }
    const int64_t Height() const { return shape[2]; }
    const int64_t Width() const { return shape[3]; }
    const int64_t Pixels() const { return Height() * Width(); }
};

ImageTensorData LoadTensorDataFromImageFilename(std::wstring_view filename, uint32_t targetHeight = 0, uint32_t targetWidth = 0)
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

    UINT originalWidth, originalHeight;
    THROW_IF_FAILED(frame->GetSize(&originalWidth, &originalHeight));

    if (targetHeight == 0)
    {
        targetHeight = originalHeight;
    }

    if (targetWidth == 0)
    {
        targetWidth = originalWidth;
    }

    WICPixelFormatGUID pixelFormat;
    THROW_IF_FAILED(frame->GetPixelFormat(&pixelFormat));

    ComPtr<IWICBitmapSource> bitmapSource = frame;

    constexpr bool modelExpectsRGB = true;
    WICPixelFormatGUID desiredFormat = modelExpectsRGB ? GUID_WICPixelFormat24bppRGB : GUID_WICPixelFormat24bppBGR;
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

    if (originalWidth != targetWidth || originalHeight != targetHeight)
    {
        if (targetWidth == targetHeight)
        {
            uint32_t minSide = std::min(originalHeight, originalWidth);
            WICRect cropRect = { 0, 0, static_cast<INT>(minSide), static_cast<INT>(minSide) };
            ComPtr<IWICBitmapClipper> clipper;
            THROW_IF_FAILED(wicFactory->CreateBitmapClipper(&clipper));
            THROW_IF_FAILED(clipper->Initialize(bitmapSource.Get(), &cropRect));
            bitmapSource = clipper;
        }

        ComPtr<IWICBitmapScaler> scaler;
        THROW_IF_FAILED(wicFactory->CreateBitmapScaler(&scaler));

        THROW_IF_FAILED(scaler->Initialize(
            bitmapSource.Get(),
            targetWidth,
            targetHeight,
            WICBitmapInterpolationModeFant
        ));

        bitmapSource = scaler;
    }

    constexpr uint32_t channels = 3;

    // Read pixel data into HWC buffer with 8 bits per channel in RGB order
    std::vector<std::byte> pixelDataHWC8bpc(targetHeight * targetWidth * channels * sizeof(std::byte));
    const uint32_t pixelDataHWC8bpcStrideH = targetWidth * channels * sizeof(uint8_t);
    WICRect rect = { 0, 0, static_cast<INT>(targetWidth), static_cast<INT>(targetHeight) };
    THROW_IF_FAILED(bitmapSource->CopyPixels(
        &rect, 
        pixelDataHWC8bpcStrideH, 
        pixelDataHWC8bpc.size(), 
        reinterpret_cast<BYTE*>(pixelDataHWC8bpc.data())
    ));

    // Convert pixel data to CHW buffer with 32 bits per channel in RGB order
    std::vector<std::byte> pixelDataCHW32bpc(channels * targetHeight * targetWidth * sizeof(float));
    float* pixelDataCHWFloat = reinterpret_cast<float*>(pixelDataCHW32bpc.data());
    for (size_t pixelIndex = 0; pixelIndex < targetHeight * targetWidth; pixelIndex++)
    {
        float r = static_cast<float>(pixelDataHWC8bpc[pixelIndex * channels + 0]) / 255.0f;
        float g = static_cast<float>(pixelDataHWC8bpc[pixelIndex * channels + 1]) / 255.0f;
        float b = static_cast<float>(pixelDataHWC8bpc[pixelIndex * channels + 2]) / 255.0f;

        pixelDataCHWFloat[pixelIndex + 0 * targetHeight * targetWidth] = r;
        pixelDataCHWFloat[pixelIndex + 1 * targetHeight * targetWidth] = g;
        pixelDataCHWFloat[pixelIndex + 2 * targetHeight * targetWidth] = b;
    }

    return { pixelDataCHW32bpc, { 1, channels, targetHeight, targetWidth } };
}

void SaveTensorDataToImageFilename(const ImageTensorData& tensorData, std::wstring_view filename)
{
    // Convert CHW tensor at 32 bits per channel to HWC tensor at 8 bits per channel
    auto src = reinterpret_cast<const float*>(tensorData.buffer.data());
    std::vector<BYTE> dst(tensorData.Pixels() * tensorData.Channels() * sizeof(std::byte));

    for (size_t pixelIndex = 0; pixelIndex < tensorData.Pixels(); pixelIndex++)
    {
        float r = src[pixelIndex + 0 * tensorData.Pixels()];
        float g = src[pixelIndex + 1 * tensorData.Pixels()];
        float b = src[pixelIndex + 2 * tensorData.Pixels()];

        dst[pixelIndex * tensorData.Channels() + 0] = static_cast<BYTE>(r * 255.0f);
        dst[pixelIndex * tensorData.Channels() + 1] = static_cast<BYTE>(g * 255.0f);
        dst[pixelIndex * tensorData.Channels() + 2] = static_cast<BYTE>(b * 255.0f);
    }


    ComPtr<IWICImagingFactory> wicFactory;
    THROW_IF_FAILED(CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&wicFactory)
    ));

    // Create a WIC bitmap
    ComPtr<IWICBitmap> bitmap;
    THROW_IF_FAILED(wicFactory->CreateBitmapFromMemory(
        tensorData.Width(),
        tensorData.Height(),
        GUID_WICPixelFormat24bppRGB,
        tensorData.Width() * tensorData.Channels(),
        dst.size(),
        dst.data(),
        &bitmap
    ));

    ComPtr<IWICStream> stream;
    THROW_IF_FAILED(wicFactory->CreateStream(&stream));

    THROW_IF_FAILED(stream->InitializeFromFilename(filename.data(), GENERIC_WRITE));

    ComPtr<IWICBitmapEncoder> encoder;
    THROW_IF_FAILED(wicFactory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder));

    THROW_IF_FAILED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache));

    ComPtr<IWICBitmapFrameEncode> frame;
    ComPtr<IPropertyBag2> propertyBag;

    THROW_IF_FAILED(encoder->CreateNewFrame(&frame, &propertyBag));

    THROW_IF_FAILED(frame->Initialize(propertyBag.Get()));

    THROW_IF_FAILED(frame->WriteSource(bitmap.Get(), nullptr));

    THROW_IF_FAILED(frame->Commit());

    THROW_IF_FAILED(encoder->Commit());

    
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

Ort::Session CreateOnnxRuntimeSession(Ort::Env& env, IDMLDevice* dmlDevice, ID3D12CommandQueue* commandQueue, std::wstring_view modelPath)
{
    const OrtApi& ortApi = Ort::GetApi();

    Ort::SessionOptions sessionOptions;
    sessionOptions.DisableMemPattern();
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    const OrtDmlApi* ortDmlApi = nullptr;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
    Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, dmlDevice, commandQueue));

    return Ort::Session(env, modelPath.data(), sessionOptions);
}

int main(int argc, char** argv)
{
    THROW_IF_FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));

    auto [dmlDevice, commandQueue] = CreateDmlDeviceAndCommandQueue();

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "DirectML_CV");
    auto ortSession = CreateOnnxRuntimeSession(env, dmlDevice.Get(), commandQueue.Get(), LR"(C:\src\ort_sr_demo\xlsr.onnx)");

    auto inputInfo = ortSession.GetInputTypeInfo(0);
    auto inputTensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
    auto inputTensorShape = inputTensorInfo.GetShape();
    auto inputTensorType = inputTensorInfo.GetElementType();

    auto outputInfo = ortSession.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputInfo.GetTensorTypeAndShapeInfo();
    auto outputTensorShape = outputTensorInfo.GetShape();
    auto outputTensorType = outputTensorInfo.GetElementType();

    auto inputTensorData = LoadTensorDataFromImageFilename(LR"(C:\src\ort_sr_demo\zebra.jpg)", inputTensorShape[2], inputTensorShape[3]);
    SaveTensorDataToImageFilename(inputTensorData, LR"(C:\src\ort_sr_demo\input_tensor.png)");

    const OrtApi& ortApi = Ort::GetApi();

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto inputTensor = Ort::Value::CreateTensor(
        memoryInfo, 
        inputTensorData.buffer.data(), 
        inputTensorData.buffer.size(), 
        inputTensorData.shape.data(), 
        inputTensorData.shape.size(),
        inputTensorType
    );

    Ort::RunOptions runOpts{nullptr};

    std::vector<const char*> inputNames = { "image" };
    std::vector<const char*> outputNames = { "output_0" };

    auto results = ortSession.Run(
        runOpts, 
        inputNames.data(), 
        &inputTensor, 
        1, 
        outputNames.data(), 
        1
    );


    // TODO: silly to copy
    float* outputData = results[0].GetTensorMutableData<float>();
    std::vector<std::byte> outputTensorDataBuffer(512 * 512 * 3 * sizeof(float));
    std::memcpy(outputTensorDataBuffer.data(), outputData, 512*512*3*sizeof(float));
    ImageTensorData outputTensorData = { 
        outputTensorDataBuffer, 
        results[0].GetTensorTypeAndShapeInfo().GetShape() 
    };

    SaveTensorDataToImageFilename(outputTensorData, LR"(C:\src\ort_sr_demo\output_tensor.png)");

    CoUninitialize();

    return 0;
}