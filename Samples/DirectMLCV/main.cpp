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
#include <wrl/client.h>
#include <wil/result.h>

#include <d3d12.h>
#include <dxgi1_6.h>

#include <optional>
#include <span>
#include <string>

#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"

#include "image_helpers.h"

using Microsoft::WRL::ComPtr;

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

    uint32_t inputChannels = 3;
    uint32_t inputHeight = 128;
    uint32_t inputWidth = 128;
    std::vector<std::byte> inputBuffer(inputChannels * inputHeight * inputWidth * sizeof(float));
    FillNCHWBufferFromImageFilename(LR"(zebra.jpg)", inputBuffer, inputHeight, inputWidth, DataType::Float32, ChannelOrder::RGB);

    // auto [dmlDevice, commandQueue] = CreateDmlDeviceAndCommandQueue();

    // Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "DirectML_CV");
    // auto ortSession = CreateOnnxRuntimeSession(env, dmlDevice.Get(), commandQueue.Get(), LR"(esrgan.onnx)");

    // auto inputInfo = ortSession.GetInputTypeInfo(0);
    // auto inputTensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
    // auto inputTensorShape = inputTensorInfo.GetShape();
    // auto inputTensorType = inputTensorInfo.GetElementType();

    // auto outputInfo = ortSession.GetOutputTypeInfo(0);
    // auto outputTensorInfo = outputInfo.GetTensorTypeAndShapeInfo();
    // auto outputTensorShape = outputTensorInfo.GetShape();
    // auto outputTensorType = outputTensorInfo.GetElementType();

    // auto inputTensorData = LoadTensorDataFromImageFilename(LR"(zebra.jpg)", inputTensorShape[2], inputTensorShape[3]);
    // SaveTensorDataToImageFilename(inputTensorData, LR"(input_tensor.png)");

    // const OrtApi& ortApi = Ort::GetApi();

    // Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // auto inputTensor = Ort::Value::CreateTensor(
    //     memoryInfo, 
    //     inputTensorData.buffer.data(), 
    //     inputTensorData.buffer.size(), 
    //     inputTensorData.shape.data(), 
    //     inputTensorData.shape.size(),
    //     inputTensorType
    // );

    // Ort::RunOptions runOpts{nullptr};

    // std::vector<const char*> inputNames = { "image" };
    // std::vector<const char*> outputNames = { "output_0" };

    // auto results = ortSession.Run(
    //     runOpts, 
    //     inputNames.data(), 
    //     &inputTensor, 
    //     1, 
    //     outputNames.data(), 
    //     1
    // );

    // // TODO: silly to copy
    // float* outputData = results[0].GetTensorMutableData<float>();
    // std::vector<std::byte> outputTensorDataBuffer(512 * 512 * 3 * sizeof(float));
    // std::memcpy(outputTensorDataBuffer.data(), outputData, 512*512*3*sizeof(float));
    // ImageTensorData outputTensorData = { 
    //     outputTensorDataBuffer, 
    //     results[0].GetTensorTypeAndShapeInfo().GetShape() 
    // };

    // SaveTensorDataToImageFilename(outputTensorData, LR"(output_tensor.png)");

    CoUninitialize();

    return 0;
}