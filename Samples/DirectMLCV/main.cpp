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
#include <dxcore.h>
#include <optional>
#include <iostream>
#include <filesystem>
#include <span>
#include <string>
#include "cxxopts.hpp"
#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"
#include "image_helpers.h"
#include "dx_helpers.h"

void RunModel(
    IDMLDevice* dmlDevice, 
    ID3D12CommandQueue* d3dQueue, 
    const std::filesystem::path& modelPath,
    const std::filesystem::path& imagePath)
{
    // DML execution provider prefers these session options.
    Ort::SessionOptions sessionOptions;
    sessionOptions.DisableMemPattern();
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    // By passing in an explicitly created DML device & queue, the DML execution provider sends work
    // to the desired device. If not used, the DML execution provider will create its own device & queue.
    const OrtApi& ortApi = Ort::GetApi();
    const OrtDmlApi* ortDmlApi = nullptr;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
    Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(
        sessionOptions, 
        dmlDevice, 
        d3dQueue
    ));

    // Load ONNX model into a session.
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "DirectML_CV");
    Ort::Session ortSession(env, modelPath.wstring().c_str(), sessionOptions);

    if (ortSession.GetInputCount() != 1 && ortSession.GetOutputCount() != 1)
    {
        throw std::invalid_argument("Model must have exactly one input and one output");
    }

    auto inputInfo = ortSession.GetInputTypeInfo(0);
    auto inputTensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
    auto inputTensorShape = inputTensorInfo.GetShape();
    auto inputTensorType = inputTensorInfo.GetElementType();
    if (inputTensorType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        throw std::invalid_argument("Model input must be of type float32");
    }

    // Load image and transform it into an NCHW tensor with the correct shape and data type.
    const uint32_t inputChannels = inputTensorShape[1];
    const uint32_t inputHeight = inputTensorShape[2];
    const uint32_t inputWidth = inputTensorShape[3];
    std::vector<std::byte> inputBuffer(inputChannels * inputHeight * inputWidth * sizeof(float));
    FillNCHWBufferFromImageFilename(imagePath.wstring(), inputBuffer, inputHeight, inputWidth, DataType::Float32, ChannelOrder::RGB);

    std::cout << "Saving cropped/scaled image to input.png" << std::endl;
    SaveNCHWBufferToImageFilename(L"input.png", inputBuffer, inputHeight, inputWidth, DataType::Float32, ChannelOrder::RGB);

    // For simplicity, this sample binds input/output buffers in system memory instead of DirectX resources.
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto inputTensor = Ort::Value::CreateTensor(
        memoryInfo, 
        inputBuffer.data(), 
        inputBuffer.size(), 
        inputTensorShape.data(), 
        inputTensorShape.size(),
        inputTensorType
    );

    auto outputInfo = ortSession.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputInfo.GetTensorTypeAndShapeInfo();
    auto outputTensorShape = outputTensorInfo.GetShape();
    auto outputTensorType = outputTensorInfo.GetElementType();
    if (outputTensorType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        throw std::invalid_argument("Model output must be of type float32");
    }

    const uint32_t outputChannels = outputTensorShape[1];
    const uint32_t outputHeight = outputTensorShape[2];
    const uint32_t outputWidth = outputTensorShape[3];

    // Run the session to get inference results.
    Ort::RunOptions runOpts;
    std::vector<const char*> inputNames = { "image" };
    std::vector<const char*> outputNames = { "output_0" };
    auto outputs = ortSession.Run(runOpts, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);

    std::span<const std::byte> outputBuffer(reinterpret_cast<const std::byte*>(outputs[0].GetTensorData<float>()), outputChannels * outputHeight * outputWidth * sizeof(float));

    std::cout << "Saving inference results to output.png" << std::endl;
    SaveNCHWBufferToImageFilename(
        L"output.png", 
        outputBuffer, 
        outputHeight, 
        outputWidth, 
        DataType::Float32, 
        ChannelOrder::RGB
    );
}

int main(int argc, char** argv)
{
    using Microsoft::WRL::ComPtr;

    // Functions in image_helpers.h use WIC APIs, which require CoInitialize.
    THROW_IF_FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));

    // Parse command-line options.
    cxxopts::Options commandLineParams("directml_cv", "DirectML Computer Vision Sample");
    commandLineParams.add_options()
        ("h,help", "Print usage")
        ("m,model", "Path to ONNX model file", cxxopts::value<std::string>()->default_value("esrgan.onnx"))
        ("i,image", "Path to input image file", cxxopts::value<std::string>()->default_value("zebra.jpg"))
        ("a,adapter", "Adapter name substring filter", cxxopts::value<std::string>()->default_value(""));

    auto commandLineArgs = commandLineParams.parse(argc, argv);

    // See dx_helpers.h for logic to select a DXCore adapter, create DML device, and create D3D command queue.
    ComPtr<IDMLDevice> dmlDevice;
    ComPtr<ID3D12CommandQueue> commandQueue;
    try
    {
        std::tie(dmlDevice, commandQueue) = CreateDmlDeviceAndCommandQueue(commandLineArgs["adapter"].as<std::string>());
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error creating device: " << e.what() << std::endl;
        return 1;
    }

    try
    {
        RunModel(
            dmlDevice.Get(), 
            commandQueue.Get(), 
            commandLineArgs["model"].as<std::string>(),
            commandLineArgs["image"].as<std::string>()
        );
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error running model: " << e.what() << std::endl;
        return 1;
    }

    // Functions in image_helpers.h use WIC APIs, which require CoUninitialize.
    CoUninitialize();

    return 0;
}