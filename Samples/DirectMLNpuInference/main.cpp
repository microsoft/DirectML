// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

#include <dxcore_interface.h>
#include <dxcore.h>

#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"

#include "TensorHelper.h"

using Microsoft::WRL::ComPtr;

void InitializeDirectML(ID3D12Device1** d3dDeviceOut, ID3D12CommandQueue** commandQueueOut, IDMLDevice** dmlDeviceOut)
{
    // Useful Flags to chage to allow, require, and disallow certain attributes of devices for testing.
    const bool allowGraphicsCapabilities = false;
    const bool requireComputeDevice = true;
    const bool requireGenericMLDevice = false;

    // Populate helper structures based on above flags
    std::vector<GUID> dxGuidAllowedAttributes = {};
    std::vector<GUID> dxGuidRequiredAttributes = {};
    std::vector<GUID> dxGuidDisallowedAttributes = {};

    // By default allow for these compute attributes
    dxGuidAllowedAttributes.push_back(DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE);
    dxGuidAllowedAttributes.push_back(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML);

    allowGraphicsCapabilities ? 
        dxGuidAllowedAttributes.push_back(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS) :
        dxGuidDisallowedAttributes.push_back(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS);

    if (requireComputeDevice)
    {
        dxGuidRequiredAttributes.push_back(DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE);
    }

    if (requireGenericMLDevice)
    {
        dxGuidRequiredAttributes.push_back(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML);
    }

    ComPtr<IDXCoreAdapterFactory> factory;
    HMODULE dxCoreModule = LoadLibraryW(L"DXCore.dll");
    if (dxCoreModule)
    {
        auto dxcoreCreateAdapterFactory = reinterpret_cast<HRESULT(WINAPI*)(REFIID, void**)>(
            GetProcAddress(dxCoreModule, "DXCoreCreateAdapterFactory")
            );
        if (dxcoreCreateAdapterFactory)
        {
            dxcoreCreateAdapterFactory(IID_PPV_ARGS(&factory));
        }
    }
    // Create the DXCore Adapter
    ComPtr<IDXCoreAdapter> adapter;
    if (factory)
    {
        // If there's any required attributes, save time by only passing in required attributes instead.
        std::vector<GUID> iteratingAdapterList = dxGuidRequiredAttributes.empty() ? dxGuidAllowedAttributes : dxGuidRequiredAttributes;
        
        for (auto& allowedGuid : iteratingAdapterList)
        {
            if (adapter != nullptr) continue;
            
            ComPtr<IDXCoreAdapterList> adapterList;
            THROW_IF_FAILED(factory->CreateAdapterList(1, &allowedGuid, IID_PPV_ARGS(&adapterList)));
                
            for (uint32_t i = 0, adapterCount = adapterList->GetAdapterCount(); i < adapterCount; i++)
            {
                ComPtr<IDXCoreAdapter> currentGpuAdapter;
                THROW_IF_FAILED(adapterList->GetAdapter(static_cast<uint32_t>(i), IID_PPV_ARGS(&currentGpuAdapter)));

                bool isAdapterValid = true;

                // filter out adapters with disallowed attributes
                for (auto& disallowedGuid : dxGuidDisallowedAttributes)
                {
                    if (currentGpuAdapter->IsAttributeSupported(disallowedGuid)) { isAdapterValid = false; }
                }

                // filter out adapters that doesn't match all required attributes
                for (auto& requiredGuid : dxGuidRequiredAttributes)
                {
                    if (!currentGpuAdapter->IsAttributeSupported(requiredGuid)) { isAdapterValid = false; }
                }

                if (!isAdapterValid) { continue; }

                adapter = std::move(currentGpuAdapter);
                break;
            }
        }
    }

    if (adapter)
    {
        size_t adapterNameSize = 0;
        THROW_IF_FAILED(adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &adapterNameSize));
        char* adapterName = (char*)malloc(adapterNameSize);
        THROW_IF_FAILED(adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, adapterNameSize, adapterName));
        printf("Successfully found adapter %s\n", adapterName);
    }

    // Create the D3D12 Device
    ComPtr<ID3D12Device1> d3dDevice;
    if (adapter)
    {
        HMODULE d3d12Module = LoadLibraryW(L"d3d12.dll");
        if (d3d12Module)
        {
            auto d3d12CreateDevice = reinterpret_cast<HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void*)>(
                GetProcAddress(d3d12Module, "D3D12CreateDevice")
                );
            if (d3d12CreateDevice)
            {
                THROW_IF_FAILED(d3d12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_1_0_GENERIC, IID_PPV_ARGS(&d3dDevice)));
            }
        }
    }
    // Create the DML Device and D3D12 Command Queue
    ComPtr<IDMLDevice> dmlDevice;
    ComPtr<ID3D12CommandQueue> commandQueue;
    if (d3dDevice)
    {
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        THROW_IF_FAILED(d3dDevice->CreateCommandQueue(
            &queueDesc,
            IID_PPV_ARGS(commandQueue.ReleaseAndGetAddressOf())));
        HMODULE dmlModule = LoadLibraryW(L"DirectML.dll");
        if (dmlModule)
        {
            auto dmlCreateDevice = reinterpret_cast<HRESULT(WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, DML_FEATURE_LEVEL, REFIID, void*)>(
                GetProcAddress(dmlModule, "DMLCreateDevice1")
                );
            if (dmlCreateDevice)
            {
                THROW_IF_FAILED(dmlCreateDevice(d3dDevice.Get(), DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_5_0, IID_PPV_ARGS(dmlDevice.ReleaseAndGetAddressOf())));
            }
        }
    }

    d3dDevice.CopyTo(d3dDeviceOut);
    commandQueue.CopyTo(commandQueueOut);
    dmlDevice.CopyTo(dmlDeviceOut);
}

void main()
{
    ComPtr<ID3D12Device1> d3dDevice;
    ComPtr<IDMLDevice> dmlDevice;
    ComPtr<ID3D12CommandQueue> commandQueue;
    InitializeDirectML(d3dDevice.GetAddressOf(), commandQueue.GetAddressOf(), dmlDevice.GetAddressOf());

    // Add the DML execution provider to ORT using the DML Device and D3D12 Command Queue created above.
    if (!dmlDevice)
    {
        printf("No NPU device found\n");
        return;
    }

    const OrtApi& ortApi = Ort::GetApi();
    static Ort::Env s_OrtEnv{ nullptr };
    s_OrtEnv = Ort::Env(Ort::ThreadingOptions{});
    s_OrtEnv.DisableTelemetryEvents();

    auto sessionOptions = Ort::SessionOptions{};
    sessionOptions.DisableMemPattern();
    sessionOptions.DisablePerSessionThreads();
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    const OrtDmlApi* ortDmlApi = nullptr;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
    Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, dmlDevice.Get(), commandQueue.Get()));

    // Create the session
    auto session = Ort::Session(s_OrtEnv, L"mobilenetv2-7-fp16.onnx", sessionOptions);
    const char* inputName = "input";
    const char* outputName = "output";

    // Create input tensor
    Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto input = CreateDmlValue(tensor_info, commandQueue.Get());
    auto inputTensor = std::move(input.first);
    
    const auto memoryInfo = inputTensor.GetTensorMemoryInfo();
    Ort::Allocator allocator(session, memoryInfo);
    
    // Get the inputResource and populate!
    ComPtr<ID3D12Resource> inputResource;
    Ort::ThrowOnError(ortDmlApi->GetD3D12ResourceFromAllocation(allocator, inputTensor.GetTensorMutableData<void*>(), &inputResource));

    // Create output tensor
    type_info = session.GetOutputTypeInfo(0);
    tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto output = CreateDmlValue(tensor_info, commandQueue.Get());
    auto outputTensor = std::move(output.first);

    // Run warmup
    session.Run(Ort::RunOptions{ nullptr }, &inputName, &inputTensor, 1, &outputName, &outputTensor, 1);

    // Queue fence, and wait for completion
    ComPtr<ID3D12Fence> fence;
    THROW_IF_FAILED(d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())));
    THROW_IF_FAILED(commandQueue->Signal(fence.Get(), 1));

    wil::unique_handle fenceEvent(CreateEvent(nullptr, FALSE, FALSE, nullptr));
    THROW_IF_FAILED(fence->SetEventOnCompletion(1, fenceEvent.get()));
    THROW_HR_IF(E_FAIL, WaitForSingleObject(fenceEvent.get(), INFINITE) != WAIT_OBJECT_0);

    // Record start
    auto start = std::chrono::high_resolution_clock::now();

    // Run performance test
    constexpr int fenceValueStart = 2;
    constexpr int numIterations = 100;
    for (int i = fenceValueStart; i < (numIterations + fenceValueStart); i++)
    {
        session.Run(Ort::RunOptions{ nullptr }, &inputName, &inputTensor, 1, &outputName, &outputTensor, 1);

        {
            // Synchronize with CPU before queuing more inference runs
            THROW_IF_FAILED(commandQueue->Signal(fence.Get(), i));
            THROW_HR_IF(E_FAIL, ResetEvent(fenceEvent.get()) == 0);
            THROW_IF_FAILED(fence->SetEventOnCompletion(i, fenceEvent.get()));
            THROW_HR_IF(E_FAIL, WaitForSingleObject(fenceEvent.get(), INFINITE) != WAIT_OBJECT_0);
        }
    }

    // Record end and calculate duration
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    printf("Evaluate Took: %fus\n", float(duration.count())/100);

    // Read results
    ComPtr<ID3D12Resource> outputResource;
    Ort::ThrowOnError(ortDmlApi->GetD3D12ResourceFromAllocation(allocator, outputTensor.GetTensorMutableData<void*>(), &outputResource));
}
