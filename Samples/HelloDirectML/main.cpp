// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

#pragma warning(disable : 4238) // References to temporary classes are okay because they are only used as function parameters.

using Microsoft::WRL::ComPtr;

// TODO: Ingest https://github.com/microsoft/DirectX-Headers
#define D3D_FEATURE_LEVEL_1_0_GENERIC_PRIVATE ((D3D_FEATURE_LEVEL)0x100)

// TODO: Ingest https://github.com/microsoft/DirectX-Headers
#define INITGUID
#include<guiddef.h>
DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, 0xb71b0d41, 0x1088, 0x422f, 0xa2, 0x7c, 0x2, 0x50, 0xb7, 0xd3, 0xa9, 0x88);

void InitializeDirect3D12(
    ComPtr<ID3D12Device> & d3D12Device,
    ComPtr<ID3D12CommandQueue> & commandQueue,
    ComPtr<ID3D12CommandAllocator> & commandAllocator,
    ComPtr<ID3D12GraphicsCommandList> & commandList)
{
#if defined(_DEBUG) && !defined(_GAMING_XBOX)
    ComPtr<ID3D12Debug> d3D12Debug;
    // Throws if the D3D12 debug layer is missing - you must install the Graphics Tools optional feature
    THROW_IF_FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(d3D12Debug.GetAddressOf())));
    d3D12Debug->EnableDebugLayer();
#endif

#if !defined(_GAMING_XBOX)
    // Whether to skip adapters which support Graphics in order to target NPU for testing
    bool forceComputeOnlyDevice = true;

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
        ComPtr<IDXCoreAdapterList> adapterList;

        THROW_IF_FAILED(factory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, IID_PPV_ARGS(&adapterList)));
        if (adapterList->GetAdapterCount() == 0)
        {
            THROW_IF_FAILED(factory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE, IID_PPV_ARGS(&adapterList)));
        }

        for (uint32_t i = 0, adapterCount = adapterList->GetAdapterCount(); i < adapterCount; i++)
        {
            ComPtr<IDXCoreAdapter> nextGpuAdapter;
            THROW_IF_FAILED(adapterList->GetAdapter(static_cast<uint32_t>(i), IID_PPV_ARGS(&nextGpuAdapter)));

            if (!forceComputeOnlyDevice || !nextGpuAdapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS))
            {
                adapter = std::move(nextGpuAdapter);
                break;
            }
        }
    }

    // Create the D3D12 Device
    if (adapter)
    {
        if (FAILED(::D3D12CreateDevice(
            adapter.Get(),
            D3D_FEATURE_LEVEL_1_0_CORE,
            IID_PPV_ARGS(d3D12Device.ReleaseAndGetAddressOf()))))
        {
            THROW_IF_FAILED(::D3D12CreateDevice(
                adapter.Get(),
                D3D_FEATURE_LEVEL_1_0_GENERIC_PRIVATE,
                IID_PPV_ARGS(d3D12Device.ReleaseAndGetAddressOf())));
        }
    }

#else
    D3D12XBOX_CREATE_DEVICE_PARAMETERS params = {};
    params.Version = D3D12_SDK_VERSION;
    params.GraphicsCommandQueueRingSizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;
    params.GraphicsScratchMemorySizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;
    params.ComputeScratchMemorySizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;
#ifdef _DEBUG
    params.ProcessDebugFlags = D3D12XBOX_PROCESS_DEBUG_FLAG_VALIDATED;
#endif

    THROW_IF_FAILED(D3D12XboxCreateDevice(
        nullptr,
        &params,
        IID_GRAPHICS_PPV_ARGS(d3D12Device.ReleaseAndGetAddressOf())
    ));
#endif

    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    THROW_IF_FAILED(d3D12Device->CreateCommandQueue(
        &commandQueueDesc,
        IID_GRAPHICS_PPV_ARGS(commandQueue.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(d3D12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_GRAPHICS_PPV_ARGS(commandAllocator.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(d3D12Device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        commandAllocator.Get(),
        nullptr,
        IID_GRAPHICS_PPV_ARGS(commandList.ReleaseAndGetAddressOf())));
}

void CloseExecuteResetWait(
    ComPtr<ID3D12Device> d3D12Device,
    ComPtr<ID3D12CommandQueue> commandQueue,
    ComPtr<ID3D12CommandAllocator> commandAllocator,
    ComPtr<ID3D12GraphicsCommandList> commandList)
{
    THROW_IF_FAILED(commandList->Close());

    ID3D12CommandList* commandLists[] = { commandList.Get() };
    commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
    
    ComPtr<ID3D12Fence> d3D12Fence;
    THROW_IF_FAILED(d3D12Device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_GRAPHICS_PPV_ARGS(d3D12Fence.GetAddressOf())));

    wil::unique_handle fenceEventHandle(::CreateEvent(nullptr, true, false, nullptr));
    THROW_LAST_ERROR_IF_NULL(fenceEventHandle);

    THROW_IF_FAILED(commandQueue->Signal(d3D12Fence.Get(), 1));
    THROW_IF_FAILED(d3D12Fence->SetEventOnCompletion(1, fenceEventHandle.get()));

    ::WaitForSingleObjectEx(fenceEventHandle.get(), INFINITE, FALSE);
    
    THROW_IF_FAILED(commandAllocator->Reset());
    THROW_IF_FAILED(commandList->Reset(commandAllocator.Get(), nullptr));
}

int main()
{
    ComPtr<ID3D12Device> d3D12Device;
    ComPtr<ID3D12CommandQueue> commandQueue;
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> commandList;

    // Set up Direct3D 12.
    InitializeDirect3D12(d3D12Device, commandQueue, commandAllocator, commandList);

    // Create the DirectML device.

    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;

#if defined (_DEBUG)
    // If the project is in a debug build, then enable debugging via DirectML debug layers with this flag.
    dmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif

    ComPtr<IDMLDevice> dmlDevice;
    THROW_IF_FAILED(DMLCreateDevice(
        d3D12Device.Get(),
        dmlCreateDeviceFlags,
        IID_PPV_ARGS(dmlDevice.GetAddressOf())));

    // Tensor with same height and width valid to use for both GEMM inputs
    constexpr UINT tensorSizes[4] = { 1, 1, 32, 32 };
    constexpr UINT tensorElementCount = tensorSizes[0] * tensorSizes[1] * tensorSizes[2] * tensorSizes[3];

    // 24 elements * 4 == 96 bytes.
    UINT64 tensorBufferSize = ((tensorElementCount * 2) + 3) & ~3;


    // Create DirectML operator(s). Operators represent abstract functions such as "multiply", "reduce", "convolution", or even
    // compound operations such as recurrent neural nets. This example creates an instance of the gemm operator

    ComPtr<IDMLCompiledOperator> dmlCompiledOperator;
    ::dml::Graph graph(dmlDevice.Get());
    ::dml::TensorDesc::Dimensions dimensions(std::begin(tensorSizes), std::end(tensorSizes));
    ::dml::TensorDesc aDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, dimensions };
    ::dml::Expression a_input = ::dml::InputTensor(graph, 0, aDesc);

    ::dml::TensorDesc bDesc = { DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_OWNED_BY_DML, dimensions };
    ::dml::Expression b_input = ::dml::InputTensor(graph, 1, bDesc);

    // Creates the DirectMLX Graph then takes the compiled operator(s) and attaches it to the relative COM Interface.
    ::dml::Expression output = ::dml::Gemm(a_input, b_input);

    DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    std::array tmp{ output }; // std::span has no initializer list ctor
    dmlCompiledOperator.Attach(graph.Compile(executionFlags, tmp).Detach());

    ComPtr<IDMLOperatorInitializer> dmlOperatorInitializer;
    IDMLCompiledOperator* dmlCompiledOperators[] = { dmlCompiledOperator.Get() };
    THROW_IF_FAILED(dmlDevice->CreateOperatorInitializer(
        ARRAYSIZE(dmlCompiledOperators),
        dmlCompiledOperators,
        IID_PPV_ARGS(dmlOperatorInitializer.GetAddressOf())));

    // Query the operator for the required size (in descriptors) of its binding table.
    // You need to initialize an operator exactly once before it can be executed, and
    // the two stages require different numbers of descriptors for binding. For simplicity,
    // we create a single descriptor heap that's large enough to satisfy them both.
    DML_BINDING_PROPERTIES initializeBindingProperties = dmlOperatorInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES executeBindingProperties = dmlCompiledOperator->GetBindingProperties();
    UINT descriptorCount = std::max(
        initializeBindingProperties.RequiredDescriptorCount,
        executeBindingProperties.RequiredDescriptorCount);

    // Create descriptor heaps.
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = descriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    THROW_IF_FAILED(d3D12Device->CreateDescriptorHeap(
        &descriptorHeapDesc,
        IID_GRAPHICS_PPV_ARGS(descriptorHeap.GetAddressOf())));

    // Set the descriptor heap(s).
    ID3D12DescriptorHeap* d3D12DescriptorHeaps[] = { descriptorHeap.Get() };
    commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    // Create a binding table over the descriptor heap we just created.
    DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
    dmlBindingTableDesc.Dispatchable = dmlOperatorInitializer.Get();
    dmlBindingTableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.SizeInDescriptors = descriptorCount;

    ComPtr<IDMLBindingTable> dmlBindingTable;
    THROW_IF_FAILED(dmlDevice->CreateBindingTable(
        &dmlBindingTableDesc,
        IID_PPV_ARGS(dmlBindingTable.GetAddressOf())));

    // Create the temporary and persistent resources that are necessary for executing an operator.

    // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
    // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.

    UINT64 temporaryResourceSize = std::max(
        initializeBindingProperties.TemporaryResourceSize,
        executeBindingProperties.TemporaryResourceSize);
    UINT64 persistentResourceSize = executeBindingProperties.PersistentResourceSize;

    // Bind and initialize the operator on the GPU.

    ComPtr<ID3D12Resource> temporaryBuffer;
    if (temporaryResourceSize != 0)
    {
        THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(temporaryBuffer.GetAddressOf())));

        if (initializeBindingProperties.TemporaryResourceSize != 0)
        {
            DML_BUFFER_BINDING bufferBinding{ temporaryBuffer.Get(), 0, temporaryResourceSize };
            DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
            dmlBindingTable->BindTemporaryResource(&bindingDesc);
        }
    }

    // Create tensor buffers for upload/input/output/readback of the tensor elements.

    ComPtr<ID3D12Resource> uploadBuffer;
    THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(uploadBuffer.GetAddressOf())));

    ComPtr<ID3D12Resource> inputBuffer;
    THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(inputBuffer.GetAddressOf())));

    std::wcout << std::fixed; std::wcout.precision(4);

    // Float16-sized element array
    std::vector<uint16_t> inputTensorElementArray(tensorElementCount);
    {
        // Fill arbitrary data into the input
        std::wcout << L"input tensor:\n";
        for (auto& element : inputTensorElementArray)
        {
            element = 0x3C00; // float16 1.0
            std::wcout << element << L' ';
        };
        std::wcout << std::endl;

        D3D12_SUBRESOURCE_DATA tensorSubresourceData{};
        tensorSubresourceData.pData = inputTensorElementArray.data();
        tensorSubresourceData.RowPitch = static_cast<LONG_PTR>(tensorBufferSize);
        tensorSubresourceData.SlicePitch = tensorSubresourceData.RowPitch;

        // Upload the input tensor to the GPU.
        ::UpdateSubresources(
            commandList.Get(),
            inputBuffer.Get(),
            uploadBuffer.Get(),
            0,
            0,
            1,
            &tensorSubresourceData);

        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                inputBuffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            )
        );
    }

    DML_BUFFER_BINDING initInputBufferBinding[2] = {{}, { inputBuffer.Get(), 0, tensorBufferSize }};
    DML_BUFFER_ARRAY_BINDING inputArrayBinding = {};
    inputArrayBinding.BindingCount = 2;
    inputArrayBinding.Bindings = initInputBufferBinding;
    DML_BINDING_DESC inputArrayBindingDesc = { DML_BINDING_TYPE_BUFFER_ARRAY, &inputArrayBinding };

    dmlBindingTable->BindInputs(1, &inputArrayBindingDesc);

    ComPtr<ID3D12Resource> persistentBuffer;
    if (persistentResourceSize != 0)
    {
        THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(persistentResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(persistentBuffer.GetAddressOf())));

        // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
        DML_BUFFER_BINDING bufferBinding{ persistentBuffer.Get(), 0, persistentResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindOutputs(1, &bindingDesc);
    }

    // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
    ComPtr<IDMLCommandRecorder> dmlCommandRecorder;
    THROW_IF_FAILED(dmlDevice->CreateCommandRecorder(
        IID_PPV_ARGS(dmlCommandRecorder.GetAddressOf())));

    // Record execution of the operator initializer.
    dmlCommandRecorder->RecordDispatch(
        commandList.Get(),
        dmlOperatorInitializer.Get(),
        dmlBindingTable.Get());

    // Close the Direct3D 12 command list, and submit it for execution as you would any other command list. You could
    // in principle record the execution into the same command list as the initialization, but you need only to Initialize
    // once, and typically you want to Execute an operator more frequently than that.
    CloseExecuteResetWait(d3D12Device, commandQueue, commandAllocator, commandList);

    // 
    // Bind and execute the operator on the GPU.
    // 

    commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    // Reset the binding table to bind for the operator we want to execute (it was previously used to bind for the
    // initializer).

    dmlBindingTableDesc.Dispatchable = dmlCompiledOperator.Get();

    THROW_IF_FAILED(dmlBindingTable->Reset(&dmlBindingTableDesc));

    if (temporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING bufferBinding{ temporaryBuffer.Get(), 0, temporaryResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindTemporaryResource(&bindingDesc);
    }

    if (persistentResourceSize != 0)
    {
        DML_BUFFER_BINDING bufferBinding{ persistentBuffer.Get(), 0, persistentResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindPersistentResource(&bindingDesc);
    }


    DML_BUFFER_BINDING inputBufferBinding = { inputBuffer.Get(), 0, tensorBufferSize };
    DML_BINDING_DESC inputBindingDesc[2] = {{ DML_BINDING_TYPE_BUFFER, &inputBufferBinding }, {}};
    dmlBindingTable->BindInputs(2, inputBindingDesc);

    ComPtr<ID3D12Resource> outputBuffer;
    THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(outputBuffer.GetAddressOf())));

    DML_BUFFER_BINDING outputBufferBinding{ outputBuffer.Get(), 0, tensorBufferSize };
    DML_BINDING_DESC outputBindingDesc{ DML_BINDING_TYPE_BUFFER, &outputBufferBinding };
    dmlBindingTable->BindOutputs(1, &outputBindingDesc);

    // Record execution of the compiled operator.
    dmlCommandRecorder->RecordDispatch(commandList.Get(), dmlCompiledOperator.Get(), dmlBindingTable.Get());

    CloseExecuteResetWait(d3D12Device, commandQueue, commandAllocator, commandList);

    // The output buffer now contains the result of the gemm operator,
    // so read it back if you want the CPU to access it.

    ComPtr<ID3D12Resource> readbackBuffer;
    THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(readbackBuffer.GetAddressOf())));

    commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            outputBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE
        )
    );

    commandList->CopyResource(readbackBuffer.Get(), outputBuffer.Get());

    CloseExecuteResetWait(d3D12Device, commandQueue, commandAllocator, commandList);

    D3D12_RANGE tensorBufferRange{ 0, static_cast<SIZE_T>(tensorBufferSize) };
    uint16_t* outputBufferData{};
    THROW_IF_FAILED(readbackBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void**>(&outputBufferData)));

    std::wstring outputString = L"output tensor.  20480 (float16 32.0) is expected:\n";
    for (size_t tensorElementIndex{ 0 }; tensorElementIndex < tensorElementCount; ++tensorElementIndex, ++outputBufferData)
    {
        outputString += std::to_wstring(*outputBufferData) + L' ';
    }

    std::wcout << outputString << std::endl;
    OutputDebugStringW(outputString.c_str());

    D3D12_RANGE emptyRange{ 0, 0 };
    readbackBuffer->Unmap(0, &emptyRange);
}
