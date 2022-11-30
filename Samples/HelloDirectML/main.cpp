// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

#pragma warning(disable : 4238) // References to temporary classes are okay because they are only used as function parameters.

using Microsoft::WRL::ComPtr;

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
    ComPtr<IDXGIFactory4> dxgiFactory;
    THROW_IF_FAILED(CreateDXGIFactory1(IID_PPV_ARGS(dxgiFactory.GetAddressOf())));

    ComPtr<IDXGIAdapter> dxgiAdapter;
    UINT adapterIndex{};
    HRESULT hr{};
    do
    {
        dxgiAdapter = nullptr;
        THROW_IF_FAILED(dxgiFactory->EnumAdapters(adapterIndex, dxgiAdapter.ReleaseAndGetAddressOf()));
        ++adapterIndex;

        hr = ::D3D12CreateDevice(
            dxgiAdapter.Get(),
            D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(d3D12Device.ReleaseAndGetAddressOf()));
        if (hr == DXGI_ERROR_UNSUPPORTED) continue;
        THROW_IF_FAILED(hr);
    } while (hr != S_OK);
#else
    THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_GRAPHICS_PPV_ARGS(d3D12Device.GetAddressOf())));
#endif

    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    THROW_IF_FAILED(d3D12Device->CreateCommandQueue(
        &commandQueueDesc,
        IID_GRAPHICS_PPV_ARGS(commandQueue.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(d3D12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_GRAPHICS_PPV_ARGS(commandAllocator.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(d3D12Device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
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

    THROW_IF_FAILED(d3D12Fence->SetEventOnCompletion(1, fenceEventHandle.get()));

    THROW_IF_FAILED(commandQueue->Signal(d3D12Fence.Get(), 1));
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

    constexpr UINT tensorSizes[4] = { 1, 2, 3, 4 };
    constexpr UINT tensorElementCount = tensorSizes[0] * tensorSizes[1] * tensorSizes[2] * tensorSizes[3];

#if !USE_DMLX

    DML_BUFFER_TENSOR_DESC dmlBufferTensorDesc = {};
    dmlBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    dmlBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    dmlBufferTensorDesc.DimensionCount = ARRAYSIZE(tensorSizes);
    dmlBufferTensorDesc.Sizes = tensorSizes;
    dmlBufferTensorDesc.Strides = nullptr;
    dmlBufferTensorDesc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
        dmlBufferTensorDesc.DataType,
        dmlBufferTensorDesc.DimensionCount,
        dmlBufferTensorDesc.Sizes,
        dmlBufferTensorDesc.Strides);

    ComPtr<IDMLOperator> dmlOperator;
    {
        // Create DirectML operator(s). Operators represent abstract functions such as "multiply", "reduce", "convolution", or even
        // compound operations such as recurrent neural nets. This example creates an instance of the Identity operator,
        // which applies the function f(x) = x for all elements in a tensor.

        DML_TENSOR_DESC dmlTensorDesc{};
        dmlTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
        dmlTensorDesc.Desc = &dmlBufferTensorDesc;

        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC dmlIdentityOperatorDesc{};
        dmlIdentityOperatorDesc.InputTensor = &dmlTensorDesc;
        dmlIdentityOperatorDesc.OutputTensor = &dmlTensorDesc; // Input and output tensors have same size/type.

        // Like Direct3D 12, these DESC structs don't need to be long-lived. This means, for example, that it's safe to place
        // the DML_OPERATOR_DESC (and all the subobjects it points to) on the stack, since they're no longer needed after
        // CreateOperator returns.
        DML_OPERATOR_DESC dmlOperatorDesc{};
        dmlOperatorDesc.Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;
        dmlOperatorDesc.Desc = &dmlIdentityOperatorDesc;

        THROW_IF_FAILED(dmlDevice->CreateOperator(
            &dmlOperatorDesc,
            IID_PPV_ARGS(dmlOperator.GetAddressOf())));
    }

    // Compile the operator into an object that can be dispatched to the GPU. In this step, DirectML performs operator
    // fusion and just-in-time (JIT) compilation of shader bytecode, then compiles it into a Direct3D 12 pipeline state object (PSO).
    // The resulting compiled operator is a baked, optimized form of an operator suitable for execution on the GPU.

    ComPtr<IDMLCompiledOperator> dmlCompiledOperator;
    THROW_IF_FAILED(dmlDevice->CompileOperator(
        dmlOperator.Get(),
        DML_EXECUTION_FLAG_NONE,
        IID_PPV_ARGS(dmlCompiledOperator.GetAddressOf())));

    // 24 elements * 4 == 96 bytes.
    UINT64 tensorBufferSize{ dmlBufferTensorDesc.TotalTensorSizeInBytes };

#else 

    // Create DirectML operator(s). Operators represent abstract functions such as "multiply", "reduce", "convolution", or even
    // compound operations such as recurrent neural nets. This example creates an instance of the Identity operator,
    // which applies the function f(x) = x for all elements in a tensor.

    ComPtr<IDMLCompiledOperator> dmlCompiledOperator;

    dml::Graph graph(dmlDevice.Get());
    dml::TensorDesc::Dimensions dimensions(std::begin(tensorSizes), std::end(tensorSizes));
    dml::TensorDesc desc = { DML_TENSOR_DATA_TYPE_FLOAT32, dimensions};
    dml::Expression input = dml::InputTensor(graph, 0, desc);

    // Creates the DirectMLX Graph then takes the compiled operator(s) and attaches it to the relative COM Interface.
    dml::Expression output = dml::Identity(input);

    DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    dmlCompiledOperator.Attach(graph.Compile(executionFlags, { output }).Detach());

    // 24 elements * 4 == 96 bytes.
    UINT64 tensorBufferSize{ desc.totalTensorSizeInBytes };

#endif

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

    ComPtr<ID3D12Resource> persistentBuffer;
    if (persistentResourceSize != 0)
    {
        THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(persistentResourceSize),
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
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(inputBuffer.GetAddressOf())));

    std::wcout << std::fixed; std::wcout.precision(2);
    std::array<FLOAT, tensorElementCount> inputTensorElementArray;
    {
        std::wcout << L"input tensor: ";
        for (auto & element : inputTensorElementArray)
        {
            element = 1.618f;
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

    DML_BUFFER_BINDING inputBufferBinding{ inputBuffer.Get(), 0, tensorBufferSize };
    DML_BINDING_DESC inputBindingDesc{ DML_BINDING_TYPE_BUFFER, &inputBufferBinding };
    dmlBindingTable->BindInputs(1, &inputBindingDesc);

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

    // The output buffer now contains the result of the identity operator,
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
    FLOAT* outputBufferData{};
    THROW_IF_FAILED(readbackBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void**>(&outputBufferData)));

    std::wcout << L"output tensor: ";
    for (size_t tensorElementIndex{ 0 }; tensorElementIndex < tensorElementCount; ++tensorElementIndex, ++outputBufferData)
    {
        std::wcout << *outputBufferData << L' ';
    }
    std::wcout << std::endl;

    D3D12_RANGE emptyRange{ 0, 0 };
    readbackBuffer->Unmap(0, &emptyRange);
}
