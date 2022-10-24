#pragma once

#include "PixCaptureHelper.h"
#include "DxModules.h"

// Simplified abstraction for submitting work to a device with a single command queue. Not thread safe.
class Device
{
public:
    Device(
        IAdapter* adapter, 
        bool debugLayersEnabled, 
        D3D12_COMMAND_LIST_TYPE commandListType, 
        std::shared_ptr<PixCaptureHelper> pixCaptureHelper,
        std::shared_ptr<D3d12Module> d3dModule,
        std::shared_ptr<DmlModule> dmlModule
        );
    ~Device();

    D3d12Module* D3DModule() { return m_d3dModule.get(); }
    ID3D12Device2* D3D() { return m_d3d.Get(); }
    IDMLDevice1* DML() { return m_dml.Get(); }
    ID3D12CommandQueue* GetCommandQueue() { return m_queue.Get(); }
    D3D12_COMMAND_LIST_TYPE GetCommandListType() const { return m_commandListType; }
    ID3D12GraphicsCommandList* GetCommandList() { return m_commandList.Get(); }
    PixCaptureHelper& GetPixCaptureHelper() { return *m_pixCaptureHelper; }

#ifndef DXCOMPILER_NONE
    IDxcUtils* GetDxcUtils();
    IDxcIncludeHandler* GetDxcIncludeHandler();
    IDxcCompiler3* GetDxcCompiler();
#endif

    // TODO: test custom heap buffer with write combine for igpu?

    Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultBuffer(
        uint64_t sizeInBytes, 
        D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        uint64_t alignment = 0,
        D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE);

    Microsoft::WRL::ComPtr<ID3D12Resource> CreateUploadBuffer(
        uint64_t sizeInBytes,
        D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_NONE,
        uint64_t alignment = 0,
        D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE);

    Microsoft::WRL::ComPtr<ID3D12Resource> CreateReadbackBuffer(
        uint64_t sizeInBytes,
        D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_NONE,
        uint64_t alignment = 0,
        D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE);

    // Waits for all work submitted to this device's queue to complete.
    void WaitForGpuWorkToComplete();

    void DispatchAndWait();

    void RecordDispatch(IDMLDispatchable* dispatchable, IDMLBindingTable* bindingTable);

    void KeepAliveUntilNextCommandListDispatch(Microsoft::WRL::ComPtr<IGraphicsUnknown>&& object)
    {
        m_temporaryResources.emplace_back(std::move(object));
    }

    Microsoft::WRL::ComPtr<ID3D12Resource> Upload(uint64_t totalSize, gsl::span<const std::byte> data, std::wstring_view name = {});

    std::vector<std::byte> Download(Microsoft::WRL::ComPtr<ID3D12Resource>);

    static uint32_t GetSizeInBytes(DML_TENSOR_DATA_TYPE dataType);
    static DXGI_FORMAT GetDxgiFormatFromDmlTensorDataType(DML_TENSOR_DATA_TYPE dataType);

private:
    void EnsureDxcInterfaces();

private:
    std::shared_ptr<PixCaptureHelper> m_pixCaptureHelper;
    std::shared_ptr<D3d12Module> m_d3dModule;
    Microsoft::WRL::ComPtr<ID3D12Device8> m_d3d;
#ifndef _GAMING_XBOX
    Microsoft::WRL::ComPtr<ID3D12InfoQueue1> m_infoQueue;
#endif
    std::shared_ptr<DmlModule> m_dmlModule;
    Microsoft::WRL::ComPtr<IDMLDevice1> m_dml;
    Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_commandRecorder;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    D3D12_COMMAND_LIST_TYPE m_commandListType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;
    std::vector<Microsoft::WRL::ComPtr<IGraphicsUnknown>> m_temporaryResources;

#ifndef DXCOMPILER_NONE
    Microsoft::WRL::ComPtr<IDxcUtils> m_dxcUtils;
    Microsoft::WRL::ComPtr<IDxcIncludeHandler> m_dxcIncludeHandler;
    Microsoft::WRL::ComPtr<IDxcCompiler3> m_dxcCompiler;
#endif
};