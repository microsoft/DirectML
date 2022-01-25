#pragma once

#include "dxcapi.h"

// Simplified abstraction for submitting work to a device with a single command queue. Not thread safe.
class Device
{
public:
    Device(IDXCoreAdapter* adapter, bool debugLayersEnabled, D3D12_COMMAND_LIST_TYPE commandListType);
    ~Device();

    ID3D12Device9* D3D() { return m_d3d.Get(); }
    IDMLDevice1* DML() { return m_dml.Get(); }
    ID3D12CommandQueue* GetCommandQueue() { return m_queue.Get(); }
    D3D12_COMMAND_LIST_TYPE GetCommandListType() const { return m_commandListType; }
    ID3D12GraphicsCommandList* GetCommandList() { return m_commandList.Get(); }

    IDxcUtils* GetDxcUtils();
    IDxcIncludeHandler* GetDxcIncludeHandler();
    IDxcCompiler3* GetDxcCompiler();

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

    Microsoft::WRL::ComPtr<IDxcResult> CompileWithDxc(
        std::filesystem::path hlslPath, 
        gsl::span<std::wstring_view> args);

    // Waits for all work submitted to this device's queue to complete.
    void WaitForGpuWorkToComplete();

    void DispatchAndWait();

    void PrintDebugLayerMessages();
    
    void RecordDispatch(IDMLDispatchable* dispatchable, IDMLBindingTable* bindingTable);

    void KeepAliveUntilNextCommandListDispatch(Microsoft::WRL::ComPtr<IUnknown>&& object)
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
    Microsoft::WRL::ComPtr<ID3D12Device9> m_d3d;
    Microsoft::WRL::ComPtr<ID3D12InfoQueue> m_infoQueue;
    Microsoft::WRL::ComPtr<IDMLDevice1> m_dml;
    Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_commandRecorder;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    Microsoft::WRL::ComPtr<ID3D12SharingContract> m_sharingContract;
    D3D12_COMMAND_LIST_TYPE m_commandListType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;
    std::vector<Microsoft::WRL::ComPtr<IUnknown>> m_temporaryResources;

    Microsoft::WRL::ComPtr<IDxcUtils> m_dxcUtils;
    Microsoft::WRL::ComPtr<IDxcIncludeHandler> m_dxcIncludeHandler;
    Microsoft::WRL::ComPtr<IDxcCompiler3> m_dxcCompiler;
};