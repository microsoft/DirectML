#pragma once

class WrappedDmlDevice : public Microsoft::WRL::Base<Microsoft::WRL::ChainInterfaces<IDMLDevice1, IDMLDevice, IDMLObject>>
{
public:
    explicit WrappedDmlDevice(
        IDMLDevice1* impl, 
        IDxDispatchLogger* logger,
        const CommandLineArgs& args
        );

    // IDMLObject
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, _Inout_ UINT* dataSize, _Out_writes_bytes_opt_(*dataSize) void* data) noexcept final;
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT dataSize, _In_reads_bytes_opt_(dataSize) const void* data) noexcept final;
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, _In_opt_ IUnknown* data) noexcept final;
    HRESULT STDMETHODCALLTYPE SetName(PCWSTR name) noexcept final;

    // IDMLDevice
    HRESULT STDMETHODCALLTYPE CheckFeatureSupport(
        DML_FEATURE feature,
        UINT featureQueryDataSize,
        _In_reads_bytes_opt_(featureQueryDataSize) const void* featureQueryData,
        UINT featureSupportDataSize,
        _Out_writes_bytes_(featureSupportDataSize) void* featureSupportData
        ) noexcept final;
    
    HRESULT STDMETHODCALLTYPE CreateOperator(
        const DML_OPERATOR_DESC* desc,
        REFIID riid,
        _COM_Outptr_opt_ void** ppv
        ) noexcept final;
    
    HRESULT STDMETHODCALLTYPE CompileOperator(
        IDMLOperator* op,
        DML_EXECUTION_FLAGS flags,
        REFIID riid,
        _COM_Outptr_opt_ void** ppv
        ) noexcept final;

    HRESULT STDMETHODCALLTYPE CreateOperatorInitializer(
        UINT operatorCount,
        _In_reads_opt_(operatorCount) IDMLCompiledOperator* const* operators,
        REFIID riid,
        _COM_Outptr_ void** ppv
        ) noexcept final;
    
    HRESULT STDMETHODCALLTYPE CreateCommandRecorder(
        REFIID riid,
        _COM_Outptr_ void** ppv
        ) noexcept final;
    
    HRESULT STDMETHODCALLTYPE CreateBindingTable(
        _In_opt_ const DML_BINDING_TABLE_DESC* desc,
        REFIID riid,
        _COM_Outptr_ void** ppv
        ) noexcept final;
    
    HRESULT STDMETHODCALLTYPE Evict(UINT count, _In_reads_(count) IDMLPageable* const* ppObjects) noexcept final;
    HRESULT STDMETHODCALLTYPE MakeResident(UINT count, _In_reads_(count) IDMLPageable* const* ppObjects) noexcept final;
    HRESULT STDMETHODCALLTYPE GetDeviceRemovedReason() noexcept final;
    HRESULT STDMETHODCALLTYPE GetParentDevice(REFIID riid, _COM_Outptr_ void** ppv) noexcept final;

    // IDMLDevice1
    HRESULT STDMETHODCALLTYPE CompileGraph(
        const DML_GRAPH_DESC* desc,
        DML_EXECUTION_FLAGS flags,
        REFIID riid,
        _COM_Outptr_opt_ void** ppv
        ) noexcept final;

private:
    Microsoft::WRL::ComPtr<IDMLDevice1> m_impl;
    Microsoft::WRL::ComPtr<IDxDispatchLogger> m_logger;
    const CommandLineArgs& m_args;
};