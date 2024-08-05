#include "pch.h"
#include "DmlTracing.h"

// {1E508EFB-18B8-4705-9C0B-98B1194E9023}
static constexpr GUID c_dmlOperatorMetadata = { 0x1e508efb, 0x18b8, 0x4705, { 0x9c, 0xb, 0x98, 0xb1, 0x19, 0x4e, 0x90, 0x23 } };

struct DmlOperatorMetadata
{
    DML_OPERATOR_TYPE type;
};

WrappedDmlDevice::WrappedDmlDevice(
    IDMLDevice1* impl, 
    IDxDispatchLogger* logger,
    const CommandLineArgs& args
    ) : m_impl(impl), m_logger(logger), m_args(args) {}

void WrappedDmlDevice::ResetTraceData()
{
    m_compileOperatorTraces.clear();
    m_compileGraphTraces.clear();
}

DmlTraceData WrappedDmlDevice::GetTraceData() const
{
    DmlTraceData traceData = {};
    traceData.compileOperatorTraces = m_compileOperatorTraces;
    traceData.compileGraphTraces = m_compileGraphTraces;
    return traceData;
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::GetPrivateData(REFGUID guid, _Inout_ UINT* dataSize, _Out_writes_bytes_opt_(*dataSize) void* data) noexcept
{
    return m_impl->GetPrivateData(guid, dataSize, data);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::SetPrivateData(REFGUID guid, UINT dataSize, _In_reads_bytes_opt_(dataSize) const void* data) noexcept
{
    return m_impl->SetPrivateData(guid, dataSize, data);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::SetPrivateDataInterface(REFGUID guid, _In_opt_ IUnknown* data) noexcept
{
    return m_impl->SetPrivateDataInterface(guid, data);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::SetName(PCWSTR name) noexcept
{
    return m_impl->SetName(name);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::CheckFeatureSupport(
    DML_FEATURE feature,
    UINT featureQueryDataSize,
    _In_reads_bytes_opt_(featureQueryDataSize) const void* featureQueryData,
    UINT featureSupportDataSize,
    _Out_writes_bytes_(featureSupportDataSize) void* featureSupportData
    ) noexcept
{
    return m_impl->CheckFeatureSupport(feature, featureQueryDataSize, featureQueryData, featureSupportDataSize, featureSupportData);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::CreateOperator(
    const DML_OPERATOR_DESC* desc,
    REFIID riid,
    _COM_Outptr_opt_ void** ppv
    ) noexcept
{
    if (!ppv)
    {
        return m_impl->CreateOperator(desc, riid, ppv);
    }
    
    Microsoft::WRL::ComPtr<IDMLOperator> op;
    auto hr = m_impl->CreateOperator(desc, IID_PPV_ARGS(&op));

    if (SUCCEEDED(hr))
    {
        DmlOperatorMetadata metadata = { desc->Type };
        THROW_IF_FAILED(op->SetPrivateData(c_dmlOperatorMetadata, sizeof(DmlOperatorMetadata), &metadata));
        THROW_IF_FAILED(op.CopyTo(riid, ppv));
    }

    return hr;
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::CompileOperator(
    IDMLOperator* op,
    DML_EXECUTION_FLAGS flags,
    REFIID riid,
    _COM_Outptr_opt_ void** ppv
    ) noexcept
{
    DmlOperatorMetadata metadata = {};
    UINT dataSize = sizeof(metadata);
    THROW_IF_FAILED(op->GetPrivateData(c_dmlOperatorMetadata, &dataSize, &metadata));

    Timer timer;
    auto hr = m_impl->CompileOperator(op, flags, riid, ppv);
    timer.End();

    std::unique_lock<std::mutex> lock(m_mutex);
    m_compileOperatorTraces.push_back({metadata.type, timer.DurationInMilliseconds()});

    return hr;
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::CreateOperatorInitializer(
    UINT operatorCount,
    _In_reads_opt_(operatorCount) IDMLCompiledOperator* const* operators,
    REFIID riid,
    _COM_Outptr_ void** ppv
    ) noexcept
{
    return m_impl->CreateOperatorInitializer(operatorCount, operators, riid, ppv);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::CreateCommandRecorder(
    REFIID riid,
    _COM_Outptr_ void** ppv
    ) noexcept
{
    return m_impl->CreateCommandRecorder(riid, ppv);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::CreateBindingTable(
    _In_opt_ const DML_BINDING_TABLE_DESC* desc,
    REFIID riid,
    _COM_Outptr_ void** ppv
    ) noexcept
{
    return m_impl->CreateBindingTable(desc, riid, ppv);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::Evict( UINT count, _In_reads_(count) IDMLPageable* const* ppObjects ) noexcept
{
    return m_impl->Evict(count, ppObjects);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::MakeResident( UINT count, _In_reads_(count) IDMLPageable* const* ppObjects ) noexcept
{
    return m_impl->MakeResident(count, ppObjects);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::GetDeviceRemovedReason() noexcept
{
    return m_impl->GetDeviceRemovedReason();
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::GetParentDevice( REFIID riid, _COM_Outptr_ void** ppv ) noexcept
{
    return m_impl->GetParentDevice(riid, ppv);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::CompileGraph(
    const DML_GRAPH_DESC* desc,
    DML_EXECUTION_FLAGS flags,
    REFIID riid,
    _COM_Outptr_opt_ void** ppv
    ) noexcept
{
    DmlCompileGraphTrace compileTrace = {};

    for (uint32_t nodeIndex = 0; nodeIndex < desc->NodeCount; nodeIndex++)
    {
        const DML_GRAPH_NODE_DESC& node = desc->Nodes[nodeIndex];

        if (node.Type == DML_GRAPH_NODE_TYPE_OPERATOR)
        {
            auto opNode = static_cast<const DML_OPERATOR_GRAPH_NODE_DESC*>(node.Desc);

            DmlOperatorMetadata opMetadata = {};
            UINT dataSize = sizeof(opMetadata);
            THROW_IF_FAILED(opNode->Operator->GetPrivateData(c_dmlOperatorMetadata, &dataSize, &opMetadata));

            compileTrace.opCounts[opMetadata.type]++;
        }
    }

    Timer timer;
    auto hr = m_impl->CompileGraph(desc, flags, riid, ppv);
    timer.End();
    
    compileTrace.durationInMilliseconds = timer.DurationInMilliseconds();

    std::unique_lock<std::mutex> lock(m_mutex);
    m_compileGraphTraces.push_back({std::move(compileTrace)});

    return hr;
}
