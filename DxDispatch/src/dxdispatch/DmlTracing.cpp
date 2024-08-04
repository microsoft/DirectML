#include "pch.h"
#include "DmlTracing.h"

WrappedDmlOperator::WrappedDmlOperator(IDMLOperator* impl, const DML_OPERATOR_DESC* desc) : 
    m_impl(impl), m_type(desc->Type)
{}

HRESULT STDMETHODCALLTYPE WrappedDmlOperator::GetDevice(REFIID riid, _COM_Outptr_ void** ppv) noexcept
{
    return m_impl->GetDevice(riid, ppv);
}

HRESULT STDMETHODCALLTYPE WrappedDmlOperator::GetPrivateData(REFGUID guid, _Inout_ UINT* dataSize, _Out_writes_bytes_opt_(*dataSize) void* data) noexcept
{
    return m_impl->GetPrivateData(guid, dataSize, data);
}

HRESULT STDMETHODCALLTYPE WrappedDmlOperator::SetPrivateData(REFGUID guid, UINT dataSize, _In_reads_bytes_opt_(dataSize) const void* data) noexcept
{
    return m_impl->SetPrivateData(guid, dataSize, data);
}

HRESULT STDMETHODCALLTYPE WrappedDmlOperator::SetPrivateDataInterface(REFGUID guid, _In_opt_ IUnknown* data) noexcept
{
    return m_impl->SetPrivateDataInterface(guid, data);
}

HRESULT STDMETHODCALLTYPE WrappedDmlOperator::SetName(PCWSTR name) noexcept
{
    return m_impl->SetName(name);
}

WrappedDmlDevice::WrappedDmlDevice(
    IDMLDevice1* impl, 
    IDxDispatchLogger* logger,
    const CommandLineArgs& args
    ) : m_impl(impl), m_logger(logger), m_args(args) {}

void WrappedDmlDevice::ResetTraceData()
{
    m_traceData = {};
}

void WrappedDmlDevice::PrintTracingInfo()
{
    for (size_t i = 0; i < m_compileOpTraces.size(); i++)
    {
        const auto& trace = m_compileOpTraces[i];
        m_logger->LogInfo(fmt::format("IDMLDevice::CompileOperator[{}] ('{}'): {:.4f} ms", 
            i, 
            (uint32_t)trace.type,
            trace.durationInMilliseconds).c_str());
    }

    for (size_t i = 0; i < m_compileGraphTraces.size(); i++)
    {
        const auto& trace = m_compileGraphTraces[i];
        m_logger->LogInfo(fmt::format("IDMLDevice::CompileGraph[{}]: {:.4f} ms", i, trace.durationInMilliseconds).c_str());
        for (const auto& [opType, count] : trace.opCounts)
        {
            m_logger->LogInfo(fmt::format("  '{}' : {}", (uint32_t)opType, count).c_str());
        }
    }
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
    Microsoft::WRL::ComPtr<IDMLOperator> op;
    auto hr = m_impl->CreateOperator(desc, IID_PPV_ARGS(&op));

    if (FAILED(hr))
    {
        return hr;
    }

    auto wrappedOp = Microsoft::WRL::Make<WrappedDmlOperator>(op.Get(), desc);
    return wrappedOp.CopyTo(riid, ppv);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::CompileOperator(
    IDMLOperator* wrappedOpInterface,
    DML_EXECUTION_FLAGS flags,
    REFIID riid,
    _COM_Outptr_opt_ void** ppv
    ) noexcept
{
    auto wrappedOp = static_cast<WrappedDmlOperator*>(wrappedOpInterface);

    Timer timer;
    auto hr = m_impl->CompileOperator(wrappedOp->Impl(), flags, riid, ppv);
    timer.End();

    std::unique_lock<std::mutex> lock(m_mutex);
    m_traceData.compileOpTraces.push_back({wrappedOp->GetType(), timer.DurationInMilliseconds()});

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
    const DML_GRAPH_DESC* wrappedDesc,
    DML_EXECUTION_FLAGS flags,
    REFIID riid,
    _COM_Outptr_opt_ void** ppv
    ) noexcept
{
    DmlCompileGraphTrace compileTrace = {};

    // DML_GRAPH_DESC operator-type nodes reference IDMLOperators, which are wrapped when tracing.
    // The rest of the graph desc can pass through unmodified, but operator-type nodes need to be unwrapped.
    DML_GRAPH_DESC unwrappedDesc = *wrappedDesc;

    std::vector<DML_GRAPH_NODE_DESC> unwrappedNodes(wrappedDesc->NodeCount);
    std::vector<DML_OPERATOR_GRAPH_NODE_DESC> unwrappedOpNodes(wrappedDesc->NodeCount);
    unwrappedDesc.Nodes = unwrappedNodes.data();

    for (uint32_t nodeIndex = 0; nodeIndex < wrappedDesc->NodeCount; nodeIndex++)
    {
        DML_GRAPH_NODE_DESC& unwrappedNode = unwrappedNodes[nodeIndex];
        const DML_GRAPH_NODE_DESC& wrappedNode = wrappedDesc->Nodes[nodeIndex];

        unwrappedNode = wrappedNode;

        if (wrappedNode.Type == DML_GRAPH_NODE_TYPE_OPERATOR)
        {
            DML_OPERATOR_GRAPH_NODE_DESC& unwrappedOpNode = unwrappedOpNodes[nodeIndex];

            const DML_OPERATOR_GRAPH_NODE_DESC& wrappedOpNode = *static_cast<const DML_OPERATOR_GRAPH_NODE_DESC*>(wrappedNode.Desc);
            auto wrappedOp = static_cast<WrappedDmlOperator*>(wrappedOpNode.Operator);

            unwrappedOpNode = wrappedOpNode;
            unwrappedOpNode.Operator = wrappedOp->Impl();
            unwrappedNode.Desc = &unwrappedOpNode;

            compileTrace.opCounts[wrappedOp->GetType()]++;
        }
    }

    Timer timer;
    auto hr = m_impl->CompileGraph(&unwrappedDesc, flags, riid, ppv);
    timer.End();
    
    compileTrace.durationInMilliseconds = timer.DurationInMilliseconds();

    std::unique_lock<std::mutex> lock(m_mutex);
    m_traceData.compileGraphTraces.push_back({std::move(compileTrace)});

    return hr;
}
