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

void WrappedDmlDevice::ClearState()
{
    m_compileGraphTimings.rawSamples.clear();
    m_compileOpTimings.rawSamples.clear();
}

void WrappedDmlDevice::PrintTracingInfo()
{
    auto compileOpStats = Timings::ComputeStats(m_compileOpTimings.rawSamples);
    auto compileGraphStats = Timings::ComputeStats(m_compileGraphTimings.rawSamples);

    m_logger->LogInfo("IDMLDevice::CompileOperator timings:");
    m_logger->LogInfo(fmt::format("  Count: {}", compileOpStats.count).c_str());
    m_logger->LogInfo(fmt::format("  Sum: {:.4f} ms", compileOpStats.sum).c_str());
    m_logger->LogInfo(fmt::format("  Average: {:.4f} ms", compileOpStats.average).c_str());
    m_logger->LogInfo(fmt::format("  Median: {:.4f} ms", compileOpStats.median).c_str());
    m_logger->LogInfo(fmt::format("  Min: {:.4f} ms", compileOpStats.min).c_str());
    m_logger->LogInfo(fmt::format("  Max: {:.4f} ms", compileOpStats.max).c_str());

    for (size_t i = 0; i < m_opCompiles.size(); i++)
    {
        const auto& compileTrace = m_opCompiles[i];
        m_logger->LogInfo(fmt::format("Operator[{}]:", (uint32_t)compileTrace.type).c_str());
    }

    m_logger->LogInfo("IDMLDevice::CompileGraph timings:");
    m_logger->LogInfo(fmt::format("  Count: {}", compileGraphStats.count).c_str());
    m_logger->LogInfo(fmt::format("  Sum: {:.4f} ms", compileGraphStats.sum).c_str());
    m_logger->LogInfo(fmt::format("  Average: {:.4f} ms", compileGraphStats.average).c_str());
    m_logger->LogInfo(fmt::format("  Median: {:.4f} ms", compileGraphStats.median).c_str());
    m_logger->LogInfo(fmt::format("  Min: {:.4f} ms", compileGraphStats.min).c_str());
    m_logger->LogInfo(fmt::format("  Max: {:.4f} ms", compileGraphStats.max).c_str());

    for (size_t i = 0; i < m_graphCompiles.size(); i++)
    {
        const auto& compileTrace = m_graphCompiles[i];
        m_logger->LogInfo(fmt::format("Graph[{}]:", i).c_str());
        for (const auto& [opType, count] : compileTrace.opCounts)
        {
            m_logger->LogInfo(fmt::format("  {} : {}", (uint32_t)opType, count).c_str());
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
    m_opCompiles.push_back({wrappedOp->GetType()});
    
    ScopeTimer timer([&](double durationInMilliseconds){
        m_compileOpTimings.rawSamples.push_back(durationInMilliseconds);
    });

    // Compile using the unwrapped IDMLOperator implementation.
    return m_impl->CompileOperator(wrappedOp->Impl(), flags, riid, ppv);
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
    CompileGraphTrace compileTrace = {};

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

    m_graphCompiles.push_back({std::move(compileTrace)});

    ScopeTimer timer([&](double durationInMilliseconds){
        m_compileGraphTimings.rawSamples.push_back(durationInMilliseconds);
    });

    return m_impl->CompileGraph(&unwrappedDesc, flags, riid, ppv);
}
