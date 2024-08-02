#include "DmlTracing.h"
#include "Timer.h"

WrappedDmlDevice::WrappedDmlDevice(IDMLDevice1* impl, IDxDispatchLogger* logger) : m_impl(impl), m_logger(logger) {}

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
    return m_impl->CreateOperator(desc, riid, ppv);
}

HRESULT STDMETHODCALLTYPE WrappedDmlDevice::CompileOperator(
    IDMLOperator* op,
    DML_EXECUTION_FLAGS flags,
    REFIID riid,
    _COM_Outptr_opt_ void** ppv
    ) noexcept
{
    m_logger->LogInfo("IDMLDevice::CompileOperator called");
    return m_impl->CompileOperator(op, flags, riid, ppv);
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
    Timer timer;
    timer.Start();
    auto graph = m_impl->CompileGraph(desc, flags, riid, ppv);
    timer.End();
    m_logger->LogInfo(fmt::format("IDMLDevice1::CompileGraph completed in {:.2f} ms", timer.DurationInMilliseconds()).c_str());
    return graph;
}
