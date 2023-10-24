#pragma once

class DxDispatchConsoleLogger : public IDxDispatchLogger
{
public:
    DxDispatchConsoleLogger() = default;
    // IUnknown
    HRESULT STDMETHODCALLTYPE QueryInterface(
        REFIID riid,
        _COM_Outptr_ void** ppvObject) final;

    ULONG STDMETHODCALLTYPE AddRef(void) final;

    ULONG STDMETHODCALLTYPE Release(void) final;

    // IDxDispatchLogger
    void STDMETHODCALLTYPE  LogInfo(
        _In_ PCSTR message) final;

    void STDMETHODCALLTYPE  LogWarning(
        _In_ PCSTR message) final;

    void STDMETHODCALLTYPE  LogError(
        _In_ PCSTR message) final;

    void STDMETHODCALLTYPE  LogCommandStarted(
        UINT32 index,
        _In_ PCSTR jsonString)  final;

    void STDMETHODCALLTYPE  LogCommandCompleted(
        UINT32 index,
        HRESULT hr,
        _In_opt_ PCSTR statusString) final;

protected:
    virtual ~DxDispatchConsoleLogger() = default;
    std::atomic<ULONG>        m_refCount = 0;
};
