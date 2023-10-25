#pragma once

class DxDispatchConsoleLogger : public Microsoft::WRL::Base<IDxDispatchLogger>
{
public:
    DxDispatchConsoleLogger() = default;

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
};
