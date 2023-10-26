#pragma once

#ifndef WIN32
WINADAPTER_IID(IDxDispatchLogger, 0xE05E128D, 0x9A97, 0x4AEE, 0x85, 0xD8, 0x17, 0x25, 0xC9, 0x2E, 0x41, 0x72);
#endif

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
