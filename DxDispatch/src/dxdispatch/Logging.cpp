#include "pch.h"
#include <iostream>

HRESULT DxDispatchConsoleLogger::QueryInterface(
    REFIID riid,
    _COM_Outptr_ void ** ppvObject)
{
    *ppvObject = nullptr;
    if ((IID_IUnknown == riid) ||
        __uuidof(IDxDispatchLogger) == riid)
    {
        *ppvObject = static_cast<IDxDispatchLogger*>(this);
    }
    else
    {
        return E_NOTIMPL;
    }
    AddRef();
    return S_OK;
}

ULONG DxDispatchConsoleLogger::AddRef(void)
{
    return ++m_refCount;
}

ULONG DxDispatchConsoleLogger::Release(void)
{
    auto ref = --m_refCount;
    if (ref == 0)
    {
        delete this;
    }
    return ref;
}

void DxDispatchConsoleLogger::LogInfo(_In_ PCSTR msg)
{
    std::cout << msg << std::endl;
#ifdef WIN32
    OutputDebugStringA(fmt::format("[INFO] : {}", msg).c_str());
#endif
}

void DxDispatchConsoleLogger::LogWarning(_In_ PCSTR msg)
{
    std::cerr << msg << std::endl;
#ifdef WIN32
    OutputDebugStringA(fmt::format("[WARNING] : {}", msg).c_str());
#endif
}

void DxDispatchConsoleLogger::LogError(_In_ PCSTR msg)
{
    std::cerr << msg << std::endl;
#ifdef WIN32
    OutputDebugStringA(fmt::format("[ERROR] : {}", msg).c_str());
#endif
}

void STDMETHODCALLTYPE  DxDispatchConsoleLogger::LogCommandStarted(
    UINT32 index,
    _In_ PCSTR jsonString)
{
    auto outputString = fmt::format("[StartCmd][{:>04}]\t: {}", index, jsonString);
    std::cout << outputString << std::endl;
#ifdef WIN32
    OutputDebugStringA(outputString.c_str());
#endif
}

void STDMETHODCALLTYPE  DxDispatchConsoleLogger::LogCommandCompleted(
    UINT32 index,
    HRESULT hr,
    _In_opt_ PCSTR statusString)
{
    auto outputString = fmt::format("[EndCmd][{:>04}]\t: hr={:<#010x} {}\n", index, hr, statusString);
    std::cout << outputString << std::endl ;
#ifdef WIN32
    OutputDebugStringA(outputString.c_str());
#endif
}
