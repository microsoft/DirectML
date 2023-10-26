#include "pch.h"
#include <iostream>

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
