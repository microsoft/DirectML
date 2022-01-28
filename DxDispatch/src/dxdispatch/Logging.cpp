#include "pch.h"
#include <iostream>

void LogInfo(std::string_view msg)
{
    std::cout << msg << std::endl;
#ifdef WIN32
    OutputDebugStringA(fmt::format("[INFO] : {}\n", msg).c_str());
#endif
}

void LogError(std::string_view msg)
{
    std::cerr << msg << std::endl;
#ifdef WIN32
    OutputDebugStringA(fmt::format("[ERROR] : {}\n", msg).c_str());
#endif
}

void LogError(std::wstring_view msg)
{
    std::wcerr << msg << std::endl;
#ifdef WIN32
    OutputDebugStringW(fmt::format(L"[ERROR] : {}\n", msg).c_str());
#endif
}