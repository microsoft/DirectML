#pragma once

struct ModuleInfo
{
    std::wstring path;
    std::wstring version;
};

ModuleInfo GetModuleInfo(std::string moduleName);