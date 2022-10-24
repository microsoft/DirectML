#pragma once

struct ModuleInfo
{
    std::wstring path;
    std::wstring version;
};

ModuleInfo GetModuleInfo(std::string moduleName);

void PrintModuleInfo(std::string name, const ModuleInfo& loadedModuleInfo, std::string_view configVersion);