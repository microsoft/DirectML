#pragma once

struct ModuleInfo
{
    std::wstring path;
    std::wstring version;
};

std::optional<ModuleInfo> GetModuleInfo(std::string moduleName);

void PrintDependencies();