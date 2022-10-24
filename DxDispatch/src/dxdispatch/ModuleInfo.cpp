#include "pch.h"
#include "ModuleInfo.h"
#include "config.h"

#ifdef _WIN32

#ifdef _GAMING_XBOX
// QueryUnbiasedInterruptTime is not declared in WINAPI_PARTITION_GAMES
// Undefining _APISETREALTIME_ will avoid a few win32_helpers declarations
// that won't work in WINAPI_PARTITION_GAMES.
#undef _APISETREALTIME_
#endif

#include <wil/win32_helpers.h>

struct LanguageAndCodePage
{
    WORD language;
    WORD codePage;
};

std::optional<ModuleInfo> GetModuleInfo(std::string moduleName)
{
    auto moduleHandle = GetModuleHandleA(moduleName.c_str());
    if (!moduleHandle)
    {
        return std::nullopt;
    }

    ModuleInfo moduleInfo = {};
    moduleInfo.path = wil::GetModuleFileNameW(moduleHandle).get();

#ifndef _GAMING_XBOX
    DWORD versionInfoHandle = 0;
    auto versionInfoSizeInBytes = GetFileVersionInfoSizeW(moduleInfo.path.data(), &versionInfoHandle);
    if (versionInfoSizeInBytes)
    {
        std::vector<std::byte> versionInfo(versionInfoSizeInBytes);
        if (GetFileVersionInfoW(moduleInfo.path.data(), versionInfoHandle, versionInfoSizeInBytes, versionInfo.data()))
        {
            // Read the list of languages and code pages stored in the file.
            LanguageAndCodePage* translationData;
            UINT translationDataSize = 0;
            if (VerQueryValueW(versionInfo.data(), L"\\VarFileInfo\\Translation", (LPVOID*)&translationData, &translationDataSize))
            {
                // Use the en-US language code (1033) if it's not stored in the module.
                WORD languageCode = translationData->language;
                if (!languageCode)
                {
                    languageCode = 1033;
                }

                // Query the product version string using the language & codepage.
                auto query = fmt::format(L"\\StringFileInfo\\{0:04x}{1:04x}\\ProductVersion", languageCode, translationData->codePage);

                LPVOID productVersionStringDataStart = nullptr;
                UINT productVersionStringDataSizeInChars = 0;
                if (VerQueryValueW(versionInfo.data(),
                    query.c_str(),
                    &productVersionStringDataStart,
                    &productVersionStringDataSizeInChars))
                {
                    moduleInfo.version.resize(productVersionStringDataSizeInChars);
                    memcpy(moduleInfo.version.data(), productVersionStringDataStart, productVersionStringDataSizeInChars * sizeof(wchar_t));
                }
            }
        }
    }
#endif

    return moduleInfo;
}

#else // !_WIN32

std::optional<ModuleInfo> GetModuleInfo(std::string moduleName)
{
    return std::nullopt;
}

#endif

void PrintModuleInfo(std::string name, const std::optional<ModuleInfo>& loadedModuleInfo, std::string_view configVersion)
{
    std::cout << name << ":\n";
    std::cout << "- Configured Version : " << configVersion << std::endl;
    if (loadedModuleInfo)
    {
        if (!loadedModuleInfo->path.empty())
        {
            std::wcout << L"- Loaded Path        : " << loadedModuleInfo->path << std::endl;
        }
        if (!loadedModuleInfo->version.empty())
        {
            std::wcout << L"- Loaded Version     : " << loadedModuleInfo->version << std::endl;
        }
    }
    std::cout << std::endl;
}

void PrintDependencies()
{
        PrintModuleInfo("DirectML", GetModuleInfo("directml"), c_directmlConfig);

#ifdef _GAMING_XBOX_SCARLETT
        PrintModuleInfo("D3D12", GetModuleInfo("d3d12_xs"), c_d3d12Config);
        PrintModuleInfo("DXCompiler", GetModuleInfo("dxcompiler_xs"), c_dxcompilerConfig);
        PrintModuleInfo("PIX", GetModuleInfo("pixevt"), c_pixConfig);
#else
        PrintModuleInfo("D3D12", GetModuleInfo("d3d12core"), c_d3d12Config);
        PrintModuleInfo("DXCompiler", GetModuleInfo("dxcompiler"), c_dxcompilerConfig);
        PrintModuleInfo("PIX", GetModuleInfo("winpixeventruntime"), c_pixConfig);
#endif

        PrintModuleInfo("ONNX Runtime", GetModuleInfo("onnxruntime"), c_ortConfig);
}