#include "pch.h"
#include "ModuleInfo.h"

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

ModuleInfo GetModuleInfo(std::string moduleName)
{
    // get modules loaded into memory and match with moduleName
    auto moduleHandle = GetModuleHandleA(moduleName.c_str());

    ModuleInfo moduleInfo = {};
    moduleInfo.path = wil::GetModuleFileNameW(moduleHandle).get();
    moduleInfo.version = L"Unknown";

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

ModuleInfo GetModuleInfo(std::string moduleName)
{
    ModuleInfo moduleInfo = {};
    return moduleInfo;
}

#endif

void PrintModuleInfo(std::string name, const ModuleInfo& loadedModuleInfo, std::string_view configVersion)
{
    std::cout << name << ":\n";
    std::cout << "- Configure Version : " << configVersion << std::endl;
#if defined(_WIN32) && !defined(_GAMING_XBOX)
    std::wcout << L"- Loaded Path       : " << loadedModuleInfo.path << std::endl;
    std::wcout << L"- Loaded Version    : " << loadedModuleInfo.version << std::endl;
#endif
    std::cout << std::endl;
}