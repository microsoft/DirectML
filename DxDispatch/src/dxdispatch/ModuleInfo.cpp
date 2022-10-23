#include "pch.h"
#include "ModuleInfo.h"
#include <wil/win32_helpers.h>

#ifdef _WIN32

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

    return moduleInfo;
}

#else

ModuleInfo GetModuleInfo(std::string moduleName)
{
    ModuleInfo moduleInfo = {};
    return moduleInfo;
}

#endif