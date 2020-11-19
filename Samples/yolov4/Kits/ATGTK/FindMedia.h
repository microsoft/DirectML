//--------------------------------------------------------------------------------------
// File: FindMedia.h
//
// Helper function to find the location of a media file for Windows desktop apps
// since they lack appx packaging support.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//-------------------------------------------------------------------------------------

#pragma once

#include <exception>
#include <string.h>


namespace DX
{
    inline void FindMediaFile(
        _Out_writes_(cchDest) wchar_t* strDestPath,
        _In_ int cchDest,
        _In_z_ const wchar_t* strFilename,
        _In_opt_ const wchar_t* const * searchFolders = nullptr)
    {
        if (!strFilename || strFilename[0] == 0 || !strDestPath || cchDest < 10)
            throw std::invalid_argument("FindMediaFile");

        // Check CWD for quick out
        wcscpy_s(strDestPath, size_t(cchDest), strFilename);
        if (GetFileAttributesW(strDestPath) != 0xFFFFFFFF)
            return;

        // Get the exe name, and exe path
        wchar_t strExePath[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, strExePath, MAX_PATH);
        strExePath[MAX_PATH - 1] = 0;

        static const wchar_t* s_defSearchFolders[] =
        {
            L"Assets",
            L"Media",
            L"Media\\Textures",
            L"Media\\Fonts",
            L"Media\\Meshes",
            L"Media\\PBR",
            L"Media\\CubeMaps",
            L"Media\\HDR",
            L"Media\\Sounds",
            L"Media\\Videos",
            0
        };

        if (!searchFolders)
            searchFolders = s_defSearchFolders;

        // Search all parent directories starting at .\ and using strFilename as the leaf name
        wchar_t strLeafName[MAX_PATH] = {};
        wcscpy_s(strLeafName, MAX_PATH, strFilename);

        wchar_t strFullPath[MAX_PATH] = {};
        wchar_t strFullFileName[MAX_PATH] = {};
        wchar_t strSearch[MAX_PATH] = {};
        wchar_t* strFilePart = nullptr;

        GetFullPathNameW(strExePath, MAX_PATH, strFullPath, &strFilePart);
        if (!strFilePart)
            throw std::exception("FindMediaFile");

        while (strFilePart && *strFilePart != '\0')
        {
            swprintf_s(strFullFileName, MAX_PATH, L"%ls\\%ls", strFullPath, strLeafName);
            if (GetFileAttributesW(strFullFileName) != 0xFFFFFFFF)
            {
                wcscpy_s(strDestPath, size_t(cchDest), strFullFileName);
                return;
            }

            for (const wchar_t* const * searchFolder = searchFolders; *searchFolder != 0; ++searchFolder)
            {
                swprintf_s(strFullFileName, MAX_PATH, L"%ls\\%ls\\%ls", strFullPath, *searchFolder, strLeafName);
                if (GetFileAttributesW(strFullFileName) != 0xFFFFFFFF)
                {
                    wcscpy_s(strDestPath, size_t(cchDest), strFullFileName);
                    return;
                }
            }

            swprintf_s(strSearch, MAX_PATH, L"%ls\\..", strFullPath);
            GetFullPathNameW(strSearch, MAX_PATH, strFullPath, &strFilePart);
        }

        // On failure, return the file as the path but also throw an error
        wcscpy_s(strDestPath, size_t(cchDest), strFilename);

        throw std::exception("File not found");
    }
}