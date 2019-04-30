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
    inline void FindMediaFile(_Out_writes_(cchDest) wchar_t* strDestPath, _In_ int cchDest, _In_z_ const wchar_t* strFilename)
    {
        bool bFound = false;

        if (!strFilename || strFilename[0] == 0 || !strDestPath || cchDest < 10)
            throw std::invalid_argument("FindMediaFile");

        // Get the exe name, and exe path
        wchar_t strExePath[MAX_PATH] = {};
        wchar_t strExeName[MAX_PATH] = {};
        wchar_t* strLastSlash = nullptr;
        GetModuleFileNameW(nullptr, strExePath, MAX_PATH);
        strExePath[MAX_PATH - 1] = 0;
        strLastSlash = wcsrchr(strExePath, TEXT('\\'));
        if (strLastSlash)
        {
            wcscpy_s(strExeName, MAX_PATH, &strLastSlash[1]);

            // Chop the exe name from the exe path
            *strLastSlash = 0;

            // Chop the .exe from the exe name
            strLastSlash = wcsrchr(strExeName, TEXT('.'));
            if (strLastSlash)
                *strLastSlash = 0;
        }

        wcscpy_s(strDestPath, cchDest, strFilename);
        if (GetFileAttributesW(strDestPath) != 0xFFFFFFFF)
            return;

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
                wcscpy_s(strDestPath, cchDest, strFullFileName);
                bFound = true;
                break;
            }

            swprintf_s(strFullFileName, MAX_PATH, L"%ls\\%ls\\%ls", strFullPath, strExeName, strLeafName);
            if (GetFileAttributesW(strFullFileName) != 0xFFFFFFFF)
            {
                wcscpy_s(strDestPath, cchDest, strFullFileName);
                bFound = true;
                break;
            }

            swprintf_s(strSearch, MAX_PATH, L"%ls\\..", strFullPath);
            GetFullPathNameW(strSearch, MAX_PATH, strFullPath, &strFilePart);
        }
        if (bFound)
            return;

        // On failure, return the file as the path but also return an error code
        wcscpy_s(strDestPath, cchDest, strFilename);

        throw std::exception("File not found");
    }
}