//--------------------------------------------------------------------------------------
// File: ReadData.h
//
// Helper for loading binary data files from disk
//
// For Windows desktop apps, it looks for files in the same folder as the running EXE if
// it can't find them in the CWD
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//-------------------------------------------------------------------------------------

#pragma once

#include <stdint.h>
#include <exception>
#include <fstream>
#include <vector>


namespace DX
{
    inline std::vector<uint8_t> ReadData(_In_z_ const wchar_t* name)
    {
        std::ifstream inFile(name, std::ios::in | std::ios::binary | std::ios::ate);

#if !defined(WINAPI_FAMILY) || (WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP)
        if (!inFile)
        {
            wchar_t moduleName[_MAX_PATH];
            if (!GetModuleFileNameW(nullptr, moduleName, _MAX_PATH))
                throw std::exception("GetModuleFileName");

            wchar_t drive[_MAX_DRIVE];
            wchar_t path[_MAX_PATH];

            if (_wsplitpath_s(moduleName, drive, _MAX_DRIVE, path, _MAX_PATH, nullptr, 0, nullptr, 0))
                throw std::exception("_wsplitpath_s");

            wchar_t filename[_MAX_PATH];
            if (_wmakepath_s(filename, _MAX_PATH, drive, path, name, nullptr))
                throw std::exception("_wmakepath_s");

            inFile.open(filename, std::ios::in | std::ios::binary | std::ios::ate);
        }
#endif

        if (!inFile)
            throw std::exception("ReadData");

        std::streampos len = inFile.tellg();
        if (!inFile)
            throw std::exception("ReadData");

        std::vector<uint8_t> blob;
        blob.resize(size_t(len));

        inFile.seekg(0, std::ios::beg);
        if (!inFile)
            throw std::exception("ReadData");

        inFile.read(reinterpret_cast<char*>(blob.data()), len);
        if (!inFile)
            throw std::exception("ReadData");

        inFile.close();

        return blob;
    }
}