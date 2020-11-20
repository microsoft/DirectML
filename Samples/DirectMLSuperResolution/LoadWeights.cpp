//--------------------------------------------------------------------------------------
// LoadWeights.cpp
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. Copyright (C) NVIDIA Corporation. All rights reserved.
// Licensed under the MIT License.
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "LoadWeights.h"
#include <iostream>
#include <string>
#include <fstream>

namespace
{
    const int c_bufferLength = 256;
}

// Loads weight values from a binary file.
bool LoadWeights(const std::string& fpath, WeightMapType& weightMap)
{
    std::ifstream input(fpath, std::ifstream::binary);
    if (!(input) || !(input.good()) || !(input.is_open()))
    {
        std::cerr << "Unable to open weight file: " << fpath << std::endl;
        return false;
    }

    int32_t count;
    try
    {
        input.read(reinterpret_cast<char*>(&count), 4);
    }
    catch (const std::ifstream::failure&)
    {
        std::cerr << "Invalid weight map file: " << fpath << std::endl;
        return false;
    }
    if (count < 0)
    {
        std::cerr << "Invalid weight map file: " << fpath << std::endl;
        return false;
    }
    std::cout << "Number of weight tensors: " + std::to_string(count) << std::endl;

    uint32_t name_len;
    uint32_t w_len;
    char name_buf[c_bufferLength];

    try
    {
        while (count--)
        {
            input.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
            if (name_len > c_bufferLength - 1)
            {
                std::cerr << "name_len exceeds c_bufferLength: " << name_len
                    << " vs " << c_bufferLength - 1 << std::endl;
                return false;
            }
            input.read(name_buf, name_len);
            name_buf[name_len] = '\0';
            std::string name(name_buf);

            input.read(reinterpret_cast<char*>(&w_len), sizeof(uint32_t));
            weightMap[name] = WeightsType(w_len);
            input.read(reinterpret_cast<char*>(weightMap[name].data()), sizeof(float) * w_len);

            std::cout << "Loaded tensor: " + name + " -> " + std::to_string(w_len) << std::endl;
        }

        input.close();
    }
    catch (const std::ifstream::failure&)
    {
        std::cerr << "Invalid tensor data" << std::endl;
        return false;
    }
    catch (const std::out_of_range&)
    {
        std::cerr << "Invalid tensor format" << std::endl;
        return false;
    }

    return true;
}
