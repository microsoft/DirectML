#pragma once

#include "pch.h"

// Reads a NumPy array file in memory.
void ReadNpy(
    std::span<std::byte const> fileData,
    /*out*/DML_TENSOR_DATA_TYPE& dataType,
    /*out*/std::vector<int32_t>& dimensions,
    /*out*/std::vector<std::byte>& arrayByteData
    );

// Writes tensor data to in-memory NumPy file data.
void WriteNpy(
    std::span<std::byte const> arrayByteData,
    DML_TENSOR_DATA_TYPE dataType,
    std::span<int32_t const> dimensions,
    /*out*/std::vector<std::byte>& fileData
    );
