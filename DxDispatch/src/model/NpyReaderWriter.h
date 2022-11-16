#pragma once

bool IsNpyFilenameExtension(std::string_view filename);

// Reads a NumPy array file in memory.
void ReadNpy(
    std::span<const std::byte> fileData,
    /*out*/DML_TENSOR_DATA_TYPE& dataType,
    /*out*/std::vector<int32_t>& dimensions,
    /*out*/std::vector<std::byte>& arrayByteData
    );

// Writes tensor data to in-memory NumPy file data.
void WriteNpy(
    std::span<const std::byte> arrayByteData,
    DML_TENSOR_DATA_TYPE dataType,
    std::span<const int32_t> dimensions,
    /*out*/std::vector<std::byte>& fileData
    );
