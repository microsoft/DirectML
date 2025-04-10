#pragma once

enum class ImageTensorLayout
{
    NCHW,
    NHWC,
};

enum class ImageTensorChannelOrder
{
    Unspecified,
    RGB,
    RGBA,
    BGR,
    Grayscale,
};

struct ImageTensorInfo
{
    DML_TENSOR_DATA_TYPE dataType;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint64_t sizeInBytes;
    ImageTensorLayout layout;
    ImageTensorChannelOrder channelOrder;
};

std::vector<std::byte> ReadTensorFromImage(
    std::filesystem::path srcPath,
    const ImageTensorInfo& dstTensorInfo
);

void WriteTensorToImage(
    std::filesystem::path dstPath,
    std::span<const std::byte> srcData,
    const ImageTensorInfo& srcInfo
);
