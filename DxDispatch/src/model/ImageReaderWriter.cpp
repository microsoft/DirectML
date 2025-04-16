#include "pch.h"
#include "StdSupport.h"
#include "ImageReaderWriter.h"

#ifndef WIN32
std::vector<std::byte> ReadTensorFromImage(
    std::filesystem::path srcPath,
    const ImageTensorInfo& dstTensorInfo)
{
    throw std::runtime_error("ReadTensorFromImage is not implemented on this platform.");
}

void WriteTensorToImage(
    std::filesystem::path dstPath,
    std::span<const std::byte> srcData,
    const ImageTensorInfo& srcInfo)
{
    throw std::runtime_error("WriteTensorToImage is not implemented on this platform.");
}

#else // Win32

#include <wincodec.h>

using Microsoft::WRL::ComPtr;

size_t GetDataTypeSize(DML_TENSOR_DATA_TYPE dataType)
{
    switch (dataType)
    {
        case DML_TENSOR_DATA_TYPE_FLOAT32: return sizeof(float);
        case DML_TENSOR_DATA_TYPE_FLOAT16: return sizeof(uint16_t);
        case DML_TENSOR_DATA_TYPE_UINT8: return sizeof(uint8_t);
        case DML_TENSOR_DATA_TYPE_INT8: return sizeof(int8_t);

        default: throw std::invalid_argument("Unsupported data type");
    }

    return 0;
}

// Converts a pixel buffer (HWC layout with uint8 components) to a tensor.
template <typename T>
std::vector<std::byte> ConvertPixelsToTensor(std::span<const std::byte> src, const ImageTensorInfo& tensorInfo)
{
    if (sizeof(T) != GetDataTypeSize(tensorInfo.dataType))
    {
        throw std::invalid_argument("Unexpected data type size");
    }

    if (tensorInfo.sizeInBytes < tensorInfo.channels * tensorInfo.height * tensorInfo.width * GetDataTypeSize(tensorInfo.dataType))
    {
        throw std::invalid_argument("Unexpected tensor size (too small)");
    }

    std::vector<std::byte> dstRaw(tensorInfo.sizeInBytes);
    std::span<T> dst(reinterpret_cast<T*>(dstRaw.data()), dstRaw.size() / sizeof(T));

    for (size_t pixelIndex = 0; pixelIndex < tensorInfo.height * tensorInfo.width; pixelIndex++)
    {
        for (size_t channelIndex = 0; channelIndex < tensorInfo.channels; channelIndex++)
        {
            // src is always HWC layout.
            size_t srcOffset = pixelIndex * tensorInfo.channels + channelIndex;

            // dst is either HWC or CHW layout.
            size_t dstOffset = tensorInfo.layout == ImageTensorLayout::NHWC ?
                srcOffset :
                pixelIndex + channelIndex * tensorInfo.height * tensorInfo.width;

            dst[dstOffset] = static_cast<float>(src[srcOffset]) / 255.0f;
        }
    }

    return dstRaw;
}

std::vector<std::byte> ReadTensorFromImage(const std::filesystem::path& srcPath, const ImageTensorInfo& dstTensorInfo)
{
    WICPixelFormatGUID desiredImagePixelFormat = GUID_WICPixelFormatDontCare;

    switch (dstTensorInfo.channelOrder)
    {
        case ImageTensorChannelOrder::RGB: desiredImagePixelFormat = GUID_WICPixelFormat24bppRGB; break;
        case ImageTensorChannelOrder::RGBA: desiredImagePixelFormat = GUID_WICPixelFormat32bppRGBA; break;
        case ImageTensorChannelOrder::BGR: desiredImagePixelFormat = GUID_WICPixelFormat24bppBGR; break;
        case ImageTensorChannelOrder::BGRA: desiredImagePixelFormat = GUID_WICPixelFormat32bppBGRA; break;
        case ImageTensorChannelOrder::Grayscale: desiredImagePixelFormat = GUID_WICPixelFormat8bppGray; break;
        default: throw std::invalid_argument("Unsupported channel order");
    }

    ComPtr<IWICImagingFactory> wicFactory;
    THROW_IF_FAILED(CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&wicFactory)
    ));

    ComPtr<IWICBitmapDecoder> bitmapDecoder;
    THROW_IF_FAILED(wicFactory->CreateDecoderFromFilename(
        srcPath.wstring().data(),
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnLoad,
        &bitmapDecoder
    ));

    ComPtr<IWICBitmapFrameDecode> bitmapFrame;
    THROW_IF_FAILED(bitmapDecoder->GetFrame(0, &bitmapFrame));

    ComPtr<IWICBitmapSource> bitmapSource = bitmapFrame;

    UINT imageWidth, imageHeight;
    THROW_IF_FAILED(bitmapFrame->GetSize(&imageWidth, &imageHeight));

    WICPixelFormatGUID imagePixelFormat;
    THROW_IF_FAILED(bitmapFrame->GetPixelFormat(&imagePixelFormat));

    if (imagePixelFormat != desiredImagePixelFormat)
    {
        Microsoft::WRL::ComPtr<IWICFormatConverter> formatConverter;
        THROW_IF_FAILED(wicFactory->CreateFormatConverter(&formatConverter));

        THROW_IF_FAILED(formatConverter->Initialize(
            bitmapFrame.Get(),
            desiredImagePixelFormat,
            WICBitmapDitherTypeNone,
            nullptr,
            0.0f,
            WICBitmapPaletteTypeCustom
        ));

        bitmapSource = formatConverter;
    }

    if (imageWidth != dstTensorInfo.width || imageHeight != dstTensorInfo.height)
    {
        // Scale image to match the tensor dimensions.
        ComPtr<IWICBitmapScaler> scaler;
        THROW_IF_FAILED(wicFactory->CreateBitmapScaler(&scaler));
        THROW_IF_FAILED(scaler->Initialize(
            bitmapSource.Get(),
            dstTensorInfo.width,
            dstTensorInfo.height,
            WICBitmapInterpolationModeHighQualityCubic
        ));

        bitmapSource = scaler;
    }

    // WIC CopyPixels writes to an interleaved (HWC) buffer with 8 bits (uint8) per channel element.
    std::vector<std::byte> pixelBuffer(dstTensorInfo.height * dstTensorInfo.width * dstTensorInfo.channels);
    WICRect pixelBufferRect = { 0, 0, static_cast<INT>(dstTensorInfo.width), static_cast<INT>(dstTensorInfo.height) };
    const uint32_t pixelBufferStride = dstTensorInfo.width * dstTensorInfo.channels * sizeof(uint8_t);

    THROW_IF_FAILED(bitmapSource->CopyPixels(
        &pixelBufferRect,
        pixelBufferStride,
        static_cast<uint32_t>(pixelBuffer.size()),
        reinterpret_cast<BYTE*>(pixelBuffer.data())
    ));

    switch (dstTensorInfo.dataType)
    {
        case DML_TENSOR_DATA_TYPE_FLOAT32: return ConvertPixelsToTensor<float>(pixelBuffer, dstTensorInfo);
        case DML_TENSOR_DATA_TYPE_FLOAT16: return ConvertPixelsToTensor<half_float::half>(pixelBuffer, dstTensorInfo);

        default: throw std::invalid_argument("Unsupported data type");
    }
}

// Converts a tensor to a pixel buffer (HWC layout with uint8 components).
template <typename T>
std::vector<std::byte> ConvertTensorToPixels(std::span<const std::byte> srcRaw, const ImageTensorInfo& tensorInfo)
{
    std::span<const T> src(reinterpret_cast<const T*>(srcRaw.data()), srcRaw.size_bytes() / sizeof(T));

    std::vector<std::byte> dst(tensorInfo.height * tensorInfo.width * tensorInfo.channels);

    for (size_t pixelIndex = 0; pixelIndex < tensorInfo.height * tensorInfo.width; pixelIndex++)
    {
        for (size_t channelIndex = 0; channelIndex < tensorInfo.channels; channelIndex++)
        {
            // dst is always HWC layout.
            size_t dstOffset = pixelIndex * tensorInfo.channels + channelIndex;

            // src is either HWC or CHW layout.
            size_t srcOffset = tensorInfo.layout == ImageTensorLayout::NHWC ?
                dstOffset :
                pixelIndex + channelIndex * tensorInfo.height * tensorInfo.width;

            dst[dstOffset] = static_cast<std::byte>(std::max(0.0f, std::min(1.0f, (float)src[srcOffset])) * 255.0f);
        }
    }

    return dst;
}

void WriteTensorToImage(const std::filesystem::path& dstPath, std::span<const std::byte> srcData, const ImageTensorInfo& srcTensorInfo)
{
    WICPixelFormatGUID desiredImagePixelFormat = GUID_WICPixelFormatDontCare;

    switch (srcTensorInfo.channelOrder)
    {
        case ImageTensorChannelOrder::RGB: desiredImagePixelFormat = GUID_WICPixelFormat24bppRGB; break;
        case ImageTensorChannelOrder::RGBA: desiredImagePixelFormat = GUID_WICPixelFormat32bppRGBA; break;
        case ImageTensorChannelOrder::BGR: desiredImagePixelFormat = GUID_WICPixelFormat24bppBGR; break;
        case ImageTensorChannelOrder::BGRA: desiredImagePixelFormat = GUID_WICPixelFormat32bppBGRA; break;
        case ImageTensorChannelOrder::Grayscale: desiredImagePixelFormat = GUID_WICPixelFormat8bppGray; break;
        default: throw std::invalid_argument("Unsupported channel order");
    }

    std::vector<std::byte> pixelBuffer;
    switch (srcTensorInfo.dataType)
    {
        case DML_TENSOR_DATA_TYPE_FLOAT32: pixelBuffer = ConvertTensorToPixels<float>(srcData, srcTensorInfo); break;
        case DML_TENSOR_DATA_TYPE_FLOAT16: pixelBuffer = ConvertTensorToPixels<half_float::half>(srcData, srcTensorInfo); break;
        default: throw std::invalid_argument("Unsupported data type");
    }

    std::span<BYTE> byteSpan(reinterpret_cast<BYTE*>(pixelBuffer.data()), pixelBuffer.size());

    ComPtr<IWICImagingFactory> wicFactory;
    THROW_IF_FAILED(CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&wicFactory)
    ));

    ComPtr<IWICBitmap> bitmap;
    THROW_IF_FAILED(wicFactory->CreateBitmapFromMemory(
        srcTensorInfo.width,
        srcTensorInfo.height,
        desiredImagePixelFormat,
        srcTensorInfo.width * srcTensorInfo.channels,
        static_cast<uint32_t>(byteSpan.size()),
        byteSpan.data(),
        &bitmap
    ));

    ComPtr<IWICStream> stream;
    THROW_IF_FAILED(wicFactory->CreateStream(&stream));
    THROW_IF_FAILED(stream->InitializeFromFilename(dstPath.wstring().data(), GENERIC_WRITE));

    std::string extension = dstPath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    GUID containerFormat;
    if (extension == ".jpg")
    {
        containerFormat = GUID_ContainerFormatJpeg;
    }
    else if (extension == ".png")
    {
        containerFormat = GUID_ContainerFormatPng;
    }
    else
    {
        throw std::invalid_argument("Unsupported image format");
    }

    ComPtr<IWICBitmapEncoder> encoder;
    THROW_IF_FAILED(wicFactory->CreateEncoder(containerFormat, nullptr, &encoder));
    THROW_IF_FAILED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache));

    ComPtr<IWICBitmapFrameEncode> frame;
    ComPtr<IPropertyBag2> propertyBag;
    THROW_IF_FAILED(encoder->CreateNewFrame(&frame, &propertyBag));
    THROW_IF_FAILED(frame->Initialize(propertyBag.Get()));
    THROW_IF_FAILED(frame->WriteSource(bitmap.Get(), nullptr));
    THROW_IF_FAILED(frame->Commit());
    THROW_IF_FAILED(encoder->Commit());
}

#endif
