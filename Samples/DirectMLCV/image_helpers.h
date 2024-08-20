#include <wincodec.h>
#include <wrl/client.h>
#include <wil/result.h>
#include <span>
#include "half.hpp"

enum class ChannelOrder
{
    RGB,
    BGR,
};

// Converts a pixel buffer to an NCHW tensor (batch size 1).
// Source: buffer of RGB pixels (HWC) using uint8 components.
// Target: buffer of RGB planes (CHW) using float32/float16 components.
template <typename T> 
void CopyPixelsToTensor(
    std::span<const std::byte> src, 
    std::span<std::byte> dst,
    uint32_t height,
    uint32_t width,
    uint32_t channels)
{
    std::span<T> dstT(reinterpret_cast<T*>(dst.data()), dst.size_bytes() / sizeof(T));

    for (size_t pixelIndex = 0; pixelIndex < height * width; pixelIndex++)
    {
        float r = static_cast<float>(src[pixelIndex * channels + 0]) / 255.0f;
        float g = static_cast<float>(src[pixelIndex * channels + 1]) / 255.0f;
        float b = static_cast<float>(src[pixelIndex * channels + 2]) / 255.0f;

        dstT[pixelIndex + 0 * height * width] = r;
        dstT[pixelIndex + 1 * height * width] = g;
        dstT[pixelIndex + 2 * height * width] = b;
    }
}

// Converts an NCHW tensor buffer (batch size 1) to a pixel buffer.
// Source: buffer of RGB planes (CHW) using float32/float16 components.
// Target: buffer of RGB pixels (HWC) using uint8 components.
template <typename T>
void CopyTensorToPixels(
    std::span<const std::byte> src, 
    std::span<BYTE> dst,
    uint32_t height,
    uint32_t width,
    uint32_t channels)
{
    std::span<const T> srcT(reinterpret_cast<const T*>(src.data()), src.size_bytes() / sizeof(T));

    for (size_t pixelIndex = 0; pixelIndex < height * width; pixelIndex++)
    {
        BYTE r = static_cast<BYTE>(std::max(0.0f, std::min(1.0f, (float)srcT[pixelIndex + 0 * height * width])) * 255.0f);
        BYTE g = static_cast<BYTE>(std::max(0.0f, std::min(1.0f, (float)srcT[pixelIndex + 1 * height * width])) * 255.0f);
        BYTE b = static_cast<BYTE>(std::max(0.0f, std::min(1.0f, (float)srcT[pixelIndex + 2 * height * width])) * 255.0f);

        dst[pixelIndex * channels + 0] = r;
        dst[pixelIndex * channels + 1] = g;
        dst[pixelIndex * channels + 2] = b;
    }
}

void FillNCHWBufferFromImageFilename(
    std::wstring_view filename,
    std::span<std::byte> tensorBuffer,
    uint32_t bufferHeight,
    uint32_t bufferWidth,
    ONNXTensorElementDataType bufferDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ChannelOrder bufferChannelOrder = ChannelOrder::RGB)
{
    using Microsoft::WRL::ComPtr;

    uint32_t bufferChannels = 0;
    WICPixelFormatGUID desiredImagePixelFormat = GUID_WICPixelFormatDontCare;
    switch (bufferChannelOrder)
    {
        case ChannelOrder::RGB:
            bufferChannels = 3;
            desiredImagePixelFormat = GUID_WICPixelFormat24bppRGB;
            break;

        case ChannelOrder::BGR:
            bufferChannels = 3;
            desiredImagePixelFormat = GUID_WICPixelFormat24bppBGR;
            break;

        default:
            throw std::invalid_argument("Unsupported channel order");
    }

    uint32_t expectedBufferSizeInBytes = bufferChannels * bufferHeight * bufferWidth;
    switch (bufferDataType)
    {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            expectedBufferSizeInBytes *= sizeof(float);
            break;

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            expectedBufferSizeInBytes *= sizeof(uint16_t);
            break;

        default:
            throw std::invalid_argument("Unsupported data type");
    }

    if (tensorBuffer.size_bytes() < expectedBufferSizeInBytes)
    {
        throw std::invalid_argument("Provided buffer is too small");
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
        filename.data(),
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

    if (imageWidth != bufferWidth || imageHeight != bufferHeight)
    {
        if (bufferWidth == bufferHeight)
        {
            // Most ML models take a square input. In this case, crop to the 
            // top-left square of the image to avoid stretching.
            INT minSide = static_cast<INT>(std::min(imageWidth, imageHeight));
            WICRect cropRect = 
            { 
                .X = 0, 
                .Y = 0, 
                .Width = minSide, 
                .Height = minSide 
            };
            
            ComPtr<IWICBitmapClipper> clipper;
            THROW_IF_FAILED(wicFactory->CreateBitmapClipper(&clipper));
            THROW_IF_FAILED(clipper->Initialize(bitmapSource.Get(), &cropRect));
            bitmapSource = clipper;
        }

        // Scale image to match the buffer size.
        ComPtr<IWICBitmapScaler> scaler;
        THROW_IF_FAILED(wicFactory->CreateBitmapScaler(&scaler));
        THROW_IF_FAILED(scaler->Initialize(
            bitmapSource.Get(),
            bufferWidth,
            bufferHeight,
            WICBitmapInterpolationModeHighQualityCubic
        ));

        bitmapSource = scaler;
    }

    // WIC doesn't support copying into "CHW" (planar) buffers except when
    // using Y'CbCr pixel formats. Instead, we copy into an interleaved
    // "HWC" buffer and then convert to "CHW" while simultaneously casting
    // data elements from uint8 to float.

    // Copy to HWC buffer with 8 bits (uint8) per channel element
    std::vector<std::byte> pixelBuffer(bufferHeight * bufferWidth * bufferChannels);
    WICRect pixelBufferRect = { 0, 0, static_cast<INT>(bufferWidth), static_cast<INT>(bufferHeight) };
    const uint32_t pixelBufferStride = bufferWidth * bufferChannels * sizeof(uint8_t);

    THROW_IF_FAILED(bitmapSource->CopyPixels(
        &pixelBufferRect, 
        pixelBufferStride, 
        pixelBuffer.size(), 
        reinterpret_cast<BYTE*>(pixelBuffer.data())
    ));

    switch (bufferDataType)
    {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            CopyPixelsToTensor<float>(pixelBuffer, tensorBuffer, bufferHeight, bufferWidth, bufferChannels);
            break;

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            CopyPixelsToTensor<half_float::half>(pixelBuffer, tensorBuffer, bufferHeight, bufferWidth, bufferChannels);
            break;

        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

void SaveNCHWBufferToImageFilename(
    std::wstring_view filename,
    std::span<const std::byte> tensorBuffer,
    uint32_t bufferHeight,
    uint32_t bufferWidth,
    ONNXTensorElementDataType bufferDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ChannelOrder bufferChannelOrder = ChannelOrder::RGB)
{
    using Microsoft::WRL::ComPtr;

    uint32_t bufferChannels = 0;
    WICPixelFormatGUID desiredImagePixelFormat = GUID_WICPixelFormatDontCare;
    switch (bufferChannelOrder)
    {
        case ChannelOrder::RGB:
            bufferChannels = 3;
            desiredImagePixelFormat = GUID_WICPixelFormat24bppRGB;
            break;

        case ChannelOrder::BGR:
            bufferChannels = 3;
            desiredImagePixelFormat = GUID_WICPixelFormat24bppBGR;
            break;

        default:
            throw std::invalid_argument("Unsupported channel order");
    }

    uint32_t outputBufferSizeInBytes = bufferChannels * bufferHeight * bufferWidth;
    switch (bufferDataType)
    {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            outputBufferSizeInBytes *= sizeof(float);
            break;

        default:
            throw std::invalid_argument("Unsupported data type");
    }

    std::vector<BYTE> pixelBuffer(outputBufferSizeInBytes);

    switch (bufferDataType)
    {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            CopyTensorToPixels<float>(tensorBuffer, pixelBuffer, bufferHeight, bufferWidth, bufferChannels);
            break;

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            CopyTensorToPixels<half_float::half>(tensorBuffer, pixelBuffer, bufferHeight, bufferWidth, bufferChannels);
            break;

        default:
            throw std::invalid_argument("Unsupported data type");
    }

    ComPtr<IWICImagingFactory> wicFactory;
    THROW_IF_FAILED(CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&wicFactory)
    ));

    ComPtr<IWICBitmap> bitmap;
    THROW_IF_FAILED(wicFactory->CreateBitmapFromMemory(
        bufferWidth,
        bufferHeight,
        desiredImagePixelFormat,
        bufferWidth * bufferChannels,
        pixelBuffer.size(),
        pixelBuffer.data(),
        &bitmap
    ));

    ComPtr<IWICStream> stream;
    THROW_IF_FAILED(wicFactory->CreateStream(&stream));
    THROW_IF_FAILED(stream->InitializeFromFilename(filename.data(), GENERIC_WRITE));

    ComPtr<IWICBitmapEncoder> encoder;
    THROW_IF_FAILED(wicFactory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder));
    THROW_IF_FAILED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache));

    ComPtr<IWICBitmapFrameEncode> frame;
    ComPtr<IPropertyBag2> propertyBag;
    THROW_IF_FAILED(encoder->CreateNewFrame(&frame, &propertyBag));
    THROW_IF_FAILED(frame->Initialize(propertyBag.Get()));
    THROW_IF_FAILED(frame->WriteSource(bitmap.Get(), nullptr));
    THROW_IF_FAILED(frame->Commit());
    THROW_IF_FAILED(encoder->Commit());
}