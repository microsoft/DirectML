#include <wincodec.h>
#include <wrl/client.h>
#include <wil/result.h>
#include <span>

enum class ChannelOrder
{
    RGB,
    BGR,
};

enum class DataType
{
    Float32,
};

void FillNCHWBufferFromImageFilename(
    std::wstring_view filename,
    std::span<std::byte> buffer,
    uint32_t bufferHeight,
    uint32_t bufferWidth,
    DataType bufferDataType = DataType::Float32,
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
        case DataType::Float32:
            expectedBufferSizeInBytes *= sizeof(float);
            break;

        default:
            throw std::invalid_argument("Unsupported data type");
    }

    if (buffer.size_bytes() != expectedBufferSizeInBytes)
    {
        throw std::invalid_argument("Buffer size does not match expected size");
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
    std::vector<std::byte> bufferHWC_UInt8(bufferHeight * bufferWidth * bufferChannels);
    {
        WICRect rect = { 0, 0, static_cast<INT>(bufferWidth), static_cast<INT>(bufferHeight) };
        const uint32_t stride = bufferWidth * bufferChannels * sizeof(uint8_t);

        THROW_IF_FAILED(bitmapSource->CopyPixels(
            &rect, 
            stride, 
            bufferHWC_UInt8.size(), 
            reinterpret_cast<BYTE*>(bufferHWC_UInt8.data())
        ));
    }

    // Copy to CHW buffer with 32 bits (float) per channel element
    std::span<float> bufferCHW_Float(reinterpret_cast<float*>(buffer.data()), buffer.size_bytes() / sizeof(float));
    for (size_t pixelIndex = 0; pixelIndex < bufferHeight * bufferWidth; pixelIndex++)
    {
        float r = static_cast<float>(bufferHWC_UInt8[pixelIndex * bufferChannels + 0]) / 255.0f;
        float g = static_cast<float>(bufferHWC_UInt8[pixelIndex * bufferChannels + 1]) / 255.0f;
        float b = static_cast<float>(bufferHWC_UInt8[pixelIndex * bufferChannels + 2]) / 255.0f;

        bufferCHW_Float[pixelIndex + 0 * bufferHeight * bufferWidth] = r;
        bufferCHW_Float[pixelIndex + 1 * bufferHeight * bufferWidth] = g;
        bufferCHW_Float[pixelIndex + 2 * bufferHeight * bufferWidth] = b;
    }
}

void SaveNCHWBufferToImageFilename(
    std::wstring_view filename,
    std::span<const std::byte> buffer,
    uint32_t bufferHeight,
    uint32_t bufferWidth,
    DataType bufferDataType = DataType::Float32,
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
        case DataType::Float32:
            outputBufferSizeInBytes *= sizeof(float);
            break;

        default:
            throw std::invalid_argument("Unsupported data type");
    }

    std::vector<BYTE> bufferHWC_Uint8(outputBufferSizeInBytes);

    std::span<const float> bufferCHW_Float(reinterpret_cast<const float*>(buffer.data()), buffer.size_bytes() / sizeof(float));
    for (size_t pixelIndex = 0; pixelIndex < bufferHeight * bufferWidth; pixelIndex++)
    {
        BYTE r = static_cast<BYTE>(std::max(0.0f, std::min(1.0f, bufferCHW_Float[pixelIndex + 0 * bufferHeight * bufferWidth])) * 255.0f);
        BYTE g = static_cast<BYTE>(std::max(0.0f, std::min(1.0f, bufferCHW_Float[pixelIndex + 1 * bufferHeight * bufferWidth])) * 255.0f);
        BYTE b = static_cast<BYTE>(std::max(0.0f, std::min(1.0f, bufferCHW_Float[pixelIndex + 2 * bufferHeight * bufferWidth])) * 255.0f);

        bufferHWC_Uint8[pixelIndex * bufferChannels + 0] = r;
        bufferHWC_Uint8[pixelIndex * bufferChannels + 1] = g;
        bufferHWC_Uint8[pixelIndex * bufferChannels + 2] = b;
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
        bufferHWC_Uint8.size(),
        bufferHWC_Uint8.data(),
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