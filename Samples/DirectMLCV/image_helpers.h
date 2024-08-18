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
    float* floatBuffer = reinterpret_cast<float*>(buffer.data());
    for (size_t pixelIndex = 0; pixelIndex < targetHeight * width; pixelIndex++)
    {
        float r = static_cast<float>(pixelDataHWC8bpc[pixelIndex * channels + 0]) / 255.0f;
        float g = static_cast<float>(pixelDataHWC8bpc[pixelIndex * channels + 1]) / 255.0f;
        float b = static_cast<float>(pixelDataHWC8bpc[pixelIndex * channels + 2]) / 255.0f;

        buffer[pixelIndex + 0 * targetHeight * width] = r;
        buffer[pixelIndex + 1 * targetHeight * width] = g;
        buffer[pixelIndex + 2 * targetHeight * width] = b;
    }

    // return { pixelDataCHW32bpc, { 1, channels, targetHeight, width } };
}

// void SaveTensorDataToImageFilename(const ImageTensorData& tensorData, std::wstring_view filename)
// {
//     // Convert CHW tensor at 32 bits per channel to HWC tensor at 8 bits per channel
//     auto src = reinterpret_cast<const float*>(tensorData.buffer.data());
//     std::vector<BYTE> dst(tensorData.Pixels() * tensorData.Channels() * sizeof(std::byte));

//     for (size_t pixelIndex = 0; pixelIndex < tensorData.Pixels(); pixelIndex++)
//     {
//         float r = src[pixelIndex + 0 * tensorData.Pixels()];
//         float g = src[pixelIndex + 1 * tensorData.Pixels()];
//         float b = src[pixelIndex + 2 * tensorData.Pixels()];

//         dst[pixelIndex * tensorData.Channels() + 0] = static_cast<BYTE>(std::max(0.0f, std::min(1.0f, r)) * 255.0f);
//         dst[pixelIndex * tensorData.Channels() + 1] = static_cast<BYTE>(std::max(0.0f, std::min(1.0f, g)) * 255.0f);
//         dst[pixelIndex * tensorData.Channels() + 2] = static_cast<BYTE>(std::max(0.0f, std::min(1.0f, b)) * 255.0f);
//     }


//     ComPtr<IWICImagingFactory> wicFactory;
//     THROW_IF_FAILED(CoCreateInstance(
//         CLSID_WICImagingFactory,
//         nullptr,
//         CLSCTX_INPROC_SERVER,
//         IID_PPV_ARGS(&wicFactory)
//     ));

//     // Create a WIC bitmap
//     ComPtr<IWICBitmap> bitmap;
//     THROW_IF_FAILED(wicFactory->CreateBitmapFromMemory(
//         tensorData.Width(),
//         tensorData.Height(),
//         GUID_WICPixelFormat24bppRGB,
//         tensorData.Width() * tensorData.Channels(),
//         dst.size(),
//         dst.data(),
//         &bitmap
//     ));

//     ComPtr<IWICStream> stream;
//     THROW_IF_FAILED(wicFactory->CreateStream(&stream));

//     THROW_IF_FAILED(stream->InitializeFromFilename(filename.data(), GENERIC_WRITE));

//     ComPtr<IWICBitmapEncoder> encoder;
//     THROW_IF_FAILED(wicFactory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder));

//     THROW_IF_FAILED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache));

//     ComPtr<IWICBitmapFrameEncode> frame;
//     ComPtr<IPropertyBag2> propertyBag;

//     THROW_IF_FAILED(encoder->CreateNewFrame(&frame, &propertyBag));

//     THROW_IF_FAILED(frame->Initialize(propertyBag.Get()));

//     THROW_IF_FAILED(frame->WriteSource(bitmap.Get(), nullptr));

//     THROW_IF_FAILED(frame->Commit());

//     THROW_IF_FAILED(encoder->Commit());
// }