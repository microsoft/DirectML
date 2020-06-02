//--------------------------------------------------------------------------------------
// File: SpriteFont.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#include "pch.h"

#include <algorithm>
#include <vector>

#include "SpriteFont.h"
#include "DirectXHelpers.h"
#include "BinaryReader.h"
#include "LoaderHelpers.h"
#include "ResourceUploadBatch.h"
#include "DescriptorHeap.h"

using namespace DirectX;
using Microsoft::WRL::ComPtr;


// Internal SpriteFont implementation class.
class SpriteFont::Impl
{
public:
    Impl(_In_ ID3D12Device* device,
        ResourceUploadBatch& upload,
        _In_ BinaryReader* reader,
        D3D12_CPU_DESCRIPTOR_HANDLE cpuDesc,
        D3D12_GPU_DESCRIPTOR_HANDLE gpuDesc,
        bool forceSRGB);
    Impl(D3D12_GPU_DESCRIPTOR_HANDLE texture,
        XMUINT2 textureSize,
        _In_reads_(glyphCount) Glyph const* glyphs,
        size_t glyphCount,
        float lineSpacing);

    Glyph const* FindGlyph(wchar_t character) const;

    void SetDefaultCharacter(wchar_t character);

    template<typename TAction>
    void ForEachGlyph(_In_z_ wchar_t const* text, TAction action) const;

    void CreateTextureResource(_In_ ID3D12Device* device,
        ResourceUploadBatch& upload,
        uint32_t width, uint32_t height,
        DXGI_FORMAT format,
        uint32_t stride, uint32_t rows,
        _In_reads_(stride * rows) const uint8_t* data);

    const wchar_t* ConvertUTF8(_In_z_ const char *text);

    // Fields.
    ComPtr<ID3D12Resource> textureResource;
    D3D12_GPU_DESCRIPTOR_HANDLE texture;
    XMUINT2 textureSize;
    std::vector<Glyph> glyphs;
    Glyph const* defaultGlyph;
    float lineSpacing;

private:
    size_t utfBufferSize;
    std::unique_ptr<wchar_t[]> utfBuffer;
};


// Constants.
const XMFLOAT2 SpriteFont::Float2Zero(0, 0);

static const char spriteFontMagic[] = "DXTKfont";


// Comparison operators make our sorted glyph vector work with std::binary_search and lower_bound.
namespace DirectX
{
    static inline bool operator< (SpriteFont::Glyph const& left, SpriteFont::Glyph const& right)
    {
        return left.Character < right.Character;
    }

    static inline bool operator< (wchar_t left, SpriteFont::Glyph const& right)
    {
        return left < right.Character;
    }

    static inline bool operator< (SpriteFont::Glyph const& left, wchar_t right)
    {
        return left.Character < right;
    }
}


// Reads a SpriteFont from the binary format created by the MakeSpriteFont utility.
_Use_decl_annotations_
SpriteFont::Impl::Impl(
    ID3D12Device* device,
    ResourceUploadBatch& upload,
    BinaryReader* reader,
    D3D12_CPU_DESCRIPTOR_HANDLE cpuDesc,
    D3D12_GPU_DESCRIPTOR_HANDLE gpuDesc,
    bool forceSRGB) :
    texture{},
    defaultGlyph(nullptr),
    utfBufferSize(0)
{
    // Validate the header.
    for (char const* magic = spriteFontMagic; *magic; magic++)
    {
        if (reader->Read<uint8_t>() != *magic)
        {
            DebugTrace("ERROR: SpriteFont provided with an invalid .spritefont file\n");
            throw std::exception("Not a MakeSpriteFont output binary");
        }
    }

    // Read the glyph data.
    auto glyphCount = reader->Read<uint32_t>();
    auto glyphData = reader->ReadArray<Glyph>(glyphCount);

    glyphs.assign(glyphData, glyphData + glyphCount);

    // Read font properties.
    lineSpacing = reader->Read<float>();

    SetDefaultCharacter(static_cast<wchar_t>(reader->Read<uint32_t>()));

    // Read the texture data.
    auto textureWidth = reader->Read<uint32_t>();
    auto textureHeight = reader->Read<uint32_t>();
    auto textureFormat = reader->Read<DXGI_FORMAT>();
    auto textureStride = reader->Read<uint32_t>();
    auto textureRows = reader->Read<uint32_t>();
    auto textureData = reader->ReadArray<uint8_t>(size_t(textureStride) * size_t(textureRows));

    if (forceSRGB)
    {
        textureFormat = LoaderHelpers::MakeSRGB(textureFormat);
    }

    // Create the D3D texture object.
    CreateTextureResource(
        device, upload,
        textureWidth, textureHeight,
        textureFormat,
        textureStride, textureRows,
        textureData);

    // Create the shader resource view
    CreateShaderResourceView(
        device, textureResource.Get(),
        cpuDesc, false);

    // Save off the GPU descriptor pointer and size.
    texture = gpuDesc;
    textureSize = XMUINT2(textureWidth, textureHeight);
}


// Constructs a SpriteFont from arbitrary user specified glyph data.
_Use_decl_annotations_
SpriteFont::Impl::Impl(D3D12_GPU_DESCRIPTOR_HANDLE itexture, XMUINT2 itextureSize, Glyph const* iglyphs, size_t glyphCount, float ilineSpacing)
    : texture(itexture),
    textureSize(itextureSize),
    glyphs(iglyphs, iglyphs + glyphCount),
    defaultGlyph(nullptr),
    lineSpacing(ilineSpacing),
    utfBufferSize(0)
{
    if (!std::is_sorted(iglyphs, iglyphs + glyphCount))
    {
        throw std::exception("Glyphs must be in ascending codepoint order");
    }
}


// Looks up the requested glyph, falling back to the default character if it is not in the font.
SpriteFont::Glyph const* SpriteFont::Impl::FindGlyph(wchar_t character) const
{
    auto glyph = std::lower_bound(glyphs.begin(), glyphs.end(), character);

    if (glyph != glyphs.end() && glyph->Character == character)
    {
        return &*glyph;
    }

    if (defaultGlyph)
    {
        return defaultGlyph;
    }

    DebugTrace("ERROR: SpriteFont encountered a character not in the font (%u, %C), and no default glyph was provided\n", character, character);
    throw std::exception("Character not in font");
}


// Sets the missing-character fallback glyph.
void SpriteFont::Impl::SetDefaultCharacter(wchar_t character)
{
    defaultGlyph = nullptr;

    if (character)
    {
        defaultGlyph = FindGlyph(character);
    }
}


// The core glyph layout algorithm, shared between DrawString and MeasureString.
template<typename TAction>
void SpriteFont::Impl::ForEachGlyph(_In_z_ wchar_t const* text, TAction action) const
{
    float x = 0;
    float y = 0;

    for (; *text; text++)
    {
        wchar_t character = *text;

        switch (character)
        {
            case '\r':
                // Skip carriage returns.
                continue;

            case '\n':
                // New line.
                x = 0;
                y += lineSpacing;
                break;

            default:
                // Output this character.
                auto glyph = FindGlyph(character);

                x += glyph->XOffset;

                if (x < 0)
                    x = 0;

                float advance = glyph->Subrect.right - glyph->Subrect.left + glyph->XAdvance;

                if (!iswspace(character)
                    || ((glyph->Subrect.right - glyph->Subrect.left) > 1)
                    || ((glyph->Subrect.bottom - glyph->Subrect.top) > 1))
                {
                    action(glyph, x, y, advance);
                }

                x += advance;
                break;
        }
    }
}


_Use_decl_annotations_
void SpriteFont::Impl::CreateTextureResource(
    ID3D12Device* device,
    ResourceUploadBatch& upload,
    uint32_t width, uint32_t height,
    DXGI_FORMAT format,
    uint32_t stride, uint32_t rows,
    const uint8_t* data)
{
    D3D12_RESOURCE_DESC desc = {};
    desc.Width = static_cast<UINT>(width);
    desc.Height = static_cast<UINT>(height);
    desc.MipLevels = 1;
    desc.DepthOrArraySize = 1;
    desc.Format = format;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

    CD3DX12_HEAP_PROPERTIES defaultHeapProperties(D3D12_HEAP_TYPE_DEFAULT);

    ThrowIfFailed(device->CreateCommittedResource(
        &defaultHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(textureResource.ReleaseAndGetAddressOf())));

    SetDebugObjectName(textureResource.Get(), L"SpriteFont:Texture");

    D3D12_SUBRESOURCE_DATA subres = {};
    subres.pData = data;
    subres.RowPitch = ptrdiff_t(stride);
    subres.SlicePitch = ptrdiff_t(stride) * ptrdiff_t(rows);

    upload.Upload(
        textureResource.Get(),
        0,
        &subres,
        1);

    upload.Transition(
        textureResource.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}


const wchar_t* SpriteFont::Impl::ConvertUTF8(_In_z_ const char *text)
{
    if (!utfBuffer)
    {
        utfBufferSize = 1024;
        utfBuffer.reset(new wchar_t[1024]);
    }

    int result = MultiByteToWideChar(CP_UTF8, 0, text, -1, utfBuffer.get(), static_cast<int>(utfBufferSize));
    if (!result && (GetLastError() == ERROR_INSUFFICIENT_BUFFER))
    {
        // Compute required buffer size
        result = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
        utfBufferSize = AlignUp(static_cast<size_t>(result), 1024);
        utfBuffer.reset(new wchar_t[utfBufferSize]);

        // Retry conversion
        result = MultiByteToWideChar(CP_UTF8, 0, text, -1, utfBuffer.get(), static_cast<int>(utfBufferSize));
    }

    if (!result)
    {
        DebugTrace("ERROR: MultiByteToWideChar failed with error %u.\n", GetLastError());
        throw std::exception("MultiByteToWideChar");
    }

    return utfBuffer.get();
}


// Construct from a binary file created by the MakeSpriteFont utility.
_Use_decl_annotations_
SpriteFont::SpriteFont(ID3D12Device* device, ResourceUploadBatch& upload, wchar_t const* fileName, D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorDest, D3D12_GPU_DESCRIPTOR_HANDLE gpuDescriptorDest, bool forceSRGB)
{
    BinaryReader reader(fileName);

    pImpl = std::make_unique<Impl>(device, upload, &reader, cpuDescriptorDest, gpuDescriptorDest, forceSRGB);
}


// Construct from a binary blob created by the MakeSpriteFont utility and already loaded into memory.
_Use_decl_annotations_
SpriteFont::SpriteFont(ID3D12Device* device, ResourceUploadBatch& upload, uint8_t const* dataBlob, size_t dataSize, D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorDest, D3D12_GPU_DESCRIPTOR_HANDLE gpuDescriptorDest, bool forceSRGB)
{
    BinaryReader reader(dataBlob, dataSize);

    pImpl = std::make_unique<Impl>(device, upload, &reader, cpuDescriptorDest, gpuDescriptorDest, forceSRGB);
}


// Construct from arbitrary user specified glyph data (for those not using the MakeSpriteFont utility).
_Use_decl_annotations_
SpriteFont::SpriteFont(D3D12_GPU_DESCRIPTOR_HANDLE texture, XMUINT2 textureSize, Glyph const* glyphs, size_t glyphCount, float lineSpacing)
    : pImpl(std::make_unique<Impl>(texture, textureSize, glyphs, glyphCount, lineSpacing))
{
}


// Move constructor.
SpriteFont::SpriteFont(SpriteFont&& moveFrom) noexcept
    : pImpl(std::move(moveFrom.pImpl))
{
}


// Move assignment.
SpriteFont& SpriteFont::operator= (SpriteFont&& moveFrom) noexcept
{
    pImpl = std::move(moveFrom.pImpl);
    return *this;
}


// Public destructor.
SpriteFont::~SpriteFont()
{
}


// Wide-character / UTF-16LE
void XM_CALLCONV SpriteFont::DrawString(_In_ SpriteBatch* spriteBatch, _In_z_ wchar_t const* text, XMFLOAT2 const& position, FXMVECTOR color, float rotation, XMFLOAT2 const& origin, float scale, SpriteEffects effects, float layerDepth) const
{
    DrawString(spriteBatch, text, XMLoadFloat2(&position), color, rotation, XMLoadFloat2(&origin), XMVectorReplicate(scale), effects, layerDepth);
}


void XM_CALLCONV SpriteFont::DrawString(_In_ SpriteBatch* spriteBatch, _In_z_ wchar_t const* text, XMFLOAT2 const& position, FXMVECTOR color, float rotation, XMFLOAT2 const& origin, XMFLOAT2 const& scale, SpriteEffects effects, float layerDepth) const
{
    DrawString(spriteBatch, text, XMLoadFloat2(&position), color, rotation, XMLoadFloat2(&origin), XMLoadFloat2(&scale), effects, layerDepth);
}


void XM_CALLCONV SpriteFont::DrawString(_In_ SpriteBatch* spriteBatch, _In_z_ wchar_t const* text, FXMVECTOR position, FXMVECTOR color, float rotation, FXMVECTOR origin, float scale, SpriteEffects effects, float layerDepth) const
{
    DrawString(spriteBatch, text, position, color, rotation, origin, XMVectorReplicate(scale), effects, layerDepth);
}


void XM_CALLCONV SpriteFont::DrawString(_In_ SpriteBatch* spriteBatch, _In_z_ wchar_t const* text, FXMVECTOR position, FXMVECTOR color, float rotation, FXMVECTOR origin, GXMVECTOR scale, SpriteEffects effects, float layerDepth) const
{
    static_assert(SpriteEffects_FlipHorizontally == 1 &&
                  SpriteEffects_FlipVertically == 2, "If you change these enum values, the following tables must be updated to match");

    // Lookup table indicates which way to move along each axis per SpriteEffects enum value.
    static XMVECTORF32 axisDirectionTable[4] =
    {
        { { { -1, -1, 0, 0 } } },
        { { {  1, -1, 0, 0 } } },
        { { { -1,  1, 0, 0 } } },
        { { {  1,  1, 0, 0 } } },
    };

    // Lookup table indicates which axes are mirrored for each SpriteEffects enum value.
    static XMVECTORF32 axisIsMirroredTable[4] =
    {
        { { { 0, 0, 0, 0 } } },
        { { { 1, 0, 0, 0 } } },
        { { { 0, 1, 0, 0 } } },
        { { { 1, 1, 0, 0 } } },
    };

    XMVECTOR baseOffset = origin;

    // If the text is mirrored, offset the start position accordingly.
    if (effects)
    {
        baseOffset = XMVectorNegativeMultiplySubtract(
            MeasureString(text),
            axisIsMirroredTable[effects & 3],
            baseOffset);
    }

    // Draw each character in turn.
    pImpl->ForEachGlyph(text, [&](Glyph const* glyph, float x, float y, float advance)
    {
        UNREFERENCED_PARAMETER(advance);

        XMVECTOR offset = XMVectorMultiplyAdd(XMVectorSet(x, y + glyph->YOffset, 0, 0), axisDirectionTable[effects & 3], baseOffset);

        if (effects)
        {
            // For mirrored characters, specify bottom and/or right instead of top left.
            XMVECTOR glyphRect = XMConvertVectorIntToFloat(XMLoadInt4(reinterpret_cast<uint32_t const*>(&glyph->Subrect)), 0);

            // xy = glyph width/height.
            glyphRect = XMVectorSubtract(XMVectorSwizzle<2, 3, 0, 1>(glyphRect), glyphRect);

            offset = XMVectorMultiplyAdd(glyphRect, axisIsMirroredTable[effects & 3], offset);
        }

        spriteBatch->Draw(pImpl->texture, pImpl->textureSize, position, &glyph->Subrect, color, rotation, offset, scale, effects, layerDepth);
    });
}


XMVECTOR XM_CALLCONV SpriteFont::MeasureString(_In_z_ wchar_t const* text) const
{
    XMVECTOR result = XMVectorZero();

    pImpl->ForEachGlyph(text, [&](Glyph const* glyph, float x, float y, float advance)
    {
        UNREFERENCED_PARAMETER(advance);

        auto w = static_cast<float>(glyph->Subrect.right - glyph->Subrect.left);
        auto h = static_cast<float>(glyph->Subrect.bottom - glyph->Subrect.top) + glyph->YOffset;

        h = std::max(h, pImpl->lineSpacing);

        result = XMVectorMax(result, XMVectorSet(x + w, y + h, 0, 0));
    });

    return result;
}


RECT SpriteFont::MeasureDrawBounds(_In_z_ wchar_t const* text, XMFLOAT2 const& position) const
{
    RECT result = { LONG_MAX, LONG_MAX, 0, 0 };

    pImpl->ForEachGlyph(text, [&](Glyph const* glyph, float x, float y, float advance)
    {
        auto w = static_cast<float>(glyph->Subrect.right - glyph->Subrect.left);
        auto h = static_cast<float>(glyph->Subrect.bottom - glyph->Subrect.top);

        float minX = position.x + x;
        float minY = position.y + y + glyph->YOffset;

        float maxX = std::max(minX + advance, minX + w);
        float maxY = minY + h;

        if (minX < result.left)
            result.left = long(minX);

        if (minY < result.top)
            result.top = long(minY);

        if (result.right < maxX)
            result.right = long(maxX);

        if (result.bottom < maxY)
            result.bottom = long(maxY);
    });

    if (result.left == LONG_MAX)
    {
        result.left = 0;
        result.top = 0;
    }

    return result;
}


RECT XM_CALLCONV SpriteFont::MeasureDrawBounds(_In_z_ wchar_t const* text, FXMVECTOR position) const
{
    XMFLOAT2 pos;
    XMStoreFloat2(&pos, position);

    return MeasureDrawBounds(text, pos);
}


// UTF-8
void XM_CALLCONV SpriteFont::DrawString(_In_ SpriteBatch* spriteBatch, _In_z_ char const* text, XMFLOAT2 const& position, FXMVECTOR color, float rotation, XMFLOAT2 const& origin, float scale, SpriteEffects effects, float layerDepth) const
{
    DrawString(spriteBatch, pImpl->ConvertUTF8(text), XMLoadFloat2(&position), color, rotation, XMLoadFloat2(&origin), XMVectorReplicate(scale), effects, layerDepth);
}


void XM_CALLCONV SpriteFont::DrawString(_In_ SpriteBatch* spriteBatch, _In_z_ char const* text, XMFLOAT2 const& position, FXMVECTOR color, float rotation, XMFLOAT2 const& origin, XMFLOAT2 const& scale, SpriteEffects effects, float layerDepth) const
{
    DrawString(spriteBatch, pImpl->ConvertUTF8(text), XMLoadFloat2(&position), color, rotation, XMLoadFloat2(&origin), XMLoadFloat2(&scale), effects, layerDepth);
}


void XM_CALLCONV SpriteFont::DrawString(_In_ SpriteBatch* spriteBatch, _In_z_ char const* text, FXMVECTOR position, FXMVECTOR color, float rotation, FXMVECTOR origin, float scale, SpriteEffects effects, float layerDepth) const
{
    DrawString(spriteBatch, pImpl->ConvertUTF8(text), position, color, rotation, origin, XMVectorReplicate(scale), effects, layerDepth);
}


void XM_CALLCONV SpriteFont::DrawString(_In_ SpriteBatch* spriteBatch, _In_z_ char const* text, FXMVECTOR position, FXMVECTOR color, float rotation, FXMVECTOR origin, GXMVECTOR scale, SpriteEffects effects, float layerDepth) const
{
    DrawString(spriteBatch, pImpl->ConvertUTF8(text), position, color, rotation, origin, scale, effects, layerDepth);
}


XMVECTOR XM_CALLCONV SpriteFont::MeasureString(_In_z_ char const* text) const
{
    return MeasureString(pImpl->ConvertUTF8(text));
}


RECT SpriteFont::MeasureDrawBounds(_In_z_ char const* text, XMFLOAT2 const& position) const
{
    return MeasureDrawBounds(pImpl->ConvertUTF8(text), position);
}


RECT XM_CALLCONV SpriteFont::MeasureDrawBounds(_In_z_ char const* text, FXMVECTOR position) const
{
    XMFLOAT2 pos;
    XMStoreFloat2(&pos, position);

    return MeasureDrawBounds(pImpl->ConvertUTF8(text), pos);
}


// Spacing properties
float SpriteFont::GetLineSpacing() const
{
    return pImpl->lineSpacing;
}


void SpriteFont::SetLineSpacing(float spacing)
{
    pImpl->lineSpacing = spacing;
}


// Font properties
wchar_t SpriteFont::GetDefaultCharacter() const
{
    return static_cast<wchar_t>(pImpl->defaultGlyph ? pImpl->defaultGlyph->Character : 0);
}


void SpriteFont::SetDefaultCharacter(wchar_t character)
{
    pImpl->SetDefaultCharacter(character);
}


bool SpriteFont::ContainsCharacter(wchar_t character) const
{
    return std::binary_search(pImpl->glyphs.begin(), pImpl->glyphs.end(), character);
}


// Custom layout/rendering
SpriteFont::Glyph const* SpriteFont::FindGlyph(wchar_t character) const
{
    return pImpl->FindGlyph(character);
}


D3D12_GPU_DESCRIPTOR_HANDLE SpriteFont::GetSpriteSheet() const
{
    return pImpl->texture;
}


XMUINT2 SpriteFont::GetSpriteSheetSize() const
{
    return pImpl->textureSize;
}
