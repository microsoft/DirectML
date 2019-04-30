//--------------------------------------------------------------------------------------
// File: ControllerFont.h
//
// Class for compositing text with Xbox controller font button sprites
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//-------------------------------------------------------------------------------------

#pragma once

#include "SpriteBatch.h"
#include "SpriteFont.h"


namespace DX
{
    enum class ControllerFont : wchar_t
    {
        LeftThumb = L' ',
        DPad = L'!',
        RightThumb = L'\"',
        View = L'#',
        Nexus = L'$',
        Menu = L'%',
        XButton = L'&',
        AButton = L'\'',
        YButton = L'(',
        BButton = L')',
        RightShoulder = L'*',
        RightTrigger = L'+',
        LeftTrigger = L',',
        LeftShoulder = L'-',
    };

    inline void XM_CALLCONV DrawControllerString(_In_ DirectX::SpriteBatch* spriteBatch, _In_ DirectX::SpriteFont* textFont, _In_ DirectX::SpriteFont* butnFont,
        _In_z_ wchar_t const* text, DirectX::XMFLOAT2 const& position, DirectX::FXMVECTOR color = DirectX::Colors::White, float scale = 1)
    {
        using namespace DirectX;

        size_t textLen = wcslen(text);
        if (textLen >= 4096)
        {
            throw std::out_of_range("String is too long");
        }

        float buttonHeight = butnFont->GetLineSpacing();
        float buttonScale = (textFont->GetLineSpacing() * scale) / buttonHeight;
        float offsetY = buttonScale / 2.f;

        size_t j = 0;
        wchar_t strBuffer[4096] = {};

        bool buttonText = false;

        XMFLOAT2 outPos = position;

        for (size_t ch = 0; ch < textLen; ++ch)
        {
            if (buttonText)
            {
                strBuffer[j++] = text[ch];

                if (text[ch] == L']')
                {
                    wchar_t button[2] = {};

                    if (_wcsicmp(strBuffer, L"[A]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::AButton);
                    }
                    else if (_wcsicmp(strBuffer, L"[B]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::BButton);
                    }
                    else if (_wcsicmp(strBuffer, L"[X]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::XButton);
                    }
                    else if (_wcsicmp(strBuffer, L"[Y]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::YButton);
                    }
                    else if (_wcsicmp(strBuffer, L"[DPad]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::DPad);
                    }
                    else if (_wcsicmp(strBuffer, L"[View]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::View);
                    }
                    else if (_wcsicmp(strBuffer, L"[Menu]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::Menu);
                    }
                    else if (_wcsicmp(strBuffer, L"[Nexus]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::Nexus);
                    }
                    else if (_wcsicmp(strBuffer, L"[RThumb]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::RightThumb);
                    }
                    else if (_wcsicmp(strBuffer, L"[LThumb]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::LeftThumb);
                    }
                    else if (_wcsicmp(strBuffer, L"[RB]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::RightShoulder);
                    }
                    else if (_wcsicmp(strBuffer, L"[LB]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::LeftShoulder);
                    }
                    else if (_wcsicmp(strBuffer, L"[RT]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::RightTrigger);
                    }
                    else if (_wcsicmp(strBuffer, L"[LT]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::LeftTrigger);
                    }

                    if (*button)
                    {
                        float bsize = XMVectorGetX(butnFont->MeasureString(button));
                        float offsetX = (bsize * buttonScale / 2.f);

                        outPos.x += offsetX;
                        outPos.y -= offsetY;
                        butnFont->DrawString(spriteBatch, button, outPos, Colors::White, 0.f, XMFLOAT2(0.f, 0.f), XMFLOAT2(buttonScale, buttonScale));
                        outPos.x += bsize * buttonScale;
                        outPos.y += offsetY;
                    }

                    memset(strBuffer, 0, sizeof(strBuffer));
                    j = 0;

                    buttonText = false;
                }
            }
            else
            {
                switch (text[ch])
                {
                case '\r':
                    break;

                case '[':
                    if (*strBuffer)
                    {
                        textFont->DrawString(spriteBatch, strBuffer, outPos, color, 0.f, XMFLOAT2(0.f, 0.f), XMFLOAT2(scale, scale));
                        outPos.x += XMVectorGetX(textFont->MeasureString(strBuffer)) * scale;
                        memset(strBuffer, 0, sizeof(strBuffer));
                        j = 0;
                    }
                    buttonText = true;
                    *strBuffer = L'[';
                    ++j;
                    break;

                case '\n':
                    if (*strBuffer)
                    {
                        textFont->DrawString(spriteBatch, strBuffer, outPos, color, 0.f, XMFLOAT2(0.f, 0.f), XMFLOAT2(scale, scale));
                        memset(strBuffer, 0, sizeof(strBuffer));
                        j = 0;
                    }
                    outPos.x = position.x;
                    outPos.y += textFont->GetLineSpacing() * scale;
                    break;

                default:
                    strBuffer[j++] = text[ch];
                    break;
                }
            }
        }

        if (*strBuffer)
        {
            textFont->DrawString(spriteBatch, strBuffer, outPos, color, 0.f, XMFLOAT2(0.f, 0.f), XMFLOAT2(scale, scale));
        }
    }

    inline RECT XM_CALLCONV MeasureControllerDrawBounds(_In_ DirectX::SpriteFont* textFont, _In_ DirectX::SpriteFont* butnFont,
        _In_z_ wchar_t const* text, DirectX::XMFLOAT2 const& position, float scale = 1)
    {
        using namespace DirectX;

        size_t textLen = wcslen(text);
        if (textLen >= 4096)
        {
            throw std::out_of_range("String is too long");
        }

        float buttonHeight = butnFont->GetLineSpacing();
        float buttonScale = (textFont->GetLineSpacing() * scale) / buttonHeight;
        float offsetY = buttonScale / 2.f;

        size_t j = 0;
        wchar_t strBuffer[4096] = {};

        bool buttonText = false;

        XMFLOAT2 outPos = position;

        RECT result = { LONG_MAX, LONG_MAX, 0, 0 };
        for (size_t ch = 0; ch < textLen; ++ch)
        {
            if (buttonText)
            {
                strBuffer[j++] = text[ch];

                if (text[ch] == L']')
                {
                    wchar_t button[2] = {};

                    if (_wcsicmp(strBuffer, L"[A]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::AButton);
                    }
                    else if (_wcsicmp(strBuffer, L"[B]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::BButton);
                    }
                    else if (_wcsicmp(strBuffer, L"[X]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::XButton);
                    }
                    else if (_wcsicmp(strBuffer, L"[Y]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::YButton);
                    }
                    else if (_wcsicmp(strBuffer, L"[DPad]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::DPad);
                    }
                    else if (_wcsicmp(strBuffer, L"[View]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::View);
                    }
                    else if (_wcsicmp(strBuffer, L"[Menu]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::Menu);
                    }
                    else if (_wcsicmp(strBuffer, L"[Nexus]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::Nexus);
                    }
                    else if (_wcsicmp(strBuffer, L"[RThumb]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::RightThumb);
                    }
                    else if (_wcsicmp(strBuffer, L"[LThumb]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::LeftThumb);
                    }
                    else if (_wcsicmp(strBuffer, L"[RB]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::RightShoulder);
                    }
                    else if (_wcsicmp(strBuffer, L"[LB]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::LeftShoulder);
                    }
                    else if (_wcsicmp(strBuffer, L"[RT]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::RightTrigger);
                    }
                    else if (_wcsicmp(strBuffer, L"[LT]") == 0)
                    {
                        *button = static_cast<wchar_t>(ControllerFont::LeftTrigger);
                    }

                    if (*button)
                    {
                        float bsize = XMVectorGetX(butnFont->MeasureString(button));
                        float offsetX = (bsize * buttonScale / 2.f);

                        if (outPos.x < result.left)
                            result.left = long(outPos.x);

                        if (outPos.y < result.top)
                            result.top = long(outPos.y);

                        outPos.x += offsetX;
                        outPos.y -= offsetY;

                        if (outPos.x < result.left)
                            result.left = long(outPos.x);

                        if (outPos.y < result.top)
                            result.top = long(outPos.y);

                        outPos.x += bsize * buttonScale;
                        outPos.y += offsetY;

                        if (result.right < outPos.x)
                            result.right = long(outPos.x);

                        if (result.bottom < outPos.y)
                            result.bottom = long(outPos.y);
                    }

                    memset(strBuffer, 0, sizeof(strBuffer));
                    j = 0;

                    buttonText = false;
                }
            }
            else
            {
                switch (text[ch])
                {
                case '\r':
                    break;

                case '[':
                    if (*strBuffer)
                    {
                        if (outPos.x < result.left)
                            result.left = long(outPos.x);

                        if (outPos.y < result.top)
                            result.top = long(outPos.y);

                        outPos.x += XMVectorGetX(textFont->MeasureString(strBuffer)) * scale;

                        if (result.right < outPos.x)
                            result.right = long(outPos.x);

                        if (result.bottom < outPos.y)
                            result.bottom = long(outPos.y);

                        memset(strBuffer, 0, sizeof(strBuffer));
                        j = 0;
                    }
                    buttonText = true;
                    *strBuffer = L'[';
                    ++j;
                    break;

                case '\n':
                    if (*strBuffer)
                    {
                        if (outPos.x < result.left)
                            result.left = long(outPos.x);

                        if (outPos.y < result.top)
                            result.top = long(outPos.y);

                        outPos.x += XMVectorGetX(textFont->MeasureString(strBuffer)) * scale;

                        if (result.right < outPos.x)
                            result.right = long(outPos.x);

                        if (result.bottom < outPos.y)
                            result.bottom = long(outPos.y);

                        memset(strBuffer, 0, sizeof(strBuffer));
                        j = 0;
                    }
                    outPos.x = position.x;
                    outPos.y += textFont->GetLineSpacing() * scale;
                    break;

                default:
                    strBuffer[j++] = text[ch];
                    break;
                }
            }
        }

        if (*strBuffer)
        {
            if (outPos.x < result.left)
                result.left = long(outPos.x);

            if (outPos.y < result.top)
                result.top = long(outPos.y);

            outPos.x += XMVectorGetX(textFont->MeasureString(strBuffer)) * scale;

            if (result.right < outPos.x)
                result.right = long(outPos.x);

            if (result.bottom < outPos.y)
                result.bottom = long(outPos.y);
        }

        if (result.left == LONG_MAX)
        {
            result.left = 0;
            result.top = 0;
        }

        return result;
    }
}