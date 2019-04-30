//--------------------------------------------------------------------------------------
// File: ATGColors.h
//
// Definitions of the standard ATG color palette.
//
// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//-------------------------------------------------------------------------------------

#pragma once

#include <DirectXMath.h>


namespace ATG
{
    namespace Colors
    {
        XMGLOBALCONST DirectX::XMVECTORF32 Background = { { { 0.254901975f, 0.254901975f, 0.254901975f, 1.f } } }; // #414141
        XMGLOBALCONST DirectX::XMVECTORF32 Green      = { { { 0.062745102f, 0.486274511f, 0.062745102f, 1.f } } }; // #107c10
        XMGLOBALCONST DirectX::XMVECTORF32 Blue       = { { { 0.019607844f, 0.372549027f, 0.803921580f, 1.f } } }; // #055fcd
        XMGLOBALCONST DirectX::XMVECTORF32 Orange     = { { { 0.764705896f, 0.176470593f, 0.019607844f, 1.f } } }; // #c32d05
        XMGLOBALCONST DirectX::XMVECTORF32 DarkGrey   = { { { 0.200000003f, 0.200000003f, 0.200000003f, 1.f } } }; // #333333
        XMGLOBALCONST DirectX::XMVECTORF32 LightGrey  = { { { 0.478431374f, 0.478431374f, 0.478431374f, 1.f } } }; // #7a7a7a
        XMGLOBALCONST DirectX::XMVECTORF32 OffWhite   = { { { 0.635294139f, 0.635294139f, 0.635294139f, 1.f } } }; // #a2a2a2
        XMGLOBALCONST DirectX::XMVECTORF32 White      = { { { 0.980392158f, 0.980392158f, 0.980392158f, 1.f } } }; // #fafafa
    };

    namespace ColorsLinear
    {
        XMGLOBALCONST DirectX::XMVECTORF32 Background = { { { 0.052860655f, 0.052860655f, 0.052860655f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 Green      = { { { 0.005181516f, 0.201556236f, 0.005181516f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 Blue       = { { { 0.001517635f, 0.114435382f, 0.610495627f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 Orange     = { { { 0.545724571f, 0.026241219f, 0.001517635f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 DarkGrey   = { { { 0.033104762f, 0.033104762f, 0.033104762f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 LightGrey  = { { { 0.194617808f, 0.194617808f, 0.194617808f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 OffWhite   = { { { 0.361306787f, 0.361306787f, 0.361306787f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 White      = { { { 0.955973506f, 0.955973506f, 0.955973506f, 1.f } } };
    };

    namespace ColorsHDR
    {
        XMGLOBALCONST DirectX::XMVECTORF32 Background = { { { 0.052860655f * 2.f, 0.052860655f * 2.f, 0.052860655f * 2.f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 Green      = { { { 0.005181516f * 2.f, 0.201556236f * 2.f, 0.005181516f * 2.f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 Blue       = { { { 0.001517635f * 2.f, 0.114435382f * 2.f, 0.610495627f * 2.f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 Orange     = { { { 0.545724571f * 2.f, 0.026241219f * 2.f, 0.001517635f * 2.f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 DarkGrey   = { { { 0.033104762f * 2.f, 0.033104762f * 2.f, 0.033104762f * 2.f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 LightGrey  = { { { 0.194617808f * 2.f, 0.194617808f * 2.f, 0.194617808f * 2.f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 OffWhite   = { { { 0.361306787f * 2.f, 0.361306787f * 2.f, 0.361306787f * 2.f, 1.f } } };
        XMGLOBALCONST DirectX::XMVECTORF32 White      = { { { 0.955973506f * 2.f, 0.955973506f * 2.f, 0.955973506f * 2.f, 1.f } } };
    };
}