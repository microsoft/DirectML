//
// Copyright (c) Microsoft Corporation.  All rights reserved.
// Licensed under the MIT License.
//
// This shader converts a FLOAT Tensor into a DX texture (BGRA/BGRX) with channel order RGB/BGR.
// This could also be done with a compute shader.
//

Buffer<float> input : register(t0); // SRV
cbuffer cbCS : register(b0)
{
    uint g_height;
    uint g_width;
    bool g_nhwc;
};

float4 VsTensorToSurf(float4 position : POSITION) : SV_POSITION
{
    return position;
}

float4 PsTensorRGB8ToSurf(float4 pos : SV_POSITION) : SV_TARGET
{
    float4 color;
    uint index = ((uint)pos.y)*g_width + (uint)pos.x;

    if (g_nhwc)
    {
        color.r = input[index * 3];
        color.g = input[index * 3 + 1];
        color.b = input[index * 3 + 2];
        color.a = 1.0f;
    }
    else
    {
        uint blockSize = g_height * g_width;

        color.r = input[index];
        color.g = input[index + blockSize];
        color.b = input[index + 2 * blockSize];
        color.a = 1.0f;
    }

    return color;
}

float4 PsTensorBGR8ToSurf(float4 pos : SV_POSITION) : SV_TARGET
{
    float4 color;
    uint index = ((uint)pos.y)*g_width + (uint)pos.x;

    if (g_nhwc)
    {
        color.b = input[index * 3];
        color.g = input[index * 3 + 1];
        color.r = input[index * 3 + 2];
        color.a = 1.0f;
    }
    else
    {
        uint blockSize = g_height * g_width;

        color.b = input[index];
        color.g = input[index + blockSize];
        color.r = input[index + 2 * blockSize];
        color.a = 1.0f;
    }

    return color;
}

float4 PsTensorGRAY8ToSurf(float4 pos : SV_POSITION) : SV_TARGET
{
    float4 color;
    uint yOffset = ((uint)pos.y)*g_width;

    color.b = input[((uint)pos.x + yOffset)];
    color.g = color.b;
    color.r = color.b;
    color.a = 1.0f;
    return color;
}