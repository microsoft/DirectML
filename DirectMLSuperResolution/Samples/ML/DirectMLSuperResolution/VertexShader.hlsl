//--------------------------------------------------------------------------------------
// VertexShader.hlsl
//
// Simple vertex shader for rendering a textured quad
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//--------------------------------------------------------------------------------------

struct Vertex
{
    float4 position     : SV_Position;
    float2 texcoord     : TEXCOORD0;
};

struct Interpolants
{
    float4 position     : SV_Position;
    float2 texcoord     : TEXCOORD0;
};

Interpolants main( Vertex In )
{
    return In;
}