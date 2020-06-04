// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkID=615561

#include "Structures.fxh"
#include "RootSig.fxh"

SamplerState Sampler       : register(s0);
Texture2D<float4> SrcMip   : register(t0);
RWTexture2D<float4> OutMip : register(u0);

cbuffer MipConstants : register(b0)
{
    float2 InvOutTexelSize; // texel size for OutMip (NOT SrcMip)
    uint SrcMipIndex;
}

float4 Mip(uint2 coord)
{
    float2 uv = (coord.xy + 0.5) * InvOutTexelSize;
    return SrcMip.SampleLevel(Sampler, uv, SrcMipIndex);
}

[RootSignature(GenerateMipsRS)]
[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    OutMip[DTid.xy] = Mip(DTid.xy);
}
