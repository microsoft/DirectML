// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

Texture2D<float4> inputImage : register(t0);
RWBuffer<float>   opTensor   : register(u0);

cbuffer ConstantBufferCS
{
    uint Height;
    uint Width;
    bool Nhwc;
};


[numthreads(32, 16, 1)]
void imageToTensor(uint3 blockID : SV_GroupID, uint3 threadID : SV_GroupThreadID)
{
    uint x = blockID.x * 32 + threadID.x;
    uint y = blockID.y * 16 + threadID.y;

    if (x < Width && y < Height)
    {
        uint index = Width * y + x;

        float3 val = inputImage[uint2(x, y)].xyz;

        if (Nhwc)
        {
            opTensor[index * 3] = val.x;
            opTensor[index * 3 + 1] = val.y;
            opTensor[index * 3 + 2] = val.z;
        }
        else
        {
            uint planeSize = Height * Width;

            // RGB plane order since model was trained on this
            opTensor[index] = val.x;
            opTensor[index + planeSize] = val.y;
            opTensor[index + planeSize * 2] = val.z;
        }		
    }
}
