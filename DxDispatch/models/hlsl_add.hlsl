// The macro T is defined by hlsl_add_fp(16|32).json
#ifndef T
#define T float
#endif

StructuredBuffer<T> inputA;
StructuredBuffer<T> inputB;
RWStructuredBuffer<T> output;

cbuffer constants
{
    uint elementCount;
    uint elementsPerThread;
    uint4 shape;
    uint4 aStrides;
    uint4 bStrides;
    uint4 outStrides;
    uint4 cumulativeSizes;
};

inline uint4 OffsetToCoords(uint offset)
{
    uint4 nchw;
    nchw[0] =  offset / cumulativeSizes[0];
    nchw[1] = (offset / cumulativeSizes[1]) % shape[1];
    nchw[2] = (offset / cumulativeSizes[2]) % shape[2];
    nchw[3] =  offset % shape[3];
    return nchw;
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    for (uint i = 0; i < elementsPerThread; i++)
    {
        uint index = dtid.x + i;

        uint4 coords = OffsetToCoords(index);
        uint aOffset = dot(coords, aStrides);
        uint bOffset = dot(coords, bStrides);
        uint outOffset = dot(coords, outStrides);

        if (index < elementCount)
        {
            T a = inputA[aOffset];
            T b = inputB[bOffset];
            T sum = a + b;
            output[outOffset] = sum;
        }
    }
}
