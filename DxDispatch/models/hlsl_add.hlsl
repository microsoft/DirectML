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
};

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x < elementCount)
    {
        T a = inputA[dtid.x];
        T b = inputB[dtid.x];
        T sum = a + b;
        output[dtid.x] = sum;
    }
}
