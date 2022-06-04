StructuredBuffer<float16_t> inputA;
StructuredBuffer<float16_t> inputB;
RWStructuredBuffer<float16_t> output;

cbuffer constants
{
    uint elementCount;
};

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x < elementCount)
    {
        float16_t a = inputA[dtid.x];
        float16_t b = inputB[dtid.x];
        float16_t sum = a + b;
        output[dtid.x] = sum;
    }
}