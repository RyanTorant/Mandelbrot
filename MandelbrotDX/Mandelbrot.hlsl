RWTexture2D<unorm float4> OutTex : register(u0);
Texture2D<float4> ColorTable : register(t0);
SamplerState PointSampler;

#if USE_DOUBLES
cbuffer CSConstantBuffer : register(b0)
{
    double2 PosOffset;

    double Zoom;
    uint Iterations;
    uint padding;
};
#else
cbuffer CSConstantBuffer : register(b0)
{
    float2 PosOffset;
    float Zoom;
    uint Iterations;
};
#endif

#define GROUP_DIM 16
groupshared float4 GroupBuffer[GROUP_DIM][GROUP_DIM];

[numthreads(GROUP_DIM, GROUP_DIM, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID)
{
#if USE_DOUBLES
    double2 mask = DTid.xy & 1;
#else
    float2 mask = DTid.xy & 1;
#endif
   
    // Using the groupshared data to do x2 SSAA
    DTid.xy /= 2;

#if USE_DOUBLES
    double2 c = double2(DTid.xy + 0.5*mask) / double2(1024, 1024);
#else
    float2 c = float2(DTid.xy + 0.5 * mask) / float2(1024, 1024);
#endif

    c = 2.0f * (c * 2 - 1);

    c *= Zoom;
    c += PosOffset;

    float iters = -1;

#if USE_DOUBLES
    double2 z = 0;
#else
    float2 z = 0;
#endif

    for (float i = 0; i < Iterations; i++)
    {
        // iterate
#if USE_DOUBLES
        z = double2(z.x * z.x - z.y * z.y, 2 * z.x * z.y) + c;
#else
        z = float2(z.x * z.x - z.y * z.y, 2 * z.x * z.y) + c;
#endif
        if (dot(z, z) > 4)
        {
            iters = i;
            break;
        }
           
    }

    // Write to groupshared
    if (iters == -1)
        GroupBuffer[GTid.x][GTid.y] = float4(0, 0, 0, 1);
    else
        GroupBuffer[GTid.x][GTid.y] = ColorTable.SampleLevel(PointSampler, float2(iters * (1.0 / 32.0f), 0.0f), 0.0f);

    // Sync
    GroupMemoryBarrierWithGroupSync();

    // Now accumulate and write. Only one thread every 4 writes
    if (mask.x == 0 && mask.y == 0)
        OutTex[DTid.xy] = 0.25f * (GroupBuffer[GTid.x][GTid.y] + GroupBuffer[GTid.x + 1][GTid.y] + GroupBuffer[GTid.x][GTid.y + 1] + GroupBuffer[GTid.x + 1][GTid.y + 1]);

}