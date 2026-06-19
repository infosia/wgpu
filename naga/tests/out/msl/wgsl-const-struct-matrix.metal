// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct MatrixHolder {
    float scale;
    char _pad1[4];
    metal::float3x2 transform;
    metal::float2 bias;
};
struct type_3 {
    metal::float3x2 inner[2];
};

kernel void main_(
) {
    float phony = ((MatrixHolder {2.0, {}, metal::float3x2(metal::float2(1.0, 2.0), metal::float2(3.0, 4.0), metal::float2(5.0, 6.0)), metal::float2(7.0, 8.0)}.scale + MatrixHolder {2.0, {}, metal::float3x2(metal::float2(1.0, 2.0), metal::float2(3.0, 4.0), metal::float2(5.0, 6.0)), metal::float2(7.0, 8.0)}.transform[2].y) + MatrixHolder {2.0, {}, metal::float3x2(metal::float2(1.0, 2.0), metal::float2(3.0, 4.0), metal::float2(5.0, 6.0)), metal::float2(7.0, 8.0)}.bias.x) + type_3 {{metal::float3x2(metal::float2(9.0, 10.0), metal::float2(11.0, 12.0), metal::float2(13.0, 14.0)), metal::float3x2(metal::float2(15.0, 16.0), metal::float2(17.0, 18.0), metal::float2(19.0, 20.0))}}.inner[1][0].y;
    return;
}
