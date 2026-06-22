// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


kernel void main_(
) {
    int phony = metal::int4(12, 0, 0, 0)[2];
    return;
}
