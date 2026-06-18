// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
};

typedef metal::float3 type_1[1];
struct Output {
    uint value;
};

kernel void main_(
  device type_1 const& values [[user(fake0)]]
, device Output& output [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    output.value = 1 + (_buffer_sizes.size0 - 0 - 16) / 16;
    return;
}
