// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct S {
    uint a;
    char _pad1[12];
    metal::float3 b;
};
struct Nested {
    S s;
    metal::float3x3 m;
};
struct type_3 {
    S inner[2];
};
struct type_4 {
    metal::float3x3 inner[2];
};

kernel void main_(
  device S& out_s [[user(fake0)]]
, device metal::float3x3& out_m [[user(fake0)]]
, device type_3& out_as [[user(fake0)]]
, device type_4& out_am [[user(fake0)]]
, device Nested& out_n [[user(fake0)]]
) {
    S s = S {42u, {}, metal::float3(1.0, 2.0, 3.0)};
    metal::float3x3 m = metal::float3x3(metal::float3(1.0, 2.0, 3.0), metal::float3(4.0, 5.0, 6.0), metal::float3(7.0, 8.0, 9.0));
    {
        const S _value = s;
        out_s.a = _value.a;
        (*reinterpret_cast<device metal::packed_float3*>(&out_s.b)) = metal::packed_float3(_value.b);
    }
    {
        const metal::float3x3 _value = m;
        (*reinterpret_cast<device metal::packed_float3*>(&out_m[0])) = metal::packed_float3(_value[0]);
        (*reinterpret_cast<device metal::packed_float3*>(&out_m[1])) = metal::packed_float3(_value[1]);
        (*reinterpret_cast<device metal::packed_float3*>(&out_m[2])) = metal::packed_float3(_value[2]);
    }
    {
        const type_3 _value = type_3 {{s, s}};
        out_as.inner[0].a = _value.inner[0].a;
        (*reinterpret_cast<device metal::packed_float3*>(&out_as.inner[0].b)) = metal::packed_float3(_value.inner[0].b);
        out_as.inner[1].a = _value.inner[1].a;
        (*reinterpret_cast<device metal::packed_float3*>(&out_as.inner[1].b)) = metal::packed_float3(_value.inner[1].b);
    }
    {
        const type_4 _value = type_4 {{m, m}};
        (*reinterpret_cast<device metal::packed_float3*>(&out_am.inner[0][0])) = metal::packed_float3(_value.inner[0][0]);
        (*reinterpret_cast<device metal::packed_float3*>(&out_am.inner[0][1])) = metal::packed_float3(_value.inner[0][1]);
        (*reinterpret_cast<device metal::packed_float3*>(&out_am.inner[0][2])) = metal::packed_float3(_value.inner[0][2]);
        (*reinterpret_cast<device metal::packed_float3*>(&out_am.inner[1][0])) = metal::packed_float3(_value.inner[1][0]);
        (*reinterpret_cast<device metal::packed_float3*>(&out_am.inner[1][1])) = metal::packed_float3(_value.inner[1][1]);
        (*reinterpret_cast<device metal::packed_float3*>(&out_am.inner[1][2])) = metal::packed_float3(_value.inner[1][2]);
    }
    {
        const Nested _value = Nested {s, m};
        out_n.s.a = _value.s.a;
        (*reinterpret_cast<device metal::packed_float3*>(&out_n.s.b)) = metal::packed_float3(_value.s.b);
        (*reinterpret_cast<device metal::packed_float3*>(&out_n.m[0])) = metal::packed_float3(_value.m[0]);
        (*reinterpret_cast<device metal::packed_float3*>(&out_n.m[1])) = metal::packed_float3(_value.m[1]);
        (*reinterpret_cast<device metal::packed_float3*>(&out_n.m[2])) = metal::packed_float3(_value.m[2]);
    }
    return;
}
