// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;
struct DefaultConstructible {
    template<typename T>
    operator T() && {
        return T {};
    }
};

struct Data {
    uint u;
    char _pad1[4];
    metal::uint2 u2_;
    metal::half2 h2_;
    char _pad3[4];
    metal::half4 h4_;
};
constant uint C_U = 3221240832u;
constant metal::half2 C_H2_ = metal::half2(1.0h, -2.0h);
constant metal::uint2 C_U2_ = metal::uint2(3221240832u, 3288348672u);
constant metal::half4 C_H4_ = metal::half4(1.0h, -2.0h, 0.5h, -4.0h);

kernel void main_(
  device Data& data [[user(fake0)]]
) {
    metal::half2 _e2 = data.h2_;
    uint r_u = as_type<uint>(_e2);
    uint _e6 = data.u;
    metal::half2 r_h2_ = as_type<metal::half2>(_e6);
    metal::half4 _e10 = data.h4_;
    metal::uint2 r_u2_ = as_type<metal::uint2>(_e10);
    metal::uint2 _e14 = data.u2_;
    metal::half4 r_h4_ = as_type<metal::half4>(_e14);
    data.u = r_u + C_U;
    data.h2_ = r_h2_ + C_H2_;
    data.u2_ = r_u2_ + C_U2_;
    data.h4_ = r_h4_ + C_H4_;
    return;
}
