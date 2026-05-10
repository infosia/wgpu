// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


struct main_Input {
    metal::uint4 prev [[color(0)]];
};
struct main_Output {
    metal::uint4 member [[color(0)]];
};
fragment main_Output main_(
  main_Input varyings [[stage_in]]
) {
    const auto prev = varyings.prev;
    return main_Output { prev };
}
