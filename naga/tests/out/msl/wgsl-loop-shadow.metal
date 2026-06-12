// language: metal1.2
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct Output {
    uint outer;
    uint body;
    uint continuing_value;
    uint iterations;
};

kernel void main_(
  device Output& output [[user(fake0)]]
) {
    uint my_idx = 7u;
    uint my_idx_1 = {};
    uint my_idx_2 = {};
    uint _e4 = my_idx;
    output.outer = _e4;
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e16 = output.iterations;
            my_idx_2 = _e16 + 20u;
            uint _e22 = my_idx_2;
            output.continuing_value = _e22;
            uint _e25 = output.iterations;
            output.iterations = _e25 + 1u;
            uint _e28 = my_idx_2;
            if (my_idx_2 >= 21u) {
                break;
            }
        }
        loop_init = false;
        uint _e7 = output.iterations;
        my_idx_1 = _e7 + 10u;
        uint _e13 = my_idx_1;
        output.body = _e13;
    }
    return;
}
