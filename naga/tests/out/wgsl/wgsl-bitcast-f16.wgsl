enable f16;

struct Data {
    u: u32,
    u2_: vec2<u32>,
    h2_: vec2<f16>,
    h4_: vec4<f16>,
}

const C_U: u32 = 3221240832u;
const C_H2_: vec2<f16> = vec2<f16>(1h, -2h);
const C_U2_: vec2<u32> = vec2<u32>(3221240832u, 3288348672u);
const C_H4_: vec4<f16> = vec4<f16>(1h, -2h, 0.5h, -4h);

@group(0) @binding(0) 
var<storage, read_write> data: Data;

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e2 = data.h2_;
    let r_u = bitcast<u32>(_e2);
    let _e6 = data.u;
    let r_h2_ = bitcast<vec2<f16>>(_e6);
    let _e10 = data.h4_;
    let r_u2_ = bitcast<vec2<u32>>(_e10);
    let _e14 = data.u2_;
    let r_h4_ = bitcast<vec4<f16>>(_e14);
    data.u = (r_u + C_U);
    data.h2_ = (r_h2_ + C_H2_);
    data.u2_ = (r_u2_ + C_U2_);
    data.h4_ = (r_h4_ + C_H4_);
    return;
}
