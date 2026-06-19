enable f16;

struct Data {
    u: u32,
    u2: vec2<u32>,
    h2: vec2<f16>,
    h4: vec4<f16>,
}

@group(0) @binding(0)
var<storage, read_write> data: Data;

const C_U: u32 = bitcast<u32>(vec2<f16>(1.0h, -2.0h));
const C_H2: vec2<f16> = bitcast<vec2<f16>>(0xc0003c00u);
const C_U2: vec2<u32> = bitcast<vec2<u32>>(vec4<f16>(1.0h, -2.0h, 0.5h, -4.0h));
const C_H4: vec4<f16> = bitcast<vec4<f16>>(vec2<u32>(0xc0003c00u, 0xc4003800u));

@compute @workgroup_size(1)
fn main() {
    let r_u: u32 = bitcast<u32>(data.h2);
    let r_h2: vec2<f16> = bitcast<vec2<f16>>(data.u);
    let r_u2: vec2<u32> = bitcast<vec2<u32>>(data.h4);
    let r_h4: vec4<f16> = bitcast<vec4<f16>>(data.u2);

    data.u = r_u + C_U;
    data.h2 = r_h2 + C_H2;
    data.u2 = r_u2 + C_U2;
    data.h4 = r_h4 + C_H4;
}
