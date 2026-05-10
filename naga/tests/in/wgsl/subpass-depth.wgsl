@group(0) @binding(0)
var gbuffer_depth: subpass_input_depth;

@fragment
fn main() -> @location(0) vec4<f32> {
    let depth = subpassLoad(gbuffer_depth);
    return vec4<f32>(depth, depth, depth, 1.0);
}
