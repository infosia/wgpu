@group(0) @binding(0)
var gbuffer_color: subpass_input_multisampled<f32>;

@fragment
fn main(@builtin(sample_index) sid: u32) -> @location(0) vec4<f32> {
    return subpassLoad(gbuffer_color, sid);
}
