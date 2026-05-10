@group(0) @binding(0) 
var gbuffer_color: subpass_input<f32>;

@fragment 
fn main() -> @location(0) vec4<f32> {
    let _e1 = subpassLoad(gbuffer_color);
    return _e1;
}
