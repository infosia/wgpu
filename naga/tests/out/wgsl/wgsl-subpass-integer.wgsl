@group(0) @binding(0) 
var gbuffer_uint: subpass_input<u32>;

@fragment 
fn main() -> @location(0) @interpolate(flat) vec4<u32> {
    let _e1 = subpassLoad(gbuffer_uint);
    return _e1;
}
