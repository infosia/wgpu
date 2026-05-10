@group(0) @binding(0)
var gbuffer_uint: subpass_input<u32>;

@fragment
fn main() -> @location(0) vec4<u32> {
    return subpassLoad(gbuffer_uint);
}
