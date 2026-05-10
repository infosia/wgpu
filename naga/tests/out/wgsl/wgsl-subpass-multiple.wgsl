@group(0) @binding(0) 
var gbuffer_albedo: subpass_input<f32>;
@group(0) @binding(1) 
var gbuffer_normal: subpass_input<f32>;
@group(0) @binding(2) 
var gbuffer_material: subpass_input<f32>;

@fragment 
fn main() -> @location(0) vec4<f32> {
    let albedo = subpassLoad(gbuffer_albedo);
    let normal = subpassLoad(gbuffer_normal);
    let material = subpassLoad(gbuffer_material);
    return (((albedo + normal) + material) / vec4(3f));
}
