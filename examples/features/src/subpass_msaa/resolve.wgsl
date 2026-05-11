@group(0) @binding(0) var msaa_input: texture_multisampled_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    out.position = vec4<f32>(x, -y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(msaa_input));
    let coords = clamp(vec2<i32>(pos.xy), vec2<i32>(0), dims - vec2<i32>(1));
    let sample_count = textureNumSamples(msaa_input);
    var color = vec4<f32>(0.0);
    for (var i = 0u; i < sample_count; i = i + 1u) {
        color = color + textureLoad(msaa_input, coords, i32(i));
    }
    return color / max(f32(sample_count), 1.0);
}
