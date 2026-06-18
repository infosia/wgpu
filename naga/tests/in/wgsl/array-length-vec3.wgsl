@group(0) @binding(0)
var<storage, read> values: array<vec3<f32>>;

struct Output {
    value: u32,
}

@group(0) @binding(1)
var<storage, read_write> output: Output;

@compute @workgroup_size(1)
fn main() {
    output.value = arrayLength(&values);
}
