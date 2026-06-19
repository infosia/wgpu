@fragment
fn main(@color(0) prev: vec4<u32>) -> @location(0) @interpolate(flat) vec4<u32> {
    return prev;
}
