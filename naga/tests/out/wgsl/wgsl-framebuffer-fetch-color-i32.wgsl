@fragment 
fn main(@color(0) prev: vec4<i32>) -> @location(0) @interpolate(flat) vec4<i32> {
    return prev;
}
