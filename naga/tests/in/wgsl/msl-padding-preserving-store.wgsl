struct S {
    a: u32,
    b: vec3<f32>,
}

struct Nested {
    s: S,
    m: mat3x3<f32>,
}

@group(0) @binding(0)
var<storage, read_write> out_s: S;

@group(0) @binding(1)
var<storage, read_write> out_m: mat3x3<f32>;

@group(0) @binding(2)
var<storage, read_write> out_as: array<S, 2>;

@group(0) @binding(3)
var<storage, read_write> out_am: array<mat3x3<f32>, 2>;

@group(0) @binding(4)
var<storage, read_write> out_n: Nested;

@compute @workgroup_size(1)
fn main() {
    let s = S(42u, vec3<f32>(1.0, 2.0, 3.0));
    let m = mat3x3<f32>(
        vec3<f32>(1.0, 2.0, 3.0),
        vec3<f32>(4.0, 5.0, 6.0),
        vec3<f32>(7.0, 8.0, 9.0),
    );

    out_s = s;
    out_m = m;
    out_as = array<S, 2>(s, s);
    out_am = array<mat3x3<f32>, 2>(m, m);
    out_n = Nested(s, m);
}
