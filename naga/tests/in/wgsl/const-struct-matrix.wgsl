struct MatrixHolder {
    scale: f32,
    transform: mat3x2<f32>,
    bias: vec2<f32>,
}

const STRUCT_WITH_MATRIX: MatrixHolder = MatrixHolder(
    2.0,
    mat3x2<f32>(
        vec2<f32>(1.0, 2.0),
        vec2<f32>(3.0, 4.0),
        vec2<f32>(5.0, 6.0),
    ),
    vec2<f32>(7.0, 8.0),
);

const ARRAY_WITH_MATRIX: array<mat3x2<f32>, 2> = array<mat3x2<f32>, 2>(
    mat3x2<f32>(
        vec2<f32>(9.0, 10.0),
        vec2<f32>(11.0, 12.0),
        vec2<f32>(13.0, 14.0),
    ),
    mat3x2<f32>(
        vec2<f32>(15.0, 16.0),
        vec2<f32>(17.0, 18.0),
        vec2<f32>(19.0, 20.0),
    ),
);

@compute @workgroup_size(1)
fn main() {
    let holder = STRUCT_WITH_MATRIX;
    let matrices = ARRAY_WITH_MATRIX;
    let result = holder.scale + holder.transform[2][1] + holder.bias.x + matrices[1][0][1];
    _ = result;
}
