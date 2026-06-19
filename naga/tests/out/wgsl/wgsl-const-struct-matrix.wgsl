struct MatrixHolder {
    scale: f32,
    transform: mat3x2<f32>,
    bias: vec2<f32>,
}

const STRUCT_WITH_MATRIX: MatrixHolder = MatrixHolder(2f, mat3x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, 4f), vec2<f32>(5f, 6f)), vec2<f32>(7f, 8f));
const ARRAY_WITH_MATRIX: array<mat3x2<f32>, 2> = array<mat3x2<f32>, 2>(mat3x2<f32>(vec2<f32>(9f, 10f), vec2<f32>(11f, 12f), vec2<f32>(13f, 14f)), mat3x2<f32>(vec2<f32>(15f, 16f), vec2<f32>(17f, 18f), vec2<f32>(19f, 20f)));

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let phony = (((STRUCT_WITH_MATRIX.scale + STRUCT_WITH_MATRIX.transform[2].y) + STRUCT_WITH_MATRIX.bias.x) + ARRAY_WITH_MATRIX[1][0].y);
    return;
}
