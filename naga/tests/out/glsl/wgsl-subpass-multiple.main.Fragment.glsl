#version 310 es

precision highp float;
precision highp int;

layout(input_attachment_index = 0) uniform subpassInput _group_0_binding_0_fs;

layout(input_attachment_index = 1) uniform subpassInput _group_0_binding_1_fs;

layout(input_attachment_index = 2) uniform subpassInput _group_0_binding_2_fs;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec4 albedo = subpassLoad(_group_0_binding_0_fs);
    vec4 normal = subpassLoad(_group_0_binding_1_fs);
    vec4 material = subpassLoad(_group_0_binding_2_fs);
    _fs2p_location0 = (((albedo + normal) + material) / vec4(3.0));
    return;
}

