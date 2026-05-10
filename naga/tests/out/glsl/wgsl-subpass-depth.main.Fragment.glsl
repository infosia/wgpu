#version 310 es

precision highp float;
precision highp int;

layout(input_attachment_index = 0) uniform subpassInput _group_0_binding_0_fs;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    float depth = subpassLoad(_group_0_binding_0_fs).x;
    _fs2p_location0 = vec4(depth, depth, depth, 1.0);
    return;
}

