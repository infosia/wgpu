#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void main() {
    int phony = ivec4(12, 0, 0, 0)[2];
    return;
}

