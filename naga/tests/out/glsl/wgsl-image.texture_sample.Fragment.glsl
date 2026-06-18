#version 430 core
uniform sampler1D _group_0_binding_0_fs;

uniform sampler2D _group_0_binding_1_fs;

uniform sampler2DArray _group_0_binding_4_fs;

uniform samplerCube _group_0_binding_5_fs;

uniform samplerCubeArray _group_0_binding_6_fs;

uniform sampler3D _group_0_binding_7_fs;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec4 a = vec4(0.0);
    vec2 _e1 = vec2(0.5);
    vec3 _e3 = vec3(0.5);
    ivec2 _e6 = ivec2(3, 1);
    vec4 _e9 = a;
    vec4 _e12 = texture(_group_0_binding_0_fs, 0.5);
    a = (_e9 + _e12);
    vec4 _e14 = a;
    vec4 _e17 = texture(_group_0_binding_1_fs, vec2(_e1));
    a = (_e14 + _e17);
    vec4 _e19 = a;
    vec4 _e25 = textureOffset(_group_0_binding_1_fs, vec2(_e1), ivec2(3, 1));
    a = (_e19 + _e25);
    vec4 _e27 = a;
    vec4 _e30 = textureLod(_group_0_binding_1_fs, vec2(_e1), 2.3);
    a = (_e27 + _e30);
    vec4 _e32 = a;
    vec4 _e35 = textureLodOffset(_group_0_binding_1_fs, vec2(_e1), 2.3, ivec2(3, 1));
    a = (_e32 + _e35);
    vec4 _e37 = a;
    vec4 _e41 = textureOffset(_group_0_binding_1_fs, vec2(_e1), ivec2(3, 1), 2.0);
    a = (_e37 + _e41);
    vec4 _e43 = a;
    vec4 _e46 = textureGrad(_group_0_binding_1_fs, vec2(_e1), _e1, _e1);
    a = (_e43 + _e46);
    vec4 _e48 = a;
    vec4 _e51 = textureLod(_group_0_binding_1_fs, vec2(_e1), 0.0);
    a = (_e48 + _e51);
    vec4 _e53 = a;
    vec4 _e57 = texture(_group_0_binding_4_fs, vec3(_e1, 0u));
    a = (_e53 + _e57);
    vec4 _e59 = a;
    vec4 _e63 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0u), ivec2(3, 1));
    a = (_e59 + _e63);
    vec4 _e65 = a;
    vec4 _e69 = textureLod(_group_0_binding_4_fs, vec3(_e1, 0u), 2.3);
    a = (_e65 + _e69);
    vec4 _e71 = a;
    vec4 _e75 = textureLodOffset(_group_0_binding_4_fs, vec3(_e1, 0u), 2.3, ivec2(3, 1));
    a = (_e71 + _e75);
    vec4 _e77 = a;
    vec4 _e82 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0u), ivec2(3, 1), 2.0);
    a = (_e77 + _e82);
    vec4 _e84 = a;
    vec4 _e88 = texture(_group_0_binding_4_fs, vec3(_e1, 0));
    a = (_e84 + _e88);
    vec4 _e90 = a;
    vec4 _e94 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0), ivec2(3, 1));
    a = (_e90 + _e94);
    vec4 _e96 = a;
    vec4 _e100 = textureLod(_group_0_binding_4_fs, vec3(_e1, 0), 2.3);
    a = (_e96 + _e100);
    vec4 _e102 = a;
    vec4 _e106 = textureLodOffset(_group_0_binding_4_fs, vec3(_e1, 0), 2.3, ivec2(3, 1));
    a = (_e102 + _e106);
    vec4 _e108 = a;
    vec4 _e113 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0), ivec2(3, 1), 2.0);
    a = (_e108 + _e113);
    vec4 _e115 = a;
    vec4 _e119 = texture(_group_0_binding_6_fs, vec4(_e3, 0u));
    a = (_e115 + _e119);
    vec4 _e121 = a;
    vec4 _e125 = textureLod(_group_0_binding_6_fs, vec4(_e3, 0u), 2.3);
    a = (_e121 + _e125);
    vec4 _e127 = a;
    vec4 _e132 = texture(_group_0_binding_6_fs, vec4(_e3, 0u), 2.0);
    a = (_e127 + _e132);
    vec4 _e134 = a;
    vec4 _e138 = texture(_group_0_binding_6_fs, vec4(_e3, 0));
    a = (_e134 + _e138);
    vec4 _e140 = a;
    vec4 _e144 = textureLod(_group_0_binding_6_fs, vec4(_e3, 0), 2.3);
    a = (_e140 + _e144);
    vec4 _e146 = a;
    vec4 _e151 = texture(_group_0_binding_6_fs, vec4(_e3, 0), 2.0);
    a = (_e146 + _e151);
    vec4 _e153 = a;
    vec4 _e156 = textureGrad(_group_0_binding_5_fs, vec3(_e3), _e3, _e3);
    a = (_e153 + _e156);
    vec4 _e158 = a;
    vec4 _e161 = textureGrad(_group_0_binding_7_fs, vec3(_e3), _e3, _e3);
    a = (_e158 + _e161);
    vec4 _e163 = a;
    _fs2p_location0 = _e163;
    return;
}

