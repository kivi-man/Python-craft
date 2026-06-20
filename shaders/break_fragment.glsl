#version 330 core
in vec2 v_texcoord;

uniform sampler2DArray u_texture;
uniform float u_layer;

out vec4 FragColor;

void main() {
    vec4 texColor = texture(u_texture, vec3(v_texcoord, u_layer));
    if (texColor.a < 0.1) discard;
    FragColor = texColor;
}
