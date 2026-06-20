#version 330 core
out vec2 v_ndc;
void main() {
    // Generate full-screen quad from gl_VertexID
    vec2 vertices[4] = vec2[4](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );
    v_ndc = vertices[gl_VertexID];
    gl_Position = vec4(v_ndc, -1.0, 1.0); // Z = -1.0 means it is placed at the near plane
}
