#version 330 core
in vec2 v_ndc;
out vec4 FragColor;

uniform mat4 u_inv_proj_view;
uniform float u_water_surface_y;

void main() {
    // Reconstruct world space coordinate on the near plane
    vec4 clip_space = vec4(v_ndc, -1.0, 1.0);
    vec4 world_space = u_inv_proj_view * clip_space;
    world_space /= world_space.w; // Perspective divide
    
    if (world_space.y <= u_water_surface_y) {
        // Pixel is underwater! Apply a semi-transparent blue tint.
        FragColor = vec4(0.05, 0.20, 0.70, 0.6); 
    } else {
        // Pixel is above water! Do not alter the screen.
        FragColor = vec4(0.0);
    }
}
