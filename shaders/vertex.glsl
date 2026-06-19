// PythonCraft Engine - Vertex Shader
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aTexCoord;
layout(location = 4) in float aAO;
layout(location = 5) in float aLight;
layout(location = 6) in float aOverlay;

uniform mat4 projection;
uniform mat4 view;

out vec3 vColor;
out vec3 vNormal;
out vec3 vFragPos;
out vec3 vTexCoord;
out float vAO;
out float vLight;
out float vOverlay;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    vColor = aColor;
    vNormal = aNormal;
    vFragPos = aPos;
    vTexCoord = aTexCoord;
    vAO = aAO;
    vLight = aLight;
    vOverlay = aOverlay;
}
