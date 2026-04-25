#version 430 core

layout(location = 0) in vec3 aPos;
// --- Add normal and UV inputs for displacement ---
layout(location = 1) in vec3 aNormal;
layout(std430, binding = 0) buffer buffer0 {
    vec2 data[];
} aTexcoord;

uniform mat4 light_view;
uniform mat4 light_projection;
uniform mat4 model;
out vec3 vertexPosition;

// --- Add displacement map sampler ---
uniform sampler2D displacementMapSampler;

void main() {
    // Reconstruct UVs
    vec2 texCoord = aTexcoord.data[gl_VertexID];
    texCoord.y = 1.0 - texCoord.y;

    // Sample height with the exact same logic and scale as the main pass
    float height = texture(displacementMapSampler, texCoord).r;
    float dispScale = 0.05; 

    // Displace the vertex position
    vec3 displacedPos = aPos + aNormal * height * dispScale;

    // Transform using light matrices
    gl_Position = light_projection * light_view * model * vec4(displacedPos, 1.0);
    vec4 vPosition = model * vec4(displacedPos, 1.0);
    vertexPosition = vPosition.xyz / vPosition.w;
}