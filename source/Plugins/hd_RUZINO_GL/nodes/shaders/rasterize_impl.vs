#version 430 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(std430, binding = 0) buffer buffer0 {
    vec2 data[];
} aTexcoord;

out vec3 vertexPosition;
out vec3 vertexNormal;
out vec2 vTexcoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// --- Add displacement map sampler ---
uniform sampler2D displacementMapSampler;

void main() {
    vec2 texCoord = aTexcoord.data[gl_VertexID];
    texCoord.y = 1.0 - texCoord.y;

    float height = texture(displacementMapSampler, texCoord).r;
    
    float dispScale = 0.05; 

    vec3 displacedPos = aPos + aNormal * height * dispScale;

    gl_Position = projection * view * model * vec4(displacedPos, 1.0);
    vec4 vPosition = model * vec4(displacedPos, 1.0);
    vertexPosition = vPosition.xyz / vPosition.w;
    
    vertexNormal = (inverse(transpose(mat3(model))) * aNormal);
    vTexcoord = texCoord;
}