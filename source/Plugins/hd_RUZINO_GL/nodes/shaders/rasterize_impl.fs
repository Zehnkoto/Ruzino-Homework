#version 430

layout(location = 0) out vec3 position;
layout(location = 1) out float depth;
layout(location = 2) out vec2 texcoords;
layout(location = 3) out vec3 diffuseColor;
layout(location = 4) out vec2 metallicRoughness;
layout(location = 5) out vec3 normal;

in vec3 vertexPosition;
in vec3 vertexNormal;
in vec2 vTexcoord;
uniform mat4 projection;
uniform mat4 view;

uniform sampler2D diffuseColorSampler;

// This only works for current scenes provided by the TAs 
// because the scenes we provide is transformed from gltf
uniform sampler2D normalMapSampler;
uniform sampler2D metallicRoughnessSampler;

void main() {
    position = vertexPosition;
    vec4 clipPos = projection * view * (vec4(position, 1.0));
    depth = clipPos.z / clipPos.w;
    texcoords = vTexcoord;

    diffuseColor = texture2D(diffuseColorSampler, vTexcoord).xyz;
    metallicRoughness = texture2D(metallicRoughnessSampler, vTexcoord).zy;

    vec3 normalmap_value = texture2D(normalMapSampler, vTexcoord).xyz;
    normal = normalize(vertexNormal);

    // Bulletproof tangent and bitangent evaluation (avoids NaNs when UVs are missing/zero)
    vec3 edge1 = dFdx(vertexPosition);
    vec3 edge2 = dFdy(vertexPosition);
    vec2 deltaUV1 = dFdx(vTexcoord);
    vec2 deltaUV2 = dFdy(vTexcoord);

    vec3 tangent;
    vec3 bitangent;
    float f = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
    
    // Check if UVs are likely missing or all zero
    if (abs(f) < 0.0001f) {
        // Find an arbitrary orthonormal basis. Z-axis as up if not pointing that way.
        vec3 up = abs(normal.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(0.0f, 1.0f, 0.0f);
        tangent = normalize(cross(up, normal));
        bitangent = normalize(cross(tangent, normal));
    } else {
        tangent = (edge1 * deltaUV2.y - edge2 * deltaUV1.y) / f;
        tangent = normalize(tangent);
        bitangent = (edge2 * deltaUV1.x - edge1 * deltaUV2.x) / f;
        bitangent = normalize(bitangent);
    }
    
    // Final sanity check before TBN construction
    if (isnan(tangent.x) || isnan(bitangent.x)) {
        // Last-resort fallback to a reliable orthonormal basis
        tangent = abs(normal.z) < 0.999f ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
        tangent = normalize(cross(tangent, normal));
        bitangent = cross(normal, tangent);
    }

    vec3 mappedNormal = normalmap_value * 2.0 - 1.0;
    mat3 TBN = mat3(tangent, bitangent, normal);
    
    // Fallback protection: use geometric normal if there is no valid normal map
    if (length(normalmap_value) < 0.1) {
        normal = normalize(vertexNormal);
    } else {
        normal = normalize(TBN * mappedNormal);
    }
}