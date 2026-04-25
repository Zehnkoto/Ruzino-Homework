#version 430 core

out vec4 FragColor;

uniform vec2 iResolution;
uniform sampler2D colorSampler;
uniform sampler2D positionSampler;
uniform sampler2D normalSampler;

// Spatial hash function for pseudo-random noise generation
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453123);
}

void main() {
    vec2 uv = gl_FragCoord.xy / iResolution;

    vec3 fragColor = texture(colorSampler, uv).rgb;
    vec3 fragPos = texture(positionSampler, uv).xyz;
    vec3 normal = texture(normalSampler, uv).xyz;

    // Background culling: skip shading if there is no geometry
    if (length(normal) < 0.1) {
        FragColor = vec4(fragColor, 1.0);
        return;
    }

    float randomAngle = hash(uv) * 6.2831853;
    float occlusion = 0.0;
    
    // HBAO-style SSAO parameters
    int numSamples = 32;       // Number of samples per pixel
    float radius2D = 0.04;     // Search radius in 2D screen UV space
    float radius3D = 0.6;      // Physical interaction radius in 3D world space
    float bias = 0.05;         // Tolerance to prevent self-shadowing acne

    for(int i = 0; i < numSamples; ++i) {
        // Golden spiral distribution in 2D space
        float r = sqrt(float(i + 0.5) / float(numSamples));
        float theta = float(i) * 2.3398 + randomAngle;
        
        // Calculate the UV offset for the neighbor sample
        vec2 offset = vec2(cos(theta), sin(theta)) * (r * radius2D);
        vec2 sampleUV = clamp(uv + offset, 0.0, 1.0);
        
        // Read neighbor's 3D position
        vec3 neighborPos = texture(positionSampler, sampleUV).xyz;

        // Skip background neighbors
        if(length(neighborPos) < 0.1) continue;

        vec3 diff = neighborPos - fragPos;
        float dist = length(diff);

        // Check if the neighbor is close enough to cast occlusion
        if(dist < radius3D && dist > 0.001) {
            vec3 dir = diff / dist;
            
            // Calculate occlusion based on how much the neighbor is "above" the current surface
            float nDotD = max(dot(normal, dir) - bias, 0.0);
            
            // Attenuate occlusion based on distance (closer = darker)
            float attenuation = smoothstep(radius3D, 0.0, dist);
            occlusion += nDotD * attenuation;
        }
    }

    // Normalize and scale the occlusion factor (2.5 is used to boost contrast)
    occlusion = 1.0 - (occlusion / float(numSamples)) * 2.5;
    occlusion = clamp(occlusion, 0.0, 1.0);
    
    // Apply a power curve for richer, darker crevices
    occlusion = pow(occlusion, 1.5);

    // Apply the ambient occlusion to the color
    FragColor = vec4(fragColor * occlusion, 1.0);
}