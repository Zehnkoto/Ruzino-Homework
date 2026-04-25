#version 430 core

// Define a uniform struct for lights
struct Light {
    mat4 light_projection;
    mat4 light_view;
    vec3 position;
    float radius;
    vec3 color; 
    int shadow_map_id;
};

layout(std430, binding = 0, row_major) buffer lightsBuffer {
    Light lights[4];
};

uniform vec2 iResolution;

uniform sampler2D diffuseColorSampler;
uniform sampler2D normalMapSampler; 
uniform sampler2D metallicRoughnessSampler;
uniform sampler2DArray shadow_maps;
uniform sampler2D position;

uniform vec3 camPos;
uniform int light_count;

layout(location = 0) out vec4 Color;

// --- PCSS Helper Functions ---

vec2 poissonDisk[16] = vec2[]( 
   vec2( -0.94201624, -0.39906216 ), vec2( 0.94558609, -0.76890725 ), 
   vec2( -0.094184101, -0.92938870 ), vec2( 0.34495938, 0.29387760 ), 
   vec2( -0.91588581, 0.45771432 ), vec2( -0.81544232, -0.87912464 ), 
   vec2( -0.38277543, 0.27676845 ), vec2( 0.97484398, 0.75648379 ), 
   vec2( 0.44323325, -0.97511554 ), vec2( 0.53742981, -0.47373420 ), 
   vec2( -0.26496911, -0.41893023 ), vec2( 0.79197514, 0.19090188 ), 
   vec2( -0.24188840, 0.99706507 ), vec2( -0.81409955, 0.91437590 ), 
   vec2( 0.19984126, 0.78641367 ), vec2( 0.14383161, -0.14100790 ) 
);

// Step 1: Blocker Search (Now passing dynamic bias)
float findBlocker(sampler2DArray shadowMap, int layer, vec2 uv, float zReceiver, float bias) {
    int blockerNum = 0;
    float blockDepth = 0.0;
    float searchRadius = 0.02; 

    for(int i = 0; i < 16; i++) {
        float shadowMapDepth = texture(shadowMap, vec3(uv + poissonDisk[i] * searchRadius, layer)).x;
        // Use the dynamically calculated bias here!
        if(zReceiver > shadowMapDepth + bias) { 
            blockDepth += shadowMapDepth;
            blockerNum++;
        }
    }
    
    if(blockerNum == 0) return -1.0; 
    
    return blockDepth / float(blockerNum);
}

// Step 3: Percentage Closer Filtering (Now passing dynamic bias)
float PCF(sampler2DArray shadowMap, int layer, vec2 uv, float zReceiver, float filterRadius, float bias) {
    float sum = 0.0;
    for(int i = 0; i < 16; i++) {
        float depth = texture(shadowMap, vec3(uv + poissonDisk[i] * filterRadius, layer)).x;
        // Use the dynamically calculated bias here!
        sum += (zReceiver > depth + bias) ? 1.0 : 0.0; 
    }
    return sum / 16.0;
}

// Main PCSS Algorithm (Requires bias parameter)
float PCSS(sampler2DArray shadowMap, int layer, vec2 uv, float zReceiver, float bias) {
    float dBlocker = findBlocker(shadowMap, layer, uv, zReceiver, bias);
    if(dBlocker < 0.0) return 0.0; 
    
    float wLight = 0.05; 
    float penumbra = (zReceiver - dBlocker) / dBlocker * wLight;
    
    return PCF(shadowMap, layer, uv, zReceiver, penumbra, bias);
}

void main() {
    vec2 uv = gl_FragCoord.xy / iResolution;

    vec3 pos = texture2D(position,uv).xyz;
    vec3 normal = texture2D(normalMapSampler,uv).xyz;
    vec3 albedo = texture2D(diffuseColorSampler, uv).xyz;

    vec4 metalnessRoughness = texture2D(metallicRoughnessSampler,uv);
    float metal = metalnessRoughness.x;
    float roughness = metalnessRoughness.y;

    vec3 finalColor = 0.15 * albedo;

    for(int i = 0; i < light_count; i ++) {

        // --- FIX 1: Calculate distance and direction for attenuation ---
        vec3 lightVector = lights[i].position - pos;
        float dist = length(lightVector);
        vec3 lightDir = normalize(lightVector);
        vec3 viewDir = normalize(camPos - pos);
        vec3 halfDir = normalize(lightDir + viewDir);

        // Apply inverse square law attenuation 
        // Adjust the '50.0' numerator to control light intensity based on scene scale
        float attenuation = 25.0 / (dist * dist + 1.0);

        float diff = max(dot(normal, lightDir), 0.0);
        // Multiply by attenuation
        vec3 diffuse = lights[i].color * diff * albedo * attenuation;

        float shininess = max((1.0 - roughness) * 128.0, 1.0); 
        float spec = pow(max(dot(normal, halfDir), 0.0), shininess);
        // Multiply by attenuation
        vec3 specular = lights[i].color * spec * attenuation;

        vec4 posLightSpace = lights[i].light_projection * lights[i].light_view * vec4(pos, 1.0);
        vec3 projCoords = posLightSpace.xyz / posLightSpace.w;
        projCoords = projCoords * 0.5 + 0.5;

        float shadow = 0.0;
        
        if(projCoords.z <= 1.0 && projCoords.x >= 0.0 && projCoords.x <= 1.0 && projCoords.y >= 0.0 && projCoords.y <= 1.0) {
            // Calculate slope-based bias depending on the angle of the surface
             float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.008);
            
            // Pass the calculated bias down into the PCSS chain
            shadow = PCSS(shadow_maps, lights[i].shadow_map_id, projCoords.xy, projCoords.z, bias);
            
        }

        finalColor += (1.0 - shadow) * (diffuse + specular);
    }
    
    // --- FIX 2: Gamma Correction (Tone Mapping) ---
    // Convert linear color to sRGB space for accurate monitor display
     finalColor = pow(finalColor, vec3(1.0 / 2.2));
    
    Color = vec4(finalColor, 1.0);
}