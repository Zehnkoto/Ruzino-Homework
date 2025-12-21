import callable_data;
#include "utils/random.slangh"
#include "utils/ray.slang"
import Scene.SceneTypes;
import Scene.BindlessMaterial;
import Scene.MaterialParams;
import Scene.MaterialParamsBuffer;
import Scene.BindlessVertexBuffer;

import Scene.StandardSurface;
import Scene.UsdPreviewSurface;
import utils.Math.ShadingFrame;
import Utils.Math.MathHelpers;

import pt_constants;
import pt_sample_lights;

RWStructuredBuffer<uint> random_seeds;
RWTexture2D<float4> output;
StructuredBuffer<uint2> inPixelTarget;
StructuredBuffer<RayInfo> rays;

struct RayPayload {
    // Geometry information from hit
    VertexInfo vertex;
    uint materialID;
    uint eta_flipped;

    // Ray state
    uint depth;
    uint seed;
    bool hit;  // Whether ray hit geometry (false means miss)
};

[shader("raygeneration")]
void RayGen()
{
    uint2 pixel = inPixelTarget[DispatchRaysIndex().x];
    RayInfo ray = rays[DispatchRaysIndex().x];

    RayPayload payload;
    payload.seed = random_seeds[DispatchRaysIndex().x];

    Spectrum output_color = Spectrum(0.0);

    const int spp = 1;
    const int max_depth = 6;

    ray.TMax = 50000.0;

    for (int i = 0; i < spp; i++) {
        Spectrum accumulated_color = Spectrum(0.0);
        Spectrum current_throughput = Spectrum(1.0);

        RayDesc current_ray = get_ray_desc(ray);
        current_ray.TMax = 50000.0;

        uint id = DispatchRaysIndex().x;

        // Iterative path tracing loop
        for (int depth = 0; depth < max_depth; depth++) {
            payload.depth = depth;
            payload.hit = false;

            // Trace ray to get geometry information
            TraceRay(
                SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, current_ray, payload);

            // Check if ray missed
            if (!payload.hit) {
                // Environment lighting (same as Miss shader)
                float3 rayDir = current_ray.Direction;
                Spectrum envColor = Spectrum(0.0);

                // Check if we have a dome light
                bool hasDomeLight = false;
                LightData domeLight;

                for (uint lightIdx = 0; lightIdx < ptConstants.lightCount;
                     lightIdx++) {
                    LightData light = lightBuffer[lightIdx];
                    if (light.type == uint(LightType::Dome)) {
                        hasDomeLight = true;
                        domeLight = light;
                        break;
                    }
                }

                if (hasDomeLight) {
                    float3 localDir =
                        mul(float4(rayDir, 0.0), domeLight.transMat).xyz;
                    localDir = normalize(localDir);

#if USE_DOME_LIGHT_CALLABLE
                    DomeLightCallableData domeData;
                    domeData.rayDirection = rayDir;
                    domeData.localDirection = localDir;
                    domeData.envColor = Spectrum(0.0);
                    domeData.intensity = domeLight.intensity;
                    domeData.textureIndex = domeLight.textureIndex;
                    domeData.seed = payload.seed;

                    CallShader(ptConstants.domeLightCallableIndex, domeData);

                    envColor = domeData.envColor;
                    payload.seed = domeData.seed;
#else
                    if (domeLight.textureIndex >= 0) {
                        float phi = atan2(localDir.z, localDir.x);
                        float theta = acos(clamp(localDir.y, -1.0, 1.0));
                        float u = (phi + 3.14159265359) / (2.0 * 3.14159265359);
                        float v = theta / 3.14159265359;

                        float4 texColor =
                            t_BindlessTextures[domeLight.textureIndex]
                                .SampleLevel(samplers[1], float2(u, v), 0);
                        envColor = Spectrum(texColor.rgb * domeLight.intensity);
                    }
                    else {
                        envColor = Spectrum(domeLight.intensity);
                    }
#endif
                }
                else {
                    float3 horizonColor = float3(0.8, 0.9, 1.0);
                    float3 zenithColor = float3(0.2, 0.4, 0.8);
                    float3 sunsetColor = float3(0.9, 0.5, 0.2);

                    float skyBlend = pow(max(0.0, rayDir.z + 0.1), 0.8);
                    float3 skyColor = lerp(horizonColor, zenithColor, skyBlend);

                    float3 sunDir = normalize(float3(0.5, 0.5, 0.5));
                    float sunDot = max(0.0, dot(rayDir, sunDir));
                    float sunIntensity = pow(sunDot, 128.0) * 0.1;
                    float sunsetIntensity =
                        pow(sunDot, 8.0) * (1.0 - rayDir.z * 0.5);

                    skyColor += float3(1.0, 0.9, 0.7) * sunIntensity;
                    skyColor +=
                        sunsetColor * sunsetIntensity * (1.0 - skyBlend) * 0.4;

                    envColor = Spectrum(skyColor * 0.0);
                }

                accumulated_color =
                    add(accumulated_color, mul(current_throughput, envColor));
                break;  // Path terminated
            }

            // Process hit - material evaluation
            VertexInfo vertex = payload.vertex;
            uint material_id = payload.materialID;
            uint eta_flipped = payload.eta_flipped;

            var material_header = materialHeaderBuffer[material_id];

            // Sample a light source for direct lighting
            LightSample lightSample =
                sampleLights(vertex.posW, vertex.faceNormalW, payload.seed);

            // Step 1: Call material data fetch callable
            FetchCallableData fetchData;
            fetchData.vertexInfo = vertex;
            fetchData.materialBlobID = material_header.material_blob_id;
            fetchData.material_params_index = id;

            CallShader(
                ptConstants.materialFetchCallableBaseIndex +
                    material_header.material_type_id,
                fetchData);

            // Step 2: Call material evaluation callable
            EvalCallableData evalData;
            evalData.color = Spectrum(0.0);
            evalData.throughput = Spectrum(1.0);
            evalData.L = lightSample.direction;
            evalData.V = -current_ray.Direction;
            evalData.sampledDir = float3(0, 0, 0);
            evalData.pdf = 0.0;
            evalData.seed = payload.seed;
            evalData.eta_flipped = eta_flipped;
            evalData.material_params_index = id;
            evalData.vertexInfo = vertex;

            CallShader(fetchData.shader_type_id, evalData);

            float pdf = evalData.pdf;
            payload.seed = evalData.seed;

            // Check for invalid PDF
            if (pdf == 0.0) {
                break;
            }

            // Validate throughput
            if (any(isnan(evalData.throughput.toFloat3())) ||
                any(isinf(evalData.throughput.toFloat3()))) {
                accumulated_color = Spectrum(float3(1, 0, 0));  // Debug color
                break;
            }

            // Add direct lighting contribution
            if (lightSample.valid) {
                Spectrum directContrib = div(
                    mul(Spectrum(
                            lightSample.intensity *
                            max(0.0,
                                dot(vertex.normalW, lightSample.direction))),
                        evalData.color),
                    lightSample.pdf);
                accumulated_color = add(
                    accumulated_color, mul(current_throughput, directContrib));
            }

            // Update throughput for next bounce
            float3 ray_dir = evalData.sampledDir;
            bool isTransmission = dot(ray_dir, vertex.faceNormalW) < 0.0;

            current_throughput =
                mul(current_throughput, div(evalData.throughput, pdf));

            // Russian Roulette path termination
            if (depth >= 3) {
                float3 thr = current_throughput.toFloat3();
                float rrProbability = min(0.95, max(thr.x, max(thr.y, thr.z)));
                if (random_float(payload.seed) > rrProbability) {
                    break;
                }
                current_throughput = div(current_throughput, rrProbability);
            }

            // Prepare next ray
            float3 offset_normal =
                isTransmission ? -vertex.faceNormalW : vertex.faceNormalW;
            float3 ray_offset = offset_normal * 1e-3f;

            current_ray.Origin = vertex.posW + ray_offset;
            current_ray.Direction = ray_dir;
            current_ray.TMin = 0.0;
            current_ray.TMax = 10000.0;
        }

        output_color = add(output_color, accumulated_color);
    }
    output_color = div(output_color, float(spp));
    output[pixel] = float4(output_color.toFloat3(), 1.0);

    random_seeds[DispatchRaysIndex().x] = payload.seed;
}

[shader("closesthit")]
void ClosestHit(
    inout RayPayload payload,
    in BuiltInTriangleIntersectionAttributes attribs)
{
    payload.hit = true;

    uint instance_id = InstanceID();
    uint prim_id = PrimitiveIndex();

    // Get vertex information
    VertexInfo vertex = get_interpolated_vertex(
        instance_id,
        prim_id,
        float2(attribs.barycentrics.x, attribs.barycentrics.y));

    // Fix normal orientation
    uint eta_flipped = 0;
    if (dot(WorldRayDirection(), vertex.normalW) >= 0.0) {
        vertex.normalW = -vertex.normalW;
        vertex.tangentW = -vertex.tangentW;
        vertex.faceNormalW = -vertex.faceNormalW;
        eta_flipped = 1;
    }

    // Get material ID
    uint material_id = instanceDescBuffer[instance_id].materialID;
    if (vertex.subsetMaterialID != -1) {
        material_id = vertex.subsetMaterialID;
    }

    // Store geometry information in payload
    payload.vertex = vertex;
    payload.materialID = material_id;
    payload.eta_flipped = eta_flipped;
}
[shader("miss")]
void Miss(inout RayPayload payload)
{
    payload.hit = false;
}

[shader("closesthit")]
void ShadowHit(
    inout ShadowPayload payload,
    in BuiltInTriangleIntersectionAttributes attribs)
{
    payload.isVisible = true;
    // payload.isVisible = false;
    // return;
    // Limit recursion depth to prevent infinite loops
    const uint MAX_SHADOW_OPACITY = 4;
    if (payload.recursionDepth >= MAX_SHADOW_OPACITY) {
        payload.isVisible = false;
        return;
    }

    uint instance_id = InstanceID();
    uint prim_id = PrimitiveIndex();

    // Get vertex information
    VertexInfo vertex = get_interpolated_vertex_simple(
        instance_id, prim_id, attribs.barycentrics);

    // Get material ID
    var material_id = instanceDescBuffer[instance_id].materialID;
    if (vertex.subsetMaterialID != -1) {
        material_id = vertex.subsetMaterialID;
    }

    var material_header = materialHeaderBuffer[material_id];

    // Optimized opacity fetch: Call the opacity-specific callable
    // This avoids full material parameter buffer access
    FetchCallableData opacityFetchData;
    opacityFetchData.vertexInfo = vertex;
    opacityFetchData.materialBlobID = material_header.material_blob_id;
    opacityFetchData.material_params_index =
        0;  // Will be filled with opacity value
    opacityFetchData.shader_type_id = 0;

    // Call opacity fetch callable (uses fetch_shader_opacity internally)
    CallShader(
        ptConstants.materialFetchCallableBaseIndex +
            material_header.material_type_id +
            ptConstants.materialOpacityCallableOffset,
        opacityFetchData);

    // Extract opacity from material_params_index (reinterpreted as float)
    float opacity = asfloat(opacityFetchData.material_params_index);
    // Simple stochastic opacity test
    float opacity_threshold = random_float(payload.seed);

    if (opacity < 0.25) {
        // Material is transparent, continue tracing shadow ray
        float3 rayOrigin = vertex.posW;
        float remainingDist = payload.rayTMax - RayTCurrent();

        if (remainingDist > 0.001) {
            RayDesc continuationRay;
            continuationRay.Origin = rayOrigin;
            continuationRay.Direction = WorldRayDirection();
            continuationRay.TMin = 0.001;
            continuationRay.TMax = remainingDist;

            ShadowPayload continuationPayload;
            continuationPayload.isVisible = true;
            continuationPayload.rayTMax = remainingDist;
            continuationPayload.recursionDepth = payload.recursionDepth + 1;
            continuationPayload.seed = payload.seed;

            TraceRay(
                SceneBVH,
                RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
                0xFF,
                1,  // Shadow hit group index
                0,
                1,  // Shadow miss shader index
                continuationRay,
                continuationPayload);
            payload.seed = continuationPayload.seed;
            payload.isVisible = continuationPayload.isVisible;
            return;
        }
    }
    // If we reach here, the surface is opaque
    payload.isVisible = false;
}

[shader("miss")]
void ShadowMiss(inout ShadowPayload payload)
{
    payload.isVisible = true;
}

// ============================================================================
// Material Evaluation Callables
// ============================================================================
// Note: Callable implementations are now in separate files:
// - callables/eval_fallback.slang
// - callables/eval_standard_surface.slang
// - callables/eval_preview_surface.slang
// They are linked via add_source_code() in the C++ code.
