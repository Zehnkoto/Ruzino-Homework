import stdlib.genslang.mx_bsdf;
import pbrlib.genslang.lib.mx_closure_type;
import pbrlib.genslang.lib.mx_transmission_refract;
import pbrlib.genslang.lib.mx_microfacet_specular;
import pbrlib.genslang.lib.mx_environment_prefilter;

void mx_dielectric_bsdf_reflection(
    float3 L,
    float3 V,
    float3 P,
    float occlusion,
    float weight,
    float3 tint,
    float ior,
    float2 roughness,
    float3 N,
    float3 X,
    int distribution,
    int scatter_mode,
    inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS) {
        return;
    }

    N = mx_forward_facing_normal(N, V);

    if (dot(N, L) < 0.0) {
        bsdf.throughput = 0.0;
        bsdf.response = float3(0.0);
        return;
    }

    X = normalize(X - dot(X, N) * N);
    float3 Y = cross(N, X);
    float3 H = normalize(L + V);

    float NdotL = clamp(dot(N, L), M_FLOAT_EPS, 1.0);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);
    float VdotH = clamp(dot(V, H), M_FLOAT_EPS, 1.0);

    float2 safeAlpha = clamp(roughness, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(safeAlpha);
    float3 Ht = float3(dot(H, X), dot(H, Y), dot(H, N));

    FresnelData fd;
    float3 safeTint = max(tint, 0.0);
    if (bsdf.thickness > 0.0) {
        fd = mx_init_fresnel_dielectric_airy(ior, bsdf.thickness, bsdf.ior);
    }
    else {
        fd = mx_init_fresnel_dielectric(ior);
    }
    float3 F = mx_compute_fresnel(VdotH, fd);
    float D = mx_ggx_NDF(Ht, safeAlpha);
    float G = mx_ggx_smith_G2(NdotL, NdotV, avgAlpha);

    float F0 = mx_ior_to_f0(ior);
    float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
    comp = 1.0f;
    float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * comp;
    bsdf.throughput = 1.0 - dirAlbedo * weight;

    // Note: NdotL is cancelled out
    bsdf.response = D * F * G * comp * safeTint * occlusion * weight /
                    (4.0 * NdotV * NdotL);
}

void mx_dielectric_bsdf_transmission(
    float3 L,
    float3 V,
    float3 P,
    float occlusion,
    float weight,
    float3 tint,
    float ior,
    float2 roughness,
    float3 N,
    float3 X,
    int distribution,
    int scatter_mode,
    inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS) {
        return;
    }

    N = mx_forward_facing_normal(N, V);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    if (dot(N, L) > 0.0) {
        bsdf.throughput = 0.0;
        bsdf.response = float3(0.0);
        return;
    }

    FresnelData fd;
    float3 safeTint = max(tint, 0.0);
    if (bsdf.thickness > 0.0) {
        fd = mx_init_fresnel_dielectric_airy(ior, bsdf.thickness, bsdf.ior);
    }
    else {
        fd = mx_init_fresnel_dielectric(ior);
    }
    float3 F = mx_compute_fresnel(NdotV, fd);

    float2 safeAlpha = clamp(roughness, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(safeAlpha);

    float F0 = mx_ior_to_f0(ior);
    float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
    comp = 1.0f;
    float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * comp;
    bsdf.throughput = 1.0 - dirAlbedo * weight;
    if (scatter_mode != 0) {
        bsdf.response = mx_surface_transmission(
                            N, V, L, X, safeAlpha, distribution, fd, safeTint) *
                        occlusion * weight;

        // Add energy compensation for transmission
        float3 comp_transmission =
            mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        comp_transmission = 1.0f;
        bsdf.response *= comp_transmission;
    }
}

void mx_dielectric_bsdf_indirect(
    float3 V,
    float weight,
    float3 tint,
    float ior,
    float2 roughness,
    float3 N,
    float3 X,
    int distribution,
    int scatter_mode,
    inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS) {
        return;
    }

    N = mx_forward_facing_normal(N, V);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    float2 safeAlpha = clamp(roughness, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(safeAlpha);

    FresnelData fd;
    float3 safeTint = max(tint, 0.0);
    if (bsdf.thickness > 0.0) {
        fd = mx_init_fresnel_dielectric_airy(ior, bsdf.thickness, bsdf.ior);
    }
    else {
        fd = mx_init_fresnel_dielectric(ior);
    }
    
    float3 F = mx_compute_fresnel(NdotV, fd);
    float F0 = mx_ior_to_f0(ior);
    
    float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
    float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * comp;
    bsdf.throughput = 1.0 - dirAlbedo * weight;

    float3 Li = mx_environment_radiance(N, V, X, safeAlpha, distribution, fd);
    bsdf.response = Li * safeTint * comp * weight;
}

void mx_dielectric_bsdf(ClosureData closureData, float weight, float3 tint, float ior, float2 roughness, float thinfilm_thickness, float thinfilm_ior, float3 N, float3 X, int distribution, int scatter_mode, inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS)
    {
        return;
    }
    if (closureData.closureType != CLOSURE_TYPE_TRANSMISSION && scatter_mode == 1)
    {
        return;
    }

    if (closureData.closureType == CLOSURE_TYPE_REFLECTION)
    {
        mx_dielectric_bsdf_reflection(closureData.L, closureData.V, closureData.P, closureData.occlusion, 
                                    weight, tint, ior, roughness, N, X, distribution, scatter_mode, bsdf);
    }
    else if (closureData.closureType == CLOSURE_TYPE_TRANSMISSION)
    {
        mx_dielectric_bsdf_transmission(closureData.L, closureData.V, closureData.P, closureData.occlusion,
                                      weight, tint, ior, roughness, N, X, distribution, scatter_mode, bsdf);
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT)
    {
        mx_dielectric_bsdf_indirect(closureData.V, weight, tint, ior, roughness, N, X, distribution, scatter_mode, bsdf);
    }
}
