import stdlib.genslang.mx_bsdf;
import pbrlib.genslang.lib.mx_microfacet_specular;
import pbrlib.genslang.lib.mx_environment_prefilter;
import pbrlib.genslang.lib.mx_transmission_refract;
__exported import pbrlib.genslang.lib.mx_closure_type;

void mx_generalized_schlick_bsdf(
    ClosureData closureData,
    float weight,
    float3 color0,
    float3 color82,
    float3 color90,
    float exponent,
    float2 roughness,
    float thinfilm_thickness,
    float thinfilm_ior,
    float3 N,
    float3 X,
    int distribution,
    int scatter_mode,
    inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS) {
        return;
    }
    if (closureData.closureType != CLOSURE_TYPE_TRANSMISSION &&
        scatter_mode == 1) {
        return;
    }

    float3 V = closureData.V;
    float3 L = closureData.L;

    N = mx_forward_facing_normal(N, V);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    float3 safeColor0 = max(color0, 0.0);
    float3 safeColor82 = max(color82, 0.0);
    float3 safeColor90 = max(color90, 0.0);
    FresnelData fd = mx_init_fresnel_schlick(
        safeColor0,
        safeColor82,
        safeColor90,
        exponent,
        thinfilm_thickness,
        thinfilm_ior);

    float2 safeAlpha = clamp(roughness, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(safeAlpha);

    if (closureData.closureType == CLOSURE_TYPE_REFLECTION) {
        X = normalize(X - dot(X, N) * N);
        float3 Y = cross(N, X);
        float3 H = normalize(L + V);

        float NdotL = clamp(dot(N, L), M_FLOAT_EPS, 1.0);
        float VdotH = clamp(dot(V, H), M_FLOAT_EPS, 1.0);

        float3 Ht = float3(dot(H, X), dot(H, Y), dot(H, N));

        float3 F = mx_compute_fresnel(VdotH, fd);
        float D = mx_ggx_NDF(Ht, safeAlpha);
        float G = mx_ggx_smith_G2(NdotL, NdotV, avgAlpha);

        float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        float3 dirAlbedo =
            mx_ggx_dir_albedo(NdotV, avgAlpha, safeColor0, safeColor90) * comp;
        float avgDirAlbedo = dot(dirAlbedo, float3(1.0 / 3.0));
        bsdf.throughput = float3(1.0 - avgDirAlbedo * weight);

        bsdf.response =
            D * F * G * comp * closureData.occlusion * weight / (4.0 * NdotV);
    }
    else if (closureData.closureType == CLOSURE_TYPE_TRANSMISSION) {
        float3 F = mx_compute_fresnel(NdotV, fd);

        float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        float3 dirAlbedo =
            mx_ggx_dir_albedo(NdotV, avgAlpha, safeColor0, safeColor90) * comp;
        float avgDirAlbedo = dot(dirAlbedo, float3(1.0 / 3.0));
        bsdf.throughput = float3(1.0 - avgDirAlbedo * weight);

        if (scatter_mode != 0) {
            float avgF0 = dot(safeColor0, float3(1.0 / 3.0));
            fd.ior = float3(mx_f0_to_ior(avgF0));
            bsdf.response =
                mx_surface_transmission(
                    N, V, L, X, safeAlpha, distribution, fd, float3(1.0)) *
                weight;
        }
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT) {
        float3 F = mx_compute_fresnel(NdotV, fd);

        float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        float3 dirAlbedo =
            mx_ggx_dir_albedo(NdotV, avgAlpha, safeColor0, safeColor90) * comp;
        float avgDirAlbedo = dot(dirAlbedo, float3(1.0 / 3.0));
        bsdf.throughput = float3(1.0 - avgDirAlbedo * weight);

        float3 Li = mx_environment_radiance(N, V, X, safeAlpha, distribution, fd);
        bsdf.response = Li * comp * weight;
    }
}
