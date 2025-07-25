import pbrlib.genslang.lib.mx_closure_type;
import pbrlib.genslang.lib.mx_microfacet_diffuse;
import pbrlib.genslang.lib.mx_environment_prefilter;

void mx_oren_nayar_diffuse_bsdf(
    ClosureData closureData,
    float weight,
    float3 color,
    float roughness,
    float3 N,
    bool energy_compensation,
    inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);

    if (weight < M_FLOAT_EPS) {
        return;
    }

    float3 V = closureData.V;
    float3 L = closureData.L;

    N = mx_forward_facing_normal(N, V);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    if (closureData.closureType == CLOSURE_TYPE_REFLECTION) {
        float NdotL = clamp(dot(N, L), M_FLOAT_EPS, 1.0);
        float LdotV = clamp(dot(L, V), M_FLOAT_EPS, 1.0);

        float3 diffuse =
            energy_compensation
                ? mx_oren_nayar_compensated_diffuse(
                      NdotV, NdotL, LdotV, roughness, color)
                : mx_oren_nayar_diffuse(NdotV, NdotL, LdotV, roughness) * color;
        bsdf.response =
            diffuse * closureData.occlusion * weight * NdotL * M_PI_INV;
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT) {
        float3 diffuse =
            energy_compensation
                ? mx_oren_nayar_compensated_diffuse_dir_albedo(
                      NdotV, roughness, color)
                : mx_oren_nayar_diffuse_dir_albedo(NdotV, roughness) * color;
        float3 Li = mx_environment_irradiance(N);
        bsdf.response = Li * diffuse * weight;
    }
}
