import stdlib.genslang.mx_bsdf;
import stdlib.genslang.lib.mx_math;
import pbrlib.genslang.lib.mx_closure_type;
import pbrlib.genslang.lib.mx_environment_prefilter;
import pbrlib.genslang.lib.mx_microfacet;

// We fake diffuse transmission by using diffuse reflection from the opposite
// side. So this BTDF is really a BRDF.
void mx_translucent_bsdf_reflection(
    float3 L,
    float3 V,
    float3 P,
    float occlusion,
    float weight,
    float3 color,
    float3 normal,
    inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);

    // Invert normal since we're transmitting light from the other side
    float NdotL = dot(L, -normal);
    if (NdotL <= 0.0 || weight < M_FLOAT_EPS) {
        return;
    }

    bsdf.response = color * weight * NdotL * M_PI_INV;
}

void mx_translucent_bsdf_indirect(
    float3 V,
    float weight,
    float3 color,
    float3 normal,
    inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);

    if (weight < M_FLOAT_EPS) {
        return;
    }

    // Invert normal since we're transmitting light from the other side
    float3 Li = mx_environment_irradiance(-normal);
    bsdf.response = Li * color * weight;
}

void mx_translucent_bsdf(ClosureData closureData, float weight, float3 color, float3 N, inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);

    if (weight < M_FLOAT_EPS)
    {
        return;
    }

    if (closureData.closureType == CLOSURE_TYPE_REFLECTION)
    {
        mx_translucent_bsdf_reflection(closureData.L, closureData.V, closureData.P, closureData.occlusion,
                                     weight, color, N, bsdf);
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT)
    {
        mx_translucent_bsdf_indirect(closureData.V, weight, color, N, bsdf);
    }
}
