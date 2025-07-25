void mx_generalized_schlick_edf(
    ClosureData closureData,
    float3 color0,
    float3 color90,
    float exponent,
    EDF base,
    out EDF result)
{
    if (closureData.closureType == CLOSURE_TYPE_EMISSION) {
        float3 N = mx_forward_facing_normal(closureData.N, closureData.V);
        float NdotV = clamp(dot(N, closureData.V), M_FLOAT_EPS, 1.0);
        float3 f = mx_fresnel_schlick(NdotV, color0, color90, exponent);
        result = base * f;
    }
}
