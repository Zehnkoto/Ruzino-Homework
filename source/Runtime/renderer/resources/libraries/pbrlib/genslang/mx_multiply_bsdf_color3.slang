import pbrlib.genslang.lib.mx_closure_type;

void mx_multiply_bsdf_color3(ClosureData closureData, BSDF in1, float3 in2, out BSDF result)
{
    float3 tint = clamp(in2, 0.0, 1.0);
    result.response = in1.response * tint;
    result.throughput = in1.throughput * tint;
}
