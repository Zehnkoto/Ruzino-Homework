import pbrlib.genslang.lib.mx_closure_type;

void mx_mix_bsdf(ClosureData closureData, BSDF fg, BSDF bg, float mixValue, out BSDF result)
{
    result.response = lerp(bg.response, fg.response, mixValue);
    result.throughput = lerp(bg.throughput, fg.throughput, mixValue);
}
