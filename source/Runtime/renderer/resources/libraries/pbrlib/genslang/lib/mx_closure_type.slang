// These are defined based on the HwShaderGenerator::ClosureContextType enum
// if that changes - these need to be updated accordingly.

static constexpr int CLOSURE_TYPE_DEFAULT = 0;
static constexpr int CLOSURE_TYPE_REFLECTION = 1;
static constexpr int CLOSURE_TYPE_TRANSMISSION = 2;
static constexpr int CLOSURE_TYPE_INDIRECT = 3;
static constexpr int CLOSURE_TYPE_EMISSION = 4;

struct ClosureData {
    int closureType;
    float3 L;
    float3 V;
    float3 N;
    float3 P;
    float occlusion;
};
