#pragma once
#include "SceneDefines.slangh"
#include "utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "utils/Math/PackedFormats.h"
using uint = uint32_t;
#endif

struct MaterialHeader {
    uint material_type_id;
    uint material_blob_id;
};

struct MaterialDataBlob {
    uint data[128];
};

#ifndef HOST_CODE
StructuredBuffer<MaterialDataBlob> materialBlobBuffer : register(t3);
StructuredBuffer<MaterialHeader> materialHeaderBuffer : register(t4);

[[vk::binding(1, 1)]]
Texture2D t_BindlessTextures[] : register(t0, space2);

SamplerState sampler : register(s2);

#endif 