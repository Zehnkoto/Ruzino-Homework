#include "material.h"

#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/materialNetwork2Interface.h>
#include <pxr/imaging/hio/image.h>

#include "nvrhi/nvrhi.h"
#include "pxr/base/arch/fileSystem.h"
#include "pxr/imaging/hd/changeTracker.h"
#include "pxr/imaging/hd/sceneDelegate.h"
#include "pxr/usd/ar/resolver.h"
#include "pxr/usd/sdr/registry.h"
USTC_CG_NAMESPACE_OPEN_SCOPE

std::mutex Hd_USTC_CG_Material::texture_mutex;
std::mutex Hd_USTC_CG_Material::material_data_handle_mutex;

Hd_USTC_CG_Material::Hd_USTC_CG_Material(SdfPath const& id) : HdMaterial(id)
{
}

HdMaterialNetwork2Interface Hd_USTC_CG_Material::FetchNetInterface(
    HdSceneDelegate* sceneDelegate,
    HdMaterialNetwork2& hdNetwork,
    SdfPath& materialPath)
{
    VtValue material = sceneDelegate->GetMaterialResource(GetId());
    HdMaterialNetworkMap networkMap = material.Get<HdMaterialNetworkMap>();

    bool isVolume;
    hdNetwork = HdConvertToHdMaterialNetwork2(networkMap, &isVolume);

    materialPath = GetId();

    HdMaterialNetwork2Interface netInterface =
        HdMaterialNetwork2Interface(materialPath, &hdNetwork);
    return netInterface;
}

HdDirtyBits Hd_USTC_CG_Material::GetInitialDirtyBitsMask() const
{
    return HdChangeTracker::AllDirty;
}

void Hd_USTC_CG_Material::Finalize(HdRenderParam* renderParam)
{
    HdMaterial::Finalize(renderParam);
}

void Hd_USTC_CG_Material::ensure_material_data_handle(
    Hd_USTC_CG_RenderParam* render_param)
{
    std::lock_guard<std::mutex> lock(material_data_handle_mutex);
    if (!material_data_handle) {
        if (!render_param) {
            throw std::runtime_error("Render param is null.");
        }

        material_header_handle =
            render_param->InstanceCollection->material_header_pool.allocate(1);

        material_data_handle =
            render_param->InstanceCollection->material_pool.allocate(1);

        MaterialHeader header;
        header.material_blob_id = material_data_handle->index();
        header.material_type_id = material_header_handle->index();
        material_header_handle->write_data(&header);
    }
}

unsigned Hd_USTC_CG_Material::GetMaterialLocation() const
{
    if (!material_data_handle) {
        return -1;
    }
    return material_header_handle->index();
}

// HLSL callable shader
std::string Hd_USTC_CG_Material::slang_source_code = R"(
import Scene.VertexInfo;

struct CallableData
{
    float4 color;
    float3 L;
    float3 V;
    uint materialBlobID;
    VertexInfo vertexInfo;
};

[shader("callable")]
void $getColor(inout CallableData data)
{
    eval(data.color, data.L, data.V, data.materialBlobID, data.vertexInfo);
}

)";

// HLSL callable shader
static std::string slang_source_code_fallback = R"(

struct VertexData
{
    float foo;
};

import Scene.VertexInfo;
// ConstantBuffer<float> cb;

struct CallableData
{
    float4 color;
    float3 L;
    float3 V;
    uint materialBlobID;
    VertexInfo vertexInfo;
};

[shader("callable")]
void $getColor(inout CallableData data)
{
    data.color = float4(0, 1, 0, 1);
}

)";
void Hd_USTC_CG_Material::ensure_shader_ready(const ShaderFactory& factory)
{
    // Use fallback shader if no source is available
    if (material_name.empty()) {
        material_name = "fallback";
    }
    final_shader_source = slang_source_code_fallback;
}

std::string Hd_USTC_CG_Material::GetShader(const ShaderFactory& factory)
{
    ensure_shader_ready(factory);
    return final_shader_source;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
