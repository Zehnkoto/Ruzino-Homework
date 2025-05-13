#pragma once
#include "material.h"
namespace pxr {
class Hio_OpenEXRImage;
}

USTC_CG_NAMESPACE_OPEN_SCOPE

class Shader;
using namespace pxr;

class Hio_StbImage;
class HD_USTC_CG_API Hd_USTC_CG_MaterialX : public Hd_USTC_CG_Material {
   public:
    explicit Hd_USTC_CG_MaterialX(SdfPath const& id);

    void Sync(
        HdSceneDelegate* sceneDelegate,
        HdRenderParam* renderParam,
        HdDirtyBits* dirtyBits) override;

    void ensure_shader_ready(const ShaderFactory& factory) override;

   protected:
    void BuildGPUTextures(Hd_USTC_CG_RenderParam* render_param);
    void CollectTextures(
        HdMaterialNetwork2Interface netInterface,
        HdMtlxTexturePrimvarData hdMtlxData);
    HdMaterialNetwork2Interface FetchMaterialNetwork(
        HdSceneDelegate* sceneDelegate,
        HdMaterialNetwork2& hdNetwork,
        SdfPath& materialPath,
        SdfPath& surfTerminalPath,
        HdMaterialNode2 const*& surfTerminal);

    std::string get_data_code;

   private:
    void MtlxGenerateShader(
        MaterialX::DocumentPtr mtlx_document,
        HdMaterialNetwork2Interface netInterface,
        HdMtlxTexturePrimvarData& hdMtlxData);

    static MaterialX::GenContextPtr shader_gen_context_;
    static MaterialX::DocumentPtr libraries;
    static std::once_flag shader_gen_initialized_;
    static std::mutex shadergen_mutex;
};

USTC_CG_NAMESPACE_CLOSE_SCOPE
