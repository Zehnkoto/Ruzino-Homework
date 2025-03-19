#pragma once
#include "Logger/Logger.h"
#include "MaterialX/SlangShaderGenerator.h"
#include "api.h"
#include "map.h"
#include "pxr/imaging/garch/glApi.h"
#include "pxr/imaging/hd/material.h"
#include "pxr/imaging/hd/materialNetwork2Interface.h"
#include "pxr/imaging/hdMtlx/hdMtlx.h"
#include "pxr/imaging/hio/image.h"
#include "renderParam.h"
#include "renderer/program_vars.hpp"

namespace pxr {
class Hio_OpenEXRImage;

}

USTC_CG_NAMESPACE_OPEN_SCOPE
class Shader;
using namespace pxr;

class Hio_StbImage;
class HD_USTC_CG_API Hd_USTC_CG_Material : public HdMaterial {
   public:
    explicit Hd_USTC_CG_Material(SdfPath const& id);

    void Sync(
        HdSceneDelegate* sceneDelegate,
        HdRenderParam* renderParam,
        HdDirtyBits* dirtyBits) override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;

    void Finalize(HdRenderParam* renderParam) override;

    std::vector<TextureHandle> GetTextures() const
    {
        std::vector<TextureHandle> textures;
        for (const auto& tex : textureResources) {
            textures.push_back(tex.second.texture);
        }
        return textures;
    }

    unsigned GetMaterialLocation() const
    {
        return material_data_handle->index();
    }

    std::shared_ptr<ProgramVars> GetShader(
        const ShaderFactory& factory,
        ResourceAllocator& allocator);

   private:
    HdMaterialNetwork2 surfaceNetwork;
    MaterialX::ShaderPtr shader;
    ProgramHandle program = nullptr;

    std::unordered_map<std::string, std::string> texturePaths;

    struct TextureResource {
        std::string filePath;
        HioImageSharedPtr image;
        nvrhi::TextureHandle texture;
        DescriptorHandle descriptor;
    };

    std::unordered_map<std::string, TextureResource> textureResources;

    void CollectTextures(
        HdMaterialNetwork2Interface netInterface,
        HdMtlxTexturePrimvarData hdMtlxData);

    void LoadTextures();

    void BuildGPUTextures(Hd_USTC_CG_RenderParam* render_param);

    void MtlxGenerateShader(
        HdMaterialNetwork2 hdNetwork,
        SdfPath materialPath,
        HdMaterialNetwork2Interface netInterface,
        SdfPath surfTerminalPath,
        HdMaterialNode2 const* surfTerminal,
        HdMtlxTexturePrimvarData& hdMtlxData);

    HdMaterialNetwork2Interface FetchMaterialNetwork(
        HdSceneDelegate* sceneDelegate,
        HdMaterialNetwork2& hdNetwork,
        SdfPath& materialPath,
        SdfPath& surfTerminalPath,
        HdMaterialNode2 const*& surfTerminal);

    DeviceMemoryPool<MaterialData>::MemoryHandle material_data_handle;

    static MaterialX::GenContextPtr shader_gen_context_;
    static MaterialX::DocumentPtr libraries;

    static std::once_flag shader_gen_initialized_;
};

USTC_CG_NAMESPACE_CLOSE_SCOPE