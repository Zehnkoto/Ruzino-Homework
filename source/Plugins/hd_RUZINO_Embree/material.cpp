// #define __GNUC__

#include "material.h"

#include <spdlog/spdlog.h>

#include "RHI/internal/map.h"
#include "pxr/imaging/hd/sceneDelegate.h"
#include "pxr/imaging/hio/image.h"
#include "pxr/usd/sdr/shaderNode.h"
#include "pxr/usd/usd/tokens.h"
#include "pxr/usdImaging/usdImaging/tokens.h"
#include "renderParam.h"
#include "texture.h"
#include "utils/sampling.hpp"

RUZINO_NAMESPACE_OPEN_SCOPE
using namespace pxr;

HdMaterialNode2 Hd_RUZINO_Material::get_input_connection(
    HdMaterialNetwork2 surfaceNetwork,
    std::map<TfToken, std::vector<HdMaterialConnection2>>::value_type&
        input_connection)
{
    HdMaterialNode2 upstream;
    assert(input_connection.second.size() == 1);
    upstream = surfaceNetwork.nodes[input_connection.second[0].upstreamNode];
    return upstream;
}

Hd_RUZINO_Material::MaterialRecord Hd_RUZINO_Material::SampleMaterialRecord(
    GfVec2f texcoord)
{
    MaterialRecord ret;
    if (diffuseColor.image) {
        auto val4 = diffuseColor.image->Evaluate(texcoord);
        ret.diffuseColor = { val4[0], val4[1], val4[2] };
    }
    else {
        ret.diffuseColor = diffuseColor.value.Get<GfVec3f>();
    }

    if (roughness.image) {
        auto val4 = roughness.image->Evaluate(texcoord);
        ret.roughness = val4[1];
    }
    else {
        ret.roughness = roughness.value.Get<float>();
    }

    if (ior.image) {
        auto val4 = ior.image->Evaluate(texcoord);
        ret.ior = val4[0];
    }
    else {
        ret.ior = ior.value.Get<float>();
    }

    if (metallic.image) {
        auto val4 = metallic.image->Evaluate(texcoord);
        ret.metallic = val4[2];
    }
    else {
        ret.metallic = metallic.value.Get<float>();
    }

    return ret;
}

void Hd_RUZINO_Material::TryLoadTexture(
    const char* name,
    InputDescriptor& descriptor,
    HdMaterialNode2& usd_preview_surface)
{
    for (auto&& input_connection : usd_preview_surface.inputConnections) {
        if (input_connection.first == TfToken(name)) {
            spdlog::info(
                "Loading texture: " + input_connection.first.GetString());
            auto texture_node =
                get_input_connection(surfaceNetwork, input_connection);
            assert(texture_node.nodeTypeId == UsdImagingTokens->UsdUVTexture);

            auto assetPath =
                texture_node.parameters[TfToken("file")].Get<SdfAssetPath>();

            HioImage::SourceColorSpace colorSpace;

            if (texture_node.parameters[TfToken("sourceColorSpace")] ==
                TfToken("sRGB")) {
                colorSpace = HioImage::SRGB;
            }
            else {
                colorSpace = HioImage::Raw;
            }

            descriptor.image =
                std::make_unique<Texture2D>(assetPath, colorSpace);
            if (!descriptor.image->isValid()) {
                descriptor.image = nullptr;
            }
            descriptor.wrapS =
                texture_node.parameters[TfToken("wrapS")].Get<TfToken>();
            descriptor.wrapT =
                texture_node.parameters[TfToken("wrapT")].Get<TfToken>();

            HdMaterialNode2 st_read_node;
            for (auto&& st_read_connection : texture_node.inputConnections) {
                st_read_node =
                    get_input_connection(surfaceNetwork, st_read_connection);
            }

            assert(
                st_read_node.nodeTypeId ==
                UsdImagingTokens->UsdPrimvarReader_float2);
            descriptor.uv_primvar_name =
                st_read_node.parameters[TfToken("varname")].Get<TfToken>();
            if (descriptor.uv_primvar_name.empty()) {
                descriptor.uv_primvar_name =
                    st_read_node.parameters[TfToken("varname")]
                        .Get<std::string>();
            }
        }
    }
}

void Hd_RUZINO_Material::TryLoadParameter(
    const char* name,
    InputDescriptor& descriptor,
    HdMaterialNode2& usd_preview_surface)
{
    for (auto&& parameter : usd_preview_surface.parameters) {
        if (parameter.first == name) {
            descriptor.value = parameter.second;
            spdlog::info("Loading parameter: " + parameter.first.GetString());
        }
    }
}

#define INPUT_LIST                                                            \
    diffuseColor, specularColor, emissiveColor, displacement, opacity,        \
        opacityThreshold, roughness, metallic, clearcoat, clearcoatRoughness, \
        occlusion, normal, ior

#define TRY_LOAD(INPUT)                                 \
    TryLoadTexture(#INPUT, INPUT, usd_preview_surface); \
    TryLoadParameter(#INPUT, INPUT, usd_preview_surface);

#define NAME_IT(INPUT) INPUT.input_name = TfToken(#INPUT);

Hd_RUZINO_Material::Hd_RUZINO_Material(const SdfPath& id) : HdMaterial(id)
{
    spdlog::info("Creating material " + id.GetString());
    diffuseColor.value = VtValue(GfVec3f(0.8f));
    roughness.value = VtValue(0.8f);

    metallic.value = VtValue(0.0f);
    normal.value = VtValue(GfVec3f(0.5, 0.5, 1.0));
    ior.value = VtValue(1.5f);

    MACRO_MAP(NAME_IT, INPUT_LIST);
}

void Hd_RUZINO_Material::Sync(
    HdSceneDelegate* sceneDelegate,
    HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits)
{
    static_cast<Hd_RUZINO_RenderParam*>(renderParam)->AcquireSceneForEdit();

    VtValue vtMat = sceneDelegate->GetMaterialResource(GetId());
    if (vtMat.IsHolding<HdMaterialNetworkMap>()) {
        const HdMaterialNetworkMap& hdNetworkMap =
            vtMat.UncheckedGet<HdMaterialNetworkMap>();
        if (!hdNetworkMap.terminals.empty() && !hdNetworkMap.map.empty()) {
            spdlog::info("Loaded a material");

            surfaceNetwork = HdConvertToHdMaterialNetwork2(hdNetworkMap);

            assert(surfaceNetwork.terminals.size() == 1);

            auto terminal =
                surfaceNetwork.terminals[HdMaterialTerminalTokens->surface];

            auto usd_preview_surface =
                surfaceNetwork.nodes[terminal.upstreamNode];
            assert(
                usd_preview_surface.nodeTypeId ==
                UsdImagingTokens->UsdPreviewSurface);

            MACRO_MAP(TRY_LOAD, INPUT_LIST)
        }
    }
    else {
        spdlog::info("Not loaded a material");
    }
    *dirtyBits = Clean;
}

HdDirtyBits Hd_RUZINO_Material::GetInitialDirtyBitsMask() const
{
    return AllDirty;
}

#define requireTexCoord(INPUT)            \
    if (!INPUT.uv_primvar_name.empty()) { \
        return INPUT.uv_primvar_name;     \
    }

std::string Hd_RUZINO_Material::requireTexcoordName()
{
    MACRO_MAP(requireTexCoord, INPUT_LIST)
    return {};
}

void Hd_RUZINO_Material::Finalize(HdRenderParam* renderParam)
{
    static_cast<Hd_RUZINO_RenderParam*>(renderParam)->AcquireSceneForEdit();
    HdMaterial::Finalize(renderParam);
}

Color Hd_RUZINO_Material::Sample(
    const GfVec3f& wo,
    GfVec3f& wi,
    float& pdf,
    GfVec2f texcoord,
    const std::function<float()>& uniform_float)
{
    auto record = SampleMaterialRecord(texcoord);
    auto roughness = std::clamp(record.roughness, 0.04f, 1.0f);
    auto ior = record.ior;
    auto metallic = record.metallic;
    GfVec3f diffuseColor = record.diffuseColor;

    float DielectricSpecular = pow((1.0f - ior) / (1.0f + ior), 2.0f);
    GfVec3f diffuseAlbedo =
        (1.0f - metallic) * diffuseColor * (1.0f - DielectricSpecular);
    GfVec3f F0 = GfVec3f(DielectricSpecular) * (1.0f - metallic) +
                 metallic * diffuseColor;
    GfVec3f F =
        F0 + (GfVec3f(1.0f) - F0) * pow(1.0f - std::max(0.0f, wo[2]), 5.0f);

    float diffuseMax = std::max(
        std::max(diffuseAlbedo[0], diffuseAlbedo[1]), diffuseAlbedo[2]);
    float specularMax = std::max(std::max(F[0], F[1]), F[2]);
    float sample_diffuse_bar = diffuseMax / (diffuseMax + specularMax + 1e-6f);

    auto sample2D = GfVec2f{ uniform_float(), uniform_float() };

    if (uniform_float() < sample_diffuse_bar) {
        float r = sqrt(sample2D[0]);
        float theta = 2.0f * M_PI * sample2D[1];
        wi = GfVec3f(
            r * cos(theta),
            r * sin(theta),
            sqrt(std::max(0.0f, 1.0f - sample2D[0])));
    }
    else {
        float alpha = roughness * roughness;
        float alpha2 = alpha * alpha;
        float phi = 2.0f * M_PI * sample2D[0];
        float cosTheta = sqrt(std::max(
            0.0f,
            (1.0f - sample2D[1]) / (1.0f + (alpha2 - 1.0f) * sample2D[1])));
        float sinTheta = sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));

        GfVec3f H(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
        wi = 2.0f * GfDot(wo, H) * H - wo;
    }

    if (wi[2] <= 0.0f || wo[2] <= 0.0f) {
        pdf = 0.0f;
        return Color(0.0f);
    }

    pdf = Pdf(wi, wo, texcoord);
    return Eval(wi, wo, texcoord);
}

Color Hd_RUZINO_Material::Eval(GfVec3f wi, GfVec3f wo, GfVec2f texcoord)
{
    if (wi[2] <= 0.0f || wo[2] <= 0.0f)
        return Color(0.0f);

    auto record = SampleMaterialRecord(texcoord);
    auto roughness = std::clamp(record.roughness, 0.04f, 1.0f);
    auto ior = record.ior;
    auto metallic = record.metallic;
    GfVec3f diffuseColor = record.diffuseColor;

    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    GfVec3f H = (wo + wi).GetNormalized();
    float NdotH = std::max(0.0f, H[2]);
    float VdotH = std::max(0.0f, GfDot(wo, H));

    float denom = (NdotH * NdotH * (alpha2 - 1.0f) + 1.0f);
    float D = alpha2 / (M_PI * denom * denom);

    float DielectricSpecular = pow((1.0f - ior) / (1.0f + ior), 2.0f);
    GfVec3f diffuseAlbedo =
        (1.0f - metallic) * diffuseColor * (1.0f - DielectricSpecular);
    GfVec3f F0 = GfVec3f(DielectricSpecular) * (1.0f - metallic) +
                 metallic * diffuseColor;
    GfVec3f F = F0 + (GfVec3f(1.0f) - F0) * pow(1.0f - VdotH, 5.0f);

    auto smith_g1 = [&](const GfVec3f& v) {
        float NdotV = std::max(0.0f, v[2]);
        return 2.0f * NdotV /
               (NdotV + sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV));
    };
    float G = smith_g1(wi) * smith_g1(wo);

    Color specular = F * (D * G / (4.0f * wi[2] * wo[2]));
    Color diffuse = diffuseAlbedo / M_PI;

    return specular + diffuse;
}

float Hd_RUZINO_Material::Pdf(GfVec3f wi, GfVec3f wo, GfVec2f texcoord)
{
    if (wi[2] <= 0.0f || wo[2] <= 0.0f)
        return 0.0f;

    auto record = SampleMaterialRecord(texcoord);
    auto roughness = std::clamp(record.roughness, 0.04f, 1.0f);
    auto ior = record.ior;
    auto metallic = record.metallic;
    GfVec3f diffuseColor = record.diffuseColor;

    float DielectricSpecular = pow((1.0f - ior) / (1.0f + ior), 2.0f);
    GfVec3f diffuseAlbedo =
        (1.0f - metallic) * diffuseColor * (1.0f - DielectricSpecular);
    GfVec3f F0 = GfVec3f(DielectricSpecular) * (1.0f - metallic) +
                 metallic * diffuseColor;
    GfVec3f F =
        F0 + (GfVec3f(1.0f) - F0) * pow(1.0f - std::max(0.0f, wo[2]), 5.0f);

    float diffuseMax = std::max(
        std::max(diffuseAlbedo[0], diffuseAlbedo[1]), diffuseAlbedo[2]);
    float specularMax = std::max(std::max(F[0], F[1]), F[2]);
    float sample_diffuse_bar = diffuseMax / (diffuseMax + specularMax + 1e-6f);

    GfVec3f H = (wo + wi).GetNormalized();
    float NdotH = std::max(0.0f, H[2]);
    float VdotH = std::max(0.0f, GfDot(wo, H));

    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH * (alpha2 - 1.0f) + 1.0f);
    float D = alpha2 / (M_PI * denom * denom);

    float ggxPdf = D * NdotH;
    float specPdf = ggxPdf / (4.0f * VdotH + 1e-6f);
    float diffPdf = wi[2] / M_PI;

    return sample_diffuse_bar * diffPdf + (1.0f - sample_diffuse_bar) * specPdf;
}

RUZINO_NAMESPACE_CLOSE_SCOPE