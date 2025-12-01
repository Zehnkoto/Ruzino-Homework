#pragma once
import Scene.VertexInfo;
import utils.Math.ShadingFrame;
import Utils.Math.MathHelpers;

#include "utils/Math/MathConstants.slangh"
#include "utils/random.slangh"

import pbrlib.genslang.mx_roughness_anisotropy;
import pbrlib.genslang.lib.mx_microfacet;
import pbrlib.genslang.lib.mx_microfacet_specular;
import pbrlib.genslang.lib.mx_microfacet_diffuse;

// Optimized material parameter structures for standard_surface
// Packed to reduce memory footprint by removing padding
struct StandardSurfaceMaterialParams {
    // Base parameters
    float base;
    float diffuse_roughness;
    float metalness;
    float specular;

    float3 base_color;
    float specular_roughness;

    float3 specular_color;
    float specular_IOR;

    // Specular and transmission
    float specular_anisotropy;
    float specular_rotation;
    float transmission;
    float transmission_depth;

    float3 transmission_color;
    float transmission_scatter_anisotropy;

    float3 transmission_scatter;
    float transmission_dispersion;

    float transmission_extra_roughness;
    float subsurface;
    float subsurface_scale;
    float subsurface_anisotropy;

    // Subsurface
    float3 subsurface_color;
    float sheen;

    float3 subsurface_radius;
    float sheen_roughness;

    // Sheen
    float3 sheen_color;
    float coat;

    // Coat
    float3 coat_color;
    float coat_roughness;

    float coat_anisotropy;
    float coat_rotation;
    float coat_IOR;
    float coat_affect_color;

    float3 coat_normal;
    float coat_affect_roughness;

    float thin_film_thickness;
    float thin_film_IOR;
    float emission;
    uint thin_walled;

    // Emission and opacity
    float3 emission_color;
    float3 opacity;

    // Normals and tangents
    float3 normal;
    float3 tangent;

    // Total: 256 bytes = 16 * float4 (reduced from 288 bytes by removing padding)
};

// Optimized material parameter structure for preview_surface
struct PreviewSurfaceMaterialParams {
    // Colors
    float3 diffuseColor;
    float metallic;

    float3 emissiveColor;
    float roughness;

    float3 specularColor;
    float clearcoat;

    // Properties
    float clearcoatRoughness;
    float opacity;
    int opacityMode;
    float opacityThreshold;

    float ior;
    int useSpecularWorkflow;
    float displacement;
    float occlusion;

    float3 normal;

    // Total: 80 bytes = 5 * float4 (reduced from 112 bytes)
};

// Pack standard_surface parameters into optimized structure
// This function passes through original values without modification
StandardSurfaceMaterialParams packStandardSurfaceMaterialParams(
    float base,
    float3 base_color,
    float diffuse_roughness,
    float metalness,
    float specular,
    float3 specular_color,
    float specular_roughness,
    float specular_IOR,
    float specular_anisotropy,
    float specular_rotation,
    float transmission,
    float3 transmission_color,
    float transmission_depth,
    float3 transmission_scatter,
    float transmission_scatter_anisotropy,
    float transmission_dispersion,
    float transmission_extra_roughness,
    float subsurface,
    float3 subsurface_color,
    float3 subsurface_radius,
    float subsurface_scale,
    float subsurface_anisotropy,
    float sheen,
    float3 sheen_color,
    float sheen_roughness,
    float coat,
    float3 coat_color,
    float coat_roughness,
    float coat_anisotropy,
    float coat_rotation,
    float coat_IOR,
    float3 coat_normal,
    float coat_affect_color,
    float coat_affect_roughness,
    float thin_film_thickness,
    float thin_film_IOR,
    float emission,
    float3 emission_color,
    float3 opacity,
    uint thin_walled,
    float3 normal,
    float3 tangent)
{
    StandardSurfaceMaterialParams params;
    
    // Store original values without pre-multiplication
    params.base = base;
    params.base_color = base_color;
    params.diffuse_roughness = diffuse_roughness;
    params.metalness = metalness;
    params.specular = specular;
    params.specular_color = specular_color;
    params.specular_roughness = specular_roughness;
    params.specular_IOR = specular_IOR;
    params.specular_anisotropy = specular_anisotropy;
    params.specular_rotation = specular_rotation;
    
    params.transmission = transmission;
    params.transmission_color = transmission_color;
    params.transmission_depth = transmission_depth;
    params.transmission_scatter = transmission_scatter;
    params.transmission_scatter_anisotropy = transmission_scatter_anisotropy;
    params.transmission_dispersion = transmission_dispersion;
    params.transmission_extra_roughness = transmission_extra_roughness;
    
    params.subsurface = subsurface;
    params.subsurface_color = subsurface_color;
    params.subsurface_radius = subsurface_radius;
    params.subsurface_scale = subsurface_scale;
    params.subsurface_anisotropy = subsurface_anisotropy;
    
    params.sheen = sheen;
    params.sheen_color = sheen_color;
    params.sheen_roughness = sheen_roughness;
    
    params.coat = coat;
    params.coat_color = coat_color;
    params.coat_roughness = coat_roughness;
    params.coat_anisotropy = coat_anisotropy;
    params.coat_rotation = coat_rotation;
    params.coat_IOR = coat_IOR;
    params.coat_normal = coat_normal;
    params.coat_affect_color = coat_affect_color;
    params.coat_affect_roughness = coat_affect_roughness;
    
    params.thin_film_thickness = thin_film_thickness;
    params.thin_film_IOR = thin_film_IOR;
    params.emission = emission;
    params.thin_walled = thin_walled;
    
    params.emission_color = emission_color;
    params.opacity = opacity;
    
    params.normal = normal;
    params.tangent = tangent;
    
    return params;
}
