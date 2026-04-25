#include "path.h"

#include <random>

#include "../surfaceInteraction.h"
#include "../utils/math.hpp"
#include "../utils/sampling.hpp"

RUZINO_NAMESPACE_OPEN_SCOPE
using namespace pxr;

VtValue PathIntegrator::Li(const GfRay& ray, std::default_random_engine& random)
{
    std::uniform_real_distribution<float> uniform_dist(
        0.0f, 1.0f - std::numeric_limits<float>::epsilon());

    std::function<float()> uniform_float = [&]() {
        return uniform_dist(random);
    };

    auto color = EstimateOutGoingRadiance(ray, uniform_float, 0);

    return VtValue(GfVec3f(color[0], color[1], color[2]));
}

GfVec3f PathIntegrator::EstimateOutGoingRadiance(
    const GfRay& ray,
    const std::function<float()>& uniform_float,
    int recursion_depth)
{
    if (recursion_depth >= 50) {
        return {};
    }

    GfVec3f lightHitPos;
    Color lightEmission = IntersectLights(ray, lightHitPos);
    float lightDist = (lightEmission == Color(0.0f))
                          ? std::numeric_limits<float>::infinity()
                          : (lightHitPos - ray.GetStartPoint()).GetLength();

    SurfaceInteraction si;
    bool hitGeom = Intersect(ray, si);
    float geomDist = hitGeom ? (si.position - ray.GetStartPoint()).GetLength()
                             : std::numeric_limits<float>::infinity();

    if (!hitGeom && lightDist == std::numeric_limits<float>::infinity()) {
        if (recursion_depth == 0) {
            return IntersectDomeLight(ray);
        }
        return GfVec3f{ 0, 0, 0 };
    }

    if (lightDist < geomDist) {
        if (recursion_depth == 0)
            return lightEmission;
        return GfVec3f{ 0, 0, 0 };
    }

    if (GfDot(si.shadingNormal, ray.GetDirection()) > 0) {
        si.flipNormal();
        si.PrepareTransforms();
    }

    GfVec3f color{ 0 };
    // 1. Calculate Direct Illumination
    GfVec3f directLight = EstimateDirectLight(si, uniform_float);
    GfVec3f globalLight = GfVec3f{ 0.f };

    // Russian Roulette
    float rr_prob = 1.0f;
    if (recursion_depth > 3) {
        rr_prob = 0.8f;
        if (uniform_float() > rr_prob) {
            return directLight;
        }
    }

    // 2. Calculate Indirect Illumination
    float pdf;
    GfVec2f u(uniform_float(), uniform_float());
    GfVec3f local_wi = CosineWeightedDirection(u, pdf);

    auto basis = constructONB(si.shadingNormal);
    GfVec3f wi = (basis * local_wi).GetNormalized();

    if (pdf > 0.0f) {
        GfVec3f start_pos = si.position + si.shadingNormal * 0.001f;
        GfVec3d ray_start(start_pos[0], start_pos[1], start_pos[2]);
        GfVec3d ray_dir(wi[0], wi[1], wi[2]);
        GfRay next_ray(ray_start, ray_dir);

        GfVec3f L_i = EstimateOutGoingRadiance(
            next_ray, uniform_float, recursion_depth + 1);

        float cos_theta = std::max(0.0f, GfDot(wi, si.shadingNormal));

        Color f_r_color = si.Eval(wi);
        GfVec3f f_r_vec(f_r_color[0], f_r_color[1], f_r_color[2]);

        globalLight = GfCompMult(L_i, f_r_vec) * cos_theta / (pdf * rr_prob);
    }

    color = directLight + globalLight;

    return color;
}

RUZINO_NAMESPACE_CLOSE_SCOPE