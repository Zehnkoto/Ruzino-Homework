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
    if (recursion_depth >= 50)
        return {};

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
        if (recursion_depth == 0)
            return IntersectDomeLight(ray);
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

    GfVec3f directLight = EstimateDirectLight(si, uniform_float);

    float rr_prob = 1.0f;
    if (recursion_depth > 3) {
        rr_prob = 0.8f;
        if (uniform_float() > rr_prob)
            return directLight;
    }

    GfVec3f wi_world;
    float pdf;
    Color f_r = si.Sample(wi_world, pdf, uniform_float);

    GfVec3f globalLight = GfVec3f{ 0.f };

    if (pdf > 1e-6f) {
        GfVec3f start_pos = si.position + si.shadingNormal * 0.001f;
        GfRay next_ray(
            GfVec3d(start_pos[0], start_pos[1], start_pos[2]),
            GfVec3d(wi_world[0], wi_world[1], wi_world[2]));

        GfVec3f L_i = EstimateOutGoingRadiance(
            next_ray, uniform_float, recursion_depth + 1);

        float cos_theta = std::max(0.0f, GfDot(wi_world, si.shadingNormal));
        globalLight = GfCompMult(L_i, f_r) * cos_theta / (pdf * rr_prob);
    }

    return directLight + globalLight;
}

RUZINO_NAMESPACE_CLOSE_SCOPE