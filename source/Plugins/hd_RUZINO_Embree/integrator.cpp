#include "integrator.h"

#include <chrono>
#include <functional>
#include <random>

#include "config.h"
#include "context.h"
#include "light.h"
#include "pxr/base/gf/matrix3f.h"
#include "pxr/base/tf/hash.h"
#include "pxr/base/tf/hashmap.h"
#include "pxr/base/work/loops.h"
#include "pxr/imaging/hd/rprim.h"
#include "pxr/pxr.h"
#include "renderParam.h"
#include "surfaceInteraction.h"

RUZINO_NAMESPACE_OPEN_SCOPE
using namespace pxr;
static unsigned channel(VtValue val)
{
    if (val.CanCast<float>()) {
        return 1;
    }
    else if (val.CanCast<GfVec4f>()) {
        return 4;
    }
    else if (val.CanCast<GfVec3f>()) {
        return 3;
    }

    TF_CODING_ERROR("val must can cast to those types");
    return 0;
}
static void _PopulateRay(
    RTCRay* ray,
    const GfVec3d& origin,
    const GfVec3d& dir,
    float nearest,
    float tfar = std::numeric_limits<float>::infinity())
{
    ray->org_x = origin[0];
    ray->org_y = origin[1];
    ray->org_z = origin[2];
    ray->tnear = nearest;

    ray->dir_x = dir[0];
    ray->dir_y = dir[1];
    ray->dir_z = dir[2];
    ray->time = 0.0f;

    ray->tfar = tfar;
    ray->mask = -1;
}

static void _PopulateRayHit(
    RTCRayHit* rayHit,
    const GfVec3d& origin,
    const GfVec3d& dir,
    float nearest)
{
    _PopulateRay(&rayHit->ray, origin, dir, nearest);
    rayHit->hit.primID = RTC_INVALID_GEOMETRY_ID;
    rayHit->hit.geomID = RTC_INVALID_GEOMETRY_ID;
}

Color Integrator::SampleLights(
    const GfVec3f& pos,
    GfVec3f& dir,
    GfVec3f& sampled_light_pos,
    float& pdf,
    const std::function<float()>& uniform_float)
{
    auto N = render_param->lights->size();
    if (N == 0) {
        pdf = 0;
        return Color{ 0 };
    }

    float select_light_pdf = 1.0f / float(N);
    auto light_id = size_t(std::floor(uniform_float() * N));
    auto light = (*render_param->lights)[light_id];

    float sample_light_pdf;
    auto color = light->Sample(
        pos, dir, sampled_light_pos, sample_light_pdf, uniform_float);
    pdf = sample_light_pdf * select_light_pdf;
    return color;
}

Color Integrator::IntersectLights(const GfRay& ray, GfVec3f& intersectPos)
{
    float currentDepth = std::numeric_limits<float>::infinity();
    Color color{ 0, 0, 0 };
    for (auto light : (*render_param->lights)) {
        float depth = std::numeric_limits<float>::infinity();
        auto intersected_radiance = light->Intersect(ray, depth);
        if (depth < currentDepth) {
            currentDepth = depth;
            intersectPos = GfVec3f(ray.GetPoint(depth));
            color = intersected_radiance;
        }
    }
    return color;
}

Color Integrator::IntersectDomeLight(const GfRay& ray)
{
    for (auto light : (*render_param->lights)) {
        if (light->IsDomeLight()) {
            float depth;
            return light->Intersect(ray, depth);
        }
    }
    return Color{ 0.0 };
}

bool Integrator::Intersect(const GfRay& ray, SurfaceInteraction& si)
{
    RTCRayHit rayHit;
    rayHit.ray.flags = 0;
    _PopulateRayHit(&rayHit, ray.GetStartPoint(), ray.GetDirection(), 0.0f);
    {
        rtcIntersect1(rtc_scene, &rayHit);
        rayHit.hit.Ng_x = -rayHit.hit.Ng_x;
        rayHit.hit.Ng_y = -rayHit.hit.Ng_y;
        rayHit.hit.Ng_z = -rayHit.hit.Ng_z;
    }

    if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return false;
    }

    const Hd_RUZINO_InstanceContext* instanceContext =
        static_cast<Hd_RUZINO_InstanceContext*>(rtcGetGeometryUserData(
            rtcGetGeometry(rtc_scene, rayHit.hit.instID[0])));

    const Hd_RUZINO_PrototypeContext* prototypeContext =
        static_cast<Hd_RUZINO_PrototypeContext*>(rtcGetGeometryUserData(
            rtcGetGeometry(instanceContext->rootScene, rayHit.hit.geomID)));

    auto hitPos = GfVec3f(
        rayHit.ray.org_x + rayHit.ray.tfar * rayHit.ray.dir_x,
        rayHit.ray.org_y + rayHit.ray.tfar * rayHit.ray.dir_y,
        rayHit.ray.org_z + rayHit.ray.tfar * rayHit.ray.dir_z);

    auto geometricNormal =
        -GfVec3f(rayHit.hit.Ng_x, rayHit.hit.Ng_y, rayHit.hit.Ng_z);

    GfVec3f shadingNormal;
    auto it = prototypeContext->primvarMap.find(HdTokens->normals);
    if (it != prototypeContext->primvarMap.end()) {
        it->second->Sample(
            rayHit.hit.primID, rayHit.hit.u, rayHit.hit.v, &shadingNormal);
    }
    else {
        shadingNormal = geometricNormal;
    }
    geometricNormal =
        instanceContext->objectToWorldMatrix.TransformDir(geometricNormal);
    shadingNormal =
        instanceContext->objectToWorldMatrix.TransformDir(shadingNormal);

    shadingNormal.Normalize();
    geometricNormal.Normalize();

    auto materialId = prototypeContext->rprim->GetMaterialId();
    si.material = (*render_param->materials)[materialId];

    auto texcoord_name = si.material->requireTexcoordName();
    it = prototypeContext->primvarMap.find(texcoord_name);
    GfVec2f texcoord;
    if (it != prototypeContext->primvarMap.end()) {
        it->second->Sample(
            rayHit.hit.primID, rayHit.hit.u, rayHit.hit.v, &texcoord);
        texcoord[1] = 1.0f - texcoord[1];
    }
    else {
        texcoord = { 0.5, 0.5 };
    }

    si.geometricNormal = geometricNormal;
    si.shadingNormal = shadingNormal;
    si.position = hitPos;
    si.barycentric = { rayHit.hit.u, rayHit.hit.v };
    si.texcoord = texcoord;
    si.PrepareTransforms();
    si.wo = GfVec3f(-ray.GetDirection().GetNormalized());

    return true;
}

bool Integrator::VisibilityTest(const GfRay& ray)
{
    RTCRay test_ray;
    _PopulateRay(&test_ray, ray.GetStartPoint(), ray.GetDirection(), 0);

    rtcOccluded1(rtc_scene, &test_ray);

    if (test_ray.tfar > 0) {
        return true;
    }
    return false;
}

bool Integrator::VisibilityTest(const GfVec3f& begin, const GfVec3f& end)
{
    GfRay ray;
    ray.SetEnds(begin, end);
    RTCRay test_ray;
    _PopulateRay(
        &test_ray,
        ray.GetStartPoint(),
        ray.GetDirection().GetNormalized(),
        0.0,
        (end - begin).GetLength() - 0.0001f);

    rtcOccluded1(rtc_scene, &test_ray);

    if (test_ray.tfar > 0)
        return true;
    return false;
}

static float PowerHeuristic(float f, float g)
{
    return f * f / (f * f + g * g);
}

Color Integrator::EstimateDirectLight(
    SurfaceInteraction& si,
    const std::function<float()>& uniform_float)
{
    GfVec3f L_dir{ 0.0f };
    int N_lights = render_param->lights->size();
    if (N_lights == 0)
        return L_dir;

    GfVec3f wi_light;
    float pdf_light_sample;
    GfVec3f sampled_light_pos;
    auto L_light = SampleLights(
        si.position,
        wi_light,
        sampled_light_pos,
        pdf_light_sample,
        uniform_float);

    if (pdf_light_sample > 1e-6f) {
        float cos_wi = GfDot(si.shadingNormal, wi_light);
        if (cos_wi > 0.0f) {
            if (this->VisibilityTest(
                    si.position + 0.0001f * si.geometricNormal,
                    sampled_light_pos)) {
                GfVec3f local_wi = si.WorldToTangent(wi_light);
                GfVec3f local_wo = si.WorldToTangent(si.wo);

                auto f_r = si.material->Eval(local_wi, local_wo, si.texcoord);
                float pdf_brdf =
                    si.material->Pdf(local_wi, local_wo, si.texcoord);

                float w_light = PowerHeuristic(pdf_light_sample, pdf_brdf);
                L_dir += GfCompMult(L_light, f_r) * cos_wi / pdf_light_sample *
                         w_light;
            }
        }
    }

    GfVec3f wi_brdf;
    float pdf_brdf_sample;
    auto f_r_brdf = si.Sample(wi_brdf, pdf_brdf_sample, uniform_float);

    if (pdf_brdf_sample > 1e-6f) {
        float cos_wi = GfDot(si.shadingNormal, wi_brdf);
        if (cos_wi > 0.0f) {
            GfRay ray_brdf(si.position + si.geometricNormal * 0.001f, wi_brdf);
            GfVec3f lightHitPos;
            Color L_hit = IntersectLights(ray_brdf, lightHitPos);

            if (L_hit != Color(0.0f) &&
                VisibilityTest(
                    si.position + 0.0001f * si.geometricNormal, lightHitPos)) {
                float pdf_light = 0.0f;

                for (auto light : *(render_param->lights)) {
                    float depth;
                    if (light->Intersect(ray_brdf, depth) != Color(0.0f)) {
                        pdf_light =
                            light->Pdf(si.position, wi_brdf) / float(N_lights);
                        break;
                    }
                }

                float w_brdf = PowerHeuristic(pdf_brdf_sample, pdf_light);
                L_dir += GfCompMult(L_hit, f_r_brdf) * cos_wi /
                         pdf_brdf_sample * w_brdf;
            }
        }
    }

    return L_dir;
}

void SamplingIntegrator::_writeBuffer(unsigned x, unsigned y, VtValue color)
{
    switch (channel(color)) {
        case 1:
            camera_->film->Write(GfVec3i(x, y, 1), 1, &color.Get<float>());
            break;
        case 3:
            camera_->film->Write(
                GfVec3i(x, y, 1), 3, color.Get<GfVec3f>().data());
            break;
        case 4:
            camera_->film->Write(
                GfVec3i(x, y, 1), 4, color.Get<GfVec4f>().data());
            break;
        default:;
    }
}

void SamplingIntegrator::accumulate_color(
    VtValue& color,
    const VtValue& vt_value)
{
    if (color.IsEmpty()) {
        color = vt_value;
    }
    else {
        switch (channel(vt_value)) {
            case 1:
                color = VtValue(color.Get<float>() + vt_value.Get<float>());
                return;
            case 3:
                color = VtValue(color.Get<GfVec3f>() + vt_value.Get<GfVec3f>());
                return;
            case 4:
                color = VtValue(color.Get<GfVec4f>() + vt_value.Get<GfVec4f>());
                return;
            default:;
        }
    }
}

VtValue SamplingIntegrator::average_samples(const VtValue& color, unsigned spp)
{
    assert(!color.IsEmpty());
    VtValue ret;

    switch (channel(color)) {
        case 1: ret = VtValue(color.Get<float>() / spp); break;
        case 3: ret = VtValue(color.Get<GfVec3f>() / spp); break;
        case 4: ret = VtValue(color.Get<GfVec4f>() / spp); break;
        default: assert(false);
    }
    return ret;
}

void SamplingIntegrator::_RenderTiles(
    HdRenderThread* renderThread,
    size_t tileStart,
    size_t tileEnd)
{
    const unsigned int minX = camera_->_dataWindow.GetMinX();
    unsigned int minY = camera_->_dataWindow.GetMinY();
    const unsigned int maxX = camera_->_dataWindow.GetMaxX() + 1;
    unsigned int maxY = camera_->_dataWindow.GetMaxY() + 1;

    std::swap(minY, maxY);
    auto height = camera_->film->GetHeight();
    minY = height - minY;
    maxY = height - maxY;

    const unsigned int tileSize = Hd_RUZINO_Config::GetInstance().tileSize;
    const unsigned int numTilesX =
        (camera_->_dataWindow.GetWidth() + tileSize - 1) / tileSize;

    size_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    seed ^= tileStart + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    std::default_random_engine random(seed);

    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
    std::function<float()> uniform_float = [&uniform_dist, &random]() {
        return uniform_dist(random);
    };

    for (unsigned int tile = tileStart; tile < tileEnd; ++tile) {
        if (renderThread && renderThread->IsStopRequested()) {
            break;
        }

        const unsigned int tileY = tile / numTilesX;
        const unsigned int tileX = tile - tileY * numTilesX;
        const unsigned int x0 = tileX * tileSize + minX;
        const unsigned int y0 = tileY * tileSize + minY;
        const unsigned int x1 = std::min(x0 + tileSize, maxX);
        const unsigned int y1 = std::min(y0 + tileSize, maxY);

        for (unsigned int y = y0; y < y1; ++y) {
            for (unsigned int x = x0; x < x1; ++x) {
                VtValue color;

                for (int sample = 0; sample < spp; ++sample) {
                    auto pixel_center_uv = GfVec2f(x, y);
                    auto ray =
                        camera_->generateRay(pixel_center_uv, uniform_float);
                    auto sampled_color = Li(ray, random);
                    accumulate_color(color, sampled_color);
                }
                color = average_samples(color, spp);

                _writeBuffer(x, y, color);
            }
        }
    }
}

void SamplingIntegrator::Render()
{
    camera_->film->Map();
    const unsigned int tileSize = Hd_RUZINO_Config::GetInstance().tileSize;

    const unsigned int numTilesX =
        (camera_->_dataWindow.GetWidth() + tileSize - 1) / tileSize;
    const unsigned int numTilesY =
        (camera_->_dataWindow.GetHeight() + tileSize - 1) / tileSize;

    WorkParallelForN(
        numTilesX * numTilesY,
        std::bind(
            &SamplingIntegrator::_RenderTiles,
            this,
            render_thread_,
            std::placeholders::_1,
            std::placeholders::_2));

    camera_->film->Unmap();

    camera_->film->SetConverged(true);
}

RUZINO_NAMESPACE_CLOSE_SCOPE