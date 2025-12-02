
#include <pxr/base/gf/vec2i.h>
#include <spdlog/spdlog.h>

#include <memory>

#include "../source/renderTLAS.h"
#include "GPUContext/program_vars.hpp"
#include "GPUContext/raytracing_context.hpp"
#include "RHI/internal/resources.hpp"
#include "Scene/MaterialParamsBuffer.slang"
#include "nodes/core/def/node_def.hpp"
#include "nvrhi/nvrhi.h"
#include "nvrhi/utils.h"
#include "render_node_base.h"
#include "shaders/shaders/utils/HitObject.h"
#include "utils/math.h"

// A traditional path tracing node

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(path_tracing)
{
    b.add_input<nvrhi::BufferHandle>("Pixel Target");
    b.add_input<nvrhi::BufferHandle>("Rays");
    b.add_input<nvrhi::BufferHandle>("Random Seeds");

    b.add_output<nvrhi::TextureHandle>("Output");

    // Function content omitted
}

struct PathTracingStorage {
    constexpr static bool has_storage = false;
    GfVec2i old_size = GfVec2i(-1, -1);

    ProgramHandle path_tracing_program;
    std::unordered_map<unsigned, std::string> callable_shaders;
    ResourceAllocator* rc;

    nvrhi::TextureHandle output;
    
    nvrhi::BufferHandle rays;
    nvrhi::BufferHandle material_params_buffer;
    nvrhi::BufferHandle lightCountBuffer;
    
    nvrhi::SamplerHandle sampler;

    std::unique_ptr<ProgramVars> cached_program_vars;
    std::unique_ptr<RaytracingContext> cached_rt_context;

    ~PathTracingStorage()
    {
        if (path_tracing_program && rc) {
            rc->destroy(path_tracing_program);
            path_tracing_program = nullptr;
        }
        if (material_params_buffer && rc) {
            rc->destroy(material_params_buffer);
            material_params_buffer = nullptr;
        }
        if (lightCountBuffer && rc) {
            rc->destroy(lightCountBuffer);
            lightCountBuffer = nullptr;
        }
        if (sampler && rc) {
            rc->destroy(sampler);
            sampler = nullptr;
        }
    }
};

NODE_EXECUTION_FUNCTION(path_tracing)
{
    using namespace nvrhi;

    auto& g = global_payload;
    auto geom_dirty =
        g.is_dirty(RenderGlobalPayload::SceneDirtyBits::DirtyGeometry);
    auto mat_dirty =
        g.is_dirty(RenderGlobalPayload::SceneDirtyBits::DirtyMaterials);
    auto light_dirty =
        g.is_dirty(RenderGlobalPayload::SceneDirtyBits::DirtyLights);

    auto size = get_free_camera(params)->dataWindow.GetSize();
    auto& storage = params.get_storage<PathTracingStorage&>();
    bool size_changed = (storage.old_size != size);
    storage.old_size = size;
    if (geom_dirty || mat_dirty || light_dirty || size_changed)
        spdlog::info(
            "Path Tracing Node: geom_dirty={}, mat_dirty={}, light_dirty={}, "
            "size_changed={}",
            geom_dirty,
            mat_dirty,
            light_dirty,
            size_changed);

    storage.rc = &(resource_allocator);

    if (mat_dirty || !storage.path_tracing_program) {
        ProgramDesc program_desc;
        program_desc.set_path("shaders/path_tracing.slang");
        program_desc.shaderType = nvrhi::ShaderType::AllRayTracing;
        program_desc.nvapi_support = true;

        auto& materials = global_payload.get_materials();

        storage.callable_shaders.clear();

        for (auto material : materials) {
            if (material.second == nullptr) {
                spdlog::warn(
                    "Null material found in path tracing node, {}",
                    material.first.GetText());
                continue;
            }
            auto location = material.second->GetMaterialLocation();
            if (location == -1) {
                continue;
            }

            program_desc.add_source_code(
                material.second->GetShader(shader_factory));

            auto callable = material.second->GetShader(shader_factory);
            storage.callable_shaders[location] =
                material.second->GetMaterialName();
        }

        if (storage.path_tracing_program) {
            resource_allocator.destroy(storage.path_tracing_program);
        }
        storage.path_tracing_program = resource_allocator.create(program_desc);
        CHECK_PROGRAM_ERROR(storage.path_tracing_program);
    }

    if (size_changed || !storage.output)
        storage.output =
            create_default_render_target(params, nvrhi::Format::RGBA32_FLOAT);

    auto upstream_rays = params.get_input<nvrhi::BufferHandle>("Rays");
    bool upstream_rays_changed = (upstream_rays != storage.rays);
    if (upstream_rays_changed) {
        storage.rays = upstream_rays;
    }
    bool is_any_dirty = geom_dirty || mat_dirty || light_dirty ||
                        size_changed || upstream_rays_changed;

    if (is_any_dirty || !storage.cached_program_vars ||
        !storage.cached_rt_context) {
        storage.cached_program_vars = std::make_unique<ProgramVars>(
            resource_allocator, storage.path_tracing_program);
        ProgramVars& program_vars = *storage.cached_program_vars;

        SamplerDesc sampler_desc;
        sampler_desc.addressU = nvrhi::SamplerAddressMode::Wrap;
        sampler_desc.addressV = nvrhi::SamplerAddressMode::Wrap;

        if (storage.sampler)
            resource_allocator.destroy(storage.sampler);
        storage.sampler = resource_allocator.create(sampler_desc);

        auto random_seeds =
            params.get_input<nvrhi::BufferHandle>("Random Seeds");

        program_vars["SceneBVH"] =
            params.get_global_payload<RenderGlobalPayload&>()
                .InstanceCollection->get_tlas();
        program_vars["inPixelTarget"] =
            params.get_input<nvrhi::BufferHandle>("Pixel Target");
        program_vars["output"] = storage.output;
        program_vars["random_seeds"] = random_seeds;
        for (int i = 0; i < 9; ++i) {
            program_vars["samplers"][i] = storage.sampler;
        }

        program_vars["rays"] = storage.rays;

        nvrhi::BufferDesc material_params_desc;
        // Each pixel should be able to store 288 bytes

        material_params_desc.byteSize = storage.rays->getDesc().byteSize /
                                        sizeof(RayInfo) *
                                        sizeof(MaterialParams);
        material_params_desc.structStride = sizeof(MaterialParams);
        material_params_desc.canHaveUAVs = true;
        material_params_desc.initialState =
            nvrhi::ResourceStates::ShaderResource;
        material_params_desc.debugName = "materialParamsBuffer";
        material_params_desc.keepInitialState = true;
        if (storage.material_params_buffer)
            resource_allocator.destroy(storage.material_params_buffer);
        storage.material_params_buffer =
            resource_allocator.create(material_params_desc);

        //    program_vars["cb"] = create_constant_buffer(params, 1);

        program_vars["instanceDescBuffer"] =
            instance_collection->instance_pool.get_device_buffer();
        program_vars["meshDescBuffer"] =
            instance_collection->mesh_pool.get_device_buffer();

        program_vars["materialBlobBuffer"] =
            instance_collection->material_pool.get_device_buffer();
        program_vars["materialHeaderBuffer"] =
            instance_collection->material_header_pool.get_device_buffer();
        program_vars["materialParamsBuffer"] = storage.material_params_buffer;

        // Bind light buffer - only include lights with valid paths
        auto& all_lights = global_payload.get_lights();
        std::vector<Hd_USTC_CG_Light*> valid_lights;

        for (auto* light : all_lights) {
            // Only include lights with non-empty paths (not fallback lights)
            if (light && !light->GetId().IsEmpty()) {
                valid_lights.push_back(light);
            }
        }

        uint32_t lightCount = static_cast<uint32_t>(valid_lights.size());

        instance_collection->light_pool.compress();
        program_vars["lightBuffer"] =
            instance_collection->light_pool.get_device_buffer();

        // Pass light count
        if (storage.lightCountBuffer)
            resource_allocator.destroy(storage.lightCountBuffer);
        storage.lightCountBuffer = create_constant_buffer(params, lightCount);
        program_vars["lightCount"] = storage.lightCountBuffer;

        program_vars.set_descriptor_table(
            "t_BindlessBuffers",
            instance_collection->bindlessData.bufferDescriptorTableManager
                ->GetDescriptorTable(),
            instance_collection->bindlessData.bufferBindlessLayout);

        program_vars.set_descriptor_table(
            "t_BindlessTextures",
            instance_collection->bindlessData.textureDescriptorTableManager
                ->GetDescriptorTable(),
            instance_collection->bindlessData.textureBindlessLayout);
        program_vars.finish_setting_vars();

        storage.cached_rt_context = std::make_unique<RaytracingContext>(
            resource_allocator, program_vars);

        RaytracingContext& context = *storage.cached_rt_context;

        context.announce_raygeneration("RayGen");
        context.announce_hitgroup(
            "ClosestHit", "", "", 0);  // Primary ray hit group at index 0
        context.announce_hitgroup(
            "ShadowHit", "", "", 1);       // Shadow ray hit group at index 1
        context.announce_miss("Miss", 0);  // Primary ray miss shader at index 0
        context.announce_miss(
            "ShadowMiss", 1);  // Shadow ray miss shader at index 1

        // Register shared material evaluation callables at fixed indices
        // Pass nullptr for local root signature since these callables use
        // global resources
        context.announce_callable(
            "eval_standard_surface",
            0,
            nullptr);  // shader_type_id = 0
        context.announce_callable(
            "eval_preview_surface", 1, nullptr);  // shader_type_id = 1
        context.announce_callable("eval_fallback", 2, nullptr);
        // Register per-material data fetch callables starting from index 2
        for (auto& callable : storage.callable_shaders) {
            context.announce_callable(
                callable.second, 3 + callable.first, nullptr);
        }

        context.finish_announcing_shader_names();
    }

    auto buffer_size = storage.rays->getDesc().byteSize / sizeof(RayInfo);

    if (buffer_size > 0) {
        storage.cached_rt_context->begin();
        storage.cached_rt_context->trace_rays(
            {}, *storage.cached_program_vars, buffer_size, 1, 1);
        storage.cached_rt_context->finish();
    }

    params.set_output("Output", storage.output);

    return true;
}

NODE_DECLARATION_UI(path_tracing);
NODE_DEF_CLOSE_SCOPE
