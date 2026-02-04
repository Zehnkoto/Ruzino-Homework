#include "stage/ecs_systems.hpp"

#include <pxr/usd/usdGeom/xformable.h>

#include <glm/gtc/type_ptr.hpp>

#include "../../../Editor/geometry/include/GCore/geom_payload.hpp"
#include "stage/animation.h"
#include "stage/stage.hpp"

#ifdef GEOM_USD_EXTENSION
#include "GCore/usd_extension.h"
#endif

RUZINO_NAMESPACE_OPEN_SCOPE
namespace ecs {

// ============================================================================
// Animation System Implementation
// ============================================================================

AnimationSystem::AnimationSystem(Stage* stage) : stage_(stage)
{
}

void AnimationSystem::update(entt::registry& registry, float delta_time)
{
    // 遍历所有拥有 AnimationComponent 和 UsdPrimComponent 的 entity
    auto view = registry.view<AnimationComponent, UsdPrimComponent>();

    for (auto entity : view) {
        auto& anim = view.get<AnimationComponent>(entity);
        auto& usd_prim = view.get<UsdPrimComponent>(entity);

        update_single_entity(registry, entity, anim, usd_prim, delta_time);
    }
}

void AnimationSystem::update_single_entity(
    entt::registry& registry,
    entt::entity entity,
    AnimationComponent& anim,
    UsdPrimComponent& usd_prim,
    float delta_time)
{
    if (!anim.node_tree || !anim.node_tree_executor) {
        return;
    }

    // 检查节点树是否变化
    auto json_attr = usd_prim.prim.GetAttribute(pxr::TfToken("node_json"));
    if (json_attr) {
        pxr::VtValue json;
        json_attr.Get(&json);
        auto new_tree_desc = json.Get<std::string>();

        if (anim.tree_desc_cache != new_tree_desc) {
            anim.tree_desc_cache = new_tree_desc;
            anim.node_tree->deserialize(anim.tree_desc_cache);
            anim.node_tree_executor->mark_tree_structure_changed();

            // 清除旧的时间采样数据
            clear_time_samples(usd_prim.prim);

            // 重置时间状态
            usd_prim.current_time = pxr::UsdTimeCode(0.0f);
            usd_prim.render_time = pxr::UsdTimeCode(0.0f);
            anim.simulation_begun = false;
        }
    }

    // 更新渲染时间
    usd_prim.render_time = stage_->get_render_time();

    // 检查是否应该进行仿真
    if (usd_prim.render_time < usd_prim.current_time) {
        return;
    }

    // 执行节点树
    auto& payload = anim.node_tree_executor->get_global_payload<GeomPayload&>();
    payload.delta_time = delta_time;

#ifdef GEOM_USD_EXTENSION
    payload.stage = usd_prim.prim.GetStage();
    payload.prim_path = usd_prim.prim.GetPath();
    payload.current_time = usd_prim.current_time;
#endif

    payload.has_simulation = false;
    payload.is_simulating = anim.simulation_begun;

    if (!anim.simulation_begun) {
        anim.simulation_begun = true;
    }

    anim.node_tree_executor->execute(anim.node_tree.get());

    // 更新仿真时间
    auto current = usd_prim.current_time.GetValue();
    current += delta_time;
    usd_prim.current_time = pxr::UsdTimeCode(current);

    // 标记为脏，需要同步到 USD
    if (!registry.all_of<DirtyComponent>(entity)) {
        registry.emplace<DirtyComponent>(entity);
    }
    auto& dirty = registry.get<DirtyComponent>(entity);
    dirty.needs_geometry_update = true;
}

void AnimationSystem::clear_time_samples(const pxr::UsdPrim& prim) const
{
    if (!prim.IsValid()) {
        return;
    }

    // 递归清除所有子prim的时间采样
    for (const auto& child : prim.GetChildren()) {
        clear_time_samples(child);
    }

    // 清除当前prim所有属性的时间采样
    for (const auto& attr : prim.GetAttributes()) {
        const auto& attr_name = attr.GetName();
        if (attr_name == pxr::TfToken("node_json") ||
            attr_name == pxr::TfToken("Animatable")) {
            continue;
        }

        if (attr.GetNumTimeSamples() > 0) {
            attr.Clear();
        }
    }
}

// ============================================================================
// USD Sync System Implementation
// ============================================================================

UsdSyncSystem::UsdSyncSystem(Stage* stage) : stage_(stage)
{
}

void UsdSyncSystem::sync(entt::registry& registry, pxr::UsdTimeCode time)
{
    // 同步所有脏 entity
    auto view = registry.view<DirtyComponent, UsdPrimComponent>();

    int dirty_count = 0;
    for (auto entity : view) {
        dirty_count++;
    }
    std::cout << "[DEBUG UsdSyncSystem::sync] 开始同步，发现 " << dirty_count
              << " 个脏entity" << std::endl;

    int synced_count = 0;
    for (auto entity : view) {
        auto& dirty = view.get<DirtyComponent>(entity);
        auto& usd_prim = view.get<UsdPrimComponent>(entity);

        std::cout << "[DEBUG UsdSyncSystem::sync] entity #"
                  << (synced_count + 1)
                  << ", prim=" << usd_prim.prim.GetPath().GetString()
                  << ", needs_geometry_update=" << dirty.needs_geometry_update
                  << std::endl;

        // 如果 geometry 发生变化，同步到 USD
        if (dirty.needs_geometry_update) {
            std::cout
                << "[DEBUG UsdSyncSystem::sync] needs_geometry_update=true"
                << std::endl;
            if (registry.all_of<GeometryComponent>(entity)) {
                auto& geom_comp = registry.get<GeometryComponent>(entity);
                std::cout << "[DEBUG UsdSyncSystem::sync] 有GeometryComponent, "
                             "geometry="
                          << (geom_comp.geometry ? "存在" : "nullptr")
                          << std::endl;
                if (geom_comp.geometry) {
                    std::cout << "[DEBUG UsdSyncSystem::sync] "
                                 "调用sync_from_geometry写入USD"
                              << std::endl;
                    usd_prim.sync_from_geometry(*geom_comp.geometry, time);
                    std::cout
                        << "[DEBUG UsdSyncSystem::sync] sync_from_geometry完成"
                        << std::endl;
                }
            }
            else {
                std::cout << "[DEBUG UsdSyncSystem::sync] 没有GeometryComponent"
                          << std::endl;
            }
        }

        // 清除脏标记
        std::cout << "[DEBUG UsdSyncSystem::sync] 清除脏标记" << std::endl;
        dirty.needs_geometry_update = false;
        dirty.needs_usd_sync = false;
        synced_count++;
    }
    std::cout << "[DEBUG UsdSyncSystem::sync] 同步完成，处理了 " << synced_count
              << " 个entity" << std::endl;
}

entt::entity UsdSyncSystem::create_entity_from_prim(
    entt::registry& registry,
    const pxr::UsdPrim& prim)
{
    auto entity = registry.create();

    // 添加 USD prim component
    registry.emplace<UsdPrimComponent>(entity, prim);

    // 检查是否有动画
    auto animatable_attr = prim.GetAttribute(pxr::TfToken("Animatable"));
    if (animatable_attr) {
        bool is_animatable = false;
        animatable_attr.Get(&is_animatable);

        if (is_animatable) {
            registry.emplace<AnimatableTag>(entity);
            // 这里可以初始化 AnimationComponent
        }
    }

    // 可以添加其他 components...

    return entity;
}

void UsdSyncSystem::update_prim_from_entity(
    entt::registry& registry,
    entt::entity entity,
    pxr::UsdTimeCode time)
{
    auto& usd_prim = registry.get<UsdPrimComponent>(entity);

    // 同步 Geometry（包括 Transform）
    if (registry.all_of<GeometryComponent>(entity)) {
        auto& geometry = registry.get<GeometryComponent>(entity);
        if (geometry.geometry) {
            usd_prim.sync_from_geometry(*geometry.geometry, time);
        }
    }
}

void UsdSyncSystem::sync_geometry(
    const GeometryComponent& geometry,
    pxr::UsdPrim& prim,
    pxr::UsdTimeCode time)
{
#ifdef GEOM_USD_EXTENSION
    if (geometry.geometry) {
        write_geometry_to_usd(
            *geometry.geometry, prim.GetStage(), prim.GetPath(), time);
    }
#endif
}

// ============================================================================
// Physics System Implementation (预留)
// ============================================================================

PhysicsSystem::PhysicsSystem()
{
}

PhysicsSystem::~PhysicsSystem()
{
    shutdown();
}

bool PhysicsSystem::initialize()
{
    if (initialized_) {
        return true;
    }

    // TODO: 初始化 PhysX SDK
    // physics_sdk_ = PxCreatePhysics(...);
    // physics_scene_ = physics_sdk_->createScene(...);

    initialized_ = true;
    return true;
}

void PhysicsSystem::shutdown()
{
    if (!initialized_) {
        return;
    }

    // TODO: 清理 PhysX 资源

    initialized_ = false;
}

void PhysicsSystem::update(entt::registry& registry, float delta_time)
{
    if (!initialized_) {
        return;
    }

    // TODO: 更新物理模拟
    // physics_scene_->simulate(delta_time);
    // physics_scene_->fetchResults(true);

    // 更新所有物理 entity 的 transform
    auto view =
        registry.view<PhysicsComponent, GeometryComponent, UsdPrimComponent>();
    for (auto entity : view) {
        auto& physics = view.get<PhysicsComponent>(entity);
        auto& geom_comp = view.get<GeometryComponent>(entity);
        auto& usd_prim = view.get<UsdPrimComponent>(entity);

        // TODO: 从 PhysX actor 获取变换并更新 Geometry 的 XformComponent
        // if (physics.physics_actor && geom_comp.geometry) {
        //     PxTransform px_transform =
        //     static_cast<PxRigidActor*>(physics.physics_actor)->getGlobalPose();
        //     auto xform_comp =
        //     geom_comp.geometry->get_component<XformComponent>(); if
        //     (xform_comp) {
        //         xform_comp->translation[0] = glm::vec3(px_transform.p.x,
        //         px_transform.p.y, px_transform.p.z); xform_comp->rotation[0]
        //         = glm::quat(px_transform.q.w, px_transform.q.x,
        //         px_transform.q.y, px_transform.q.z);
        //     }
        // }
    }
}

void PhysicsSystem::add_physics_actor(
    entt::registry& registry,
    entt::entity entity)
{
    // TODO: 创建 PhysX actor
}

void PhysicsSystem::remove_physics_actor(
    entt::registry& registry,
    entt::entity entity)
{
    // TODO: 销毁 PhysX actor
}

// ============================================================================
// Scene Query System Implementation (预留)
// ============================================================================

SceneQuerySystem::SceneQuerySystem(PhysicsSystem* physics_system)
    : physics_system_(physics_system)
{
}

bool SceneQuerySystem::raycast(
    const glm::vec3& origin,
    const glm::vec3& direction,
    float max_distance,
    entt::entity& hit_entity,
    glm::vec3& hit_position)
{
    // TODO: 实现 raycast
    return false;
}

bool SceneQuerySystem::overlap_sphere(
    const glm::vec3& center,
    float radius,
    std::vector<entt::entity>& overlapping_entities)
{
    // TODO: 实现 overlap test
    return false;
}

}  // namespace ecs
RUZINO_NAMESPACE_CLOSE_SCOPE
