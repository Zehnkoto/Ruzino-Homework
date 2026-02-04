#pragma once

#include <pxr/usd/usd/stage.h>

#include <entt/entt.hpp>

#include "api.h"
#include "stage/ecs_components.hpp"

RUZINO_NAMESPACE_OPEN_SCOPE

// Forward declaration
class Stage;

namespace ecs {

// ============================================================================
// Animation System - 处理动画逻辑的更新
// ============================================================================
class STAGE_API AnimationSystem {
   public:
    AnimationSystem(Stage* stage);

    // 更新所有动画 entity
    void update(entt::registry& registry, float delta_time);

   private:
    void update_single_entity(
        entt::registry& registry,
        entt::entity entity,
        AnimationComponent& anim,
        UsdPrimComponent& usd_prim,
        float delta_time);

    void clear_time_samples(const pxr::UsdPrim& prim) const;

    Stage* stage_;
};

// ============================================================================
// USD Sync System - 将 ECS 数据同步到 USD
// ============================================================================
class STAGE_API UsdSyncSystem {
   public:
    UsdSyncSystem(Stage* stage);

    // 同步所有脏 entity 到 USD
    void sync(entt::registry& registry, pxr::UsdTimeCode time);

    // 从 USD 创建 entity
    entt::entity create_entity_from_prim(
        entt::registry& registry,
        const pxr::UsdPrim& prim);

    // 从 entity 更新 USD prim
    void update_prim_from_entity(
        entt::registry& registry,
        entt::entity entity,
        pxr::UsdTimeCode time);

   private:
    void sync_geometry(
        const GeometryComponent& geometry,
        pxr::UsdPrim& prim,
        pxr::UsdTimeCode time);

    Stage* stage_;
};

// ============================================================================
// Physics System - 物理模拟系统（为 PhysX 预留）
// ============================================================================
class STAGE_API PhysicsSystem {
   public:
    PhysicsSystem();
    ~PhysicsSystem();

    // 初始化物理引擎
    bool initialize();

    // 清理物理引擎
    void shutdown();

    // 更新物理模拟
    void update(entt::registry& registry, float delta_time);

    // 添加物理 actor
    void add_physics_actor(entt::registry& registry, entt::entity entity);

    // 移除物理 actor
    void remove_physics_actor(entt::registry& registry, entt::entity entity);

   private:
    // PhysX 相关的成员（暂时用 void* 避免依赖）
    void* physics_sdk_ = nullptr;
    void* physics_scene_ = nullptr;
    bool initialized_ = false;
};

// ============================================================================
// Scene Query System - 用于碰撞检测等查询
// ============================================================================
class STAGE_API SceneQuerySystem {
   public:
    SceneQuerySystem(PhysicsSystem* physics_system);

    // Raycast
    bool raycast(
        const glm::vec3& origin,
        const glm::vec3& direction,
        float max_distance,
        entt::entity& hit_entity,
        glm::vec3& hit_position);

    // Overlap test
    bool overlap_sphere(
        const glm::vec3& center,
        float radius,
        std::vector<entt::entity>& overlapping_entities);

   private:
    PhysicsSystem* physics_system_;
};

}  // namespace ecs

RUZINO_NAMESPACE_CLOSE_SCOPE
