#pragma once

#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/timeCode.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <string>

#include "api.h"

RUZINO_NAMESPACE_OPEN_SCOPE

// Forward declarations
class NodeTree;
class NodeTreeExecutor;
class Geometry;

namespace ecs {

// ============================================================================
// USD Component - 存储 USD prim 的引用和 USD↔Geometry 同步功能
// ============================================================================
struct STAGE_API UsdPrimComponent {
    pxr::UsdPrim prim;

    // Prim自己的时间状态（用于独立的动画时间线）
    pxr::UsdTimeCode current_time = pxr::UsdTimeCode(0.0);
    pxr::UsdTimeCode render_time = pxr::UsdTimeCode(0.0);

    UsdPrimComponent() = default;
    explicit UsdPrimComponent(const pxr::UsdPrim& p) : prim(p)
    {
    }

    // 从 USD prim 同步几何数据到 Geometry 对象
    // 参数: geometry - 目标 Geometry 对象
    //       time - USD 时间码（默认使用 current_time）
    // 返回: 是否成功同步
    bool sync_to_geometry(
        class Geometry& geometry,
        const pxr::UsdTimeCode& time) const;
    bool sync_to_geometry(class Geometry& geometry) const;

    // 从 Geometry 对象同步数据到 USD prim
    // 参数: geometry - 源 Geometry 对象
    //       time - USD 时间码（默认使用 current_time）
    // 返回: 是否成功同步
    bool sync_from_geometry(
        const class Geometry& geometry,
        const pxr::UsdTimeCode& time);
    bool sync_from_geometry(const class Geometry& geometry);
};

// ============================================================================
// Animation Component - 存储动画/节点逻辑
// ============================================================================
struct STAGE_API AnimationComponent {
    std::shared_ptr<NodeTree> node_tree;
    std::shared_ptr<NodeTreeExecutor> node_tree_executor;

    // 缓存的树描述，用于检测变化
    mutable std::string tree_desc_cache;

    // 仿真状态
    mutable bool simulation_begun = false;

    AnimationComponent() = default;
};

// ============================================================================
// Geometry Component - 包装 Geometry 系统
// ============================================================================
struct STAGE_API GeometryComponent {
    std::shared_ptr<Geometry> geometry;

    GeometryComponent() = default;
    explicit GeometryComponent(std::shared_ptr<Geometry> geom)
        : geometry(std::move(geom))
    {
    }
};

// ============================================================================
// Material Component - 材质引用
// ============================================================================
struct STAGE_API MaterialComponent {
    pxr::SdfPath material_path;
    std::string shader_path;  // Custom shader path

    MaterialComponent() = default;
};

// ============================================================================
// Physics Component - 为 PhysX 预留
// ============================================================================
struct STAGE_API PhysicsComponent {
    enum class Type {
        Static,    // 静态物体
        Dynamic,   // 动态物体
        Kinematic  // 运动学物体
    };

    Type type = Type::Static;
    float mass = 1.0f;
    bool is_trigger = false;

    // PhysX actor 指针（暂时用 void* 以避免依赖）
    void* physics_actor = nullptr;

    PhysicsComponent() = default;
};

// ============================================================================
// Dirty Flag Component - 标记需要同步到 USD 的 entity
// ============================================================================
struct STAGE_API DirtyComponent {
    bool needs_usd_sync = false;
    bool needs_geometry_update = false;  // Geometry发生变化，需要写入USD

    DirtyComponent() = default;
};

// ============================================================================
// Tag Components - 用于筛选的标签
// ============================================================================
struct STAGE_API AnimatableTag { };  // 标记可动画的 entity
struct STAGE_API RenderableTag { };  // 标记可渲染的 entity
struct STAGE_API SimulationTag { };  // 标记参与物理模拟的 entity

}  // namespace ecs

RUZINO_NAMESPACE_CLOSE_SCOPE
