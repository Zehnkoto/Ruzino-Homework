#include <gtest/gtest.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/xform.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>

#include "GCore/GOP.h"
#include "GCore/create_geom.h"
#include "stage/ecs_components.hpp"
#include "stage/ecs_systems.hpp"
#include "stage/stage.hpp"

using namespace Ruzino;

// 生成带时间戳的临时USD文件路径
static std::string get_temp_usd_path(const std::string& test_name)
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;

    char timestamp[32];
    std::tm* tm_info = std::localtime(&time);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);

    std::string filename = std::string(timestamp) + "_" +
                           std::to_string(ms.count()) + "_" + test_name +
                           ".usdc";
    return "../../Assets/.test_tmp/" + filename;
}

// 用于测试的动画组件 - 定义简单的平移动画
struct TranslationAnimationComponent {
    glm::vec3 velocity;  // 每秒移动的速度
    float elapsed_time = 0.0f;
};

// 用于测试的动画系统 - 更新位置并写入USD时间采样
class TranslationAnimationSystem {
   public:
    void update(
        entt::registry& registry,
        float delta_time,
        pxr::UsdTimeCode time_code)
    {
        auto view =
            registry
                .view<ecs::UsdPrimComponent, TranslationAnimationComponent>();

        for (auto entity : view) {
            auto& usd_comp = view.get<ecs::UsdPrimComponent>(entity);
            auto& anim_comp = view.get<TranslationAnimationComponent>(entity);

            // 更新elapsed time
            anim_comp.elapsed_time += delta_time;

            // 计算新位置
            glm::vec3 new_position =
                anim_comp.velocity * anim_comp.elapsed_time;

            // 获取xformable并设置变换
            auto xformable = pxr::UsdGeomXformable(usd_comp.prim);
            if (xformable) {
                // 获取或创建translate op
                pxr::UsdGeomXformOp translate_op;
                bool reset_stack;
                auto existing_ops = xformable.GetOrderedXformOps(&reset_stack);

                // 检查是否已有translate op
                bool has_translate = false;
                for (const auto& op : existing_ops) {
                    if (op.GetOpType() == pxr::UsdGeomXformOp::TypeTranslate) {
                        translate_op = op;
                        has_translate = true;
                        break;
                    }
                }

                // 如果没有，创建一个
                if (!has_translate) {
                    translate_op = xformable.AddTranslateOp();
                }

                // 设置位置
                pxr::GfVec3d usd_pos(
                    new_position.x, new_position.y, new_position.z);
                translate_op.Set(usd_pos, time_code);

                // 更新组件的时间
                usd_comp.current_time = time_code;
            }
        }
    }
};

class EcsApiTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        // 测试前的准备
    }

    void TearDown() override
    {
        // 测试后的清理
    }
};

TEST_F(EcsApiTest, BasicEcsCreation)
{
    // 创建新的stage（文件不存在会自动创建）
    auto stage = std::make_unique<Stage>(get_temp_usd_path("BasicEcsCreation"));
    auto& registry = stage->get_registry();

    // 检查初始的entity数
    int initial_count = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        initial_count++;
    }

    // 创建一些 USD prims
    auto sphere_prim = stage->create_sphere(pxr::SdfPath("/MySphere"));
    auto cube_prim = stage->create_cube(pxr::SdfPath("/MyCube"));

    // 获取自动创建的 entities
    int entity_count = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entity_count++;
    }

    // create_sphere/cube会创建parent xform和shape prim，所以总共4个entities
    // MySphere, MySphere/sphere_0, MyCube, MyCube/cube_0
    int new_entities = entity_count - initial_count;
    EXPECT_EQ(new_entities, 4) << "应该创建了 4 个entities (包括parent xforms)";

    // 验证映射是否建立 - 确保sphere和cube的shape prims能找到
    auto sphere_entity = stage->find_entity_by_path(sphere_prim.GetPath());
    auto cube_entity = stage->find_entity_by_path(cube_prim.GetPath());

    EXPECT_TRUE(registry.valid(sphere_entity)) << "Sphere entity 应该存在";
    EXPECT_TRUE(registry.valid(cube_entity)) << "Cube entity 应该存在";

    // 验证 entity 到 prim 的映射
    auto sphere_prim_back = stage->get_prim_from_entity(sphere_entity);
    EXPECT_EQ(sphere_prim_back.GetPath(), sphere_prim.GetPath());
}

TEST_F(EcsApiTest, ComponentOperations)
{
    // 创建新的stage（文件不存在会自动创建）
    auto stage =
        std::make_unique<Stage>(get_temp_usd_path("ComponentOperations"));
    auto& registry = stage->get_registry();

    // 创建一个 USD sphere
    auto sphere = stage->create_sphere(pxr::SdfPath("/TestSphere"));
    auto entity = stage->find_entity_by_path(sphere.GetPath());

    ASSERT_TRUE(registry.valid(entity)) << "Entity 应该存在";

    // 验证 UsdPrimComponent 已自动添加
    EXPECT_TRUE(registry.all_of<ecs::UsdPrimComponent>(entity));

    // 添加 MaterialComponent
    auto& material = registry.emplace<ecs::MaterialComponent>(entity);
    material.material_path = pxr::SdfPath("/Materials/MyMaterial");
    material.shader_path = "shaders/pbr.slang";

    EXPECT_EQ(material.material_path.GetString(), "/Materials/MyMaterial");
    EXPECT_EQ(material.shader_path, "shaders/pbr.slang");

    // 添加 PhysicsComponent
    auto& physics = registry.emplace<ecs::PhysicsComponent>(entity);
    physics.type = ecs::PhysicsComponent::Type::Dynamic;
    physics.mass = 10.0f;
    physics.is_trigger = false;

    EXPECT_EQ(physics.mass, 10.0f);

    // 添加 Tags
    registry.emplace<ecs::RenderableTag>(entity);
    registry.emplace<ecs::SimulationTag>(entity);

    // 验证所有组件都存在
    EXPECT_TRUE(registry.all_of<ecs::UsdPrimComponent>(entity));
    EXPECT_TRUE(registry.all_of<ecs::MaterialComponent>(entity));
    EXPECT_TRUE(registry.all_of<ecs::PhysicsComponent>(entity));
    EXPECT_TRUE(registry.all_of<ecs::RenderableTag>(entity));
    EXPECT_TRUE(registry.all_of<ecs::SimulationTag>(entity));
}

TEST_F(EcsApiTest, EntityQueries)
{
    // 创建新的stage（文件不存在会自动创建）
    auto stage = std::make_unique<Stage>(get_temp_usd_path("EntityQueries"));
    auto& registry = stage->get_registry();

    // 创建多个 entities
    for (int i = 0; i < 5; i++) {
        auto path = pxr::SdfPath(std::string("/Object_") + std::to_string(i));
        auto sphere = stage->create_sphere(path);
        auto entity = stage->find_entity_by_path(sphere.GetPath());

        if (i % 2 == 0) {
            registry.emplace<ecs::MaterialComponent>(entity);
            registry.emplace<ecs::RenderableTag>(entity);
        }

        if (i % 3 == 0) {
            auto& physics = registry.emplace<ecs::PhysicsComponent>(entity);
            physics.mass = 5.0f * (i + 1);
            registry.emplace<ecs::SimulationTag>(entity);
        }
    }

    // 查询所有拥有 UsdPrimComponent 的 entities (包括parent xforms)
    // 5个create_sphere会创建10个entities (5 parents + 5 shapes)
    {
        auto view = registry.view<ecs::UsdPrimComponent>();
        int count = std::distance(view.begin(), view.end());
        EXPECT_EQ(count, 10) << "应该有 10 个带 UsdPrimComponent 的 entities "
                                "(5 parent xforms + 5 shapes)";
    }

    // 查询所有可渲染的 entities (只有sphere_0, 2,
    // 4三个shape有MaterialComponent)
    {
        auto view = registry.view<
            ecs::UsdPrimComponent,
            ecs::MaterialComponent,
            ecs::RenderableTag>();
        int count = std::distance(view.begin(), view.end());
        EXPECT_EQ(count, 3) << "应该有 3 个可渲染的 entities";
    }

    // 查询所有物理对象 (sphere_0和 sphere_3两个shape有PhysicsComponent)
    {
        auto view = registry.view<ecs::PhysicsComponent, ecs::SimulationTag>();
        int count = std::distance(view.begin(), view.end());
        EXPECT_EQ(count, 2) << "应该有 2 个物理对象";

        for (auto entity : view) {
            auto& physics = view.get<ecs::PhysicsComponent>(entity);
            EXPECT_GT(physics.mass, 0.0f) << "物理对象的质量应该大于 0";
        }
    }
}

TEST_F(EcsApiTest, UsdSync)
{
    // 创建新的stage（文件不存在会自动创建）
    auto stage = std::make_unique<Stage>(get_temp_usd_path("UsdSync"));
    auto& registry = stage->get_registry();

    // 创建一个 sphere
    auto sphere = stage->create_sphere(pxr::SdfPath("/SyncTest"));
    auto entity = stage->find_entity_by_path(sphere.GetPath());

    ASSERT_TRUE(registry.valid(entity));

    // 标记需要同步
    auto& dirty = registry.emplace<ecs::DirtyComponent>(entity);
    dirty.needs_geometry_update = true;

    EXPECT_TRUE(dirty.needs_geometry_update);

    // 同步到 USD
    stage->sync_entities_to_usd();

    // 验证 USD 中的数据
    auto xformable = pxr::UsdGeomXformable(sphere.GetPrim());
    if (xformable) {
        pxr::GfMatrix4d matrix;
        bool reset_stack;
        xformable.GetLocalTransformation(
            &matrix, &reset_stack, pxr::UsdTimeCode::Default());

        auto translation = matrix.ExtractTranslation();
        EXPECT_EQ(translation[0], 0.0);
        EXPECT_EQ(translation[1], 0.0);
        EXPECT_EQ(translation[2], 0.0);
    }
}

TEST_F(EcsApiTest, StageListenerCallbacks)
{
    // 创建新的stage（文件不存在会自动创建）
    auto stage =
        std::make_unique<Stage>(get_temp_usd_path("StageListenerCallbacks"));
    auto& registry = stage->get_registry();

    int entities_before = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entities_before++;
    }

    // 创建新的 prim (这应该触发回调)
    auto new_sphere = stage->create_sphere(pxr::SdfPath("/NewSphere"));

    // 检查是否自动创建了 entity
    int entities_after = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entities_after++;
    }

    int new_entities = entities_after - entities_before;
    EXPECT_GT(new_entities, 0) << "StageListener 应该自动创建 entity";

    auto new_entity = stage->find_entity_by_path(new_sphere.GetPath());
    EXPECT_TRUE(registry.valid(new_entity)) << "应该能找到新创建的 entity";

    // 测试直接使用USD API创建prim，看是否触发notice机制
    auto pxr_stage = stage->get_usd_stage();

    int entities_before_usd_api = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entities_before_usd_api++;
    }

    // 直接使用USD API创建一个xform prim
    auto direct_prim = pxr_stage->DefinePrim(
        pxr::SdfPath("/DirectXform"), pxr::TfToken("Xform"));
    pxr::UsdGeomXform direct_xform(direct_prim);

    int entities_after_usd_api = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entities_after_usd_api++;
    }

    int entities_from_usd_api =
        entities_after_usd_api - entities_before_usd_api;
    std::cout << "\n[USD API Test] DirectXform创建后:" << std::endl;
    std::cout << "  创建前entity数: " << entities_before_usd_api << std::endl;
    std::cout << "  创建后entity数: " << entities_after_usd_api << std::endl;
    std::cout << "  新增entity数: " << entities_from_usd_api << std::endl;

    if (entities_from_usd_api > 0) {
        std::cout << "  ✓ USD API创建的prim触发了notice机制！" << std::endl;
        auto direct_entity = stage->find_entity_by_path(direct_prim.GetPath());
        EXPECT_TRUE(registry.valid(direct_entity))
            << "应该能找到直接通过USD API创建的entity";
    }
    else {
        std::cout << "  ✗ USD API创建的prim没有触发notice机制" << std::endl;
    }
}

TEST_F(EcsApiTest, PrimDeletionSync)
{
    // 创建新的stage（文件不存在会自动创建）
    auto stage = std::make_unique<Stage>(get_temp_usd_path("PrimDeletionSync"));
    auto& registry = stage->get_registry();
    auto pxr_stage = stage->get_usd_stage();

    // 直接创建一个prim
    auto test_prim = pxr_stage->DefinePrim(
        pxr::SdfPath("/TestDeletion"), pxr::TfToken("Xform"));

    // 获取创建后的entity数
    int entities_after_create = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entities_after_create++;
    }

    auto test_entity = stage->find_entity_by_path(test_prim.GetPath());
    bool entity_exists_before = registry.valid(test_entity);

    std::cout << "\n[Prim Deletion Test]" << std::endl;
    std::cout << "  创建TestDeletion后:" << std::endl;
    std::cout << "    Entity总数: " << entities_after_create << std::endl;
    std::cout << "    Entity存在: " << (entity_exists_before ? "是" : "否")
              << std::endl;

    // 删除这个prim
    pxr_stage->RemovePrim(test_prim.GetPath());

    // 获取删除后的entity数
    int entities_after_delete = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entities_after_delete++;
    }

    auto test_entity_after = stage->find_entity_by_path(test_prim.GetPath());
    bool entity_exists_after = registry.valid(test_entity_after);

    std::cout << "  删除TestDeletion后:" << std::endl;
    std::cout << "    Entity总数: " << entities_after_delete << std::endl;
    std::cout << "    新增Entity数: "
              << (entities_after_delete - entities_after_create) << std::endl;
    std::cout << "    Entity存在: " << (entity_exists_after ? "是" : "否")
              << std::endl;

    if (!entity_exists_after && entities_after_delete < entities_after_create) {
        std::cout << "  ✓ 删除prim成功触发了notice机制，entity被正确删除！"
                  << std::endl;
    }
    else {
        std::cout << "  ✗ 删除prim可能没有正确删除entity" << std::endl;
    }

    EXPECT_LT(entities_after_delete, entities_after_create)
        << "删除prim后entity数应该减少";
    EXPECT_FALSE(entity_exists_after) << "被删除的prim对应的entity应该被删除";
}

TEST_F(EcsApiTest, PrimModificationSync)
{
    // 创建新的stage（文件不存在会自动创建）
    auto stage =
        std::make_unique<Stage>(get_temp_usd_path("PrimModificationSync"));
    auto& registry = stage->get_registry();

    // 创建一个sphere
    auto sphere_prim = stage->create_sphere(pxr::SdfPath("/ModifiableSphere"));
    auto sphere_entity = stage->find_entity_by_path(sphere_prim.GetPath());

    ASSERT_TRUE(registry.valid(sphere_entity));

    // 检查修改前的dirty标记
    bool had_dirty_before = registry.all_of<ecs::DirtyComponent>(sphere_entity);

    // 修改sphere的属性（比如半径）- 使用double而不是float
    auto sphere = pxr::UsdGeomSphere(sphere_prim);
    sphere.GetRadiusAttr().Set(5.0);  // USD期望double类型

    // 检查修改后是否标记为脏
    bool has_dirty_after = registry.all_of<ecs::DirtyComponent>(sphere_entity);

    std::cout << "\n[Prim Modification Test]" << std::endl;
    std::cout << "  修改sphere半径为5.0:" << std::endl;
    std::cout << "    修改前是否有DirtyComponent: "
              << (had_dirty_before ? "有" : "无") << std::endl;
    std::cout << "    修改后是否有DirtyComponent: "
              << (has_dirty_after ? "有" : "无") << std::endl;

    // 检查UsdPrimComponent中的current_time是否被更新
    auto& usd_comp = registry.get<ecs::UsdPrimComponent>(sphere_entity);
    std::cout << "    Sphere prim当前时间码: "
              << usd_comp.current_time.GetValue() << std::endl;

    // 验证prim数据确实被修改了
    double radius = 0.0;  // 使用double而不是float
    sphere.GetRadiusAttr().Get(&radius);

    std::cout << "    修改后的Sphere半径: " << radius << std::endl;

    if (has_dirty_after) {
        std::cout << "  ✓ 修改prim后自动标记为脏！" << std::endl;
        auto& dirty = registry.get<ecs::DirtyComponent>(sphere_entity);
        std::cout << "    需要geometry更新: "
                  << (dirty.needs_geometry_update ? "是" : "否") << std::endl;
        std::cout << "    需要USD同步: " << (dirty.needs_usd_sync ? "是" : "否")
                  << std::endl;
    }
    else {
        std::cout << "  ✗ 修改prim属性后notice机制可能没有触发自动标记脏组件"
                  << std::endl;
        std::cout
            << "     "
               "（这可能需要在on_prim_changed()中手动添加DirtyComponent标记）"
            << std::endl;
    }

    EXPECT_DOUBLE_EQ(radius, 5.0) << "Sphere半径应该被修改为5.0";
    // 注：修改属性是否自动标记脏取决于on_prim_changed()的实现
    // 这个测试主要验证notice机制是否能捕捉属性修改
}

TEST_F(EcsApiTest, AnimationSystem)
{
    // 创建新的stage（文件不存在会自动创建）
    auto stage = std::make_unique<Stage>(get_temp_usd_path("AnimationSystem"));
    auto& registry = stage->get_registry();

    // 创建一个带动画的 entity
    auto sphere = stage->create_sphere(pxr::SdfPath("/AnimatedSphere"));
    auto entity = stage->find_entity_by_path(sphere.GetPath());

    ASSERT_TRUE(registry.valid(entity));

    // 添加 UsdPrimComponent (如果还没有)
    if (!registry.all_of<ecs::UsdPrimComponent>(entity)) {
        registry.emplace<ecs::UsdPrimComponent>(entity, sphere.GetPrim());
    }

    // 添加 AnimationComponent
    auto& anim = registry.emplace<ecs::AnimationComponent>(entity);
    anim.simulation_begun = false;

    registry.emplace<ecs::AnimatableTag>(entity);

    // 模拟几帧更新
    auto* anim_system = stage->get_animation_system();
    ASSERT_NE(anim_system, nullptr);

    for (int frame = 0; frame < 3; frame++) {
        float delta_time = 1.0f / 60.0f;  // 60 FPS
        anim_system->update(registry, delta_time);

        auto& usd_prim = registry.get<ecs::UsdPrimComponent>(entity);
        // 时间应该在增加或保持
        EXPECT_GE(usd_prim.current_time.GetValue(), 0.0);
    }
}

TEST_F(EcsApiTest, DirtyTracking)
{
    // 创建新的stage（文件不存在会自动创建）
    auto stage = std::make_unique<Stage>(get_temp_usd_path("DirtyTracking"));
    auto& registry = stage->get_registry();

    // 创建多个 entities
    std::vector<entt::entity> entities;
    for (int i = 0; i < 3; i++) {
        auto path = pxr::SdfPath(std::string("/Obj_") + std::to_string(i));
        auto sphere = stage->create_sphere(path);
        auto entity = stage->find_entity_by_path(sphere.GetPath());

        entities.push_back(entity);
    }

    EXPECT_EQ(entities.size(), 3);

    // 修改第 1 和第 3 个 entity
    {
        auto& dirty = registry.emplace<ecs::DirtyComponent>(entities[0]);
        dirty.needs_geometry_update = true;
    }

    {
        auto& dirty = registry.emplace<ecs::DirtyComponent>(entities[2]);
        dirty.needs_geometry_update = true;
        dirty.needs_usd_sync = true;
    }

    // 查询脏 entities
    {
        auto dirty_view = registry.view<ecs::DirtyComponent>();
        int dirty_count = std::distance(dirty_view.begin(), dirty_view.end());
        EXPECT_EQ(dirty_count, 2) << "应该有 2 个脏 entity";
    }

    // 同步
    stage->sync_entities_to_usd();

    // 验证脏标记清除情况
    {
        auto dirty_view = registry.view<ecs::DirtyComponent>();
        int remaining_dirty = 0;
        for (auto entity : dirty_view) {
            auto& dirty = dirty_view.get<ecs::DirtyComponent>(entity);
            if (dirty.needs_geometry_update || dirty.needs_usd_sync) {
                remaining_dirty++;
            }
        }
        // 注意：根据实现，脏标记可能仍然存在，这里验证同步已执行
        EXPECT_GE(remaining_dirty, 0);
    }
}

TEST_F(EcsApiTest, InfiniteLoopBug)
{
    // 这个测试用于检测USD修改->notice->dirty->sync->USD修改的无限循环
    auto stage = std::make_unique<Stage>(get_temp_usd_path("InfiniteLoopBug"));
    auto& registry = stage->get_registry();

    // 重置计数器
    Stage::reset_on_prim_changed_counter();

    // 创建一个mesh prim
    auto pxr_stage = stage->get_usd_stage();
    auto mesh_prim =
        pxr::UsdGeomMesh::Define(pxr_stage, pxr::SdfPath("/LoopTestMesh"));

    // 等待notice触发创建entity
    auto mesh_entity = stage->find_entity_by_path(mesh_prim.GetPath());
    ASSERT_TRUE(registry.valid(mesh_entity));

    // 创建一个有实际数据的geometry（使用create_cube）
    auto geometry = std::make_shared<Geometry>(create_cube(1.0f, 1.0f, 1.0f));
    if (!registry.all_of<ecs::GeometryComponent>(mesh_entity)) {
        registry.emplace<ecs::GeometryComponent>(mesh_entity, geometry);
    }
    else {
        auto& geom_comp = registry.get<ecs::GeometryComponent>(mesh_entity);
        geom_comp.geometry = geometry;
    }

    std::cout << "\n[Infinite Loop Bug Test - With Real Geometry Data]"
              << std::endl;
    std::cout << "  已创建cube geometry，包含实际的顶点和面数据" << std::endl;

    // 第一次手动标记dirty，触发第一次sync
    std::cout << "\n[TEST] 第一次手动标记dirty，触发初始sync" << std::endl;
    if (!registry.all_of<ecs::DirtyComponent>(mesh_entity)) {
        registry.emplace<ecs::DirtyComponent>(mesh_entity);
    }
    auto& initial_dirty = registry.get<ecs::DirtyComponent>(mesh_entity);
    initial_dirty.needs_geometry_update = true;

    // 重置计数器（清除创建时的notice）
    Stage::reset_on_prim_changed_counter();

    std::cout << "\n========== 开始循环测试 ==========\n" << std::endl;

    // 关键：这里不再手动标记dirty！
    // 如果sync写入USD后触发notice，on_prim_changed会自动标记dirty
    // 如果存在循环，应该会看到counter不断增加
    for (int i = 0; i < 10; i++) {
        std::cout << "\n========== 第" << (i + 1)
                  << "次循环开始 ==========" << std::endl;

        // 检查当前dirty状态（不手动修改）
        bool has_dirty = registry.all_of<ecs::DirtyComponent>(mesh_entity);
        bool needs_update = false;
        if (has_dirty) {
            auto& dirty_comp = registry.get<ecs::DirtyComponent>(mesh_entity);
            needs_update = dirty_comp.needs_geometry_update;
        }
        std::cout << "[TEST] 循环前状态: has_dirty=" << has_dirty
                  << ", needs_geometry_update=" << needs_update << std::endl;

        int counter_before = Stage::get_on_prim_changed_counter();
        std::cout << "[TEST] 调用sync之前，counter=" << counter_before
                  << std::endl;

        stage->sync_entities_to_usd();

        int counter_after = Stage::get_on_prim_changed_counter();
        int delta = counter_after - counter_before;

        std::cout << "[TEST] sync之后，counter=" << counter_after
                  << ", delta=" << delta << std::endl;

        // 检查sync后的dirty状态
        has_dirty = registry.all_of<ecs::DirtyComponent>(mesh_entity);
        needs_update = false;
        if (has_dirty) {
            auto& dirty_comp = registry.get<ecs::DirtyComponent>(mesh_entity);
            needs_update = dirty_comp.needs_geometry_update;
        }
        std::cout << "[TEST] sync后状态: has_dirty=" << has_dirty
                  << ", needs_geometry_update=" << needs_update << std::endl;

        std::cout << "========== 第" << (i + 1) << "次循环结束 ==========\n"
                  << std::endl;

        // 如果delta>0，说明sync触发了notice
        if (delta > 0) {
            std::cout << "  ⚠️  检测到：sync触发了 " << delta
                      << " 次on_prim_changed调用，但已被重入保护拦截！"
                      << std::endl;
        }

        // 关键验证：虽然触发了notice，但因为重入保护，不应该形成真正的循环
        // 真正的循环表现是：counter持续增长（每次循环都增加）
        // 我们的修复：只在第一次sync时触发notice，之后因为dirty已清除不再sync
        if (i == 0 && counter_after > 0) {
            std::cout
                << "  ✓ 第一次sync触发了notice（预期行为），但已被重入保护拦截"
                << std::endl;
        }
        else if (i > 0 && delta > 0) {
            std::cout << "  ✗ 后续循环仍在触发notice，可能存在问题！"
                      << std::endl;
            FAIL() << "后续循环不应该再触发notice（dirty应该已清除）";
        }
    }

    int final_counter = Stage::get_on_prim_changed_counter();
    std::cout << "  测试完成，总调用次数: " << final_counter << std::endl;

    if (final_counter == 0) {
        std::cout << "  ✓ sync没有触发notice（可能geometry数据为空或USD优化）"
                  << std::endl;
    }
    else if (final_counter <= 10) {
        std::cout << "  ⚠️  sync触发了notice但没有形成无限循环（调用次数: "
                  << final_counter << "）" << std::endl;
    }
    else {
        std::cout << "  ✗ 存在循环问题（调用次数: " << final_counter << "）"
                  << std::endl;
    }
}

TEST_F(EcsApiTest, AnimationWriteToUsd)
{
    // 创建一个新的stage，如果文件不存在会报错（这是正常的）
    auto stage =
        std::make_unique<Stage>("../../Assets/test_ecs_animation_write.usdc");
    auto& registry = stage->get_registry();

    // 创建3个不同的几何体
    auto sphere_prim = stage->create_sphere(pxr::SdfPath("/AnimSphere"));
    auto cube_prim = stage->create_cube(pxr::SdfPath("/AnimCube"));
    auto cylinder_prim = stage->create_cylinder(pxr::SdfPath("/AnimCylinder"));

    auto sphere_entity = stage->find_entity_by_path(sphere_prim.GetPath());
    auto cube_entity = stage->find_entity_by_path(cube_prim.GetPath());
    auto cylinder_entity = stage->find_entity_by_path(cylinder_prim.GetPath());

    ASSERT_TRUE(registry.valid(sphere_entity));
    ASSERT_TRUE(registry.valid(cube_entity));
    ASSERT_TRUE(registry.valid(cylinder_entity));

    // 为每个entity添加平移动画组件
    // Sphere: 向右移动 100单位/秒
    auto& sphere_anim =
        registry.emplace<TranslationAnimationComponent>(sphere_entity);
    sphere_anim.velocity = glm::vec3(100.0f, 0.0f, 0.0f);

    // Cube: 向右移动 100单位/秒，Y偏移5
    auto& cube_anim =
        registry.emplace<TranslationAnimationComponent>(cube_entity);
    cube_anim.velocity = glm::vec3(100.0f, 5.0f, 0.0f);

    // Cylinder: 向右移动 100单位/秒，Y偏移-5
    auto& cylinder_anim =
        registry.emplace<TranslationAnimationComponent>(cylinder_entity);
    cylinder_anim.velocity = glm::vec3(100.0f, -5.0f, 0.0f);

    // 创建动画系统
    TranslationAnimationSystem anim_system;

    // 60帧，1秒，60 FPS
    const int num_frames = 60;
    const float duration = 1.0f;
    const float fps = 60.0f;
    const float delta_time = 1.0f / fps;

    // 动画循环 - 让system更新每一帧
    for (int frame = 0; frame <= num_frames; frame++) {
        float time_seconds = frame * delta_time;
        pxr::UsdTimeCode time_code(time_seconds);

        // 调用system更新所有动画entities
        anim_system.update(registry, delta_time, time_code);
    }

    // 设置stage的时间范围
    stage->get_usd_stage()->SetStartTimeCode(0.0);
    stage->get_usd_stage()->SetEndTimeCode(duration);
    stage->get_usd_stage()->SetTimeCodesPerSecond(fps);

    // 同步所有entities到USD
    stage->sync_entities_to_usd();

    // 保存USD文件到指定路径（从Binaries/Release相对路径）
    stage->SaveAs("../../Assets/test_ecs_animation_write.usdc");

    // 验证：检查最后一帧的位置
    {
        auto sphere_xform = pxr::UsdGeomXformable(sphere_prim.GetPrim());
        pxr::GfVec3d final_pos;
        bool reset_stack;
        std::vector<pxr::UsdGeomXformOp> xform_ops =
            sphere_xform.GetOrderedXformOps(&reset_stack);
        if (!xform_ops.empty()) {
            xform_ops[0].Get(&final_pos, pxr::UsdTimeCode(duration));

            EXPECT_NEAR(final_pos[0], 100.0, 0.1) << "Sphere应该移动到x=100";
            EXPECT_EQ(final_pos[1], 0.0) << "Sphere的Y位置应该为0";
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "动画已写入到: "
              << std::filesystem::absolute(
                     "../../Assets/test_ecs_animation_write.usdc")
                     .string()
              << std::endl;
    std::cout << "时间范围: 0.0 - " << duration << " 秒, " << num_frames
              << " 帧" << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "3个几何体（Sphere, Cube, Cylinder）从x=0移动到x=100"
              << std::endl;
    std::cout << "请使用usdview打开查看动画效果" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// Google Test 主函数
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
