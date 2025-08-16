#include <gtest/gtest.h>
#include <pxr/base/gf/ray.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include <memory>

#include "GCore/Components/MeshComponent.h"
#include "GCore/algorithms/intersection.h"
#include "RHI/rhi.hpp"

using namespace USTC_CG;

class IntersectionTests : public ::testing::Test {
   protected:
    void SetUp() override
    {
        RHI::init();
        init_gpu_geometry_algorithms();
    }

    void TearDown() override
    {
        deinit_gpu_geometry_algorithms();
    }

    // Helper to create a simple triangle mesh
    Geometry CreateTriangleMesh()
    {
        Geometry mesh = Geometry::CreateMesh();
        auto meshComp = mesh.get_component<MeshComponent>();

        // Create a simple triangle
        pxr::VtArray<pxr::GfVec3f> vertices;
        vertices.push_back(pxr::GfVec3f(0.0f, 0.0f, 0.0f));
        vertices.push_back(pxr::GfVec3f(1.0f, 0.0f, 0.0f));
        vertices.push_back(pxr::GfVec3f(0.0f, 1.0f, 0.0f));

        pxr::VtArray<int> faceVertexCounts;
        faceVertexCounts.push_back(3);

        pxr::VtArray<int> faceVertexIndices;
        faceVertexIndices.push_back(0);
        faceVertexIndices.push_back(1);
        faceVertexIndices.push_back(2);

        pxr::VtArray<pxr::GfVec3f> normals;
        normals.push_back(pxr::GfVec3f(0.0f, 0.0f, 1.0f));
        normals.push_back(pxr::GfVec3f(0.0f, 0.0f, 1.0f));
        normals.push_back(pxr::GfVec3f(0.0f, 0.0f, 1.0f));

        meshComp->set_vertices(vertices);
        meshComp->set_face_vertex_counts(faceVertexCounts);
        meshComp->set_face_vertex_indices(faceVertexIndices);
        meshComp->set_normals(normals);

        return mesh;
    }
};

TEST_F(IntersectionTests, BasicRayIntersection)
{
    // Create a simple mesh
    Geometry mesh = CreateTriangleMesh();

    // Create rays for intersection
    pxr::VtArray<pxr::GfRay> rays;

    // Ray that should hit the triangle (inside the triangle bounds)
    rays.push_back(pxr::GfRay(
        pxr::GfVec3f(0.25f, 0.25f, 1.0f), pxr::GfVec3f(0.0f, 0.0f, -1.0f)));

    // Ray that should miss the triangle
    rays.push_back(pxr::GfRay(
        pxr::GfVec3f(-0.5f, 0.5f, 1.0f), pxr::GfVec3f(0.0f, 0.0f, -1.0f)));

    // Test intersection
    pxr::VtArray<PointSample> samples = Intersect(rays, mesh);

    // Check results
    ASSERT_EQ(samples.size(), 2);

    // First ray should hit
    EXPECT_TRUE(samples[0].valid);
    EXPECT_NEAR(samples[0].position[0], 0.25f, 1e-4);
    EXPECT_NEAR(samples[0].position[1], 0.25f, 1e-4);
    EXPECT_NEAR(samples[0].position[2], 0.0f, 1e-4);

    // Normal should be correct
    EXPECT_NEAR(samples[0].normal[0], 0.0f, 1e-4);
    EXPECT_NEAR(samples[0].normal[1], 0.0f, 1e-4);
    EXPECT_NEAR(samples[0].normal[2], 1.0f, 1e-4);

    // Second ray should miss
    EXPECT_FALSE(samples[1].valid);
}

TEST_F(IntersectionTests, RayParallelToTriangle)
{
    Geometry mesh = CreateTriangleMesh();

    pxr::VtArray<pxr::GfRay> rays;
    // Ray parallel to triangle
    rays.push_back(pxr::GfRay(
        pxr::GfVec3f(0.5f, 0.5f, 1.0f), pxr::GfVec3f(1.0f, 0.0f, 0.0f)));

    pxr::VtArray<PointSample> samples = Intersect(rays, mesh);

    ASSERT_EQ(samples.size(), 1);
    EXPECT_FALSE(samples[0].valid);
}

TEST_F(IntersectionTests, MultipleTriangles)
{
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    // Create two triangles
    pxr::VtArray<pxr::GfVec3f> vertices;
    vertices.push_back(pxr::GfVec3f(0.0f, 0.0f, 0.0f));
    vertices.push_back(pxr::GfVec3f(1.0f, 0.0f, 0.0f));
    vertices.push_back(pxr::GfVec3f(0.0f, 1.0f, 0.0f));
    vertices.push_back(pxr::GfVec3f(1.0f, 1.0f, 0.0f));

    pxr::VtArray<int> faceVertexCounts;
    faceVertexCounts.push_back(3);
    faceVertexCounts.push_back(3);

    pxr::VtArray<int> faceVertexIndices;
    faceVertexIndices.push_back(0);
    faceVertexIndices.push_back(1);
    faceVertexIndices.push_back(2);
    faceVertexIndices.push_back(1);
    faceVertexIndices.push_back(3);
    faceVertexIndices.push_back(2);

    pxr::VtArray<pxr::GfVec3f> normals;
    for (int i = 0; i < 6; i++) {
        normals.push_back(pxr::GfVec3f(0.0f, 0.0f, 1.0f));
    }

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);
    meshComp->set_normals(normals);

    // Create rays for intersection
    pxr::VtArray<pxr::GfRay> rays;
    // Hit first triangle
    rays.push_back(pxr::GfRay(
        pxr::GfVec3f(0.25f, 0.25f, 1.0f), pxr::GfVec3f(0.0f, 0.0f, -1.0f)));
    // Hit second triangle
    rays.push_back(pxr::GfRay(
        pxr::GfVec3f(0.75f, 0.75f, 1.0f), pxr::GfVec3f(0.0f, 0.0f, -1.0f)));

    pxr::VtArray<PointSample> samples = Intersect(rays, mesh);

    ASSERT_EQ(samples.size(), 2);
    EXPECT_TRUE(samples[0].valid);
    EXPECT_TRUE(samples[1].valid);

    EXPECT_NEAR(samples[0].position[0], 0.25f, 1e-4);
    EXPECT_NEAR(samples[0].position[1], 0.25f, 1e-4);
    EXPECT_NEAR(samples[0].position[2], 0.0f, 1e-4);

    EXPECT_NEAR(samples[1].position[0], 0.75f, 1e-4);
    EXPECT_NEAR(samples[1].position[1], 0.75f, 1e-4);
    EXPECT_NEAR(samples[1].position[2], 0.0f, 1e-4);
}

TEST_F(IntersectionTests, EmptyMesh)
{
    Geometry mesh = Geometry::CreateMesh();

    pxr::VtArray<pxr::GfRay> rays;
    rays.push_back(pxr::GfRay(
        pxr::GfVec3f(0.5f, 0.5f, 1.0f), pxr::GfVec3f(0.0f, 0.0f, -1.0f)));

    pxr::VtArray<PointSample> samples = Intersect(rays, mesh);

    ASSERT_EQ(samples.size(), 1);
    EXPECT_FALSE(samples[0].valid);
}

TEST_F(IntersectionTests, TransformedMesh)
{
    Geometry mesh = CreateTriangleMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    // Apply a transform to the mesh (move it to z=-1)
    pxr::GfMatrix4d transform;
    transform.SetTranslate(pxr::GfVec3d(0.0, 0.0, -1.0));
    meshComp->apply_transform(transform);

    pxr::VtArray<pxr::GfRay> rays;
    rays.push_back(pxr::GfRay(
        pxr::GfVec3f(0.5f, 0.5f, 1.0f), pxr::GfVec3f(0.0f, 0.0f, -1.0f)));

    pxr::VtArray<PointSample> samples = Intersect(rays, mesh);

    ASSERT_EQ(samples.size(), 1);
    EXPECT_TRUE(samples[0].valid);

    EXPECT_NEAR(samples[0].position[0], 0.5f, 1e-4);
    EXPECT_NEAR(samples[0].position[1], 0.5f, 1e-4);
    EXPECT_NEAR(samples[0].position[2], -1.0f, 1e-4);
}
