#include <GCore/Components/MeshComponent.h>
#include <GCore/GOP.h>
#include <GCore/util_openmesh_bind.h>
#include <GCore/create_geom.h>
#include <GCore/algorithms/tetgen_algorithm.h>
#include <GCore/util_openmesh_bind.h>
#include <gtest/gtest.h>
#include <rzsim/rzsim.h>

#include <OpenVolumeMesh/Mesh/TetrahedralMesh.hh>

// Forward declarations for CUDA initialization
namespace Ruzino {
namespace cuda {
    extern int cuda_init();
    extern int cuda_shutdown();
}  // namespace cuda
}  // namespace Ruzino

#include <iostream>
#include <set>
#include <tuple>
#include <algorithm>

using namespace Ruzino;

// ============================================================================
// Surface Mesh Tests (Triangles)
// ============================================================================

// Helper to verify surface adjacency structure
void verify_surface_adjacency(
    const std::vector<unsigned>& adjacency_data,
    const std::vector<unsigned>& offset_buffer,
    size_t expected_vertex_count,
    const std::vector<std::vector<std::pair<unsigned, unsigned>>>&
        expected_pairs)
{
    ASSERT_EQ(offset_buffer.size(), expected_vertex_count);
    ASSERT_EQ(expected_pairs.size(), expected_vertex_count);

    for (size_t v = 0; v < expected_vertex_count; ++v) {
        unsigned offset = offset_buffer[v];
        unsigned count = adjacency_data[offset];

        ASSERT_EQ(count, expected_pairs[v].size())
            << "Vertex " << v << " should have " << expected_pairs[v].size()
            << " opposite edge pairs";

        // Collect actual pairs
        std::vector<std::pair<unsigned, unsigned>> actual_pairs;
        for (unsigned i = 0; i < count; ++i) {
            unsigned a = adjacency_data[offset + 1 + i * 2];
            unsigned b = adjacency_data[offset + 1 + i * 2 + 1];
            actual_pairs.push_back({ a, b });
        }

        // Verify all expected pairs exist (order may vary)
        for (const auto& expected_pair : expected_pairs[v]) {
            bool found = false;
            for (const auto& actual_pair : actual_pairs) {
                if (actual_pair == expected_pair) {
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found)
                << "Vertex " << v << " missing edge pair ("
                << expected_pair.first << ", " << expected_pair.second << ")";
        }
    }
}

TEST(SurfaceAdjacency, SingleTriangle)
{
    // Triangle (0, 1, 2)
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    std::vector<glm::vec3> vertices = {
        glm::vec3(0.0f, 0.0f, 0.0f),  // 0
        glm::vec3(1.0f, 0.0f, 0.0f),  // 1
        glm::vec3(0.0f, 1.0f, 0.0f)   // 2
    };

    std::vector<int> faceVertexCounts = { 3 };
    std::vector<int> faceVertexIndices = { 0, 1, 2 };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto [adjacencyCPU, offsetCPU] = get_surface_adjacency(mesh);

    std::cout << "\n=== Single Triangle Surface Adjacency ===\n";
    for (size_t i = 0; i < offsetCPU.size(); ++i) {
        unsigned offset = offsetCPU[i];
        unsigned count = adjacencyCPU[offset];
        std::cout << "V" << i << " (count=" << count << "): ";
        for (unsigned j = 0; j < count; ++j) {
            unsigned a = adjacencyCPU[offset + 1 + j * 2];
            unsigned b = adjacencyCPU[offset + 1 + j * 2 + 1];
            std::cout << "(" << a << "," << b << ") ";
        }
        std::cout << "\n";
    }

    verify_surface_adjacency(
        adjacencyCPU,
        offsetCPU,
        3,
        {
            { { 1, 2 } },  // v0: opposite edge (v1, v2)
            { { 2, 0 } },  // v1: opposite edge (v2, v0)
            { { 0, 1 } }   // v2: opposite edge (v0, v1)
        });
}

TEST(SurfaceAdjacency, TwoTriangles)
{
    // Two triangles sharing edge: (0,1,2) and (0,2,3)
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    std::vector<glm::vec3> vertices = {
        glm::vec3(0.5f, 1.0f, 0.0f),  // 0
        glm::vec3(0.0f, 0.0f, 0.0f),  // 1
        glm::vec3(0.5f, 0.0f, 0.0f),  // 2
        glm::vec3(1.0f, 0.0f, 0.0f)   // 3
    };

    std::vector<int> faceVertexCounts = { 3, 3 };
    std::vector<int> faceVertexIndices = { 0, 1, 2, 0, 2, 3 };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto [adjacencyCPU, offsetCPU] = get_surface_adjacency(mesh);

    std::cout << "\n=== Two Triangles Surface Adjacency ===\n";
    for (size_t i = 0; i < offsetCPU.size(); ++i) {
        unsigned offset = offsetCPU[i];
        unsigned count = adjacencyCPU[offset];
        std::cout << "V" << i << " (count=" << count << "): ";
        for (unsigned j = 0; j < count; ++j) {
            unsigned a = adjacencyCPU[offset + 1 + j * 2];
            unsigned b = adjacencyCPU[offset + 1 + j * 2 + 1];
            std::cout << "(" << a << "," << b << ") ";
        }
        std::cout << "\n";
    }

    verify_surface_adjacency(
        adjacencyCPU,
        offsetCPU,
        4,
        {
            { { 1, 2 }, { 2, 3 } },  // v0: in 2 triangles (0,1,2) and (0,2,3)
            { { 2, 0 } },  // v1: in triangle (0,1,2), opposite edge (2,0)
            { { 3, 0 }, { 0, 1 } },  // v2: in both triangles
            { { 0, 2 } }  // v3: in triangle (0,2,3), opposite edge (0,2)
        });
}

TEST(SurfaceAdjacency, TriangleFan)
{
    // 4 triangles sharing central vertex 0: (0,1,2), (0,2,3), (0,3,4), (0,4,1)
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    std::vector<glm::vec3> vertices = {
        glm::vec3(0.5f, 0.5f, 0.0f),  // 0 (center)
        glm::vec3(1.0f, 0.0f, 0.0f),  // 1
        glm::vec3(1.0f, 1.0f, 0.0f),  // 2
        glm::vec3(0.0f, 1.0f, 0.0f),  // 3
        glm::vec3(0.0f, 0.0f, 0.0f)   // 4
    };

    std::vector<int> faceVertexCounts = { 3, 3, 3, 3 };
    std::vector<int> faceVertexIndices = { 0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1 };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto [adjacencyCPU, offsetCPU] = get_surface_adjacency(mesh);

    std::cout << "\n=== Triangle Fan Surface Adjacency ===\n";
    for (size_t i = 0; i < offsetCPU.size(); ++i) {
        unsigned offset = offsetCPU[i];
        unsigned count = adjacencyCPU[offset];
        std::cout << "V" << i << " (count=" << count << "): ";
        for (unsigned j = 0; j < count; ++j) {
            unsigned a = adjacencyCPU[offset + 1 + j * 2];
            unsigned b = adjacencyCPU[offset + 1 + j * 2 + 1];
            std::cout << "(" << a << "," << b << ") ";
        }
        std::cout << "\n";
    }

    // Vertex 0 is in 4 triangles
    unsigned offset0 = offsetCPU[0];
    unsigned count0 = adjacencyCPU[offset0];
    EXPECT_EQ(count0, 4) << "Central vertex should have 4 opposite edge pairs";
}

// ============================================================================
// Volume Mesh Tests (Tetrahedra)
// ============================================================================

// Check if two triangles are the same (allow any cyclic permutation, including reverse)
// E.g., (1,2,3), (2,3,1), (3,1,2), (3,2,1), (2,1,3), (1,3,2) all represent the same triangle
bool triangles_match_oriented(
    unsigned a0, unsigned a1, unsigned a2,
    unsigned b0, unsigned b1, unsigned b2)
{
    // Check all cyclic permutations (CCW)
    if ((a0 == b0 && a1 == b1 && a2 == b2) ||
        (a0 == b1 && a1 == b2 && a2 == b0) ||
        (a0 == b2 && a1 == b0 && a2 == b1))
        return true;
    
    // Check reverse cyclic permutations (CW)
    if ((a0 == b0 && a1 == b2 && a2 == b1) ||
        (a0 == b1 && a1 == b0 && a2 == b2) ||
        (a0 == b2 && a1 == b1 && a2 == b0))
        return true;
    
    return false;
}

// Helper to verify volume adjacency structure
void verify_volume_adjacency(
    const std::vector<unsigned>& adjacency_data,
    const std::vector<unsigned>& offset_buffer,
    size_t expected_vertex_count,
    const std::vector<std::vector<std::tuple<unsigned, unsigned, unsigned>>>&
        expected_triplets)
{
    ASSERT_EQ(offset_buffer.size(), expected_vertex_count);
    ASSERT_EQ(expected_triplets.size(), expected_vertex_count);

    for (size_t v = 0; v < expected_vertex_count; ++v) {
        unsigned offset = offset_buffer[v];
        unsigned count = adjacency_data[offset];

        ASSERT_EQ(count, expected_triplets[v].size())
            << "Vertex " << v << " should have " << expected_triplets[v].size()
            << " opposite face triplets";

        // Collect actual triplets
        std::vector<std::tuple<unsigned, unsigned, unsigned>> actual_triplets;
        for (unsigned i = 0; i < count; ++i) {
            unsigned a = adjacency_data[offset + 1 + i * 3];
            unsigned b = adjacency_data[offset + 1 + i * 3 + 1];
            unsigned c = adjacency_data[offset + 1 + i * 3 + 2];
            actual_triplets.push_back({ a, b, c });
        }

        // Verify all expected triplets exist (allowing cyclic permutations)
        for (const auto& expected_triplet : expected_triplets[v]) {
            bool found = false;
            auto [exp_a, exp_b, exp_c] = expected_triplet;
            for (const auto& actual_triplet : actual_triplets) {
                auto [act_a, act_b, act_c] = actual_triplet;
                if (triangles_match_oriented(exp_a, exp_b, exp_c, act_a, act_b, act_c)) {
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found) << "Vertex " << v << " missing face triplet ("
                               << exp_a << ", " << exp_b << ", " << exp_c << ")";
        }
    }
}

TEST(VolumeAdjacency, SingleTetrahedron)
{
    // Single tetrahedron (0, 1, 2, 3)
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    std::vector<glm::vec3> vertices = {
        glm::vec3(0.0f, 0.0f, 0.0f),  // 0
        glm::vec3(1.0f, 0.0f, 0.0f),  // 1
        glm::vec3(0.5f, 1.0f, 0.0f),  // 2
        glm::vec3(0.5f, 0.5f, 1.0f)   // 3
    };

    // Tetrahedron has 4 triangular faces
    std::vector<int> faceVertexCounts = { 3, 3, 3, 3 };
    std::vector<int> faceVertexIndices = {
        1, 2, 3,  // face opposite to v0
        0, 3, 2,  // face opposite to v1
        0, 1, 3,  // face opposite to v2
        0, 2, 1   // face opposite to v3
    };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto [adjacencyCPU, offsetCPU] = get_volume_adjacency(mesh);

    std::cout << "\n=== Single Tetrahedron Volume Adjacency ===\n";
    for (size_t i = 0; i < offsetCPU.size(); ++i) {
        unsigned offset = offsetCPU[i];
        unsigned count = adjacencyCPU[offset];
        std::cout << "V" << i << " (count=" << count << "): ";
        for (unsigned j = 0; j < count; ++j) {
            unsigned a = adjacencyCPU[offset + 1 + j * 3];
            unsigned b = adjacencyCPU[offset + 1 + j * 3 + 1];
            unsigned c = adjacencyCPU[offset + 1 + j * 3 + 2];
            std::cout << "(" << a << "," << b << "," << c << ") ";
        }
        std::cout << "\n";
    }

    verify_volume_adjacency(
        adjacencyCPU,
        offsetCPU,
        4,
        {
            { { 1, 2, 3 } },  // v0: opposite face (v1, v2, v3)
            { { 0, 3, 2 } },  // v1: opposite face (v0, v3, v2)
            { { 0, 1, 3 } },  // v2: opposite face (v0, v1, v3)
            { { 0, 2, 1 } }   // v3: opposite face (v0, v2, v1)
        });
}

TEST(VolumeAdjacency, TwoTetrahedra)
{
    // Two tets sharing a face: (0,1,2,3) and (0,2,1,4)
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    std::vector<glm::vec3> vertices = {
        glm::vec3(0.0f, 0.0f, 0.0f),  // 0
        glm::vec3(1.0f, 0.0f, 0.0f),  // 1
        glm::vec3(0.5f, 1.0f, 0.0f),  // 2
        glm::vec3(0.5f, 0.5f, 1.0f),  // 3
        glm::vec3(0.5f, 0.5f, -1.0f)  // 4
    };

    // Two tetrahedra sharing face (0,1,2)
    // Tet 1: (0,1,2,3)
    // Tet 2: (0,1,2,4)
    std::vector<int> faceVertexCounts = { 3, 3, 3, 3, 3, 3, 3 };
    std::vector<int> faceVertexIndices = {
        // Tet 1 faces
        1, 2, 3,  // opposite to v0
        0, 2, 3,  // opposite to v1
        0, 1, 3,  // opposite to v2
        // Tet 2 faces (excluding shared face)
        1, 2, 4,  // opposite to v0
        0, 2, 4,  // opposite to v1
        0, 1, 4,  // opposite to v2
        // Shared face
        0, 1, 2   // opposite to v3 (in Tet1) and v4 (in Tet2)
    };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto [adjacencyCPU, offsetCPU] = get_volume_adjacency(mesh);

    std::cout << "\n=== Two Tetrahedra Volume Adjacency ===\n";
    for (size_t i = 0; i < offsetCPU.size(); ++i) {
        unsigned offset = offsetCPU[i];
        unsigned count = adjacencyCPU[offset];
        std::cout << "V" << i << " (count=" << count << "): ";
        for (unsigned j = 0; j < count; ++j) {
            unsigned a = adjacencyCPU[offset + 1 + j * 3];
            unsigned b = adjacencyCPU[offset + 1 + j * 3 + 1];
            unsigned c = adjacencyCPU[offset + 1 + j * 3 + 2];
            std::cout << "(" << a << "," << b << "," << c << ") ";
        }
        std::cout << "\n";
    }

    // V0, V1, V2 are shared by both tets, each has 2 opposite faces
    // V3 only in Tet1, has 1 opposite face: (0,1,2)
    // V4 only in Tet2, has 1 opposite face: (0,1,2)
    verify_volume_adjacency(
        adjacencyCPU,
        offsetCPU,
        5,
        {
            { { 1, 2, 3 }, { 1, 2, 4 } },  // v0 in both tets
            { { 0, 2, 3 }, { 0, 2, 4 } },  // v1 in both tets
            { { 0, 1, 3 }, { 0, 1, 4 } },  // v2 in both tets
            { { 0, 1, 2 } },  // v3 only in Tet1, opposite face is shared face
            { { 0, 1, 2 } }   // v4 only in Tet2, opposite face is shared face
        });
}

// ============================================================================
// OpenVolumeMesh Validation Tests
// ============================================================================

// Compare our adjacency results with OpenVolumeMesh
void verify_against_openvolulemesh(const Geometry& mesh)
{
    auto volumemesh = operand_to_openvolumemesh(const_cast<Geometry*>(&mesh));
    auto [adjacencyCPU, offsetCPU] = get_volume_adjacency(mesh);
    
    std::cout << "\n=== OpenVolumeMesh Validation ===\n";
    std::cout << "Vertices: " << volumemesh->n_vertices() << "\n";
    std::cout << "Cells: " << volumemesh->n_cells() << "\n";
    std::cout << "Faces: " << volumemesh->n_faces() << "\n";
    
    // For each vertex, get opposite faces from OpenVolumeMesh
    // Note: We iterate through ALL cells and extract faces for each vertex
    // rather than using vc_iter, since vc_iter may be incomplete when _topologyCheck=false
    std::vector<std::vector<std::tuple<unsigned, unsigned, unsigned>>> ovm_opposite_by_vertex(
        volumemesh->n_vertices());
    
    for (auto c_it = volumemesh->cells_begin(); c_it != volumemesh->cells_end(); ++c_it) {
        // Get all faces of this cell
        auto cell_faces = volumemesh->cell(*c_it).halffaces();
        
        for (auto hf : cell_faces) {
            // Get vertices of this face in order
            std::vector<unsigned> face_v;
            auto he_begin = volumemesh->halfface(hf).halfedges()[0];
            auto he = he_begin;
            do {
                auto from_v = volumemesh->halfedge(he).from_vertex();
                face_v.push_back(from_v.idx());
                he = volumemesh->next_halfedge_in_halfface(he, hf);
            } while (he != he_begin && face_v.size() < 10);
            
            if (face_v.size() == 3) {
                // This face is opposite to the 4th vertex of the cell
                auto cv_it = volumemesh->cv_iter(*c_it);
                std::set<unsigned> cell_verts;
                for (; cv_it.valid(); ++cv_it) {
                    cell_verts.insert((*cv_it).idx());
                }
                
                // Find the vertex NOT in this face
                for (unsigned v_idx : cell_verts) {
                    if (std::find(face_v.begin(), face_v.end(), v_idx) == face_v.end()) {
                        // v_idx is the opposite vertex
                        if (v_idx < volumemesh->n_vertices()) {
                            ovm_opposite_by_vertex[v_idx].push_back(
                                {face_v[0], face_v[1], face_v[2]});
                        }
                    }
                }
            }
        }
    }
    
    // Now compare OVM (extracted from full cell iteration) with GPU results
    int mismatch_count = 0;
    for (unsigned v_idx = 0; v_idx < volumemesh->n_vertices(); ++v_idx) {
        // Collect opposite faces from our implementation (with orientation)
        std::vector<std::tuple<unsigned, unsigned, unsigned>> our_opposite_faces;
        unsigned offset = offsetCPU[v_idx];
        unsigned count = adjacencyCPU[offset];
        
        for (unsigned i = 0; i < count; ++i) {
            unsigned a = adjacencyCPU[offset + 1 + i * 3];
            unsigned b = adjacencyCPU[offset + 1 + i * 3 + 1];
            unsigned c = adjacencyCPU[offset + 1 + i * 3 + 2];
            our_opposite_faces.push_back({a, b, c});
        }
        
        const auto& ovm_opposite_faces = ovm_opposite_by_vertex[v_idx];
        
        // Check if counts match
        if (ovm_opposite_faces.size() != our_opposite_faces.size()) {
            std::cout << "V" << v_idx << ": OVM=" << ovm_opposite_faces.size() 
                      << ", Ours=" << our_opposite_faces.size() << " ✗ COUNT MISMATCH!\n";
            mismatch_count++;
            EXPECT_EQ(ovm_opposite_faces.size(), our_opposite_faces.size()) 
                << "Vertex " << v_idx << " opposite face count mismatch";
            continue;
        }
        
        // Check if all faces match with correct orientation
        bool all_match = true;
        std::vector<bool> ovm_matched(ovm_opposite_faces.size(), false);
        
        for (const auto& [a, b, c] : our_opposite_faces) {
            bool found = false;
            for (size_t j = 0; j < ovm_opposite_faces.size(); ++j) {
                if (!ovm_matched[j]) {
                    auto [oa, ob, oc] = ovm_opposite_faces[j];
                    if (triangles_match_oriented(a, b, c, oa, ob, oc)) {
                        ovm_matched[j] = true;
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                all_match = false;
                std::cout << "V" << v_idx << " ✗ ORIENTATION MISMATCH!\n";
                std::cout << "  Missing oriented face: (" << a << "," << b << "," << c << ")\n";
                mismatch_count++;
                break;
            }
        }
        
        EXPECT_TRUE(all_match) 
            << "Vertex " << v_idx << " opposite faces orientation mismatch";
    }
    
    if (mismatch_count == 0) {
        std::cout << "Full OVM validation successful for " << volumemesh->n_cells() << " tetrahedra\n";
    }
}

TEST(VolumeAdjacency, CubeWithFiveTets)
{
    // A cube subdivided into 5 tetrahedra (correct subdivision)
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    // Cube vertices
    std::vector<glm::vec3> vertices = {
        glm::vec3(0.0f, 0.0f, 0.0f),  // 0
        glm::vec3(1.0f, 0.0f, 0.0f),  // 1
        glm::vec3(1.0f, 1.0f, 0.0f),  // 2
        glm::vec3(0.0f, 1.0f, 0.0f),  // 3
        glm::vec3(0.0f, 0.0f, 1.0f),  // 4
        glm::vec3(1.0f, 0.0f, 1.0f),  // 5
        glm::vec3(1.0f, 1.0f, 1.0f),  // 6
        glm::vec3(0.0f, 1.0f, 1.0f)   // 7
    };

    // 5 tetrahedra along diagonal (0,6) - standard cube subdivision
    std::vector<int> faceVertexCounts(20, 3);
    std::vector<int> faceVertexIndices = {
        // Tet 1: (0,1,2,6)
        1,2,6, 0,6,2, 0,1,6, 0,2,1,
        // Tet 2: (0,4,5,6)
        4,5,6, 0,6,5, 0,4,6, 0,5,4,
        // Tet 3: (0,1,5,6)
        1,5,6, 0,6,5, 0,1,6, 0,5,1,
        // Tet 4: (0,2,3,6)
        2,3,6, 0,6,3, 0,2,6, 0,3,2,
        // Tet 5: (0,4,6,7)
        4,6,7, 0,7,6, 0,4,7, 0,6,4
    };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    verify_against_openvolulemesh(mesh);
}

TEST(VolumeAdjacency, OctahedronWithEightTets)
{
    // Regular octahedron subdivided into 8 tetrahedra
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    std::vector<glm::vec3> vertices = {
        glm::vec3(0.0f, 0.0f, 0.0f),   // 0 center
        glm::vec3(1.0f, 0.0f, 0.0f),   // 1
        glm::vec3(0.0f, 1.0f, 0.0f),   // 2
        glm::vec3(-1.0f, 0.0f, 0.0f),  // 3
        glm::vec3(0.0f, -1.0f, 0.0f),  // 4
        glm::vec3(0.0f, 0.0f, 1.0f),   // 5 top
        glm::vec3(0.0f, 0.0f, -1.0f)   // 6 bottom
    };

    // 8 tetrahedra from center to each octahedron face
    std::vector<int> faceVertexCounts(32, 3);
    std::vector<int> faceVertexIndices = {
        // Top pyramid (4 tets)
        // Tet: (0,1,2,5)
        1,2,5, 0,5,2, 0,1,5, 0,2,1,
        // Tet: (0,2,3,5)
        2,3,5, 0,5,3, 0,2,5, 0,3,2,
        // Tet: (0,3,4,5)
        3,4,5, 0,5,4, 0,3,5, 0,4,3,
        // Tet: (0,4,1,5)
        4,1,5, 0,5,1, 0,4,5, 0,1,4,
        // Bottom pyramid (4 tets)
        // Tet: (0,2,1,6)
        2,1,6, 0,6,1, 0,2,6, 0,1,2,
        // Tet: (0,3,2,6)
        3,2,6, 0,6,2, 0,3,6, 0,2,3,
        // Tet: (0,4,3,6)
        4,3,6, 0,6,3, 0,4,6, 0,3,4,
        // Tet: (0,1,4,6)
        1,4,6, 0,6,4, 0,1,6, 0,4,1
    };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    verify_against_openvolulemesh(mesh);
}

TEST(VolumeAdjacency, PrismWithThreeTets)
{
    // Triangular prism subdivided into 3 tetrahedra
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    std::vector<glm::vec3> vertices = {
        // Bottom triangle
        glm::vec3(0.0f, 0.0f, 0.0f),   // 0
        glm::vec3(1.0f, 0.0f, 0.0f),   // 1
        glm::vec3(0.5f, 1.0f, 0.0f),   // 2
        // Top triangle
        glm::vec3(0.0f, 0.0f, 1.0f),   // 3
        glm::vec3(1.0f, 0.0f, 1.0f),   // 4
        glm::vec3(0.5f, 1.0f, 1.0f)    // 5
    };

    // 3 tetrahedra subdividing the prism
    std::vector<int> faceVertexCounts(12, 3);
    std::vector<int> faceVertexIndices = {
        // Tet 1: (0,1,2,3)
        1,2,3, 0,3,2, 0,1,3, 0,2,1,
        // Tet 2: (1,2,3,4)
        2,3,4, 1,4,3, 1,2,4, 1,3,2,
        // Tet 3: (2,3,4,5)
        3,4,5, 2,5,4, 2,3,5, 2,4,3
    };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    verify_against_openvolulemesh(mesh);
}

// ============================================================================
// TetGen-based Volume Mesh Tests
// ============================================================================

TEST(VolumeAdjacency, TetgenCube)
{
    // Create a cube and tetrahedralize it
    Geometry cube = create_cube(2.0f, 2.0f, 2.0f);
    
    // Triangulate first (cube has quad faces)
    auto omesh = operand_to_openmesh(&cube);
    omesh->triangulate();
    Geometry triangulated_cube = *openmesh_to_operand(omesh.get());
    
    geom_algorithm::TetgenParams params;
    params.max_volume = 0.5f;
    params.quality_ratio = 2.0f;
    
    Geometry tet_mesh = geom_algorithm::tetrahedralize(triangulated_cube, params);
    auto mesh_comp = tet_mesh.get_const_component<MeshComponent>();
    
    // Count tetrahedra (each tet has 4 faces)
    int num_faces = mesh_comp->get_face_vertex_counts().size();
    int num_tets = num_faces / 4;
    
    std::cout << "\n=== TetGen Cube ===\n";
    std::cout << "Tetrahedra: " << num_tets << "\n";
    std::cout << "Vertices: " << mesh_comp->get_vertices().size() << "\n";
    
    // Debug: Check how many times face (9,12,13) appears
    {
        // Print all tetrahedra in TetGen output
        const auto& indices = mesh_comp->get_face_vertex_indices();
        std::map<std::set<int>, int> tet_map;
        
        for (int t = 0; t < num_tets; ++t) {
            int base = t * 12;
            std::set<int> tet_verts;
            for (int f = 0; f < 4; ++f) {
                for (int v = 0; v < 3; ++v) {
                    tet_verts.insert(indices[base + f * 3 + v]);
                }
            }
            tet_map[tet_verts]++;
        }
        
        // Check for specific tets
        std::set<int> target_1 = {6, 8, 10, 11};
        std::set<int> target_2 = {6, 8, 11, 10};
        std::set<int> target_3 = {8, 10, 11, 6};
        
        bool found_6_8_10_11 = false;
        if (tet_map.count(target_1)) {
            std::cout << "✓ Tetrahedron {6,8,10,11} FOUND in TetGen output\n";
            found_6_8_10_11 = true;
        } else {
            std::cout << "✗ Tetrahedron {6,8,10,11} NOT in TetGen output\n";
        }
        
        if (!found_6_8_10_11) {
            std::cout << "Available tets containing {6,8,10,11} vertices:\n";
            for (const auto& [tet, count] : tet_map) {
                bool has_6 = tet.count(6) > 0;
                bool has_8 = tet.count(8) > 0;
                bool has_10 = tet.count(10) > 0;
                bool has_11 = tet.count(11) > 0;
                
                if ((has_6 + has_8 + has_10 + has_11) >= 3) {
                    std::cout << "  {";
                    bool first = true;
                    for (int v : tet) {
                        if (!first) std::cout << ",";
                        std::cout << v;
                        first = false;
                    }
                    std::cout << "}\n";
                }
            }
        }
    }

    
    // Debug: print all tetrahedra to check for (2,9,12,13)
    {
        const auto& indices = mesh_comp->get_face_vertex_indices();
        std::map<std::set<int>, int> tet_sets;
        for (int t = 0; t < num_tets; ++t) {
            int base = t * 12; // 4 faces * 3 vertices
            std::set<int> tet_vertices;
            for (int f = 0; f < 4; ++f) {
                for (int v = 0; v < 3; ++v) {
                    tet_vertices.insert(indices[base + f * 3 + v]);
                }
            }
            tet_sets[tet_vertices]++;
        }
        
        // Check if (2,9,12,13) exists
        std::set<int> target_tet = {2, 9, 12, 13};
        if (tet_sets.count(target_tet)) {
            std::cout << "WARNING: Tetrahedron (2,9,12,13) FOUND in TetGen output!\n";
            // Print vertex coordinates
            const auto& verts = mesh_comp->get_vertices();
            std::cout << "  V2: (" << verts[2].x << ", " << verts[2].y << ", " << verts[2].z << ")\n";
            std::cout << "  V9: (" << verts[9].x << ", " << verts[9].y << ", " << verts[9].z << ")\n";
            std::cout << "  V12: (" << verts[12].x << ", " << verts[12].y << ", " << verts[12].z << ")\n";
            std::cout << "  V13: (" << verts[13].x << ", " << verts[13].y << ", " << verts[13].z << ")\n";
            
            // Compute volume to check if degenerate
            glm::vec3 v2 = verts[2];
            glm::vec3 v9 = verts[9];
            glm::vec3 v12 = verts[12];
            glm::vec3 v13 = verts[13];
            glm::vec3 a = v9 - v2;
            glm::vec3 b = v12 - v2;
            glm::vec3 c = v13 - v2;
            float volume = std::abs(glm::dot(a, glm::cross(b, c))) / 6.0f;
            std::cout << "  Tetrahedron volume: " << volume << "\n";
            if (volume < 1e-6f) {
                std::cout << "  -> DEGENERATE (near-zero volume)!\n";
            }
        } else {
            std::cout << "Tetrahedron (2,9,12,13) NOT found in TetGen output (as expected)\n";
        }
    }
    
    EXPECT_GT(num_tets, 10) << "Should generate at least 10 tetrahedra";
    EXPECT_LT(num_tets, 2000) << "Should not generate excessive tetrahedra";
    
    // Verify with OpenVolumeMesh
    verify_against_openvolulemesh(tet_mesh);
}

TEST(VolumeAdjacency, TetgenIcoSphere)
{
    // Create an icosphere and tetrahedralize it
    Geometry sphere = create_ico_sphere(1, 1.0f);
    
    geom_algorithm::TetgenParams params;
    params.max_volume = 0.3f;
    params.quality_ratio = 2.0f;
    
    Geometry tet_mesh = geom_algorithm::tetrahedralize(sphere, params);
    auto mesh_comp = tet_mesh.get_const_component<MeshComponent>();
    
    int num_faces = mesh_comp->get_face_vertex_counts().size();
    int num_tets = num_faces / 4;
    
    std::cout << "\n=== TetGen IcoSphere ===\n";
    std::cout << "Tetrahedra: " << num_tets << "\n";
    std::cout << "Vertices: " << mesh_comp->get_vertices().size() << "\n";
    
    EXPECT_GT(num_tets, 20) << "Should generate reasonable number of tetrahedra";
    EXPECT_LT(num_tets, 3000) << "Should not generate excessive tetrahedra";
    
    // Verify with OpenVolumeMesh
    verify_against_openvolulemesh(tet_mesh);
}

TEST(VolumeAdjacency, TetgenUVSphere)
{
    // Create a UV sphere and tetrahedralize it
    Geometry sphere = create_uv_sphere(8, 6, 1.5f);
    
    // Triangulate first (UV sphere has quad faces)
    auto omesh = operand_to_openmesh(&sphere);
    omesh->triangulate();
    Geometry triangulated_sphere = *openmesh_to_operand(omesh.get());
    
    geom_algorithm::TetgenParams params;
    params.max_volume = 0.4f;
    params.quality_ratio = 2.0f;
    
    Geometry tet_mesh = geom_algorithm::tetrahedralize(triangulated_sphere, params);
    auto mesh_comp = tet_mesh.get_const_component<MeshComponent>();
    
    int num_faces = mesh_comp->get_face_vertex_counts().size();
    int num_tets = num_faces / 4;
    
    std::cout << "\n=== TetGen UV Sphere ===\n";
    std::cout << "Tetrahedra: " << num_tets << "\n";
    std::cout << "Vertices: " << mesh_comp->get_vertices().size() << "\n";
    
    EXPECT_GT(num_tets, 50) << "Should generate substantial tetrahedra";
    EXPECT_LT(num_tets, 5000) << "Should not generate excessive tetrahedra";
    
    // Verify with OpenVolumeMesh
    verify_against_openvolulemesh(tet_mesh);
}

TEST(VolumeAdjacency, TetgenLargeMesh)
{
    // Create a larger mesh to test performance with hundreds of tets
    Geometry sphere = create_uv_sphere(16, 12, 2.0f);
    
    // Triangulate first (UV sphere has quad faces)
    auto omesh = operand_to_openmesh(&sphere);
    omesh->triangulate();
    Geometry triangulated_sphere = *openmesh_to_operand(omesh.get());
    
    geom_algorithm::TetgenParams params;
    params.max_volume = 0.15f;  // Smaller volume for more tets
    params.quality_ratio = 2.0f;
    
    Geometry tet_mesh = geom_algorithm::tetrahedralize(triangulated_sphere, params);
    auto mesh_comp = tet_mesh.get_const_component<MeshComponent>();
    
    int num_faces = mesh_comp->get_face_vertex_counts().size();
    int num_tets = num_faces / 4;
    
    std::cout << "\n=== TetGen Large Mesh ===\n";
    std::cout << "Tetrahedra: " << num_tets << "\n";
    std::cout << "Vertices: " << mesh_comp->get_vertices().size() << "\n";
    
    EXPECT_GT(num_tets, 200) << "Should generate hundreds of tetrahedra";
    EXPECT_LT(num_tets, 10000) << "Should not generate excessive tetrahedra";
    
    // Verify with OpenVolumeMesh (full validation even for large mesh)
    verify_against_openvolulemesh(tet_mesh);
    
    std::cout << "Full OVM validation successful for " << num_tets << " tetrahedra\n";
}

TEST(VolumeAdjacency, SubDevidedTet)
{
    std::cout << "\n=== Subdivided Tetrahedron Tests ===\n";
    
    // Test subdivision level 1 (8 tets)
    Geometry tet1 = create_subdivided_tetrahedron(1, 1.0f);
    auto mesh1 = tet1.get_component<MeshComponent>();
    
    std::cout << "\n--- Subdivision Level 1 ---\n";
    std::cout << "Vertices: " << mesh1->get_vertices().size() << "\n";
    std::cout << "Faces: " << mesh1->get_face_vertex_counts().size() << "\n";
    std::cout << "Face indices: " << mesh1->get_face_vertex_indices().size() << "\n";
    
    // Print first few triangles
    const auto& indices1 = mesh1->get_face_vertex_indices();
    const auto& counts1 = mesh1->get_face_vertex_counts();
    std::cout << "First few triangles:\n";
    for (size_t i = 0; i < std::min((size_t)5, counts1.size()); i++) {
        std::cout << "  Triangle " << i << ": (" << indices1[i*3] << "," << indices1[i*3+1] << "," << indices1[i*3+2] << ")\n";
    }
    
    auto [adj1, off1] = get_volume_adjacency(tet1);
    
    // Count total opposite faces
    int total_opposite_1 = 0;
    for (size_t v = 0; v < off1.size(); v++) {
        total_opposite_1 += adj1[off1[v]];
    }
    std::cout << "Total opposite faces (subdivision 1): " << total_opposite_1 << "\n";
    
    // Print first few vertices' adjacency
    for (size_t v = 0; v < std::min((size_t)5, off1.size()); v++) {
        unsigned offset = off1[v];
        unsigned count = adj1[offset];
        std::cout << "V" << v << " (count=" << count << "): ";
        for (unsigned i = 0; i < std::min(count, 3u); i++) {
            unsigned a = adj1[offset + 1 + i * 3];
            unsigned b = adj1[offset + 1 + i * 3 + 1];
            unsigned c = adj1[offset + 1 + i * 3 + 2];
            std::cout << "(" << a << "," << b << "," << c << ") ";
        }
        if (count > 3) std::cout << "...";
        std::cout << "\n";
    }
    
    EXPECT_GT(total_opposite_1, 0) << "Subdivision 1 should have opposite faces";
    
    // Test subdivision level 2 (64 tets)
    Geometry tet2 = create_subdivided_tetrahedron(2, 1.0f);
    auto mesh2 = tet2.get_component<MeshComponent>();
    
    std::cout << "\n--- Subdivision Level 2 ---\n";
    std::cout << "Vertices: " << mesh2->get_vertices().size() << "\n";
    std::cout << "Faces: " << mesh2->get_face_vertex_counts().size() << "\n";
    std::cout << "Face indices: " << mesh2->get_face_vertex_indices().size() << "\n";
    
    auto [adj2, off2] = get_volume_adjacency(tet2);
    
    // Count total opposite faces
    int total_opposite_2 = 0;
    for (size_t v = 0; v < off2.size(); v++) {
        total_opposite_2 += adj2[off2[v]];
    }
    std::cout << "Total opposite faces (subdivision 2): " << total_opposite_2 << "\n";
    
    // Print first few vertices' adjacency
    for (size_t v = 0; v < std::min((size_t)5, off2.size()); v++) {
        unsigned offset = off2[v];
        unsigned count = adj2[offset];
        std::cout << "V" << v << " (count=" << count << "): ";
        for (unsigned i = 0; i < std::min(count, 3u); i++) {
            unsigned a = adj2[offset + 1 + i * 3];
            unsigned b = adj2[offset + 1 + i * 3 + 1];
            unsigned c = adj2[offset + 1 + i * 3 + 2];
            std::cout << "(" << a << "," << b << "," << c << ") ";
        }
        if (count > 3) std::cout << "...";
        std::cout << "\n";
    }
    
    EXPECT_GT(total_opposite_2, 0) << "Subdivision 2 should have opposite faces";
    
    // Verify with OpenVolumeMesh
    std::cout << "\n--- OVM Validation for Subdivision 1 ---\n";
    verify_against_openvolulemesh(tet1);
    
    std::cout << "\n--- OVM Validation for Subdivision 2 ---\n";
    verify_against_openvolulemesh(tet2);
}

TEST(VolumeAdjacency, KnownVolumeMesh)
{
    
}

int main(int argc, char** argv)
{
    Ruzino::cuda::cuda_init();

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    Ruzino::cuda::cuda_shutdown();

    return result;
}
