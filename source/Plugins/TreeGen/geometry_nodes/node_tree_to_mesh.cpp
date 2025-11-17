#include "GCore/Components/CurveComponent.h"
#include "GCore/Components/MeshComponent.h"
#include "geom_node_base.h"
#include <glm/gtx/rotate_vector.hpp>
#include <cmath>

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(tree_to_mesh)
{
    b.add_input<Geometry>("Tree Branches");
    b.add_input<int>("Radial Segments").min(3).max(32).default_val(8);
    
    b.add_output<Geometry>("Mesh");
}

NODE_EXECUTION_FUNCTION(tree_to_mesh)
{
    auto tree_branches = params.get_input<Geometry>("Tree Branches");
    int radial_segments = params.get_input<int>("Radial Segments");
    tree_branches.apply_transform();
    auto curve = tree_branches.get_component<CurveComponent>();
    if (!curve) {
        return false;
    }
    
    // Create mesh geometry
    Geometry mesh_geom = Geometry::CreateMesh();
    auto mesh = mesh_geom.get_component<MeshComponent>();
    
    std::vector<glm::vec3> vertices;
    std::vector<int> face_vertex_counts;
    std::vector<int> face_vertex_indices;
    std::vector<glm::vec3> normals;
    
    auto curve_verts = curve->get_vertices();
    auto curve_counts = curve->get_curve_counts();
    auto curve_widths = curve->get_widths();
    
    // Process each branch (curve segment)
    int vert_offset = 0;
    int curve_offset = 0;
    
    for (int branch_idx = 0; branch_idx < curve_counts.size(); ++branch_idx) {
        int segment_count = curve_counts[branch_idx];
        if (segment_count < 2) {
            curve_offset += segment_count;
            continue;
        }
        
        // Get branch start and end
        glm::vec3 start = curve_verts[curve_offset];
        glm::vec3 end = curve_verts[curve_offset + 1];
        glm::vec3 dir = glm::normalize(end - start);
        
        float start_radius = curve_widths[curve_offset];
        float end_radius = curve_widths[curve_offset + 1];
        
        // Get perpendicular vector for creating circle
        glm::vec3 perp;
        if (std::abs(dir.y) < 0.99f) {
            perp = glm::normalize(glm::cross(dir, glm::vec3(0.0f, 1.0f, 0.0f)));
        } else {
            perp = glm::normalize(glm::cross(dir, glm::vec3(1.0f, 0.0f, 0.0f)));
        }
        
        int start_vert_idx = vertices.size();
        
        // Create vertices for start and end circles
        for (int ring = 0; ring < 2; ++ring) {
            glm::vec3 center = (ring == 0) ? start : end;
            float radius = (ring == 0) ? start_radius : end_radius;
            
            for (int i = 0; i < radial_segments; ++i) {
                float angle = (2.0f * 3.14159f * i) / radial_segments;
                glm::vec3 offset = glm::rotate(perp, angle, dir) * radius;
                
                vertices.push_back(center + offset);
                normals.push_back(glm::normalize(offset));
            }
        }
        
        // Create faces connecting the two rings
        for (int i = 0; i < radial_segments; ++i) {
            int next_i = (i + 1) % radial_segments;
            
            int v0 = start_vert_idx + i;
            int v1 = start_vert_idx + next_i;
            int v2 = start_vert_idx + radial_segments + next_i;
            int v3 = start_vert_idx + radial_segments + i;
            
            face_vertex_indices.push_back(v0);
            face_vertex_indices.push_back(v1);
            face_vertex_indices.push_back(v2);
            face_vertex_indices.push_back(v3);
            face_vertex_counts.push_back(4);
        }
        
        curve_offset += segment_count;
    }
    
    mesh->set_vertices(vertices);
    mesh->set_face_vertex_counts(face_vertex_counts);
    mesh->set_face_vertex_indices(face_vertex_indices);
    mesh->set_normals(normals);
    
    params.set_output("Mesh", mesh_geom);
    return true;
}

NODE_DECLARATION_UI(tree_to_mesh);

NODE_DEF_CLOSE_SCOPE
