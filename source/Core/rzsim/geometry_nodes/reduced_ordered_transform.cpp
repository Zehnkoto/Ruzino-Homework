#include <glm/glm.hpp>
#include <memory>

#include "GCore/Components/MeshComponent.h"
#include "GCore/geom_payload.hpp"
#include "nodes/core/def/node_def.hpp"
#include "nodes/core/io/json.hpp"
#include "rzsim/reduced_order_basis.h"

using namespace Ruzino;

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(reduced_ordered_transform)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<std::shared_ptr<ReducedOrderedBasis>>("Reduced Basis");
    b.add_input<int>("Mode Index").default_val(0).min(0).max(99);
    b.add_input<float>("Weight").default_val(1.0f).min(-10.0f).max(10.0f);

    b.add_input<float>("Translate X").min(-10).max(10).default_val(0);
    b.add_input<float>("Translate Y").min(-10).max(10).default_val(0);
    b.add_input<float>("Translate Z").min(-10).max(10).default_val(0);

    b.add_input<float>("Rotate X").min(-180).max(180).default_val(0);
    b.add_input<float>("Rotate Y").min(-180).max(180).default_val(0);
    b.add_input<float>("Rotate Z").min(-180).max(180).default_val(0);

    b.add_input<float>("Scale X").min(0.1f).max(10).default_val(1);
    b.add_input<float>("Scale Y").min(0.1f).max(10).default_val(1);
    b.add_input<float>("Scale Z").min(0.1f).max(10).default_val(1);

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(reduced_ordered_transform)
{
    // Get inputs
    auto input_geom = params.get_input<Geometry>("Geometry");
    auto reduced_basis = params.get_input<std::shared_ptr<ReducedOrderedBasis>>("Reduced Basis");
    int mode_index = params.get_input<int>("Mode Index");
    float weight = params.get_input<float>("Weight");

    float t_x = params.get_input<float>("Translate X");
    float t_y = params.get_input<float>("Translate Y");
    float t_z = params.get_input<float>("Translate Z");

    float r_x = params.get_input<float>("Rotate X");
    float r_y = params.get_input<float>("Rotate Y");
    float r_z = params.get_input<float>("Rotate Z");

    float s_x = params.get_input<float>("Scale X");
    float s_y = params.get_input<float>("Scale Y");
    float s_z = params.get_input<float>("Scale Z");

    // Validate reduced basis
    if (!reduced_basis || reduced_basis->basis.empty()) {
        spdlog::warn("Reduced basis is empty or null");
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    // Validate mode index
    if (mode_index < 0 || mode_index >= static_cast<int>(reduced_basis->basis.size())) {
        spdlog::warn("Mode index {} out of range [0, {}), clamping",
                     mode_index, reduced_basis->basis.size());
        mode_index = std::clamp(mode_index, 0, static_cast<int>(reduced_basis->basis.size()) - 1);
    }

    // Get mesh component
    auto mesh_component = input_geom.get_component<MeshComponent>();
    if (!mesh_component) {
        spdlog::warn("Geometry has no mesh component");
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    // Get vertices
    std::vector<glm::vec3> vertices = mesh_component->get_vertices();
    if (vertices.empty()) {
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    // Get the selected eigenvector (mode)
    const auto& mode = reduced_basis->basis[mode_index];
    
    // Validate eigenvector size matches vertex count
    if (mode.size() != vertices.size()) {
        spdlog::error("Mode size {} does not match vertex count {}",
                      mode.size(), vertices.size());
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return false;
    }

    // Convert rotation from degrees to radians
    float rx_rad = glm::radians(r_x);
    float ry_rad = glm::radians(r_y);
    float rz_rad = glm::radians(r_z);

    // Build rotation matrices
    glm::mat3 rot_x = glm::mat3(
        1, 0, 0,
        0, cos(rx_rad), -sin(rx_rad),
        0, sin(rx_rad), cos(rx_rad)
    );
    
    glm::mat3 rot_y = glm::mat3(
        cos(ry_rad), 0, sin(ry_rad),
        0, 1, 0,
        -sin(ry_rad), 0, cos(ry_rad)
    );
    
    glm::mat3 rot_z = glm::mat3(
        cos(rz_rad), -sin(rz_rad), 0,
        sin(rz_rad), cos(rz_rad), 0,
        0, 0, 1
    );
    
    // Combined rotation matrix (Z * Y * X order)
    glm::mat3 rotation = rot_z * rot_y * rot_x;

    // Apply weighted transformation to each vertex based on mode value
    std::vector<glm::vec3> transformed_vertices = vertices;
    
    for (size_t i = 0; i < vertices.size(); ++i) {
        float mode_value = mode(i);
        float vertex_weight = weight * mode_value;
        
        // Apply affine transformation with per-vertex weight
        glm::vec3 v = vertices[i];
        
        // Scale (interpolate between original and scaled)
        glm::vec3 scale_vec = glm::vec3(
            1.0f + vertex_weight * (s_x - 1.0f),
            1.0f + vertex_weight * (s_y - 1.0f),
            1.0f + vertex_weight * (s_z - 1.0f)
        );
        v = v * scale_vec;
        
        // Rotate (interpolate between no rotation and full rotation)
        // For small vertex_weight, apply partial rotation
        glm::mat3 vertex_rotation = glm::mat3(1.0f) + vertex_weight * (rotation - glm::mat3(1.0f));
        v = vertex_rotation * v;
        
        // Translate
        glm::vec3 translation = glm::vec3(t_x, t_y, t_z) * vertex_weight;
        v = v + translation;
        
        transformed_vertices[i] = v;
    }

    // Create output geometry with transformed vertices
    Geometry output_geom = input_geom;
    auto output_mesh = output_geom.get_component<MeshComponent>();
    if (output_mesh) {
        output_mesh->set_vertices(transformed_vertices);
    }

    params.set_output<Geometry>("Geometry", std::move(output_geom));
    return true;
}

NODE_DECLARATION_UI(reduced_ordered_transform);

NODE_DEF_CLOSE_SCOPE
