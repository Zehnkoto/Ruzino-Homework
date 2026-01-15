#include <glm/glm.hpp>
#include <memory>

#include "GCore/Components/MeshComponent.h"
#include "nodes/core/def/node_def.hpp"
#include "rzsim/reduced_order_basis.h"
#include "spdlog/spdlog.h"

using namespace Ruzino;

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(reduced_ordered_transform)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<std::shared_ptr<ReducedOrderedBasis>>("Reduced Basis");
    b.add_input<int>("Mode Index").default_val(0).min(0).max(99);
    b.add_input<bool>("Apply to All Modes").default_val(true);
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
    b.add_output<std::shared_ptr<Ruzino::AffineTransform>>("Transform");
}

NODE_EXECUTION_FUNCTION(reduced_ordered_transform)
{
    // Get inputs
    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();
    auto reduced_basis =
        params.get_input<std::shared_ptr<ReducedOrderedBasis>>("Reduced Basis");
    int mode_index = params.get_input<int>("Mode Index");
    bool apply_to_all_modes = params.get_input<bool>("Apply to All Modes");
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
        params.set_output(
            "Transform", std::shared_ptr<Ruzino::AffineTransform>(nullptr));
        return true;
    }

    // Validate mode index (only if not applying to all modes)
    if (!apply_to_all_modes) {
        if (mode_index < 0 ||
            mode_index >= static_cast<int>(reduced_basis->basis.size())) {
            spdlog::warn(
                "Mode index {} out of range [0, {}), clamping",
                mode_index,
                reduced_basis->basis.size());
            mode_index = std::clamp(
                mode_index,
                0,
                static_cast<int>(reduced_basis->basis.size()) - 1);
        }
    }

    // Get mesh component
    auto mesh_component = input_geom.get_component<MeshComponent>();
    if (!mesh_component) {
        spdlog::warn("Geometry has no mesh component");
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        params.set_output(
            "Transform", std::shared_ptr<Ruzino::AffineTransform>(nullptr));
        return true;
    }

    // Get vertices
    std::vector<glm::vec3> vertices = mesh_component->get_vertices();
    if (vertices.empty()) {
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        params.set_output(
            "Transform", std::shared_ptr<Ruzino::AffineTransform>(nullptr));
        return true;
    }

    // Determine number of modes to process
    int num_modes =
        apply_to_all_modes ? static_cast<int>(reduced_basis->basis.size()) : 1;
    int start_mode = apply_to_all_modes ? 0 : mode_index;
    int end_mode = apply_to_all_modes ? num_modes : (mode_index + 1);

    // Validate eigenvector size matches vertex count for the first mode
    const auto& first_mode = reduced_basis->basis[start_mode];
    if (first_mode.size() != vertices.size()) {
        spdlog::error(
            "Mode size {} does not match vertex count {}",
            first_mode.size(),
            vertices.size());
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        params.set_output(
            "Transform", std::shared_ptr<Ruzino::AffineTransform>(nullptr));
        return false;
    }

    // Convert rotation from degrees to radians
    float rx_rad = glm::radians(r_x);
    float ry_rad = glm::radians(r_y);
    float rz_rad = glm::radians(r_z);

    // Build rotation matrices
    glm::mat3 rot_x = glm::mat3(
        1, 0, 0, 0, cos(rx_rad), -sin(rx_rad), 0, sin(rx_rad), cos(rx_rad));

    glm::mat3 rot_y = glm::mat3(
        cos(ry_rad), 0, sin(ry_rad), 0, 1, 0, -sin(ry_rad), 0, cos(ry_rad));

    glm::mat3 rot_z = glm::mat3(
        cos(rz_rad), -sin(rz_rad), 0, sin(rz_rad), cos(rz_rad), 0, 0, 0, 1);

    // Combined rotation matrix (Z * Y * X order)
    glm::mat3 rotation = rot_z * rot_y * rot_x;

    // Create AffineTransform output
    auto transform_output = std::make_shared<Ruzino::AffineTransform>(
        apply_to_all_modes ? num_modes
                           : static_cast<int>(reduced_basis->basis.size()));

    // For each mode in the range, compute and store the transformation
    for (int m = start_mode; m < end_mode; ++m) {
        // Compute transform for this mode
        std::vector<float> transform(12);

        // Store rotation matrix (row-major)
        transform[0] = rotation[0][0];
        transform[1] = rotation[0][1];
        transform[2] = rotation[0][2];
        transform[3] = rotation[1][0];
        transform[4] = rotation[1][1];
        transform[5] = rotation[1][2];
        transform[6] = rotation[2][0];
        transform[7] = rotation[2][1];
        transform[8] = rotation[2][2];

        // Store translation
        transform[9] = t_x;
        transform[10] = t_y;
        transform[11] = t_z;

        transform_output->set_transform(m, transform);
    }

    spdlog::info(
        "Generated affine transform for {} mode(s) (apply_to_all={})",
        end_mode - start_mode,
        apply_to_all_modes);

    // Apply weighted transformation to each vertex based on mode value
    std::vector<glm::vec3> transformed_vertices = vertices;

    // Process each mode in the range
    for (int m = start_mode; m < end_mode; ++m) {
        const auto& mode = reduced_basis->basis[m];

        for (size_t i = 0; i < vertices.size(); ++i) {
            float mode_value = mode(i);
            float vertex_weight = weight * mode_value;

            // Apply affine transformation with per-vertex weight
            glm::vec3 v = transformed_vertices[i];

            // Scale (interpolate between original and scaled)
            glm::vec3 scale_vec = glm::vec3(
                1.0f + vertex_weight * (s_x - 1.0f),
                1.0f + vertex_weight * (s_y - 1.0f),
                1.0f + vertex_weight * (s_z - 1.0f));
            v = v * scale_vec;

            // Rotate (interpolate between no rotation and full rotation)
            // For small vertex_weight, apply partial rotation
            glm::mat3 vertex_rotation =
                glm::mat3(1.0f) + vertex_weight * (rotation - glm::mat3(1.0f));
            v = vertex_rotation * v;

            // Translate
            glm::vec3 translation = glm::vec3(t_x, t_y, t_z) * vertex_weight;
            v = v + translation;

            transformed_vertices[i] = v;
        }
    }

    // Create output geometry with transformed vertices
    Geometry output_geom = input_geom;
    auto output_mesh = output_geom.get_component<MeshComponent>();
    if (output_mesh) {
        output_mesh->set_vertices(transformed_vertices);
    }

    params.set_output<Geometry>("Geometry", std::move(output_geom));
    params.set_output("Transform", std::move(transform_output));
    return true;
}

NODE_DECLARATION_UI(reduced_ordered_transform);

NODE_DEF_CLOSE_SCOPE
