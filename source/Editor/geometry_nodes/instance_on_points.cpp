#include "GCore/Components/InstancerComponent.h"
#include "GCore/Components/PointsComponent.h"
#include "GCore/GOP.h"
#include "glm/ext/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(instance_on_points)
{
    // Function content omitted

    b.add_input<Geometry>("Geometry");
    b.add_input<Geometry>("Points");
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(instance_on_points)
{
    // Function content omitted
    auto points = params.get_input<Geometry>("Points");
    points.apply_transform();
    auto geometry = params.get_input<Geometry>("Geometry");
    geometry.apply_transform();

    auto instancer = std::make_shared<InstancerComponent>(&geometry);
    geometry.attach_component(instancer);

    auto points_component = points.get_component<PointsComponent>();

    if (!points_component) {
        params.set_error("No points component found in input Points");
        return false;
    }

    auto points_vertices = points_component->get_vertices();
    auto points_normals = points_component->get_normals();

    // Check if we have normals to orient instances
    bool has_normals = !points_normals.empty() &&
                       points_normals.size() == points_vertices.size();

    instancer->set_has_rotations_enabled(has_normals);

    // Batch prepare all instances
    size_t num_points = points_vertices.size();
    std::vector<glm::vec3> positions;
    std::vector<glm::quat> orientations;
    std::vector<glm::vec3> scales;  // Keep empty, no scaling needed

    positions.reserve(num_points);
    if (has_normals) {
        orientations.reserve(num_points);
    }

    for (size_t i = 0; i < num_points; ++i) {
        const auto& point = points_vertices[i];
        positions.push_back(point);

        // Only compute orientations if we have normals
        if (has_normals) {
            const auto& normal = points_normals[i];
            glm::vec3 normalized_normal = glm::normalize(normal);

            // Default up direction (Z-axis)
            glm::vec3 default_up(0.0f, 0.0f, 1.0f);

            // Calculate rotation from default up to normal
            float dot = glm::dot(default_up, normalized_normal);

            glm::quat orientation;
            if (std::abs(dot - 1.0f) < 1e-6f) {
                // Normal is already aligned with default up, no rotation needed
                orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            }
            else if (std::abs(dot + 1.0f) < 1e-6f) {
                // Normal is opposite to default up, rotate 180 degrees around X
                orientation = glm::angleAxis(
                    glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
            }
            else {
                // General case: create rotation from default up to normal
                glm::vec3 axis =
                    glm::normalize(glm::cross(default_up, normalized_normal));
                float angle = std::acos(glm::clamp(dot, -1.0f, 1.0f));
                orientation = glm::angleAxis(angle, axis);
            }
            orientations.push_back(orientation);
        }
    }

    // Batch add all instances at once
    // orientations and scales can be empty
    instancer->add_instances(positions, orientations, scales);

    params.set_output("Geometry", std::move(geometry));

    return true;
}

NODE_DECLARATION_UI(instance_on_points);
NODE_DEF_CLOSE_SCOPE