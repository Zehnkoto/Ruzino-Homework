
#include <random>

#include "GCore/Components/PointsComponent.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(random_points)
{
    // Function content omitted
    b.add_input<float>("x_min").min(-3).max(3).default_val(-1);
    b.add_input<float>("x_max").min(-3).max(3).default_val(1);
    b.add_input<float>("y_min").min(-3).max(3).default_val(-1);
    b.add_input<float>("y_max").min(-3).max(3).default_val(1);
    b.add_input<float>("z_min").min(-3).max(3).default_val(-1);
    b.add_input<float>("z_max").min(-3).max(3).default_val(1);

    b.add_input<float>("width").min(0.01).max(1).default_val(0.1);
    b.add_input<int>("num_points").min(1).max(10000).default_val(100);

    b.add_input<int>("Seed").default_val(0).min(0).max(100);

    b.add_input<float>("theta_min").min(0).max(6.28).default_val(0);
    b.add_input<float>("theta_max").min(0).max(6.28).default_val(6.28);
    b.add_input<float>("phi_min").min(0).max(3.14).default_val(0);
    b.add_input<float>("phi_max").min(0).max(3.14).default_val(3.14);

    b.add_output<Geometry>("Points");
}

NODE_EXECUTION_FUNCTION(random_points)
{
    // Function content omitted

    Geometry points_geometry = Geometry();

    auto points_component = std::make_shared<PointsComponent>(&points_geometry);
    points_geometry.attach_component(points_component);

    std::vector<glm::vec3> vertices;

    float x_min = params.get_input<float>("x_min");
    float x_max = params.get_input<float>("x_max");
    float y_min = params.get_input<float>("y_min");
    float y_max = params.get_input<float>("y_max");
    float z_min = params.get_input<float>("z_min");
    float z_max = params.get_input<float>("z_max");

    // Ensure min <= max for all dimensions
    if (x_min > x_max)
        std::swap(x_min, x_max);
    if (y_min > y_max)
        std::swap(y_min, y_max);
    if (z_min > z_max)
        std::swap(z_min, z_max);

    std::mt19937 rng(params.get_input<int>("Seed"));

    std::uniform_real_distribution<float> dist_x(x_min, x_max);
    std::uniform_real_distribution<float> dist_y(y_min, y_max);
    std::uniform_real_distribution<float> dist_z(z_min, z_max);

    const int num_points = params.get_input<int>("num_points");

    std::vector<float> widths(num_points, params.get_input<float>("width"));
    vertices.resize(num_points);
    std::vector<glm::vec3> normals(num_points);

    float theta_min = params.get_input<float>("theta_min");
    float theta_max = params.get_input<float>("theta_max");
    float phi_min = params.get_input<float>("phi_min");
    float phi_max = params.get_input<float>("phi_max");

    // Ensure min <= max
    if (theta_min > theta_max)
        std::swap(theta_min, theta_max);
    if (phi_min > phi_max)
        std::swap(phi_min, phi_max);

    std::uniform_real_distribution<float> dist_theta(theta_min, theta_max);
    // For uniform solid angle sampling, we need to sample cos(phi) uniformly
    std::uniform_real_distribution<float> dist_cos_phi(
        std::cos(phi_max), std::cos(phi_min));

    for (int i = 0; i < num_points; i++) {
        vertices[i] = glm::vec3(dist_x(rng), dist_y(rng), dist_z(rng));

        // Generate random normal using spherical coordinates
        // Uniform solid angle sampling: sample theta uniformly, cos(phi)
        // uniformly
        float theta = dist_theta(rng);
        float cos_phi = dist_cos_phi(rng);
        float sin_phi = std::sqrt(1.0f - cos_phi * cos_phi);

        normals[i] = glm::vec3(
            sin_phi * std::cos(theta), sin_phi * std::sin(theta), cos_phi);
    }

    points_component->set_vertices(vertices);
    points_component->set_width(widths);
    points_component->set_normals(normals);

    // Set the output
    params.set_output("Points", std::move(points_geometry));
    return true;
}

NODE_DECLARATION_UI(random_points);
NODE_DEF_CLOSE_SCOPE
