#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <iostream>
#include <vector>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

OpenMesh::Vec3uc ErrorToColor(double error, double max_error)
{
    if (error < 0)
        return OpenMesh::Vec3uc(0, 0, 0); 

    double t = (error - 1.0) / (max_error - 1.0);
    t = std::max(0.0, std::min(1.0, t));  

    float r = std::max(0.0, std::min(1.0, 1.5 - std::abs(4.0 * t - 3.0)));
    float g = std::max(0.0, std::min(1.0, 1.5 - std::abs(4.0 * t - 2.0)));
    float b = std::max(0.0, std::min(1.0, 1.5 - std::abs(4.0 * t - 1.0)));

    return OpenMesh::Vec3uc(
        static_cast<unsigned char>(r * 255.0f),
        static_cast<unsigned char>(g * 255.0f),
        static_cast<unsigned char>(b * 255.0f));
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw5_distortion_eval)
{
    b.add_input<Geometry>("Original 3D Mesh");
    b.add_input<Geometry>("Parametrized 2D Mesh");
    b.add_input<float>("Max Error Threshold").default_val(3.0f);

    b.add_output<Geometry>("Colored Mesh");
}

NODE_EXECUTION_FUNCTION(hw5_distortion_eval)
{
    auto orig_input = params.get_input<Geometry>("Original 3D Mesh");
    auto param_input = params.get_input<Geometry>("Parametrized 2D Mesh");
    float max_error = params.get_input<float>("Max Error Threshold");

    if (!orig_input.get_component<MeshComponent>() ||
        !param_input.get_component<MeshComponent>()) {
        throw std::runtime_error(
            "Distortion Eval: Need both Original and Parametrized Geometry "
            "Inputs.");
        return false;
    }

    auto orig_mesh = operand_to_openmesh(&orig_input);
    auto color_mesh = operand_to_openmesh(&param_input);

    color_mesh->request_vertex_colors();

    int n_vertices = color_mesh->n_vertices();
    std::vector<double> vert_error_sum(n_vertices, 0.0);
    std::vector<int> vert_face_count(n_vertices, 0);
    int flipped_triangles = 0;

    for (auto f_handle : color_mesh->faces()) {
        std::vector<OpenMesh::VertexHandle> f_v;
        for (auto fv_it = color_mesh->cfv_iter(f_handle); fv_it.is_valid();
             ++fv_it) {
            f_v.push_back(*fv_it);
        }
        if (f_v.size() != 3)
            continue;  

        OpenMesh::Vec3f p1 = orig_mesh->point(f_v[0]);
        OpenMesh::Vec3f p2 = orig_mesh->point(f_v[1]);
        OpenMesh::Vec3f p3 = orig_mesh->point(f_v[2]);

        OpenMesh::Vec3f u1 = color_mesh->point(f_v[0]);
        OpenMesh::Vec3f u2 = color_mesh->point(f_v[1]);
        OpenMesh::Vec3f u3 = color_mesh->point(f_v[2]);

        OpenMesh::Vec3f e1 = p2 - p1;
        OpenMesh::Vec3f e2 = p3 - p1;
        double len_e1 = e1.norm();
        if (len_e1 < 1e-8)
            continue; 

        OpenMesh::Vec3f X = e1 / len_e1;
        OpenMesh::Vec3f N = (e1 % e2).normalized();  
        OpenMesh::Vec3f Y = (N % X).normalized();

        Eigen::Matrix2d Q;
        Q(0, 0) = len_e1;
        Q(0, 1) = (e2 | X);  // e2 dot X
        Q(1, 0) = 0.0;
        Q(1, 1) = (e2 | Y);  // e2 dot Y

        Eigen::Matrix2d V;
        V(0, 0) = u2[0] - u1[0];
        V(0, 1) = u3[0] - u1[0];
        V(1, 0) = u2[1] - u1[1];
        V(1, 1) = u3[1] - u1[1];

        if (std::abs(Q.determinant()) < 1e-8)
            continue;
        Eigen::Matrix2d J = V * Q.inverse();

        Eigen::JacobiSVD<Eigen::Matrix2d> svd(
            J, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector2d singular_values = svd.singularValues();
        double s1 = singular_values(0);
        double s2 = singular_values(1);

        double conformal_error = 1.0;
        if (s2 > 1e-8) {
            conformal_error = s1 / s2; 
        }

        if (J.determinant() <= 0) {
            flipped_triangles++;
        }

        for (int k = 0; k < 3; ++k) {
            int v_idx = f_v[k].idx();
            if (conformal_error < 0) {
                vert_error_sum[v_idx] =
                    -1000.0;  
            }
            else if (vert_error_sum[v_idx] >= 0) {
                vert_error_sum[v_idx] += conformal_error;
            }
            vert_face_count[v_idx]++;
        }
    }

    for (auto v_handle : color_mesh->vertices()) {
        int idx = v_handle.idx();
        double final_error = 1.0;
        if (vert_error_sum[idx] < 0) {
            final_error = -1.0;  
        }
        else if (vert_face_count[idx] > 0) {
            final_error =
                vert_error_sum[idx] / vert_face_count[idx];  
        }

        auto color = ErrorToColor(final_error, max_error);
        color_mesh->set_color(v_handle, color);
    }

    std::cout << "[Distortion Eval] Flipped Triangles: " << flipped_triangles
              << std::endl;

    auto geometry = openmesh_to_operand(color_mesh.get());
    params.set_output("Colored Mesh", std::move(*geometry));

    return true;
}

NODE_DECLARATION_UI(hw5_distortion_eval);
NODE_DEF_CLOSE_SCOPE