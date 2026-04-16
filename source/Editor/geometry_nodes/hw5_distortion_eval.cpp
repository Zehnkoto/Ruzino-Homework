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

NODE_DECLARATION_FUNCTION(hw6_distortion_eval)
{
    b.add_input<Geometry>("Original 3D Mesh");
    b.add_input<std::vector<glm::vec2>>("Parametrized UVs");
    b.add_input<float>("Max Error Threshold").default_val(3.0f);

    b.add_input<bool>("Output as 2D Flat Mesh").default_val(true);

    b.add_output<Geometry>("Angle Distortion");
    b.add_output<Geometry>("Area Distortion");
}

NODE_EXECUTION_FUNCTION(hw6_distortion_eval)
{
    auto orig_input = params.get_input<Geometry>("Original 3D Mesh");
    auto uvs = params.get_input<std::vector<glm::vec2>>("Parametrized UVs");
    float max_error = params.get_input<float>("Max Error Threshold");
    bool flat_output = params.get_input<bool>("Output as 2D Flat Mesh");

    if (!orig_input.get_component<MeshComponent>() || uvs.empty()) {
        throw std::runtime_error(
            "Distortion Eval: Need Original Mesh and UV Inputs.");
        return false;
    }

    auto orig_mesh = operand_to_openmesh(&orig_input);

    auto angle_mesh = operand_to_openmesh(&orig_input);
    angle_mesh->request_vertex_colors();

    auto area_mesh = operand_to_openmesh(&orig_input);
    area_mesh->request_vertex_colors();

    int n_vertices = angle_mesh->n_vertices();
    if (uvs.size() != n_vertices) {
        throw std::runtime_error("UV size does not match vertex count.");
    }

    std::vector<double> vert_angle_sum(n_vertices, 0.0);
    std::vector<double> vert_area_sum(n_vertices, 0.0);
    std::vector<int> vert_face_count(n_vertices, 0);
    int flipped_triangles = 0;

    double total_3d_area = 0.0;
    double global_angle_err = 0.0;
    double global_area_err = 0.0;

    for (auto f_handle : angle_mesh->faces()) {
        std::vector<OpenMesh::VertexHandle> f_v;
        for (auto fv_it = angle_mesh->cfv_iter(f_handle); fv_it.is_valid();
             ++fv_it) {
            f_v.push_back(*fv_it);
        }
        if (f_v.size() != 3)
            continue;

        OpenMesh::Vec3f p1 = orig_mesh->point(f_v[0]);
        OpenMesh::Vec3f p2 = orig_mesh->point(f_v[1]);
        OpenMesh::Vec3f p3 = orig_mesh->point(f_v[2]);

        OpenMesh::Vec3f u1(uvs[f_v[0].idx()].x, uvs[f_v[0].idx()].y, 0.0f);
        OpenMesh::Vec3f u2(uvs[f_v[1].idx()].x, uvs[f_v[1].idx()].y, 0.0f);
        OpenMesh::Vec3f u3(uvs[f_v[2].idx()].x, uvs[f_v[2].idx()].y, 0.0f);

        OpenMesh::Vec3f e1 = p2 - p1;
        OpenMesh::Vec3f e2 = p3 - p1;
        double len_e1 = e1.norm();
        if (len_e1 < 1e-8)
            continue;

        double face_area_3d = 0.5 * (e1 % e2).norm();

        OpenMesh::Vec3f X = e1 / len_e1;
        OpenMesh::Vec3f N = (e1 % e2).normalized();
        OpenMesh::Vec3f Y = (N % X).normalized();

        Eigen::Matrix2d Q;
        Q(0, 0) = len_e1;
        Q(0, 1) = (e2 | X);
        Q(1, 0) = 0.0;
        Q(1, 1) = (e2 | Y);

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

        if (J.determinant() <= 0) {
            flipped_triangles++;
        }

        double err_angle = 1.0;
        double err_area = 1.0;

        if (s2 > 1e-8 && s1 > 1e-8) {
            err_angle = 0.5 * (s1 / s2 + s2 / s1);
            double area = s1 * s2;
            err_area = 0.5 * (area + 1.0 / area);

            double paper_angle_err = (s1 / s2 + s2 / s1);
            double paper_area_err = (area + 1.0 / area);
            total_3d_area += face_area_3d;
            global_angle_err += face_area_3d * paper_angle_err;
            global_area_err += face_area_3d * paper_area_err;
        }
        else {
            err_angle = -1000.0;
            err_area = -1000.0;
        }

        for (int k = 0; k < 3; ++k) {
            int v_idx = f_v[k].idx();
            if (J.determinant() <= 0) {
                vert_angle_sum[v_idx] = -1000.0;
                vert_area_sum[v_idx] = -1000.0;
            }
            else {
                if (vert_angle_sum[v_idx] >= 0)
                    vert_angle_sum[v_idx] += err_angle;
                if (vert_area_sum[v_idx] >= 0)
                    vert_area_sum[v_idx] += err_area;
            }
            vert_face_count[v_idx]++;
        }
    }

    if (total_3d_area > 1e-8) {
        global_angle_err /= total_3d_area;
        global_area_err /= total_3d_area;
        std::cout << "\n==================================================\n";
        std::cout << "[Distortion Metrics - Follows ARAP Paper Eq.]\n";
        std::cout << " -> Angle Distortion (D_angle) : " << global_angle_err << "  (Perfect = 2.0)\n";
        std::cout << " -> Area Distortion  (D_area)  : " << global_area_err << "  (Perfect = 2.0)\n";
        std::cout << " -> Flipped Triangles          : " << flipped_triangles << "\n";
        std::cout << "==================================================\n\n";
    }

    for (auto v_handle : angle_mesh->vertices()) {
        int idx = v_handle.idx();
        double f_angle = 1.0;
        double f_area = 1.0;

        if (vert_angle_sum[idx] < 0) {
            f_angle = -1.0;
            f_area = -1.0;
        }
        else if (vert_face_count[idx] > 0) {
            f_angle = vert_angle_sum[idx] / vert_face_count[idx];
            f_area = vert_area_sum[idx] / vert_face_count[idx];
        }

        angle_mesh->set_color(v_handle, ErrorToColor(f_angle, max_error));
        area_mesh->set_color(v_handle, ErrorToColor(f_area, max_error));

        if (flat_output) {
            OpenMesh::Vec3f flat_pt(uvs[idx].x, uvs[idx].y, 0.0f);
            angle_mesh->set_point(v_handle, flat_pt);
            area_mesh->set_point(v_handle, flat_pt);
        }
    }

    auto geom_angle = openmesh_to_operand(angle_mesh.get());
    auto geom_area = openmesh_to_operand(area_mesh.get());

    params.set_output("Angle Distortion", std::move(*geom_angle));
    params.set_output("Area Distortion", std::move(*geom_area));

    return true;
}

NODE_DECLARATION_UI(hw6_distortion_eval);
NODE_DEF_CLOSE_SCOPE