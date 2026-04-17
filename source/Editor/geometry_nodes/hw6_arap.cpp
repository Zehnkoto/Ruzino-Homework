#include <time.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw6_arap)
{
    b.add_input<Geometry>("Input");
    b.add_input<Geometry>("InputUV");

    b.add_input<int>("Iterations").default_val(20).min(1).max(50);
    b.add_input<float>("Hybrid Lambda").default_val(1.0f).min(0.0f).max(50.0f);

    b.add_input<bool>("Use OpenMP").default_val(true);

    b.add_input<bool>("Use Soft Constraint").default_val(false);
    b.add_input<float>("Soft Weight").default_val(5.0f).min(0.1f).max(100.0f);
    b.add_input<float>("Pin2 Pull Factor")
        .default_val(1.0f)
        .min(1.0f)
        .max(10.0f);

    b.add_output<std::vector<glm::vec2>>("ARAP");
    b.add_output<std::vector<glm::vec2>>("ASAP");
    b.add_output<std::vector<glm::vec2>>("Hybrid");

    b.add_output<Geometry>("ARAP Mesh");
    b.add_output<Geometry>("ASAP Mesh");
    b.add_output<Geometry>("Hybrid Mesh");
}

NODE_EXECUTION_FUNCTION(hw6_arap)
{
    auto input = params.get_input<Geometry>("Input");
    auto init_uv_geom = params.get_input<Geometry>("InputUV");

    if (!input.get_component<MeshComponent>() ||
        !init_uv_geom.get_component<MeshComponent>()) {
        throw std::runtime_error("ARAP/ASAP: Need valid Input and InputUV.");
    }

    auto halfedge_mesh = operand_to_openmesh(&input);
    auto init_uv_mesh = operand_to_openmesh(&init_uv_geom);

    const int n_vertices = halfedge_mesh->n_vertices();
    const int n_faces = halfedge_mesh->n_faces();

    if (init_uv_mesh->n_vertices() != n_vertices) {
        throw std::runtime_error("Vertex count mismatch.");
    }

    std::vector<std::array<Eigen::Vector2d, 3>> local_x(n_faces);
    std::vector<std::array<double, 3>> cot_weights(n_faces);
    std::vector<std::array<int, 3>> face_vidx(n_faces);
    std::vector<double> local_K(n_faces, 0.0);

    std::vector<double> face_area_3d(n_faces, 0.0);
    double total_3d_area = 0.0;

    for (auto f_it = halfedge_mesh->faces_begin();
         f_it != halfedge_mesh->faces_end();
         ++f_it) {
        int f_idx = f_it->idx();
        std::array<OpenMesh::Vec3d, 3> v;
        int k = 0;
        for (auto fv_it = halfedge_mesh->fv_iter(*f_it); fv_it.is_valid();
             ++fv_it) {
            v[k] = halfedge_mesh->point(*fv_it);
            face_vidx[f_idx][k] = fv_it->idx();
            k++;
        }

        Eigen::Vector3d e0(
            v[2][0] - v[1][0], v[2][1] - v[1][1], v[2][2] - v[1][2]);
        Eigen::Vector3d e1(
            v[0][0] - v[2][0], v[0][1] - v[2][1], v[0][2] - v[2][2]);
        Eigen::Vector3d e2(
            v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]);

        double l0 = e0.norm(), l1 = e1.norm(), l2 = e2.norm();

        OpenMesh::Vec3d om_e1 = v[1] - v[0];
        OpenMesh::Vec3d om_e2 = v[2] - v[0];
        double area3d = 0.5 * (om_e1 % om_e2).norm();
        face_area_3d[f_idx] = area3d;
        total_3d_area += area3d;

        auto calc_cot_by_len = [](double a, double b, double c) {
            double denom = 2.0 * b * c;
            if (denom < 1e-8)
                return 0.0;
            double cos_val = (b * b + c * c - a * a) / denom;
            cos_val = std::max(-0.99999, std::min(0.99999, cos_val));
            return cos_val / std::sqrt(1.0 - cos_val * cos_val);
        };

        cot_weights[f_idx][0] = std::max(1e-4, calc_cot_by_len(l0, l1, l2));
        cot_weights[f_idx][1] = std::max(1e-4, calc_cot_by_len(l1, l2, l0));
        cot_weights[f_idx][2] = std::max(1e-4, calc_cot_by_len(l2, l0, l1));

        double cos_v0 = std::max(
            -0.99999, std::min(0.99999, -e1.dot(e2) / (l1 * l2 + 1e-8)));
        double sin_v0 = std::sqrt(1.0 - cos_v0 * cos_v0);

        local_x[f_idx][0] = Eigen::Vector2d(0.0, 0.0);
        local_x[f_idx][1] = Eigen::Vector2d(l2, 0.0);
        local_x[f_idx][2] = Eigen::Vector2d(l1 * cos_v0, l1 * sin_v0);

        Eigen::Vector2d dx0 = local_x[f_idx][1] - local_x[f_idx][2];
        Eigen::Vector2d dx1 = local_x[f_idx][2] - local_x[f_idx][0];
        Eigen::Vector2d dx2 = local_x[f_idx][0] - local_x[f_idx][1];
        local_K[f_idx] = cot_weights[f_idx][0] * dx0.squaredNorm() +
                         cot_weights[f_idx][1] * dx1.squaredNorm() +
                         cot_weights[f_idx][2] * dx2.squaredNorm();
    }

    int pin1 = -1, pin2 = -1;
    double max_dist_sq = -1.0;
    std::vector<int> bnd_vertices;
    for (auto v_it = halfedge_mesh->vertices_begin();
         v_it != halfedge_mesh->vertices_end();
         ++v_it) {
        if (halfedge_mesh->is_boundary(*v_it))
            bnd_vertices.push_back(v_it->idx());
    }

    if (bnd_vertices.size() >= 2) {
        for (size_t i = 0; i < bnd_vertices.size(); ++i) {
            for (size_t j = i + 1; j < bnd_vertices.size(); ++j) {
                auto p1 = halfedge_mesh->point(
                    halfedge_mesh->vertex_handle(bnd_vertices[i]));
                auto p2 = halfedge_mesh->point(
                    halfedge_mesh->vertex_handle(bnd_vertices[j]));
                double d = (p1 - p2).sqrnorm();
                if (d > max_dist_sq) {
                    max_dist_sq = d;
                    pin1 = bnd_vertices[i];
                    pin2 = bnd_vertices[j];
                }
            }
        }
    }
    else {
        pin1 = 0;
        pin2 = 1;
    }

    std::vector<Eigen::Vector2d> base_init_u(n_vertices);
    for (auto v_it = init_uv_mesh->vertices_begin();
         v_it != init_uv_mesh->vertices_end();
         ++v_it) {
        base_init_u[v_it->idx()] = Eigen::Vector2d(
            init_uv_mesh->point(*v_it)[0], init_uv_mesh->point(*v_it)[1]);
    }

    double signed_area = 0.0;
    for (int f = 0; f < n_faces; ++f) {
        Eigen::Vector2d u0 = base_init_u[face_vidx[f][0]];
        Eigen::Vector2d u1 = base_init_u[face_vidx[f][1]];
        Eigen::Vector2d u2 = base_init_u[face_vidx[f][2]];
        signed_area += (u1.x() - u0.x()) * (u2.y() - u0.y()) -
                       (u2.x() - u0.x()) * (u1.y() - u0.y());
    }
    if (signed_area < 0) {
        for (int i = 0; i < n_vertices; ++i)
            base_init_u[i].y() = -base_init_u[i].y();
    }

    double dist_3D = std::sqrt(max_dist_sq);
    double dist_2D = (base_init_u[pin1] - base_init_u[pin2]).norm();
    double init_scale = (dist_2D > 1e-8) ? (dist_3D / dist_2D) : 1.0;
    for (int i = 0; i < n_vertices; ++i)
        base_init_u[i] *= init_scale;

    float pull_factor = params.get_input<float>("Pin2 Pull Factor");
    if (pin2 != -1) {
        base_init_u[pin2].x() *= pull_factor;
        base_init_u[pin2].y() *= pull_factor;
    }

    int num_iterations = std::max(1, params.get_input<int>("Iterations"));
    float lambda = params.get_input<float>("Hybrid Lambda");
    bool use_omp = params.get_input<bool>("Use OpenMP");

    bool use_soft_constraint = params.get_input<bool>("Use Soft Constraint");
    double w_soft = static_cast<double>(params.get_input<float>("Soft Weight"));

    const char* output_names[3] = { "ARAP", "ASAP", "Hybrid" };
    const char* mesh_names[3] = { "ARAP Mesh", "ASAP Mesh", "Hybrid Mesh" };

    for (int method = 0; method < 3; ++method) {
        bool fix_two_points = true;

        Eigen::SparseMatrix<double> A(n_vertices, n_vertices);
        std::vector<Eigen::Triplet<double>> triplets;
        std::vector<double> diag_sum(n_vertices, 0.0);

        for (int f = 0; f < n_faces; ++f) {
            for (int i = 0; i < 3; ++i) {
                int v0 = face_vidx[f][i];
                int v1 = face_vidx[f][(i + 1) % 3];
                double w = cot_weights[f][(i + 2) % 3];

                bool v0_pinned =
                    !use_soft_constraint &&
                    ((v0 == pin1) || (fix_two_points && v0 == pin2));
                bool v1_pinned =
                    !use_soft_constraint &&
                    ((v1 == pin1) || (fix_two_points && v1 == pin2));

                if (!v0_pinned) {
                    triplets.push_back(Eigen::Triplet<double>(v0, v1, -w));
                    diag_sum[v0] += w;
                }
                if (!v1_pinned) {
                    triplets.push_back(Eigen::Triplet<double>(v1, v0, -w));
                    diag_sum[v1] += w;
                }
            }
        }

        for (int i = 0; i < n_vertices; ++i) {
            bool pinned = (i == pin1) || (fix_two_points && i == pin2);

            if (!use_soft_constraint && pinned) {
                triplets.push_back(Eigen::Triplet<double>(i, i, 1.0));
            }
            else {
                double diag_val = diag_sum[i];
                if (use_soft_constraint && pinned) {
                    diag_val += w_soft;
                }
                triplets.push_back(Eigen::Triplet<double>(i, i, diag_val));
            }
        }

        A.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Linear solver factorization failed.");

        std::vector<Eigen::Vector2d> u = base_init_u;
        Eigen::VectorXd b_x(n_vertices), b_y(n_vertices);
        std::vector<Eigen::Matrix2d> L(n_faces);

        std::vector<double> history_angle_err;
        std::vector<double> history_area_err;

        double total_local_time_ms = 0.0;
        double total_global_time_ms = 0.0;

        for (int iter = 0; iter < num_iterations; ++iter) {
            auto t_local_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for if (use_omp)
            for (int f = 0; f < n_faces; ++f) {
                Eigen::Matrix2d S = Eigen::Matrix2d::Zero();
                for (int i = 0; i < 3; ++i) {
                    int j = (i + 1) % 3;
                    int opp = (i + 2) % 3;
                    double w = cot_weights[f][opp];
                    Eigen::Vector2d du =
                        u[face_vidx[f][i]] - u[face_vidx[f][j]];
                    Eigen::Vector2d dx = local_x[f][i] - local_x[f][j];
                    S += w * (du * dx.transpose());
                }

                Eigen::JacobiSVD<Eigen::Matrix2d> svd(
                    S, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix2d R = svd.matrixU() * svd.matrixV().transpose();

                if (R.determinant() < 0) {
                    Eigen::Matrix2d U = svd.matrixU();
                    U.col(1) *= -1.0;
                    R = U * svd.matrixV().transpose();
                }

                double s = 1.0;
                if (method != 0) {
                    double C1 = local_K[f];
                    double C23 = (R.transpose() * S).trace();

                    if (method == 1) {
                        s = C23 / (C1 + 1e-8);
                    }
                    else if (method == 2) {
                        if (lambda < 1e-6) {
                            s = C23 / (C1 + 1e-8);
                        }
                        else {
                            double p = (C1 - 2.0 * lambda) / (2.0 * lambda);
                            double q = -C23 / (2.0 * lambda);
                            double delta = (q / 2.0) * (q / 2.0) +
                                           (p / 3.0) * (p / 3.0) * (p / 3.0);

                            if (delta > 0.0) {
                                double sqrt_delta = std::sqrt(delta);
                                s = std::cbrt(-q / 2.0 + sqrt_delta) +
                                    std::cbrt(-q / 2.0 - sqrt_delta);
                            }
                            else {
                                double r = std::sqrt(
                                    -(p / 3.0) * (p / 3.0) * (p / 3.0));
                                double theta = std::acos(std::max(
                                    -1.0,
                                    std::min(1.0, -q / 2.0 / (r + 1e-12))));
                                s = 2.0 * std::sqrt(-p / 3.0) *
                                    std::cos(theta / 3.0);
                            }
                            if (s < 1e-4)
                                s = 1e-4;
                        }
                    }
                }
                L[f] = s * R;
            }

            auto t_local_end = std::chrono::high_resolution_clock::now();
            total_local_time_ms += std::chrono::duration<double, std::milli>(
                                       t_local_end - t_local_start)
                                       .count();

            auto t_global_start = std::chrono::high_resolution_clock::now();

            b_x.setZero();
            b_y.setZero();

            for (int f = 0; f < n_faces; ++f) {
                for (int i = 0; i < 3; ++i) {
                    int j = (i + 1) % 3;
                    int opp = (i + 2) % 3;
                    int v0 = face_vidx[f][i];
                    int v1 = face_vidx[f][j];
                    double w = cot_weights[f][opp];

                    Eigen::Vector2d dx = local_x[f][i] - local_x[f][j];
                    Eigen::Vector2d rhs_term = w * L[f] * dx;

                    bool v0_pinned =
                        !use_soft_constraint &&
                        ((v0 == pin1) || (fix_two_points && v0 == pin2));
                    bool v1_pinned =
                        !use_soft_constraint &&
                        ((v1 == pin1) || (fix_two_points && v1 == pin2));

                    if (!v0_pinned) {
                        b_x[v0] += rhs_term.x();
                        b_y[v0] += rhs_term.y();
                    }
                    if (!v1_pinned) {
                        b_x[v1] -= rhs_term.x();
                        b_y[v1] -= rhs_term.y();
                    }
                }
            }

            if (!use_soft_constraint) {
                b_x[pin1] = base_init_u[pin1].x();
                b_y[pin1] = base_init_u[pin1].y();
                if (fix_two_points) {
                    b_x[pin2] = base_init_u[pin2].x();
                    b_y[pin2] = base_init_u[pin2].y();
                }
            }
            else {
                b_x[pin1] += w_soft * base_init_u[pin1].x();
                b_y[pin1] += w_soft * base_init_u[pin1].y();
                if (fix_two_points) {
                    b_x[pin2] += w_soft * base_init_u[pin2].x();
                    b_y[pin2] += w_soft * base_init_u[pin2].y();
                }
            }

            Eigen::VectorXd u_x = solver.solve(b_x);
            Eigen::VectorXd u_y = solver.solve(b_y);

            for (int i = 0; i < n_vertices; ++i) {
                u[i] = Eigen::Vector2d(u_x[i], u_y[i]);
            }

            auto t_global_end = std::chrono::high_resolution_clock::now();
            total_global_time_ms += std::chrono::duration<double, std::milli>(
                                        t_global_end - t_global_start)
                                        .count();

            double current_angle_err = 0.0;
            double current_area_err = 0.0;
            for (int f = 0; f < n_faces; ++f) {
                Eigen::Matrix2d V, Q;
                V.col(0) = u[face_vidx[f][1]] - u[face_vidx[f][0]];
                V.col(1) = u[face_vidx[f][2]] - u[face_vidx[f][0]];
                Q.col(0) = local_x[f][1] - local_x[f][0];
                Q.col(1) = local_x[f][2] - local_x[f][0];

                Eigen::Matrix2d J = V * Q.inverse();
                Eigen::JacobiSVD<Eigen::Matrix2d> svd_J(
                    J, Eigen::ComputeFullU | Eigen::ComputeFullV);
                double s1 = svd_J.singularValues()(0);
                double s2 = svd_J.singularValues()(1);

                if (s1 > 1e-8 && s2 > 1e-8) {
                    current_angle_err += face_area_3d[f] * (s1 / s2 + s2 / s1);
                    current_area_err +=
                        face_area_3d[f] * (s1 * s2 + 1.0 / (s1 * s2));
                }
            }
            history_angle_err.push_back(current_angle_err / total_3d_area);
            history_area_err.push_back(current_area_err / total_3d_area);
        }

        std::cout << "\n[Performance] " << output_names[method]
                  << " (Iterations: " << num_iterations << ")\n"
                  << "   -> OpenMP Mode  : "
                  << (use_omp ? "ON (Multi-Threaded)" : "OFF (Single-Threaded)")
                  << "\n"
                  << "   -> Constraint   : "
                  << (use_soft_constraint
                          ? ("SOFT (W=" + std::to_string(w_soft) + ")")
                          : "HARD")
                  << "\n"
                  << "   -> Local Phase  : " << total_local_time_ms << " ms\n"
                  << "   -> Global Phase : " << total_global_time_ms << " ms\n";

        std::string filename =
            std::string(output_names[method]) + "_Convergence_Log.csv";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << "Iteration,Angle_Distortion,Area_Distortion\n";
            for (size_t i = 0; i < history_angle_err.size(); ++i) {
                file << (i + 1) << "," << history_angle_err[i] << ","
                     << history_area_err[i] << "\n";
            }
            file.close();
        }

        std::vector<glm::vec2> uv_result(n_vertices);
        for (int i = 0; i < n_vertices; ++i) {
            uv_result[i] = glm::vec2(
                static_cast<float>(u[i].x()), static_cast<float>(u[i].y()));
        }
        params.set_output(output_names[method], uv_result);

        auto flat_mesh = operand_to_openmesh(&input);
        for (int i = 0; i < n_vertices; ++i) {
            flat_mesh->set_point(
                flat_mesh->vertex_handle(i),
                OpenMesh::Vec3f(u[i].x(), u[i].y(), 0.0f));
        }
        auto flat_geom = openmesh_to_operand(flat_mesh.get());
        params.set_output(mesh_names[method], std::move(*flat_geom));
    }

    return true;
}

NODE_DECLARATION_UI(hw6_arap);
NODE_DEF_CLOSE_SCOPE