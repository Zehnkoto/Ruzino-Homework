#include <time.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
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
    int num_iterations = std::max(1, params.get_input<int>("Iterations"));
    float lambda = params.get_input<float>("Hybrid Lambda");

    const char* output_names[3] = { "ARAP", "ASAP", "Hybrid" };
    const char* mesh_names[3] = { "ARAP Mesh", "ASAP Mesh", "Hybrid Mesh" };

    for (int method = 0; method < 3; ++method) {
        bool fix_two_points = (method != 0);

        Eigen::SparseMatrix<double> A(n_vertices, n_vertices);
        std::vector<Eigen::Triplet<double>> triplets;
        std::vector<double> diag_sum(n_vertices, 0.0);

        for (int f = 0; f < n_faces; ++f) {
            for (int i = 0; i < 3; ++i) {
                int v0 = face_vidx[f][i];
                int v1 = face_vidx[f][(i + 1) % 3];
                double w = cot_weights[f][(i + 2) % 3];

                bool v0_pinned = (v0 == pin1) || (fix_two_points && v0 == pin2);
                bool v1_pinned = (v1 == pin1) || (fix_two_points && v1 == pin2);

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
            if (pinned)
                triplets.push_back(Eigen::Triplet<double>(i, i, 1.0));
            else
                triplets.push_back(Eigen::Triplet<double>(i, i, diag_sum[i]));
        }

        A.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Linear solver factorization failed.");

        std::vector<Eigen::Vector2d> u = base_init_u;
        Eigen::VectorXd b_x(n_vertices), b_y(n_vertices);
        std::vector<Eigen::Matrix2d> L(n_faces);

        for (int iter = 0; iter < num_iterations; ++iter) {
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
                            s = 1.0;
                            for (int n_it = 0; n_it < 10; ++n_it) {
                                double f_val = 2.0 * lambda * s * s * s +
                                               (C1 - 2.0 * lambda) * s - C23;
                                double df_val =
                                    6.0 * lambda * s * s + C1 - 2.0 * lambda;
                                if (std::abs(df_val) < 1e-8)
                                    break;
                                double ds = f_val / df_val;
                                s -= ds;
                                if (std::abs(ds) < 1e-6)
                                    break;
                            }
                        }
                    }
                }
                L[f] = s * R;
            }

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
                        (v0 == pin1) || (fix_two_points && v0 == pin2);
                    bool v1_pinned =
                        (v1 == pin1) || (fix_two_points && v1 == pin2);

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

            b_x[pin1] = base_init_u[pin1].x();
            b_y[pin1] = base_init_u[pin1].y();
            if (fix_two_points) {
                b_x[pin2] = base_init_u[pin2].x();
                b_y[pin2] = base_init_u[pin2].y();
            }

            Eigen::VectorXd u_x = solver.solve(b_x);
            Eigen::VectorXd u_y = solver.solve(b_y);

            for (int i = 0; i < n_vertices; ++i) {
                u[i] = Eigen::Vector2d(u_x[i], u_y[i]);
            }
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
                OpenMesh::Vec3f(
                    static_cast<float>(u[i].x()),
                    static_cast<float>(u[i].y()),
                    0.0f));
        }
        auto flat_geom = openmesh_to_operand(flat_mesh.get());
        params.set_output(mesh_names[method], std::move(*flat_geom));
    }

    return true;
}

NODE_DECLARATION_UI(hw6_arap);
NODE_DEF_CLOSE_SCOPE