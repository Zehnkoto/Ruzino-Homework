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

    b.add_output<std::vector<glm::vec2>>("OutputUV");
}

NODE_EXECUTION_FUNCTION(hw6_arap)
{
    auto input = params.get_input<Geometry>("Input");
    auto init_uv_geom = params.get_input<Geometry>("InputUV");

    if (!input.get_component<MeshComponent>() ||
        !init_uv_geom.get_component<MeshComponent>()) {
        throw std::runtime_error("ARAP: Need valid Input and InputUV.");
    }

    auto halfedge_mesh = operand_to_openmesh(&input);
    auto init_uv_mesh = operand_to_openmesh(&init_uv_geom);

    const int n_vertices = halfedge_mesh->n_vertices();
    const int n_faces = halfedge_mesh->n_faces();

    if (init_uv_mesh->n_vertices() != n_vertices) {
        throw std::runtime_error("ARAP: Vertex count mismatch.");
    }

    std::vector<std::array<Eigen::Vector2d, 3>> local_x(n_faces);
    std::vector<std::array<double, 3>> cot_weights(n_faces);
    std::vector<std::array<int, 3>> face_vidx(n_faces);

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
    }

    int pin_v = 0;

    Eigen::SparseMatrix<double> A(n_vertices, n_vertices);
    std::vector<Eigen::Triplet<double>> triplets;
    std::vector<double> diag_sum(n_vertices, 0.0);

    for (int f = 0; f < n_faces; ++f) {
        for (int i = 0; i < 3; ++i) {
            int v0 = face_vidx[f][i];
            int v1 = face_vidx[f][(i + 1) % 3];
            double w = cot_weights[f][(i + 2) % 3];

            if (v0 != pin_v) {
                triplets.push_back(Eigen::Triplet<double>(v0, v1, -w));
                diag_sum[v0] += w;
            }
            if (v1 != pin_v) {
                triplets.push_back(Eigen::Triplet<double>(v1, v0, -w));
                diag_sum[v1] += w;
            }
        }
    }

    for (int i = 0; i < n_vertices; ++i) {
        if (i == pin_v) {
            triplets.push_back(Eigen::Triplet<double>(i, i, 1.0));
        }
        else {
            triplets.push_back(Eigen::Triplet<double>(i, i, diag_sum[i]));
        }
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("ARAP pre-factorization failed.");
    }

    std::vector<Eigen::Vector2d> u(n_vertices);
    for (auto v_it = init_uv_mesh->vertices_begin();
         v_it != init_uv_mesh->vertices_end();
         ++v_it) {
        int i = v_it->idx();
        auto pt = init_uv_mesh->point(*v_it);
        u[i] = Eigen::Vector2d(pt[0], pt[1]);
    }

    double signed_area = 0.0;
    for (int f = 0; f < n_faces; ++f) {
        Eigen::Vector2d u0 = u[face_vidx[f][0]];
        Eigen::Vector2d u1 = u[face_vidx[f][1]];
        Eigen::Vector2d u2 = u[face_vidx[f][2]];
        signed_area += (u1.x() - u0.x()) * (u2.y() - u0.y()) -
                       (u2.x() - u0.x()) * (u1.y() - u0.y());
    }
    if (signed_area < 0) {
        std::cout
            << "[ARAP] Detected inside-out initialization! Flipping Y axis..."
            << std::endl;
        for (int i = 0; i < n_vertices; ++i) {
            u[i].y() = -u[i].y();
        }
    }

    Eigen::VectorXd b_x(n_vertices);
    Eigen::VectorXd b_y(n_vertices);
    std::vector<Eigen::Matrix2d> L(n_faces);

    int num_iterations = params.get_input<int>("Iterations");
    num_iterations = std::max(1, num_iterations);

    for (int iter = 0; iter < num_iterations; ++iter) {
        for (int f = 0; f < n_faces; ++f) {
            Eigen::Matrix2d S = Eigen::Matrix2d::Zero();
            for (int i = 0; i < 3; ++i) {
                int j = (i + 1) % 3;
                int opp = (i + 2) % 3;
                int v0 = face_vidx[f][i];
                int v1 = face_vidx[f][j];
                double w = cot_weights[f][opp];

                Eigen::Vector2d du = u[v0] - u[v1];
                Eigen::Vector2d dx = local_x[f][i] - local_x[f][j];
                S += w * (du * dx.transpose());
            }

            Eigen::JacobiSVD<Eigen::Matrix2d> svd(
                S, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2d U = svd.matrixU();
            Eigen::Matrix2d V = svd.matrixV();
            Eigen::Matrix2d R = U * V.transpose();

            if (R.determinant() < 0) {
                U.col(1) *= -1.0;
                R = U * V.transpose();
            }
            L[f] = R;
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

                if (v0 != pin_v) {
                    b_x[v0] += rhs_term.x();
                    b_y[v0] += rhs_term.y();
                }
                if (v1 != pin_v) {
                    b_x[v1] -= rhs_term.x();
                    b_y[v1] -= rhs_term.y();
                }
            }
        }

        b_x[pin_v] = 0.0;
        b_y[pin_v] = 0.0;

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

    params.set_output("OutputUV", uv_result);
    return true;
}

NODE_DECLARATION_UI(hw6_arap);
NODE_DEF_CLOSE_SCOPE