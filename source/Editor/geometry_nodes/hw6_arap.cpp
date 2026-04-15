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
    // Input-1: Original 3D mesh with boundary (used for 3D rest shape)
    b.add_input<Geometry>("Input");

    // Input-2: Initial flattened mesh from HW5
    // Changed from vector to Geometry to accept hw5_param output directly
    b.add_input<Geometry>("InputUV");

    // Output-1: The UV coordinate of the mesh, provided by ARAP algorithm
    b.add_output<std::vector<glm::vec2>>("OutputUV");
}

NODE_EXECUTION_FUNCTION(hw6_arap)
{
    // 1. Get input data
    auto input = params.get_input<Geometry>("Input");
    auto init_uv_geom = params.get_input<Geometry>("InputUV");

    if (!input.get_component<MeshComponent>()) {
        throw std::runtime_error("ARAP: Need Original Geometry Input.");
    }
    if (!init_uv_geom.get_component<MeshComponent>()) {
        throw std::runtime_error(
            "ARAP: Need Initial Flattened Geometry (InputUV).");
    }

    auto halfedge_mesh = operand_to_openmesh(&input);
    auto init_uv_mesh = operand_to_openmesh(&init_uv_geom);

    const int n_vertices = halfedge_mesh->n_vertices();
    const int n_faces = halfedge_mesh->n_faces();

    if (init_uv_mesh->n_vertices() != n_vertices) {
        throw std::runtime_error(
            "ARAP: Initial UV mesh vertex count mismatch.");
    }

    // 2. Preprocess 1: Calculate local 2D flattenings and cotangent weights
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

        auto calc_cot = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
            double dot = a.dot(b);
            double cross = a.cross(b).norm();
            return (cross > 1e-8) ? dot / cross : 0.0;
        };
        cot_weights[f_idx][0] = calc_cot(-e1, e2);
        cot_weights[f_idx][1] = calc_cot(-e2, e0);
        cot_weights[f_idx][2] = calc_cot(-e0, e1);

        double l01 = e2.norm();
        double l02 = e1.norm();
        double cos_v0 = -e1.dot(e2) / (l01 * l02);
        double sin_v0 = std::sqrt(std::max(0.0, 1.0 - cos_v0 * cos_v0));

        local_x[f_idx][0] = Eigen::Vector2d(0.0, 0.0);
        local_x[f_idx][1] = Eigen::Vector2d(l01, 0.0);
        local_x[f_idx][2] = Eigen::Vector2d(l02 * cos_v0, l02 * sin_v0);
    }

    // 3. Preprocess 2: Find two furthest boundary points to pin (remove
    // nullspace)
    int pin1 = -1, pin2 = -1;
    double max_dist_sq = -1.0;

    std::vector<int> bnd_vertices;
    for (auto v_it = halfedge_mesh->vertices_begin();
         v_it != halfedge_mesh->vertices_end();
         ++v_it) {
        if (halfedge_mesh->is_boundary(*v_it)) {
            bnd_vertices.push_back(v_it->idx());
        }
    }

    if (bnd_vertices.size() >= 2) {
        for (size_t i = 0; i < bnd_vertices.size(); ++i) {
            for (size_t j = i + 1; j < bnd_vertices.size(); ++j) {
                int v1 = bnd_vertices[i];
                int v2 = bnd_vertices[j];
                auto p1 =
                    halfedge_mesh->point(halfedge_mesh->vertex_handle(v1));
                auto p2 =
                    halfedge_mesh->point(halfedge_mesh->vertex_handle(v2));
                double dist_sq = (p1 - p2).sqrnorm();
                if (dist_sq > max_dist_sq) {
                    max_dist_sq = dist_sq;
                    pin1 = v1;
                    pin2 = v2;
                }
            }
        }
    }
    else {
        pin1 = 0;
        pin2 = 1;
    }

    // 4. Preprocess 3: Build global sparse matrix and pre-factorize
    Eigen::SparseMatrix<double> A(n_vertices, n_vertices);
    std::vector<Eigen::Triplet<double>> triplets;
    std::vector<double> diag_sum(n_vertices, 0.0);

    for (int f = 0; f < n_faces; ++f) {
        for (int i = 0; i < 3; ++i) {
            int v0 = face_vidx[f][i];
            int v1 = face_vidx[f][(i + 1) % 3];
            double w = cot_weights[f][(i + 2) % 3];

            if (v0 != pin1 && v0 != pin2) {
                triplets.push_back(Eigen::Triplet<double>(v0, v1, -w));
                diag_sum[v0] += w;
            }
            if (v1 != pin1 && v1 != pin2) {
                triplets.push_back(Eigen::Triplet<double>(v1, v0, -w));
                diag_sum[v1] += w;
            }
        }
    }

    for (int i = 0; i < n_vertices; ++i) {
        if (i == pin1 || i == pin2) {
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

    // 5. Initialize UVs from HW5 geometry
    std::vector<Eigen::Vector2d> u(n_vertices);
    for (auto v_it = init_uv_mesh->vertices_begin();
         v_it != init_uv_mesh->vertices_end();
         ++v_it) {
        int i = v_it->idx();
        auto pt = init_uv_mesh->point(*v_it);
        u[i] = Eigen::Vector2d(pt[0], pt[1]);
    }

    Eigen::VectorXd b_x(n_vertices);
    Eigen::VectorXd b_y(n_vertices);
    std::vector<Eigen::Matrix2d> L(n_faces);

    // 6. ARAP Local-Global Iterations
    const int num_iterations = 20;
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Local Phase: Optimize L_t
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

        // Global Phase: Optimize u
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

                if (v0 != pin1 && v0 != pin2) {
                    b_x[v0] += rhs_term.x();
                    b_y[v0] += rhs_term.y();
                }
                if (v1 != pin1 && v1 != pin2) {
                    b_x[v1] -= rhs_term.x();
                    b_y[v1] -= rhs_term.y();
                }
            }
        }

        b_x[pin1] = u[pin1].x();
        b_y[pin1] = u[pin1].y();
        b_x[pin2] = u[pin2].x();
        b_y[pin2] = u[pin2].y();

        Eigen::VectorXd u_x = solver.solve(b_x);
        Eigen::VectorXd u_y = solver.solve(b_y);

        for (int i = 0; i < n_vertices; ++i) {
            u[i] = Eigen::Vector2d(u_x[i], u_y[i]);
        }
    }

    // 7. Format output
    std::vector<glm::vec2> uv_result(n_vertices);
    for (int i = 0; i < n_vertices; ++i) {
        uv_result[i] = glm::vec2(
            static_cast<float>(u[i].x()), static_cast<float>(u[i].y()));
    }

    params.set_output("OutputUV", uv_result);
}

NODE_DECLARATION_UI(hw6_arap);
NODE_DEF_CLOSE_SCOPE