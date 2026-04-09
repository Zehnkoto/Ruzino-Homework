#include <pxr/usd/usdGeom/mesh.h>
#include <time.h>

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
** @brief HW4_TutteParameterization
**
** This file presents the basic framework of a "node", which processes inputs
** received from the left and outputs specific variables for downstream nodes to
** use.
** - In the first function, node_declare, you can set up the node's input and
** output variables.
** - The second function, node_exec is the execution part of the node, where we
** need to implement the node's functionality.
** Your task is to fill in the required logic at the specified locations
** within this template, especially in node_exec.
*/

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(hw5_param)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");

    // Input-2: 原始三维网格，用于计算 Cotangent 和 Floater 权重的几何信息
    b.add_input<Geometry>("Original Mesh");

    b.add_output<Geometry>("Uniform");
    b.add_output<Geometry>("Cotangent");
    b.add_output<Geometry>("Floater");
}

NODE_EXECUTION_FUNCTION(hw5_param)
{
    // Get the input from params
    auto input = params.get_input<Geometry>("Input");

    if (!input.get_component<MeshComponent>()) {
        throw std::runtime_error("Minimal Surface: Need Geometry Input.");
        return false;
    }

    std::shared_ptr<Ruzino::PolyMesh> ref_mesh = nullptr;
    if (params.has_input("Original Mesh")) {
        auto ref_input = params.get_input<Geometry>("Original Mesh");
        if (ref_input.get_component<MeshComponent>()) {
            ref_mesh = operand_to_openmesh(&ref_input);
        }
    }

    // 辅助函数：计算余切值 cot = cos/sin = dot(u,v) / norm(cross(u,v))
    auto get_cotan = [](const OpenMesh::Vec3f& center,
                        const OpenMesh::Vec3f& p1,
                        const OpenMesh::Vec3f& p2) {
        OpenMesh::Vec3f u = p1 - center;
        OpenMesh::Vec3f v = p2 - center;
        double dot_prod = (u | v);          // OpenMesh dot product
        double cross_len = (u % v).norm();  // OpenMesh cross product length
        if (cross_len < 1e-6)
            return 0.0;
        return dot_prod / cross_len;
    };

    const char* output_names[3] = { "Uniform", "Cotangent", "Floater" };

    // 循环 3 次，一次性算出 3 种权重，送到 3 个输出端口
    for (int weight_type = 0; weight_type < 3; ++weight_type) {
        int actual_type = weight_type;
        // 如果选择了高级权重但没有连入 Original Mesh，自动退化为Uniform
        if ((actual_type == 1 || actual_type == 2) && ref_mesh == nullptr) {
            actual_type = 0;
        }

        // 每次循环都需要从 Input 获取一个干净的全新网格拷贝
        auto halfedge_mesh = operand_to_openmesh(&input);

        int n_vertices = halfedge_mesh->n_vertices();
        Eigen::SparseMatrix<double> A(n_vertices, n_vertices);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_vertices, 3);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(n_vertices * 7);

        for (const auto& v_handle : halfedge_mesh->vertices()) {
            int i = v_handle.idx();

            // 边界点始终固定为当前坐标 (Dirichlet 边界条件)
            if (halfedge_mesh->is_boundary(v_handle)) {
                triplets.push_back(Eigen::Triplet<double>(i, i, 1.0));
                auto pt = halfedge_mesh->point(v_handle);
                B(i, 0) = pt[0];
                B(i, 1) = pt[1];
                B(i, 2) = pt[2];
            }
            // 内部点
            else {
                if (actual_type == 0) {
                    // 1. Uniform Weights
                    double degree = 0.0;
                    for (const auto& he_handle :
                         v_handle.outgoing_halfedges()) {
                        int j = he_handle.to().idx();
                        triplets.push_back(Eigen::Triplet<double>(i, j, -1.0));
                        degree += 1.0;
                    }
                    triplets.push_back(Eigen::Triplet<double>(i, i, degree));
                }
                else if (actual_type == 1) {
                    // 2. Cotangent Weights
                    double weight_sum = 0.0;
                    for (const auto& he_handle :
                         v_handle.outgoing_halfedges()) {
                        int j = he_handle.to().idx();
                        double w = 0.0;

                        // 去 Original
                        // Mesh中寻找对应的半边以获取真实的三维几何信息
                        auto ref_he =
                            ref_mesh->halfedge_handle(he_handle.idx());

                        if (!ref_mesh->is_boundary(ref_he)) {
                            auto he_next =
                                ref_mesh->next_halfedge_handle(ref_he);
                            auto p_alpha = ref_mesh->point(
                                ref_mesh->to_vertex_handle(he_next));
                            auto p_i = ref_mesh->point(
                                ref_mesh->from_vertex_handle(ref_he));
                            auto p_j = ref_mesh->point(
                                ref_mesh->to_vertex_handle(ref_he));
                            w += get_cotan(p_alpha, p_i, p_j);
                        }

                        auto ref_he_opp =
                            ref_mesh->opposite_halfedge_handle(ref_he);
                        if (!ref_mesh->is_boundary(ref_he_opp)) {
                            auto he_opp_next =
                                ref_mesh->next_halfedge_handle(ref_he_opp);
                            auto p_beta = ref_mesh->point(
                                ref_mesh->to_vertex_handle(he_opp_next));
                            auto p_i = ref_mesh->point(
                                ref_mesh->to_vertex_handle(ref_he_opp));
                            auto p_j = ref_mesh->point(
                                ref_mesh->from_vertex_handle(ref_he_opp));
                            w += get_cotan(p_beta, p_i, p_j);
                        }

                        if (w < 1e-4)
                            w = 1e-4;  // 防止负权重导致系统不稳定

                        triplets.push_back(Eigen::Triplet<double>(i, j, -w));
                        weight_sum += w;
                    }
                    triplets.push_back(
                        Eigen::Triplet<double>(i, i, weight_sum));
                }
                else if (actual_type == 2) {
                    // 3. Floater's Shape-Preserving Weights
                    std::vector<OpenMesh::SmartHalfedgeHandle> out_hes;
                    for (auto he : v_handle.outgoing_halfedges()) {
                        out_hes.push_back(he);
                    }
                    int d = out_hes.size();
                    std::vector<double> r(d);
                    std::vector<double> alpha(d);
                    double theta = 0.0;

                    auto ref_v = ref_mesh->vertex_handle(v_handle.idx());
                    auto p_i = ref_mesh->point(ref_v);

                    // 提取原网格中的距离和角度信息
                    for (int k = 0; k < d; ++k) {
                        auto he = out_hes[k];
                        auto ref_he = ref_mesh->halfedge_handle(he.idx());
                        auto p_k =
                            ref_mesh->point(ref_mesh->to_vertex_handle(ref_he));
                        r[k] = (p_k - p_i).norm();

                        int k_next = (k + 1) % d;
                        auto he_next = out_hes[k_next];
                        auto ref_he_next =
                            ref_mesh->halfedge_handle(he_next.idx());
                        auto p_k_next = ref_mesh->point(
                            ref_mesh->to_vertex_handle(ref_he_next));

                        OpenMesh::Vec3f u = p_k - p_i;
                        OpenMesh::Vec3f v = p_k_next - p_i;
                        double dot = u | v;
                        double cross = (u % v).norm();
                        alpha[k] = std::atan2(cross, dot);
                        theta += alpha[k];
                    }

                    // 映射到 2D 局部极坐标平面
                    std::vector<double> phi(d + 1, 0.0);
                    for (int k = 0; k < d; ++k) {
                        phi[k + 1] = phi[k] + 2.0 * M_PI * (alpha[k] / theta);
                    }

                    struct Point2D {
                        double x, y;
                    };
                    std::vector<Point2D> p2d(d);
                    for (int k = 0; k < d; ++k) {
                        p2d[k].x = r[k] * std::cos(phi[k]);
                        p2d[k].y = r[k] * std::sin(phi[k]);
                    }

                    // 计算原点在三个顶点构成的三角形中的重心坐标分量
                    auto area2D = [](Point2D p1, Point2D p2) {
                        return 0.5 * (p1.x * p2.y - p2.x * p1.y);
                    };

                    std::vector<double> lambda(d, 0.0);
                    for (int l = 0; l < d; ++l) {
                        // 反向射线的角度
                        double psi = std::fmod(phi[l] + M_PI, 2.0 * M_PI);
                        if (psi < 0)
                            psi += 2.0 * M_PI;

                        // 寻找反向射线穿过的线段 [p_m, p_{m+1}]
                        int m = 0;
                        for (int k = 0; k < d; ++k) {
                            if (psi >= phi[k] && psi < phi[k + 1]) {
                                m = k;
                                break;
                            }
                        }
                        int next = (m + 1) % d;

                        // 利用面积法计算重心坐标
                        double a1 = area2D(p2d[m], p2d[next]);
                        double a2 = area2D(p2d[next], p2d[l]);
                        double a3 = area2D(p2d[l], p2d[m]);
                        double sum_a = a1 + a2 + a3;

                        lambda[l] += (a1 / sum_a) / d;
                        lambda[m] += (a2 / sum_a) / d;
                        lambda[next] += (a3 / sum_a) / d;
                    }

                    //  填充稀疏矩阵
                    for (int k = 0; k < d; ++k) {
                        int j = out_hes[k].to().idx();
                        triplets.push_back(
                            Eigen::Triplet<double>(i, j, -lambda[k]));
                    }
                    triplets.push_back(Eigen::Triplet<double>(i, i, 1.0));
                }
            }
        }

        A.setFromTriplets(triplets.begin(), triplets.end());

        // 求解稀疏线性方程组
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);

        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "Minimal Surface: Failed to factorize the coefficient matrix.");
            return false;
        }

        Eigen::MatrixXd X = solver.solve(B);

        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "Minimal Surface: Failed to solve the linear system.");
            return false;
        }

        // 将求解出的新坐标更新回 Mesh
        for (auto v_handle : halfedge_mesh->vertices()) {
            int i = v_handle.idx();
            halfedge_mesh->set_point(
                v_handle, OpenMesh::Vec3f(X(i, 0), X(i, 1), X(i, 2)));
        }

        /* ----------------------------- Postprocess
         * ------------------------------
         */
        auto geometry = openmesh_to_operand(halfedge_mesh.get());

        //  根据当前的类型，把跑完的网格推送到对应的输出端口
        params.set_output(output_names[weight_type], std::move(*geometry));
    }

    return true;
}

NODE_DECLARATION_UI(hw5_param);
NODE_DEF_CLOSE_SCOPE