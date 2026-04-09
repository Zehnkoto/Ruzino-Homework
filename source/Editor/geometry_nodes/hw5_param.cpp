#include <time.h>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>  // 注意：求解非对称稀疏矩阵需要引入此头文件
#include <cmath>

// #include "GCore/Components/MeshOperand.h"
#include <pxr/usd/usdGeom/mesh.h>

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <cfloat>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

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

    /*
    ** NOTE: You can add more inputs or outputs if necessary. For example, in
    *some cases,
    ** additional information (e.g. other mesh geometry, other parameters) is
    *required to perform
    ** the computation.
    **
    ** Be sure that the input/outputs do not share the same name. You can add
    *one geometry as
    **
    **                b.add_input<Geometry>("Input");
    **
    ** Or maybe you need a value buffer like:
    **
    **                b.add_input<float1Buffer>("Weights");
    */

    // Output-1: Minimal surface with fixed boundary
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_param)
{
    // Get the input from params
    auto input = params.get_input<Geometry>("Input");

    // (TO BE UPDATED) Avoid processing the node when there is no input
    if (!input.get_component<MeshComponent>()) {
        throw std::runtime_error("Minimal Surface: Need Geometry Input.");
        return false;
    }

    /* ----------------------------- Preprocess -------------------------------
    ** Create a halfedge structure (using OpenMesh) for the input mesh. The
    ** half-edge data structure is a widely used data structure in geometric
    ** processing, offering convenient operations for traversing and modifying
    ** mesh elements.
    */
    auto halfedge_mesh = operand_to_openmesh(&input);

    /* ---------------- [HW4_TODO] TASK 1: Minimal Surface --------------------
    ** In this task, you are required to generate a 'minimal surface' mesh with
    ** the boundary of the input mesh as its boundary.
    **
    ** Specifically, the positions of the boundary vertices of the input mesh
    ** should be fixed. By solving a global Laplace equation on the mesh,
    ** recalculate the coordinates of the vertices inside the mesh to achieve
    ** the minimal surface configuration.
    **
    ** (Recall the Poisson equation with Dirichlet Boundary Condition in HW3)
    */

    /*
    ** Algorithm Pseudocode for Minimal Surface Calculation
    ** ------------------------------------------------------------------------
    ** 1. Initialize mesh with input boundary conditions.
    **    - For each boundary vertex, fix its position.
    **    - For internal vertices, initialize with initial guess if necessary.
    **
    ** 2. Construct Laplacian matrix for the mesh.
    **    - Compute weights for each edge based on the chosen weighting scheme
    **      (e.g., uniform weights for simplicity).
    **    - Assemble the global Laplacian matrix.
    **
    ** 3. Solve the Laplace equation for interior vertices.
    **    - Apply Dirichlet boundary conditions for boundary vertices.
    **    - Solve the linear system (Laplacian * X = 0) to find new positions
    **      for internal vertices.
    **
    ** 4. Update mesh geometry with new vertex positions.
    **    - Ensure the mesh respects the minimal surface configuration.
    **
    ** Note: This pseudocode outlines the general steps for calculating a
    ** minimal surface mesh given fixed boundary conditions using the Laplace
    ** equation. The specific implementation details may vary based on the mesh
    ** representation and numerical methods used.
    **
    */
    // 获取顶点总数，初始化 Eigen 稀疏矩阵 A 和右端项矩阵 B
    int n_vertices = halfedge_mesh->n_vertices();
    Eigen::SparseMatrix<double> A(n_vertices, n_vertices);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_vertices, 3);

    // 使用 Triplet 列表来高效构建稀疏矩阵
    std::vector<Eigen::Triplet<double>> triplets;
    // 预分配空间，假设每个点平均度数为 6，加上主对角线 1 个
    triplets.reserve(n_vertices * 7);

    // 遍历所有顶点，构建线性方程组
    for (const auto& v_handle : halfedge_mesh->vertices()) {
        int i = v_handle.idx();

        // 如果是边界点
        if (halfedge_mesh->is_boundary(v_handle)) {
            triplets.push_back(Eigen::Triplet<double>(i, i, 1.0));

            // 右端项为其原始三维空间坐标
            auto pt = halfedge_mesh->point(v_handle);
            B(i, 0) = pt[0];
            B(i, 1) = pt[1];
            B(i, 2) = pt[2];
        }
        // 如果是内部点
        else {
            double degree = 0.0;
            // 遍历 1-邻域的半边
            for (const auto& he_handle : v_handle.outgoing_halfedges()) {
                int j = he_handle.to().idx();  // 邻居顶点的索引
                triplets.push_back(Eigen::Triplet<double>(i, j, -1.0));
                degree += 1.0;
            }
            // 主对角线为度数 d_i
            triplets.push_back(Eigen::Triplet<double>(i, i, degree));

            // 右端项 B 矩阵对应行在初始化时已经是 0，无需操作
        }
    }

    // 从 Triplet 构建稀疏矩阵 A
    A.setFromTriplets(triplets.begin(), triplets.end());

    // 求解稀疏线性方程组 AX = B
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "Minimal Surface: Failed to factorize the coefficient matrix.");
        return false;
    }

    // 求解得到所有顶点的新坐标
    Eigen::MatrixXd X = solver.solve(B);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "Minimal Surface: Failed to solve the linear system.");
        return false;
    }

    // 将求解出的新坐标更新回 Mesh 结构中
    for (auto v_handle : halfedge_mesh->vertices()) {
        int i = v_handle.idx();
        halfedge_mesh->set_point(
            v_handle, OpenMesh::Vec3f(X(i, 0), X(i, 1), X(i, 2)));
    }

    /* ----------------------------- Postprocess ------------------------------
    ** Convert the minimal surface mesh from the halfedge structure back to
    ** Geometry format as the node's output.
    */
    auto geometry = openmesh_to_operand(halfedge_mesh.get());

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(hw5_param);
NODE_DEF_CLOSE_SCOPE