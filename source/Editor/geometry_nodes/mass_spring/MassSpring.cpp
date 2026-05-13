#include "MassSpring.h"

#include <iostream>

namespace USTC_CG::mass_spring {
MassSpring::MassSpring(const Eigen::MatrixXd& X, const EdgeSet& E)
{
    this->X = this->init_X = X;
    this->vel = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    this->E = E;

    std::cout << "number of edges: " << E.size() << std::endl;
    std::cout << "init mass spring" << std::endl;

    // Compute the rest pose edge length
    for (const auto& e : E) {
        Eigen::Vector3d x0 = X.row(e.first);
        Eigen::Vector3d x1 = X.row(e.second);
        this->E_rest_length.push_back((x0 - x1).norm());
    }

    // Initialize the mask for Dirichlet boundary condition
    dirichlet_bc_mask.resize(X.rows(), false);

    // (HW_TODO) Fix two vertices, feel free to modify this
    unsigned n_fix = sqrt(X.rows());  // Here we assume the cloth is square
    dirichlet_bc_mask[0] = true;
    dirichlet_bc_mask[n_fix - 1] = true;
}

void MassSpring::step()
{
    Eigen::Vector3d acceleration_ext = gravity + wind_ext_acc;

    unsigned n_vertices = X.rows();

    // The reason to not use 1.0 as mass per vertex: the cloth gets heavier as
    // we increase the resolution
    double mass_per_vertex = mass / n_vertices;

    //----------------------------------------------------
    // (HW Optional) Bonus part: Sphere collision
    Eigen::MatrixXd acceleration_collision =
        getSphereCollisionForce(sphere_center.cast<double>(), sphere_radius) /
        mass_per_vertex;
    //----------------------------------------------------

    if (time_integrator == IMPLICIT_EULER) {
        // Implicit Euler
        TIC(step)

        // (HW TODO)
        auto H_elastic = computeHessianSparse(stiffness);  // size = [nx3, nx3]

        // compute Y and assemble the full Hessian H_g
        // H_g = M / h^2 + H_elastic
        Eigen::SparseMatrix<double> H_g(n_vertices * 3, n_vertices * 3);
        std::vector<Eigen::Triplet<double>> triplets_M;
        double m_h2 = mass_per_vertex / (h * h);
        for (int i = 0; i < n_vertices * 3; i++) {
            triplets_M.emplace_back(i, i, m_h2);
        }
        Eigen::SparseMatrix<double> M_h2(n_vertices * 3, n_vertices * 3);
        M_h2.setFromTriplets(triplets_M.begin(), triplets_M.end());

        H_g = M_h2 + H_elastic;

        // Compute grad_g
        // Add collision acceleration to the external acceleration for implicit
        // integration
        Eigen::Vector3d total_acc = acceleration_ext;

        Eigen::MatrixXd grad_E = computeGrad(stiffness);
        Eigen::MatrixXd grad_g_mat = Eigen::MatrixXd::Zero(X.rows(), X.cols());
        for (int i = 0; i < n_vertices; i++) {
            // Include collision acceleration specific to each vertex
            Eigen::Vector3d vertex_acc =
                total_acc + acceleration_collision.row(i).transpose();

            grad_g_mat.row(i) = -(mass_per_vertex / h) * vel.row(i) -
                                mass_per_vertex * vertex_acc.transpose() +
                                grad_E.row(i);
        }
        Eigen::VectorXd grad_g_flatten = flatten(grad_g_mat);

        // Enforce Dirichlet boundary conditions by modifying the linear system
        for (int i = 0; i < n_vertices; i++) {
            if (dirichlet_bc_mask[i]) {
                for (int j = 0; j < 3; j++) {
                    int idx = 3 * i + j;
                    // Set the target gradient to 0 for fixed points
                    grad_g_flatten(idx) = 0.0;

                    // Use a massive penalty value (1e11) to strictly enforce
                    // delta_X = 0 This prevents numerical pollution to adjacent
                    // vertices.
                    H_g.coeffRef(idx, idx) += 1e11;
                }
            }
        }

        // Solve Newton's search direction with linear solver
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H_g);
        Eigen::VectorXd delta_X_flatten = solver.solve(-grad_g_flatten);

        // update X and vel
        Eigen::MatrixXd delta_X = unflatten(delta_X_flatten);
        for (int i = 0; i < n_vertices; i++) {
            if (dirichlet_bc_mask[i]) {
                vel.row(i).setZero();
            }
            else {
                X.row(i) += delta_X.row(i);
                vel.row(i) = delta_X.row(i) / h;  // V = (X_new - X_old) / h
            }
        }

        TOC(step)
    }
    else if (time_integrator == SEMI_IMPLICIT_EULER) {
        // Semi-implicit Euler
        Eigen::MatrixXd acceleration =
            -computeGrad(stiffness) / mass_per_vertex;
        acceleration.rowwise() += acceleration_ext.transpose();

        // -----------------------------------------------
        // (HW Optional)
        if (enable_sphere_collision) {
            acceleration += acceleration_collision;
        }
        // -----------------------------------------------

        // (HW TODO): Implement semi-implicit Euler time integration

        // Update velocity first
        vel += h * acceleration;

        // Apply damping
        if (enable_damping) {
            vel *= damping;
        }

        // Update X and vel with boundary conditions
        for (int i = 0; i < n_vertices; i++) {
            if (dirichlet_bc_mask[i]) {
                vel.row(i).setZero();  // Fix points have zero velocity
            }
            else {
                X.row(i) += h * vel.row(i);  // Update positions
            }
        }
    }
    else {
        std::cerr << "Unknown time integrator!" << std::endl;
        return;
    }
}

// There are different types of mass spring energy:
// For this homework we will adopt Prof. Huamin Wang's energy definition
// introduced in GAMES103 course Lecture 2 E = 0.5 * stiffness * sum_{i=1}^{n}
// (||x_i - x_j|| - l)^2 There exist other types of energy definition, e.g.,
// Prof. Minchen Li's energy definition
// https://www.cs.cmu.edu/~15769-f23/lec/3_Mass_Spring_Systems.pdf
double MassSpring::computeEnergy(double stiffness)
{
    double sum = 0.;
    unsigned i = 0;
    for (const auto& e : E) {
        auto diff = X.row(e.first) - X.row(e.second);
        auto l = E_rest_length[i];
        sum += 0.5 * stiffness * std::pow((diff.norm() - l), 2);
        i++;
    }
    return sum;
}

Eigen::MatrixXd MassSpring::computeGrad(double stiffness)
{
    Eigen::MatrixXd g = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    unsigned i = 0;
    for (const auto& e : E) {
        // --------------------------------------------------
        // (HW TODO): Implement the gradient computation
        Eigen::Vector3d diff = X.row(e.first) - X.row(e.second);
        double length = diff.norm();

        // Avoid division by zero
        if (length > 1e-6) {
            Eigen::Vector3d force_dir = diff / length;
            Eigen::Vector3d grad_e =
                stiffness * (length - E_rest_length[i]) * force_dir;

            g.row(e.first) += grad_e.transpose();
            g.row(e.second) -= grad_e.transpose();
        }
        // --------------------------------------------------
        i++;
    }
    return g;
}

Eigen::SparseMatrix<double> MassSpring::computeHessianSparse(double stiffness)
{
    unsigned n_vertices = X.rows();
    Eigen::SparseMatrix<double> H(n_vertices * 3, n_vertices * 3);
    std::vector<Eigen::Triplet<double>> triplets;  // Store non-zero entries

    unsigned i = 0;
    auto k = stiffness;
    const auto I =
        Eigen::Matrix3d::Identity();  // Using Matrix3d instead of MatrixXd
    for (const auto& e : E) {
        // --------------------------------------------------
        // (HW TODO): Implement the sparse version Hessian computation
        // Remember to consider fixed points
        // You can also consider positive definiteness here

        Eigen::Vector3d diff = X.row(e.first) - X.row(e.second);
        double len = diff.norm();
        double l = E_rest_length[i];

        Eigen::Matrix3d H_block = Eigen::Matrix3d::Zero();
        if (len > 1e-6) {
            Eigen::Vector3d dir = diff / len;
            Eigen::Matrix3d dir_dirT = dir * dir.transpose();

            if (len >= l) {
                // Exact Hessian when the spring is stretched
                H_block = k * dir_dirT + k * (1.0 - l / len) * (I - dir_dirT);
            }
            else {
                // Approximate Hessian when compressed to ensure Positive
                // Definiteness
                H_block = k * dir_dirT;
            }
        }

        // Add 3x3 block to the full 3Nx3N Hessian
        int v1 = e.first;
        int v2 = e.second;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                triplets.emplace_back(3 * v1 + r, 3 * v1 + c, H_block(r, c));
                triplets.emplace_back(3 * v2 + r, 3 * v2 + c, H_block(r, c));
                triplets.emplace_back(3 * v1 + r, 3 * v2 + c, -H_block(r, c));
                triplets.emplace_back(3 * v2 + r, 3 * v1 + c, -H_block(r, c));
            }
        }
        // --------------------------------------------------

        i++;
    }

    H.setFromTriplets(triplets.begin(), triplets.end());
    H.makeCompressed();
    return H;
}

bool MassSpring::checkSPD(const Eigen::SparseMatrix<double>& A)
{
    // Eigen::SimplicialLDLT<SparseMatrix_d> ldlt(A);
    // return ldlt.info() == Eigen::Success;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    auto eigen_values = es.eigenvalues();
    return eigen_values.minCoeff() >= 1e-10;
}

void MassSpring::reset()
{
    std::cout << "reset" << std::endl;
    this->X = this->init_X;
    this->vel.setZero();
}

// ----------------------------------------------------------------------------------
// (HW Optional) Bonus part
Eigen::MatrixXd MassSpring::getSphereCollisionForce(
    Eigen::Vector3d center,
    double radius)
{
    Eigen::MatrixXd force = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); i++) {
        // (HW Optional) Implement penalty-based force here
        Eigen::Vector3d x_i = X.row(i);
        Eigen::Vector3d dir = x_i - center;
        double dist = dir.norm();

        // Define the threshold radius for penalty force
        double threshold = collision_scale_factor * radius;

        // If the vertex penetrates the threshold boundary
        if (dist < threshold && dist > 1e-6) {
            Eigen::Vector3d n =
                dir / dist;  // Outward normal from sphere center
            // Penalty force magnitude
            double magnitude = collision_penalty_k * (threshold - dist);
            // Apply the force in the direction of the normal
            force.row(i) = (magnitude * n).transpose();
        }
    }
    return force;
}
// ----------------------------------------------------------------------------------

bool MassSpring::set_dirichlet_bc_mask(const std::vector<bool>& mask)
{
    if (mask.size() == X.rows()) {
        dirichlet_bc_mask = mask;
        return true;
    }
    else
        return false;
}

bool MassSpring::update_dirichlet_bc_vertices(const MatrixXd& control_vertices)
{
    for (int i = 0; i < dirichlet_bc_control_pair.size(); i++) {
        int idx = dirichlet_bc_control_pair[i].first;
        int control_idx = dirichlet_bc_control_pair[i].second;
        X.row(idx) = control_vertices.row(control_idx);
    }

    return true;
}

bool MassSpring::init_dirichlet_bc_vertices_control_pair(
    const MatrixXd& control_vertices,
    const std::vector<bool>& control_mask)
{
    if (control_mask.size() != control_vertices.rows())
        return false;

    // TODO: optimize this part from O(n) to O(1)
    // First, get selected_control_vertices
    std::vector<VectorXd> selected_control_vertices;
    std::vector<int> selected_control_idx;
    for (int i = 0; i < control_mask.size(); i++) {
        if (control_mask[i]) {
            selected_control_vertices.push_back(control_vertices.row(i));
            selected_control_idx.push_back(i);
        }
    }

    // Then update mass spring fixed vertices
    for (int i = 0; i < dirichlet_bc_mask.size(); i++) {
        if (dirichlet_bc_mask[i]) {
            // O(n^2) nearest point search, can be optimized
            // -----------------------------------------
            int nearest_idx = 0;
            double nearst_dist = 1e6;
            VectorXd X_i = X.row(i);
            for (int j = 0; j < selected_control_vertices.size(); j++) {
                double dist = (X_i - selected_control_vertices[j]).norm();
                if (dist < nearst_dist) {
                    nearst_dist = dist;
                    nearest_idx = j;
                }
            }
            //-----------------------------------------

            X.row(i) = selected_control_vertices[nearest_idx];
            dirichlet_bc_control_pair.push_back(
                std::make_pair(i, selected_control_idx[nearest_idx]));
        }
    }

    return true;
}

}  // namespace USTC_CG::mass_spring