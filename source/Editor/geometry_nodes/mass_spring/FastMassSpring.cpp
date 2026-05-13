#include "FastMassSpring.h"

#include <iostream>

#include "utils.h"  // Assuming flatten/unflatten are defined here

namespace USTC_CG::mass_spring {
FastMassSpring::FastMassSpring(
    const Eigen::MatrixXd& X,
    const EdgeSet& E,
    const float stiffness,
    const float h)
    : MassSpring(X, E)
{
    // construct L and J at initialization
    std::cout << "init fast mass spring" << std::endl;

    unsigned n_vertices = X.rows();
    this->stiffness = stiffness;
    this->h = h;

    Eigen::SparseMatrix<double> A(n_vertices * 3, n_vertices * 3);
    A.setZero();

    // (HW Optional) precompute A and prefactorize
    // Note: one thing to take care of: A is related with stiffness, if
    // stiffness changes, A need to be recomputed

    double mass_per_vertex = mass / n_vertices;
    std::vector<Eigen::Triplet<double>> triplets_A;

    // 1. Build Mass matrix M
    for (int i = 0; i < n_vertices * 3; i++) {
        triplets_A.emplace_back(i, i, mass_per_vertex);
    }

    // 2. Build h^2 * L matrix (Laplacian scaled by stiffness and time step
    // squared)
    double h2 = h * h;
    for (const auto& e : E) {
        int v1 = e.first;
        int v2 = e.second;

        for (int r = 0; r < 3; ++r) {
            // Add k to diagonal blocks (v1, v1) and (v2, v2)
            triplets_A.emplace_back(3 * v1 + r, 3 * v1 + r, h2 * stiffness);
            triplets_A.emplace_back(3 * v2 + r, 3 * v2 + r, h2 * stiffness);

            // Subtract k from off-diagonal blocks (v1, v2) and (v2, v1)
            triplets_A.emplace_back(3 * v1 + r, 3 * v2 + r, -h2 * stiffness);
            triplets_A.emplace_back(3 * v2 + r, 3 * v1 + r, -h2 * stiffness);
        }
    }

    A.setFromTriplets(triplets_A.begin(), triplets_A.end());

    // 3. Enforce Dirichlet boundary conditions using the penalty method
    for (int i = 0; i < n_vertices; i++) {
        if (dirichlet_bc_mask[i]) {
            for (int j = 0; j < 3; j++) {
                int idx = 3 * i + j;
                A.coeffRef(idx, idx) +=
                    1e11;  // Huge penalty value to fix the vertex
            }
        }
    }

    A.makeCompressed();

    // 4. Pre-factorize A matrix (This is the core of the speedup)
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "FastMassSpring: A matrix pre-factorization failed!"
                  << std::endl;
    }
    else {
        std::cout << "FastMassSpring: A matrix pre-factorization succeeded!"
                  << std::endl;
    }
}

void FastMassSpring::step()
{
    // (HW Optional) Necessary preparation
    TIC(step)

    unsigned n_vertices = X.rows();
    double mass_per_vertex = mass / n_vertices;

    // Compute external accelerations (Gravity + Wind)
    Eigen::MatrixXd acc_ext_mat = Eigen::MatrixXd::Zero(n_vertices, 3);
    Eigen::Vector3d acceleration_ext = gravity + wind_ext_acc;
    acc_ext_mat.rowwise() += acceleration_ext.transpose();

    // Add collision penalty force if enabled
    if (enable_sphere_collision) {
        Eigen::MatrixXd collision_force = getSphereCollisionForce(
            sphere_center.cast<double>(), sphere_radius);
        acc_ext_mat += collision_force / mass_per_vertex;
    }

    // Compute inertial position Y = X_old + h * v_old + h^2 * M^{-1} * f_ext
    Eigen::MatrixXd Y = X + h * vel + h * h * acc_ext_mat;

    // Pre-compute My = M * Y for the right-hand side (b)
    Eigen::VectorXd My_flatten = flatten(Y) * mass_per_vertex;

    // Initialize the current guess for the new positions
    Eigen::MatrixXd X_new = X;

    for (unsigned iter = 0; iter < max_iter; iter++) {
        // (HW Optional)
        // local_step and global_step alternating solving

        // --- Local Step: Compute spring directions (d_i) and assemble Jd ---
        Eigen::MatrixXd Jd_mat = Eigen::MatrixXd::Zero(n_vertices, 3);
        unsigned i = 0;
        for (const auto& e : E) {
            int v1 = e.first;
            int v2 = e.second;
            Eigen::Vector3d diff = X_new.row(v1) - X_new.row(v2);
            double length = diff.norm();
            double rest_length = E_rest_length[i];

            if (length > 1e-6) {
                // Determine the ideal rest-state vector for this spring
                Eigen::Vector3d d_i = (diff / length) * rest_length;

                // Multiply by stiffness and apply to incidence matrix
                // components
                Eigen::Vector3d kd = stiffness * d_i;
                Jd_mat.row(v1) += kd.transpose();
                Jd_mat.row(v2) -= kd.transpose();
            }
            i++;
        }

        // --- Global Step: Solve Ax = b ---
        // b = h^2 * Jd + My
        Eigen::VectorXd b = h * h * flatten(Jd_mat) + My_flatten;

        // Apply penalty method for boundary conditions on the right-hand side
        for (int i = 0; i < n_vertices; i++) {
            if (dirichlet_bc_mask[i]) {
                for (int j = 0; j < 3; j++) {
                    int idx = 3 * i + j;
                    b(idx) = 1e11 * init_X(i, j);  // Force the solution to stay
                                                   // at the initial position
                }
            }
        }

        // Extremely fast solve using the pre-factorized Cholesky solver
        Eigen::VectorXd X_new_flatten = solver.solve(b);
        X_new = unflatten(X_new_flatten);
    }

    // Update the final velocities and positions
    for (int i = 0; i < n_vertices; i++) {
        if (dirichlet_bc_mask[i]) {
            vel.row(i).setZero();
            X.row(i) = init_X.row(i);
        }
        else {
            vel.row(i) = (X_new.row(i) - X.row(i)) / h;
            X.row(i) = X_new.row(i);
        }
    }

    TOC(step)
}

}  // namespace USTC_CG::mass_spring