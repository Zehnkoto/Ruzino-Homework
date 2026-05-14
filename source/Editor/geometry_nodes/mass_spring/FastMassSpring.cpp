#include "FastMassSpring.h"

#include <chrono>
#include <cmath>
#include <fstream>
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

    auto t_start = std::chrono::high_resolution_clock::now();

    // 4. Pre-factorize A matrix (This is the core of the speedup)
    solver.compute(A);

    auto t_end = std::chrono::high_resolution_clock::now();
    double precompute_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
            .count() /
        1000.0;

    if (solver.info() != Eigen::Success) {
        std::cerr << "FastMassSpring: A matrix pre-factorization failed!"
                  << std::endl;
    }
    else {
        std::cout << "FastMassSpring: A matrix pre-factorization succeeded! "
                  << "Time taken: " << precompute_ms << " ms." << std::endl;

        // Use "./" prefix to explicitly output the CSV file to the current
        // working directory
        std::ofstream init_log("./liu13_init_perf_log.csv", std::ios::app);
        init_log << "Vertices,Stiffness,PrefactorizeTime_ms\n";
        init_log << n_vertices << "," << stiffness << "," << precompute_ms
                 << "\n";
    }
}

void FastMassSpring::step()
{
    // (HW Optional) Necessary preparation
    TIC(step)

    // Use "./" prefix to explicitly output the CSV files to the current working
    // directory
    static std::ofstream perf_log(
        "./liu13_performance_log.csv", std::ios::out | std::ios::trunc);
    static std::ofstream conv_log(
        "./liu13_convergence_log.csv", std::ios::out | std::ios::trunc);
    static std::ofstream energy_log(
        "./liu13_energy_log.csv", std::ios::out | std::ios::trunc);
    static int frame_count = 0;
    static bool is_first_frame = true;

    if (is_first_frame) {
        perf_log << "Frame,TotalStepTime_ms,AvgIterTime_ms\n";
        conv_log << "Frame,Iteration,Error\n";
        energy_log
            << "Frame,KineticEnergy,ElasticEnergy,GravityEnergy,TotalEnergy\n";
        is_first_frame = false;
    }

    auto step_start_time = std::chrono::high_resolution_clock::now();

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

    double total_iter_time_ms = 0.0;

    for (unsigned iter = 0; iter < max_iter; iter++) {
        auto iter_start_time = std::chrono::high_resolution_clock::now();

        Eigen::MatrixXd X_guess_before = X_new;

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

        auto iter_end_time = std::chrono::high_resolution_clock::now();
        total_iter_time_ms +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                iter_end_time - iter_start_time)
                .count() /
            1000.0;

        double iter_error = 0.0;
        for (int v = 0; v < n_vertices; v++) {
            iter_error += (X_new.row(v) - X_guess_before.row(v)).norm();
        }
        iter_error /= n_vertices;

        conv_log << frame_count << "," << iter << "," << iter_error << "\n";
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

    double kinetic_energy = 0.5 * mass_per_vertex * vel.squaredNorm();

    double elastic_energy = computeEnergy(stiffness);

    double gravity_energy = 0.0;
    for (int i = 0; i < n_vertices; i++) {
        gravity_energy -= mass_per_vertex * gravity.dot(X.row(i));
    }

    double total_energy = kinetic_energy + elastic_energy + gravity_energy;

    energy_log << frame_count << "," << kinetic_energy << "," << elastic_energy
               << "," << gravity_energy << "," << total_energy << "\n";

    auto step_end_time = std::chrono::high_resolution_clock::now();
    double step_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                         step_end_time - step_start_time)
                         .count() /
                     1000.0;
    double avg_iter_ms = total_iter_time_ms / max_iter;

    perf_log << frame_count << "," << step_ms << "," << avg_iter_ms << "\n";

    frame_count++;

    TOC(step)
}

}  // namespace USTC_CG::mass_spring