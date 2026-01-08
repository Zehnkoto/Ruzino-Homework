#include <RHI/cuda.hpp>
#include <glm/glm.hpp>
#include <limits>
#include <set>

#include "GCore/Components/MeshComponent.h"
#include "GCore/Components/PointsComponent.h"
#include "GCore/geom_payload.hpp"
#include "RHI/internal/cuda_extension.hpp"
#include "RZSolver/Solver.hpp"
#include "nodes/core/def/node_def.hpp"
#include "rzsim_cuda/adjacency_map.cuh"
#include "rzsim_cuda/mass_spring_implicit.cuh"
#include "spdlog/spdlog.h"

NODE_DEF_OPEN_SCOPE

// Storage for persistent GPU simulation state
struct MassSpringImplicitGPUStorage {
    cuda::CUDALinearBufferHandle positions_buffer;
    cuda::CUDALinearBufferHandle velocities_buffer;
    cuda::CUDALinearBufferHandle springs_buffer;
    cuda::CUDALinearBufferHandle rest_lengths_buffer;
    cuda::CUDALinearBufferHandle next_positions_buffer;
    cuda::CUDALinearBufferHandle mass_matrix_buffer;
    cuda::CUDALinearBufferHandle gradients_buffer;
    cuda::CUDALinearBufferHandle f_ext_buffer;

    size_t geom_hash = 0;

    constexpr static bool has_storage = false;
};

NODE_DECLARATION_FUNCTION(mass_spring_implicit_gpu)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<float>("Mass").default_val(1.0f).min(0.01f).max(100.0f);
    b.add_input<float>("Stiffness")
        .default_val(1000.0f)
        .min(1.0f)
        .max(10000.0f);
    b.add_input<float>("Damping").default_val(1.0f).min(0.0f).max(1.0f);
    b.add_input<int>("Newton Iterations").default_val(30).min(1).max(100);
    b.add_input<float>("Newton Tolerance")
        .default_val(1e-2f)
        .min(1e-8f)
        .max(1e-1f);
    b.add_input<float>("Gravity").default_val(-9.81f).min(-20.0f).max(0.0f);
    b.add_input<float>("Ground Restitution")
        .default_val(0.3f)
        .min(0.0f)
        .max(1.0f);
    b.add_input<bool>("Flip Normal").default_val(false);

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(mass_spring_implicit_gpu)
{
    auto& global_payload = params.get_global_payload<GeomPayload&>();
    auto& storage = params.get_storage<MassSpringImplicitGPUStorage&>();

    // Get inputs
    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();

    float mass = params.get_input<float>("Mass");
    float stiffness = params.get_input<float>("Stiffness");
    float damping = params.get_input<float>("Damping");
    int max_iterations = params.get_input<int>("Newton Iterations");
    float tolerance = params.get_input<float>("Newton Tolerance");
    tolerance = std::max(tolerance, 1e-4f);
    float gravity = params.get_input<float>("Gravity");
    float restitution = params.get_input<float>("Ground Restitution");
    bool flip_normal = params.get_input<bool>("Flip Normal");
    float dt = global_payload.delta_time;

    // Get mesh component
    auto mesh_component = input_geom.get_component<MeshComponent>();
    std::vector<glm::vec3> positions;
    std::vector<int> face_vertex_indices;
    std::vector<int> face_counts;

    if (mesh_component) {
        positions = mesh_component->get_vertices();
        face_vertex_indices = mesh_component->get_face_vertex_indices();
        face_counts = mesh_component->get_face_vertex_counts();
    }
    else {
        auto points_component = input_geom.get_component<PointsComponent>();
        if (!points_component) {
            params.set_output<Geometry>("Geometry", std::move(input_geom));
            return true;
        }
        positions = points_component->get_vertices();
    }

    int num_particles = positions.size();
    if (num_particles == 0) {
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    if (input_geom.hash() != storage.geom_hash) {
        // Write positions to GPU buffer
        storage.positions_buffer = cuda::create_cuda_linear_buffer(positions);
        storage.velocities_buffer =
            cuda::create_cuda_linear_buffer<glm::vec3>(num_particles);
        storage.next_positions_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        storage.gradients_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        storage.f_ext_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Create mass matrix (diagonal with mass value per DOF, matching CPU)
        std::vector<float> mass_diag(num_particles * 3, mass);
        storage.mass_matrix_buffer = cuda::create_cuda_linear_buffer(mass_diag);

        auto face_indices = mesh_component->get_face_vertex_indices();
        auto triangles = cuda::create_cuda_linear_buffer(face_indices);

        storage.springs_buffer =
            rzsim_cuda::build_edge_set_gpu(storage.positions_buffer, triangles);

        // Compute rest lengths from initial positions
        storage.rest_lengths_buffer = rzsim_cuda::compute_rest_lengths_gpu(
            storage.positions_buffer, storage.springs_buffer);

        storage.geom_hash = input_geom.hash();
    }

    auto d_positions = storage.positions_buffer;
    auto d_velocities = storage.velocities_buffer;
    auto d_springs = storage.springs_buffer;
    auto d_rest_lengths = storage.rest_lengths_buffer;
    auto d_next_positions = storage.next_positions_buffer;
    auto d_M_diag = storage.mass_matrix_buffer;
    auto d_gradients = storage.gradients_buffer;
    auto d_f_ext = storage.f_ext_buffer;

    printf(
        "[GPU Params] mass=%.2f, k=%.1f, damp=%.3f, maxIter=%d, tol=%.2e, "
        "g=%.2f, rest=%.2f, dt=%.6f\n",
        mass,
        stiffness,
        damping,
        max_iterations,
        tolerance,
        gravity,
        restitution,
        dt);

    spdlog::info(
        "[GPU] Implicit solver: {} particles, {} springs",
        num_particles,
        storage.springs_buffer->getDesc().element_count / 2);

    // Setup external forces on GPU
    rzsim_cuda::setup_external_forces_gpu(
        mass, gravity, num_particles, d_f_ext);

    // Compute x_tilde = x + dt * v on GPU
    rzsim_cuda::explicit_step_gpu(
        d_positions, d_velocities, dt, num_particles, d_next_positions);

    // Debug: print first 3 particles before simulation
    auto positions_debug = d_positions->get_host_vector<glm::vec3>();
    auto velocities_debug = d_velocities->get_host_vector<glm::vec3>();

    // Print all positions if geometry is small
    if (num_particles <= 10) {
        printf("[GPU Before] All positions:\n");
        for (int i = 0; i < num_particles; i++) {
            printf(
                "  p[%d]=(%.6f, %.6f, %.6f)\n",
                i,
                positions_debug[i].x,
                positions_debug[i].y,
                positions_debug[i].z);
        }
        printf("[GPU Before] All velocities:\n");
        for (int i = 0; i < num_particles; i++) {
            printf(
                "  v[%d]=(%.6f, %.6f, %.6f)\n",
                i,
                velocities_debug[i].x,
                velocities_debug[i].y,
                velocities_debug[i].z);
        }
    }
    else {
        printf(
            "[GPU Before] p[0]=(%.6f, %.6f, %.6f), p[1]=(%.6f, %.6f, %.6f), "
            "p[2]=(%.6f, %.6f, %.6f)\n",
            positions_debug[0].x,
            positions_debug[0].y,
            positions_debug[0].z,
            positions_debug[1].x,
            positions_debug[1].y,
            positions_debug[1].z,
            positions_debug[2].x,
            positions_debug[2].y,
            positions_debug[2].z);
        printf(
            "[GPU Before] v[0]=(%.6f, %.6f, %.6f), v[1]=(%.6f, %.6f, %.6f)\n",
            velocities_debug[0].x,
            velocities_debug[0].y,
            velocities_debug[0].z,
            velocities_debug[1].x,
            velocities_debug[1].y,
            velocities_debug[1].z);
    }

    // Newton's method iterations
    auto d_x_new = cuda::create_cuda_linear_buffer<float>(num_particles * 3);
    // Initialize x_new = x (current position, NOT x_tilde) - matching CPU
    // implementation
    auto positions_flat = d_positions->get_host_vector<glm::vec3>();
    std::vector<float> x_init_flat(num_particles * 3);
    for (int i = 0; i < num_particles; i++) {
        x_init_flat[i * 3 + 0] = positions_flat[i].x;
        x_init_flat[i * 3 + 1] = positions_flat[i].y;
        x_init_flat[i * 3 + 2] = positions_flat[i].z;
    }
    d_x_new->assign_host_vector(x_init_flat);

    spdlog::info(
        "[GPU] Starting Newton iterations, max_iter={}, tol={:.2e}",
        max_iterations,
        tolerance);

    bool converged = false;
    for (int iter = 0; iter < max_iterations; iter++) {
        spdlog::info("[GPU] === Newton iteration {} ===", iter);

        // Debug: print current x_new at start of iteration
        auto x_curr_debug = d_x_new->get_host_vector<float>();
        if (iter <= 2) {
            printf(
                "[GPU Debug] Start iter %d: x[0:3]=(%.12e, %.12e, %.12e), x[6:9]=(%.12e, %.12e, %.12e)\n",
                iter,
                x_curr_debug[0],
                x_curr_debug[1],
                x_curr_debug[2],
                x_curr_debug[6],
                x_curr_debug[7],
                x_curr_debug[8]);
        }

        // Create fresh buffer for Newton direction each iteration to avoid warm
        // start issues
        auto d_p = cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Compute gradient at current x_new
        rzsim_cuda::compute_gradient_gpu(
            d_x_new,
            d_next_positions,  // x_tilde (unchanged)
            d_M_diag,
            d_f_ext,
            d_springs,
            d_rest_lengths,
            stiffness,
            dt,
            num_particles,
            d_gradients);

        // Check gradient norm for convergence
        auto grad_host = d_gradients->get_host_vector<float>();
        float grad_inf_norm = 0.0f;
        for (int i = 0; i < num_particles * 3; i++) {
            grad_inf_norm = std::max(grad_inf_norm, std::abs(grad_host[i]));
        }

        // Print full gradient for small problems (iter 0 and 1)
        if ((iter == 0 || iter == 1) && num_particles * 3 <= 100) {
            printf("[GPU] === Full gradient at iter %d ===\n", iter);
            for (int i = 0; i < num_particles * 3; i++) {
                printf("  grad[%d] = %.12e\n", i, grad_host[i]);
            }
            printf("[GPU] === End gradient ===\n");
        }
        else if (iter == 0) {
            printf("[GPU] First 10 gradient elements at iter 0:\n");
            for (int i = 0; i < std::min(10, num_particles * 3); i++) {
                printf("  grad[%d] = %.6e\n", i, grad_host[i]);
            }
        }

        spdlog::info(
            "[GPU] Iteration {}: grad_inf_norm={:.6e}, grad_inf_norm/dt={:.6e}",
            iter,
            grad_inf_norm,
            grad_inf_norm / dt);

        // Check for convergence
        if (!std::isfinite(grad_inf_norm)) {
            spdlog::error(
                "[GPU] Gradient contains NaN/Inf at iteration {}", iter);
            break;
        }

        if (grad_inf_norm / dt < tolerance) {
            spdlog::info(
                "[GPU] Converged at iteration {} with grad_norm={:.6e}",
                iter,
                grad_inf_norm / dt);
            converged = true;
            break;
        }

        // Assemble Hessian matrix
        auto hessian = rzsim_cuda::assemble_hessian_gpu(
            d_x_new,
            d_M_diag,
            d_springs,
            d_rest_lengths,
            stiffness,
            dt,
            num_particles);

        // Get Hessian data for printing
        auto row_offsets_host =
            hessian.row_offsets->get_host_vector<int>();
        auto col_indices_host =
            hessian.col_indices->get_host_vector<int>();
        auto values_host = hessian.values->get_host_vector<float>();

        // Print all entries for iter 0 and 1 if small matrix
        if (hessian.num_rows <= 100 && (iter == 0 || iter == 1)) {
            printf("[GPU] Assembling Hessian at iter %d: %d particles, %zu springs\n",
                iter, num_particles, storage.springs_buffer->getDesc().element_count / 2);
            printf("[GPU] === Full Hessian at iter %d (CSR format) ===\n", iter);
            for (int row = 0; row < hessian.num_rows; row++) {
                for (int idx = row_offsets_host[row];
                     idx < row_offsets_host[row + 1];
                     idx++) {
                    printf(
                        "  H(%d, %d) = %.12e\n",
                        row,
                        col_indices_host[idx],
                        values_host[idx]);
                }
            }
            printf("[GPU] === End Hessian ===\n");
        }
        else if (iter == 0) {
            printf(
                "[GPU] Assembling Hessian: %d particles, %zu springs\n",
                num_particles,
                storage.springs_buffer->getDesc().element_count / 2);
            printf("[GPU] dt = %.6f, stiffness = %.6f\n", dt, stiffness);
            printf(
                "[GPU] Matrix size: %d x %d\n",
                hessian.num_rows,
                hessian.num_cols);
            spdlog::info("[GPU] Hessian non-zeros: {}", hessian.nnz);

            printf("[GPU] First 10 non-zero entries (COO format):\n");
            int count = 0;
            for (int row = 0; row < hessian.num_rows && count < 10; row++) {
                for (int idx = row_offsets_host[row];
                     idx < row_offsets_host[row + 1] && count < 10;
                     idx++) {
                    printf(
                        "  (%d, %d) = %.6e\n",
                        row,
                        col_indices_host[idx],
                        values_host[idx]);
                    count++;
                }
            }

            printf("[GPU] CSR Format:\n");
            printf("  First 5 row_offsets: ");
            for (int i = 0; i < std::min(5, hessian.num_rows + 1); i++) {
                printf("%d ", row_offsets_host[i]);
            }
            printf("\n");
        }

        // Solve H * p = -grad using CUDA CG
        auto solver = Ruzino::Solver::SolverFactory::create(
            Ruzino::Solver::SolverType::CUDA_CG);

        // Adaptive CG tolerance based on gradient magnitude
        // CG residual should be 0.1% of gradient norm, but not too small
        float cg_tol = std::max(1e-8f, grad_inf_norm * 1e-3f);

        Ruzino::Solver::SolverConfig solver_config;
        solver_config.tolerance = cg_tol;
        solver_config.max_iterations = 1000;
        solver_config.use_preconditioner = true;
        solver_config.verbose =
            (iter <= 1);  // Verbose for first TWO iterations

        if (iter <= 1) {
            printf(
                "[GPU] Iter %d: CG tolerance set to %.6e (grad_inf_norm = "
                "%.6e)\n",
                iter, cg_tol, grad_inf_norm);
        }

        // Negate gradient for RHS: -grad
        auto d_neg_grad =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        std::vector<float> neg_grad_host(num_particles * 3);
        for (int i = 0; i < num_particles * 3; i++) {
            neg_grad_host[i] =
                -grad_host[i];  // Reuse grad_host from convergence check
        }
        d_neg_grad->assign_host_vector(neg_grad_host);

        if (iter == 0) {
            float neg_grad_norm = 0.0f;
            for (int i = 0; i < num_particles * 3; i++) {
                neg_grad_norm += neg_grad_host[i] * neg_grad_host[i];
            }
            neg_grad_norm = std::sqrt(neg_grad_norm);
            spdlog::info(
                "[GPU] CG RHS ||-grad|| = {:.6e}, grad[0:3]=({:.6f}, {:.6f}, "
                "{:.6f})",
                neg_grad_norm,
                -grad_host[0],
                -grad_host[1],
                -grad_host[2]);
        }

        // Solve on GPU
        auto result = solver->solveGPU(
            hessian.num_rows,
            hessian.nnz,
            reinterpret_cast<const int*>(hessian.row_offsets->get_device_ptr()),
            reinterpret_cast<const int*>(hessian.col_indices->get_device_ptr()),
            reinterpret_cast<const float*>(hessian.values->get_device_ptr()),
            reinterpret_cast<const float*>(d_neg_grad->get_device_ptr()),
            reinterpret_cast<float*>(d_p->get_device_ptr()),
            solver_config);

        //Debug: check what CG actually wrote to d_p
        auto p_after_cg = d_p->get_host_vector<float>();
        if ((iter == 0 || iter == 1) && num_particles * 3 <= 100) {
            printf("[GPU] === Full Newton direction p at iter %d ===\n", iter);
            for (int i = 0; i < num_particles * 3; i++) {
                printf("  p[%d] = %.12e\n", i, p_after_cg[i]);
            }
            printf("[GPU] === End p ===\n");
        }
        else if (iter <= 2) {
            printf(
                "[GPU Debug] Iter %d CG result: p[0:3]=(%.6e, %.6e, %.6e), p[3:6]=(%.6e, %.6e, %.6e)\n",
                iter,
                p_after_cg[0],
                p_after_cg[1],
                p_after_cg[2],
                p_after_cg[3],
                p_after_cg[4],
                p_after_cg[5]);
        }

        // Check if p is a descent direction: p^T * grad should be < 0
        float p_dot_grad = 0.0f;
        float p_norm_sq = 0.0f;
        float grad_norm_sq = 0.0f;
        for (int i = 0; i < num_particles * 3; i++) {
            p_dot_grad += p_after_cg[i] * grad_host[i];
            p_norm_sq += p_after_cg[i] * p_after_cg[i];
            grad_norm_sq += grad_host[i] * grad_host[i];
        }
        float cosine =
            p_dot_grad /
            (std::sqrt(p_norm_sq) * std::sqrt(grad_norm_sq) + 1e-20f);
        spdlog::info(
            "[GPU] Iter {}: p^T * grad = {:.6e}, cos(angle)={:.6f}",
            iter,
            p_dot_grad,
            cosine);

        if (!result.converged) {
            spdlog::error(
                "[GPU] Newton solve failed at iteration {}: {} (iters={}, "
                "residual={:.6e})",
                iter,
                result.error_message,
                result.iterations,
                result.final_residual);
            break;
        }

        // Check if Newton direction is valid
        auto p_check = d_p->get_host_vector<float>();
        float p_norm = 0.0f;
        for (int i = 0; i < num_particles * 3; i++) {
            p_norm += p_check[i] * p_check[i];
        }
        p_norm = std::sqrt(p_norm);

        if (iter == 0) {
            spdlog::info(
                "[GPU] CG converged: iters={}, residual={:.6e}, ||p||={:.6e}",
                result.iterations,
                result.final_residual,
                p_norm);
        }

        if (p_norm < 1e-12f) {
            spdlog::warn(
                "[GPU] Newton direction is nearly zero (||p||={:.6e})", p_norm);
            converged = true;
            break;
        }

        spdlog::info(
            "[GPU] Newton iter {}: grad_norm={:.6e}, CG_iters={}",
            iter,
            grad_inf_norm / dt,
            result.iterations);

        // Line search with energy descent
        float E_current = rzsim_cuda::compute_energy_gpu(
            d_x_new,
            d_next_positions,  // x_tilde
            d_M_diag,
            d_f_ext,
            d_springs,
            d_rest_lengths,
            stiffness,
            dt,
            num_particles);

        float alpha = 1.0f;
        auto d_x_candidate =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        int ls_iter = 0;
        float E_candidate =
            std::numeric_limits<float>::infinity();  // Start with +infinity so
                                                     // first check passes

        while (E_candidate > E_current && alpha > 1e-8f && ls_iter < 20) {
            // x_candidate = x_new + alpha * p
            auto x_new_host = d_x_new->get_host_vector<float>();
            auto p_host = d_p->get_host_vector<float>();
            std::vector<float> x_cand_host(num_particles * 3);
            for (int i = 0; i < num_particles * 3; i++) {
                x_cand_host[i] = x_new_host[i] + alpha * p_host[i];
            }
            d_x_candidate->assign_host_vector(x_cand_host);

            E_candidate = rzsim_cuda::compute_energy_gpu(
                d_x_candidate,
                d_next_positions,
                d_M_diag,
                d_f_ext,
                d_springs,
                d_rest_lengths,
                stiffness,
                dt,
                num_particles);

            // Log line search progress for first few iterations
            if (iter < 3 || (iter < 10 && ls_iter < 3)) {
                float energy_reduction = E_current - E_candidate;
                spdlog::info(
                    "[GPU] Iter {}, LS {}: alpha={:.3e}, E: {:.6e} -> {:.6e}, "
                    "reduction={:.6e}",
                    iter,
                    ls_iter,
                    alpha,
                    E_current,
                    E_candidate,
                    energy_reduction);
            }

            if (E_candidate <= E_current) {
                // Accept step
                spdlog::info(
                    "[GPU] Iter {}: Line search accepted at LS iter {}, "
                    "alpha={:.3e}",
                    iter,
                    ls_iter,
                    alpha);

                auto x_cand_final = d_x_candidate->get_host_vector<float>();
                d_x_new->assign_host_vector(x_cand_final);
                
                // Debug: verify x_new was actually updated
                if (iter <= 2) {
                    auto x_new_verify = d_x_new->get_host_vector<float>();
                    printf(
                        "[GPU Debug] After update iter %d: x[0:3]=(%.12e, %.12e, %.12e), x[6:9]=(%.12e, %.12e, %.12e)\n",
                        iter,
                        x_new_verify[0],
                        x_new_verify[1],
                        x_new_verify[2],
                        x_new_verify[6],
                        x_new_verify[7],
                        x_new_verify[8]);
                }
                break;
            }

            alpha *= 0.5f;
            ls_iter++;
        }

        if (alpha < 1e-8f) {
            spdlog::warn(
                "[GPU] Iter {}: Line search failed after {} attempts, could "
                "not reduce energy",
                iter,
                ls_iter);
            // Do not accept step if we couldn't find energy descent
            // This prevents divergence - we should break from Newton iterations
            break;
        }
    }

    if (!converged) {
        spdlog::warn(
            "[GPU] Newton method did not converge in {} iterations",
            max_iterations);
    }

    // Update velocities: v = (x_new - x_n) / dt
    auto x_new_final = d_x_new->get_host_vector<float>();
    auto x_n_host = d_positions->get_host_vector<glm::vec3>();
    std::vector<float> v_new(num_particles * 3);
    for (int i = 0; i < num_particles; i++) {
        v_new[i * 3 + 0] = (x_new_final[i * 3 + 0] - x_n_host[i].x) / dt * damping;
        v_new[i * 3 + 1] = (x_new_final[i * 3 + 1] - x_n_host[i].y) / dt * damping;
        v_new[i * 3 + 2] = (x_new_final[i * 3 + 2] - x_n_host[i].z) / dt * damping;
    }
    d_velocities->assign_host_vector(v_new);
    d_positions->assign_host_vector(x_new_final);

    // Debug print
    auto positions_debug_after = d_positions->get_host_vector<glm::vec3>();
    auto velocities_debug_after = d_velocities->get_host_vector<glm::vec3>();

    // Print all positions if geometry is small
    if (num_particles <= 10) {
        printf("[GPU After] All positions:\n");
        for (int i = 0; i < num_particles; i++) {
            printf(
                "  p[%d]=(%.6f, %.6f, %.6f)\n",
                i,
                positions_debug_after[i].x,
                positions_debug_after[i].y,
                positions_debug_after[i].z);
        }
        printf("[GPU After] All velocities:\n");
        for (int i = 0; i < num_particles; i++) {
            printf(
                "  v[%d]=(%.6f, %.6f, %.6f)\n",
                i,
                velocities_debug_after[i].x,
                velocities_debug_after[i].y,
                velocities_debug_after[i].z);
        }
    }
    else {
        printf(
            "[GPU After]  p[0]=(%.6f, %.6f, %.6f), p[1]=(%.6f, %.6f, %.6f), "
            "p[2]=(%.6f, %.6f, %.6f)\n",
            positions_debug_after[0].x,
            positions_debug_after[0].y,
            positions_debug_after[0].z,
            positions_debug_after[1].x,
            positions_debug_after[1].y,
            positions_debug_after[1].z,
            positions_debug_after[2].x,
            positions_debug_after[2].y,
            positions_debug_after[2].z);
        printf(
            "[GPU After]  v[0]=(%.6f, %.6f, %.6f), v[1]=(%.6f, %.6f, %.6f)\n",
            velocities_debug_after[0].x,
            velocities_debug_after[0].y,
            velocities_debug_after[0].z,
            velocities_debug_after[1].x,
            velocities_debug_after[1].y,
            velocities_debug_after[1].z);
    }

    // Convert to output format
    std::vector<glm::vec3> new_positions = positions_debug_after;
    std::vector<float> positions_out(3 * num_particles);
    std::vector<float> velocities_out(3 * num_particles);

    // new_positions already populated from GPU buffers above

    // Update geometry with new positions
    if (mesh_component) {
        mesh_component->set_vertices(new_positions);

        // Recalculate normals
        std::vector<glm::vec3> normals;
        normals.reserve(face_vertex_indices.size());

        int idx = 0;
        for (int face_count : face_counts) {
            if (face_count >= 3) {
                int i0 = face_vertex_indices[idx];
                int i1 = face_vertex_indices[idx + 1];
                int i2 = face_vertex_indices[idx + 2];

                glm::vec3 edge1 = new_positions[i1] - new_positions[i0];
                glm::vec3 edge2 = new_positions[i2] - new_positions[i0];
                glm::vec3 normal = glm::cross(
                    flip_normal ? edge1 : edge2, flip_normal ? edge2 : edge1);

                float length = glm::length(normal);
                if (length > 1e-8f) {
                    normal = normal / length;
                }
                else {
                    normal = glm::vec3(0.0f, 0.0f, 1.0f);
                }

                for (int i = 0; i < face_count; ++i) {
                    normals.push_back(normal);
                }
            }
            idx += face_count;
        }

        if (!normals.empty()) {
            mesh_component->set_normals(normals);
        }
    }
    else {
        auto points_component = input_geom.get_component<PointsComponent>();
        points_component->set_vertices(new_positions);
    }

    params.set_output<Geometry>("Geometry", std::move(input_geom));
    return true;
}

NODE_DECLARATION_UI(mass_spring_implicit_gpu);
NODE_DEF_CLOSE_SCOPE
