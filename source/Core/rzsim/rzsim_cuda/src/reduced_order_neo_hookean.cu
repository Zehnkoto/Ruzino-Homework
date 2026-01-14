#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Eigen/Eigen>
#include <cmath>
#include <cstdio>  // For printf as error reporting
#include <vector>

#include "RHI/internal/cuda_extension.hpp"
#include "rzsim_cuda/adjacency_map.cuh"
#include "rzsim_cuda/reduced_order_neo_hookean.cuh"

// Include glm after CUDA headers to avoid conflicts
#ifndef __CUDACC__
#include <glm/glm.hpp>
#else
// For CUDA device code, we need basic vec3 definition
namespace glm {
struct vec3 {
    float x, y, z;
};
}  // namespace glm
#endif

RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

// ============================================================================
// Kernel: Build reduced order data
// ============================================================================

ReducedOrderData build_reduced_order_data_gpu(
    const void* basis_data,
    const void* rest_positions_data)
{
    // Cast back to actual types
    const auto& basis =
        *reinterpret_cast<const std::vector<Eigen::VectorXf>*>(basis_data);
    const auto& rest_positions =
        *reinterpret_cast<const std::vector<glm::vec3>*>(rest_positions_data);

    ReducedOrderData ro_data;
    ro_data.num_basis = basis.size();
    ro_data.num_particles = rest_positions.size();

    if (ro_data.num_basis == 0 || ro_data.num_particles == 0) {
        printf("[ReducedOrder] Empty basis or rest positions\n");
        return ro_data;
    }

    // Build basis weights matrix [num_particles, num_basis]
    // Each entry is the eigenvector weight for that vertex and basis
    std::vector<float> weights(ro_data.num_particles * ro_data.num_basis);

    for (int i = 0; i < ro_data.num_basis; ++i) {
        const auto& eigenvec = basis[i];
        if (eigenvec.size() != ro_data.num_particles) {
            printf(
                "[ReducedOrder] Basis %d size mismatch: %d vs %d\n",
                i,
                (int)eigenvec.size(),
                ro_data.num_particles);
            continue;
        }

        for (int v = 0; v < ro_data.num_particles; ++v) {
            weights[v * ro_data.num_basis + i] = eigenvec(v);
        }
    }

    ro_data.basis_weights = cuda::create_cuda_linear_buffer(weights);
    ro_data.rest_positions = cuda::create_cuda_linear_buffer(rest_positions);

    return ro_data;
}

// ============================================================================
// Kernel: Map reduced coordinates to full space
// x[v] = rest[v] + Σ_i weight[v,i] * (R_i * rest[v] + t_i - rest[v])
// ============================================================================

void map_reduced_to_full_gpu(
    cuda::CUDALinearBufferHandle q_reduced,
    const ReducedOrderData& ro_data,
    cuda::CUDALinearBufferHandle x_full)
{
    int num_particles = ro_data.num_particles;
    int num_basis = ro_data.num_basis;

    const float* q_reduced_ptr = q_reduced->get_device_ptr<float>();
    const float* basis_weights_ptr =
        ro_data.basis_weights->get_device_ptr<float>();
    const float* rest_positions_ptr =
        ro_data.rest_positions->get_device_ptr<float>();
    float* x_full_ptr = x_full->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "map_reduced_to_full", num_particles, [=] __device__(int v) {
            // Load rest position
            float rest_x = rest_positions_ptr[3 * v + 0];
            float rest_y = rest_positions_ptr[3 * v + 1];
            float rest_z = rest_positions_ptr[3 * v + 2];

            // Initialize with rest position
            float x = rest_x;
            float y = rest_y;
            float z = rest_z;

            // Add contributions from each basis
            for (int i = 0; i < num_basis; ++i) {
                float weight = basis_weights_ptr[v * num_basis + i];

                // Load affine transform parameters for basis i
                // Rotation matrix R (row-major, 3x3)
                float R00 = q_reduced_ptr[i * 12 + 0];
                float R01 = q_reduced_ptr[i * 12 + 1];
                float R02 = q_reduced_ptr[i * 12 + 2];
                float R10 = q_reduced_ptr[i * 12 + 3];
                float R11 = q_reduced_ptr[i * 12 + 4];
                float R12 = q_reduced_ptr[i * 12 + 5];
                float R20 = q_reduced_ptr[i * 12 + 6];
                float R21 = q_reduced_ptr[i * 12 + 7];
                float R22 = q_reduced_ptr[i * 12 + 8];

                // Translation t
                float tx = q_reduced_ptr[i * 12 + 9];
                float ty = q_reduced_ptr[i * 12 + 10];
                float tz = q_reduced_ptr[i * 12 + 11];

                // Apply affine transform: R * rest_pos + t
                float transformed_x =
                    R00 * rest_x + R01 * rest_y + R02 * rest_z + tx;
                float transformed_y =
                    R10 * rest_x + R11 * rest_y + R12 * rest_z + ty;
                float transformed_z =
                    R20 * rest_x + R21 * rest_y + R22 * rest_z + tz;

                // Add weighted contribution
                x += weight * (transformed_x - rest_x);
                y += weight * (transformed_y - rest_y);
                z += weight * (transformed_z - rest_z);
            }

            x_full_ptr[3 * v + 0] = x;
            x_full_ptr[3 * v + 1] = y;
            x_full_ptr[3 * v + 2] = z;
        });
}

// ============================================================================
// Kernel: Compute Jacobian matrix
// J[3v+d, 12i+p] = weight[v,i] * ∂(R_i*rest[v] + t_i)[d] / ∂q[12i+p]
// ============================================================================

void compute_jacobian_gpu(
    cuda::CUDALinearBufferHandle q_reduced,
    const ReducedOrderData& ro_data,
    cuda::CUDALinearBufferHandle jacobian)
{
    int num_particles = ro_data.num_particles;
    int num_basis = ro_data.num_basis;

    // Zero out jacobian first
    size_t jacobian_size = num_particles * 3 * num_basis * 12 * sizeof(float);
    cudaMemset(
        reinterpret_cast<void*>(jacobian->get_device_ptr()), 0, jacobian_size);

    const float* basis_weights_ptr =
        ro_data.basis_weights->get_device_ptr<float>();
    const float* rest_positions_ptr =
        ro_data.rest_positions->get_device_ptr<float>();
    float* jacobian_ptr = jacobian->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "compute_jacobian", num_particles, [=] __device__(int v) {
            float rest_x = rest_positions_ptr[3 * v + 0];
            float rest_y = rest_positions_ptr[3 * v + 1];
            float rest_z = rest_positions_ptr[3 * v + 2];

            // For each basis mode
            for (int i = 0; i < num_basis; ++i) {
                float weight = basis_weights_ptr[v * num_basis + i];

                // Derivatives w.r.t. rotation matrix elements
                // x_component: d(x)/d(R[row][col]) = weight * rest[col] if row
                // == 0 For x-component of vertex v
                int row_x = 3 * v + 0;
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 0] =
                    weight * rest_x;  // dR00
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 1] =
                    weight * rest_y;  // dR01
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 2] =
                    weight * rest_z;  // dR02
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 3] =
                    0.0f;  // dR10
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 4] =
                    0.0f;  // dR11
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 5] =
                    0.0f;  // dR12
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 6] =
                    0.0f;  // dR20
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 7] =
                    0.0f;  // dR21
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 8] =
                    0.0f;  // dR22
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 9] =
                    weight;  // dtx
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 10] =
                    0.0f;  // dty
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 11] =
                    0.0f;  // dtz

                // For y-component
                int row_y = 3 * v + 1;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 0] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 1] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 2] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 3] =
                    weight * rest_x;  // dR10
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 4] =
                    weight * rest_y;  // dR11
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 5] =
                    weight * rest_z;  // dR12
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 6] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 7] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 8] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 9] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 10] =
                    weight;  // dty
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 11] = 0.0f;

                // For z-component
                int row_z = 3 * v + 2;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 0] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 1] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 2] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 3] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 4] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 5] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 6] =
                    weight * rest_x;  // dR20
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 7] =
                    weight * rest_y;  // dR21
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 8] =
                    weight * rest_z;  // dR22
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 9] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 10] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 11] =
                    weight;  // dtz
            }
        });
}

// ============================================================================
// Kernel: Compute reduced gradient (J^T * grad_x)
// ============================================================================

void compute_reduced_gradient_gpu(
    cuda::CUDALinearBufferHandle jacobian,
    cuda::CUDALinearBufferHandle grad_x,
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle grad_q)
{
    int reduced_dof = num_basis * 12;
    int full_dof = num_particles * 3;

    const float* jacobian_ptr = jacobian->get_device_ptr<float>();
    const float* grad_x_ptr = grad_x->get_device_ptr<float>();
    float* grad_q_ptr = grad_q->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "compute_reduced_gradient", reduced_dof, [=] __device__(int idx) {
            float sum = 0.0f;

            // grad_q[idx] = J[:, idx]^T * grad_x = sum_i J[i, idx] * grad_x[i]
            for (int i = 0; i < full_dof; ++i) {
                sum += jacobian_ptr[i * reduced_dof + idx] * grad_x_ptr[i];
            }

            grad_q_ptr[idx] = sum;
        });
}

void compute_reduced_neg_gradient_gpu(
    cuda::CUDALinearBufferHandle jacobian,
    cuda::CUDALinearBufferHandle grad_x,
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle neg_grad_q)
{
    int reduced_dof = num_basis * 12;
    int full_dof = num_particles * 3;

    const float* jacobian_ptr = jacobian->get_device_ptr<float>();
    const float* grad_x_ptr = grad_x->get_device_ptr<float>();
    float* neg_grad_q_ptr = neg_grad_q->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "compute_reduced_neg_gradient", reduced_dof, [=] __device__(int idx) {
            float sum = 0.0f;

            // neg_grad_q[idx] = -J[:, idx]^T * grad_x = -sum_i J[i, idx] *
            // grad_x[i]
            for (int i = 0; i < full_dof; ++i) {
                sum += jacobian_ptr[i * reduced_dof + idx] * grad_x_ptr[i];
            }

            neg_grad_q_ptr[idx] = -sum;  // Negate the result
        });
}

// ============================================================================
// Kernel: Map reduced velocities to full space (v_full = J * q_dot)
// ============================================================================

void map_reduced_velocities_to_full_gpu(
    cuda::CUDALinearBufferHandle jacobian,
    cuda::CUDALinearBufferHandle q_dot,
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle v_full)
{
    int reduced_dof = num_basis * 12;
    int full_dof = num_particles * 3;

    const float* jacobian_ptr = jacobian->get_device_ptr<float>();
    const float* q_dot_ptr = q_dot->get_device_ptr<float>();
    float* v_full_ptr = v_full->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "map_reduced_velocities", full_dof, [=] __device__(int idx) {
            float sum = 0.0f;
            // v_full[idx] = J[idx, :] * q_dot = sum_j J[idx, j] * q_dot[j]
            for (int j = 0; j < reduced_dof; ++j) {
                sum += jacobian_ptr[idx * reduced_dof + j] * q_dot_ptr[j];
            }

            v_full_ptr[idx] = sum;
        });
}

// ============================================================================
// Kernel: Sparse matrix-dense matrix multiply (CSR * dense)
// ============================================================================

// ============================================================================
// Kernel: Dense transpose matrix multiply (A^T * B)
// ============================================================================

void compute_reduced_hessian_gpu(
    const NeoHookeanCSRStructure& hessian_structure,
    cuda::CUDALinearBufferHandle hessian_values,
    cuda::CUDALinearBufferHandle jacobian,
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle temp_buffer,
    cuda::CUDALinearBufferHandle H_q)
{
    int full_dof = num_particles * 3;
    int reduced_dof = num_basis * 12;

    // Step 1: temp = H_x * J
    // H_x is [full_dof, full_dof] sparse CSR
    // J is [full_dof, reduced_dof] dense
    // temp is [full_dof, reduced_dof] dense

    const int* row_offsets_ptr =
        hessian_structure.row_offsets->get_device_ptr<int>();
    const int* col_indices_ptr =
        hessian_structure.col_indices->get_device_ptr<int>();
    const float* hessian_values_ptr = hessian_values->get_device_ptr<float>();
    const float* jacobian_ptr = jacobian->get_device_ptr<float>();
    float* temp_buffer_ptr = temp_buffer->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "sparse_dense_multiply", full_dof, [=] __device__(int row) {
            int row_start = row_offsets_ptr[row];
            int row_end = row_offsets_ptr[row + 1];

            // For each column in the dense output
            for (int dc = 0; dc < reduced_dof; ++dc) {
                float sum = 0.0f;

                // Multiply row of sparse matrix with column dc of dense matrix
                for (int idx = row_start; idx < row_end; ++idx) {
                    int col = col_indices_ptr[idx];
                    float val = hessian_values_ptr[idx];
                    sum += val * jacobian_ptr[col * reduced_dof + dc];
                }

                temp_buffer_ptr[row * reduced_dof + dc] = sum;
            }
        });

    // Step 2: H_q = J^T * temp
    // J^T is [reduced_dof, full_dof]
    // temp is [full_dof, reduced_dof]
    // H_q is [reduced_dof, reduced_dof]

    float* H_q_ptr = H_q->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "dense_transpose_multiply",
        reduced_dof * reduced_dof,
        [=] __device__(int idx) {
            int row = idx / reduced_dof;
            int col = idx % reduced_dof;

            float sum = 0.0f;
            for (int i = 0; i < full_dof; ++i) {
                sum += jacobian_ptr[i * reduced_dof + row] *
                       temp_buffer_ptr[i * reduced_dof + col];
            }

            H_q_ptr[row * reduced_dof + col] = sum;
        });
}

// ============================================================================
// Initialize reduced coordinates to identity (R=I, t=0 for each basis)
// ============================================================================

void initialize_reduced_coords_identity_gpu(
    int num_basis,
    cuda::CUDALinearBufferHandle q)
{
    int reduced_dof = num_basis * 12;
    std::vector<float> host_q(reduced_dof, 0.0f);

    // For each basis: set rotation to identity (R00=R11=R22=1), translation to
    // 0
    for (int i = 0; i < num_basis; ++i) {
        host_q[i * 12 + 0] = 1.0f;  // R00
        host_q[i * 12 + 4] = 1.0f;  // R11
        host_q[i * 12 + 8] = 1.0f;  // R22
        // Rest are already 0
    }

    cudaMemcpy(
        reinterpret_cast<void*>(q->get_device_ptr()),
        host_q.data(),
        reduced_dof * sizeof(float),
        cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(
            "[ReducedOrder] CUDA error in "
            "initialize_reduced_coords_identity_gpu: %s\n",
            cudaGetErrorString(err));
    }
}

// ============================================================================
// Kernel: Explicit step in reduced space
// ============================================================================

void explicit_step_reduced_gpu(
    cuda::CUDALinearBufferHandle q,
    cuda::CUDALinearBufferHandle q_dot,
    float dt,
    int num_basis,
    cuda::CUDALinearBufferHandle q_tilde)
{
    int reduced_dof = num_basis * 12;

    const float* q_ptr = q->get_device_ptr<float>();
    const float* q_dot_ptr = q_dot->get_device_ptr<float>();
    float* q_tilde_ptr = q_tilde->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "explicit_step_reduced", reduced_dof, [=] __device__(int idx) {
            q_tilde_ptr[idx] = q_ptr[idx] + dt * q_dot_ptr[idx];
        });
}

// ============================================================================
// Kernel: Update reduced velocities
// ============================================================================

void update_reduced_velocities_gpu(
    cuda::CUDALinearBufferHandle q_new,
    cuda::CUDALinearBufferHandle q_old,
    float dt,
    float damping,
    int num_basis,
    cuda::CUDALinearBufferHandle q_dot)
{
    int reduced_dof = num_basis * 12;

    const float* q_new_ptr = q_new->get_device_ptr<float>();
    const float* q_old_ptr = q_old->get_device_ptr<float>();
    float* q_dot_ptr = q_dot->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "update_reduced_velocities", reduced_dof, [=] __device__(int idx) {
            q_dot_ptr[idx] = (q_new_ptr[idx] - q_old_ptr[idx]) / dt * damping;
        });
}

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
