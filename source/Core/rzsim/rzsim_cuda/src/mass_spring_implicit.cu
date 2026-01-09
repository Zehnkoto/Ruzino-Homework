#include <cusparse.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <Eigen/Dense>
#include <RHI/cuda.hpp>
#include <cstddef>

#include "rzsim_cuda/mass_spring_implicit.cuh"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

// Functor for comparing edge pairs
struct EdgePairEqual {
    __host__ __device__ bool operator()(
        const thrust::tuple<int, int>& a,
        const thrust::tuple<int, int>& b) const
    {
        return thrust::get<0>(a) == thrust::get<0>(b) &&
               thrust::get<1>(a) == thrust::get<1>(b);
    }
};

cuda::CUDALinearBufferHandle build_edge_set_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle edges)
{
    // Get triangle count
    size_t num_triangles = edges->getDesc().element_count / 3;

    // Allocate temporary buffer for all edges (3 edges per triangle)
    thrust::device_vector<int> all_edges(num_triangles * 6);
    const int* triangles = edges->get_device_ptr<int>();
    int* edge_pairs = thrust::raw_pointer_cast(all_edges.data());

    // Extract edges from triangles
    cuda::GPUParallelFor(
        "extract_edges", num_triangles, GPU_LAMBDA_Ex(int tid) {
            int base_idx = tid * 3;
            int v0 = triangles[base_idx];
            int v1 = triangles[base_idx + 1];
            int v2 = triangles[base_idx + 2];

            int output_base = tid * 6;
            // Edge 0-1
            edge_pairs[output_base + 0] = min(v0, v1);
            edge_pairs[output_base + 1] = max(v0, v1);
            // Edge 1-2
            edge_pairs[output_base + 2] = min(v1, v2);
            edge_pairs[output_base + 3] = max(v1, v2);
            // Edge 2-0
            edge_pairs[output_base + 4] = min(v2, v0);
            edge_pairs[output_base + 5] = max(v2, v0);
        });

    // Create vectors for edge pairs
    thrust::device_vector<int> edge_first(num_triangles * 3);
    thrust::device_vector<int> edge_second(num_triangles * 3);
    const int* interleaved = thrust::raw_pointer_cast(all_edges.data());
    int* first = thrust::raw_pointer_cast(edge_first.data());
    int* second = thrust::raw_pointer_cast(edge_second.data());

    // Separate the interleaved edge data
    cuda::GPUParallelFor(
        "separate_edges", num_triangles * 3, GPU_LAMBDA_Ex(int tid) {
            first[tid] = interleaved[tid * 2];
            second[tid] = interleaved[tid * 2 + 1];
        });

    // Create zip iterator
    auto edge_begin = thrust::make_zip_iterator(
        thrust::make_tuple(edge_first.begin(), edge_second.begin()));
    auto edge_end = thrust::make_zip_iterator(
        thrust::make_tuple(edge_first.end(), edge_second.end()));

    // Sort edges
    thrust::sort(
        edge_begin,
        edge_end,
        [] __device__(
            const thrust::tuple<int, int>& a,
            const thrust::tuple<int, int>& b) {
            if (thrust::get<0>(a) != thrust::get<0>(b))
                return thrust::get<0>(a) < thrust::get<0>(b);
            return thrust::get<1>(a) < thrust::get<1>(b);
        });

    // Remove duplicates
    auto new_end = thrust::unique(edge_begin, edge_end, EdgePairEqual());

    // Calculate unique edge count
    size_t num_unique_edges = new_end - edge_begin;

    // Copy unique edges to output buffer (interleaved format)
    auto output_buffer =
        cuda::create_cuda_linear_buffer<int>(size_t(num_unique_edges * 2));
    int* output_ptr = output_buffer->get_device_ptr<int>();
    const int* edge_first_ptr = thrust::raw_pointer_cast(edge_first.data());
    const int* edge_second_ptr = thrust::raw_pointer_cast(edge_second.data());

    // Interleave the data
    cuda::GPUParallelFor(
        "interleave_edges", num_unique_edges, GPU_LAMBDA_Ex(int tid) {
            output_ptr[tid * 2] = edge_first_ptr[tid];
            output_ptr[tid * 2 + 1] = edge_second_ptr[tid];
        });

    return output_buffer;
}

// Build adjacency list: for each vertex, store list of adjacent vertices
// Returns: (adjacent_vertices, vertex_offsets, rest_lengths)
// Format: adjacent_vertices[vertex_offsets[v]..vertex_offsets[v+1]] = adjacent
// vertex indices for vertex v
std::tuple<
    cuda::CUDALinearBufferHandle,
    cuda::CUDALinearBufferHandle,
    cuda::CUDALinearBufferHandle>
build_adjacency_list_gpu(
    cuda::CUDALinearBufferHandle triangles,
    cuda::CUDALinearBufferHandle positions,
    int num_particles)
{
    // Step 1: Extract and deduplicate edges
    auto springs =
        build_edge_set_gpu(cuda::CUDALinearBufferHandle(), triangles);
    int num_springs = springs->getDesc().element_count / 2;
    const int* springs_ptr = springs->get_device_ptr<int>();

    // Step 2: Compute rest lengths for each edge
    auto rest_lengths_per_edge = compute_rest_lengths_gpu(positions, springs);

    // Step 3: Count adjacent vertices per vertex
    auto d_adj_count = cuda::create_cuda_linear_buffer<int>(num_particles);
    int* count_ptr = d_adj_count->get_device_ptr<int>();
    cudaMemset(count_ptr, 0, num_particles * sizeof(int));

    // Each edge contributes 2 adjacencies
    cuda::GPUParallelFor(
        "count_adjacencies", num_springs, GPU_LAMBDA_Ex(int s) {
            int i = springs_ptr[s * 2];
            int j = springs_ptr[s * 2 + 1];
            atomicAdd(&count_ptr[i], 1);
            atomicAdd(&count_ptr[j], 1);
        });

    // Step 4: Build offset buffer (prefix sum)
    auto d_offsets = cuda::create_cuda_linear_buffer<int>(num_particles + 1);
    int* offsets_ptr = d_offsets->get_device_ptr<int>();

    thrust::device_ptr<int> count_thrust(count_ptr);
    thrust::device_ptr<int> offsets_thrust(offsets_ptr);
    thrust::exclusive_scan(
        thrust::device,
        count_thrust,
        count_thrust + num_particles,
        offsets_thrust);

    // Get total count
    int total_entries;
    cudaMemcpy(
        &total_entries,
        count_ptr + num_particles - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost);
    int last_offset;
    cudaMemcpy(
        &last_offset,
        offsets_ptr + num_particles - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost);
    total_entries += last_offset;
    cudaMemcpy(
        offsets_ptr + num_particles,
        &total_entries,
        sizeof(int),
        cudaMemcpyHostToDevice);

    // Step 5: Allocate adjacency and rest length buffers
    auto d_adjacent_vertices =
        cuda::create_cuda_linear_buffer<int>(total_entries);
    auto d_rest_lengths = cuda::create_cuda_linear_buffer<float>(total_entries);
    int* adj_ptr = d_adjacent_vertices->get_device_ptr<int>();
    float* rest_ptr = d_rest_lengths->get_device_ptr<float>();
    const float* edge_rest_ptr = rest_lengths_per_edge->get_device_ptr<float>();

    // Reset counts for filling
    cudaMemset(count_ptr, 0, num_particles * sizeof(int));

    // Step 6: Fill adjacency lists
    cuda::GPUParallelFor(
        "fill_adjacency", num_springs, GPU_LAMBDA_Ex(int s) {
            int i = springs_ptr[s * 2];
            int j = springs_ptr[s * 2 + 1];
            float rest_len = edge_rest_ptr[s];

            // Add j to i's adjacency list
            int pos_i = offsets_ptr[i] + atomicAdd(&count_ptr[i], 1);
            adj_ptr[pos_i] = j;
            rest_ptr[pos_i] = rest_len;

            // Add i to j's adjacency list
            int pos_j = offsets_ptr[j] + atomicAdd(&count_ptr[j], 1);
            adj_ptr[pos_j] = i;
            rest_ptr[pos_j] = rest_len;
        });

    cudaDeviceSynchronize();

    return { d_adjacent_vertices, d_offsets, d_rest_lengths };
}

void explicit_step_gpu(
    cuda::CUDALinearBufferHandle x,
    cuda::CUDALinearBufferHandle v,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle x_tilde)
{
    const float* x_ptr = x->get_device_ptr<float>();
    const float* v_ptr = v->get_device_ptr<float>();
    float* x_tilde_ptr = x_tilde->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "explicit_step", num_particles, GPU_LAMBDA_Ex(int i) {
            x_tilde_ptr[i * 3 + 0] = x_ptr[i * 3 + 0] + dt * v_ptr[i * 3 + 0];
            x_tilde_ptr[i * 3 + 1] = x_ptr[i * 3 + 1] + dt * v_ptr[i * 3 + 1];
            x_tilde_ptr[i * 3 + 2] = x_ptr[i * 3 + 2] + dt * v_ptr[i * 3 + 2];
        });
}

void setup_external_forces_gpu(
    float mass,
    float gravity,
    int num_particles,
    cuda::CUDALinearBufferHandle f_ext)
{
    float* f_ext_ptr = f_ext->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "setup_forces", num_particles, GPU_LAMBDA_Ex(int i) {
            f_ext_ptr[i * 3 + 0] = 0.0f;
            f_ext_ptr[i * 3 + 1] = 0.0f;
            f_ext_ptr[i * 3 + 2] = mass * gravity;
        });
}

cuda::CUDALinearBufferHandle compute_rest_lengths_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle springs)
{
    size_t num_springs = springs->getDesc().element_count / 2;
    auto rest_lengths_buffer =
        cuda::create_cuda_linear_buffer<float>(num_springs);

    const float* pos_ptr = positions->get_device_ptr<float>();
    const int* springs_ptr = springs->get_device_ptr<int>();
    float* rest_ptr = rest_lengths_buffer->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "compute_rest_lengths", num_springs, GPU_LAMBDA_Ex(int s) {
            int i = springs_ptr[s * 2];
            int j = springs_ptr[s * 2 + 1];

            float dx = pos_ptr[i * 3] - pos_ptr[j * 3];
            float dy = pos_ptr[i * 3 + 1] - pos_ptr[j * 3 + 1];
            float dz = pos_ptr[i * 3 + 2] - pos_ptr[j * 3 + 2];

            rest_ptr[s] = sqrtf(dx * dx + dy * dy + dz * dz);
        });

    return rest_lengths_buffer;
}

// Gradient kernel using adjacency list
__global__ void compute_gradient_kernel_adjacency(
    const float* x_curr,
    const float* x_tilde,
    const float* M_diag,
    const float* f_ext,
    const int* adjacent_vertices,
    const int* vertex_offsets,
    const float* rest_lengths,
    float stiffness,
    float dt,
    int num_particles,
    float* grad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles)
        return;

    // Initialize with inertial term: M * (x - x_tilde)
    grad[i * 3 + 0] =
        M_diag[i * 3 + 0] * (x_curr[i * 3 + 0] - x_tilde[i * 3 + 0]);
    grad[i * 3 + 1] =
        M_diag[i * 3 + 1] * (x_curr[i * 3 + 1] - x_tilde[i * 3 + 1]);
    grad[i * 3 + 2] =
        M_diag[i * 3 + 2] * (x_curr[i * 3 + 2] - x_tilde[i * 3 + 2]);

    // Add spring forces - iterate over adjacent vertices
    int start = vertex_offsets[i];
    int end = vertex_offsets[i + 1];

    for (int idx = start; idx < end; ++idx) {
        int j = adjacent_vertices[idx];
        float l0 = rest_lengths[idx];
        float l0_sq = l0 * l0;

        float dx = x_curr[i * 3 + 0] - x_curr[j * 3 + 0];
        float dy = x_curr[i * 3 + 1] - x_curr[j * 3 + 1];
        float dz = x_curr[i * 3 + 2] - x_curr[j * 3 + 2];
        float diff_sq = dx * dx + dy * dy + dz * dz;

        float factor = 2.0f * stiffness * (diff_sq / l0_sq - 1.0f) * dt * dt;

        // Spring force on vertex i
        grad[i * 3 + 0] += factor * dx;
        grad[i * 3 + 1] += factor * dy;
        grad[i * 3 + 2] += factor * dz;
    }

    // Subtract external forces
    grad[i * 3 + 0] -= dt * dt * f_ext[i * 3 + 0];
    grad[i * 3 + 1] -= dt * dt * f_ext[i * 3 + 1];
    grad[i * 3 + 2] -= dt * dt * f_ext[i * 3 + 2];
}

void compute_gradient_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle adjacent_vertices,
    cuda::CUDALinearBufferHandle vertex_offsets,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle grad)
{
    int block_size = 256;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    compute_gradient_kernel_adjacency<<<num_blocks, block_size>>>(
        x_curr->get_device_ptr<float>(),
        x_tilde->get_device_ptr<float>(),
        M_diag->get_device_ptr<float>(),
        f_ext->get_device_ptr<float>(),
        adjacent_vertices->get_device_ptr<int>(),
        vertex_offsets->get_device_ptr<int>(),
        rest_lengths->get_device_ptr<float>(),
        stiffness,
        dt,
        num_particles,
        grad->get_device_ptr<float>());

    cudaDeviceSynchronize();
}

// Custom 3x3 symmetric eigenvalue decomposition using Jacobi rotation
// This is needed because Eigen's SelfAdjointEigenSolver doesn't work on CUDA
// device
__device__ void eigen_decomposition_3x3(
    const Eigen::Matrix3f& A,
    Eigen::Vector3f& eigenvalues,
    Eigen::Matrix3f& eigenvectors)
{
    const int MAX_ITER = 50;
    const float EPSILON = 1e-10f;

    // Initialize eigenvectors as identity
    eigenvectors.setIdentity();

    // Copy A to working matrix
    Eigen::Matrix3f a = A;

    // Jacobi rotation
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Find largest off-diagonal element
        float max_offdiag = 0.0f;
        int p = 0, q = 1;

        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                float abs_aij = fabsf(a(i, j));
                if (abs_aij > max_offdiag) {
                    max_offdiag = abs_aij;
                    p = i;
                    q = j;
                }
            }
        }

        // Check convergence
        if (max_offdiag < EPSILON) {
            break;
        }

        // Compute rotation angle
        float diff = a(q, q) - a(p, p);
        float theta = 0.5f * atan2f(2.0f * a(p, q), diff);
        float c = cosf(theta);
        float s = sinf(theta);

        // Apply rotation to a: a = J^T * a * J
        Eigen::Matrix3f J;
        J.setIdentity();
        J(p, p) = c;
        J(q, q) = c;
        J(p, q) = s;
        J(q, p) = -s;

        a = J.transpose() * a * J;

        // Accumulate eigenvectors
        eigenvectors = eigenvectors * J;
    }

    // Extract eigenvalues from diagonal
    eigenvalues(0) = a(0, 0);
    eigenvalues(1) = a(1, 1);
    eigenvalues(2) = a(2, 2);
}

// Custom PSD projection for 3x3 symmetric matrix
__device__ Eigen::Matrix3f project_psd_custom(const Eigen::Matrix3f& H)
{
    Eigen::Vector3f eigenvalues;
    Eigen::Matrix3f eigenvectors;

    // Compute eigendecomposition
    eigen_decomposition_3x3(H, eigenvalues, eigenvectors);

    // Clamp negative eigenvalues to zero
    for (int i = 0; i < 3; i++) {
        if (eigenvalues(i) < 0.0f) {
            eigenvalues(i) = 0.0f;
        }
    }

    // Reconstruct: H_psd = V * Lambda * V^T
    Eigen::Matrix3f result =
        eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

    return result;
}

// ============================================================================
// NEW: Zero-sort direct-fill implementation
// ============================================================================

// Kernel to count total edges (for matrix structure)
__global__ void count_total_edges_kernel(
    const int* adjacent_vertices,
    const int* vertex_offsets,
    int num_particles,
    int* total_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles)
        return;

    int start = vertex_offsets[i];
    int end = vertex_offsets[i + 1];
    int count = 0;
    for (int idx = start; idx < end; idx++) {
        int j = adjacent_vertices[idx];
        if (j > i)
            count++;
    }
    atomicAdd(total_count, count);
}

// Kernel to generate matrix entries - parallelized per vertex+neighbor
__global__ void generate_matrix_entries_kernel(
    const int* adjacent_vertices,
    const int* vertex_offsets,
    int num_particles,
    int* rows,
    int* cols,
    int* write_offset)  // Atomic counter for writing position
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles)
        return;

    int n = num_particles * 3;

    // First, write mass diagonal entries
    for (int d = 0; d < 3; d++) {
        int dof = i * 3 + d;
        int pos = atomicAdd(write_offset, 1);
        rows[pos] = dof;
        cols[pos] = dof;
    }

    // Then, write spring entries for neighbors with j > i
    int start = vertex_offsets[i];
    int end = vertex_offsets[i + 1];

    for (int idx = start; idx < end; idx++) {
        int j = adjacent_vertices[idx];
        if (j <= i)
            continue;  // Only process j > i to avoid duplicates

        // Generate 36 entries for this edge (vi, vj)
        for (int block_r = 0; block_r < 2; block_r++) {
            for (int block_c = 0; block_c < 2; block_c++) {
                int base_r = (block_r == 0 ? i : j) * 3;
                int base_c = (block_c == 0 ? i : j) * 3;

                for (int local_r = 0; local_r < 3; local_r++) {
                    for (int local_c = 0; local_c < 3; local_c++) {
                        int pos = atomicAdd(write_offset, 1);
                        rows[pos] = base_r + local_r;
                        cols[pos] = base_c + local_c;
                    }
                }
            }
        }
    }
}

// Binary search helper for finding (row, col) in sorted unique arrays
__device__ int find_entry_position(
    const int* unique_rows,
    const int* unique_cols,
    int nnz,
    int target_row,
    int target_col)
{
    int left = 0;
    int right = nnz - 1;

    while (left <= right) {
        int mid = (left + right) / 2;
        int mid_row = unique_rows[mid];
        int mid_col = unique_cols[mid];

        if (mid_row == target_row && mid_col == target_col) {
            return mid;
        }

        // Compare as (row, col) pairs
        if (mid_row < target_row ||
            (mid_row == target_row && mid_col < target_col)) {
            left = mid + 1;
        }
        else {
            right = mid - 1;
        }
    }

    return -1;  // Not found (should never happen for valid structure)
}

// Kernel to build mass diagonal positions using binary search
__global__ void build_mass_positions_kernel(
    const int* unique_rows,
    const int* unique_cols,
    int nnz,
    int num_dofs,
    int* mass_positions)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_dofs)
        return;

    // Mass diagonal: find position of (tid, tid)
    mass_positions[tid] =
        find_entry_position(unique_rows, unique_cols, nnz, tid, tid);
}

// Kernel to build spring positions using vertex iteration and binary search
// For each adjacency index, store 36 positions if j > i, otherwise store -1
__global__ void build_spring_positions_kernel(
    const int* unique_rows,
    const int* unique_cols,
    const int* adjacent_vertices,
    const int* vertex_offsets,
    int nnz,
    int num_particles,
    int total_adjacencies,
    int* spring_positions)  // Output: [total_adjacencies * 36]
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles)
        return;

    int start = vertex_offsets[i];
    int end = vertex_offsets[i + 1];

    for (int idx = start; idx < end; idx++) {
        int j = adjacent_vertices[idx];
        int base_out = idx * 36;

        if (j <= i) {
            // Mark as unused
            for (int k = 0; k < 36; k++) {
                spring_positions[base_out + k] = -1;
            }
            continue;
        }

        // Store 36 positions for this edge
        int count = 0;

        // Same order as generation: (i,i), (i,j), (j,i), (j,j)
        for (int block_r = 0; block_r < 2; block_r++) {
            for (int block_c = 0; block_c < 2; block_c++) {
                int vi = (block_r == 0 ? i : j);
                int vj = (block_c == 0 ? i : j);

                for (int local_r = 0; local_r < 3; local_r++) {
                    for (int local_c = 0; local_c < 3; local_c++) {
                        int row = vi * 3 + local_r;
                        int col = vj * 3 + local_c;
                        int pos = find_entry_position(
                            unique_rows, unique_cols, nnz, row, col);
                        spring_positions[base_out + count] = pos;
                        count++;
                    }
                }
            }
        }
    }
}

// Build CSR sparsity pattern once during initialization (GPU version)
CSRStructure build_hessian_structure_gpu(
    cuda::CUDALinearBufferHandle adjacent_vertices,
    cuda::CUDALinearBufferHandle vertex_offsets,
    int num_particles)
{
    CSRStructure structure;
    int n = num_particles * 3;
    int block_size = 256;

    // Count total edges (j > i) across all vertices
    auto d_total_edges = cuda::create_cuda_linear_buffer<int>(1);
    cudaMemset(d_total_edges->get_device_ptr<int>(), 0, sizeof(int));

    int count_blocks = (num_particles + block_size - 1) / block_size;
    count_total_edges_kernel<<<count_blocks, block_size>>>(
        adjacent_vertices->get_device_ptr<int>(),
        vertex_offsets->get_device_ptr<int>(),
        num_particles,
        d_total_edges->get_device_ptr<int>());
    cudaDeviceSynchronize();

    int num_edges;
    cudaMemcpy(
        &num_edges,
        d_total_edges->get_device_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost);

    int num_mass_entries = n;
    int num_spring_entries = num_edges * 36;
    int total_entries = num_mass_entries + num_spring_entries;

    // Allocate temporary buffers for all entries
    auto d_all_rows = cuda::create_cuda_linear_buffer<int>(total_entries);
    auto d_all_cols = cuda::create_cuda_linear_buffer<int>(total_entries);
    auto d_write_offset = cuda::create_cuda_linear_buffer<int>(1);
    cudaMemset(d_write_offset->get_device_ptr<int>(), 0, sizeof(int));

    // Generate all (row, col) pairs on GPU
    int num_blocks = (num_particles + block_size - 1) / block_size;

    generate_matrix_entries_kernel<<<num_blocks, block_size>>>(
        adjacent_vertices->get_device_ptr<int>(),
        vertex_offsets->get_device_ptr<int>(),
        num_particles,
        d_all_rows->get_device_ptr<int>(),
        d_all_cols->get_device_ptr<int>(),
        d_write_offset->get_device_ptr<int>());

    cudaDeviceSynchronize();

    // Sort by (row, col) using thrust
    thrust::device_ptr<int> rows_ptr(d_all_rows->get_device_ptr<int>());
    thrust::device_ptr<int> cols_ptr(d_all_cols->get_device_ptr<int>());

    auto zip_begin =
        thrust::make_zip_iterator(thrust::make_tuple(rows_ptr, cols_ptr));

    thrust::sort(
        thrust::device,
        zip_begin,
        zip_begin + total_entries,
        [] __device__(
            const thrust::tuple<int, int>& a,
            const thrust::tuple<int, int>& b) {
            if (thrust::get<0>(a) != thrust::get<0>(b))
                return thrust::get<0>(a) < thrust::get<0>(b);
            return thrust::get<1>(a) < thrust::get<1>(b);
        });

    cudaDeviceSynchronize();

    // Remove duplicates
    auto new_end = thrust::unique(
        thrust::device,
        zip_begin,
        zip_begin + total_entries,
        [] __device__(
            const thrust::tuple<int, int>& a,
            const thrust::tuple<int, int>& b) {
            return thrust::get<0>(a) == thrust::get<0>(b) &&
                   thrust::get<1>(a) == thrust::get<1>(b);
        });

    int nnz = new_end - zip_begin;

    cudaDeviceSynchronize();

    // Get total adjacencies count
    int total_adjacencies;
    cudaMemcpy(
        &total_adjacencies,
        vertex_offsets->get_device_ptr<int>() + num_particles,
        sizeof(int),
        cudaMemcpyDeviceToHost);

    // Allocate CSR arrays
    structure.col_indices = cuda::create_cuda_linear_buffer<int>(nnz);
    structure.row_offsets = cuda::create_cuda_linear_buffer<int>(n + 1);
    structure.mass_value_positions = cuda::create_cuda_linear_buffer<int>(n);
    structure.spring_value_positions =
        cuda::create_cuda_linear_buffer<int>(total_adjacencies * 36);

    // Copy unique columns
    cudaMemcpy(
        structure.col_indices->get_device_ptr<int>(),
        d_all_cols->get_device_ptr<int>(),
        nnz * sizeof(int),
        cudaMemcpyDeviceToDevice);

    // Build row_offsets using histogram
    thrust::device_ptr<int> row_offsets_ptr(
        structure.row_offsets->get_device_ptr<int>());
    thrust::fill(thrust::device, row_offsets_ptr, row_offsets_ptr + n + 1, 0);

    int* row_offsets_raw = structure.row_offsets->get_device_ptr<int>();
    int* unique_rows_raw = d_all_rows->get_device_ptr<int>();

    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nnz),
        [row_offsets_raw, unique_rows_raw] __device__(int idx) {
            int row = unique_rows_raw[idx];
            atomicAdd(&row_offsets_raw[row], 1);
        });

    cudaDeviceSynchronize();

    // Compute prefix sum for row offsets
    thrust::exclusive_scan(
        thrust::device,
        row_offsets_ptr,
        row_offsets_ptr + n + 1,
        row_offsets_ptr);

    cudaDeviceSynchronize();

    // Build position mappings on GPU using binary search
    int mass_blocks = (n + block_size - 1) / block_size;
    build_mass_positions_kernel<<<mass_blocks, block_size>>>(
        d_all_rows->get_device_ptr<int>(),
        d_all_cols->get_device_ptr<int>(),
        nnz,
        n,
        structure.mass_value_positions->get_device_ptr<int>());

    int vertex_blocks = (num_particles + block_size - 1) / block_size;
    build_spring_positions_kernel<<<vertex_blocks, block_size>>>(
        d_all_rows->get_device_ptr<int>(),
        d_all_cols->get_device_ptr<int>(),
        adjacent_vertices->get_device_ptr<int>(),
        vertex_offsets->get_device_ptr<int>(),
        nnz,
        num_particles,
        total_adjacencies,
        structure.spring_value_positions->get_device_ptr<int>());

    cudaDeviceSynchronize();

    structure.num_rows = n;
    structure.num_cols = n;
    structure.nnz = nnz;

    return structure;
}

// Kernel: Directly fill spring Hessian values into CSR using vertex iteration
__global__ void fill_spring_hessian_values_kernel(
    const float* x_curr,
    const int* adjacent_vertices,
    const int* vertex_offsets,
    const float* rest_lengths,
    const int*
        value_positions,  // Pre-computed positions: [total_adjacencies * 36]
    float stiffness,
    float dt,
    int num_particles,
    float* values)  // Output CSR values array
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles)
        return;

    int start = vertex_offsets[i];
    int end = vertex_offsets[i + 1];

    for (int idx = start; idx < end; idx++) {
        int j = adjacent_vertices[idx];
        if (j <= i)
            continue;  // Only process j > i

        float l0 = rest_lengths[idx];
        if (l0 < 1e-10f)
            continue;

        float k = stiffness;
        float l0_sq = l0 * l0;

        // Get positions
        Eigen::Vector3f xi(x_curr[i * 3], x_curr[i * 3 + 1], x_curr[i * 3 + 2]);
        Eigen::Vector3f xj(x_curr[j * 3], x_curr[j * 3 + 1], x_curr[j * 3 + 2]);
        Eigen::Vector3f diff = xi - xj;
        float diff_sq = diff.squaredNorm();

        // H_diff = 2*k/l0^2 * (2*outer(diff,diff) + (diff_sq - l0^2)*I)
        Eigen::Matrix3f outer = diff * diff.transpose();
        Eigen::Matrix3f H_diff =
            2.0f * k / l0_sq *
            (2.0f * outer + (diff_sq - l0_sq) * Eigen::Matrix3f::Identity());

        // PSD projection
        H_diff = project_psd_custom(H_diff);

        // Scale by dt^2
        float scale = dt * dt;
        H_diff *= scale;

        // Use adjacency index directly for position lookup
        int base_pos = idx * 36;
        int count = 0;

        // 4 blocks: (i,i), (i,j), (j,i), (j,j)
        for (int block_r = 0; block_r < 2; block_r++) {
            for (int block_c = 0; block_c < 2; block_c++) {
                float sign_row = (block_r == 0) ? 1.0f : -1.0f;
                float sign_col = (block_c == 0) ? 1.0f : -1.0f;

                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        float val = H_diff(r, c) * sign_row * sign_col;
                        int pos = value_positions[base_pos + count++];
                        atomicAdd(&values[pos], val);
                    }
                }
            }
        }
    }
}

// Fast update: directly fill values (NO SORTING!)
void update_hessian_values_gpu(
    const CSRStructure& csr_structure,
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle adjacent_vertices,
    cuda::CUDALinearBufferHandle vertex_offsets,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle values)
{
    int num_dofs = num_particles * 3;
    int block_size = 256;

    // Zero out values array
    cudaMemset(
        values->get_device_ptr<float>(), 0, csr_structure.nnz * sizeof(float));

    // Fill mass diagonal
    const float* M_diag_ptr = M_diag->get_device_ptr<float>();
    const int* mass_positions =
        csr_structure.mass_value_positions->get_device_ptr<int>();
    float* values_ptr = values->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "fill_mass_diagonal", num_dofs, GPU_LAMBDA_Ex(int i) {
            float regularization = 1e-6f;
            int pos = mass_positions[i];
            values_ptr[pos] = M_diag_ptr[i] + regularization;
        });

    // Fill spring contributions - use vertex iteration
    int vertex_blocks = (num_particles + block_size - 1) / block_size;
    fill_spring_hessian_values_kernel<<<vertex_blocks, block_size>>>(
        x_curr->get_device_ptr<float>(),
        adjacent_vertices->get_device_ptr<int>(),
        vertex_offsets->get_device_ptr<int>(),
        rest_lengths->get_device_ptr<float>(),
        csr_structure.spring_value_positions->get_device_ptr<int>(),
        stiffness,
        dt,
        num_particles,
        values->get_device_ptr<float>());

    cudaDeviceSynchronize();
}

// ============================================================================
// End of zero-sort implementation
// ============================================================================

// Kernel to compute spring energy on GPU using adjacency list
__global__ void compute_spring_energy_kernel(
    const float* x_curr,
    const int* adjacent_vertices,
    const int* vertex_offsets,
    const float* rest_lengths,
    float stiffness,
    int num_particles,
    float* spring_energies_per_vertex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles)
        return;

    int start = vertex_offsets[i];
    int end = vertex_offsets[i + 1];

    float xi[3] = { x_curr[i * 3], x_curr[i * 3 + 1], x_curr[i * 3 + 2] };

    float total_energy = 0.0f;

    for (int idx = start; idx < end; idx++) {
        int j = adjacent_vertices[idx];

        // Only compute energy for j > i to avoid double counting
        if (j <= i)
            continue;

        float l0 = rest_lengths[idx];
        float l0_sq = l0 * l0;

        // Get neighbor position
        float xj[3] = { x_curr[j * 3], x_curr[j * 3 + 1], x_curr[j * 3 + 2] };

        // Compute squared distance
        float diff[3] = { xi[0] - xj[0], xi[1] - xj[1], xi[2] - xj[2] };
        float diff_sq =
            diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];

        // Spring energy matching gradient: 0.5 * k * l0^2 * ((diff_sq / l0^2) -
        // 1)^2
        float ratio = diff_sq / l0_sq - 1.0f;
        float energy = 0.5f * stiffness * l0_sq * ratio * ratio;

        total_energy += energy;
    }

    // Store per-vertex total energy
    spring_energies_per_vertex[i] = total_energy;
}

// Compute total energy: E = 0.5 * M * ||x - x_tilde||^2 + spring_energy -
// f_ext^T * x
float compute_energy_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle d_adjacent_vertices,
    cuda::CUDALinearBufferHandle d_vertex_offsets,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle d_inertial_terms,
    cuda::CUDALinearBufferHandle d_spring_energies,
    cuda::CUDALinearBufferHandle d_potential_terms)
{
    int n = num_particles * 3;

    float* x_ptr = reinterpret_cast<float*>(x_curr->get_device_ptr());
    float* x_tilde_ptr = reinterpret_cast<float*>(x_tilde->get_device_ptr());
    float* M_ptr = reinterpret_cast<float*>(M_diag->get_device_ptr());
    float* f_ptr = reinterpret_cast<float*>(f_ext->get_device_ptr());

    // Use pre-allocated buffers instead of creating new ones
    float* inertial_ptr =
        reinterpret_cast<float*>(d_inertial_terms->get_device_ptr());

    cuda::GPUParallelFor(
        "compute_inertial_energy", n, GPU_LAMBDA_Ex(int i) {
            float diff = x_ptr[i] - x_tilde_ptr[i];
            inertial_ptr[i] = 0.5f * M_ptr[i] * diff * diff;
        });

    // Sum inertial energy
    thrust::device_ptr<float> d_inertial_thrust(inertial_ptr);
    float E_inertial =
        thrust::reduce(d_inertial_thrust, d_inertial_thrust + n, 0.0f);

    // Use pre-allocated buffer for spring energies (per-vertex)
    float* spring_energy_ptr =
        reinterpret_cast<float*>(d_spring_energies->get_device_ptr());

    // Zero out spring energies buffer (size = num_particles)
    cudaMemset(spring_energy_ptr, 0, num_particles * sizeof(float));

    int block_size = 256;
    int num_blocks = (num_particles + block_size - 1) / block_size;
    compute_spring_energy_kernel<<<num_blocks, block_size>>>(
        x_ptr,
        reinterpret_cast<const int*>(d_adjacent_vertices->get_device_ptr()),
        reinterpret_cast<const int*>(d_vertex_offsets->get_device_ptr()),
        reinterpret_cast<const float*>(rest_lengths->get_device_ptr()),
        stiffness,
        num_particles,
        spring_energy_ptr);

    cudaDeviceSynchronize();

    // Sum spring energy (over num_particles, not total_adjacencies)
    thrust::device_ptr<float> d_spring_thrust(spring_energy_ptr);
    float E_spring =
        thrust::reduce(d_spring_thrust, d_spring_thrust + num_particles, 0.0f);

    // Use pre-allocated buffer for potential energy
    float* potential_ptr =
        reinterpret_cast<float*>(d_potential_terms->get_device_ptr());

    cuda::GPUParallelFor(
        "compute_potential_energy", n, GPU_LAMBDA_Ex(int i) {
            potential_ptr[i] = -f_ptr[i] * x_ptr[i] * dt * dt;
        });

    thrust::device_ptr<float> d_potential_thrust(potential_ptr);
    float E_potential =
        thrust::reduce(d_potential_thrust, d_potential_thrust + n, 0.0f);

    float total_energy = E_inertial + dt * dt * E_spring + E_potential;

    return total_energy;
}

// Functors for thrust operations (must be defined outside functions for CUDA
// compatibility)
struct square_op {
    __device__ float operator()(float x) const
    {
        return x * x;
    }
};

// GPU vector operations to avoid CPU-GPU transfers
float compute_vector_norm_gpu(cuda::CUDALinearBufferHandle vec, int size)
{
    float* vec_ptr = reinterpret_cast<float*>(vec->get_device_ptr());
    thrust::device_ptr<float> d_vec(vec_ptr);

    // Compute sum of squares using functor
    float sum_sq = thrust::transform_reduce(
        thrust::device,
        d_vec,
        d_vec + size,
        square_op(),
        0.0f,
        thrust::plus<float>());

    return sqrtf(sum_sq);
}

float compute_dot_product_gpu(
    cuda::CUDALinearBufferHandle vec1,
    cuda::CUDALinearBufferHandle vec2,
    int size)
{
    float* vec1_ptr = reinterpret_cast<float*>(vec1->get_device_ptr());
    float* vec2_ptr = reinterpret_cast<float*>(vec2->get_device_ptr());

    thrust::device_ptr<float> d_vec1(vec1_ptr);
    thrust::device_ptr<float> d_vec2(vec2_ptr);

    return thrust::inner_product(
        thrust::device, d_vec1, d_vec1 + size, d_vec2, 0.0f);
}

void axpy_gpu(
    float alpha,
    cuda::CUDALinearBufferHandle x,
    cuda::CUDALinearBufferHandle y,
    cuda::CUDALinearBufferHandle result,
    int size)
{
    const float* x_ptr = x->get_device_ptr<float>();
    const float* y_ptr = y->get_device_ptr<float>();
    float* result_ptr = result->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "axpy", size, GPU_LAMBDA_Ex(int i) {
            result_ptr[i] = y_ptr[i] + alpha * x_ptr[i];
        });
}

void negate_gpu(
    cuda::CUDALinearBufferHandle in,
    cuda::CUDALinearBufferHandle out,
    int size)
{
    const float* in_ptr = in->get_device_ptr<float>();
    float* out_ptr = out->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "negate", size, GPU_LAMBDA_Ex(int i) { out_ptr[i] = -in_ptr[i]; });
}

void project_to_ground_gpu(
    cuda::CUDALinearBufferHandle positions,
    int num_particles,
    float ground_height)
{
    float* pos_ptr = positions->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "project_to_ground", num_particles, GPU_LAMBDA_Ex(int i) {
            if (pos_ptr[i * 3 + 2] < ground_height) {
                pos_ptr[i * 3 + 2] = ground_height;
            }
        });
}

// Kernel to compute face normals with precomputed face_offsets
__global__ void compute_normals_kernel(
    const float* positions,
    const int* face_vertex_indices,
    const int* face_counts,
    const int* face_offsets,  // Precomputed prefix sum
    int num_faces,
    bool flip_normal,
    float* normals)
{
    int face_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_id >= num_faces)
        return;

    int face_start = face_offsets[face_id];
    int face_count = face_counts[face_id];
    if (face_count < 3)
        return;

    // Get first 3 vertices of the face
    int i0 = face_vertex_indices[face_start];
    int i1 = face_vertex_indices[face_start + 1];
    int i2 = face_vertex_indices[face_start + 2];

    // Compute edges
    float e1x = positions[i1 * 3] - positions[i0 * 3];
    float e1y = positions[i1 * 3 + 1] - positions[i0 * 3 + 1];
    float e1z = positions[i1 * 3 + 2] - positions[i0 * 3 + 2];

    float e2x = positions[i2 * 3] - positions[i0 * 3];
    float e2y = positions[i2 * 3 + 1] - positions[i0 * 3 + 1];
    float e2z = positions[i2 * 3 + 2] - positions[i0 * 3 + 2];

    // Compute cross product
    float nx, ny, nz;
    if (flip_normal) {
        nx = e1y * e2z - e1z * e2y;
        ny = e1z * e2x - e1x * e2z;
        nz = e1x * e2y - e1y * e2x;
    }
    else {
        nx = e2y * e1z - e2z * e1y;
        ny = e2z * e1x - e2x * e1z;
        nz = e2x * e1y - e2y * e1x;
    }

    // Normalize
    float length = sqrtf(nx * nx + ny * ny + nz * nz);
    if (length > 1e-8f) {
        nx /= length;
        ny /= length;
        nz /= length;
    }
    else {
        nx = 0.0f;
        ny = 0.0f;
        nz = 1.0f;
    }

    // Write normal for all vertices in this face
    for (int v = 0; v < face_count; v++) {
        int out_idx = (face_start + v) * 3;
        normals[out_idx] = nx;
        normals[out_idx + 1] = ny;
        normals[out_idx + 2] = nz;
    }
}

void compute_normals_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle face_vertex_indices,
    cuda::CUDALinearBufferHandle face_counts,
    bool flip_normal,
    cuda::CUDALinearBufferHandle normals)
{
    int num_faces = face_counts->getDesc().element_count;

    // Precompute face offsets using prefix sum (exclusive scan)
    thrust::device_vector<int> face_offsets(num_faces);
    const int* face_counts_ptr = face_counts->get_device_ptr<int>();
    thrust::device_ptr<const int> face_counts_thrust(face_counts_ptr);
    thrust::exclusive_scan(
        thrust::device,
        face_counts_thrust,
        face_counts_thrust + num_faces,
        face_offsets.begin());

    int block_size = 256;
    int num_blocks = (num_faces + block_size - 1) / block_size;

    compute_normals_kernel<<<num_blocks, block_size>>>(
        positions->get_device_ptr<float>(),
        face_vertex_indices->get_device_ptr<int>(),
        face_counts->get_device_ptr<int>(),
        thrust::raw_pointer_cast(face_offsets.data()),
        num_faces,
        flip_normal,
        normals->get_device_ptr<float>());

    cudaDeviceSynchronize();
}

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
