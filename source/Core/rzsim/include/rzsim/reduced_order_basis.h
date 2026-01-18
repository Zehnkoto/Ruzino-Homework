#pragma once
#include <rzsim/api.h>

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <memory>
#include <vector>

RUZINO_NAMESPACE_OPEN_SCOPE

class Geometry;

// Structure to store affine transformation for each basis mode
struct RZSIM_API AffineTransform {
    // Each basis mode has 12 DOF: 9 for rotation matrix (3x3) + 3 for
    // translation We store them as a vector: [R00, R01, R02, R10, R11, R12,
    // R20, R21, R22, tx, ty, tz]
    std::vector<std::vector<float>> transforms;  // [num_basis][12]

    AffineTransform() = default;

    // Initialize with identity transforms for num_basis modes
    explicit AffineTransform(int num_basis);

    // Set transformation for a specific mode
    void set_transform(int mode_index, const std::vector<float>& transform);

    // Get transformation for a specific mode
    const std::vector<float>& get_transform(int mode_index) const;

    // Get number of basis modes
    int num_modes() const
    {
        return static_cast<int>(transforms.size());
    }
};

struct RZSIM_API ReducedOrderedBasis {
    // Default constructor
    ReducedOrderedBasis() = default;

    // dimension: 2 for surface mesh (triangles/quads), 3 for volume mesh
    // (tetrahedra) use_libigl: if true, use libigl's cotmatrix; otherwise use
    // custom implementation
    // bc_vertices: vertices with Dirichlet boundary conditions (will be clamped
    // to zero in eigenvectors)
    ReducedOrderedBasis(
        const Geometry& g,
        int num_modes = 10,
        int dimension = 2,
        bool use_libigl = false,
        const std::vector<int>& bc_vertices = {});

    // Compute eigenvalue decomposition and store the first N eigenvectors
    void compute_eigenmodes(int num_modes);

    std::vector<Eigen::VectorXf> basis;
    std::vector<float> eigenvalues;
    Eigen::SparseMatrix<float> laplacian_matrix_;
    std::vector<int> bc_vertices_;  // Vertices with Dirichlet BC

   private:
    void assemble_laplacian_2d(void* mesh);
    void assemble_laplacian_3d(void* mesh);
    void assemble_laplacian_2d_libigl(void* mesh);
    void assemble_laplacian_3d_libigl(void* mesh);
    void apply_bc_to_laplacian();  // Apply BC constraints to Laplacian matrix
};

RUZINO_NAMESPACE_CLOSE_SCOPE