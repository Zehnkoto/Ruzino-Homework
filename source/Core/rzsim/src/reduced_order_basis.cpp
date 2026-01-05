#include <GCore/Components/MeshComponent.h>
#include <GCore/util_openmesh_bind.h>
#include <rzsim/reduced_order_basis.h>
#include <spdlog/spdlog.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <iostream>

RUZINO_NAMESPACE_OPEN_SCOPE

using PolyMesh = OpenMesh::PolyMesh_ArrayKernelT<>;
using VolumeMesh = OpenVolumeMesh::GeometricTetrahedralMeshV3d;

ReducedOrderedBasis::ReducedOrderedBasis(
    const Geometry& g,
    int num_modes,
    int dimension)
{
    // Get mesh component
    auto mesh_comp = g.get_component<MeshComponent>();
    if (!mesh_comp) {
        throw std::runtime_error("Geometry must have MeshComponent");
    }

    auto geom_ptr = const_cast<Geometry*>(&g);

    if (dimension == 3) {
        // 3D case: assemble Laplace operator for tetrahedral mesh
        auto volumemesh = operand_to_openvolumemesh(geom_ptr);
        if (!volumemesh || volumemesh->n_vertices() == 0) {
            throw std::runtime_error(
                "Invalid volume mesh for 3D reduced order basis");
        }
        spdlog::info(
            "Assembling 3D Laplacian for tetrahedral mesh (vertices={}, "
            "cells={})",
            volumemesh->n_vertices(),
            volumemesh->n_cells());
        assemble_laplacian_3d(volumemesh.get());
    }
    else if (dimension == 2) {
        // 2D case: assemble Laplace operator for surface mesh (triangles/quads)
        auto openmesh = operand_to_openmesh(geom_ptr);
        if (!openmesh || openmesh->n_vertices() == 0 ||
            openmesh->n_faces() == 0) {
            throw std::runtime_error(
                "Invalid surface mesh for 2D reduced order basis");
        }
        spdlog::info(
            "Assembling 2D Laplacian for surface mesh (vertices={}, faces={})",
            openmesh->n_vertices(),
            openmesh->n_faces());
        assemble_laplacian_2d(openmesh.get());
    }
    else {
        throw std::runtime_error("Dimension must be 2 (surface) or 3 (volume)");
    }

    // Compute eigenmodes
    compute_eigenmodes(num_modes);
}

void ReducedOrderedBasis::assemble_laplacian_3d(void* mesh_ptr)
{
    auto mesh = static_cast<VolumeMesh*>(mesh_ptr);
    int n_vertices = mesh->n_vertices();

    // Triplets for sparse matrix assembly
    std::vector<Eigen::Triplet<double>> triplets;

    // Iterate over all cells (tetrahedra) and compute cotangent Laplacian
    for (auto c_it = mesh->cells_begin(); c_it != mesh->cells_end(); ++c_it) {
        // Get the four vertices of the tetrahedron
        std::vector<int> vertex_ids;
        std::vector<pxr::GfVec3d> positions;

        for (auto cv_it = mesh->cv_iter(*c_it); cv_it.valid(); ++cv_it) {
            vertex_ids.push_back((*cv_it).idx());
            auto pt = mesh->vertex(*cv_it);
            positions.push_back(pxr::GfVec3d(pt[0], pt[1], pt[2]));
        }

        if (vertex_ids.size() != 4)
            continue;  // Skip non-tetrahedral cells

        // Compute cotangent weights for all 6 edges of the tetrahedron
        // Edge (i,j) has cotangent weight from the two opposite vertices
        for (int e = 0; e < 6; e++) {
            int edge_pairs[6][2] = { { 0, 1 }, { 0, 2 }, { 0, 3 },
                                     { 1, 2 }, { 1, 3 }, { 2, 3 } };
            int i = edge_pairs[e][0];
            int j = edge_pairs[e][1];

            // Find the two vertices not on this edge
            std::vector<int> opposite;
            for (int k = 0; k < 4; k++) {
                if (k != i && k != j) {
                    opposite.push_back(k);
                }
            }

            if (opposite.size() != 2)
                continue;

            // Compute dihedral angle cotangent contribution
            auto vi = positions[i];
            auto vj = positions[j];
            auto vk = positions[opposite[0]];
            auto vl = positions[opposite[1]];

            // Edge vector
            auto eij = vj - vi;
            double eij_len_sq =
                eij[0] * eij[0] + eij[1] * eij[1] + eij[2] * eij[2];
            double eij_len = std::sqrt(eij_len_sq);

            if (eij_len < 1e-12)
                continue;

            // Vectors from edge to opposite vertices
            auto eki = vk - vi;
            auto eli = vl - vi;

            // Cross products to get face normals
            auto nk_x = eij[1] * eki[2] - eij[2] * eki[1];
            auto nk_y = eij[2] * eki[0] - eij[0] * eki[2];
            auto nk_z = eij[0] * eki[1] - eij[1] * eki[0];
            double nk_norm = std::sqrt(nk_x * nk_x + nk_y * nk_y + nk_z * nk_z);

            auto nl_x = eij[1] * eli[2] - eij[2] * eli[1];
            auto nl_y = eij[2] * eli[0] - eij[0] * eli[2];
            auto nl_z = eij[0] * eli[1] - eij[1] * eli[0];
            double nl_norm = std::sqrt(nl_x * nl_x + nl_y * nl_y + nl_z * nl_z);

            if (nk_norm < 1e-12 || nl_norm < 1e-12)
                continue;

            // Dihedral angle cosine
            double cos_theta =
                (nk_x * nl_x + nk_y * nl_y + nk_z * nl_z) / (nk_norm * nl_norm);
            double sin_theta =
                std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));

            if (sin_theta < 1e-12)
                continue;

            double cot_theta = cos_theta / sin_theta;
            double weight = cot_theta * eij_len / 6.0;

            int vi_idx = vertex_ids[i];
            int vj_idx = vertex_ids[j];

            triplets.emplace_back(vi_idx, vj_idx, weight);
            triplets.emplace_back(vj_idx, vi_idx, weight);
            triplets.emplace_back(vi_idx, vi_idx, -weight);
            triplets.emplace_back(vj_idx, vj_idx, -weight);
        }
    }

    // Assemble global Laplacian matrix
    Eigen::SparseMatrix<double> laplacian_double(n_vertices, n_vertices);
    laplacian_double.setFromTriplets(triplets.begin(), triplets.end());
    laplacian_double.makeCompressed();

    // Negate to get positive semi-definite matrix
    laplacian_matrix_ = (-laplacian_double).cast<float>();

    std::cout << "Assembled 3D cotangent Laplacian matrix: " << n_vertices
              << " x " << n_vertices << " with " << laplacian_matrix_.nonZeros()
              << " non-zeros" << std::endl;
}

void ReducedOrderedBasis::compute_eigenmodes(int num_modes)
{
    int n = laplacian_matrix_.rows();

    if (num_modes > n) {
        std::cout << "Warning: Requested " << num_modes
                  << " modes but matrix only has " << n << " dimensions. Using "
                  << n << " modes instead." << std::endl;
        num_modes = n;
    }

    std::cout << "Computing " << num_modes
              << " eigenmodes using sparse solver (Spectra)..." << std::endl;

    // Convert to double precision sparse matrix for Spectra
    Eigen::SparseMatrix<double> laplacian_double = laplacian_matrix_.cast<double>();

    // Use Spectra to compute the smallest eigenvalues
    // We want the smallest eigenvalues in magnitude (closest to zero)
    Spectra::SparseSymMatProd<double> op(laplacian_double);
    
    // Request more eigenvalues than needed for better convergence
    int ncv = std::min(std::max(2 * num_modes + 1, 20), n);
    
    Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(op, num_modes, ncv);
    
    // Initialize and compute
    // Use SmallestAlge to get smallest eigenvalues in algebraic order (ascending)
    eigs.init();
    int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 4000, 1e-12);
    
    if (eigs.info() != Spectra::CompInfo::Successful) {
        std::cerr << "Spectra eigenvalue computation failed, falling back to dense solver..." << std::endl;
        
        // Fallback to dense solver
        Eigen::MatrixXd dense_laplacian = laplacian_double;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(dense_laplacian);
        
        if (eigensolver.info() != Eigen::Success) {
            throw std::runtime_error("Both sparse and dense eigenvalue decomposition failed");
        }
        
        Eigen::VectorXd all_eigenvalues = eigensolver.eigenvalues();
        Eigen::MatrixXd all_eigenvectors = eigensolver.eigenvectors();
        
        basis.clear();
        eigenvalues.clear();
        
        int actual_modes = std::min(num_modes, static_cast<int>(all_eigenvalues.size()));
        for (int i = 0; i < actual_modes; i++) {
            eigenvalues.push_back(static_cast<float>(all_eigenvalues(i)));
            basis.push_back(all_eigenvectors.col(i).cast<float>());
            std::cout << "  Mode " << i << ": eigenvalue = " << all_eigenvalues(i) << std::endl;
        }
    }
    else {
        // Get eigenvalues and eigenvectors from Spectra
        Eigen::VectorXd eigenvalues_d = eigs.eigenvalues();
        Eigen::MatrixXd eigenvectors_d = eigs.eigenvectors();
        
        // Sort eigenvalues and eigenvectors in ascending order
        std::vector<std::pair<double, int>> sorted_indices;
        int actual_modes = std::min(nconv, num_modes);
        for (int i = 0; i < actual_modes; i++) {
            sorted_indices.push_back({eigenvalues_d(i), i});
        }
        std::sort(sorted_indices.begin(), sorted_indices.end());
        
        basis.clear();
        eigenvalues.clear();
        
        for (int i = 0; i < actual_modes; i++) {
            int original_index = sorted_indices[i].second;
            double eigenvalue = sorted_indices[i].first;
            eigenvalues.push_back(static_cast<float>(eigenvalue));
            basis.push_back(eigenvectors_d.col(original_index).cast<float>());
            std::cout << "  Mode " << i << ": eigenvalue = " << eigenvalue << std::endl;
        }
    }

    std::cout << "Eigenmode computation complete. Stored " << basis.size()
              << " basis vectors." << std::endl;
}

void ReducedOrderedBasis::assemble_laplacian_2d(void* mesh_ptr)
{
    auto mesh = static_cast<PolyMesh*>(mesh_ptr);
    int n_vertices = mesh->n_vertices();

    // Triplets for sparse matrix assembly
    std::vector<Eigen::Triplet<double>> triplets;

    // Iterate over all faces and compute cotangent Laplacian
    for (auto f_it : mesh->faces()) {
        // Get vertices of the face
        std::vector<int> vertex_ids;
        std::vector<pxr::GfVec3d> positions;

        for (auto fv_it : mesh->fv_range(f_it)) {
            vertex_ids.push_back(fv_it.idx());
            auto pt = mesh->point(fv_it);
            positions.push_back(pxr::GfVec3d(pt[0], pt[1], pt[2]));
        }

        int num_verts = vertex_ids.size();

        // Process triangles and quads (triangulate quads)
        if (num_verts == 3) {
            // Triangle - compute cotangent weights
            for (int i = 0; i < 3; i++) {
                int v0 = vertex_ids[i];
                int v1 = vertex_ids[(i + 1) % 3];
                int v2 = vertex_ids[(i + 2) % 3];

                auto p0 = positions[i];
                auto p1 = positions[(i + 1) % 3];
                auto p2 = positions[(i + 2) % 3];

                // Edge vectors
                auto e1 = p1 - p0;
                auto e2 = p2 - p0;

                // Compute cotangent of angle at v0
                double dot_prod = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
                double cross_x = e1[1] * e2[2] - e1[2] * e2[1];
                double cross_y = e1[2] * e2[0] - e1[0] * e2[2];
                double cross_z = e1[0] * e2[1] - e1[1] * e2[0];
                double cross_norm = std::sqrt(
                    cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);

                if (cross_norm > 1e-12) {
                    double cot = dot_prod / cross_norm;

                    // Add cotangent weight to edge (v1, v2)
                    triplets.emplace_back(v1, v2, cot);
                    triplets.emplace_back(v2, v1, cot);
                    triplets.emplace_back(v1, v1, -cot);
                    triplets.emplace_back(v2, v2, -cot);
                }
            }
        }
        else if (num_verts == 4) {
            // Quad - split into two triangles and process each
            // Triangle 1: v0, v1, v2
            for (int tri = 0; tri < 2; tri++) {
                std::vector<int> tri_verts;
                std::vector<pxr::GfVec3d> tri_pos;

                if (tri == 0) {
                    tri_verts = { vertex_ids[0], vertex_ids[1], vertex_ids[2] };
                    tri_pos = { positions[0], positions[1], positions[2] };
                }
                else {
                    tri_verts = { vertex_ids[0], vertex_ids[2], vertex_ids[3] };
                    tri_pos = { positions[0], positions[2], positions[3] };
                }

                for (int i = 0; i < 3; i++) {
                    int v0 = tri_verts[i];
                    int v1 = tri_verts[(i + 1) % 3];
                    int v2 = tri_verts[(i + 2) % 3];

                    auto p0 = tri_pos[i];
                    auto p1 = tri_pos[(i + 1) % 3];
                    auto p2 = tri_pos[(i + 2) % 3];

                    auto e1 = p1 - p0;
                    auto e2 = p2 - p0;

                    double dot_prod =
                        e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
                    double cross_x = e1[1] * e2[2] - e1[2] * e2[1];
                    double cross_y = e1[2] * e2[0] - e1[0] * e2[2];
                    double cross_z = e1[0] * e2[1] - e1[1] * e2[0];
                    double cross_norm = std::sqrt(
                        cross_x * cross_x + cross_y * cross_y +
                        cross_z * cross_z);

                    if (cross_norm > 1e-12) {
                        double cot = dot_prod / cross_norm;

                        triplets.emplace_back(v1, v2, cot);
                        triplets.emplace_back(v2, v1, cot);
                        triplets.emplace_back(v1, v1, -cot);
                        triplets.emplace_back(v2, v2, -cot);
                    }
                }
            }
        }
    }

    // Assemble global Laplacian matrix (use double precision)
    Eigen::SparseMatrix<double> laplacian_double(n_vertices, n_vertices);
    laplacian_double.setFromTriplets(triplets.begin(), triplets.end());
    laplacian_double.makeCompressed();

    // Convert to float, multiply by 0.5, and negate to get positive
    // semi-definite matrix Standard cotangent Laplacian is negative
    // semi-definite, we want eigenvalues >= 0
    laplacian_matrix_ = (-0.5 * laplacian_double).cast<float>();

    std::cout << "Assembled 2D cotangent Laplacian matrix: " << n_vertices
              << " x " << n_vertices << " with " << laplacian_matrix_.nonZeros()
              << " non-zeros" << std::endl;
}

RUZINO_NAMESPACE_CLOSE_SCOPE
