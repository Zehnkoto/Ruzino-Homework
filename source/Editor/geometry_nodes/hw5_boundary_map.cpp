#include <Eigen/Sparse>
#include <cmath>
#include <vector>

#include "GCore/Components/MeshComponent.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
** @brief HW4_TutteParameterization
**
** This file contains two nodes whose primary function is to map the boundary of
*a mesh to a plain
** convex closed curve (circle of square), setting the stage for subsequent
*Laplacian equation
** solution and mesh parameterization tasks.
**
** Key to this node's implementation is the adept manipulation of half-edge data
*structures
** to identify and modify the boundary of the mesh.
**
** Task Overview:
** - The two execution functions (node_map_boundary_to_square_exec,
** node_map_boundary_to_circle_exec) require an update to accurately map the
*mesh boundary to a and
** circles. This entails identifying the boundary edges, evenly distributing
*boundary vertices along
** the square's perimeter, and ensuring the internal vertices' positions remain
*unchanged.
** - A focus on half-edge data structures to efficiently traverse and modify
*mesh boundaries.
*/

NODE_DEF_OPEN_SCOPE

/*
** HW4_TODO: Node to map the mesh boundary to a circle.
*/

NODE_DECLARATION_FUNCTION(hw5_circle_boundary_mapping)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");
    // Output-1: Processed 3D mesh whose boundary is mapped to a square and the
    // interior vertices remains the same
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_circle_boundary_mapping)
{
    // Get the input from params
    auto input = params.get_input<Geometry>("Input");

    // (TO BE UPDATED) Avoid processing the node when there is no input
    if (!input.get_component<MeshComponent>()) {
        throw std::runtime_error("Boundary Mapping: Need Geometry Input.");
    }

    /* ----------------------------- Preprocess -------------------------------
    ** Create a halfedge structure (using OpenMesh) for the input mesh. The
    ** half-edge data structure is a widely used data structure in geometric
    ** processing, offering convenient operations for traversing and modifying
    ** mesh elements.
    */
    auto halfedge_mesh = operand_to_openmesh(&input);

    /* ----------- [HW4_TODO] TASK 2.1: Boundary Mapping (to circle)
     *------------
     ** In this task, you are required to map the boundary of the mesh to a
     *circle
     ** shape while ensuring the internal vertices remain unaffected. This step
     *is
     ** crucial for setting up the mesh for subsequent parameterization tasks.
     **
     ** Algorithm Pseudocode for Boundary Mapping to Circle
     ** ------------------------------------------------------------------------
     ** 1. Identify the boundary loop(s) of the mesh using the half-edge
     *structure.
     **
     ** 2. Calculate the total length of the boundary loop to determine the
     *spacing
     **    between vertices when mapped to a square.
     **
     ** 3. Sequentially assign each boundary vertex a new position along the
     *square's
     **    perimeter, maintaining the calculated spacing to ensure proper
     *distribution.
     **
     ** 4. Keep the interior vertices' positions unchanged during this process.
     **
     ** Note: How to distribute the points on the circle?
     **
     ** Note: It would be better to normalize the boundary to a unit circle in
     *[0,1]x[0,1] for
     ** texture mapping.
     */

    // 1. Find a starting boundary halfedge
    OpenMesh::SmartHalfedgeHandle start_he;
    bool found_boundary = false;
    for (auto he : halfedge_mesh->halfedges()) {
        if (he.is_boundary()) {
            start_he = he;
            found_boundary = true;
            break;
        }
    }

    if (!found_boundary) {
        throw std::runtime_error("Circle Mapping: The mesh has no boundary!");
        return false;
    }

    // 2. Traverse the boundary loop and calculate total length
    std::vector<OpenMesh::SmartVertexHandle> bnd_vertices;
    std::vector<double> edge_lengths;
    double total_length = 0.0;

    auto he = start_he;
    do {
        bnd_vertices.push_back(he.from());

        auto p1 = halfedge_mesh->point(he.from());
        auto p2 = halfedge_mesh->point(he.to());
        double len = (p1 - p2).norm();

        edge_lengths.push_back(len);
        total_length += len;

        he = he.next();  // For boundary halfedge, next() goes along the
                         // boundary loop
    } while (he != start_he);

    // 3. Map boundary vertices to a unit circle [0,1]x[0,1] based on arc length
    double current_length = 0.0;
    for (size_t i = 0; i < bnd_vertices.size(); ++i) {
        double theta = 2.0 * M_PI * (current_length / total_length);

        // Map to [0,1]x[0,1] range: Center at (0.5, 0.5), radius = 0.5
        float x = 0.5f + 0.5f * std::cos(theta);
        float y = 0.5f + 0.5f * std::sin(theta);

        halfedge_mesh->set_point(bnd_vertices[i], OpenMesh::Vec3f(x, y, 0.0f));

        current_length += edge_lengths[i];
    }

    /* ----------------------------- Postprocess ------------------------------
    ** Convert the result mesh from the halfedge structure back to Geometry
    *format as the node's
    ** output.
    */
    auto geometry = openmesh_to_operand(halfedge_mesh.get());

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));
    return true;
}

/*
** HW4_TODO: Node to map the mesh boundary to a square.
*/

NODE_DECLARATION_FUNCTION(hw5_square_boundary_mapping)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");

    // Output-1: Processed 3D mesh whose boundary is mapped to a square and the
    // interior vertices remains the same
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_square_boundary_mapping)
{
    // Get the input from params
    auto input = params.get_input<Geometry>("Input");

    // (TO BE UPDATED) Avoid processing the node when there is no input
    if (!input.get_component<MeshComponent>()) {
        throw std::runtime_error("Input does not contain a mesh");
    }

    /* ----------------------------- Preprocess -------------------------------
    ** Create a halfedge structure (using OpenMesh) for the input mesh.
    */
    auto halfedge_mesh = operand_to_openmesh(&input);

    /* ----------- [HW4_TODO] TASK 2.2: Boundary Mapping (to square)
     *------------
     ** In this task, you are required to map the boundary of the mesh to a
     *circle
     ** shape while ensuring the internal vertices remain unaffected.
     **
     ** Algorithm Pseudocode for Boundary Mapping to Square
     ** ------------------------------------------------------------------------
     ** (omitted)
     **
     ** Note: Can you perserve the 4 corners of the square after boundary
     *mapping?
     **
     ** Note: It would be better to normalize the boundary to a unit circle in
     *[0,1]x[0,1] for
     ** texture mapping.
     */

    // 1. Find a starting boundary halfedge
    OpenMesh::SmartHalfedgeHandle start_he;
    bool found_boundary = false;
    for (auto he : halfedge_mesh->halfedges()) {
        if (he.is_boundary()) {
            start_he = he;
            found_boundary = true;
            break;
        }
    }

    if (!found_boundary) {
        throw std::runtime_error("Square Mapping: The mesh has no boundary!");
        return false;
    }

    // 2. Traverse the boundary loop and calculate segment lengths
    std::vector<OpenMesh::SmartVertexHandle> bnd_vertices;
    std::vector<double> edge_lengths;
    double total_length = 0.0;

    auto he = start_he;
    do {
        bnd_vertices.push_back(he.from());
        auto p1 = halfedge_mesh->point(he.from());
        auto p2 = halfedge_mesh->point(he.to());
        double len = (p1 - p2).norm();
        edge_lengths.push_back(len);
        total_length += len;
        he = he.next();
    } while (he != start_he);

    // 3. Compute accumulated length to find the 4 corners of the square
    std::vector<double> accum_length(bnd_vertices.size() + 1, 0.0);
    for (size_t i = 0; i < bnd_vertices.size(); ++i) {
        accum_length[i + 1] = accum_length[i] + edge_lengths[i];
    }

    // Target lengths for the 4 corners: 0, L/4, L/2, 3L/4
    int c[4] = { 0, 0, 0, 0 };
    double target[4] = {
        0.0, total_length * 0.25, total_length * 0.5, total_length * 0.75
    };

    for (int k = 1; k < 4; ++k) {
        double min_diff = 1e9;  // arbitrarily large number
        for (size_t i = 0; i < bnd_vertices.size(); ++i) {
            double diff = std::abs(accum_length[i] - target[k]);
            if (diff < min_diff) {
                min_diff = diff;
                c[k] = i;
            }
        }
    }

    // 4. Map the segments to the 4 edges of the square in [0,1]x[0,1]
    OpenMesh::Vec3f corners2D[4] = {
        OpenMesh::Vec3f(0.0f, 0.0f, 0.0f),  // Bottom-Left
        OpenMesh::Vec3f(1.0f, 0.0f, 0.0f),  // Bottom-Right
        OpenMesh::Vec3f(1.0f, 1.0f, 0.0f),  // Top-Right
        OpenMesh::Vec3f(0.0f, 1.0f, 0.0f)   // Top-Left
    };

    for (int k = 0; k < 4; ++k) {
        int start_idx = c[k];
        int end_idx = (k == 3) ? bnd_vertices.size() : c[k + 1];
        double segment_len = accum_length[end_idx] - accum_length[start_idx];

        for (int i = start_idx; i < end_idx; ++i) {
            double t =
                (segment_len > 1e-8)
                    ? (accum_length[i] - accum_length[start_idx]) / segment_len
                    : 0.0;

            // Linear interpolation between two corners
            OpenMesh::Vec3f pos =
                corners2D[k] * (1.0f - t) + corners2D[(k + 1) % 4] * t;
            halfedge_mesh->set_point(bnd_vertices[i], pos);
        }
    }

    /* ----------------------------- Postprocess ------------------------------
    ** Convert the result mesh from the halfedge structure back to Geometry
    *format as the node's
    ** output.
    */
    auto geometry = openmesh_to_operand(halfedge_mesh.get());

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(boundary_mapping);
NODE_DEF_CLOSE_SCOPE