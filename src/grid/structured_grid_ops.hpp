#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "cartesian_grid_2d.hpp"
#include "cartesian_grid_3d.hpp"

namespace kfbim::structured_grid {

struct Index2D {
    int i = 0;
    int j = 0;
};

struct Index3D {
    int i = 0;
    int j = 0;
    int k = 0;
};

inline Index2D index_2d(const CartesianGrid2D& grid, int idx)
{
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    return {idx % nx, idx / nx};
}

inline Index3D index_3d(const CartesianGrid3D& grid, int idx)
{
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nxy = nx * ny;
    const int k = idx / nxy;
    const int rem = idx % nxy;
    return {rem % nx, rem / nx, k};
}

inline bool is_boundary_index(const CartesianGrid2D& grid, Index2D idx)
{
    const auto dims = grid.dof_dims();
    return idx.i == 0 || idx.i == dims[0] - 1
        || idx.j == 0 || idx.j == dims[1] - 1;
}

inline bool is_boundary_index(const CartesianGrid3D& grid, Index3D idx)
{
    const auto dims = grid.dof_dims();
    return idx.i == 0 || idx.i == dims[0] - 1
        || idx.j == 0 || idx.j == dims[1] - 1
        || idx.k == 0 || idx.k == dims[2] - 1;
}

inline bool is_boundary_node(const CartesianGrid2D& grid, int idx)
{
    return is_boundary_index(grid, index_2d(grid, idx));
}

inline bool is_boundary_node(const CartesianGrid3D& grid, int idx)
{
    return is_boundary_index(grid, index_3d(grid, idx));
}

inline Eigen::Vector2d point(const CartesianGrid2D& grid, int idx)
{
    const auto c = grid.coord(idx);
    return {c[0], c[1]};
}

inline Eigen::Vector3d point(const CartesianGrid3D& grid, int idx)
{
    const auto c = grid.coord(idx);
    return {c[0], c[1], c[2]};
}

inline double max_spacing(const CartesianGrid2D& grid)
{
    const auto h = grid.spacing();
    return std::max(h[0], h[1]);
}

inline double max_spacing(const CartesianGrid3D& grid)
{
    const auto h = grid.spacing();
    return std::max(h[0], std::max(h[1], h[2]));
}

inline double stencil_weight_for_neighbor(const CartesianGrid2D& grid,
                                          int                    neighbor_slot)
{
    const auto h = grid.spacing();
    if (neighbor_slot == 0 || neighbor_slot == 1)
        return 1.0 / (h[0] * h[0]);
    return 1.0 / (h[1] * h[1]);
}

inline double stencil_weight_for_neighbor(const CartesianGrid3D& grid,
                                          int                    neighbor_slot)
{
    const auto h = grid.spacing();
    if (neighbor_slot == 0 || neighbor_slot == 1)
        return 1.0 / (h[0] * h[0]);
    if (neighbor_slot == 2 || neighbor_slot == 3)
        return 1.0 / (h[1] * h[1]);
    return 1.0 / (h[2] * h[2]);
}

inline int restrict_stencil_side(double target_coord, double node_coord)
{
    return (target_coord > node_coord) ? 1 : -1;
}

inline std::array<Index2D, 6> quadratic_restrict_stencil_offsets_2d(
    int d1,
    int d2)
{
    return {{{0, 0},
             {d1, 0},
             {0, d2},
             {d1, d2},
             {-d1, 0},
             {0, -d2}}};
}

inline std::array<Index3D, 10> quadratic_restrict_stencil_offsets_3d(
    int d1,
    int d2,
    int d3)
{
    return {{{0, 0, 0},
             {0, d2, 0},
             {0, 0, d3},
             {d1, 0, 0},
             {0, d2, d3},
             {0, -d2, 0},
             {0, 0, -d3},
             {d1, d2, 0},
             {d1, 0, d3},
             {-d1, 0, 0}}};
}

inline bool is_valid_index(const CartesianGrid2D& grid, Index2D idx)
{
    const auto dims = grid.dof_dims();
    return idx.i >= 0 && idx.i < dims[0]
        && idx.j >= 0 && idx.j < dims[1];
}

inline bool is_valid_index(const CartesianGrid3D& grid, Index3D idx)
{
    const auto dims = grid.dof_dims();
    return idx.i >= 0 && idx.i < dims[0]
        && idx.j >= 0 && idx.j < dims[1]
        && idx.k >= 0 && idx.k < dims[2];
}

inline std::array<int, 6> quadratic_restrict_stencil_nodes_2d(
    const char* context,
    const CartesianGrid2D& grid,
    int closest_node,
    Eigen::Vector2d target)
{
    if (closest_node < 0 || closest_node >= grid.num_dofs()) {
        throw std::invalid_argument(
            std::string(context) + ": closest_node is outside the grid");
    }

    const Index2D base = index_2d(grid, closest_node);
    const Eigen::Vector2d base_pt = point(grid, closest_node);
    const int d1 = restrict_stencil_side(target[0], base_pt[0]);
    const int d2 = restrict_stencil_side(target[1], base_pt[1]);
    const auto offsets = quadratic_restrict_stencil_offsets_2d(d1, d2);

    std::array<int, 6> nodes{};
    for (std::size_t s = 0; s < offsets.size(); ++s) {
        const Index2D idx{base.i + offsets[s].i,
                          base.j + offsets[s].j};
        if (!is_valid_index(grid, idx)) {
            throw std::runtime_error(
                std::string(context)
                + ": fixed 2D quadratic restrict stencil leaves grid bounds");
        }
        nodes[s] = grid.index(idx.i, idx.j);
    }
    return nodes;
}

inline std::array<int, 10> quadratic_restrict_stencil_nodes_3d(
    const char* context,
    const CartesianGrid3D& grid,
    int closest_node,
    Eigen::Vector3d target)
{
    if (closest_node < 0 || closest_node >= grid.num_dofs()) {
        throw std::invalid_argument(
            std::string(context) + ": closest_node is outside the grid");
    }

    const Index3D base = index_3d(grid, closest_node);
    const Eigen::Vector3d base_pt = point(grid, closest_node);
    const int d1 = restrict_stencil_side(target[0], base_pt[0]);
    const int d2 = restrict_stencil_side(target[1], base_pt[1]);
    const int d3 = restrict_stencil_side(target[2], base_pt[2]);
    const auto offsets = quadratic_restrict_stencil_offsets_3d(d1, d2, d3);

    std::array<int, 10> nodes{};
    for (std::size_t s = 0; s < offsets.size(); ++s) {
        const Index3D idx{base.i + offsets[s].i,
                          base.j + offsets[s].j,
                          base.k + offsets[s].k};
        if (!is_valid_index(grid, idx)) {
            throw std::runtime_error(
                std::string(context)
                + ": fixed 3D quadratic restrict stencil leaves grid bounds");
        }
        nodes[s] = grid.index(idx.i, idx.j, idx.k);
    }
    return nodes;
}

template <typename Func>
void for_each_node_box(const CartesianGrid2D& grid,
                       int i_min,
                       int i_max,
                       int j_min,
                       int j_max,
                       Func&& fn)
{
    const auto dims = grid.dof_dims();
    i_min = std::max(0, i_min);
    j_min = std::max(0, j_min);
    i_max = std::min(dims[0] - 1, i_max);
    j_max = std::min(dims[1] - 1, j_max);
    for (int j = j_min; j <= j_max; ++j)
        for (int i = i_min; i <= i_max; ++i)
            fn(i, j, grid.index(i, j));
}

template <typename Func>
void for_each_node_box(const CartesianGrid3D& grid,
                       int i_min,
                       int i_max,
                       int j_min,
                       int j_max,
                       int k_min,
                       int k_max,
                       Func&& fn)
{
    const auto dims = grid.dof_dims();
    i_min = std::max(0, i_min);
    j_min = std::max(0, j_min);
    k_min = std::max(0, k_min);
    i_max = std::min(dims[0] - 1, i_max);
    j_max = std::min(dims[1] - 1, j_max);
    k_max = std::min(dims[2] - 1, k_max);
    for (int k = k_min; k <= k_max; ++k)
        for (int j = j_min; j <= j_max; ++j)
            for (int i = i_min; i <= i_max; ++i)
                fn(i, j, k, grid.index(i, j, k));
}

inline void require_grid_vector_size(const char* context,
                                     const char* name,
                                     Eigen::Index actual,
                                     Eigen::Index expected)
{
    if (actual != expected) {
        throw std::invalid_argument(
            std::string(context) + ": " + name
            + " has size " + std::to_string(actual)
            + ", expected " + std::to_string(expected));
    }
}

inline Eigen::VectorXd apply_dirichlet_boundary_elimination(
    const char* context,
    const CartesianGrid2D& grid,
    const Eigen::VectorXd& rhs,
    const Eigen::VectorXd& outer_dirichlet_values)
{
    const int n_dof = grid.num_dofs();
    require_grid_vector_size(context, "rhs", rhs.size(), n_dof);
    if (outer_dirichlet_values.size() == 0)
        return rhs;
    require_grid_vector_size(context,
                             "outer_dirichlet_values",
                             outer_dirichlet_values.size(),
                             n_dof);

    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    Eigen::VectorXd modified = rhs;
    const auto h = grid.spacing();
    const double inv_hx2 = 1.0 / (h[0] * h[0]);
    const double inv_hy2 = 1.0 / (h[1] * h[1]);

    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const int n = grid.index(i, j);
            double boundary_lift = 0.0;
            if (i == 1)
                boundary_lift += outer_dirichlet_values[grid.index(0, j)] * inv_hx2;
            if (i == nx - 2)
                boundary_lift += outer_dirichlet_values[grid.index(nx - 1, j)] * inv_hx2;
            if (j == 1)
                boundary_lift += outer_dirichlet_values[grid.index(i, 0)] * inv_hy2;
            if (j == ny - 2)
                boundary_lift += outer_dirichlet_values[grid.index(i, ny - 1)] * inv_hy2;
            modified[n] += boundary_lift;
        }
    }

    for (int n = 0; n < n_dof; ++n)
        if (is_boundary_node(grid, n))
            modified[n] = 0.0;

    return modified;
}

inline Eigen::VectorXd apply_dirichlet_boundary_elimination(
    const char* context,
    const CartesianGrid3D& grid,
    const Eigen::VectorXd& rhs,
    const Eigen::VectorXd& outer_dirichlet_values)
{
    const int n_dof = grid.num_dofs();
    require_grid_vector_size(context, "rhs", rhs.size(), n_dof);
    if (outer_dirichlet_values.size() == 0)
        return rhs;
    require_grid_vector_size(context,
                             "outer_dirichlet_values",
                             outer_dirichlet_values.size(),
                             n_dof);

    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = dims[2];
    Eigen::VectorXd modified = rhs;
    const auto h = grid.spacing();
    const double inv_hx2 = 1.0 / (h[0] * h[0]);
    const double inv_hy2 = 1.0 / (h[1] * h[1]);
    const double inv_hz2 = 1.0 / (h[2] * h[2]);

    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                const int n = grid.index(i, j, k);
                double boundary_lift = 0.0;
                if (i == 1)
                    boundary_lift += outer_dirichlet_values[grid.index(0, j, k)] * inv_hx2;
                if (i == nx - 2)
                    boundary_lift += outer_dirichlet_values[grid.index(nx - 1, j, k)] * inv_hx2;
                if (j == 1)
                    boundary_lift += outer_dirichlet_values[grid.index(i, 0, k)] * inv_hy2;
                if (j == ny - 2)
                    boundary_lift += outer_dirichlet_values[grid.index(i, ny - 1, k)] * inv_hy2;
                if (k == 1)
                    boundary_lift += outer_dirichlet_values[grid.index(i, j, 0)] * inv_hz2;
                if (k == nz - 2)
                    boundary_lift += outer_dirichlet_values[grid.index(i, j, nz - 1)] * inv_hz2;
                modified[n] += boundary_lift;
            }
        }
    }

    for (int n = 0; n < n_dof; ++n)
        if (is_boundary_node(grid, n))
            modified[n] = 0.0;

    return modified;
}

inline void restore_dirichlet_boundary(const char* context,
                                       const CartesianGrid2D& grid,
                                       Eigen::VectorXd&       u_bulk,
                                       const Eigen::VectorXd& outer_dirichlet_values)
{
    if (outer_dirichlet_values.size() == 0)
        return;
    const int n_dof = grid.num_dofs();
    require_grid_vector_size(context, "u_bulk", u_bulk.size(), n_dof);
    require_grid_vector_size(context,
                             "outer_dirichlet_values",
                             outer_dirichlet_values.size(),
                             n_dof);
    for (int n = 0; n < n_dof; ++n)
        if (is_boundary_node(grid, n))
            u_bulk[n] = outer_dirichlet_values[n];
}

inline void restore_dirichlet_boundary(const char* context,
                                       const CartesianGrid3D& grid,
                                       Eigen::VectorXd&       u_bulk,
                                       const Eigen::VectorXd& outer_dirichlet_values)
{
    if (outer_dirichlet_values.size() == 0)
        return;
    const int n_dof = grid.num_dofs();
    require_grid_vector_size(context, "u_bulk", u_bulk.size(), n_dof);
    require_grid_vector_size(context,
                             "outer_dirichlet_values",
                             outer_dirichlet_values.size(),
                             n_dof);
    for (int n = 0; n < n_dof; ++n)
        if (is_boundary_node(grid, n))
            u_bulk[n] = outer_dirichlet_values[n];
}

} // namespace kfbim::structured_grid
