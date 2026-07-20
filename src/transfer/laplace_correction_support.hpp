#pragma once

#include <array>
#include <vector>

#include "../geometry/corner_patch_2d.hpp"
#include "../geometry/grid_pair_2d.hpp"
#include "../geometry/grid_pair_3d.hpp"
#include "../grid/structured_grid_ops.hpp"

namespace kfbim {

enum class LaplaceCrossingKind {
    InterfaceJump,
    CornerPatchBoundary
};

enum class LaplaceJumpCorrectionSource2D {
    RawInterface,
    PatchRegularizedInterface,
    PatchBoundarySingular
};

enum class PatchInterfaceJumpMode2D {
    PhysicalJump,
    SingularRemovedJump
};

struct LaplaceCrossingCorrectionOp {
    LaplaceCrossingKind kind = LaplaceCrossingKind::InterfaceJump;
    int rhs_node = -1;
    int correction_node = -1;
    int neighbor_slot = -1;
    int side_delta = 0;
    int patch = -1;
    double stencil_weight = 0.0;
    int rhs_patch = -1;
    int correction_patch = -1;
};

inline LaplaceJumpCorrectionSource2D laplace_crossing_source_2d(
    const LaplaceCrossingCorrectionOp& op)
{
    if (op.kind == LaplaceCrossingKind::CornerPatchBoundary)
        return LaplaceJumpCorrectionSource2D::PatchBoundarySingular;
    return op.patch >= 0
             ? LaplaceJumpCorrectionSource2D::PatchRegularizedInterface
             : LaplaceJumpCorrectionSource2D::RawInterface;
}

struct LaplaceCorrectionSupport2D {
    std::vector<LaplaceCrossingCorrectionOp> crossing_ops;
    std::vector<std::array<int, 6>> restrict_stencils;
    std::vector<int> correction_nodes;
    std::vector<int> projection_nodes;
    int restrict_sample_visits = 0;
};

struct LaplaceCorrectionSupport3D {
    std::vector<LaplaceCrossingCorrectionOp> crossing_ops;
    std::vector<std::array<int, 10>> restrict_stencils;
    std::vector<int> correction_nodes;
    std::vector<int> projection_nodes;
    int restrict_sample_visits = 0;
};

namespace laplace_correction_support_detail {

inline int side_from_label(int label)
{
    return label == 0 ? 0 : 1;
}

inline Eigen::Vector2d interface_point(const Interface2D& iface, int q)
{
    return iface.points().row(q).transpose();
}

inline Eigen::Vector3d interface_point(const Interface3D& iface, int q)
{
    return iface.points().row(q).transpose();
}

inline void append_unique_projection_nodes(const std::vector<char>& mark,
                                           std::vector<int>&        nodes)
{
    nodes.clear();
    for (int idx = 0; idx < static_cast<int>(mark.size()); ++idx) {
        if (mark[idx])
            nodes.push_back(idx);
    }
}

} // namespace laplace_correction_support_detail

inline LaplaceCorrectionSupport2D build_laplace_correction_support_2d(
    const GridPair2D& grid_pair,
    const char*       context)
{
    const auto& grid = grid_pair.grid();
    const auto& iface = grid_pair.interface();
    const int n_grid = grid.num_dofs();
    std::vector<char> needs_c(n_grid, 0);

    LaplaceCorrectionSupport2D support;
    for (int n = 0; n < n_grid; ++n) {
        if (structured_grid::is_boundary_node(grid, n))
            continue;

        const int side_n =
            laplace_correction_support_detail::side_from_label(
                grid_pair.domain_label(n));
        const auto neighbors = grid.neighbors(n);
        for (int slot = 0; slot < 4; ++slot) {
            const int nb = neighbors[slot];
            if (nb < 0 || structured_grid::is_boundary_node(grid, nb))
                continue;

            const int side_nb =
                laplace_correction_support_detail::side_from_label(
                    grid_pair.domain_label(nb));
            const int patch_n = grid_pair.corner_patch_label(n);
            const int patch_nb = grid_pair.corner_patch_label(nb);
            if (side_nb != side_n) {
                const int patch_id = patch_n >= 0 ? patch_n : patch_nb;
                needs_c[n] = 1;
                needs_c[nb] = 1;
                LaplaceCrossingCorrectionOp op;
                op.kind = LaplaceCrossingKind::InterfaceJump;
                op.rhs_node = n;
                op.correction_node = nb;
                op.neighbor_slot = slot;
                op.side_delta = side_n - side_nb;
                op.patch = patch_id;
                op.stencil_weight =
                    structured_grid::stencil_weight_for_neighbor(grid, slot);
                op.rhs_patch = patch_n;
                op.correction_patch = patch_nb;
                support.crossing_ops.push_back(op);
            }

            // Corner-patch boundary corrections are same-phase decomposition
            // corrections. Edges crossing the physical interface use only the
            // regularized interface jump correction.
            if (patch_n != patch_nb && side_n == side_nb) {
                const int patch_id = patch_n >= 0 ? patch_n : patch_nb;

                needs_c[n] = 1;
                needs_c[nb] = 1;
                const int in_patch_n = patch_n >= 0 ? 1 : 0;
                const int in_patch_nb = patch_nb >= 0 ? 1 : 0;
                LaplaceCrossingCorrectionOp op;
                op.kind = LaplaceCrossingKind::CornerPatchBoundary;
                op.rhs_node = n;
                op.correction_node = nb;
                op.neighbor_slot = slot;
                op.side_delta = in_patch_n - in_patch_nb;
                op.patch = patch_id;
                op.stencil_weight =
                    structured_grid::stencil_weight_for_neighbor(grid, slot);
                op.rhs_patch = patch_n;
                op.correction_patch = patch_nb;
                support.crossing_ops.push_back(op);
            }
        }
    }

    support.restrict_stencils.reserve(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const int closest = grid_pair.closest_bulk_node(q);
        const auto stencil =
            structured_grid::quadratic_restrict_stencil_nodes_2d(
                context,
                grid,
                closest,
                laplace_correction_support_detail::interface_point(iface, q));
        support.restrict_stencils.push_back(stencil);
        for (int node : stencil) {
            needs_c[node] = 1;
            ++support.restrict_sample_visits;
        }
    }

    laplace_correction_support_detail::append_unique_projection_nodes(
        needs_c, support.correction_nodes);
    support.projection_nodes = support.correction_nodes;
    return support;
}

inline LaplaceCorrectionSupport3D build_laplace_correction_support_3d(
    const GridPair3D& grid_pair,
    const char*       context)
{
    const auto& grid = grid_pair.grid();
    const auto& iface = grid_pair.interface();
    const int n_grid = grid.num_dofs();
    std::vector<char> needs_c(n_grid, 0);

    LaplaceCorrectionSupport3D support;
    for (int n = 0; n < n_grid; ++n) {
        if (structured_grid::is_boundary_node(grid, n))
            continue;

        const int side_n =
            laplace_correction_support_detail::side_from_label(
                grid_pair.domain_label(n));
        const auto neighbors = grid.neighbors(n);
        for (int slot = 0; slot < 6; ++slot) {
            const int nb = neighbors[slot];
            if (nb < 0 || structured_grid::is_boundary_node(grid, nb))
                continue;

            const int side_nb =
                laplace_correction_support_detail::side_from_label(
                    grid_pair.domain_label(nb));
            if (side_nb == side_n)
                continue;

            needs_c[nb] = 1;
            support.crossing_ops.push_back(
                {LaplaceCrossingKind::InterfaceJump,
                 n,
                 nb,
                 slot,
                 side_n - side_nb,
                 -1,
                 structured_grid::stencil_weight_for_neighbor(grid, slot)});
        }
    }

    support.restrict_stencils.reserve(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const int closest = grid_pair.closest_bulk_node(q);
        const auto stencil =
            structured_grid::quadratic_restrict_stencil_nodes_3d(
                context,
                grid,
                closest,
                laplace_correction_support_detail::interface_point(iface, q));
        support.restrict_stencils.push_back(stencil);
        for (int node : stencil) {
            needs_c[node] = 1;
            ++support.restrict_sample_visits;
        }
    }

    laplace_correction_support_detail::append_unique_projection_nodes(
        needs_c, support.correction_nodes);
    support.projection_nodes = support.correction_nodes;
    return support;
}

} // namespace kfbim
