#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>

#include "i_restrict.hpp"
#include "laplace_correction_support.hpp"

namespace kfbim {

struct CornerPatchAverageRestoreBasis2D {
    int patch = -1;
    std::vector<double> value_basis;
    std::vector<Eigen::Vector2d> gradient_basis;
    bool valid = false;
};

struct RestrictHalfCorrectionRowBasis2D {
    int grid_node = -1;
    Eigen::Matrix<double, 1, 6> full_basis =
        Eigen::Matrix<double, 1, 6>::Zero();
    Eigen::Matrix<double, 1, 6> patch_basis =
        Eigen::Matrix<double, 1, 6>::Zero();
};

struct RestrictHalfCorrectionBasis2D {
    int patch = -1;
    int anchor_center = -1;
    int patch_center = -1;
    std::array<RestrictHalfCorrectionRowBasis2D, 6> rows;
};

// ---------------------------------------------------------------------------
// Restrict companion for LaplaceQuadraticPanelCenterSpread2D.
//
// The incoming correction_polys are expansion-center Taylor polynomials, not
// interface-point polynomials.  Each interface-point fit anchors the fixed
// six-point square stencil to the closest bulk node's expansion-center type,
// so all samples in that fit use one jump source.  Grid samples are then
// shifted onto the average branch:
// interior samples subtract C/2, exterior samples add C/2.
// stencil_radius is retained for API compatibility and nearest-center cache
// construction; it no longer changes the interpolation stencil size.
// ---------------------------------------------------------------------------

class LaplaceQuadraticPanelCenterRestrict2D final : public ILaplaceRestrict2D {
public:
    explicit LaplaceQuadraticPanelCenterRestrict2D(
        const GridPair2D& grid_pair,
        int               stencil_radius = 2);

    std::vector<LocalPoly2D> apply(
        const Eigen::VectorXd&          bulk_solution,
        const LaplaceSpreadResult2D&    spread_result) const override;

    std::vector<LocalPoly2D> apply(
        const Eigen::VectorXd&          bulk_solution,
        const std::vector<LocalPoly2D>& correction_polys) const;

    // Recover a fixed-route virtual continuation of the interior branch.
    // Exterior stencil samples are shifted by the full value jump while
    // interior samples are left unchanged.  This route deliberately rejects
    // corner/singular correction states: those need a dedicated physical
    // interior-trace restore rather than the averaged-trace restore above.
    std::vector<LocalPoly2D> apply_interior_virtual(
        const Eigen::VectorXd&       bulk_solution,
        const LaplaceSpreadResult2D& spread_result) const;

    // Symmetric fixed-route diagnostic: map interior stencil samples onto
    // the exterior branch by subtracting the full value jump.  Like the
    // interior-virtual route, this first implementation excludes singular
    // correction states.
    std::vector<LocalPoly2D> apply_exterior_virtual(
        const Eigen::VectorXd&       bulk_solution,
        const LaplaceSpreadResult2D& spread_result) const;

    const GridPair2D& grid_pair() const override { return grid_pair_; }

private:
    enum class TraceRoute {
        Average,
        InteriorVirtual,
        ExteriorVirtual
    };

    std::vector<LocalPoly2D> apply_impl(
        const Eigen::VectorXd&       bulk_solution,
        const LaplaceSpreadResult2D& spread_result,
        TraceRoute                   trace_route) const;

    LocalPoly2D fit_at_interface_point(
        const Eigen::VectorXd&                    bulk_solution,
        int                                       q,
        const class LaplaceRestrictCorrectionEvaluator2D& correction_evaluator,
        TraceRoute                                trace_route) const;

    const GridPair2D& grid_pair_;
    int               stencil_radius_;
    LaplaceCorrectionSupport2D support_;
    std::vector<Eigen::Matrix<double, 6, 6>> fit_inverse_by_point_;
    std::vector<int> interface_panel_by_point_;
    std::vector<int> patch_boundary_center_by_point_;
    std::vector<int> full_boundary_center_by_point_;
    std::vector<RestrictHalfCorrectionBasis2D> half_correction_basis_by_point_;
    mutable bool average_restore_basis_cache_ready_ = false;
    mutable std::vector<CornerPatchAverageRestoreBasis2D>
        average_restore_basis_by_point_;
};

using LaplaceLobattoCenterRestrict2D = LaplaceQuadraticPanelCenterRestrict2D;

} // namespace kfbim
