#pragma once

#include <vector>

#include "i_spread.hpp"
#include "laplace_correction_support.hpp"
#include "../local_cauchy/laplace_panel_solver_2d.hpp"

namespace kfbim {

struct PatchBoundarySingularValueBasis2D {
    int patch = -1;
    std::vector<double> mode_basis;
    double legacy_basis = 0.0;
};

struct PatchBoundarySingularDifferenceBasis2D {
    PatchBoundarySingularValueBasis2D rhs;
    PatchBoundarySingularValueBasis2D correction;
    bool valid = false;
};

// ---------------------------------------------------------------------------
// Legacy 2D Laplace spread using the Gauss-point panel Cauchy solver.
//
// The Cauchy solve recovers a degree-2 Taylor correction polynomial C = u+ - u-
// at each legacy Gauss interface quadrature point, where u+ is the interior (label 1) and
// u- is the exterior (label 0). The jump is defined as [u] = u_int - u_ext.
// stencil defect at irregular grid nodes:
//
//   rhs[n] += side(nb)-side(n) * C(x_nb) / h_axis^2
//
// for each face neighbor nb across the interface.
// ---------------------------------------------------------------------------

class LaplacePanelSpread2D final : public ILaplaceSpread2D {
public:
    explicit LaplacePanelSpread2D(const GridPair2D& grid_pair,
                                  double            kappa = 0.0);

    LaplaceSpreadResult2D apply(
        const std::vector<LaplaceJumpData2D>& jumps,
        Eigen::VectorXd&                      rhs_correction) const override;

    const GridPair2D& grid_pair() const override { return grid_pair_; }

private:
    const GridPair2D& grid_pair_;
    double            kappa_;
};

// ---------------------------------------------------------------------------
// Preferred 2D Laplace spread for quadratic Lagrange P2 panel geometry.
//
// Interface points are 3 P2 geometry/jump DOFs per panel. Correction
// polynomials are solved at four generated panel-center locations. RHS and
// restrict grid-node corrections evaluate the Taylor polynomial from the
// nearest generated expansion center.
// ---------------------------------------------------------------------------

class LaplaceQuadraticPanelCenterSpread2D final : public ILaplaceSpread2D {
public:
    explicit LaplaceQuadraticPanelCenterSpread2D(
        const GridPair2D& grid_pair,
        double            kappa = 0.0,
        LaplaceCorrectionMethod2D correction_method =
            LaplaceCorrectionMethod2D::NearestExpansionCenter,
        int               projection_restrict_stencil_radius = 2,
        CornerPatchSolverOptions2D corner_patch_solver_options = {},
        SdCornerLiftingOptions2D sd_corner_lifting_options = {});

    LaplaceSpreadResult2D apply(
        const std::vector<LaplaceJumpData2D>& jumps,
        Eigen::VectorXd&                      rhs_correction) const override;

    const GridPair2D& grid_pair() const override { return grid_pair_; }

private:
    const GridPair2D& grid_pair_;
    double            kappa_;
    LaplaceCorrectionMethod2D correction_method_;
    int               projection_restrict_stencil_radius_;
    LaplaceCorrectionSupport2D support_;
    std::vector<int> crossing_center_indices_;
    std::vector<std::vector<int>> patch_regularized_center_indices_by_patch_;
    PanelCenterCauchyGeometryCache2D cauchy_geometry_cache_;
    NarrowBandProjection2D projection_cache_;
    mutable bool patch_boundary_singular_basis_cache_ready_ = false;
    mutable std::vector<PatchBoundarySingularDifferenceBasis2D>
        patch_boundary_singular_basis_by_crossing_;
    CornerPatchSolverOptions2D corner_patch_solver_options_;
    SdCornerLiftingOptions2D sd_corner_lifting_options_;
};

using LaplaceLobattoCenterSpread2D = LaplaceQuadraticPanelCenterSpread2D;

} // namespace kfbim
