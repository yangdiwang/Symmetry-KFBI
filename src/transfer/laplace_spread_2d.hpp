#pragma once

#include <memory>
#include <vector>

#include "i_spread.hpp"
#include "laplace_correction_support.hpp"
#include "../local_cauchy/laplace_panel_solver_2d.hpp"

namespace kfbim {

enum class LaplaceP2PanelCenterSpreadMode2D {
    QuadraticCauchy,
    CubicHarmonic
};

struct LaplaceP2CubicHarmonicSpreadOptions2D {
    int value_neighbors = 4;
    int normal_neighbors = 3;
    double relative_svd_cutoff = 2.0e-12;
    double distance_weight_shift = 0.35;
    double normal_row_weight = 0.85;
};

struct LaplaceP2CubicHarmonicCache2D;

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
// Interface points are 3 P2 geometry/jump DOFs per panel. The legacy route
// solves correction polynomials at four generated panel-center locations.
// CrossingOwner instead retains every reliable segment intersection and
// constructs a complete quadratic spatial jet there from a BVP-selected
// phi-P3, psi-P2, or combined trace stencil, evaluates the local geometry, and
// closes the normal Hessian with the PDE. Generated center polynomials remain
// available for corner, gap, multiple-hit, and other unresolved fallbacks, and
// for optional center-based restrict continuation.
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
        SdCornerLiftingOptions2D sd_corner_lifting_options = {},
        LaplaceP2PanelCenterSpreadMode2D spread_mode =
            LaplaceP2PanelCenterSpreadMode2D::QuadraticCauchy,
        LaplaceP2CubicHarmonicSpreadOptions2D
            cubic_harmonic_options = {},
        LaplaceCrossingTraceStencil2D crossing_trace_stencil =
            LaplaceCrossingTraceStencil2D::PhiP3PsiP2,
        LaplaceCrossingJetScheme2D crossing_jet_scheme =
            LaplaceCrossingJetScheme2D::ArcLengthBSplineCrossingJet);

    LaplaceSpreadResult2D apply(
        const std::vector<LaplaceJumpData2D>& jumps,
        Eigen::VectorXd&                      rhs_correction) const override;

    LaplaceSpreadResult2D apply_with_crossing_trace_stencil(
        const std::vector<LaplaceJumpData2D>& jumps,
        Eigen::VectorXd&                      rhs_correction,
        LaplaceCrossingTraceStencil2D         trace_stencil) const override;

    // Direct-coefficient route. `jumps` still supplies full-interface values
    // for legacy fallbacks and the PDE forcing jump, but every certified true
    // crossing consumes `direct_state` without invoking the historical
    // nodal-to-spline fit.
    LaplaceSpreadResult2D apply_with_direct_nurbs_density_state(
        const std::vector<LaplaceJumpData2D>& jumps,
        Eigen::VectorXd& rhs_correction,
        std::shared_ptr<const LaplaceNurbsDensityTraceState2D>
            direct_state) const;

    const GridPair2D& grid_pair() const override { return grid_pair_; }
    LaplaceCrossingTraceStencil2D crossing_trace_stencil() const override
    {
        return crossing_trace_stencil_;
    }
    LaplaceCrossingJetScheme2D crossing_jet_scheme() const
    {
        return crossing_jet_scheme_;
    }
    LaplaceP2PanelCenterSpreadMode2D spread_mode() const {
        return spread_mode_;
    }
    double cubic_harmonic_max_condition() const;

private:
    LaplaceSpreadResult2D apply_impl(
        const std::vector<LaplaceJumpData2D>& jumps,
        Eigen::VectorXd& rhs_correction,
        LaplaceCrossingTraceStencil2D trace_stencil,
        std::shared_ptr<const LaplaceNurbsDensityTraceState2D>
            direct_state) const;

    const GridPair2D& grid_pair_;
    double            kappa_;
    LaplaceCorrectionMethod2D correction_method_;
    int               projection_restrict_stencil_radius_;
    LaplaceCorrectionSupport2D support_;
    std::vector<P2CrossingOwner2D> crossing_owners_;
    std::vector<int> crossing_center_indices_;
    std::vector<std::vector<int>> patch_regularized_center_indices_by_patch_;
    PanelCenterCauchyGeometryCache2D cauchy_geometry_cache_;
    NarrowBandProjection2D projection_cache_;
    mutable bool patch_boundary_singular_basis_cache_ready_ = false;
    mutable std::vector<PatchBoundarySingularDifferenceBasis2D>
        patch_boundary_singular_basis_by_crossing_;
    CornerPatchSolverOptions2D corner_patch_solver_options_;
    SdCornerLiftingOptions2D sd_corner_lifting_options_;
    LaplaceP2PanelCenterSpreadMode2D spread_mode_;
    LaplaceP2CubicHarmonicSpreadOptions2D cubic_harmonic_options_;
    LaplaceCrossingTraceStencil2D crossing_trace_stencil_;
    LaplaceCrossingJetScheme2D crossing_jet_scheme_;
    std::shared_ptr<const LaplaceArcLengthBSplineCrossingJetPlan2D>
        arc_length_bspline_crossing_jet_plan_;
    std::shared_ptr<const LaplaceP2CubicHarmonicCache2D>
        cubic_harmonic_cache_;
};

using LaplaceLobattoCenterSpread2D = LaplaceQuadraticPanelCenterSpread2D;

} // namespace kfbim
