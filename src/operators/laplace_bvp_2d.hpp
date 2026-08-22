#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "../geometry/grid_pair_2d.hpp"
#include "../geometry/corner_patch_2d.hpp"
#include "../grid/cartesian_grid_2d.hpp"
#include "../interface/interface_2d.hpp"
#include "i_kfbi_operator.hpp"
#include "../potentials/laplace_potential.hpp"
#include "../bulk_solvers/laplace_zfft_bulk_solver_2d.hpp"
#include "../transfer/laplace_restrict_2d.hpp"
#include "../transfer/laplace_spread_2d.hpp"

namespace kfbim {

enum class LaplaceBvpType2D {
    InteriorDirichlet,
    ExteriorDirichlet,
    InteriorNeumann,
    ExteriorNeumann
};

// Default crossing-local stencil for the legacy layer-density formulation.
// ExteriorTraceCauchy selects PhiP3/PsiP2 per potential subsolve instead.
LaplaceCrossingTraceStencil2D laplace_crossing_trace_stencil_for_bvp_2d(
    LaplaceBvpType2D type);

enum class LaplaceBvpFormulation2D {
    LayerDensity,
    ExteriorTraceCauchy
};

enum class LaplaceBvpPanelMethod2D {
    QuadraticPanelCenter,

    // Backward-compatible aliases for older active 2D call sites.
    ChebyshevLobattoCenter = QuadraticPanelCenter,
    LobattoCenter = QuadraticPanelCenter
};

enum class LaplaceBvpRestrictMethod2D {
    // Fixed six-node complete-quadratic fit. In CrossingOwner mode every
    // half-jump sample is evaluated from the ALS-CJ polynomial frozen at the
    // interface trace point; unavailable trace points retain center fallback.
    SixPointQuadraticCrossingOwner,

    // Three/four normal layers with spatial quadratic/cubic interpolation.
    // Every wrong-side interpolation node is converted through its actual
    // query-to-node crossing and the shared ALS-CJ state.
    JointPolynomialCrossingOwner,

    // USN-P2: one strict complete spatial P2 stencil per trace side, shared
    // by all three normal samples; a trace-point Cauchy jump polynomial and
    // one unified normal P2 recover the interface value and normal derivative.
    UnifiedSpatialNormalP2CrossingOwner,

    // USN-P2-DOF-EXT: all spatial corrections use the Cauchy polynomial at
    // the current interface DOF. Interior normal samples are converted
    // directly to the exterior virtual field and only that normal P2 is fit.
    UnifiedSpatialNormalP2DofCauchyExterior
};

inline constexpr const char* laplace_bvp_restrict_method_name_2d(
    LaplaceBvpRestrictMethod2D method)
{
    switch (method) {
    case LaplaceBvpRestrictMethod2D::
             SixPointQuadraticCrossingOwner:
        return "six_point_quadratic_crossing_owner";
    case LaplaceBvpRestrictMethod2D::
             JointPolynomialCrossingOwner:
        return "joint_polynomial_crossing_owner";
    case LaplaceBvpRestrictMethod2D::
             UnifiedSpatialNormalP2CrossingOwner:
        return "unified_spatial_normal_p2_crossing_owner";
    case LaplaceBvpRestrictMethod2D::
             UnifiedSpatialNormalP2DofCauchyExterior:
        return "unified_spatial_normal_p2_dof_cauchy_exterior";
    }
    return "unknown";
}

struct LaplaceBvpOptions2D {
    LaplaceBvpPanelMethod2D panel_method =
        LaplaceBvpPanelMethod2D::QuadraticPanelCenter;
    double eta = 0.0;
    Eigen::VectorXd outer_dirichlet_values;
    int restrict_stencil_radius = 2;
    LaplaceBvpRestrictMethod2D restrict_method =
        LaplaceBvpRestrictMethod2D::SixPointQuadraticCrossingOwner;
    LaplaceP2JointPolynomialRestrictOptions2D joint_restrict;
    LaplaceCorrectionMethod2D correction_method =
        LaplaceCorrectionMethod2D::NearestExpansionCenter;
    LaplaceBvpFormulation2D formulation =
        LaplaceBvpFormulation2D::LayerDensity;
    // Formal crossing reconstruction policy. ALS-CJ uses a component-global
    // physical-arclength coordinate with periodic P3(phi)/P2(psi) B-splines.
    LaplaceCrossingJetScheme2D crossing_jet_scheme =
        LaplaceCrossingJetScheme2D::ArcLengthBSplineCrossingJet;
    CornerPatchConfig2D corner_patch;
    CornerPatchSolverOptions2D corner_patch_solver;
    // Optional S_D corner lifting. With strength_source=P2EndpointFromUJump,
    // every BVP operator application extracts corner-edge strengths from the
    // current u_jump/density trace using P2 arclength endpoint derivatives.
    SdCornerLiftingOptions2D sd_corner_lifting;
    // Optional active boundary unknowns. When empty, every interface point is
    // an unknown. When nonempty, GMRES vectors use this index list and the
    // remaining interface points are geometry/metadata-only in the BVP solve.
    std::vector<int> active_interface_points;
};

struct LaplaceBvpSolveResult2D {
    Eigen::VectorXd     u_bulk;
    Eigen::VectorXd     u_physical;
    Eigen::VectorXd     density;
    std::vector<CornerPatchCorrectionData2D> corner_patch_corrections;
    std::vector<double> residuals;
    int                 iterations;
    bool                converged;
    // Active-interface traces.  For ExteriorTraceCauchy, density is the
    // complementary Cauchy jump, while these vectors expose the recovered
    // physical- and ghost-side data directly.  Kept as trailing fields so
    // existing aggregate initializers remain source compatible.
    Eigen::VectorXd     trace;
    Eigen::VectorXd     normal_trace;
    Eigen::VectorXd     ghost_trace;
    Eigen::VectorXd     ghost_normal_trace;
};

// ---------------------------------------------------------------------------
// LaplaceBvp2D
//
// Solves 2D interior/exterior Dirichlet and Neumann BVPs:
//   -Delta u + eta*u = f
//
// For LayerDensity, boundary_data stores physical Dirichlet values or physical
// normal-derivative values. For ExteriorTraceCauchy, it stores the prescribed
// phi=[u] block (Dirichlet) or psi=[u_n] block (Neumann) directly in the
// repository's interior-minus-exterior jump convention; no physical-side sign
// is applied to that density block.
// ---------------------------------------------------------------------------
class LaplaceBvp2D : public IKFBIOperator {
public:
    LaplaceBvp2D(const CartesianGrid2D& grid,
                 const Interface2D&     iface,
                 LaplaceBvpType2D       type,
                 LaplaceBvpOptions2D    options = {});

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;
    int problem_size() const override;

    LaplaceBvpSolveResult2D solve(const Eigen::VectorXd&              boundary_data,
                                  const Eigen::VectorXd&              f_bulk,
                                  const std::vector<Eigen::VectorXd>& rhs_derivs,
                                  int max_iter = 100,
                                  double tol = 1e-8,
                                  int restart = 50) const;

    const GridPair2D& grid_pair() const { return grid_pair_; }
    LaplaceBvpType2D type() const { return type_; }
    double eta() const { return eta_; }
    LaplaceBvpFormulation2D formulation() const { return formulation_; }
    int active_interface_point(int active_index) const {
        return active_interface_points_[static_cast<std::size_t>(active_index)];
    }
    const std::vector<int>& active_interface_points() const {
        return active_interface_points_;
    }
    const LaplaceP2JointPolynomialRestrictDiagnostics2D*
    joint_polynomial_restrict_diagnostics() const
    {
        const auto* joint = dynamic_cast<const
            LaplaceP2CrossingOwnerJointPolynomialRestrict2D*>(
                restrict_op_.get());
        return joint != nullptr ? &joint->diagnostics() : nullptr;
    }

private:
    Eigen::VectorXd apply_dirichlet_boundary_elimination(
        const Eigen::VectorXd& rhs) const;

    void restore_dirichlet_boundary(Eigen::VectorXd& u_bulk) const;

    void apply_with_rhs(const Eigen::VectorXd&              density,
                        const Eigen::VectorXd&              rhs,
                        const std::vector<Eigen::VectorXd>& rhs_derivs,
                        Eigen::VectorXd&                    y) const;

    void apply_exterior_trace_unknown(const Eigen::VectorXd& density,
                                      Eigen::VectorXd&       y) const;

    LaplaceBvpSolveResult2D solve_exterior_trace(
        const Eigen::VectorXd&              boundary_data,
        const Eigen::VectorXd&              f_bulk,
        const std::vector<Eigen::VectorXd>& rhs_derivs,
        int max_iter,
        double tol,
        int restart) const;

    Eigen::VectorXd expand_active_density(
        const Eigen::VectorXd& active_density) const;
    std::vector<Eigen::VectorXd> expand_active_rhs_derivs(
        const std::vector<Eigen::VectorXd>& rhs_derivs) const;

    std::shared_ptr<const Interface2D> effective_iface_;
    GridPair2D                       grid_pair_;
    std::unique_ptr<ILaplaceSpread2D> spread_;
    LaplaceFftBulkSolverZfft2D       bulk_solver_;
    std::unique_ptr<ILaplaceRestrict2D> restrict_op_;
    LaplacePotentialEval2D           potentials_;

    LaplaceBvpType2D             type_;
    LaplaceBvpFormulation2D      formulation_;
    double                       rhs_deriv_sign_;
    double                       eta_;
    Eigen::VectorXd              outer_dirichlet_values_;
    std::vector<int>             active_interface_points_;
    std::vector<int>             iface_to_active_;
};

} // namespace kfbim
