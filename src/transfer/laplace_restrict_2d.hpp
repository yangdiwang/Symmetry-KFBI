#pragma once

#include <array>
#include <memory>
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

enum class LaplaceP2JointRestrictDegree2D {
    Quadratic = 2,
    Cubic = 3
};

enum class LaplaceP2JointJumpEvaluation2D {
    InterfaceLinear,
    NearestIncidentPanelCenterCauchy,
    InterfaceTracePointCauchy
};

enum class LaplaceP2JointNormalFit2D {
    PiecewiseSharedValueAndDerivative,
    UnifiedQuadratic
};

enum class LaplaceP2JointVirtualTraceRoute2D {
    // Build both virtual branches at the normal samples and average their
    // independently recovered trace data.
    SymmetricTwoBranchAverage,

    // Use the local Cauchy polynomial at the current interface DOF for every
    // spatial correction. Convert the interior normal samples directly to
    // the exterior virtual solution, fit only that branch, and restore the
    // requested trace at the interface point itself.
    InterfaceDofCauchyExterior
};

enum class LaplaceP2QuadraticGridStencil2D {
    TensorProduct3x3,
    SameSideCrossUniqueDiagonal,
    SameSideCrossGridEdgeDiagonal,
    SameSideCrossGridEdgeDiagonalSharedSidePolynomial,
    UnifiedCompleteP2SharedSidePolynomial
};

struct LaplaceP2JointPolynomialRestrictOptions2D {
    LaplaceP2JointRestrictDegree2D degree =
        LaplaceP2JointRestrictDegree2D::Cubic;
    int normal_layer_count = 4;
    // After the physical one-sided normal samples have been interpolated,
    // select how the jump is continued from the trace point to those sample
    // locations before the joint normal fit.  InterfaceLinear preserves the
    // original [u] +/- rho [u_n] continuation.  The panel-center route binds
    // each trace DOF to the nearest generated Cauchy center on an incident P2
    // panel and evaluates that complete quadratic polynomial at all samples.
    LaplaceP2JointJumpEvaluation2D jump_evaluation =
        LaplaceP2JointJumpEvaluation2D::InterfaceLinear;
    LaplaceP2JointNormalFit2D normal_fit =
        LaplaceP2JointNormalFit2D::PiecewiseSharedValueAndDerivative;
    LaplaceP2JointVirtualTraceRoute2D virtual_trace_route =
        LaplaceP2JointVirtualTraceRoute2D::
            SymmetricTwoBranchAverage;
    // Quadratic sampling normally uses a tensor 3x3 stencil.  The query-ray
    // cross route instead uses the nearest same-phase node, its four Cartesian
    // neighbors, and the nearest diagonal whose query-to-node segment has one
    // distinct crossing.  The grid-edge route binds the four Cartesian nodes
    // through center-to-node grid edges and accepts an opposite-phase diagonal
    // when at least one of its two incident Cartesian edges has an unambiguous
    // crossing; two incident owners are blended symmetrically.  The shared
    // side-polynomial variant selects one such six-node stencil at the
    // midpoint of the normal sampling interval on each side and evaluates
    // that same complete quadratic at all three normal sample locations.
    LaplaceP2QuadraticGridStencil2D quadratic_grid_stencil =
        LaplaceP2QuadraticGridStencil2D::TensorProduct3x3;
    // For the 3x3 quadratic route, allow the interpolated virtual phase to
    // differ from the physical phase of the normal sample.  This lets a
    // convex-corner interior sample use an exterior-dominant stencil, convert
    // only the few interior nodes to the exterior virtual field, and restore
    // the interior value at the query.
    bool allow_virtual_side_flip = false;
    // The cubic route uses all four layers; the quadratic route uses the
    // first three.
    std::array<double, 4> normal_layers{{0.2, 0.6, 1.0, 1.4}};
};

inline LaplaceP2JointPolynomialRestrictOptions2D
make_laplace_p2_joint_quadratic_restrict_options_2d()
{
    LaplaceP2JointPolynomialRestrictOptions2D options;
    options.degree = LaplaceP2JointRestrictDegree2D::Quadratic;
    options.normal_layer_count = 3;
    return options;
}

inline LaplaceP2JointPolynomialRestrictOptions2D
make_laplace_p2_joint_quadratic_cross_stencil_interface_jump_options_2d()
{
    // Strongly decoupled jump route: first recover the three physical
    // samples on each side with the spatial cross stencil, then perform the
    // shared quadratic normal fit.  InterfaceLinear is algebraically the
    // constrained least-squares fit that enforces [u] and [u_n] at the trace
    // point; it does not bind the trace point to an incident panel center.
    LaplaceP2JointPolynomialRestrictOptions2D options =
        make_laplace_p2_joint_quadratic_restrict_options_2d();
    options.quadratic_grid_stencil =
        LaplaceP2QuadraticGridStencil2D::
            SameSideCrossUniqueDiagonal;
    return options;
}

inline LaplaceP2JointPolynomialRestrictOptions2D
make_laplace_p2_joint_quadratic_grid_edge_cross_stencil_interface_jump_options_2d()
{
    LaplaceP2JointPolynomialRestrictOptions2D options =
        make_laplace_p2_joint_quadratic_restrict_options_2d();
    options.quadratic_grid_stencil =
        LaplaceP2QuadraticGridStencil2D::
            SameSideCrossGridEdgeDiagonal;
    return options;
}

inline LaplaceP2JointPolynomialRestrictOptions2D
make_laplace_p2_joint_quadratic_grid_edge_shared_side_polynomial_interface_jump_options_2d()
{
    LaplaceP2JointPolynomialRestrictOptions2D options =
        make_laplace_p2_joint_quadratic_restrict_options_2d();
    options.quadratic_grid_stencil =
        LaplaceP2QuadraticGridStencil2D::
            SameSideCrossGridEdgeDiagonalSharedSidePolynomial;
    return options;
}

// USN-P2: one strict complete spatial P2 per physical side, shared by all
// three normal samples; a complete trace-point jump polynomial converts the
// samples to one virtual branch, followed by one unified normal P2 fit.
inline LaplaceP2JointPolynomialRestrictOptions2D
make_laplace_p2_unified_spatial_normal_restrict_options_2d()
{
    LaplaceP2JointPolynomialRestrictOptions2D options =
        make_laplace_p2_joint_quadratic_restrict_options_2d();
    options.jump_evaluation =
        LaplaceP2JointJumpEvaluation2D::InterfaceTracePointCauchy;
    options.normal_fit = LaplaceP2JointNormalFit2D::UnifiedQuadratic;
    options.quadratic_grid_stencil =
        LaplaceP2QuadraticGridStencil2D::
            UnifiedCompleteP2SharedSidePolynomial;
    return options;
}

// USN-P2-DOF-EXT: use one interface-DOF Cauchy polynomial for all spatial
// corrections, map the three interior normal samples directly to the
// exterior virtual field, and fit only one unified normal P2. The public
// restrict result still observes the average/interior/exterior trace contract
// by restoring the jump at rho=0 after the exterior fit.
inline LaplaceP2JointPolynomialRestrictOptions2D
make_laplace_p2_dof_cauchy_exterior_restrict_options_2d()
{
    LaplaceP2JointPolynomialRestrictOptions2D options =
        make_laplace_p2_unified_spatial_normal_restrict_options_2d();
    options.virtual_trace_route =
        LaplaceP2JointVirtualTraceRoute2D::
            InterfaceDofCauchyExterior;
    return options;
}

inline LaplaceP2JointPolynomialRestrictOptions2D
make_laplace_p2_joint_quadratic_center_cauchy_jump_options_2d()
{
    LaplaceP2JointPolynomialRestrictOptions2D options =
        make_laplace_p2_joint_quadratic_restrict_options_2d();
    options.jump_evaluation =
        LaplaceP2JointJumpEvaluation2D::
            NearestIncidentPanelCenterCauchy;
    return options;
}

inline LaplaceP2JointPolynomialRestrictOptions2D
make_laplace_p2_joint_quadratic_cross_stencil_center_cauchy_jump_options_2d()
{
    LaplaceP2JointPolynomialRestrictOptions2D options =
        make_laplace_p2_joint_quadratic_center_cauchy_jump_options_2d();
    options.quadratic_grid_stencil =
        LaplaceP2QuadraticGridStencil2D::
            SameSideCrossUniqueDiagonal;
    return options;
}

inline LaplaceP2JointPolynomialRestrictOptions2D
make_laplace_p2_joint_quadratic_virtual_side_flip_options_2d()
{
    LaplaceP2JointPolynomialRestrictOptions2D options =
        make_laplace_p2_joint_quadratic_restrict_options_2d();
    options.allow_virtual_side_flip = true;
    return options;
}

using LaplaceP2JointCubicRestrictOptions2D =
    LaplaceP2JointPolynomialRestrictOptions2D;

struct LaplaceP2JointPolynomialRestrictDiagnostics2D {
    // Reference-only counters are populated when arbitrary exact-NURBS trace
    // targets are configured.  They intentionally exclude the compatibility
    // Interface2D templates retained for other callers.
    int reference_trace_points = 0;
    int reference_trace_samples = 0;
    int reference_interpolation_nodes = 0;
    int reference_wrong_side_nodes = 0;
    int reference_corrected_nodes = 0;
    int reference_exact_crossing_owners = 0;
    int reference_gap_fallback_owners = 0;
    int reference_unresolved_gap_fallback_owners = 0;
    int reference_endpoint_fallback_owners = 0;
    int reference_unified_p2_stencils = 0;
    int reference_unified_p2_relocated_stencils = 0;
    int reference_unified_p2_candidate_rejections = 0;
    double reference_max_six_point_weight_l1 = 0.0;
    int trace_samples = 0;
    int interpolation_nodes = 0;
    int wrong_side_nodes = 0;
    int corrected_nodes = 0;
    int exact_crossing_owners = 0;
    // GapFallback is retained as a legacy status name. These two counters
    // distinguish a certified hit on the explicitly represented corner gap
    // from the true nearest-center fallback with no identified intersection.
    int gap_fallback_owners = 0;
    int identified_gap_crossing_owners = 0;
    int unresolved_gap_fallback_owners = 0;
    int endpoint_fallback_owners = 0;
    int virtual_side_flips = 0;
    int interior_sample_flips = 0;
    int exterior_sample_flips = 0;
    int shifted_stencils = 0;
    int center_cauchy_jump_samples = 0;
    int trace_points_without_incident_center = 0;
    int six_point_cross_stencils = 0;
    int diagonal_zero_crossing_rejections = 0;
    int diagonal_multiple_crossing_rejections = 0;
    int grid_edge_cross_stencils = 0;
    int grid_edge_cardinal_corrections = 0;
    int grid_edge_cardinal_ambiguous_rejections = 0;
    int grid_edge_diagonal_zero_owner_rejections = 0;
    int grid_edge_diagonal_ambiguous_edge_rejections = 0;
    int grid_edge_diagonal_same_side_stencils = 0;
    int grid_edge_diagonal_single_owner_stencils = 0;
    int grid_edge_diagonal_blended_owner_stencils = 0;
    int grid_edge_blended_correction_nodes = 0;
    int grid_edge_owner_terms = 0;
    int shared_side_spatial_polynomials = 0;
    int shared_side_spatial_polynomial_samples = 0;
    int unified_p2_stencils = 0;
    int unified_p2_relocated_stencils = 0;
    int unified_p2_candidate_rejections = 0;
    // Number of samples for which the absolute nearest same-side distance
    // shell had no unique-crossing diagonal and the next admissible shell was
    // used. The final two fields monitor locality and interpolation growth.
    int expanded_same_side_center_stencils = 0;
    double max_same_side_center_distance_over_h = 0.0;
    double max_six_point_weight_l1 = 0.0;
    struct GridEdgeSample {
        int q = -1;
        int side = -1;
        int layer = -1;
        Eigen::Vector2d query = Eigen::Vector2d::Zero();
        Eigen::Vector2d selection_query = Eigen::Vector2d::Zero();
        int center_node = -1;
        int diagonal_node = -1;
        double nearest_center_distance_over_h = 0.0;
        double selected_center_distance_over_h = 0.0;
        double evaluation_center_distance_over_h = 0.0;
        bool expanded_center = false;
        std::array<int, 6> grid_nodes{{-1, -1, -1, -1, -1, -1}};
        std::array<Eigen::Vector2d, 6> grid_points;
        std::array<double, 6> interpolation_weights{{
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        std::array<bool, 4> cardinal_corrected{{
            false, false, false, false}};
        std::array<P2CrossingOwner2D, 4> cardinal_owners;
        bool diagonal_corrected = false;
        int diagonal_owner_count = 0;
        std::array<P2CrossingOwner2D, 2> diagonal_owners;
    };
    std::vector<GridEdgeSample> grid_edge_samples;
};

using LaplaceP2JointCubicRestrictDiagnostics2D =
    LaplaceP2JointPolynomialRestrictDiagnostics2D;

// Exact-NURBS target used by coefficient exterior-trace equations.  These
// points need not be Interface2D compatibility nodes.  Their spatial P2
// interpolation templates are built once by configure_trace_references() and
// then reused unchanged by every operator application.
struct LaplaceNurbsTraceReferencePoint2D {
    int branch = -1;
    int density_span = -1;
    int gauss_node = -1;
    double parameter = 0.0;
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    double physical_weight = 0.0;
};

struct LaplaceRestrictedReferenceTrace2D {
    Eigen::VectorXd value;
    Eigen::VectorXd normal_derivative;
};

// P2 companion restrict with matched quadratic or cubic sampling:
//   1. sample at three/four equal inside/outside normal offsets;
//   2. recover each physical one-sided value by 3x3 biquadratic, 4x4
//      bicubic, or a six-node complete-quadratic cross-plus-diagonal stencil;
//   3. convert wrong-phase stencil nodes with a complete quadratic spatial
//      polynomial constructed at the actual query-to-support P2 crossing;
//      legacy routes retain center fallback, while strict unified routes
//      relocate the same P2 format and reject an unresolved final stencil;
//   4. continue the jump to each normal sample either linearly from the trace
//      DOF, with an incident-panel Cauchy polynomial, or with the complete
//      local Cauchy polynomial constructed at the current interface DOF;
//   5. fit two shared-trace normal polynomials, one unified polynomial per
//      branch, or one exterior virtual polynomial after direct DOF-Cauchy
//      conversion of the interior samples.
class LaplaceP2CrossingOwnerJointPolynomialRestrict2D final
    : public ILaplaceRestrict2D {
public:
    explicit LaplaceP2CrossingOwnerJointPolynomialRestrict2D(
        const GridPair2D& grid_pair,
        LaplaceP2JointPolynomialRestrictOptions2D options = {});
    ~LaplaceP2CrossingOwnerJointPolynomialRestrict2D();

    std::vector<LocalPoly2D> apply(
        const Eigen::VectorXd&       bulk_solution,
        const LaplaceSpreadResult2D& spread_result) const override;

    std::vector<LocalPoly2D> apply_interior_virtual(
        const Eigen::VectorXd&       bulk_solution,
        const LaplaceSpreadResult2D& spread_result) const;

    std::vector<LocalPoly2D> apply_exterior_virtual(
        const Eigen::VectorXd&       bulk_solution,
        const LaplaceSpreadResult2D& spread_result) const;

    // Configure once, before the first application.  The strict unified
    // complete-P2 spatial stencil and its crossing metadata are cached for
    // all normal samples of every exact-NURBS reference point.
    void configure_trace_references(
        std::vector<LaplaceNurbsTraceReferencePoint2D> references);

    LaplaceRestrictedReferenceTrace2D
    apply_exterior_virtual_at_references(
        const Eigen::VectorXd&       bulk_solution,
        const LaplaceSpreadResult2D& spread_result) const;

    const std::vector<LaplaceNurbsTraceReferencePoint2D>&
    trace_references() const;

    const GridPair2D& grid_pair() const override { return grid_pair_; }
    const LaplaceP2JointPolynomialRestrictDiagnostics2D& diagnostics() const;

private:
    enum class TraceRoute {
        Average,
        InteriorVirtual,
        ExteriorVirtual
    };

    std::vector<LocalPoly2D> apply_impl(
        const Eigen::VectorXd&       bulk_solution,
        const LaplaceSpreadResult2D& spread_result,
        TraceRoute                   route) const;

    struct Impl;
    const GridPair2D& grid_pair_;
    std::unique_ptr<Impl> impl_;
};

using LaplaceP2CrossingOwnerJointCubicRestrict2D =
    LaplaceP2CrossingOwnerJointPolynomialRestrict2D;

} // namespace kfbim
