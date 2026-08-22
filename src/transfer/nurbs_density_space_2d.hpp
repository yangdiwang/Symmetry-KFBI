#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "../geometry/nurbs_boundary_2d.hpp"
#include "laplace_crossing_local_polynomial_2d.hpp"

namespace kfbim {

// Numerical continuity requested for a scalar density at the connection from
// branch b to branch b+1 (cyclically for a closed boundary).  This is kept
// separate from CAD/NURBS geometric continuity: callers can, for example,
// request C0 for phi and Discontinuous for psi at the same G0 corner.
enum class NurbsDensityContinuity2D {
    Discontinuous = -1,
    C0 = 0,
    C1 = 1,
    C2 = 2
};

// Coordinate used by the scalar density spline.  PhysicalArclength makes the
// scalar basis itself intrinsic.  NurbsParameter keeps the scalar basis in the
// exact CAD parameter and converts its differential jet with the metric of the
// NURBS curve.  LegacyNurbsParameter is retained as a source-compatible name.
enum class NurbsDensityCoordinate2D {
    NurbsParameter,
    LegacyNurbsParameter = NurbsParameter,
    PhysicalArclength
};

// How tangential derivatives of a density are evaluated when building a
// crossing trace jet.  AnalyticBasis differentiates the B-spline basis.
// SampledFiniteDifference evaluates density values at neighbouring physical-
// arclength points. CovariantParameterFiniteDifference samples in the NURBS
// parameter xi, forms rho_xi/rho_xixi, and converts them to the intrinsic
// rho_s/rho_ss jet with J=|gamma_xi| and J_xi. Analytic rows remain available
// as a reference and for constructing exact physical continuity constraints.
enum class NurbsDensityDerivativeScheme2D {
    AnalyticBasis,
    SampledFiniteDifference,
    CovariantParameterFiniteDifference
};

struct NurbsDensityLocation2D {
    int branch = -1;
    double parameter = 0.0;
};

struct NurbsDensityJet2D {
    double value = 0.0;
    double arclength_first = 0.0;
    double arclength_second = 0.0;
};

// Dense reduced rows are intentional at this first integration boundary.
// The raw rows have only degree+1 nonzeros; reduction may couple endpoints
// when a closed or multi-branch continuity constraint is imposed.
struct NurbsDensityEvaluationRows2D {
    Eigen::RowVectorXd value;
    Eigen::RowVectorXd arclength_first;
    Eigen::RowVectorXd arclength_second;
};

struct NurbsDensitySample2D {
    NurbsDensityLocation2D location;
    double value = 0.0;
    double weight = 1.0;
};

struct NurbsDensityBranchInfo2D {
    int branch = -1;
    double parameter_start = 0.0;
    double parameter_end = 0.0;
    double arclength = 0.0;
    int spline_span_count = 0;
    int raw_coefficient_offset = 0;
    int raw_coefficient_count = 0;
    // A smooth one-branch closed component with the full C^(p-1) request is
    // represented by a native cyclic basis with exactly one coefficient per
    // span.  No endpoint constraint or dense null-space reduction is needed.
    bool native_periodic = false;
    // Clamped density-spline breakpoints. Interior entries are obtained by
    // distributing a fixed total span count over the smooth parameter charts
    // and inverting equal physical-arclength targets inside each chart.
    std::vector<double> parameter_breaks;
    // Exact interior NURBS knots used as covariant-chart boundaries. Empty for
    // a PhysicalArclength density basis. Parameter-coordinate bases lower
    // their xi continuity here and restore intrinsic C^(p-1) by reduction.
    std::vector<double> parameter_chart_breaks;
};

enum class NurbsDensityTraceRole2D {
    PhiValueJump,
    PsiNormalJump
};

// Direct coefficient space for one scalar interface density.
//
// In the production PhysicalArclength coordinate, every exact NURBS branch
// receives a spline in local physical arclength.  A smooth one-branch closed
// component uses a native cyclic basis; feature-separated branches use open
// clamped bases and only their requested connection constraints are assembled
// in C.  The parameter coordinate instead evaluates the same break topology in
// the original NURBS parameter, split into smooth parameter charts and joined
// by intrinsic derivative constraints. Remaining constraints are eliminated through
//
//                  raw_coefficients = R * reduced_coefficients.
//
// The reduced coefficients are the unknowns intended for GMRES.  No nodal
// density interpolation is performed during a matrix-vector product.
class NurbsDensitySpace2D {
public:
    using Function = std::function<double(
        int branch,
        double parameter,
        const Eigen::Vector2d& point)>;

    NurbsDensitySpace2D(
        const geometry2d::NurbsBoundaryPanelGeometry2D& geometry,
        int degree,
        double target_spacing,
        std::vector<NurbsDensityContinuity2D> connections,
        NurbsDensityCoordinate2D coordinate =
            NurbsDensityCoordinate2D::PhysicalArclength);

    int degree() const;
    double target_spacing() const;
    int branch_count() const;
    int raw_coefficient_count() const;
    int reduced_coefficient_count() const;
    int constraint_rank() const;
    NurbsDensityCoordinate2D coordinate() const;

    const geometry2d::NurbsBoundaryPanelGeometry2D& geometry() const;
    const std::vector<NurbsDensityContinuity2D>& connections() const;
    const NurbsDensityBranchInfo2D& branch_info(int branch) const;
    const Eigen::MatrixXd& constraint_matrix() const;
    const Eigen::MatrixXd& reduction_matrix() const;

    Eigen::VectorXd expand_reduced_coefficients(
        const Eigen::VectorXd& reduced_coefficients) const;

    NurbsDensityEvaluationRows2D evaluation_rows(
        int branch,
        double parameter) const;
    // Evaluate value/first/second rows by sampling values around the point in
    // local physical arclength.  difference_step_over_span multiplies the
    // mean density-knot span on the owning branch.  Smooth closed components
    // wrap periodically; feature-separated branches use one-sided formulas
    // when a centered stencil would leave the owning branch.
    NurbsDensityEvaluationRows2D sampled_finite_difference_rows(
        int branch,
        double parameter,
        double difference_step_over_span) const;
    // Evaluate neighbouring values in the native NURBS parameter and apply
    // the one-dimensional covariant conversion
    //
    //   rho_s  = rho_xi / J,
    //   rho_ss = rho_xixi / J^2 - J_xi rho_xi / J^3.
    //
    // The physical target step is difference_step_over_span times the mean
    // density span and is mapped locally to a xi step with the exact J.
    NurbsDensityEvaluationRows2D
    covariant_parameter_finite_difference_rows(
        int branch,
        double parameter,
        double difference_step_over_span) const;
    NurbsDensityJet2D evaluate(
        int branch,
        double parameter,
        const Eigen::VectorXd& reduced_coefficients) const;
    NurbsDensityJet2D evaluate_sampled_finite_difference(
        int branch,
        double parameter,
        const Eigen::VectorXd& reduced_coefficients,
        double difference_step_over_span) const;
    NurbsDensityJet2D evaluate_covariant_parameter_finite_difference(
        int branch,
        double parameter,
        const Eigen::VectorXd& reduced_coefficients,
        double difference_step_over_span) const;
    NurbsDensityJet2D evaluate(
        NurbsDensityLocation2D location,
        const Eigen::VectorXd& reduced_coefficients) const;

    // Resolve the exact NURBS branch/parameter retained by a compatibility
    // P2 crossing owner. Explicitly represented corner-gap crossings are
    // projected one-sidedly to their owning branch; unresolved fallbacks are
    // rejected.
    bool can_resolve_crossing(const P2CrossingOwner2D& crossing) const;
    NurbsDensityLocation2D resolve_crossing(
        const P2CrossingOwner2D& crossing) const;
    NurbsDensityJet2D evaluate_crossing(
        const P2CrossingOwner2D& crossing,
        const Eigen::VectorXd& reduced_coefficients) const;
    NurbsDensityJet2D evaluate_crossing_sampled_finite_difference(
        const P2CrossingOwner2D& crossing,
        const Eigen::VectorXd& reduced_coefficients,
        double difference_step_over_span) const;
    NurbsDensityJet2D
    evaluate_crossing_covariant_parameter_finite_difference(
        const P2CrossingOwner2D& crossing,
        const Eigen::VectorXd& reduced_coefficients,
        double difference_step_over_span) const;
    LaplaceCrossingTraceJet2D evaluate_crossing_trace(
        const P2CrossingOwner2D& crossing,
        const Eigen::VectorXd& reduced_coefficients,
        NurbsDensityTraceRole2D role) const;

    // Weighted least-squares fit into the reduced coefficient space.  This is
    // intended for fixed boundary data and manufactured solutions, not for a
    // GMRES matvec.
    Eigen::VectorXd fit_samples(
        const std::vector<NurbsDensitySample2D>& samples,
        double relative_rank_tolerance = 1.0e-12) const;
    Eigen::VectorXd fit_function(
        const Function& function,
        double relative_rank_tolerance = 1.0e-12) const;
    Eigen::VectorXd fit_parameterized_point_samples(
        const Eigen::VectorXd& values,
        double relative_rank_tolerance = 1.0e-12) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

// Complete P2 Cauchy polynomial assembled directly from two coefficient
// spaces at the same true crossing. forcing_jump is [f] in the repository
// convention (-Delta + alpha) C = [f].
LaplaceP2CrossingLocalPolynomial2D
build_nurbs_density_local_polynomial_2d(
    const NurbsDensitySpace2D& phi_space,
    const Eigen::VectorXd& phi_coefficients,
    const NurbsDensitySpace2D& psi_space,
    const Eigen::VectorXd& psi_coefficients,
    const P2CrossingOwner2D& crossing,
    double forcing_jump,
    double alpha);

} // namespace kfbim
