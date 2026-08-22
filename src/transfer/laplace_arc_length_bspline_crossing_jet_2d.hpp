#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "laplace_crossing_local_polynomial_2d.hpp"
#include "laplace_crossing_trace_stencil_2d.hpp"

namespace kfbim {

// Per-application ALS-CJ density state.  Geometry, arclength coordinates,
// periodic spline matrices, and their factorizations live in the immutable
// plan; this state contains only coefficients fitted from the current GMRES
// jump vectors. The historical by_component field names index spline plans;
// one cornered component may contribute several open branch plans.
struct LaplaceArcLengthBSplineTraceState2D {
    LaplaceCrossingTraceStencil2D trace_stencil =
        LaplaceCrossingTraceStencil2D::RhsOnly;
    std::vector<Eigen::VectorXd> phi_coefficients_by_component;
    std::vector<Eigen::VectorXd> psi_coefficients_by_component;
};

// Arc-Length B-Spline Crossing Jet (ALS-CJ).
//
// A smooth closed component is reconstructed in one periodic physical-
// arclength coordinate. Cornered or open components are split into maximal
// smooth panel chains and reconstructed with independent open clamped
// splines, so value/derivative continuity is enforced along each branch but
// never across a physical corner. Raw interface DOFs interpolate a cubic C2
// spline for phi and a quadratic C1 spline for psi. Crossing evaluation
// therefore remains continuous when the owner panel or active B-spline span
// changes inside a smooth branch.
class LaplaceArcLengthBSplineCrossingJetPlan2D {
public:
    explicit LaplaceArcLengthBSplineCrossingJetPlan2D(
        const Interface2D& iface,
        LaplaceCrossingJetScheme2D scheme =
            LaplaceCrossingJetScheme2D::ArcLengthBSplineCrossingJet);

    bool can_evaluate(
        const P2CrossingOwner2D& crossing,
        LaplaceCrossingTraceStencil2D trace_stencil) const;

    LaplaceArcLengthBSplineTraceState2D fit(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        LaplaceCrossingTraceStencil2D trace_stencil) const;

    LaplaceCrossingTraceJet2D evaluate(
        const P2CrossingOwner2D& crossing,
        const LaplaceArcLengthBSplineTraceState2D& state) const;

    // Build the complete spatial Cauchy polynomial using the same crossing
    // representation as evaluate().  In NSP-CJ mode this queries exact NURBS
    // geometry at crossing.crossing_point, including explicit corner-gap
    // intersections; in ALS-CJ mode it delegates to the retained panel
    // geometry closure.
    LaplaceP2CrossingLocalPolynomial2D build_local_polynomial(
        const P2CrossingOwner2D& crossing,
        const LaplaceArcLengthBSplineTraceState2D& state,
        const Eigen::VectorXd& rhs_jump,
        double alpha) const;

    bool uses_nurbs_same_parameter() const;
    int nurbs_span_count() const;

    int smooth_component_count() const;
    int periodic_component_count() const;
    int open_branch_count() const;
    int total_component_count() const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace kfbim
