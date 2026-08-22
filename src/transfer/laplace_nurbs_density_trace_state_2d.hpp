#pragma once

#include <memory>

#include <Eigen/Dense>

#include "nurbs_density_space_2d.hpp"

namespace kfbim {

// Immutable per-application density state for the direct-coefficient NURBS
// route.  Unlike LaplaceArcLengthBSplineTraceState2D, construction performs
// no interpolation solve: the stored vectors already are the reduced spline
// coefficients used by GMRES (or by a fitted, fixed boundary datum).
class LaplaceNurbsDensityTraceState2D {
public:
    LaplaceNurbsDensityTraceState2D(
        std::shared_ptr<const NurbsDensitySpace2D> phi_space,
        Eigen::VectorXd phi_reduced_coefficients,
        std::shared_ptr<const NurbsDensitySpace2D> psi_space,
        Eigen::VectorXd psi_reduced_coefficients,
        LaplaceCrossingTraceStencil2D trace_stencil,
        NurbsDensityDerivativeScheme2D derivative_scheme =
            NurbsDensityDerivativeScheme2D::AnalyticBasis,
        double difference_step_over_span = 0.5);

    LaplaceCrossingTraceStencil2D trace_stencil() const noexcept
    {
        return trace_stencil_;
    }
    NurbsDensityDerivativeScheme2D derivative_scheme() const noexcept
    {
        return derivative_scheme_;
    }
    double difference_step_over_span() const noexcept
    {
        return difference_step_over_span_;
    }

    const std::shared_ptr<const NurbsDensitySpace2D>& phi_space() const
    {
        return phi_space_;
    }
    const std::shared_ptr<const NurbsDensitySpace2D>& psi_space() const
    {
        return psi_space_;
    }
    const Eigen::VectorXd& phi_reduced_coefficients() const
    {
        return phi_reduced_coefficients_;
    }
    const Eigen::VectorXd& psi_reduced_coefficients() const
    {
        return psi_reduced_coefficients_;
    }

    const geometry2d::NurbsBoundaryPanelGeometry2D& geometry() const;

    bool can_evaluate(const P2CrossingOwner2D& crossing) const noexcept;

    LaplaceCrossingTraceJet2D evaluate(
        const P2CrossingOwner2D& crossing) const;

    // Evaluate the same immutable coefficient state at an exact NURBS
    // branch/parameter location.  This is the trace-reference counterpart of
    // evaluate(crossing): it does not require the requested point to coincide
    // with a compatibility P2 node or grid-edge crossing.
    LaplaceCrossingTraceJet2D evaluate(
        NurbsDensityLocation2D location) const;

    // Build the complete crossing-local P2 Cauchy polynomial. rhs_jump is
    // sampled at the parameterized Interface2D compatibility points. Empty
    // selects the homogeneous jump [f]=0. The interpolation is used only for
    // the scalar PDE closure. Phi/psi values come directly from the stored
    // reduced coefficients; tangential derivatives use analytic basis rows,
    // physical-arclength differences, or NURBS-parameter differences followed
    // by exact metric conversion according to the state's derivative scheme.
    LaplaceP2CrossingLocalPolynomial2D build_local_polynomial(
        const P2CrossingOwner2D& crossing,
        const Eigen::VectorXd& rhs_jump,
        double alpha) const;

    LaplaceP2CrossingLocalPolynomial2D build_local_polynomial_at_location(
        NurbsDensityLocation2D location,
        const Eigen::VectorXd& rhs_jump,
        double alpha) const;

private:
    const NurbsDensitySpace2D& reference_space() const;
    double forcing_jump_at_crossing(
        const P2CrossingOwner2D& crossing,
        const Eigen::VectorXd& rhs_jump) const;
    double forcing_jump_at_location(
        NurbsDensityLocation2D location,
        const Eigen::VectorXd& rhs_jump) const;

    std::shared_ptr<const NurbsDensitySpace2D> phi_space_;
    Eigen::VectorXd phi_reduced_coefficients_;
    std::shared_ptr<const NurbsDensitySpace2D> psi_space_;
    Eigen::VectorXd psi_reduced_coefficients_;
    LaplaceCrossingTraceStencil2D trace_stencil_ =
        LaplaceCrossingTraceStencil2D::RhsOnly;
    NurbsDensityDerivativeScheme2D derivative_scheme_ =
        NurbsDensityDerivativeScheme2D::AnalyticBasis;
    double difference_step_over_span_ = 0.5;
};

} // namespace kfbim
