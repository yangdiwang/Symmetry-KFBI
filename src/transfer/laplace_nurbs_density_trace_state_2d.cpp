#include "laplace_nurbs_density_trace_state_2d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfbim {
namespace {

void validate_density_block(
    const char* name,
    const std::shared_ptr<const NurbsDensitySpace2D>& space,
    const Eigen::VectorXd& coefficients,
    int required_degree,
    bool required)
{
    if (!space) {
        if (required || coefficients.size() != 0) {
            throw std::invalid_argument(
                std::string("direct NURBS density state has no ") + name
                + " space");
        }
        return;
    }
    if (space->degree() != required_degree) {
        throw std::invalid_argument(
            std::string("direct NURBS density state requires degree ")
            + std::to_string(required_degree) + " for " + name);
    }
    if (coefficients.size() != space->reduced_coefficient_count()
        || !coefficients.allFinite()) {
        throw std::invalid_argument(
            std::string("direct NURBS density state has invalid ") + name
            + " reduced coefficients");
    }
}

struct ParameterSample {
    double parameter = 0.0;
    int point = -1;
};

NurbsDensityJet2D evaluate_density_jet(
    const NurbsDensitySpace2D& space,
    const P2CrossingOwner2D& crossing,
    const Eigen::VectorXd& coefficients,
    NurbsDensityDerivativeScheme2D scheme,
    double difference_step_over_span)
{
    switch (scheme) {
    case NurbsDensityDerivativeScheme2D::AnalyticBasis:
        return space.evaluate_crossing(crossing, coefficients);
    case NurbsDensityDerivativeScheme2D::SampledFiniteDifference:
        return space.evaluate_crossing_sampled_finite_difference(
            crossing, coefficients, difference_step_over_span);
    case NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference:
        return space.evaluate_crossing_covariant_parameter_finite_difference(
            crossing, coefficients, difference_step_over_span);
    }
    throw std::invalid_argument(
        "direct NURBS density state has an invalid derivative scheme");
}

NurbsDensityJet2D evaluate_density_jet(
    const NurbsDensitySpace2D& space,
    NurbsDensityLocation2D location,
    const Eigen::VectorXd& coefficients,
    NurbsDensityDerivativeScheme2D scheme,
    double difference_step_over_span)
{
    switch (scheme) {
    case NurbsDensityDerivativeScheme2D::AnalyticBasis:
        return space.evaluate(location, coefficients);
    case NurbsDensityDerivativeScheme2D::SampledFiniteDifference:
        return space.evaluate_sampled_finite_difference(
            location.branch,
            location.parameter,
            coefficients,
            difference_step_over_span);
    case NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference:
        return space.evaluate_covariant_parameter_finite_difference(
            location.branch,
            location.parameter,
            coefficients,
            difference_step_over_span);
    }
    throw std::invalid_argument(
        "direct NURBS density state has an invalid derivative scheme");
}

} // namespace

LaplaceNurbsDensityTraceState2D::LaplaceNurbsDensityTraceState2D(
    std::shared_ptr<const NurbsDensitySpace2D> phi_space,
    Eigen::VectorXd phi_reduced_coefficients,
    std::shared_ptr<const NurbsDensitySpace2D> psi_space,
    Eigen::VectorXd psi_reduced_coefficients,
    LaplaceCrossingTraceStencil2D trace_stencil,
    NurbsDensityDerivativeScheme2D derivative_scheme,
    double difference_step_over_span)
    : phi_space_(std::move(phi_space))
    , phi_reduced_coefficients_(std::move(phi_reduced_coefficients))
    , psi_space_(std::move(psi_space))
    , psi_reduced_coefficients_(std::move(psi_reduced_coefficients))
    , trace_stencil_(trace_stencil)
    , derivative_scheme_(derivative_scheme)
    , difference_step_over_span_(difference_step_over_span)
{
    if (!is_valid_crossing_trace_stencil(trace_stencil_)) {
        throw std::invalid_argument(
            "direct NURBS density state has an invalid trace stencil");
    }
    validate_density_block(
        "phi",
        phi_space_,
        phi_reduced_coefficients_,
        3,
        crossing_trace_uses_phi_p3(trace_stencil_));
    validate_density_block(
        "psi",
        psi_space_,
        psi_reduced_coefficients_,
        2,
        crossing_trace_uses_psi_p2(trace_stencil_));
    if (phi_space_ && psi_space_
        && &phi_space_->geometry() != &psi_space_->geometry()) {
        throw std::invalid_argument(
            "direct NURBS phi/psi spaces must share one geometry provider");
    }
    if (!(difference_step_over_span_ > 0.0)
        || !std::isfinite(difference_step_over_span_)) {
        throw std::invalid_argument(
            "direct NURBS density difference spacing must be finite and positive");
    }
    if (derivative_scheme_
            == NurbsDensityDerivativeScheme2D::SampledFiniteDifference) {
        if ((phi_space_ && phi_space_->coordinate()
                 != NurbsDensityCoordinate2D::PhysicalArclength)
            || (psi_space_ && psi_space_->coordinate()
                 != NurbsDensityCoordinate2D::PhysicalArclength)) {
            throw std::invalid_argument(
                "sampled NURBS density differences require physical-arclength spaces");
        }
    }
    if (derivative_scheme_
            == NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference) {
        if ((phi_space_ && phi_space_->coordinate()
                 != NurbsDensityCoordinate2D::LegacyNurbsParameter)
            || (psi_space_ && psi_space_->coordinate()
                 != NurbsDensityCoordinate2D::LegacyNurbsParameter)) {
            throw std::invalid_argument(
                "covariant parameter differences require NURBS-parameter spaces");
        }
    }
    (void)reference_space();
}

const NurbsDensitySpace2D&
LaplaceNurbsDensityTraceState2D::reference_space() const
{
    if (crossing_trace_uses_phi_p3(trace_stencil_) && phi_space_)
        return *phi_space_;
    if (crossing_trace_uses_psi_p2(trace_stencil_) && psi_space_)
        return *psi_space_;
    if (phi_space_)
        return *phi_space_;
    if (psi_space_)
        return *psi_space_;
    throw std::invalid_argument(
        "direct NURBS density state has no geometry-bearing space");
}

const geometry2d::NurbsBoundaryPanelGeometry2D&
LaplaceNurbsDensityTraceState2D::geometry() const
{
    return reference_space().geometry();
}

bool LaplaceNurbsDensityTraceState2D::can_evaluate(
    const P2CrossingOwner2D& crossing) const noexcept
{
    try {
        const NurbsDensityLocation2D reference =
            reference_space().resolve_crossing(crossing);
        if (crossing_trace_uses_phi_p3(trace_stencil_)) {
            const NurbsDensityLocation2D phi =
                phi_space_->resolve_crossing(crossing);
            if (phi.branch != reference.branch
                || std::abs(phi.parameter - reference.parameter) > 1.0e-12
                       * std::max(1.0, std::abs(reference.parameter))) {
                return false;
            }
        }
        if (crossing_trace_uses_psi_p2(trace_stencil_)) {
            const NurbsDensityLocation2D psi =
                psi_space_->resolve_crossing(crossing);
            if (psi.branch != reference.branch
                || std::abs(psi.parameter - reference.parameter) > 1.0e-12
                       * std::max(1.0, std::abs(reference.parameter))) {
                return false;
            }
        }
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

LaplaceCrossingTraceJet2D LaplaceNurbsDensityTraceState2D::evaluate(
    const P2CrossingOwner2D& crossing) const
{
    if (!can_evaluate(crossing)) {
        throw std::invalid_argument(
            "direct NURBS density state cannot evaluate this crossing");
    }

    LaplaceCrossingTraceJet2D trace;
    if (crossing_trace_uses_phi_p3(trace_stencil_)) {
        const NurbsDensityJet2D phi = evaluate_density_jet(
            *phi_space_,
            crossing,
            phi_reduced_coefficients_,
            derivative_scheme_,
            difference_step_over_span_);
        trace.value = phi.value;
        trace.value_tangent_derivative = phi.arclength_first;
        trace.value_tangent_second_derivative = phi.arclength_second;
    }
    if (crossing_trace_uses_psi_p2(trace_stencil_)) {
        const NurbsDensityJet2D psi = evaluate_density_jet(
            *psi_space_,
            crossing,
            psi_reduced_coefficients_,
            derivative_scheme_,
            difference_step_over_span_);
        trace.normal_value = psi.value;
        trace.normal_tangent_derivative = psi.arclength_first;
    }
    return trace;
}

LaplaceCrossingTraceJet2D LaplaceNurbsDensityTraceState2D::evaluate(
    NurbsDensityLocation2D location) const
{
    const auto interval = geometry().span_interval(location.branch);
    const double scale = std::max(
        1.0, std::abs(interval.second - interval.first));
    if (!std::isfinite(location.parameter)
        || location.parameter < interval.first - 1.0e-13 * scale
        || location.parameter > interval.second + 1.0e-13 * scale) {
        throw std::invalid_argument(
            "direct NURBS density location is outside its branch");
    }
    location.parameter = std::clamp(
        location.parameter, interval.first, interval.second);

    LaplaceCrossingTraceJet2D trace;
    if (crossing_trace_uses_phi_p3(trace_stencil_)) {
        const NurbsDensityJet2D phi = evaluate_density_jet(
            *phi_space_,
            location,
            phi_reduced_coefficients_,
            derivative_scheme_,
            difference_step_over_span_);
        trace.value = phi.value;
        trace.value_tangent_derivative = phi.arclength_first;
        trace.value_tangent_second_derivative = phi.arclength_second;
    }
    if (crossing_trace_uses_psi_p2(trace_stencil_)) {
        const NurbsDensityJet2D psi = evaluate_density_jet(
            *psi_space_,
            location,
            psi_reduced_coefficients_,
            derivative_scheme_,
            difference_step_over_span_);
        trace.normal_value = psi.value;
        trace.normal_tangent_derivative = psi.arclength_first;
    }
    return trace;
}

double LaplaceNurbsDensityTraceState2D::forcing_jump_at_crossing(
    const P2CrossingOwner2D& crossing,
    const Eigen::VectorXd& rhs_jump) const
{
    if (rhs_jump.size() == 0)
        return 0.0;
    const auto& boundary = geometry();
    if (rhs_jump.size() != boundary.num_parameterized_points()
        || !rhs_jump.allFinite()) {
        throw std::invalid_argument(
            "direct NURBS density forcing samples must match the geometry points");
    }
    if (rhs_jump.cwiseAbs().maxCoeff() == 0.0)
        return 0.0;

    const NurbsDensityLocation2D location =
        reference_space().resolve_crossing(crossing);
    return forcing_jump_at_location(location, rhs_jump);
}

double LaplaceNurbsDensityTraceState2D::forcing_jump_at_location(
    NurbsDensityLocation2D location,
    const Eigen::VectorXd& rhs_jump) const
{
    if (rhs_jump.size() == 0)
        return 0.0;
    const auto& boundary = geometry();
    if (rhs_jump.size() != boundary.num_parameterized_points()
        || !rhs_jump.allFinite()) {
        throw std::invalid_argument(
            "direct NURBS density forcing samples must match the geometry points");
    }
    if (rhs_jump.cwiseAbs().maxCoeff() == 0.0)
        return 0.0;
    const auto interval = boundary.span_interval(location.branch);
    location.parameter = std::clamp(
        location.parameter, interval.first, interval.second);
    std::vector<ParameterSample> candidates;
    candidates.reserve(
        static_cast<std::size_t>(boundary.num_parameterized_points()));
    for (int point = 0; point < boundary.num_parameterized_points(); ++point) {
        if (boundary.point_span(point) == location.branch) {
            candidates.push_back(
                {boundary.point_parameter(point), point});
        }
    }
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [&](const ParameterSample& lhs, const ParameterSample& rhs) {
            return std::abs(lhs.parameter - location.parameter)
                 < std::abs(rhs.parameter - location.parameter);
        });

    std::vector<ParameterSample> samples;
    samples.reserve(3);
    const double parameter_scale = std::max(
        1.0,
        std::abs(reference_space().branch_info(location.branch).parameter_end
                 - reference_space().branch_info(location.branch).parameter_start));
    for (const ParameterSample& candidate : candidates) {
        const bool duplicate = std::any_of(
            samples.begin(), samples.end(),
            [&](const ParameterSample& saved) {
                return std::abs(saved.parameter - candidate.parameter)
                    <= 1.0e-14 * parameter_scale;
            });
        if (!duplicate)
            samples.push_back(candidate);
        if (samples.size() == 3)
            break;
    }
    if (samples.empty()) {
        throw std::invalid_argument(
            "direct NURBS density forcing interpolation has no branch samples");
    }
    std::sort(samples.begin(), samples.end(),
              [](const ParameterSample& lhs, const ParameterSample& rhs) {
                  return lhs.parameter < rhs.parameter;
              });

    double value = 0.0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        double basis = 1.0;
        for (std::size_t j = 0; j < samples.size(); ++j) {
            if (j == i)
                continue;
            basis *= (location.parameter - samples[j].parameter)
                   / (samples[i].parameter - samples[j].parameter);
        }
        value += basis * rhs_jump[samples[i].point];
    }
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "direct NURBS density forcing interpolation is nonfinite");
    }
    return value;
}

LaplaceP2CrossingLocalPolynomial2D
LaplaceNurbsDensityTraceState2D::build_local_polynomial(
    const P2CrossingOwner2D& crossing,
    const Eigen::VectorXd& rhs_jump,
    double alpha) const
{
    if (!std::isfinite(alpha) || !can_evaluate(crossing)) {
        throw std::invalid_argument(
            "direct NURBS density local polynomial has invalid input");
    }
    const NurbsDensityLocation2D location =
        reference_space().resolve_crossing(crossing);
    const geometry2d::NurbsBoundaryGeometry2D differential =
        geometry().evaluate_on_span(location.branch, location.parameter);

    LaplaceCrossingGeometryJet2D geometry_jet;
    geometry_jet.center = differential.point;
    geometry_jet.tangent = differential.tangent;
    geometry_jet.normal = differential.normal;
    geometry_jet.curvature = differential.curvature;
    geometry_jet.forcing = forcing_jump_at_crossing(crossing, rhs_jump);
    return build_laplace_crossing_local_polynomial_from_geometry_trace_jet_2d(
        geometry_jet, evaluate(crossing), alpha);
}

LaplaceP2CrossingLocalPolynomial2D
LaplaceNurbsDensityTraceState2D::build_local_polynomial_at_location(
    NurbsDensityLocation2D location,
    const Eigen::VectorXd& rhs_jump,
    double alpha) const
{
    if (!std::isfinite(alpha)) {
        throw std::invalid_argument(
            "direct NURBS density local polynomial has invalid input");
    }
    const geometry2d::NurbsBoundaryGeometry2D differential =
        geometry().evaluate_on_span(location.branch, location.parameter);

    LaplaceCrossingGeometryJet2D geometry_jet;
    geometry_jet.center = differential.point;
    geometry_jet.tangent = differential.tangent;
    geometry_jet.normal = differential.normal;
    geometry_jet.curvature = differential.curvature;
    geometry_jet.forcing = forcing_jump_at_location(location, rhs_jump);
    return build_laplace_crossing_local_polynomial_from_geometry_trace_jet_2d(
        geometry_jet, evaluate(location), alpha);
}

} // namespace kfbim
