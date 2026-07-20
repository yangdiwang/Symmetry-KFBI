#pragma once

#include "nurbs_basis.hpp"
#include "nurbs_utils.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfbim::geometry {

template <int Dim>
class NurbsCurve {
public:
    using Vector = Eigen::Matrix<double, Dim, 1>;
    using ControlPointVector =
        std::vector<Vector, Eigen::aligned_allocator<Vector>>;
    using ProjectionResult = NurbsProjectionResult<Dim>;

    NurbsCurve(NurbsBasis1D basis,
               ControlPointVector control_points,
               std::vector<double> weights)
        : basis_(std::move(basis))
        , control_points_(std::move(control_points))
        , weights_(std::move(weights))
    {
        validate();
    }

    [[nodiscard]] const NurbsBasis1D& basis() const noexcept { return basis_; }

    [[nodiscard]] const ControlPointVector& control_points() const noexcept
    {
        return control_points_;
    }

    [[nodiscard]] const std::vector<double>& weights() const noexcept
    {
        return weights_;
    }

    [[nodiscard]] double domain_start() const noexcept
    {
        return basis_.domain_start();
    }

    [[nodiscard]] double domain_end() const noexcept
    {
        return basis_.domain_end();
    }

    [[nodiscard]] Vector evaluate(double t) const
    {
        const double parameter = clamped_parameter(t);
        const auto active = basis_.active_basis_indices(parameter);
        const auto values = basis_.evaluate_nonzero(parameter);

        Vector numerator = Vector::Zero();
        double denominator = 0.0;
        for (std::size_t local = 0; local < active.size(); ++local) {
            const int i = active[local];
            const double weighted_basis =
                values[local] * weights_[static_cast<std::size_t>(i)];
            numerator += weighted_basis * control_points_[static_cast<std::size_t>(i)];
            denominator += weighted_basis;
        }

        if (denominator <= basis_.tolerance())
            throw std::runtime_error("NURBS curve denominator is zero or near zero");
        return numerator / denominator;
    }

    [[nodiscard]] double rational_basis_value(int i, double t) const
    {
        if (i < 0 || i >= basis_.num_basis_functions())
            throw std::invalid_argument("rational basis index is out of range");

        const double parameter = clamped_parameter(t);
        const auto active = basis_.active_basis_indices(parameter);
        const auto values = basis_.evaluate_nonzero(parameter);

        double numerator = 0.0;
        double denominator = 0.0;
        for (std::size_t local = 0; local < active.size(); ++local) {
            const int active_index = active[local];
            const double weighted_basis =
                values[local] * weights_[static_cast<std::size_t>(active_index)];
            if (active_index == i)
                numerator = weighted_basis;
            denominator += weighted_basis;
        }

        if (denominator <= basis_.tolerance())
            throw std::runtime_error("NURBS curve denominator is zero or near zero");
        return numerator / denominator;
    }

    [[nodiscard]] std::vector<int> active_basis_indices(double t) const
    {
        return basis_.active_basis_indices(clamped_parameter(t));
    }

    [[nodiscard]] Vector derivative(double t) const
    {
        const double parameter = clamped_parameter(t);
        const auto active = basis_.active_basis_indices(parameter);
        const auto values = basis_.evaluate_nonzero(parameter);
        const auto derivatives = basis_.evaluate_nonzero_first_derivatives(parameter);

        Vector numerator = Vector::Zero();
        Vector numerator_derivative = Vector::Zero();
        double denominator = 0.0;
        double denominator_derivative = 0.0;

        for (std::size_t local = 0; local < active.size(); ++local) {
            const int i = active[local];
            const double weight = weights_[static_cast<std::size_t>(i)];
            const Vector& control_point =
                control_points_[static_cast<std::size_t>(i)];

            const double weighted_basis = values[local] * weight;
            const double weighted_basis_derivative = derivatives[local] * weight;

            numerator += weighted_basis * control_point;
            numerator_derivative += weighted_basis_derivative * control_point;
            denominator += weighted_basis;
            denominator_derivative += weighted_basis_derivative;
        }

        if (denominator <= basis_.tolerance())
            throw std::runtime_error("NURBS curve denominator is zero or near zero");

        return (numerator_derivative * denominator
              - numerator * denominator_derivative)
             / (denominator * denominator);
    }

    [[nodiscard]] Vector tangent(double t) const
    {
        return normalize_safe<Dim>(derivative(t), basis_.tolerance());
    }

    [[nodiscard]] ProjectionResult project(const Vector& x) const
    {
        require_finite<Dim>(x, "projection point");

        const auto samples = projection_samples();
        int best_index = 0;
        double best_distance_squared = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double distance_squared = squared_distance_to(x, samples[i]);
            if (distance_squared < best_distance_squared) {
                best_distance_squared = distance_squared;
                best_index = static_cast<int>(i);
            }
        }

        double lower =
            samples[static_cast<std::size_t>(std::max(0, best_index - 1))];
        double upper = samples[static_cast<std::size_t>(
            std::min(static_cast<int>(samples.size()) - 1, best_index + 1))];
        if (upper < lower)
            std::swap(lower, upper);

        ProjectionResult result;
        result.coarse_samples = static_cast<int>(samples.size());
        if (upper - lower <= basis_.tolerance()) {
            result.parameter = clamped_parameter(samples[static_cast<std::size_t>(best_index)]);
            result.u = result.parameter;
            result.point = evaluate(result.parameter);
            result.distance = (result.point - x).norm();
            result.converged = true;
            result.iterations = 0;
            result.status = NurbsProjectionStatus::ConvergedBracketSmall;
            result.residual_norm = result.distance;
            result.message =
                "curve projection bracket collapsed during coarse initialization";
            return result;
        }

        const double parameter_tolerance =
            64.0 * basis_.tolerance() * std::max(1.0, domain_end() - domain_start());
        constexpr double inverse_golden_ratio = 0.6180339887498948482;
        constexpr int max_iterations = 96;

        double a = lower;
        double b = upper;
        double c = b - inverse_golden_ratio * (b - a);
        double d = a + inverse_golden_ratio * (b - a);
        double fc = squared_distance_to(x, c);
        double fd = squared_distance_to(x, d);

        int iterations = 0;
        for (; iterations < max_iterations && b - a > parameter_tolerance;
             ++iterations) {
            if (fc <= fd) {
                b = d;
                d = c;
                fd = fc;
                c = b - inverse_golden_ratio * (b - a);
                fc = squared_distance_to(x, c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + inverse_golden_ratio * (b - a);
                fd = squared_distance_to(x, d);
            }
        }

        result.parameter = clamped_parameter(0.5 * (a + b));
        result.u = result.parameter;
        result.point = evaluate(result.parameter);
        result.distance = (result.point - x).norm();
        result.converged = (b - a <= parameter_tolerance);
        result.iterations = iterations;
        result.status = result.converged
            ? NurbsProjectionStatus::ConvergedBracketSmall
            : NurbsProjectionStatus::MaxIterationsReached;
        result.residual_norm = result.distance;
        result.step_norm = b - a;
        result.message = result.converged
            ? "curve projection converged by bracket tolerance"
            : "curve projection reached maximum golden-section iterations";
        return result;
    }

    [[nodiscard]] static NurbsCurve make_line_segment(const Vector& a,
                                                      const Vector& b)
    {
        ControlPointVector control_points;
        control_points.reserve(2);
        control_points.push_back(a);
        control_points.push_back(b);
        return NurbsCurve(
            NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
            std::move(control_points),
            {1.0, 1.0});
    }

private:
    NurbsBasis1D basis_;
    ControlPointVector control_points_;
    std::vector<double> weights_;

    void validate() const
    {
        if (control_points_.empty())
            throw std::invalid_argument(
                "NURBS curve must have at least one control point");
        if (control_points_.size() != weights_.size())
            throw std::invalid_argument(
                "NURBS curve control point and weight sizes must match");
        if (static_cast<int>(control_points_.size()) !=
            basis_.num_basis_functions()) {
            throw std::invalid_argument(
                "NURBS curve control point count must match basis function count");
        }

        for (const Vector& control_point : control_points_)
            require_finite<Dim>(control_point, "NURBS curve control point");
        for (const double weight : weights_) {
            require_finite(weight, "NURBS curve weight");
            if (weight <= 0.0)
                throw std::invalid_argument("NURBS curve weights must be positive");
        }
    }

    [[nodiscard]] double clamped_parameter(double t) const
    {
        require_finite(t, "curve parameter");
        return std::clamp(t, domain_start(), domain_end());
    }

    [[nodiscard]] double squared_distance_to(const Vector& x,
                                             double        parameter) const
    {
        const Vector displacement = evaluate(parameter) - x;
        return displacement.squaredNorm();
    }

    [[nodiscard]] std::vector<double> projection_samples() const
    {
        constexpr int samples_per_span = 16;
        std::vector<double> samples;
        samples.push_back(domain_start());

        const auto& knots = basis_.knots();
        for (std::size_t i = 1; i < knots.size(); ++i) {
            const double span_start =
                std::clamp(knots[i - 1], domain_start(), domain_end());
            const double span_end =
                std::clamp(knots[i], domain_start(), domain_end());
            if (span_end <= span_start + basis_.tolerance())
                continue;
            for (int sample = 1; sample <= samples_per_span; ++sample) {
                const double alpha =
                    static_cast<double>(sample) / samples_per_span;
                samples.push_back((1.0 - alpha) * span_start + alpha * span_end);
            }
        }

        if (samples.back() < domain_end() - basis_.tolerance())
            samples.push_back(domain_end());

        std::sort(samples.begin(), samples.end());
        samples.erase(
            std::unique(
                samples.begin(),
                samples.end(),
                [this](double lhs, double rhs) {
                    return std::abs(lhs - rhs) <= basis_.tolerance();
                }),
            samples.end());
        return samples;
    }
};

} // namespace kfbim::geometry
