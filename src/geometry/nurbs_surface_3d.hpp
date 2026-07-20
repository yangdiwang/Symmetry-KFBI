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

namespace kfbim::geometry3d {

struct NurbsSurfaceDerivatives3D {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d du = Eigen::Vector3d::Zero();
    Eigen::Vector3d dv = Eigen::Vector3d::Zero();
};

class NurbsSurfacePatch3D {
public:
    using ProjectionResult = kfbim::geometry::NurbsProjectionResult3D;

    NurbsSurfacePatch3D(kfbim::geometry::NurbsBasis1D basis_u,
                        kfbim::geometry::NurbsBasis1D basis_v,
                        std::vector<std::vector<Eigen::Vector3d>> control_net,
                        std::vector<std::vector<double>> weights)
        : basis_u_(std::move(basis_u))
        , basis_v_(std::move(basis_v))
        , control_net_(std::move(control_net))
        , weights_(std::move(weights))
    {
        validate();
    }

    [[nodiscard]] const kfbim::geometry::NurbsBasis1D& basis_u() const noexcept
    {
        return basis_u_;
    }

    [[nodiscard]] const kfbim::geometry::NurbsBasis1D& basis_v() const noexcept
    {
        return basis_v_;
    }

    [[nodiscard]] const std::vector<std::vector<Eigen::Vector3d>>&
    control_net() const noexcept
    {
        return control_net_;
    }

    [[nodiscard]] const std::vector<std::vector<double>>& weights() const noexcept
    {
        return weights_;
    }

    [[nodiscard]] double domain_start_u() const noexcept
    {
        return basis_u_.domain_start();
    }

    [[nodiscard]] double domain_end_u() const noexcept
    {
        return basis_u_.domain_end();
    }

    [[nodiscard]] double domain_start_v() const noexcept
    {
        return basis_v_.domain_start();
    }

    [[nodiscard]] double domain_end_v() const noexcept
    {
        return basis_v_.domain_end();
    }

    [[nodiscard]] Eigen::Vector3d evaluate(double u, double v) const
    {
        const double parameter_u = clamped_u(u);
        const double parameter_v = clamped_v(v);
        const auto active_u = basis_u_.active_basis_indices(parameter_u);
        const auto active_v = basis_v_.active_basis_indices(parameter_v);
        const auto values_u = basis_u_.evaluate_nonzero(parameter_u);
        const auto values_v = basis_v_.evaluate_nonzero(parameter_v);

        Eigen::Vector3d numerator = Eigen::Vector3d::Zero();
        double denominator = 0.0;
        for (std::size_t local_u = 0; local_u < active_u.size(); ++local_u) {
            const int i = active_u[local_u];
            for (std::size_t local_v = 0; local_v < active_v.size(); ++local_v) {
                const int j = active_v[local_v];
                const double weighted_basis =
                    values_u[local_u] * values_v[local_v] * weight(i, j);
                numerator += weighted_basis * point(i, j);
                denominator += weighted_basis;
            }
        }

        if (denominator <= tolerance())
            throw std::runtime_error("NURBS surface denominator is zero or near zero");
        return numerator / denominator;
    }

    [[nodiscard]] NurbsSurfaceDerivatives3D evaluate_with_derivatives(
        double u,
        double v) const
    {
        const double parameter_u = clamped_u(u);
        const double parameter_v = clamped_v(v);
        try {
            const NurbsSurfaceDerivatives3D derivatives =
                analytic_evaluation_and_partials(parameter_u, parameter_v);
            if (derivatives.point.allFinite() &&
                derivatives.du.allFinite() &&
                derivatives.dv.allFinite()) {
                return derivatives;
            }
        } catch (const std::exception&) {
        }

        NurbsSurfaceDerivatives3D derivatives;
        derivatives.point = evaluate(parameter_u, parameter_v);
        derivatives.du = finite_difference_partial(parameter_u, parameter_v, true);
        derivatives.dv = finite_difference_partial(parameter_u, parameter_v, false);
        return derivatives;
    }

    [[nodiscard]] double rational_basis_value(int i,
                                              int j,
                                              double u,
                                              double v) const
    {
        check_control_index(i, j);

        const double parameter_u = clamped_u(u);
        const double parameter_v = clamped_v(v);
        const auto active_u = basis_u_.active_basis_indices(parameter_u);
        const auto active_v = basis_v_.active_basis_indices(parameter_v);
        const auto values_u = basis_u_.evaluate_nonzero(parameter_u);
        const auto values_v = basis_v_.evaluate_nonzero(parameter_v);

        double numerator = 0.0;
        double denominator = 0.0;
        for (std::size_t local_u = 0; local_u < active_u.size(); ++local_u) {
            const int active_i = active_u[local_u];
            for (std::size_t local_v = 0; local_v < active_v.size(); ++local_v) {
                const int active_j = active_v[local_v];
                const double weighted_basis =
                    values_u[local_u] * values_v[local_v] * weight(active_i, active_j);
                if (active_i == i && active_j == j)
                    numerator = weighted_basis;
                denominator += weighted_basis;
            }
        }

        if (denominator <= tolerance())
            throw std::runtime_error("NURBS surface denominator is zero or near zero");
        return numerator / denominator;
    }

    [[nodiscard]] std::vector<std::pair<int, int>> active_basis_pairs(
        double u,
        double v) const
    {
        const auto active_u = basis_u_.active_basis_indices(clamped_u(u));
        const auto active_v = basis_v_.active_basis_indices(clamped_v(v));

        std::vector<std::pair<int, int>> pairs;
        pairs.reserve(active_u.size() * active_v.size());
        for (const int i : active_u) {
            for (const int j : active_v)
                pairs.emplace_back(i, j);
        }
        return pairs;
    }

    [[nodiscard]] Eigen::Vector3d du(double u, double v) const
    {
        return evaluate_with_derivatives(u, v).du;
    }

    [[nodiscard]] Eigen::Vector3d dv(double u, double v) const
    {
        return evaluate_with_derivatives(u, v).dv;
    }

    [[nodiscard]] Eigen::Vector3d normal(double u, double v) const
    {
        const NurbsSurfaceDerivatives3D derivatives =
            evaluate_with_derivatives(u, v);
        const Eigen::Vector3d cross = derivatives.du.cross(derivatives.dv);
        const double norm = cross.norm();
        if (!std::isfinite(norm) || norm <= tolerance())
            return Eigen::Vector3d::Zero();
        return cross / norm;
    }

    [[nodiscard]] ProjectionResult project(const Eigen::Vector3d& x) const
    {
        kfbim::geometry::require_finite<3>(x, "projection point");

        const auto samples_u = projection_samples(basis_u_);
        const auto samples_v = projection_samples(basis_v_);

        double best_u = domain_start_u();
        double best_v = domain_start_v();
        double best_distance_squared = std::numeric_limits<double>::infinity();
        for (const double sample_u : samples_u) {
            for (const double sample_v : samples_v) {
                const double distance_squared =
                    squared_distance_to(x, sample_u, sample_v);
                if (distance_squared < best_distance_squared) {
                    best_distance_squared = distance_squared;
                    best_u = sample_u;
                    best_v = sample_v;
                }
            }
        }

        double u = best_u;
        double v = best_v;
        constexpr int max_iterations = 64;
        bool converged = false;
        int iterations = 0;
        double previous_distance_squared = best_distance_squared;
        auto status = kfbim::geometry::NurbsProjectionStatus::MaxIterationsReached;
        std::string message =
            "surface projection reached maximum Gauss-Newton iterations";
        double gradient_norm = std::numeric_limits<double>::infinity();
        double step_norm = std::numeric_limits<double>::infinity();
        double normal_matrix_determinant = 0.0;
        int total_line_search_iterations = 0;
        bool used_coordinate_search = false;

        for (; iterations < max_iterations; ++iterations) {
            const NurbsSurfaceDerivatives3D derivatives =
                evaluate_with_derivatives(u, v);
            const Eigen::Vector3d residual = derivatives.point - x;
            const Eigen::Vector3d tangent_u = derivatives.du;
            const Eigen::Vector3d tangent_v = derivatives.dv;

            Eigen::Matrix2d normal_matrix;
            normal_matrix(0, 0) = tangent_u.dot(tangent_u);
            normal_matrix(0, 1) = tangent_u.dot(tangent_v);
            normal_matrix(1, 0) = normal_matrix(0, 1);
            normal_matrix(1, 1) = tangent_v.dot(tangent_v);

            Eigen::Vector2d rhs;
            rhs(0) = -tangent_u.dot(residual);
            rhs(1) = -tangent_v.dot(residual);
            gradient_norm = rhs.norm();

            Eigen::Vector2d step = Eigen::Vector2d::Zero();
            const double determinant = normal_matrix.determinant();
            normal_matrix_determinant = determinant;
            if (std::isfinite(determinant) &&
                std::abs(determinant) >
                    64.0 * tolerance() * normal_matrix.squaredNorm()) {
                step = normal_matrix.ldlt().solve(rhs);
            } else {
                used_coordinate_search = true;
                const bool improved =
                    coordinate_search_step(x, u, v, previous_distance_squared);
                if (improved)
                    continue;
                status = kfbim::geometry::NurbsProjectionStatus::DegenerateNormalSystem;
                message =
                    "surface projection stopped because the local normal system is degenerate";
                converged = false;
                break;
            }

            if (!step.allFinite()) {
                status = kfbim::geometry::NurbsProjectionStatus::NonFiniteStep;
                message =
                    "surface projection stopped because the Gauss-Newton step is not finite";
                converged = false;
                break;
            }

            step_norm = step.norm();
            if (gradient_norm <= 1.0e-9) {
                status = kfbim::geometry::NurbsProjectionStatus::ConvergedGradientSmall;
                message =
                    "surface projection converged by projected-gradient tolerance";
                converged = true;
                break;
            }

            if (step_norm <= parameter_tolerance()) {
                status = kfbim::geometry::NurbsProjectionStatus::ConvergedStepSmall;
                message = "surface projection converged by parameter-step tolerance";
                converged = true;
                break;
            }

            bool accepted = false;
            double alpha = 1.0;
            for (int line_search = 0; line_search < 16; ++line_search) {
                ++total_line_search_iterations;
                const double trial_u = clamped_u(u + alpha * step(0));
                const double trial_v = clamped_v(v + alpha * step(1));
                const double trial_distance_squared =
                    squared_distance_to(x, trial_u, trial_v);
                if (trial_distance_squared <= previous_distance_squared) {
                    const double parameter_step =
                        std::hypot(trial_u - u, trial_v - v);
                    u = trial_u;
                    v = trial_v;
                    previous_distance_squared = trial_distance_squared;
                    accepted = true;
                    if (parameter_step <= parameter_tolerance()) {
                        status = kfbim::geometry::NurbsProjectionStatus::
                            ConvergedLineSearchStepSmall;
                        message =
                            "surface projection converged after line search by parameter-step tolerance";
                        converged = true;
                    }
                    break;
                }
                alpha *= 0.5;
            }

            if (!accepted) {
                used_coordinate_search = true;
                const bool improved =
                    coordinate_search_step(x, u, v, previous_distance_squared);
                if (!improved) {
                    status = kfbim::geometry::NurbsProjectionStatus::
                        ConvergedCoordinateSearchStalled;
                    message =
                        "surface projection stopped because line search and coordinate search did not improve distance";
                    converged = true;
                    break;
                }
            }

            if (converged)
                break;
        }

        ProjectionResult result;
        result.parameter = u;
        result.u = u;
        result.v = v;
        result.point = evaluate(u, v);
        result.distance = (result.point - x).norm();
        result.converged = converged;
        result.iterations = iterations;
        result.status = status;
        result.residual_norm = result.distance;
        result.gradient_norm = gradient_norm;
        result.step_norm = step_norm;
        result.normal_matrix_determinant = normal_matrix_determinant;
        result.coarse_samples =
            static_cast<int>(samples_u.size() * samples_v.size());
        result.line_search_iterations = total_line_search_iterations;
        result.used_coordinate_search = used_coordinate_search;
        result.message = message;
        return result;
    }

    [[nodiscard]] static NurbsSurfacePatch3D make_bilinear_plane(
        const Eigen::Vector3d& p00,
        const Eigen::Vector3d& p10,
        const Eigen::Vector3d& p01,
        const Eigen::Vector3d& p11)
    {
        return NurbsSurfacePatch3D(
            kfbim::geometry::NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
            kfbim::geometry::NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
            {{p00, p01}, {p10, p11}},
            {{1.0, 1.0}, {1.0, 1.0}});
    }

    [[nodiscard]] static NurbsSurfacePatch3D make_unit_square_xy()
    {
        return make_bilinear_plane(
            Eigen::Vector3d(0.0, 0.0, 0.0),
            Eigen::Vector3d(1.0, 0.0, 0.0),
            Eigen::Vector3d(0.0, 1.0, 0.0),
            Eigen::Vector3d(1.0, 1.0, 0.0));
    }

    [[nodiscard]] static NurbsSurfacePatch3D make_quarter_cylinder_patch(
        double radius,
        double z0,
        double z1)
    {
        kfbim::geometry::require_finite(radius, "radius");
        kfbim::geometry::require_finite(z0, "z0");
        kfbim::geometry::require_finite(z1, "z1");
        if (radius <= 0.0)
            throw std::invalid_argument("quarter cylinder radius must be positive");
        if (std::abs(z1 - z0) <= kfbim::geometry::kNurbsGeometryTolerance)
            throw std::invalid_argument("quarter cylinder height must be nonzero");

        const double middle_weight = std::sqrt(2.0) / 2.0;
        return NurbsSurfacePatch3D(
            kfbim::geometry::NurbsBasis1D(2, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}),
            kfbim::geometry::NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
            {{Eigen::Vector3d(radius, 0.0, z0),
              Eigen::Vector3d(radius, 0.0, z1)},
             {Eigen::Vector3d(radius, radius, z0),
              Eigen::Vector3d(radius, radius, z1)},
             {Eigen::Vector3d(0.0, radius, z0),
              Eigen::Vector3d(0.0, radius, z1)}},
            {{1.0, 1.0}, {middle_weight, middle_weight}, {1.0, 1.0}});
    }

private:
    kfbim::geometry::NurbsBasis1D basis_u_;
    kfbim::geometry::NurbsBasis1D basis_v_;
    std::vector<std::vector<Eigen::Vector3d>> control_net_;
    std::vector<std::vector<double>> weights_;

    void validate() const
    {
        const int num_u = basis_u_.num_basis_functions();
        const int num_v = basis_v_.num_basis_functions();

        if (static_cast<int>(control_net_.size()) != num_u)
            throw std::invalid_argument(
                "NURBS surface control net U dimension must match basis U function count");
        if (static_cast<int>(weights_.size()) != num_u)
            throw std::invalid_argument(
                "NURBS surface weight U dimension must match basis U function count");

        for (int i = 0; i < num_u; ++i) {
            if (static_cast<int>(control_net_[static_cast<std::size_t>(i)].size())
                != num_v) {
                throw std::invalid_argument(
                    "NURBS surface control net V dimension must match basis V function count");
            }
            if (static_cast<int>(weights_[static_cast<std::size_t>(i)].size())
                != num_v) {
                throw std::invalid_argument(
                    "NURBS surface weight V dimension must match basis V function count");
            }

            for (int j = 0; j < num_v; ++j) {
                kfbim::geometry::require_finite<3>(
                    point(i, j), "NURBS surface control point");
                kfbim::geometry::require_finite(
                    weight(i, j), "NURBS surface weight");
                if (weight(i, j) <= 0.0)
                    throw std::invalid_argument(
                        "NURBS surface weights must be positive");
            }
        }
    }

    [[nodiscard]] double tolerance() const noexcept
    {
        return std::max(basis_u_.tolerance(), basis_v_.tolerance());
    }

    [[nodiscard]] double parameter_tolerance() const noexcept
    {
        const double domain_scale = std::max(
            {1.0,
             domain_end_u() - domain_start_u(),
             domain_end_v() - domain_start_v()});
        return 128.0 * tolerance() * domain_scale;
    }

    [[nodiscard]] double clamped_u(double u) const
    {
        kfbim::geometry::require_finite(u, "surface u parameter");
        return std::clamp(u, domain_start_u(), domain_end_u());
    }

    [[nodiscard]] double clamped_v(double v) const
    {
        kfbim::geometry::require_finite(v, "surface v parameter");
        return std::clamp(v, domain_start_v(), domain_end_v());
    }

    [[nodiscard]] const Eigen::Vector3d& point(int i, int j) const
    {
        return control_net_[static_cast<std::size_t>(i)]
                           [static_cast<std::size_t>(j)];
    }

    [[nodiscard]] double weight(int i, int j) const
    {
        return weights_[static_cast<std::size_t>(i)]
                       [static_cast<std::size_t>(j)];
    }

    void check_control_index(int i, int j) const
    {
        if (i < 0 || i >= basis_u_.num_basis_functions() ||
            j < 0 || j >= basis_v_.num_basis_functions()) {
            throw std::invalid_argument(
                "rational surface basis index is out of range");
        }
    }

    [[nodiscard]] NurbsSurfaceDerivatives3D analytic_evaluation_and_partials(
        double u,
        double v) const
    {
        const auto active_u = basis_u_.active_basis_indices(u);
        const auto active_v = basis_v_.active_basis_indices(v);
        const auto values_u = basis_u_.evaluate_nonzero(u);
        const auto values_v = basis_v_.evaluate_nonzero(v);
        const auto derivatives_u =
            basis_u_.evaluate_nonzero_first_derivatives(u);
        const auto derivatives_v =
            basis_v_.evaluate_nonzero_first_derivatives(v);

        Eigen::Vector3d numerator = Eigen::Vector3d::Zero();
        Eigen::Vector3d numerator_derivative_u = Eigen::Vector3d::Zero();
        Eigen::Vector3d numerator_derivative_v = Eigen::Vector3d::Zero();
        double denominator = 0.0;
        double denominator_derivative_u = 0.0;
        double denominator_derivative_v = 0.0;

        for (std::size_t local_u = 0; local_u < active_u.size(); ++local_u) {
            const int i = active_u[local_u];
            for (std::size_t local_v = 0; local_v < active_v.size(); ++local_v) {
                const int j = active_v[local_v];
                const double basis_product =
                    values_u[local_u] * values_v[local_v];
                const double derivative_product_u =
                    derivatives_u[local_u] * values_v[local_v];
                const double derivative_product_v =
                    values_u[local_u] * derivatives_v[local_v];
                const double weighted_basis = basis_product * weight(i, j);
                const double weighted_derivative_u =
                    derivative_product_u * weight(i, j);
                const double weighted_derivative_v =
                    derivative_product_v * weight(i, j);

                numerator += weighted_basis * point(i, j);
                numerator_derivative_u += weighted_derivative_u * point(i, j);
                numerator_derivative_v += weighted_derivative_v * point(i, j);
                denominator += weighted_basis;
                denominator_derivative_u += weighted_derivative_u;
                denominator_derivative_v += weighted_derivative_v;
            }
        }

        if (denominator <= tolerance())
            throw std::runtime_error("NURBS surface denominator is zero or near zero");

        const double denominator_squared = denominator * denominator;
        NurbsSurfaceDerivatives3D derivatives;
        derivatives.point = numerator / denominator;
        derivatives.du =
            (numerator_derivative_u * denominator
           - numerator * denominator_derivative_u)
          / denominator_squared;
        derivatives.dv =
            (numerator_derivative_v * denominator
           - numerator * denominator_derivative_v)
          / denominator_squared;
        return derivatives;
    }

    [[nodiscard]] Eigen::Vector3d finite_difference_partial(double u,
                                                            double v,
                                                            bool with_respect_to_u) const
    {
        const double domain_length = with_respect_to_u
            ? domain_end_u() - domain_start_u()
            : domain_end_v() - domain_start_v();
        const double h =
            std::max(1.0e-7 * std::max(1.0, domain_length),
                     parameter_tolerance());

        if (with_respect_to_u) {
            const double minus = clamped_u(u - h);
            const double plus = clamped_u(u + h);
            if (plus <= minus + tolerance())
                return Eigen::Vector3d::Zero();
            return (evaluate(plus, v) - evaluate(minus, v)) / (plus - minus);
        }

        const double minus = clamped_v(v - h);
        const double plus = clamped_v(v + h);
        if (plus <= minus + tolerance())
            return Eigen::Vector3d::Zero();
        return (evaluate(u, plus) - evaluate(u, minus)) / (plus - minus);
    }

    [[nodiscard]] double squared_distance_to(const Eigen::Vector3d& x,
                                             double u,
                                             double v) const
    {
        const Eigen::Vector3d displacement = evaluate(u, v) - x;
        return displacement.squaredNorm();
    }

    [[nodiscard]] static std::vector<double> projection_samples(
        const kfbim::geometry::NurbsBasis1D& basis)
    {
        constexpr int samples_per_span = 6;
        std::vector<double> samples;
        samples.push_back(basis.domain_start());

        const auto& knots = basis.knots();
        for (std::size_t i = 1; i < knots.size(); ++i) {
            const double span_start =
                std::clamp(knots[i - 1], basis.domain_start(), basis.domain_end());
            const double span_end =
                std::clamp(knots[i], basis.domain_start(), basis.domain_end());
            if (span_end <= span_start + basis.tolerance())
                continue;
            for (int sample = 1; sample <= samples_per_span; ++sample) {
                const double alpha =
                    static_cast<double>(sample) / samples_per_span;
                samples.push_back((1.0 - alpha) * span_start + alpha * span_end);
            }
        }

        if (samples.back() < basis.domain_end() - basis.tolerance())
            samples.push_back(basis.domain_end());

        std::sort(samples.begin(), samples.end());
        samples.erase(
            std::unique(
                samples.begin(),
                samples.end(),
                [&basis](double lhs, double rhs) {
                    return std::abs(lhs - rhs) <= basis.tolerance();
                }),
            samples.end());
        return samples;
    }

    bool coordinate_search_step(const Eigen::Vector3d& x,
                                double& u,
                                double& v,
                                double& best_distance_squared) const
    {
        const double base_step =
            0.05 * std::max(domain_end_u() - domain_start_u(),
                            domain_end_v() - domain_start_v());
        bool improved = false;

        for (int reduction = 0; reduction < 12; ++reduction) {
            const double scale = std::ldexp(1.0, -reduction);
            const double step_u = std::max(base_step * scale, parameter_tolerance());
            const double step_v = std::max(base_step * scale, parameter_tolerance());
            for (const auto& offset : {std::pair<double, double>{step_u, 0.0},
                                       std::pair<double, double>{-step_u, 0.0},
                                       std::pair<double, double>{0.0, step_v},
                                       std::pair<double, double>{0.0, -step_v},
                                       std::pair<double, double>{step_u, step_v},
                                       std::pair<double, double>{step_u, -step_v},
                                       std::pair<double, double>{-step_u, step_v},
                                       std::pair<double, double>{-step_u, -step_v}}) {
                const double trial_u = clamped_u(u + offset.first);
                const double trial_v = clamped_v(v + offset.second);
                const double trial_distance_squared =
                    squared_distance_to(x, trial_u, trial_v);
                if (trial_distance_squared < best_distance_squared) {
                    u = trial_u;
                    v = trial_v;
                    best_distance_squared = trial_distance_squared;
                    improved = true;
                }
            }
            if (improved)
                break;
        }

        return improved;
    }
};

} // namespace kfbim::geometry3d
