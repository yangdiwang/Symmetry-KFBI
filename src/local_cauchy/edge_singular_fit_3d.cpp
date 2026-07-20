#include "edge_singular_fit_3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace kfbim {

namespace {

bool finite_positive(double value)
{
    return std::isfinite(value) && value > 0.0;
}

double integer_power(double x, int power)
{
    if (power == 0)
        return 1.0;
    return std::pow(x, static_cast<double>(power));
}

double falling_factorial(double power, int derivative_order)
{
    double value = 1.0;
    for (int k = 0; k < derivative_order; ++k)
        value *= power - static_cast<double>(k);
    return value;
}

double factorial(int value)
{
    double result = 1.0;
    for (int k = 2; k <= value; ++k)
        result *= static_cast<double>(k);
    return result;
}

bool nearly_equal(double a, double b)
{
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= 1.0e-12 * scale;
}

} // namespace

EdgeSingularFit3D::EdgeSingularFit3D(
    std::vector<EdgeSingularSample3D> samples,
    double edge_station,
    EdgeSingularFitOptions3D options)
    : samples_(std::move(samples))
    , edge_station_(edge_station)
    , options_(std::move(options))
{
    if (!std::isfinite(edge_station_))
        throw std::invalid_argument(
            "EdgeSingularFit3D edge_station must be finite");
    if (options_.regular_total_degree < 0)
        throw std::invalid_argument(
            "EdgeSingularFit3D regular_total_degree must be nonnegative");
    if (options_.strength_degree < 0)
        throw std::invalid_argument(
            "EdgeSingularFit3D strength_degree must be nonnegative");
    if (!finite_positive(options_.rho_scale)
        || !finite_positive(options_.s_scale)) {
        throw std::invalid_argument(
            "EdgeSingularFit3D coordinate scales must be finite and positive");
    }
    if (!finite_positive(options_.svd_relative_tolerance)
        || options_.svd_relative_tolerance >= 1.0) {
        throw std::invalid_argument(
            "EdgeSingularFit3D SVD tolerance must lie in (0,1)");
    }
    if (options_.exponents.empty())
        throw std::invalid_argument(
            "EdgeSingularFit3D requires at least one singular exponent");

    for (std::size_t k = 0; k < options_.exponents.size(); ++k) {
        const double exponent = options_.exponents[k];
        if (!finite_positive(exponent))
            throw std::invalid_argument(
                "EdgeSingularFit3D exponents must be finite and positive");
        for (std::size_t l = 0; l < k; ++l) {
            if (nearly_equal(exponent, options_.exponents[l]))
                throw std::invalid_argument(
                    "EdgeSingularFit3D exponents must be distinct");
        }
        for (int integer = 0; integer <= options_.regular_total_degree;
             ++integer) {
            if (nearly_equal(exponent, static_cast<double>(integer)))
                throw std::invalid_argument(
                    "EdgeSingularFit3D singular exponent duplicates the regular basis");
        }
    }

    // Total-degree regular polynomial basis.
    for (int total = 0; total <= options_.regular_total_degree; ++total) {
        for (int rho_power = 0; rho_power <= total; ++rho_power) {
            basis_terms_.push_back(
                {static_cast<double>(rho_power), total - rho_power, -1});
        }
    }
    // Local Taylor expansion of every a_lambda(s).
    for (int exponent_index = 0;
         exponent_index < static_cast<int>(options_.exponents.size());
         ++exponent_index) {
        for (int s_power = 0; s_power <= options_.strength_degree;
             ++s_power) {
            basis_terms_.push_back(
                {options_.exponents[static_cast<std::size_t>(exponent_index)],
                 s_power,
                 exponent_index});
        }
    }

    const int n = static_cast<int>(samples_.size());
    const int m = static_cast<int>(basis_terms_.size());
    if (n < m) {
        throw std::invalid_argument(
            "EdgeSingularFit3D needs at least as many samples as basis functions");
    }

    Eigen::MatrixXd design(n, m);
    Eigen::VectorXd sqrt_weights(n);
    for (int row = 0; row < n; ++row) {
        const EdgeSingularSample3D& sample =
            samples_[static_cast<std::size_t>(row)];
        if (!finite_positive(sample.rho) || !std::isfinite(sample.s)
            || !finite_positive(sample.weight)) {
            throw std::invalid_argument(
                "EdgeSingularFit3D samples require rho>0 and finite s/positive weight");
        }
        const double R = sample.rho / options_.rho_scale;
        const double T = (sample.s - edge_station_) / options_.s_scale;
        sqrt_weights[row] = std::sqrt(sample.weight);
        for (int col = 0; col < m; ++col) {
            const BasisTerm& term = basis_terms_[static_cast<std::size_t>(col)];
            design(row, col) = std::pow(R, term.rho_power)
                             * integer_power(T, term.s_power);
        }
    }

    Eigen::MatrixXd weighted_design = design;
    for (int row = 0; row < n; ++row)
        weighted_design.row(row) *= sqrt_weights[row];

    Eigen::VectorXd column_norms(m);
    Eigen::MatrixXd scaled_design = weighted_design;
    for (int col = 0; col < m; ++col) {
        column_norms[col] = scaled_design.col(col).norm();
        if (!finite_positive(column_norms[col]))
            throw std::runtime_error(
                "EdgeSingularFit3D has a zero or non-finite design column");
        scaled_design.col(col) /= column_norms[col];
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        scaled_design, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(options_.svd_relative_tolerance);
    const int rank = svd.rank();
    if (rank != m) {
        throw std::runtime_error(
            "EdgeSingularFit3D enriched design is rank deficient: rank="
            + std::to_string(rank) + " basis=" + std::to_string(m));
    }

    const Eigen::VectorXd singular_values = svd.singularValues();
    const double sigma_max = singular_values[0];
    const double sigma_min = singular_values[m - 1];
    if (!finite_positive(sigma_min))
        throw std::runtime_error(
            "EdgeSingularFit3D has a non-positive retained singular value");

    // scaled_design^+ maps weighted samples to column-scaled coefficients.
    // Undo both column and row scaling to obtain c = P * sample_values.
    const Eigen::MatrixXd scaled_pseudoinverse =
        svd.solve(Eigen::MatrixXd::Identity(n, n));
    coefficient_operator_ = column_norms.cwiseInverse().asDiagonal()
                          * scaled_pseudoinverse
                          * sqrt_weights.asDiagonal();

    diagnostics_.sample_count = n;
    diagnostics_.basis_size = m;
    diagnostics_.numerical_rank = rank;
    diagnostics_.scaled_condition_number = sigma_max / sigma_min;
    diagnostics_.reproduction_residual =
        (coefficient_operator_ * design
         - Eigen::MatrixXd::Identity(m, m))
            .cwiseAbs()
            .maxCoeff();
}

int EdgeSingularFit3D::sample_count() const
{
    return static_cast<int>(samples_.size());
}

int EdgeSingularFit3D::basis_size() const
{
    return static_cast<int>(basis_terms_.size());
}

double EdgeSingularFit3D::edge_station() const
{
    return edge_station_;
}

const std::vector<EdgeSingularSample3D>& EdgeSingularFit3D::samples() const
{
    return samples_;
}

const EdgeSingularFitOptions3D& EdgeSingularFit3D::options() const
{
    return options_;
}

const EdgeSingularFitDiagnostics3D& EdgeSingularFit3D::diagnostics() const
{
    return diagnostics_;
}

void EdgeSingularFit3D::require_sample_values(
    const Eigen::Ref<const Eigen::VectorXd>& sample_values) const
{
    if (sample_values.size() != sample_count())
        throw std::invalid_argument(
            "EdgeSingularFit3D sample value count does not match geometry");
    if (!sample_values.allFinite())
        throw std::invalid_argument(
            "EdgeSingularFit3D sample values must be finite");
}

Eigen::VectorXd EdgeSingularFit3D::fit_coefficients(
    const Eigen::Ref<const Eigen::VectorXd>& sample_values) const
{
    require_sample_values(sample_values);
    return coefficient_operator_ * sample_values;
}

Eigen::VectorXd EdgeSingularFit3D::basis_functional(
    double rho,
    double s,
    int rho_derivative_order,
    int s_derivative_order) const
{
    if (!finite_positive(rho) || !std::isfinite(s))
        throw std::invalid_argument(
            "EdgeSingularFit3D evaluation requires rho>0 and finite s");
    if (rho_derivative_order < 0 || s_derivative_order < 0)
        throw std::invalid_argument(
            "EdgeSingularFit3D derivative orders must be nonnegative");

    const double R = rho / options_.rho_scale;
    const double T = (s - edge_station_) / options_.s_scale;
    Eigen::VectorXd functional(basis_size());
    for (int i = 0; i < basis_size(); ++i) {
        const BasisTerm& term = basis_terms_[static_cast<std::size_t>(i)];
        // Fractional powers remain differentiable at every rho>0 even when
        // rho_power < derivative_order.  Integer polynomial powers vanish
        // automatically through their falling factorial, so only the
        // integer tangential exponent needs an early guard against a negative
        // power of T.
        if (term.s_power < s_derivative_order) {
            functional[i] = 0.0;
            continue;
        }
        const double radial =
            falling_factorial(term.rho_power, rho_derivative_order)
            * std::pow(R,
                       term.rho_power
                           - static_cast<double>(rho_derivative_order))
            / std::pow(options_.rho_scale,
                       static_cast<double>(rho_derivative_order));
        const double tangential =
            falling_factorial(static_cast<double>(term.s_power),
                              s_derivative_order)
            * integer_power(T, term.s_power - s_derivative_order)
            / std::pow(options_.s_scale,
                       static_cast<double>(s_derivative_order));
        functional[i] = radial * tangential;
    }
    return functional;
}

Eigen::VectorXd EdgeSingularFit3D::sample_weights_from_functional(
    const Eigen::Ref<const Eigen::VectorXd>& functional) const
{
    if (functional.size() != basis_size())
        throw std::invalid_argument(
            "EdgeSingularFit3D functional size does not match basis");
    return coefficient_operator_.transpose() * functional;
}

Eigen::VectorXd EdgeSingularFit3D::value_weights(double rho, double s) const
{
    return derivative_weights(rho, s, 0, 0);
}

Eigen::VectorXd EdgeSingularFit3D::derivative_weights(
    double rho,
    double s,
    int rho_derivative_order,
    int s_derivative_order) const
{
    return sample_weights_from_functional(
        basis_functional(rho,
                         s,
                         rho_derivative_order,
                         s_derivative_order));
}

Eigen::VectorXd EdgeSingularFit3D::radial_second_derivative_weights(
    double rho,
    double s) const
{
    return derivative_weights(rho, s, 2, 0);
}

Eigen::VectorXd EdgeSingularFit3D::tangential_second_derivative_weights(
    double rho,
    double s) const
{
    return derivative_weights(rho, s, 0, 2);
}

Eigen::VectorXd EdgeSingularFit3D::flat_surface_laplacian_weights(
    double rho,
    double s) const
{
    const Eigen::VectorXd functional =
        basis_functional(rho, s, 2, 0)
        + basis_functional(rho, s, 0, 2);
    return sample_weights_from_functional(functional);
}

double EdgeSingularFit3D::evaluate(
    const Eigen::Ref<const Eigen::VectorXd>& weights,
    const Eigen::Ref<const Eigen::VectorXd>& sample_values) const
{
    if (weights.size() != sample_count())
        throw std::invalid_argument(
            "EdgeSingularFit3D weight count does not match geometry");
    require_sample_values(sample_values);
    return weights.dot(sample_values);
}

Eigen::VectorXd EdgeSingularFit3D::singular_strength_weights(
    int exponent_index,
    int derivative_order) const
{
    if (exponent_index < 0
        || exponent_index >= static_cast<int>(options_.exponents.size())) {
        throw std::out_of_range(
            "EdgeSingularFit3D exponent index is out of range");
    }
    if (derivative_order < 0
        || derivative_order > options_.strength_degree) {
        throw std::out_of_range(
            "EdgeSingularFit3D strength derivative order is unavailable");
    }

    int basis_index = -1;
    for (int i = 0; i < basis_size(); ++i) {
        const BasisTerm& term = basis_terms_[static_cast<std::size_t>(i)];
        if (term.exponent_index == exponent_index
            && term.s_power == derivative_order) {
            basis_index = i;
            break;
        }
    }
    if (basis_index < 0)
        throw std::logic_error(
            "EdgeSingularFit3D internal singular basis lookup failed");

    const double exponent =
        options_.exponents[static_cast<std::size_t>(exponent_index)];
    const double scale = factorial(derivative_order)
                       / (std::pow(options_.rho_scale, exponent)
                          * std::pow(options_.s_scale,
                                     static_cast<double>(derivative_order)));
    Eigen::VectorXd functional = Eigen::VectorXd::Zero(basis_size());
    functional[basis_index] = scale;
    return sample_weights_from_functional(functional);
}

double EdgeSingularFit3D::singular_strength(
    const Eigen::Ref<const Eigen::VectorXd>& sample_values,
    int exponent_index,
    int derivative_order) const
{
    return evaluate(singular_strength_weights(exponent_index,
                                              derivative_order),
                    sample_values);
}

} // namespace kfbim
