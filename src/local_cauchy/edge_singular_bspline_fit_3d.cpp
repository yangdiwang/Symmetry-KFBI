#include "edge_singular_bspline_fit_3d.hpp"

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

bool finite_nonnegative(double value)
{
    return std::isfinite(value) && value >= 0.0;
}

bool nearly_equal(double a, double b)
{
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= 1.0e-12 * scale;
}

double integer_power(double value, int power)
{
    if (power == 0)
        return 1.0;
    return std::pow(value, static_cast<double>(power));
}

double falling_factorial(double power, int derivative_order)
{
    double value = 1.0;
    for (int k = 0; k < derivative_order; ++k)
        value *= power - static_cast<double>(k);
    return value;
}

double condition_number(const Eigen::VectorXd& singular_values)
{
    if (singular_values.size() == 0)
        return std::numeric_limits<double>::infinity();
    const double smallest = singular_values[singular_values.size() - 1];
    if (!finite_positive(smallest))
        return std::numeric_limits<double>::infinity();
    return singular_values[0] / smallest;
}

} // namespace

EdgeSingularBSplineFit3D::EdgeSingularBSplineFit3D(
    std::vector<EdgeSingularBSplineSample3D> samples,
    EdgeSingularBSplineFitOptions3D options)
    : samples_(std::move(samples))
    , options_(std::move(options))
{
    if (options_.regular_total_degree < 0)
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D regular degree must be nonnegative");
    if (options_.spline_degree < 1)
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D spline degree must be positive");
    if (!std::isfinite(options_.s_min) || !std::isfinite(options_.s_max)
        || options_.s_max <= options_.s_min) {
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D requires s_min < s_max");
    }
    if (!finite_positive(options_.rho_scale))
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D rho_scale must be positive");
    if (options_.spline_elements < 0
        || !finite_nonnegative(options_.spline_spacing)) {
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D has invalid spline resolution");
    }
    if (options_.spline_elements == 0
        && !finite_positive(options_.spline_spacing)) {
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D needs spline_elements or spline_spacing");
    }
    if (!finite_nonnegative(options_.coefficient_ridge)
        || !finite_nonnegative(options_.smoothness_penalty)
        || !finite_nonnegative(options_.spectral_ridge)) {
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D penalties must be nonnegative");
    }
    if (!finite_positive(options_.svd_relative_tolerance)
        || options_.svd_relative_tolerance >= 1.0) {
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D SVD tolerance must lie in (0,1)");
    }
    if (options_.exponents.empty())
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D needs at least one exponent");
    for (std::size_t k = 0; k < options_.exponents.size(); ++k) {
        const double exponent = options_.exponents[k];
        if (!finite_positive(exponent))
            throw std::invalid_argument(
                "EdgeSingularBSplineFit3D exponents must be positive");
        for (std::size_t j = 0; j < k; ++j) {
            if (nearly_equal(exponent, options_.exponents[j]))
                throw std::invalid_argument(
                    "EdgeSingularBSplineFit3D exponents must be distinct");
        }
        for (int integer = 0; integer <= options_.regular_total_degree;
             ++integer) {
            if (nearly_equal(exponent, static_cast<double>(integer)))
                throw std::invalid_argument(
                    "EdgeSingularBSplineFit3D exponent duplicates regular basis");
        }
    }

    if (options_.regular_basis
        == EdgeSingularRegularBasis3D::PythonFourTerm) {
        regular_terms_ = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    } else {
        for (int total = 0; total <= options_.regular_total_degree; ++total) {
            for (int rho_power = 0; rho_power <= total; ++rho_power)
                regular_terms_.push_back({rho_power, total - rho_power});
        }
    }

    int spline_elements = options_.spline_elements;
    if (spline_elements == 0) {
        spline_elements = std::max(
            2,
            static_cast<int>(std::ceil(
                (options_.s_max - options_.s_min)
                / options_.spline_spacing)));
    }
    options_.spline_elements = spline_elements;
    const int degree = options_.spline_degree;
    knots_.reserve(static_cast<std::size_t>(
        2 * (degree + 1) + std::max(0, spline_elements - 1)));
    for (int k = 0; k <= degree; ++k)
        knots_.push_back(options_.s_min);
    for (int element = 1; element < spline_elements; ++element) {
        const double fraction = static_cast<double>(element)
                              / static_cast<double>(spline_elements);
        knots_.push_back(options_.s_min
                         + fraction * (options_.s_max - options_.s_min));
    }
    for (int k = 0; k <= degree; ++k)
        knots_.push_back(options_.s_max);
    spline_basis_size_ = static_cast<int>(knots_.size()) - degree - 1;

    if (samples_.empty())
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D needs surface samples");
    int maximum_group = -1;
    for (const auto& sample : samples_) {
        if (!finite_positive(sample.rho) || !std::isfinite(sample.s)
            || !std::isfinite(sample.psi)
            || !finite_positive(sample.weight)
            || sample.regular_group < 0
            || !std::isfinite(sample.singular_sign)) {
            throw std::invalid_argument(
                "EdgeSingularBSplineFit3D has an invalid sample");
        }
        const double tolerance = 1.0e-12
                               * std::max(1.0,
                                          options_.s_max - options_.s_min);
        if (sample.s < options_.s_min - tolerance
            || sample.s > options_.s_max + tolerance) {
            throw std::invalid_argument(
                "EdgeSingularBSplineFit3D sample lies outside spline interval");
        }
        maximum_group = std::max(maximum_group, sample.regular_group);
    }
    regular_group_count_ = maximum_group + 1;
    std::vector<bool> group_seen(static_cast<std::size_t>(regular_group_count_),
                                 false);
    for (const auto& sample : samples_)
        group_seen[static_cast<std::size_t>(sample.regular_group)] = true;
    if (std::find(group_seen.begin(), group_seen.end(), false)
        != group_seen.end()) {
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D regular groups must be contiguous");
    }

    regular_basis_size_ = regular_group_count_
                        * static_cast<int>(regular_terms_.size());
    singular_basis_offset_ = regular_basis_size_;
    const int singular_size = spline_basis_size_
                            * static_cast<int>(options_.exponents.size());
    const int m = regular_basis_size_ + singular_size;
    const int n = static_cast<int>(samples_.size());
    if (n < m) {
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D needs at least as many samples as basis functions");
    }

    const double s_center = 0.5 * (options_.s_min + options_.s_max);
    const double s_scale = 0.5 * (options_.s_max - options_.s_min);
    Eigen::MatrixXd design = Eigen::MatrixXd::Zero(n, m);
    Eigen::VectorXd sqrt_weights(n);
    for (int row = 0; row < n; ++row) {
        const auto& sample = samples_[static_cast<std::size_t>(row)];
        const double R = sample.rho / options_.rho_scale;
        const double S = (sample.s - s_center) / s_scale;
        const int regular_offset = sample.regular_group
                                 * static_cast<int>(regular_terms_.size());
        for (int j = 0; j < static_cast<int>(regular_terms_.size()); ++j) {
            const RegularTerm& term =
                regular_terms_[static_cast<std::size_t>(j)];
            design(row, regular_offset + j) =
                integer_power(R, term.rho_power)
                * integer_power(S, term.s_power);
        }
        const Eigen::VectorXd splines = bspline_values(sample.s, 0);
        for (int exponent_index = 0;
             exponent_index < static_cast<int>(options_.exponents.size());
             ++exponent_index) {
            const double exponent =
                options_.exponents[static_cast<std::size_t>(exponent_index)];
            const double radial_angular =
                sample.singular_sign * std::pow(R, exponent)
                * std::cos(exponent * sample.psi);
            for (int spline_index = 0; spline_index < spline_basis_size_;
                 ++spline_index) {
                design(row,
                       singular_basis_index(exponent_index, spline_index)) =
                    radial_angular * splines[spline_index];
            }
        }
        sqrt_weights[row] = std::sqrt(sample.weight);
    }

    Eigen::MatrixXd weighted_data = design;
    for (int row = 0; row < n; ++row)
        weighted_data.row(row) *= sqrt_weights[row];
    Eigen::VectorXd column_norms(m);
    Eigen::MatrixXd scaled_data = weighted_data;
    std::vector<int> active_columns;
    active_columns.reserve(static_cast<std::size_t>(m));
    for (int col = 0; col < m; ++col) {
        column_norms[col] = scaled_data.col(col).norm();
        if (finite_positive(column_norms[col]) && column_norms[col] > 1.0e-13) {
            scaled_data.col(col) /= column_norms[col];
            active_columns.push_back(col);
        } else {
            scaled_data.col(col).setZero();
            column_norms[col] = 1.0;
        }
    }

    if (active_columns.empty())
        throw std::runtime_error(
            "EdgeSingularBSplineFit3D has no active design columns");

    Eigen::MatrixXd active_data(n, static_cast<int>(active_columns.size()));
    for (int j = 0; j < static_cast<int>(active_columns.size()); ++j)
        active_data.col(j) = scaled_data.col(active_columns[static_cast<std::size_t>(j)]);

    Eigen::JacobiSVD<Eigen::MatrixXd> data_svd(
        active_data, Eigen::ComputeThinU | Eigen::ComputeThinV);
    data_svd.setThreshold(options_.svd_relative_tolerance);
    if (options_.require_full_rank
        && data_svd.rank() != static_cast<int>(active_columns.size())) {
        throw std::runtime_error(
            "EdgeSingularBSplineFit3D data design is rank deficient: rank="
            + std::to_string(data_svd.rank())
            + " active_basis=" + std::to_string(active_columns.size()));
    }

    const int ridge_rows = options_.coefficient_ridge > 0.0
                         ? singular_size : 0;
    const int smooth_rows = options_.smoothness_penalty > 0.0
                          ? static_cast<int>(options_.exponents.size())
                                * std::max(0, spline_basis_size_ - 2)
                          : 0;
    Eigen::MatrixXd augmented_full = Eigen::MatrixXd::Zero(
        n + ridge_rows + smooth_rows, m);
    augmented_full.topRows(n) = scaled_data;
    int penalty_row = n;
    if (ridge_rows > 0) {
        const double scale = std::sqrt(options_.coefficient_ridge);
        for (int j = 0; j < singular_size; ++j)
            augmented_full(penalty_row++, singular_basis_offset_ + j) = scale;
    }
    if (smooth_rows > 0) {
        const double scale = std::sqrt(options_.smoothness_penalty);
        for (int exponent_index = 0;
             exponent_index < static_cast<int>(options_.exponents.size());
             ++exponent_index) {
            for (int j = 0; j + 2 < spline_basis_size_; ++j) {
                augmented_full(penalty_row,
                          singular_basis_index(exponent_index, j)) = scale;
                augmented_full(penalty_row,
                          singular_basis_index(exponent_index, j + 1)) =
                    -2.0 * scale;
                augmented_full(penalty_row,
                          singular_basis_index(exponent_index, j + 2)) = scale;
                ++penalty_row;
            }
        }
    }

    Eigen::MatrixXd augmented(augmented_full.rows(),
                              static_cast<int>(active_columns.size()));
    for (int j = 0; j < static_cast<int>(active_columns.size()); ++j)
        augmented.col(j) =
            augmented_full.col(active_columns[static_cast<std::size_t>(j)]);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        augmented, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(options_.svd_relative_tolerance);
    const int rank = svd.rank();
    if (options_.require_full_rank
        && rank != static_cast<int>(active_columns.size())) {
        throw std::runtime_error(
            "EdgeSingularBSplineFit3D augmented design is rank deficient: rank="
            + std::to_string(rank)
            + " active_basis=" + std::to_string(active_columns.size()));
    }
    const Eigen::VectorXd singular_values = svd.singularValues();
    Eigen::VectorXd inverse_singular_values(singular_values.size());
    const double cutoff = options_.svd_relative_tolerance
                        * singular_values[0];
    int retained_rank = 0;
    double smallest_retained = std::numeric_limits<double>::infinity();
    for (int j = 0; j < singular_values.size(); ++j) {
        const double sigma = singular_values[j];
        if (sigma > cutoff) {
            inverse_singular_values[j] = sigma
                / (sigma * sigma + options_.spectral_ridge);
            smallest_retained = sigma;
            ++retained_rank;
        } else {
            inverse_singular_values[j] = 0.0;
        }
    }
    const Eigen::MatrixXd augmented_pseudoinverse =
        svd.matrixV() * inverse_singular_values.asDiagonal()
        * svd.matrixU().transpose();
    coefficient_operator_ = Eigen::MatrixXd::Zero(m, n);
    const Eigen::MatrixXd active_operator =
        augmented_pseudoinverse.leftCols(n) * sqrt_weights.asDiagonal();
    for (int j = 0; j < static_cast<int>(active_columns.size()); ++j) {
        const int col = active_columns[static_cast<std::size_t>(j)];
        coefficient_operator_.row(col) =
            active_operator.row(j) / column_norms[col];
    }

    diagnostics_.sample_count = n;
    diagnostics_.regular_group_count = regular_group_count_;
    diagnostics_.regular_basis_size = regular_basis_size_;
    diagnostics_.spline_basis_size = spline_basis_size_;
    diagnostics_.singular_basis_size = singular_size;
    diagnostics_.total_basis_size = m;
    diagnostics_.active_basis_size = static_cast<int>(active_columns.size());
    diagnostics_.numerical_rank = retained_rank;
    diagnostics_.scaled_data_condition_number =
        condition_number(data_svd.singularValues());
    diagnostics_.augmented_condition_number = retained_rank > 0
        ? singular_values[0] / smallest_retained
        : std::numeric_limits<double>::infinity();
    diagnostics_.data_reproduction_residual =
        (coefficient_operator_ * design
         - Eigen::MatrixXd::Identity(m, m))
            .cwiseAbs()
            .maxCoeff();
}

int EdgeSingularBSplineFit3D::sample_count() const
{
    return static_cast<int>(samples_.size());
}

int EdgeSingularBSplineFit3D::basis_size() const
{
    return coefficient_operator_.rows();
}

int EdgeSingularBSplineFit3D::regular_group_count() const
{
    return regular_group_count_;
}

int EdgeSingularBSplineFit3D::regular_basis_size_per_group() const
{
    return static_cast<int>(regular_terms_.size());
}

int EdgeSingularBSplineFit3D::spline_basis_size() const
{
    return spline_basis_size_;
}

int EdgeSingularBSplineFit3D::singular_coefficient_count() const
{
    return basis_size() - singular_basis_offset_;
}

const std::vector<double>& EdgeSingularBSplineFit3D::knots() const
{
    return knots_;
}

const std::vector<EdgeSingularBSplineSample3D>&
EdgeSingularBSplineFit3D::samples() const
{
    return samples_;
}

const EdgeSingularBSplineFitOptions3D&
EdgeSingularBSplineFit3D::options() const
{
    return options_;
}

const EdgeSingularBSplineFitDiagnostics3D&
EdgeSingularBSplineFit3D::diagnostics() const
{
    return diagnostics_;
}

void EdgeSingularBSplineFit3D::require_sample_values(
    const Eigen::Ref<const Eigen::VectorXd>& sample_values) const
{
    if (sample_values.size() != sample_count())
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D sample value count does not match geometry");
    if (!sample_values.allFinite())
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D sample values must be finite");
}

Eigen::VectorXd EdgeSingularBSplineFit3D::fit_coefficients(
    const Eigen::Ref<const Eigen::VectorXd>& sample_values) const
{
    require_sample_values(sample_values);
    return coefficient_operator_ * sample_values;
}

Eigen::VectorXd EdgeSingularBSplineFit3D::singular_coefficients(
    const Eigen::Ref<const Eigen::VectorXd>& sample_values) const
{
    return fit_coefficients(sample_values).tail(singular_coefficient_count());
}

double EdgeSingularBSplineFit3D::bspline_basis_value(
    int index,
    int degree,
    int derivative_order,
    double s) const
{
    if (derivative_order < 0 || derivative_order > degree || index < 0)
        return 0.0;
    if (degree == 0) {
        if (derivative_order != 0
            || index + 1 >= static_cast<int>(knots_.size())) {
            return 0.0;
        }
        return knots_[static_cast<std::size_t>(index)] <= s
                    && s < knots_[static_cast<std::size_t>(index + 1)]
             ? 1.0 : 0.0;
    }

    const double left_denominator =
        knots_[static_cast<std::size_t>(index + degree)]
        - knots_[static_cast<std::size_t>(index)];
    const double right_denominator =
        knots_[static_cast<std::size_t>(index + degree + 1)]
        - knots_[static_cast<std::size_t>(index + 1)];
    if (derivative_order == 0) {
        double value = 0.0;
        if (left_denominator > 0.0) {
            value += (s - knots_[static_cast<std::size_t>(index)])
                   / left_denominator
                   * bspline_basis_value(index, degree - 1, 0, s);
        }
        if (right_denominator > 0.0) {
            value += (knots_[static_cast<std::size_t>(index + degree + 1)]
                      - s)
                   / right_denominator
                   * bspline_basis_value(index + 1, degree - 1, 0, s);
        }
        return value;
    }

    double value = 0.0;
    if (left_denominator > 0.0) {
        value += static_cast<double>(degree) / left_denominator
               * bspline_basis_value(index,
                                     degree - 1,
                                     derivative_order - 1,
                                     s);
    }
    if (right_denominator > 0.0) {
        value -= static_cast<double>(degree) / right_denominator
               * bspline_basis_value(index + 1,
                                     degree - 1,
                                     derivative_order - 1,
                                     s);
    }
    return value;
}

Eigen::VectorXd EdgeSingularBSplineFit3D::bspline_values(
    double s,
    int derivative_order) const
{
    if (!std::isfinite(s) || derivative_order < 0)
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D has an invalid spline evaluation");
    const double tolerance = 1.0e-12
                           * std::max(1.0,
                                      options_.s_max - options_.s_min);
    if (s < options_.s_min - tolerance || s > options_.s_max + tolerance)
        throw std::out_of_range(
            "EdgeSingularBSplineFit3D spline evaluation is outside the edge");
    if (derivative_order > options_.spline_degree)
        return Eigen::VectorXd::Zero(spline_basis_size_);

    double evaluation_s = std::max(options_.s_min, std::min(options_.s_max, s));
    if (nearly_equal(evaluation_s, options_.s_max))
        evaluation_s = std::nextafter(options_.s_max, options_.s_min);
    Eigen::VectorXd values(spline_basis_size_);
    for (int j = 0; j < spline_basis_size_; ++j) {
        values[j] = bspline_basis_value(j,
                                        options_.spline_degree,
                                        derivative_order,
                                        evaluation_s);
    }
    if (derivative_order == 0 && nearly_equal(s, options_.s_max)) {
        values.setZero();
        values[spline_basis_size_ - 1] = 1.0;
    }
    return values;
}

int EdgeSingularBSplineFit3D::singular_basis_index(
    int exponent_index,
    int spline_index) const
{
    return singular_basis_offset_
         + exponent_index * spline_basis_size_
         + spline_index;
}

Eigen::VectorXd
EdgeSingularBSplineFit3D::sample_weights_from_coefficient_functional(
    const Eigen::Ref<const Eigen::VectorXd>& functional) const
{
    if (functional.size() != basis_size())
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D functional size does not match basis");
    return coefficient_operator_.transpose() * functional;
}

Eigen::VectorXd EdgeSingularBSplineFit3D::singular_coefficient_functional(
    double rho,
    double s,
    double psi,
    int rho_derivative_order,
    int s_derivative_order,
    double singular_sign) const
{
    if (!finite_positive(rho) || !std::isfinite(psi)
        || rho_derivative_order < 0 || s_derivative_order < 0
        || !std::isfinite(singular_sign)) {
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D has an invalid singular functional");
    }
    const double R = rho / options_.rho_scale;
    const Eigen::VectorXd splines = bspline_values(s, s_derivative_order);
    Eigen::VectorXd functional = Eigen::VectorXd::Zero(basis_size());
    for (int exponent_index = 0;
         exponent_index < static_cast<int>(options_.exponents.size());
         ++exponent_index) {
        const double exponent =
            options_.exponents[static_cast<std::size_t>(exponent_index)];
        const double radial =
            falling_factorial(exponent, rho_derivative_order)
            * std::pow(R,
                       exponent - static_cast<double>(rho_derivative_order))
            / std::pow(options_.rho_scale,
                       static_cast<double>(rho_derivative_order));
        const double factor = singular_sign * radial
                            * std::cos(exponent * psi);
        for (int spline_index = 0; spline_index < spline_basis_size_;
             ++spline_index) {
            functional[singular_basis_index(exponent_index, spline_index)] =
                factor * splines[spline_index];
        }
    }
    return functional;
}

Eigen::VectorXd EdgeSingularBSplineFit3D::normalized_strength_weights(
    double s,
    int exponent_index,
    int derivative_order) const
{
    if (exponent_index < 0
        || exponent_index >= static_cast<int>(options_.exponents.size())) {
        throw std::out_of_range(
            "EdgeSingularBSplineFit3D exponent index is out of range");
    }
    const Eigen::VectorXd splines = bspline_values(s, derivative_order);
    Eigen::VectorXd functional = Eigen::VectorXd::Zero(basis_size());
    for (int j = 0; j < spline_basis_size_; ++j)
        functional[singular_basis_index(exponent_index, j)] = splines[j];
    return sample_weights_from_coefficient_functional(functional);
}

double EdgeSingularBSplineFit3D::normalized_strength(
    const Eigen::Ref<const Eigen::VectorXd>& sample_values,
    double s,
    int exponent_index,
    int derivative_order) const
{
    return evaluate(normalized_strength_weights(s,
                                                exponent_index,
                                                derivative_order),
                    sample_values);
}

Eigen::VectorXd EdgeSingularBSplineFit3D::singular_strength_weights(
    double s,
    int exponent_index,
    int derivative_order) const
{
    const double exponent =
        options_.exponents.at(static_cast<std::size_t>(exponent_index));
    return normalized_strength_weights(s,
                                       exponent_index,
                                       derivative_order)
         / std::pow(options_.rho_scale, exponent);
}

double EdgeSingularBSplineFit3D::singular_strength(
    const Eigen::Ref<const Eigen::VectorXd>& sample_values,
    double s,
    int exponent_index,
    int derivative_order) const
{
    return evaluate(singular_strength_weights(s,
                                              exponent_index,
                                              derivative_order),
                    sample_values);
}

double EdgeSingularBSplineFit3D::singular_strength_from_coefficients(
    const Eigen::Ref<const Eigen::VectorXd>& singular_coefficients,
    double s,
    int exponent_index,
    int derivative_order) const
{
    if (singular_coefficients.size() != singular_coefficient_count()) {
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D singular coefficient size mismatch");
    }
    if (!singular_coefficients.allFinite()) {
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D singular coefficients must be finite");
    }
    if (exponent_index < 0
        || exponent_index >= static_cast<int>(options_.exponents.size())) {
        throw std::out_of_range(
            "EdgeSingularBSplineFit3D exponent index is out of range");
    }
    const Eigen::VectorXd splines = bspline_values(s, derivative_order);
    const int offset = exponent_index * spline_basis_size_;
    const double normalized =
        singular_coefficients.segment(offset, spline_basis_size_).dot(splines);
    return normalized
         / std::pow(options_.rho_scale,
                    options_.exponents[static_cast<std::size_t>(exponent_index)]);
}

Eigen::VectorXd EdgeSingularBSplineFit3D::singular_value_weights(
    double rho,
    double s,
    double psi,
    double singular_sign) const
{
    return sample_weights_from_coefficient_functional(
        singular_coefficient_functional(rho, s, psi, 0, 0, singular_sign));
}

Eigen::VectorXd
EdgeSingularBSplineFit3D::singular_radial_second_derivative_weights(
    double rho,
    double s,
    double psi,
    double singular_sign) const
{
    return sample_weights_from_coefficient_functional(
        singular_coefficient_functional(rho, s, psi, 2, 0, singular_sign));
}

Eigen::VectorXd
EdgeSingularBSplineFit3D::singular_tangential_second_derivative_weights(
    double rho,
    double s,
    double psi,
    double singular_sign) const
{
    return sample_weights_from_coefficient_functional(
        singular_coefficient_functional(rho, s, psi, 0, 2, singular_sign));
}

Eigen::VectorXd
EdgeSingularBSplineFit3D::singular_flat_face_laplacian_weights(
    double rho,
    double s,
    double psi,
    double singular_sign) const
{
    const Eigen::VectorXd functional =
        singular_coefficient_functional(rho, s, psi, 2, 0, singular_sign)
        + singular_coefficient_functional(rho, s, psi, 0, 2, singular_sign);
    return sample_weights_from_coefficient_functional(functional);
}

Eigen::VectorXd EdgeSingularBSplineFit3D::singular_volume_residual_weights(
    double rho,
    double s,
    double psi,
    double singular_sign) const
{
    return singular_tangential_second_derivative_weights(rho,
                                                         s,
                                                         psi,
                                                         singular_sign);
}

double EdgeSingularBSplineFit3D::evaluate(
    const Eigen::Ref<const Eigen::VectorXd>& weights,
    const Eigen::Ref<const Eigen::VectorXd>& sample_values) const
{
    if (weights.size() != sample_count())
        throw std::invalid_argument(
            "EdgeSingularBSplineFit3D weight count does not match samples");
    require_sample_values(sample_values);
    return weights.dot(sample_values);
}

} // namespace kfbim
