#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfbim::geometry {

inline constexpr double kNurbsBasisTolerance = 1.0e-12;

class NurbsBasis1D {
public:
    NurbsBasis1D(int degree,
                 std::vector<double> knots,
                 double tolerance = kNurbsBasisTolerance)
        : degree_(degree)
        , knots_(std::move(knots))
        , tolerance_(tolerance)
    {
        validate();
    }

    [[nodiscard]] int degree() const noexcept { return degree_; }

    [[nodiscard]] int num_basis_functions() const noexcept
    {
        return static_cast<int>(knots_.size()) - degree_ - 1;
    }

    [[nodiscard]] const std::vector<double>& knots() const noexcept
    {
        return knots_;
    }

    [[nodiscard]] double tolerance() const noexcept { return tolerance_; }

    [[nodiscard]] double domain_start() const noexcept
    {
        return knots_[static_cast<std::size_t>(degree_)];
    }

    [[nodiscard]] double domain_end() const noexcept
    {
        return knots_[static_cast<std::size_t>(num_basis_functions())];
    }

    [[nodiscard]] int find_span(double u) const
    {
        const double parameter = checked_parameter(u);
        const int n = num_basis_functions() - 1;

        if (parameter <= domain_start() + tolerance_)
            return degree_;
        if (parameter >= domain_end() - tolerance_)
            return n;

        const auto upper =
            std::upper_bound(knots_.begin(), knots_.end(), parameter);
        int span = static_cast<int>(std::distance(knots_.begin(), upper)) - 1;
        return std::clamp(span, degree_, n);
    }

    [[nodiscard]] std::vector<int> active_basis_indices(double u) const
    {
        const int span = find_span(u);
        std::vector<int> indices(static_cast<std::size_t>(degree_ + 1));
        for (int local = 0; local <= degree_; ++local)
            indices[static_cast<std::size_t>(local)] = span - degree_ + local;
        return indices;
    }

    [[nodiscard]] std::vector<double> evaluate_nonzero(double u) const
    {
        const double parameter = checked_parameter(u);
        const int span = find_span(parameter);

        std::vector<double> values(static_cast<std::size_t>(degree_ + 1), 0.0);
        values[0] = 1.0;

        std::vector<double> left(static_cast<std::size_t>(degree_ + 1), 0.0);
        std::vector<double> right(static_cast<std::size_t>(degree_ + 1), 0.0);

        for (int j = 1; j <= degree_; ++j) {
            left[static_cast<std::size_t>(j)] =
                parameter - knots_[static_cast<std::size_t>(span + 1 - j)];
            right[static_cast<std::size_t>(j)] =
                knots_[static_cast<std::size_t>(span + j)] - parameter;

            double saved = 0.0;
            for (int r = 0; r < j; ++r) {
                const double denominator =
                    right[static_cast<std::size_t>(r + 1)]
                  + left[static_cast<std::size_t>(j - r)];
                const double temp =
                    divide_or_zero(values[static_cast<std::size_t>(r)], denominator);
                values[static_cast<std::size_t>(r)] =
                    saved + right[static_cast<std::size_t>(r + 1)] * temp;
                saved = left[static_cast<std::size_t>(j - r)] * temp;
            }
            values[static_cast<std::size_t>(j)] = saved;
        }

        return values;
    }

    [[nodiscard]] double basis_value(int i, double u) const
    {
        if (i < 0 || i >= num_basis_functions())
            throw std::invalid_argument("basis index is out of range");

        const auto indices = active_basis_indices(u);
        const auto values = evaluate_nonzero(u);
        for (std::size_t local = 0; local < indices.size(); ++local) {
            if (indices[local] == i)
                return values[local];
        }
        return 0.0;
    }

    [[nodiscard]] std::vector<double> evaluate_nonzero_first_derivatives(
        double u) const
    {
        const double parameter = checked_parameter(u);
        const int span = find_span(parameter);

        std::vector<double> derivatives(
            static_cast<std::size_t>(degree_ + 1), 0.0);
        if (degree_ == 0)
            return derivatives;

        const int p = degree_;
        std::vector<std::vector<double>> ndu(
            static_cast<std::size_t>(p + 1),
            std::vector<double>(static_cast<std::size_t>(p + 1), 0.0));
        std::vector<double> left(static_cast<std::size_t>(p + 1), 0.0);
        std::vector<double> right(static_cast<std::size_t>(p + 1), 0.0);

        ndu[0][0] = 1.0;
        for (int j = 1; j <= p; ++j) {
            left[static_cast<std::size_t>(j)] =
                parameter - knots_[static_cast<std::size_t>(span + 1 - j)];
            right[static_cast<std::size_t>(j)] =
                knots_[static_cast<std::size_t>(span + j)] - parameter;

            double saved = 0.0;
            for (int r = 0; r < j; ++r) {
                ndu[static_cast<std::size_t>(j)][static_cast<std::size_t>(r)] =
                    right[static_cast<std::size_t>(r + 1)]
                  + left[static_cast<std::size_t>(j - r)];
                const double temp = divide_or_zero(
                    ndu[static_cast<std::size_t>(r)]
                       [static_cast<std::size_t>(j - 1)],
                    ndu[static_cast<std::size_t>(j)]
                       [static_cast<std::size_t>(r)]);
                ndu[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)] =
                    saved + right[static_cast<std::size_t>(r + 1)] * temp;
                saved = left[static_cast<std::size_t>(j - r)] * temp;
            }
            ndu[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)] = saved;
        }

        std::vector<std::vector<double>> a(
            2, std::vector<double>(static_cast<std::size_t>(p + 1), 0.0));

        for (int r = 0; r <= p; ++r) {
            int s1 = 0;
            int s2 = 1;
            a[0][0] = 1.0;

            constexpr int k = 1;
            double d = 0.0;
            const int rk = r - k;
            const int pk = p - k;

            if (r >= k) {
                a[static_cast<std::size_t>(s2)][0] = divide_or_zero(
                    a[static_cast<std::size_t>(s1)][0],
                    ndu[static_cast<std::size_t>(pk + 1)]
                       [static_cast<std::size_t>(rk)]);
                d = a[static_cast<std::size_t>(s2)][0]
                  * ndu[static_cast<std::size_t>(rk)]
                       [static_cast<std::size_t>(pk)];
            }

            const int j1 = (rk >= -1) ? 1 : -rk;
            const int j2 = (r - 1 <= pk) ? k - 1 : p - r;

            for (int j = j1; j <= j2; ++j) {
                a[static_cast<std::size_t>(s2)][static_cast<std::size_t>(j)] =
                    divide_or_zero(
                        a[static_cast<std::size_t>(s1)]
                         [static_cast<std::size_t>(j)]
                      - a[static_cast<std::size_t>(s1)]
                         [static_cast<std::size_t>(j - 1)],
                        ndu[static_cast<std::size_t>(pk + 1)]
                           [static_cast<std::size_t>(rk + j)]);
                d += a[static_cast<std::size_t>(s2)][static_cast<std::size_t>(j)]
                   * ndu[static_cast<std::size_t>(rk + j)]
                        [static_cast<std::size_t>(pk)];
            }

            if (r <= pk) {
                a[static_cast<std::size_t>(s2)][static_cast<std::size_t>(k)] =
                    divide_or_zero(
                        -a[static_cast<std::size_t>(s1)]
                          [static_cast<std::size_t>(k - 1)],
                        ndu[static_cast<std::size_t>(pk + 1)]
                           [static_cast<std::size_t>(r)]);
                d += a[static_cast<std::size_t>(s2)][static_cast<std::size_t>(k)]
                   * ndu[static_cast<std::size_t>(r)]
                        [static_cast<std::size_t>(pk)];
            }

            std::swap(s1, s2);
            derivatives[static_cast<std::size_t>(r)] =
                static_cast<double>(p) * d;
        }

        return derivatives;
    }

    [[nodiscard]] double basis_first_derivative(int i, double u) const
    {
        if (i < 0 || i >= num_basis_functions())
            throw std::invalid_argument("basis index is out of range");

        const auto indices = active_basis_indices(u);
        const auto derivatives = evaluate_nonzero_first_derivatives(u);
        for (std::size_t local = 0; local < indices.size(); ++local) {
            if (indices[local] == i)
                return derivatives[local];
        }
        return 0.0;
    }

private:
    int degree_;
    std::vector<double> knots_;
    double tolerance_;

    void validate() const
    {
        if (degree_ < 0)
            throw std::invalid_argument("NURBS basis degree must be nonnegative");
        if (!std::isfinite(tolerance_) || tolerance_ <= 0.0)
            throw std::invalid_argument(
                "NURBS basis tolerance must be positive and finite");
        if (knots_.size() < static_cast<std::size_t>(2 * degree_ + 2))
            throw std::invalid_argument(
                "knot vector is too small for the requested degree");

        for (std::size_t i = 0; i < knots_.size(); ++i) {
            if (!std::isfinite(knots_[i]))
                throw std::invalid_argument("knot vector entries must be finite");
            if (i > 0 && knots_[i] + tolerance_ < knots_[i - 1])
                throw std::invalid_argument("knot vector must be nondecreasing");
        }

        if (domain_end() <= domain_start() + tolerance_)
            throw std::invalid_argument(
                "NURBS basis parameter domain must have positive length");
    }

    [[nodiscard]] double checked_parameter(double u) const
    {
        if (!std::isfinite(u))
            throw std::invalid_argument("basis parameter must be finite");
        if (u < domain_start() - tolerance_ || u > domain_end() + tolerance_)
            throw std::invalid_argument(
                "basis parameter is outside the active knot domain");
        if (u <= domain_start() + tolerance_)
            return domain_start();
        if (u >= domain_end() - tolerance_)
            return domain_end();
        return u;
    }

    [[nodiscard]] double divide_or_zero(double numerator,
                                        double denominator) const noexcept
    {
        if (std::abs(denominator) <= tolerance_)
            return 0.0;
        return numerator / denominator;
    }
};

} // namespace kfbim::geometry
