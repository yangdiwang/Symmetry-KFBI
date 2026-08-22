#include "laplace_arc_length_bspline_crossing_jet_2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include "../geometry/nurbs_boundary_2d.hpp"
#include "../geometry/p2_curve_2d.hpp"

namespace kfbim {
namespace {

constexpr std::array<double, 8> kGaussNodes{{
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
     0.1834346424956498,
     0.5255324099163290,
     0.7966664774136267,
     0.9602898564975363}};

constexpr std::array<double, 8> kGaussWeights{{
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763}};

double panel_arclength(const Interface2D& iface,
                       int panel,
                       double from_s,
                       double to_s)
{
    if (from_s == to_s)
        return 0.0;
    const double midpoint = 0.5 * (from_s + to_s);
    const double half_span = 0.5 * (to_s - from_s);
    double integral = 0.0;
    for (std::size_t q = 0; q < kGaussNodes.size(); ++q) {
        const double s = midpoint + half_span * kGaussNodes[q];
        const double speed =
            geometry2d::panel_tangent(iface, panel, s).norm();
        if (!(speed > 1.0e-14) || !std::isfinite(speed)) {
            throw std::invalid_argument(
                "ALS-CJ encountered degenerate panel arclength");
        }
        integral += kGaussWeights[q] * speed;
    }
    return half_span * integral;
}

double factorial(int value)
{
    double result = 1.0;
    for (int i = 2; i <= value; ++i)
        result *= static_cast<double>(i);
    return result;
}

int binomial(int n, int k)
{
    if (k < 0 || k > n)
        return 0;
    k = std::min(k, n - k);
    int result = 1;
    for (int i = 1; i <= k; ++i)
        result = result * (n - k + i) / i;
    return result;
}

// Centered cardinal B-spline beta_p and its derivatives with respect to its
// dimensionless argument.  The truncated-power representation is compact and
// exact enough for the only degrees used by ALS-CJ (p=2,3).
double centered_cardinal_bspline(int degree,
                                 int derivative_order,
                                 double x)
{
    if (degree < 0 || derivative_order < 0
        || derivative_order > degree) {
        return 0.0;
    }
    const double radius = 0.5 * static_cast<double>(degree + 1);
    if (x < -radius || x > radius)
        return 0.0;

    const int power = degree - derivative_order;
    double value = 0.0;
    for (int k = 0; k <= degree + 1; ++k) {
        const double shifted =
            x + radius - static_cast<double>(k);
        if (!(shifted > 0.0))
            continue;
        const double term = power == 0
            ? 1.0 : std::pow(shifted, power);
        value += (k % 2 == 0 ? 1.0 : -1.0)
               * static_cast<double>(binomial(degree + 1, k))
               * term;
    }
    return value / factorial(power);
}

struct PeriodicBasisEntry {
    int coefficient = -1;
    std::array<double, 3> derivative{{0.0, 0.0, 0.0}};
};

std::vector<PeriodicBasisEntry> periodic_basis_entries(
    int degree,
    int coefficient_count,
    double arclength,
    double component_length,
    int maximum_derivative)
{
    if (coefficient_count <= degree + 1
        || !(component_length > 0.0)) {
        return {};
    }
    arclength = std::fmod(arclength, component_length);
    if (arclength < 0.0)
        arclength += component_length;
    const double z = static_cast<double>(coefficient_count)
                   * arclength / component_length;
    const int base = static_cast<int>(std::floor(z));
    const double derivative_scale =
        static_cast<double>(coefficient_count) / component_length;

    std::vector<PeriodicBasisEntry> entries;
    entries.reserve(static_cast<std::size_t>(degree + 1));
    for (int raw = base - degree - 2;
         raw <= base + degree + 2;
         ++raw) {
        int index = raw % coefficient_count;
        if (index < 0)
            index += coefficient_count;
        if (std::any_of(entries.begin(), entries.end(),
                        [index](const PeriodicBasisEntry& entry) {
                            return entry.coefficient == index;
                        })) {
            continue;
        }

        double distance = z - static_cast<double>(index);
        distance = std::remainder(
            distance, static_cast<double>(coefficient_count));
        PeriodicBasisEntry entry;
        entry.coefficient = index;
        double scale = 1.0;
        for (int derivative = 0;
             derivative <= maximum_derivative;
             ++derivative) {
            entry.derivative[static_cast<std::size_t>(derivative)] =
                scale * centered_cardinal_bspline(
                    degree, derivative, distance);
            scale *= derivative_scale;
        }
        if (std::abs(entry.derivative[0]) > 1.0e-15
            || std::abs(entry.derivative[1]) > 1.0e-15
            || std::abs(entry.derivative[2]) > 1.0e-15) {
            entries.push_back(entry);
        }
    }
    std::sort(entries.begin(), entries.end(),
              [](const PeriodicBasisEntry& lhs,
                 const PeriodicBasisEntry& rhs) {
                  return lhs.coefficient < rhs.coefficient;
              });
    return entries;
}

struct PeriodicSplineSystem {
    using SparseMatrix = Eigen::SparseMatrix<double>;
    using Solver = Eigen::SparseLU<SparseMatrix,
                                   Eigen::COLAMDOrdering<int>>;

    int degree = 0;
    SparseMatrix interpolation;
    std::unique_ptr<Solver> solver;

    bool build(int requested_degree,
               const std::vector<double>& sample_arclength,
               double component_length)
    {
        degree = requested_degree;
        const int n = static_cast<int>(sample_arclength.size());
        if (n <= degree + 1)
            return false;

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<std::size_t>((degree + 1) * n));
        for (int row = 0; row < n; ++row) {
            const auto entries = periodic_basis_entries(
                degree,
                n,
                sample_arclength[static_cast<std::size_t>(row)],
                component_length,
                0);
            for (const PeriodicBasisEntry& entry : entries) {
                if (std::abs(entry.derivative[0]) > 1.0e-15) {
                    triplets.emplace_back(
                        row, entry.coefficient, entry.derivative[0]);
                }
            }
        }
        interpolation.resize(n, n);
        interpolation.setFromTriplets(triplets.begin(), triplets.end());
        interpolation.makeCompressed();

        solver = std::make_unique<Solver>();
        solver->analyzePattern(interpolation);
        solver->factorize(interpolation);
        if (solver->info() != Eigen::Success)
            return false;

        const Eigen::VectorXd ones = Eigen::VectorXd::Ones(n);
        const Eigen::VectorXd coefficients = solver->solve(ones);
        if (solver->info() != Eigen::Success || !coefficients.allFinite())
            return false;
        const double residual =
            (interpolation * coefficients - ones).lpNorm<Eigen::Infinity>();
        return residual <= 1.0e-10;
    }

    Eigen::VectorXd fit(const Eigen::VectorXd& samples) const
    {
        if (!solver || samples.size() != interpolation.rows()) {
            throw std::invalid_argument(
                "ALS-CJ periodic spline sample count mismatch");
        }
        const Eigen::VectorXd coefficients = solver->solve(samples);
        if (solver->info() != Eigen::Success || !coefficients.allFinite()) {
            throw std::runtime_error(
                "ALS-CJ periodic spline solve failed");
        }
        return coefficients;
    }
};

int open_knot_span(int degree,
                   int coefficient_count,
                   double parameter,
                   const std::vector<double>& knots)
{
    const int last_coefficient = coefficient_count - 1;
    if (parameter >= knots[static_cast<std::size_t>(last_coefficient + 1)])
        return last_coefficient;
    if (parameter <= knots[static_cast<std::size_t>(degree)])
        return degree;

    int low = degree;
    int high = last_coefficient + 1;
    int middle = (low + high) / 2;
    while (parameter < knots[static_cast<std::size_t>(middle)]
           || parameter
                  >= knots[static_cast<std::size_t>(middle + 1)]) {
        if (parameter < knots[static_cast<std::size_t>(middle)])
            high = middle;
        else
            low = middle;
        middle = (low + high) / 2;
    }
    return middle;
}

std::vector<PeriodicBasisEntry> open_basis_entries(
    int degree,
    int coefficient_count,
    double arclength,
    double interval_length,
    const std::vector<double>& knots,
    int maximum_derivative)
{
    if (coefficient_count < degree + 1
        || !(interval_length > 0.0)
        || static_cast<int>(knots.size())
               != coefficient_count + degree + 1) {
        return {};
    }
    maximum_derivative = std::min(maximum_derivative, degree);
    const double parameter = std::clamp(
        arclength / interval_length, 0.0, 1.0);
    const int span = open_knot_span(
        degree, coefficient_count, parameter, knots);

    std::vector<std::vector<double>> ndu(
        static_cast<std::size_t>(degree + 1),
        std::vector<double>(static_cast<std::size_t>(degree + 1), 0.0));
    std::vector<double> left(static_cast<std::size_t>(degree + 1), 0.0);
    std::vector<double> right(static_cast<std::size_t>(degree + 1), 0.0);
    ndu[0][0] = 1.0;
    for (int column = 1; column <= degree; ++column) {
        left[static_cast<std::size_t>(column)] =
            parameter
            - knots[static_cast<std::size_t>(span + 1 - column)];
        right[static_cast<std::size_t>(column)] =
            knots[static_cast<std::size_t>(span + column)] - parameter;
        double saved = 0.0;
        for (int row = 0; row < column; ++row) {
            const double denominator =
                right[static_cast<std::size_t>(row + 1)]
                + left[static_cast<std::size_t>(column - row)];
            ndu[static_cast<std::size_t>(column)]
               [static_cast<std::size_t>(row)] = denominator;
            const double temporary = std::abs(denominator) > 1.0e-15
                ? ndu[static_cast<std::size_t>(row)]
                     [static_cast<std::size_t>(column - 1)] / denominator
                : 0.0;
            ndu[static_cast<std::size_t>(row)]
               [static_cast<std::size_t>(column)] =
                saved
                + right[static_cast<std::size_t>(row + 1)] * temporary;
            saved = left[static_cast<std::size_t>(column - row)]
                  * temporary;
        }
        ndu[static_cast<std::size_t>(column)]
           [static_cast<std::size_t>(column)] = saved;
    }

    std::vector<std::vector<double>> derivatives(
        static_cast<std::size_t>(maximum_derivative + 1),
        std::vector<double>(static_cast<std::size_t>(degree + 1), 0.0));
    for (int basis = 0; basis <= degree; ++basis) {
        derivatives[0][static_cast<std::size_t>(basis)] =
            ndu[static_cast<std::size_t>(basis)]
               [static_cast<std::size_t>(degree)];
    }

    std::vector<std::vector<double>> work(
        2, std::vector<double>(static_cast<std::size_t>(degree + 1), 0.0));
    for (int basis = 0; basis <= degree; ++basis) {
        int previous = 0;
        int current = 1;
        work[0][0] = 1.0;
        for (int derivative = 1;
             derivative <= maximum_derivative;
             ++derivative) {
            std::fill(work[static_cast<std::size_t>(current)].begin(),
                      work[static_cast<std::size_t>(current)].end(), 0.0);
            double value = 0.0;
            const int shifted_basis = basis - derivative;
            const int reduced_degree = degree - derivative;
            if (basis >= derivative) {
                const double denominator =
                    ndu[static_cast<std::size_t>(reduced_degree + 1)]
                       [static_cast<std::size_t>(shifted_basis)];
                work[static_cast<std::size_t>(current)][0] =
                    std::abs(denominator) > 1.0e-15
                        ? work[static_cast<std::size_t>(previous)][0]
                              / denominator
                        : 0.0;
                value = work[static_cast<std::size_t>(current)][0]
                      * ndu[static_cast<std::size_t>(shifted_basis)]
                           [static_cast<std::size_t>(reduced_degree)];
            }
            const int first = shifted_basis >= -1 ? 1 : -shifted_basis;
            const int last = basis - 1 <= reduced_degree
                ? derivative - 1 : degree - basis;
            for (int column = first; column <= last; ++column) {
                const double denominator =
                    ndu[static_cast<std::size_t>(reduced_degree + 1)]
                       [static_cast<std::size_t>(shifted_basis + column)];
                work[static_cast<std::size_t>(current)]
                    [static_cast<std::size_t>(column)] =
                    std::abs(denominator) > 1.0e-15
                        ? (work[static_cast<std::size_t>(previous)]
                               [static_cast<std::size_t>(column)]
                           - work[static_cast<std::size_t>(previous)]
                               [static_cast<std::size_t>(column - 1)])
                              / denominator
                        : 0.0;
                value += work[static_cast<std::size_t>(current)]
                              [static_cast<std::size_t>(column)]
                       * ndu[static_cast<std::size_t>(
                                 shifted_basis + column)]
                            [static_cast<std::size_t>(reduced_degree)];
            }
            if (basis <= reduced_degree) {
                const double denominator =
                    ndu[static_cast<std::size_t>(reduced_degree + 1)]
                       [static_cast<std::size_t>(basis)];
                work[static_cast<std::size_t>(current)]
                    [static_cast<std::size_t>(derivative)] =
                    std::abs(denominator) > 1.0e-15
                        ? -work[static_cast<std::size_t>(previous)]
                               [static_cast<std::size_t>(derivative - 1)]
                              / denominator
                        : 0.0;
                value += work[static_cast<std::size_t>(current)]
                              [static_cast<std::size_t>(derivative)]
                       * ndu[static_cast<std::size_t>(basis)]
                            [static_cast<std::size_t>(reduced_degree)];
            }
            derivatives[static_cast<std::size_t>(derivative)]
                       [static_cast<std::size_t>(basis)] = value;
            std::swap(previous, current);
        }
    }

    double factorial_scale = static_cast<double>(degree);
    for (int derivative = 1;
         derivative <= maximum_derivative;
         ++derivative) {
        const double arclength_scale =
            std::pow(interval_length, -derivative);
        for (int basis = 0; basis <= degree; ++basis) {
            derivatives[static_cast<std::size_t>(derivative)]
                       [static_cast<std::size_t>(basis)] *=
                factorial_scale * arclength_scale;
        }
        factorial_scale *= static_cast<double>(degree - derivative);
    }

    std::vector<PeriodicBasisEntry> entries;
    entries.reserve(static_cast<std::size_t>(degree + 1));
    for (int local = 0; local <= degree; ++local) {
        PeriodicBasisEntry entry;
        entry.coefficient = span - degree + local;
        for (int derivative = 0;
             derivative <= maximum_derivative;
             ++derivative) {
            entry.derivative[static_cast<std::size_t>(derivative)] =
                derivatives[static_cast<std::size_t>(derivative)]
                           [static_cast<std::size_t>(local)];
        }
        entries.push_back(entry);
    }
    return entries;
}

struct OpenSplineSystem {
    using SparseMatrix = Eigen::SparseMatrix<double>;
    using Solver = Eigen::SparseLU<SparseMatrix,
                                   Eigen::COLAMDOrdering<int>>;

    int degree = 0;
    double interval_length = 0.0;
    std::vector<double> knots;
    SparseMatrix interpolation;
    std::unique_ptr<Solver> solver;

    bool build(int requested_degree,
               const std::vector<double>& sample_arclength)
    {
        degree = requested_degree;
        const int count = static_cast<int>(sample_arclength.size());
        if (count < degree + 1)
            return false;
        const double start = sample_arclength.front();
        interval_length = sample_arclength.back() - start;
        if (!(interval_length > 0.0))
            return false;

        std::vector<double> parameters(static_cast<std::size_t>(count));
        for (int sample = 0; sample < count; ++sample) {
            parameters[static_cast<std::size_t>(sample)] =
                (sample_arclength[static_cast<std::size_t>(sample)] - start)
                / interval_length;
            if (sample > 0
                && !(parameters[static_cast<std::size_t>(sample)]
                     > parameters[static_cast<std::size_t>(sample - 1)])) {
                return false;
            }
        }
        parameters.front() = 0.0;
        parameters.back() = 1.0;

        knots.assign(static_cast<std::size_t>(count + degree + 1), 0.0);
        for (int index = count; index < count + degree + 1; ++index)
            knots[static_cast<std::size_t>(index)] = 1.0;
        const int last_coefficient = count - 1;
        for (int knot = 1; knot <= last_coefficient - degree; ++knot) {
            double average = 0.0;
            for (int sample = knot; sample < knot + degree; ++sample) {
                average += parameters[static_cast<std::size_t>(sample)];
            }
            knots[static_cast<std::size_t>(knot + degree)] =
                average / static_cast<double>(degree);
        }

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<std::size_t>((degree + 1) * count));
        for (int row = 0; row < count; ++row) {
            const auto entries = open_basis_entries(
                degree,
                count,
                sample_arclength[static_cast<std::size_t>(row)] - start,
                interval_length,
                knots,
                0);
            for (const PeriodicBasisEntry& entry : entries) {
                if (std::abs(entry.derivative[0]) > 1.0e-15) {
                    triplets.emplace_back(
                        row, entry.coefficient, entry.derivative[0]);
                }
            }
        }
        interpolation.resize(count, count);
        interpolation.setFromTriplets(triplets.begin(), triplets.end());
        interpolation.makeCompressed();

        solver = std::make_unique<Solver>();
        solver->analyzePattern(interpolation);
        solver->factorize(interpolation);
        if (solver->info() != Eigen::Success)
            return false;
        const Eigen::VectorXd ones = Eigen::VectorXd::Ones(count);
        const Eigen::VectorXd coefficients = solver->solve(ones);
        if (solver->info() != Eigen::Success || !coefficients.allFinite())
            return false;
        return (interpolation * coefficients - ones)
                   .lpNorm<Eigen::Infinity>() <= 1.0e-10;
    }

    Eigen::VectorXd fit(const Eigen::VectorXd& samples) const
    {
        if (!solver || samples.size() != interpolation.rows()) {
            throw std::invalid_argument(
                "ALS-CJ open spline sample count mismatch");
        }
        const Eigen::VectorXd coefficients = solver->solve(samples);
        if (solver->info() != Eigen::Success || !coefficients.allFinite()) {
            throw std::runtime_error("ALS-CJ open spline solve failed");
        }
        return coefficients;
    }

    std::vector<PeriodicBasisEntry> basis_entries(
        int coefficient_count,
        double arclength,
        int maximum_derivative) const
    {
        return open_basis_entries(degree,
                                  coefficient_count,
                                  arclength,
                                  interval_length,
                                  knots,
                                  maximum_derivative);
    }
};

struct SplineSystem {
    bool periodic = true;
    int degree = 0;
    double length = 0.0;
    PeriodicSplineSystem periodic_system;
    OpenSplineSystem open_system;

    bool build(int requested_degree,
               const std::vector<double>& sample_arclength,
               double curve_length,
               bool use_periodic)
    {
        periodic = use_periodic;
        degree = requested_degree;
        length = curve_length;
        return periodic
            ? periodic_system.build(
                  requested_degree, sample_arclength, curve_length)
            : open_system.build(requested_degree, sample_arclength);
    }

    Eigen::VectorXd fit(const Eigen::VectorXd& samples) const
    {
        return periodic
            ? periodic_system.fit(samples)
            : open_system.fit(samples);
    }

    std::vector<PeriodicBasisEntry> basis_entries(
        int coefficient_count,
        double arclength,
        int maximum_derivative) const
    {
        return periodic
            ? periodic_basis_entries(degree,
                                     coefficient_count,
                                     arclength,
                                     length,
                                     maximum_derivative)
            : open_system.basis_entries(coefficient_count,
                                        arclength,
                                        maximum_derivative);
    }
};

struct ComponentPlan {
    int interface_component = -1;
    bool periodic = true;
    double length = 0.0;
    std::vector<int> ordered_panels;
    std::vector<int> sample_points;
    std::vector<double> sample_arclength;
    // NSP-CJ fields. sample_points are the interior density DOFs on one
    // complete NURBS knot span; sample_arclength stores their global NURBS
    // parameters. The spline interpolation rows additionally include both
    // true span endpoints.
    int nurbs_span = -1;
    double nurbs_parameter_start = 0.0;
    double nurbs_parameter_end = 0.0;
    SplineSystem phi_system;
    SplineSystem psi_system;
};

bool periodic_distance_close(double lhs,
                             double rhs,
                             double period,
                             double tolerance)
{
    double difference = std::abs(lhs - rhs);
    difference = std::min(difference, period - difference);
    return difference <= tolerance;
}

double lagrange_endpoint_value(const ComponentPlan& plan,
                               const Eigen::VectorXd& values,
                               bool left,
                               int requested_degree)
{
    const int available = static_cast<int>(plan.sample_points.size());
    const int count = std::min(available, requested_degree + 1);
    if (count < 1)
        throw std::invalid_argument(
            "NSP-CJ endpoint extrapolation has no density DOFs");
    const int first = left ? 0 : available - count;
    const double target = left
        ? plan.nurbs_parameter_start : plan.nurbs_parameter_end;
    double result = 0.0;
    for (int local = 0; local < count; ++local) {
        const int i = first + local;
        const double node =
            plan.sample_arclength[static_cast<std::size_t>(i)];
        double basis = 1.0;
        for (int other = 0; other < count; ++other) {
            if (other == local)
                continue;
            const int j = first + other;
            const double other_node =
                plan.sample_arclength[static_cast<std::size_t>(j)];
            const double denominator = node - other_node;
            if (std::abs(denominator) <= 1.0e-14
                    * std::max(1.0,
                               std::abs(plan.nurbs_parameter_end
                                        - plan.nurbs_parameter_start))) {
                throw std::invalid_argument(
                    "NSP-CJ endpoint extrapolation has repeated parameters");
            }
            basis *= (target - other_node) / denominator;
        }
        result += values[plan.sample_points[static_cast<std::size_t>(i)]]
                * basis;
    }
    return result;
}

double local_parameter_interpolation(const ComponentPlan& plan,
                                     const Eigen::VectorXd& values,
                                     double parameter,
                                     int degree)
{
    const int available = static_cast<int>(plan.sample_points.size());
    const int count = std::min(available, degree + 1);
    if (count < 1)
        throw std::invalid_argument("NSP-CJ interpolation has no samples");
    std::vector<int> ordering(static_cast<std::size_t>(available));
    for (int i = 0; i < available; ++i)
        ordering[static_cast<std::size_t>(i)] = i;
    std::stable_sort(ordering.begin(), ordering.end(),
                     [&](int lhs, int rhs) {
                         return std::abs(
                                    plan.sample_arclength[
                                        static_cast<std::size_t>(lhs)]
                                    - parameter)
                              < std::abs(
                                    plan.sample_arclength[
                                        static_cast<std::size_t>(rhs)]
                                    - parameter);
                     });
    ordering.resize(static_cast<std::size_t>(count));
    std::sort(ordering.begin(), ordering.end());
    double result = 0.0;
    for (int local = 0; local < count; ++local) {
        const int i = ordering[static_cast<std::size_t>(local)];
        const double node =
            plan.sample_arclength[static_cast<std::size_t>(i)];
        double basis = 1.0;
        for (int other = 0; other < count; ++other) {
            if (other == local)
                continue;
            const int j = ordering[static_cast<std::size_t>(other)];
            const double other_node =
                plan.sample_arclength[static_cast<std::size_t>(j)];
            basis *= (parameter - other_node) / (node - other_node);
        }
        result += values[plan.sample_points[static_cast<std::size_t>(i)]]
                * basis;
    }
    return result;
}

} // namespace

struct LaplaceArcLengthBSplineCrossingJetPlan2D::Impl {
    explicit Impl(const Interface2D& source_iface,
                  LaplaceCrossingJetScheme2D source_scheme)
        : iface(source_iface)
        , scheme(source_scheme)
    {
        component_to_plan.assign(
            static_cast<std::size_t>(iface.num_components()), -1);
        panel_to_plan.assign(
            static_cast<std::size_t>(iface.num_panels()), -1);
        panel_start_arclength.assign(
            static_cast<std::size_t>(iface.num_panels()), 0.0);
        if (!geometry2d::is_quadratic_lagrange_panel_layout(iface)) {
            return;
        }

        if (scheme
            == LaplaceCrossingJetScheme2D::NurbsSameParameterCrossingJet) {
            if (!iface.has_panel_geometry())
                return;
            nurbs_geometry = dynamic_cast<
                const geometry2d::NurbsBoundaryPanelGeometry2D*>(
                    &iface.panel_geometry());
            if (nurbs_geometry == nullptr
                || nurbs_geometry->num_parameterized_points()
                       != iface.num_points()) {
                return;
            }
            const auto invalidate_nurbs_plan = [&]() {
                component_plans.clear();
                std::fill(component_to_plan.begin(),
                          component_to_plan.end(), -1);
                std::fill(panel_to_plan.begin(), panel_to_plan.end(), -1);
                nurbs_geometry = nullptr;
            };

            for (int span = 0; span < nurbs_geometry->num_spans(); ++span) {
                auto plan = std::make_unique<ComponentPlan>();
                plan->periodic = nurbs_geometry->closed()
                              && nurbs_geometry->num_spans() == 1;
                plan->nurbs_span = span;
                const auto span_interval =
                    nurbs_geometry->span_interval(span);
                plan->nurbs_parameter_start = span_interval.first;
                plan->nurbs_parameter_end = span_interval.second;
                plan->length = span_interval.second - span_interval.first;

                for (int point = 0; point < iface.num_points(); ++point) {
                    if (iface.is_corner_point(point)
                        || nurbs_geometry->point_span(point) != span) {
                        continue;
                    }
                    const double parameter =
                        nurbs_geometry->point_parameter(point);
                    const double tolerance = 1.0e-12
                        * std::max(1.0, plan->length);
                    if (parameter <= span_interval.first + tolerance
                        || parameter >= span_interval.second - tolerance) {
                        continue;
                    }
                    plan->sample_points.push_back(point);
                    plan->sample_arclength.push_back(parameter);
                }
                std::vector<std::size_t> permutation(
                    plan->sample_points.size());
                for (std::size_t i = 0; i < permutation.size(); ++i)
                    permutation[i] = i;
                std::sort(permutation.begin(), permutation.end(),
                          [&](std::size_t lhs, std::size_t rhs) {
                              return plan->sample_arclength[lhs]
                                   < plan->sample_arclength[rhs];
                          });
                std::vector<int> sorted_points;
                std::vector<double> sorted_parameters;
                sorted_points.reserve(permutation.size());
                sorted_parameters.reserve(permutation.size());
                for (std::size_t index : permutation) {
                    sorted_points.push_back(plan->sample_points[index]);
                    sorted_parameters.push_back(
                        plan->sample_arclength[index]);
                }
                plan->sample_points = std::move(sorted_points);
                plan->sample_arclength = std::move(sorted_parameters);
                if ((plan->periodic && plan->sample_points.size() < 5)
                    || (!plan->periodic
                        && plan->sample_points.size() < 3)) {
                    invalidate_nurbs_plan();
                    return;
                }

                std::vector<double> interpolation_parameters;
                interpolation_parameters.reserve(plan->sample_points.size()
                    + (plan->periodic ? 0 : 2));
                if (!plan->periodic)
                    interpolation_parameters.push_back(0.0);
                for (double parameter : plan->sample_arclength) {
                    interpolation_parameters.push_back(
                        parameter - span_interval.first);
                }
                if (!plan->periodic)
                    interpolation_parameters.push_back(plan->length);
                if (!plan->phi_system.build(
                        3,
                        interpolation_parameters,
                        plan->length,
                        plan->periodic)
                    || !plan->psi_system.build(
                        2,
                        interpolation_parameters,
                        plan->length,
                        plan->periodic)) {
                    invalidate_nurbs_plan();
                    return;
                }

                for (int panel = 0; panel < iface.num_panels(); ++panel) {
                    if (nurbs_geometry->panel_span(panel) != span)
                        continue;
                    plan->ordered_panels.push_back(panel);
                }
                if (plan->ordered_panels.empty()) {
                    invalidate_nurbs_plan();
                    return;
                }
                const int plan_index =
                    static_cast<int>(component_plans.size());
                plan->interface_component =
                    iface.panel_components()[plan->ordered_panels.front()];
                if (component_to_plan[static_cast<std::size_t>(
                        plan->interface_component)] < 0) {
                    component_to_plan[static_cast<std::size_t>(
                        plan->interface_component)] = plan_index;
                }
                for (int panel : plan->ordered_panels) {
                    panel_to_plan[static_cast<std::size_t>(panel)] =
                        plan_index;
                    panel_start_arclength[static_cast<std::size_t>(panel)] =
                        span_interval.first;
                }
                component_plans.push_back(std::move(plan));
            }
            if (static_cast<int>(component_plans.size())
                != nurbs_geometry->num_spans()) {
                invalidate_nurbs_plan();
            }
            return;
        }

        const auto add_plan = [&](int component,
                                  const std::vector<int>& ordered,
                                  bool periodic) {
            if (ordered.empty())
                return false;
            auto plan = std::make_unique<ComponentPlan>();
            plan->interface_component = component;
            plan->periodic = periodic;
            plan->ordered_panels = ordered;
            std::vector<double> point_arclength(
                static_cast<std::size_t>(iface.num_points()),
                std::numeric_limits<double>::quiet_NaN());
            double cumulative = 0.0;
            for (int current : ordered) {
                panel_start_arclength[
                    static_cast<std::size_t>(current)] = cumulative;
                const double half = panel_arclength(
                    iface, current, -1.0, 0.0);
                const double full = panel_arclength(
                    iface, current, -1.0, 1.0);
                const std::array<int, 3> points{{
                    iface.point_index(current, 0),
                    iface.point_index(current, 1),
                    iface.point_index(current, 2)}};
                const std::array<double, 3> positions{{
                    cumulative, cumulative + half, cumulative + full}};
                for (std::size_t local = 0; local < points.size(); ++local) {
                    double& saved = point_arclength[
                        static_cast<std::size_t>(points[local])];
                    if (!std::isfinite(saved))
                        saved = positions[local];
                }
                cumulative += full;
            }
            plan->length = cumulative;
            if (!(plan->length > 1.0e-14))
                return false;

            for (int point = 0; point < iface.num_points(); ++point) {
                double arclength = point_arclength[
                    static_cast<std::size_t>(point)];
                if (!std::isfinite(arclength))
                    continue;
                if (periodic) {
                    arclength = std::fmod(arclength, plan->length);
                    if (arclength < 0.0)
                        arclength += plan->length;
                }
                plan->sample_points.push_back(point);
                plan->sample_arclength.push_back(arclength);
            }
            std::vector<std::size_t> permutation(plan->sample_points.size());
            for (std::size_t i = 0; i < permutation.size(); ++i)
                permutation[i] = i;
            std::sort(permutation.begin(), permutation.end(),
                      [&](std::size_t lhs, std::size_t rhs) {
                          return plan->sample_arclength[lhs]
                               < plan->sample_arclength[rhs];
                      });
            std::vector<int> sorted_points;
            std::vector<double> sorted_arclength;
            sorted_points.reserve(permutation.size());
            sorted_arclength.reserve(permutation.size());
            for (std::size_t index : permutation) {
                sorted_points.push_back(plan->sample_points[index]);
                sorted_arclength.push_back(plan->sample_arclength[index]);
            }
            plan->sample_points = std::move(sorted_points);
            plan->sample_arclength = std::move(sorted_arclength);

            if (plan->sample_points.size() < 5)
                return false;
            const double point_tolerance = 1.0e-12 * plan->length;
            for (std::size_t i = 1;
                 i < plan->sample_arclength.size();
                 ++i) {
                if (!(plan->sample_arclength[i]
                      - plan->sample_arclength[i - 1]
                      > point_tolerance)) {
                    return false;
                }
            }
            if (periodic
                && periodic_distance_close(
                    plan->sample_arclength.front(),
                    plan->sample_arclength.back(),
                    plan->length,
                    point_tolerance)) {
                return false;
            }
            if (!periodic
                && (std::abs(plan->sample_arclength.front())
                        > point_tolerance
                    || std::abs(plan->sample_arclength.back()
                                - plan->length)
                           > point_tolerance)) {
                return false;
            }
            if (!plan->phi_system.build(
                    3, plan->sample_arclength, plan->length, periodic)
                || !plan->psi_system.build(
                    2, plan->sample_arclength, plan->length, periodic)) {
                return false;
            }

            const int plan_index = static_cast<int>(component_plans.size());
            if (component_to_plan[static_cast<std::size_t>(component)] < 0) {
                component_to_plan[static_cast<std::size_t>(component)] =
                    plan_index;
            }
            for (int current : ordered) {
                panel_to_plan[static_cast<std::size_t>(current)] = plan_index;
            }
            component_plans.push_back(std::move(plan));
            return true;
        };

        const auto smooth_connection = [&](int current, int next) {
            const int right = iface.point_index(current, 2);
            const int left = iface.point_index(next, 0);
            if (right != left || iface.is_corner_point(right))
                return false;
            const Eigen::Vector2d current_point =
                geometry2d::panel_point(iface, current, 1.0);
            const Eigen::Vector2d next_point =
                geometry2d::panel_point(iface, next, -1.0);
            const double geometry_scale = std::max(
                1.0, std::max(current_point.norm(), next_point.norm()));
            Eigen::Vector2d current_tangent =
                geometry2d::panel_tangent(iface, current, 1.0);
            Eigen::Vector2d next_tangent =
                geometry2d::panel_tangent(iface, next, -1.0);
            if (!(current_tangent.norm() > 1.0e-14)
                || !(next_tangent.norm() > 1.0e-14)) {
                return false;
            }
            current_tangent.normalize();
            next_tangent.normalize();
            return (current_point - next_point).norm()
                       <= 1.0e-10 * geometry_scale
                && current_tangent.dot(next_tangent) >= 1.0 - 1.0e-9;
        };

        for (int component = 0;
             component < iface.num_components();
             ++component) {
            std::vector<int> panels;
            for (int panel = 0; panel < iface.num_panels(); ++panel) {
                if (iface.panel_components()[panel] == component)
                    panels.push_back(panel);
            }
            if (panels.empty())
                continue;

            std::unordered_map<int, int> panel_from_left_endpoint;
            for (int panel : panels) {
                const int left = iface.point_index(panel, 0);
                const auto inserted =
                    panel_from_left_endpoint.emplace(left, panel);
                if (!inserted.second)
                    inserted.first->second = -1;
            }
            std::vector<int> next_by_panel(
                static_cast<std::size_t>(iface.num_panels()), -1);
            std::vector<int> previous_by_panel(
                static_cast<std::size_t>(iface.num_panels()), -1);
            for (int panel : panels) {
                const int right = iface.point_index(panel, 2);
                const auto found = panel_from_left_endpoint.find(right);
                if (found == panel_from_left_endpoint.end()
                    || found->second < 0
                    || !smooth_connection(panel, found->second)) {
                    continue;
                }
                const int next = found->second;
                int& previous = previous_by_panel[
                    static_cast<std::size_t>(next)];
                if (previous >= 0) {
                    next_by_panel[static_cast<std::size_t>(previous)] = -1;
                    previous = -2;
                    continue;
                }
                if (previous == -2)
                    continue;
                next_by_panel[static_cast<std::size_t>(panel)] = next;
                previous = panel;
            }

            std::vector<char> visited(
                static_cast<std::size_t>(iface.num_panels()), 0);
            for (int start : panels) {
                if (previous_by_panel[static_cast<std::size_t>(start)] >= 0)
                    continue;
                std::vector<int> ordered;
                int current = start;
                while (current >= 0
                       && !visited[static_cast<std::size_t>(current)]) {
                    visited[static_cast<std::size_t>(current)] = 1;
                    ordered.push_back(current);
                    current = next_by_panel[static_cast<std::size_t>(current)];
                }
                add_plan(component, ordered, false);
            }
            for (int start : panels) {
                if (visited[static_cast<std::size_t>(start)])
                    continue;
                std::vector<int> ordered;
                int current = start;
                while (current >= 0
                       && !visited[static_cast<std::size_t>(current)]) {
                    visited[static_cast<std::size_t>(current)] = 1;
                    ordered.push_back(current);
                    current = next_by_panel[static_cast<std::size_t>(current)];
                }
                const bool periodic = current == start;
                add_plan(component, ordered, periodic);
            }
        }
    }

    int plan_for_crossing(const P2CrossingOwner2D& crossing) const
    {
        if (crossing.panel_index < 0
            || crossing.panel_index >= iface.num_panels()) {
            return -1;
        }
        return panel_to_plan[
            static_cast<std::size_t>(crossing.panel_index)];
    }

    double crossing_arclength(
        const P2CrossingOwner2D& crossing,
        int plan_index) const
    {
        const ComponentPlan& plan =
            *component_plans[static_cast<std::size_t>(plan_index)];
        if (nurbs_geometry != nullptr && plan.nurbs_span >= 0) {
            geometry2d::NurbsSpanProjection2D projection;
            if (crossing.status
                    == P2CrossingOwnerStatus2D::ExactIntersection
                && crossing.local_s >= -1.0 - 1.0e-10
                && crossing.local_s <= 1.0 + 1.0e-10) {
                projection.span = plan.nurbs_span;
                projection.parameter = nurbs_geometry->panel_parameter(
                    crossing.panel_index, crossing.local_s);
                projection.point = nurbs_geometry->curve().evaluate(
                    projection.parameter);
                projection.distance =
                    (projection.point - crossing.crossing_point).norm();
                projection.converged = true;
            } else {
                projection = nurbs_geometry->project_to_span(
                    plan.nurbs_span, crossing.crossing_point);
            }
            const double geometry_scale = std::max(
                1.0, crossing.crossing_point.norm());
            if (!projection.converged
                || projection.distance > 1.0e-8 * geometry_scale) {
                throw std::invalid_argument(
                    "NSP-CJ crossing does not lie on its NURBS span");
            }
            return projection.parameter - plan.nurbs_parameter_start;
        }
        double arclength =
            panel_start_arclength[
                static_cast<std::size_t>(crossing.panel_index)]
            + panel_arclength(
                iface,
                crossing.panel_index,
                -1.0,
                geometry2d::clamp_to_panel(crossing.local_s));
        if (plan.periodic) {
            arclength = std::fmod(arclength, plan.length);
            if (arclength < 0.0)
                arclength += plan.length;
        } else {
            arclength = std::clamp(arclength, 0.0, plan.length);
        }
        return arclength;
    }

    const Interface2D& iface;
    LaplaceCrossingJetScheme2D scheme =
        LaplaceCrossingJetScheme2D::ArcLengthBSplineCrossingJet;
    const geometry2d::NurbsBoundaryPanelGeometry2D* nurbs_geometry = nullptr;
    std::vector<std::unique_ptr<ComponentPlan>> component_plans;
    std::vector<int> component_to_plan;
    std::vector<int> panel_to_plan;
    std::vector<double> panel_start_arclength;
};

LaplaceArcLengthBSplineCrossingJetPlan2D::
LaplaceArcLengthBSplineCrossingJetPlan2D(
    const Interface2D& iface,
    LaplaceCrossingJetScheme2D scheme)
    : impl_(std::make_shared<Impl>(iface, scheme))
{}

bool LaplaceArcLengthBSplineCrossingJetPlan2D::can_evaluate(
    const P2CrossingOwner2D& crossing,
    LaplaceCrossingTraceStencil2D trace_stencil) const
{
    if (!impl_ || !is_valid_crossing_trace_stencil(trace_stencil)
        || !std::isfinite(crossing.local_s)
        || !crossing.crossing_point.allFinite()) {
        return false;
    }
    const int plan_index = impl_->plan_for_crossing(crossing);
    if (plan_index < 0)
        return false;
    if (impl_->nurbs_geometry != nullptr) {
        const bool identified_crossing =
            (crossing.status
                 == P2CrossingOwnerStatus2D::ExactIntersection
             && crossing.exact_intersection_count == 1)
            || (crossing.status
                    == P2CrossingOwnerStatus2D::GapFallback
                && crossing.explicit_gap_intersection);
        if (!identified_crossing)
            return false;
        try {
            (void)impl_->crossing_arclength(crossing, plan_index);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }
    return crossing.status == P2CrossingOwnerStatus2D::ExactIntersection
        && crossing.exact_intersection_count == 1
        && crossing.local_s >= -1.0 - 1.0e-10
        && crossing.local_s <= 1.0 + 1.0e-10;
}

LaplaceArcLengthBSplineTraceState2D
LaplaceArcLengthBSplineCrossingJetPlan2D::fit(
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump,
    LaplaceCrossingTraceStencil2D trace_stencil) const
{
    if (!impl_ || !is_valid_crossing_trace_stencil(trace_stencil)
        || value_jump.size() != impl_->iface.num_points()
        || normal_jump.size() != impl_->iface.num_points()) {
        throw std::invalid_argument(
            "ALS-CJ fit requires one finite jump value per interface DOF");
    }
    LaplaceArcLengthBSplineTraceState2D state;
    state.trace_stencil = trace_stencil;
    state.phi_coefficients_by_component.resize(
        impl_->component_plans.size());
    state.psi_coefficients_by_component.resize(
        impl_->component_plans.size());

    if (impl_->nurbs_geometry != nullptr) {
        const std::size_t plan_count = impl_->component_plans.size();
        std::vector<double> phi_left(plan_count, 0.0);
        std::vector<double> phi_right(plan_count, 0.0);
        std::vector<double> psi_left(plan_count, 0.0);
        std::vector<double> psi_right(plan_count, 0.0);
        for (std::size_t index = 0; index < plan_count; ++index) {
            const ComponentPlan& plan =
                *impl_->component_plans[index];
            if (plan.periodic)
                continue;
            if (crossing_trace_uses_phi_p3(trace_stencil)) {
                phi_left[index] = lagrange_endpoint_value(
                    plan, value_jump, true, 3);
                phi_right[index] = lagrange_endpoint_value(
                    plan, value_jump, false, 3);
            }
            if (crossing_trace_uses_psi_p2(trace_stencil)) {
                psi_left[index] = lagrange_endpoint_value(
                    plan, normal_jump, true, 2);
                psi_right[index] = lagrange_endpoint_value(
                    plan, normal_jump, false, 2);
            }
        }
        if (crossing_trace_uses_phi_p3(trace_stencil)
            && impl_->nurbs_geometry->closed() && plan_count > 1) {
            // Phi is a scalar trace value, so the two one-sided extrapolants
            // share one derived corner value. Psi is a normal derivative and
            // intentionally remains one-sided because the normal jumps at a
            // geometric corner.
            std::vector<double> shared(plan_count, 0.0);
            for (std::size_t corner = 0; corner < plan_count; ++corner) {
                const std::size_t previous =
                    (corner + plan_count - 1) % plan_count;
                shared[corner] =
                    0.5 * (phi_right[previous] + phi_left[corner]);
            }
            for (std::size_t index = 0; index < plan_count; ++index) {
                phi_left[index] = shared[index];
                phi_right[index] = shared[(index + 1) % plan_count];
            }
        }

        for (std::size_t index = 0; index < plan_count; ++index) {
            const ComponentPlan& plan =
                *impl_->component_plans[index];
            const int interior_count =
                static_cast<int>(plan.sample_points.size());
            if (plan.periodic) {
                if (crossing_trace_uses_phi_p3(trace_stencil)) {
                    Eigen::VectorXd samples(interior_count);
                    for (int i = 0; i < interior_count; ++i) {
                        samples[i] = value_jump[plan.sample_points[
                            static_cast<std::size_t>(i)]];
                    }
                    state.phi_coefficients_by_component[index] =
                        plan.phi_system.fit(samples);
                }
                if (crossing_trace_uses_psi_p2(trace_stencil)) {
                    Eigen::VectorXd samples(interior_count);
                    for (int i = 0; i < interior_count; ++i) {
                        samples[i] = normal_jump[plan.sample_points[
                            static_cast<std::size_t>(i)]];
                    }
                    state.psi_coefficients_by_component[index] =
                        plan.psi_system.fit(samples);
                }
                continue;
            }
            if (crossing_trace_uses_phi_p3(trace_stencil)) {
                Eigen::VectorXd samples(interior_count + 2);
                samples[0] = phi_left[index];
                for (int i = 0; i < interior_count; ++i) {
                    samples[i + 1] =
                        value_jump[plan.sample_points[
                            static_cast<std::size_t>(i)]];
                }
                samples[interior_count + 1] = phi_right[index];
                state.phi_coefficients_by_component[index] =
                    plan.phi_system.fit(samples);
            }
            if (crossing_trace_uses_psi_p2(trace_stencil)) {
                Eigen::VectorXd samples(interior_count + 2);
                samples[0] = psi_left[index];
                for (int i = 0; i < interior_count; ++i) {
                    samples[i + 1] =
                        normal_jump[plan.sample_points[
                            static_cast<std::size_t>(i)]];
                }
                samples[interior_count + 1] = psi_right[index];
                state.psi_coefficients_by_component[index] =
                    plan.psi_system.fit(samples);
            }
        }
        return state;
    }

    for (std::size_t component = 0;
         component < impl_->component_plans.size();
         ++component) {
        const ComponentPlan& plan = *impl_->component_plans[component];
        const int n = static_cast<int>(plan.sample_points.size());
        if (crossing_trace_uses_phi_p3(trace_stencil)) {
            Eigen::VectorXd samples(n);
            for (int i = 0; i < n; ++i)
                samples[i] = value_jump[plan.sample_points[i]];
            if (!samples.allFinite())
                throw std::invalid_argument("ALS-CJ phi samples are nonfinite");
            state.phi_coefficients_by_component[component] =
                plan.phi_system.fit(samples);
        }
        if (crossing_trace_uses_psi_p2(trace_stencil)) {
            Eigen::VectorXd samples(n);
            for (int i = 0; i < n; ++i)
                samples[i] = normal_jump[plan.sample_points[i]];
            if (!samples.allFinite())
                throw std::invalid_argument("ALS-CJ psi samples are nonfinite");
            state.psi_coefficients_by_component[component] =
                plan.psi_system.fit(samples);
        }
    }
    return state;
}

LaplaceCrossingTraceJet2D
LaplaceArcLengthBSplineCrossingJetPlan2D::evaluate(
    const P2CrossingOwner2D& crossing,
    const LaplaceArcLengthBSplineTraceState2D& state) const
{
    if (!can_evaluate(crossing, state.trace_stencil)) {
        throw std::invalid_argument(
            "ALS-CJ cannot evaluate this crossing");
    }
    const int plan_index = impl_->plan_for_crossing(crossing);
    const ComponentPlan& plan =
        *impl_->component_plans[static_cast<std::size_t>(plan_index)];
    const double arclength =
        impl_->crossing_arclength(crossing, plan_index);

    LaplaceCrossingTraceJet2D jet;
    if (crossing_trace_uses_phi_p3(state.trace_stencil)) {
        const Eigen::VectorXd& coefficients =
            state.phi_coefficients_by_component[
                static_cast<std::size_t>(plan_index)];
        const auto entries = plan.phi_system.basis_entries(
            static_cast<int>(coefficients.size()),
            arclength,
            2);
        for (const PeriodicBasisEntry& entry : entries) {
            const double coefficient = coefficients[entry.coefficient];
            jet.value += coefficient * entry.derivative[0];
            jet.value_tangent_derivative +=
                coefficient * entry.derivative[1];
            jet.value_tangent_second_derivative +=
                coefficient * entry.derivative[2];
        }
    }
    if (crossing_trace_uses_psi_p2(state.trace_stencil)) {
        const Eigen::VectorXd& coefficients =
            state.psi_coefficients_by_component[
                static_cast<std::size_t>(plan_index)];
        const auto entries = plan.psi_system.basis_entries(
            static_cast<int>(coefficients.size()),
            arclength,
            1);
        for (const PeriodicBasisEntry& entry : entries) {
            const double coefficient = coefficients[entry.coefficient];
            jet.normal_value += coefficient * entry.derivative[0];
            jet.normal_tangent_derivative +=
                coefficient * entry.derivative[1];
        }
    }
    if (impl_->nurbs_geometry != nullptr && plan.nurbs_span >= 0) {
        const double parameter = plan.nurbs_parameter_start + arclength;
        const geometry2d::NurbsBoundaryGeometry2D geometry =
            impl_->nurbs_geometry->evaluate_on_span(
                plan.nurbs_span, parameter);
        const double inverse_speed = 1.0 / geometry.speed;
        const double speed_parameter_derivative =
            geometry.parameter_tangent.dot(geometry.parameter_second)
            * inverse_speed;
        const double value_parameter_first =
            jet.value_tangent_derivative;
        const double value_parameter_second =
            jet.value_tangent_second_derivative;
        jet.value_tangent_derivative =
            value_parameter_first * inverse_speed;
        jet.value_tangent_second_derivative =
            value_parameter_second * inverse_speed * inverse_speed
            - value_parameter_first * speed_parameter_derivative
                * inverse_speed * inverse_speed * inverse_speed;
        jet.normal_tangent_derivative *= inverse_speed;
    }
    const Eigen::Matrix<double, 5, 1> values =
        (Eigen::Matrix<double, 5, 1>()
             << jet.value,
                jet.value_tangent_derivative,
                jet.value_tangent_second_derivative,
                jet.normal_value,
                jet.normal_tangent_derivative)
            .finished();
    if (!values.allFinite())
        throw std::runtime_error("ALS-CJ produced a nonfinite trace jet");
    return jet;
}

LaplaceP2CrossingLocalPolynomial2D
LaplaceArcLengthBSplineCrossingJetPlan2D::build_local_polynomial(
    const P2CrossingOwner2D& crossing,
    const LaplaceArcLengthBSplineTraceState2D& state,
    const Eigen::VectorXd& rhs_jump,
    double alpha) const
{
    if (!can_evaluate(crossing, state.trace_stencil)
        || !impl_ || rhs_jump.size() != impl_->iface.num_points()
        || !rhs_jump.allFinite() || !std::isfinite(alpha)) {
        throw std::invalid_argument(
            "crossing-jet local polynomial has invalid input");
    }
    const LaplaceCrossingTraceJet2D trace = evaluate(crossing, state);
    const int plan_index = impl_->plan_for_crossing(crossing);
    const ComponentPlan& plan =
        *impl_->component_plans[static_cast<std::size_t>(plan_index)];
    if (impl_->nurbs_geometry == nullptr || plan.nurbs_span < 0) {
        return build_laplace_p2_crossing_local_polynomial_from_trace_jet_2d(
            impl_->iface,
            crossing.panel_index,
            crossing.local_s,
            trace,
            rhs_jump,
            alpha);
    }

    const double relative_parameter =
        impl_->crossing_arclength(crossing, plan_index);
    const double parameter =
        plan.nurbs_parameter_start + relative_parameter;
    const geometry2d::NurbsBoundaryGeometry2D nurbs =
        impl_->nurbs_geometry->evaluate_on_span(
            plan.nurbs_span, parameter);
    LaplaceCrossingGeometryJet2D geometry;
    geometry.center = nurbs.point;
    geometry.tangent = nurbs.tangent;
    geometry.normal = nurbs.normal;
    geometry.curvature = nurbs.curvature;
    // The PDE closure needs only [f] at the crossing. A local P2 evaluation
    // in the same NURBS parameter gives O(h^3) value error and therefore does
    // not reduce the requested O(h^2) extension accuracy.
    geometry.forcing = local_parameter_interpolation(
        plan, rhs_jump, parameter, 2);
    return build_laplace_crossing_local_polynomial_from_geometry_trace_jet_2d(
        geometry, trace, alpha);
}

bool LaplaceArcLengthBSplineCrossingJetPlan2D::
uses_nurbs_same_parameter() const
{
    return impl_ && impl_->nurbs_geometry != nullptr
        && !impl_->component_plans.empty();
}

int LaplaceArcLengthBSplineCrossingJetPlan2D::nurbs_span_count() const
{
    return uses_nurbs_same_parameter()
        ? static_cast<int>(impl_->component_plans.size()) : 0;
}

int LaplaceArcLengthBSplineCrossingJetPlan2D::
smooth_component_count() const
{
    return impl_ ? static_cast<int>(impl_->component_plans.size()) : 0;
}

int LaplaceArcLengthBSplineCrossingJetPlan2D::
periodic_component_count() const
{
    if (!impl_)
        return 0;
    return static_cast<int>(std::count_if(
        impl_->component_plans.begin(),
        impl_->component_plans.end(),
        [](const std::unique_ptr<ComponentPlan>& plan) {
            return plan->periodic;
        }));
}

int LaplaceArcLengthBSplineCrossingJetPlan2D::open_branch_count() const
{
    if (!impl_)
        return 0;
    return static_cast<int>(std::count_if(
        impl_->component_plans.begin(),
        impl_->component_plans.end(),
        [](const std::unique_ptr<ComponentPlan>& plan) {
            return !plan->periodic;
        }));
}

int LaplaceArcLengthBSplineCrossingJetPlan2D::
total_component_count() const
{
    return impl_ ? impl_->iface.num_components() : 0;
}

} // namespace kfbim
