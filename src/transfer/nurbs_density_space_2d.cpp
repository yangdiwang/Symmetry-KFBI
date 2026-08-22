#include "nurbs_density_space_2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/QR>
#include <Eigen/SVD>

#include "../geometry/nurbs_basis.hpp"

namespace kfbim {
namespace {

using BoundaryGeometry = geometry2d::NurbsBoundaryPanelGeometry2D;

constexpr std::array<double, 8> kGaussNodes{{
    -0.96028985649753623168,
    -0.79666647741362673959,
    -0.52553240991632898582,
    -0.18343464249564980494,
     0.18343464249564980494,
     0.52553240991632898582,
     0.79666647741362673959,
     0.96028985649753623168}};

constexpr std::array<double, 8> kGaussWeights{{
    0.10122853629037625915,
    0.22238103445337447054,
    0.31370664587788728734,
    0.36268378337836198297,
    0.36268378337836198297,
    0.31370664587788728734,
    0.22238103445337447054,
    0.10122853629037625915}};

void require_branch(int branch, int count)
{
    if (branch < 0 || branch >= count)
        throw std::out_of_range("NURBS density branch is out of range");
}

double gauss_speed_integral(const BoundaryGeometry& geometry,
                            int branch,
                            double a,
                            double b)
{
    if (!(b > a))
        return 0.0;
    const double midpoint = 0.5 * (a + b);
    const double half = 0.5 * (b - a);
    double integral = 0.0;
    for (std::size_t q = 0; q < kGaussNodes.size(); ++q) {
        const double parameter = midpoint + half * kGaussNodes[q];
        const double speed =
            geometry.evaluate_on_span(branch, parameter).speed;
        if (!(speed > 0.0) || !std::isfinite(speed)) {
            throw std::runtime_error(
                "NURBS density arclength integration found degenerate geometry");
        }
        integral += kGaussWeights[q] * speed;
    }
    return half * integral;
}

double adaptive_speed_integral(const BoundaryGeometry& geometry,
                               int branch,
                               double a,
                               double b,
                               double coarse,
                               double tolerance,
                               int depth)
{
    const double midpoint = 0.5 * (a + b);
    const double left = gauss_speed_integral(
        geometry, branch, a, midpoint);
    const double right = gauss_speed_integral(
        geometry, branch, midpoint, b);
    const double fine = left + right;
    if (depth <= 0
        || std::abs(fine - coarse)
               <= tolerance * std::max(1.0, std::abs(fine))) {
        return fine;
    }
    return adaptive_speed_integral(
               geometry,
               branch,
               a,
               midpoint,
               left,
               0.5 * tolerance,
               depth - 1)
         + adaptive_speed_integral(
               geometry,
               branch,
               midpoint,
               b,
               right,
               0.5 * tolerance,
               depth - 1);
}

double arclength_between(const BoundaryGeometry& geometry,
                         int branch,
                         double a,
                         double b)
{
    if (a == b)
        return 0.0;
    double sign = 1.0;
    if (b < a) {
        std::swap(a, b);
        sign = -1.0;
    }

    std::vector<double> pieces{a, b};
    for (double knot : geometry.curve().basis().knots()) {
        if (knot > a && knot < b)
            pieces.push_back(knot);
    }
    std::sort(pieces.begin(), pieces.end());
    pieces.erase(
        std::unique(pieces.begin(), pieces.end()), pieces.end());

    double integral = 0.0;
    for (std::size_t interval = 1; interval < pieces.size(); ++interval) {
        const double left = pieces[interval - 1];
        const double right = pieces[interval];
        if (!(right > left))
            continue;
        const double coarse = gauss_speed_integral(
            geometry, branch, left, right);
        integral += adaptive_speed_integral(
            geometry,
            branch,
            left,
            right,
            coarse,
            2.0e-13,
            10);
    }
    return sign * integral;
}

double parameter_at_arclength(const BoundaryGeometry& geometry,
                              int branch,
                              double parameter_start,
                              double parameter_end,
                              double branch_length,
                              double target)
{
    if (target <= 0.0)
        return parameter_start;
    if (target >= branch_length)
        return parameter_end;

    double left = parameter_start;
    double right = parameter_end;
    const double length_tolerance =
        2.0e-13 * std::max(1.0, branch_length);
    const double parameter_tolerance =
        32.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(parameter_end - parameter_start));
    for (int iteration = 0; iteration < 80; ++iteration) {
        const double midpoint = 0.5 * (left + right);
        if (midpoint == left || midpoint == right)
            break;
        const double length = arclength_between(
            geometry, branch, parameter_start, midpoint);
        if (std::abs(length - target) <= length_tolerance)
            return midpoint;
        if (length < target)
            left = midpoint;
        else
            right = midpoint;
        if (right - left <= parameter_tolerance)
            break;
    }
    return 0.5 * (left + right);
}

std::vector<double> make_clamped_knots(
    const std::vector<double>& breaks,
    int degree)
{
    if (breaks.size() < 2)
        throw std::invalid_argument(
            "NURBS density spline needs at least one span");
    std::vector<double> knots;
    knots.reserve(breaks.size() + 2 * static_cast<std::size_t>(degree));
    knots.insert(knots.end(),
                 static_cast<std::size_t>(degree + 1),
                 breaks.front());
    for (std::size_t i = 1; i + 1 < breaks.size(); ++i)
        knots.push_back(breaks[i]);
    knots.insert(knots.end(),
                 static_cast<std::size_t>(degree + 1),
                 breaks.back());
    return knots;
}

std::vector<double> make_chart_aligned_parameter_knots(
    const std::vector<double>& density_breaks,
    const std::vector<double>& chart_breaks,
    int degree)
{
    if (density_breaks.size() < 2)
        throw std::invalid_argument(
            "chart-aligned density spline needs at least one span");
    const double start = density_breaks.front();
    const double end = density_breaks.back();
    const double tolerance = 2.0e-12
        * std::max(1.0, std::abs(end - start));

    std::vector<double> interior;
    interior.reserve(density_breaks.size() + chart_breaks.size());
    for (std::size_t i = 1; i + 1 < density_breaks.size(); ++i) {
        const double value = density_breaks[i];
        const bool coincides_with_chart = std::any_of(
            chart_breaks.begin(),
            chart_breaks.end(),
            [&](double chart) {
                return std::abs(value - chart) <= tolerance;
            });
        if (!coincides_with_chart)
            interior.push_back(value);
    }
    interior.insert(interior.end(), chart_breaks.begin(), chart_breaks.end());
    std::sort(interior.begin(), interior.end());
    interior.erase(
        std::unique(
            interior.begin(),
            interior.end(),
            [&](double lhs, double rhs) {
                return std::abs(lhs - rhs) <= tolerance;
            }),
        interior.end());

    std::vector<double> knots;
    knots.insert(knots.end(), static_cast<std::size_t>(degree + 1), start);
    for (double value : interior) {
        const bool is_chart = std::any_of(
            chart_breaks.begin(),
            chart_breaks.end(),
            [&](double chart) {
                return std::abs(value - chart) <= tolerance;
            });
        knots.insert(
            knots.end(),
            // Multiplicity p leaves the chart pieces C0 in xi.  Their
            // physical first/second jets are then connected explicitly;
            // NurbsBasis1D intentionally rejects an internal p+1 break.
            static_cast<std::size_t>(is_chart ? std::max(1, degree) : 1),
            value);
    }
    knots.insert(knots.end(), static_cast<std::size_t>(degree + 1), end);
    return knots;
}

std::vector<double> internal_geometry_chart_breaks(
    const BoundaryGeometry& geometry,
    double start,
    double end)
{
    const double tolerance = 2.0e-12
        * std::max(1.0, std::abs(end - start));
    std::vector<double> result;
    for (double knot : geometry.curve().basis().knots()) {
        if (knot > start + tolerance && knot < end - tolerance)
            result.push_back(knot);
    }
    std::sort(result.begin(), result.end());
    result.erase(
        std::unique(
            result.begin(),
            result.end(),
            [&](double lhs, double rhs) {
                return std::abs(lhs - rhs) <= tolerance;
            }),
        result.end());
    return result;
}

std::vector<double> make_chart_aligned_parameter_breaks(
    const BoundaryGeometry& geometry,
    int branch,
    double start,
    double end,
    int total_span_count,
    const std::vector<double>& chart_breaks)
{
    std::vector<double> boundaries;
    boundaries.reserve(chart_breaks.size() + 2);
    boundaries.push_back(start);
    boundaries.insert(
        boundaries.end(), chart_breaks.begin(), chart_breaks.end());
    boundaries.push_back(end);
    const int chart_count = static_cast<int>(boundaries.size()) - 1;
    if (total_span_count < chart_count) {
        throw std::invalid_argument(
            "density span count cannot cover every NURBS parameter chart");
    }

    std::vector<double> lengths(static_cast<std::size_t>(chart_count));
    double total_length = 0.0;
    for (int chart = 0; chart < chart_count; ++chart) {
        lengths[static_cast<std::size_t>(chart)] = arclength_between(
            geometry,
            branch,
            boundaries[static_cast<std::size_t>(chart)],
            boundaries[static_cast<std::size_t>(chart + 1)]);
        total_length += lengths[static_cast<std::size_t>(chart)];
    }

    std::vector<int> counts(static_cast<std::size_t>(chart_count), 1);
    std::vector<std::pair<double, int>> remainders;
    remainders.reserve(static_cast<std::size_t>(chart_count));
    const int extra_total = total_span_count - chart_count;
    int assigned_extra = 0;
    for (int chart = 0; chart < chart_count; ++chart) {
        const double ideal = static_cast<double>(extra_total)
            * lengths[static_cast<std::size_t>(chart)] / total_length;
        const int extra = static_cast<int>(std::floor(ideal));
        counts[static_cast<std::size_t>(chart)] += extra;
        assigned_extra += extra;
        remainders.emplace_back(ideal - static_cast<double>(extra), chart);
    }
    std::stable_sort(
        remainders.begin(), remainders.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.first > rhs.first;
        });
    for (int extra = assigned_extra; extra < extra_total; ++extra) {
        const int chart = remainders[
            static_cast<std::size_t>(extra - assigned_extra)].second;
        ++counts[static_cast<std::size_t>(chart)];
    }

    std::vector<double> result;
    result.reserve(static_cast<std::size_t>(total_span_count + 1));
    result.push_back(start);
    for (int chart = 0; chart < chart_count; ++chart) {
        const double chart_start =
            boundaries[static_cast<std::size_t>(chart)];
        const double chart_end =
            boundaries[static_cast<std::size_t>(chart + 1)];
        const double chart_length = lengths[static_cast<std::size_t>(chart)];
        const int count = counts[static_cast<std::size_t>(chart)];
        for (int split = 1; split < count; ++split) {
            result.push_back(parameter_at_arclength(
                geometry,
                branch,
                chart_start,
                chart_end,
                chart_length,
                chart_length * static_cast<double>(split)
                    / static_cast<double>(count)));
        }
        result.push_back(chart_end);
    }
    if (static_cast<int>(result.size()) != total_span_count + 1) {
        throw std::logic_error(
            "chart-aligned density break allocation has the wrong size");
    }
    return result;
}

std::vector<double> make_uniform_arclength_breaks(double length,
                                                   int span_count)
{
    std::vector<double> breaks(
        static_cast<std::size_t>(span_count + 1));
    for (int split = 0; split <= span_count; ++split) {
        breaks[static_cast<std::size_t>(split)] =
            length * static_cast<double>(split)
            / static_cast<double>(span_count);
    }
    return breaks;
}

std::vector<double> make_periodic_extension_knots(double length,
                                                  int span_count,
                                                  int degree)
{
    const double spacing = length / static_cast<double>(span_count);
    std::vector<double> knots;
    knots.reserve(static_cast<std::size_t>(
        span_count + 2 * degree + 1));
    for (int knot = -degree; knot <= span_count + degree; ++knot)
        knots.push_back(spacing * static_cast<double>(knot));
    return knots;
}

double wrap_periodic_coordinate(double coordinate, double period)
{
    coordinate = std::fmod(coordinate, period);
    if (coordinate < 0.0)
        coordinate += period;
    if (coordinate >= period)
        coordinate = 0.0;
    return coordinate;
}

int continuity_order(NurbsDensityContinuity2D continuity)
{
    return static_cast<int>(continuity);
}

} // namespace

struct NurbsDensitySpace2D::Impl {
    struct ArcLengthSegment {
        double parameter_start = 0.0;
        double parameter_end = 0.0;
        double arclength_start = 0.0;
        double arclength_end = 0.0;
        double speed_start = 0.0;
        double speed_end = 0.0;
    };

    struct BranchData {
        NurbsDensityBranchInfo2D info;
        geometry::NurbsBasis1D basis;
        // Boundaries of smooth NURBS parameter charts inside one physical
        // geometry branch. Parameter-coordinate density pieces are clamped
        // here and reconnected through intrinsic C^(p-1) constraints.
        std::vector<double> parameter_chart_breaks;
        std::vector<ArcLengthSegment> arclength_map;

        BranchData(NurbsDensityBranchInfo2D source_info,
                   geometry::NurbsBasis1D source_basis,
                   std::vector<double> source_chart_breaks = {})
            : info(std::move(source_info))
            , basis(std::move(source_basis))
            , parameter_chart_breaks(std::move(source_chart_breaks))
        {}
    };

    Impl(const BoundaryGeometry& source_geometry,
         int source_degree,
         double source_target_spacing,
         std::vector<NurbsDensityContinuity2D> source_connections,
         NurbsDensityCoordinate2D source_coordinate)
        : geometry(&source_geometry)
        , degree(source_degree)
        , target_spacing(source_target_spacing)
        , connections(std::move(source_connections))
        , coordinate(source_coordinate)
    {
        if (degree < 0 || degree > 5)
            throw std::invalid_argument(
                "NURBS density degree must be between zero and five");
        if (!(target_spacing > 0.0)
            || !std::isfinite(target_spacing)) {
            throw std::invalid_argument(
                "NURBS density target spacing must be finite and positive");
        }
        const int count = geometry->num_spans();
        if (count < 1)
            throw std::invalid_argument(
                "NURBS density space requires at least one branch");
        const int expected_connections = geometry->closed()
            ? count : count - 1;
        if (static_cast<int>(connections.size())
            != expected_connections) {
            throw std::invalid_argument(
                "NURBS density connection count does not match boundary topology");
        }
        for (NurbsDensityContinuity2D continuity : connections) {
            const int order = continuity_order(continuity);
            if (order < -1 || order > 2 || order > degree) {
                throw std::invalid_argument(
                    "NURBS density continuity is unsupported by its degree");
            }
        }

        const bool one_branch_native_periodic =
            coordinate == NurbsDensityCoordinate2D::PhysicalArclength
            && geometry->closed()
            && count == 1
            && degree >= 1
            && continuity_order(connections.front()) == degree - 1;

        int raw_offset = 0;
        branches.reserve(static_cast<std::size_t>(count));
        for (int branch = 0; branch < count; ++branch) {
            const auto interval = geometry->span_interval(branch);
            const double length = arclength_between(
                *geometry, branch, interval.first, interval.second);
            if (!(length > 1.0e-14) || !std::isfinite(length)) {
                throw std::invalid_argument(
                    "NURBS density branch has zero or invalid arclength");
            }
            const std::vector<double> chart_breaks =
                coordinate == NurbsDensityCoordinate2D::LegacyNurbsParameter
                ? internal_geometry_chart_breaks(
                      *geometry, interval.first, interval.second)
                : std::vector<double>();
            const int span_count = std::max(
                {one_branch_native_periodic ? degree + 1 : 1,
                 static_cast<int>(std::ceil(length / target_spacing)),
                 static_cast<int>(chart_breaks.size()) + 1});
            std::vector<double> breaks;
            if (coordinate
                == NurbsDensityCoordinate2D::LegacyNurbsParameter) {
                breaks = make_chart_aligned_parameter_breaks(
                    *geometry,
                    branch,
                    interval.first,
                    interval.second,
                    span_count,
                    chart_breaks);
            } else {
                breaks.resize(static_cast<std::size_t>(span_count + 1));
                breaks.front() = interval.first;
                breaks.back() = interval.second;
                for (int split = 1; split < span_count; ++split) {
                    breaks[static_cast<std::size_t>(split)] =
                        parameter_at_arclength(
                            *geometry,
                            branch,
                            interval.first,
                            interval.second,
                            length,
                            length * static_cast<double>(split)
                                / static_cast<double>(span_count));
                }
            }
            for (std::size_t i = 1; i < breaks.size(); ++i) {
                if (!(breaks[i] > breaks[i - 1])) {
                    throw std::runtime_error(
                        "NURBS density equal-arclength knot inversion was not monotone");
                }
            }

            NurbsDensityBranchInfo2D info;
            info.branch = branch;
            info.parameter_start = interval.first;
            info.parameter_end = interval.second;
            info.arclength = length;
            info.spline_span_count = span_count;
            info.raw_coefficient_offset = raw_offset;
            info.native_periodic = one_branch_native_periodic;
            info.parameter_breaks = breaks;
            info.parameter_chart_breaks = chart_breaks;

            const std::vector<double> spline_breaks =
                coordinate == NurbsDensityCoordinate2D::PhysicalArclength
                ? make_uniform_arclength_breaks(length, span_count)
                : breaks;
            geometry::NurbsBasis1D basis(
                degree,
                info.native_periodic
                    ? make_periodic_extension_knots(
                          length, span_count, degree)
                    : (coordinate
                               == NurbsDensityCoordinate2D::LegacyNurbsParameter
                           ? make_chart_aligned_parameter_knots(
                                 spline_breaks, chart_breaks, degree)
                           : make_clamped_knots(spline_breaks, degree)));
            info.raw_coefficient_count = info.native_periodic
                ? span_count : basis.num_basis_functions();
            const int expected_basis_count = info.native_periodic
                ? span_count + degree
                : info.raw_coefficient_count;
            if (basis.num_basis_functions() != expected_basis_count) {
                throw std::logic_error(
                    "NURBS density raw coefficient count is inconsistent");
            }
            raw_offset += info.raw_coefficient_count;
            branches.emplace_back(
                std::move(info), std::move(basis), chart_breaks);
        }
        if (coordinate == NurbsDensityCoordinate2D::PhysicalArclength) {
            for (BranchData& branch : branches)
                build_arclength_map(branch);
        }
        raw_count = raw_offset;
        build_reduction();
    }

    struct RawRows {
        Eigen::RowVectorXd value;
        Eigen::RowVectorXd first;
        Eigen::RowVectorXd second;
    };

    static double hermite_arclength(const ArcLengthSegment& segment,
                                    double parameter)
    {
        const double width =
            segment.parameter_end - segment.parameter_start;
        const double t = std::clamp(
            (parameter - segment.parameter_start) / width,
            0.0,
            1.0);
        const double t2 = t * t;
        const double t3 = t2 * t;
        const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        const double h10 = t3 - 2.0 * t2 + t;
        const double h01 = -2.0 * t3 + 3.0 * t2;
        const double h11 = t3 - t2;
        return h00 * segment.arclength_start
             + h10 * width * segment.speed_start
             + h01 * segment.arclength_end
             + h11 * width * segment.speed_end;
    }

    void append_certified_arclength_segment(
        BranchData& data,
        double parameter_start,
        double parameter_end,
        double arclength_start,
        double arclength_end,
        int depth)
    {
        ArcLengthSegment candidate;
        candidate.parameter_start = parameter_start;
        candidate.parameter_end = parameter_end;
        candidate.arclength_start = arclength_start;
        candidate.arclength_end = arclength_end;
        candidate.speed_start = geometry->evaluate_on_span(
            data.info.branch, parameter_start).speed;
        candidate.speed_end = geometry->evaluate_on_span(
            data.info.branch, parameter_end).speed;

        const double midpoint =
            0.5 * (parameter_start + parameter_end);
        const double exact_midpoint = arclength_start
            + arclength_between(
                  *geometry,
                  data.info.branch,
                  parameter_start,
                  midpoint);
        const double interpolation_error = std::abs(
            hermite_arclength(candidate, midpoint) - exact_midpoint);
        const double tolerance = 5.0e-13
            * std::max(1.0, data.info.arclength);
        const double parameter_tolerance =
            64.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(parameter_end));
        if (interpolation_error > tolerance
            && depth < 12
            && parameter_end - parameter_start > parameter_tolerance) {
            append_certified_arclength_segment(
                data,
                parameter_start,
                midpoint,
                arclength_start,
                exact_midpoint,
                depth + 1);
            append_certified_arclength_segment(
                data,
                midpoint,
                parameter_end,
                exact_midpoint,
                arclength_end,
                depth + 1);
            return;
        }
        data.arclength_map.push_back(candidate);
    }

    void build_arclength_map(BranchData& data)
    {
        data.arclength_map.clear();
        const int spans = data.info.spline_span_count;
        for (int span = 0; span < spans; ++span) {
            const double density_start = data.info.parameter_breaks[
                static_cast<std::size_t>(span)];
            const double density_end = data.info.parameter_breaks[
                static_cast<std::size_t>(span + 1)];
            std::vector<double> anchors{density_start, density_end};
            for (double knot : geometry->curve().basis().knots()) {
                if (knot > density_start && knot < density_end)
                    anchors.push_back(knot);
            }
            std::sort(anchors.begin(), anchors.end());
            anchors.erase(
                std::unique(anchors.begin(), anchors.end()), anchors.end());

            double local_start = data.info.arclength
                * static_cast<double>(span)
                / static_cast<double>(spans);
            const double density_arclength_end = data.info.arclength
                * static_cast<double>(span + 1)
                / static_cast<double>(spans);
            for (std::size_t part = 1; part < anchors.size(); ++part) {
                const double parameter_start = anchors[part - 1];
                const double parameter_end = anchors[part];
                const double local_end = part + 1 == anchors.size()
                    ? density_arclength_end
                    : local_start + arclength_between(
                          *geometry,
                          data.info.branch,
                          parameter_start,
                          parameter_end);
                append_certified_arclength_segment(
                    data,
                    parameter_start,
                    parameter_end,
                    local_start,
                    local_end,
                    0);
                local_start = local_end;
            }
        }
        if (data.arclength_map.empty()) {
            throw std::logic_error(
                "NURBS density arclength map has no segments");
        }
    }

    double local_arclength(const BranchData& data,
                           double parameter) const
    {
        const double parameter_scale = std::max(
            1.0,
            std::abs(data.info.parameter_end
                     - data.info.parameter_start));
        const double tolerance = 1.0e-12 * parameter_scale;
        if (parameter <= data.info.parameter_start + tolerance)
            return 0.0;
        if (parameter >= data.info.parameter_end - tolerance)
            return data.info.arclength;
        const auto found = std::lower_bound(
            data.arclength_map.begin(),
            data.arclength_map.end(),
            parameter,
            [](const ArcLengthSegment& segment, double value) {
                return segment.parameter_end < value;
            });
        if (found == data.arclength_map.end())
            return data.info.arclength;
        return std::clamp(
            hermite_arclength(*found, parameter),
            0.0,
            data.info.arclength);
    }

    double parameter_at_local_arclength(const BranchData& data,
                                        double local) const
    {
        if (data.info.native_periodic)
            local = wrap_periodic_coordinate(local, data.info.arclength);
        local = std::clamp(local, 0.0, data.info.arclength);
        const double length_tolerance = 5.0e-13
            * std::max(1.0, data.info.arclength);
        if (local <= length_tolerance)
            return data.info.parameter_start;
        if (local >= data.info.arclength - length_tolerance)
            return data.info.parameter_end;

        const auto found = std::lower_bound(
            data.arclength_map.begin(),
            data.arclength_map.end(),
            local,
            [](const ArcLengthSegment& segment, double value) {
                return segment.arclength_end < value;
            });
        if (found == data.arclength_map.end())
            return data.info.parameter_end;

        double left = found->parameter_start;
        double right = found->parameter_end;
        const double parameter_tolerance =
            32.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(right));
        for (int iteration = 0; iteration < 64; ++iteration) {
            const double midpoint = 0.5 * (left + right);
            if (midpoint == left || midpoint == right)
                break;
            if (hermite_arclength(*found, midpoint) < local)
                left = midpoint;
            else
                right = midpoint;
            if (right - left <= parameter_tolerance)
                break;
        }
        return 0.5 * (left + right);
    }

    Eigen::RowVectorXd raw_value_row_at_local_arclength(
        int branch,
        double local) const
    {
        if (coordinate != NurbsDensityCoordinate2D::PhysicalArclength) {
            throw std::invalid_argument(
                "sampled density differences require a physical-arclength spline");
        }
        require_branch(branch, static_cast<int>(branches.size()));
        const BranchData& data =
            branches[static_cast<std::size_t>(branch)];
        if (!std::isfinite(local)) {
            throw std::invalid_argument(
                "sampled density arclength is nonfinite");
        }
        if (data.info.native_periodic)
            local = wrap_periodic_coordinate(local, data.info.arclength);
        else
            local = std::clamp(local, 0.0, data.info.arclength);

        Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(raw_count);
        const std::vector<int> active =
            data.basis.active_basis_indices(local);
        const std::vector<double> values =
            data.basis.evaluate_nonzero(local);
        for (std::size_t index = 0; index < active.size(); ++index) {
            const int branch_coefficient = data.info.native_periodic
                ? active[index] % data.info.raw_coefficient_count
                : active[index];
            const int raw = data.info.raw_coefficient_offset
                          + branch_coefficient;
            row[raw] += values[index];
        }
        return row;
    }

    Eigen::RowVectorXd raw_value_row_at_parameter(
        int branch,
        double parameter) const
    {
        if (coordinate
            != NurbsDensityCoordinate2D::LegacyNurbsParameter) {
            throw std::invalid_argument(
                "covariant parameter differences require a NURBS-parameter spline");
        }
        require_branch(branch, static_cast<int>(branches.size()));
        const BranchData& data =
            branches[static_cast<std::size_t>(branch)];
        const double scale = std::max(
            1.0,
            std::abs(data.info.parameter_end
                     - data.info.parameter_start));
        const double tolerance = 1.0e-12 * scale;
        if (!std::isfinite(parameter)
            || parameter < data.info.parameter_start - tolerance
            || parameter > data.info.parameter_end + tolerance) {
            throw std::invalid_argument(
                "covariant density sample is outside its NURBS branch");
        }
        parameter = std::clamp(
            parameter,
            data.info.parameter_start,
            data.info.parameter_end);

        Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(raw_count);
        const std::vector<int> active =
            data.basis.active_basis_indices(parameter);
        const std::vector<double> values =
            data.basis.evaluate_nonzero(parameter);
        for (std::size_t index = 0; index < active.size(); ++index) {
            const int branch_coefficient = active[index];
            const int raw = data.info.raw_coefficient_offset
                          + branch_coefficient;
            row[raw] += values[index];
        }
        return row;
    }

    RawRows raw_rows(int branch, double parameter) const
    {
        require_branch(branch, static_cast<int>(branches.size()));
        const BranchData& data =
            branches[static_cast<std::size_t>(branch)];
        const double scale = std::max(
            1.0,
            std::abs(data.info.parameter_end
                     - data.info.parameter_start));
        const double tolerance = 1.0e-12 * scale;
        if (!std::isfinite(parameter)
            || parameter < data.info.parameter_start - tolerance
            || parameter > data.info.parameter_end + tolerance) {
            throw std::invalid_argument(
                "NURBS density evaluation parameter is outside its branch");
        }
        parameter = std::clamp(
            parameter,
            data.info.parameter_start,
            data.info.parameter_end);

        double spline_coordinate = parameter;
        if (coordinate == NurbsDensityCoordinate2D::PhysicalArclength) {
            spline_coordinate = local_arclength(data, parameter);
            if (data.info.native_periodic
                && spline_coordinate >= data.info.arclength
                       - 1.0e-12 * std::max(1.0, data.info.arclength)) {
                spline_coordinate = 0.0;
            }
        }

        RawRows rows;
        rows.value = Eigen::RowVectorXd::Zero(raw_count);
        rows.first = Eigen::RowVectorXd::Zero(raw_count);
        rows.second = Eigen::RowVectorXd::Zero(raw_count);

        const std::vector<int> active =
            data.basis.active_basis_indices(spline_coordinate);
        const std::vector<double> values =
            data.basis.evaluate_nonzero(spline_coordinate);
        const std::vector<double> first =
            data.basis.evaluate_nonzero_first_derivatives(spline_coordinate);
        const std::vector<double> second =
            data.basis.evaluate_nonzero_second_derivatives(
                spline_coordinate);

        double inverse_speed = 1.0;
        double speed_parameter_derivative = 0.0;
        if (coordinate == NurbsDensityCoordinate2D::LegacyNurbsParameter) {
            const geometry2d::NurbsBoundaryGeometry2D differential =
                geometry->evaluate_on_span(branch, parameter);
            inverse_speed = 1.0 / differential.speed;
            speed_parameter_derivative =
                differential.parameter_tangent.dot(
                    differential.parameter_second)
                * inverse_speed;
        }

        for (std::size_t local = 0; local < active.size(); ++local) {
            const int branch_coefficient = data.info.native_periodic
                ? active[local] % data.info.raw_coefficient_count
                : active[local];
            const int raw = data.info.raw_coefficient_offset
                          + branch_coefficient;
            rows.value[raw] += values[local];
            rows.first[raw] += first[local] * inverse_speed;
            rows.second[raw] +=
                second[local] * inverse_speed * inverse_speed
                - first[local] * speed_parameter_derivative
                    * inverse_speed * inverse_speed * inverse_speed;
        }
        return rows;
    }

    void build_reduction()
    {
        int row_count = 0;
        const int chart_continuity_order = std::min(2, degree - 1);
        if (coordinate
                == NurbsDensityCoordinate2D::LegacyNurbsParameter
            && chart_continuity_order >= 0) {
            for (const BranchData& branch : branches) {
                row_count += static_cast<int>(
                    branch.parameter_chart_breaks.size())
                    * chart_continuity_order;
            }
        }
        for (int connection = 0;
             connection < static_cast<int>(connections.size());
             ++connection) {
            if (branches.size() == 1
                && branches.front().info.native_periodic)
                continue;
            const NurbsDensityContinuity2D continuity =
                connections[static_cast<std::size_t>(connection)];
            const int order = continuity_order(continuity);
            if (order >= 0)
                row_count += order + 1;
        }
        constraints = Eigen::MatrixXd::Zero(row_count, raw_count);

        int row = 0;
        if (coordinate
                == NurbsDensityCoordinate2D::LegacyNurbsParameter
            && chart_continuity_order >= 0) {
            for (const BranchData& branch : branches) {
                for (double chart : branch.parameter_chart_breaks) {
                    const double left_parameter = std::nextafter(
                        chart, branch.info.parameter_start);
                    const double right_parameter = std::nextafter(
                        chart, branch.info.parameter_end);
                    const RawRows left_rows = raw_rows(
                        branch.info.branch, left_parameter);
                    const RawRows right_rows = raw_rows(
                        branch.info.branch, right_parameter);
                    if (chart_continuity_order >= 1) {
                        constraints.row(row++) =
                            left_rows.first - right_rows.first;
                    }
                    if (chart_continuity_order >= 2) {
                        constraints.row(row++) =
                            left_rows.second - right_rows.second;
                    }
                }
            }
        }
        for (int connection = 0;
             connection < static_cast<int>(connections.size());
             ++connection) {
            if (branches.size() == 1
                && branches.front().info.native_periodic)
                continue;
            const int order = continuity_order(
                connections[static_cast<std::size_t>(connection)]);
            if (order < 0)
                continue;
            const int left_branch = connection;
            const int right_branch = geometry->closed()
                ? (connection + 1) % static_cast<int>(branches.size())
                : connection + 1;
            const BranchData& left =
                branches[static_cast<std::size_t>(left_branch)];
            const BranchData& right =
                branches[static_cast<std::size_t>(right_branch)];
            const RawRows left_rows = raw_rows(
                left_branch, left.info.parameter_end);
            const RawRows right_rows = raw_rows(
                right_branch, right.info.parameter_start);

            constraints.row(row++) = left_rows.value - right_rows.value;
            if (order >= 1)
                constraints.row(row++) = left_rows.first - right_rows.first;
            if (order >= 2)
                constraints.row(row++) = left_rows.second - right_rows.second;
        }
        if (row != row_count)
            throw std::logic_error(
                "NURBS density constraint row count is inconsistent");

        if (row_count == 0) {
            reduction = Eigen::MatrixXd::Identity(raw_count, raw_count);
            rank = 0;
            return;
        }

        Eigen::MatrixXd scaled = constraints;
        for (int r = 0; r < scaled.rows(); ++r) {
            const double norm = scaled.row(r).norm();
            if (norm > 0.0)
                scaled.row(r) /= norm;
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            scaled, Eigen::ComputeFullV);
        const Eigen::VectorXd singular = svd.singularValues();
        const double maximum = singular.size() > 0
            ? singular.maxCoeff() : 0.0;
        const double threshold =
            256.0 * std::numeric_limits<double>::epsilon()
            * static_cast<double>(std::max(scaled.rows(), scaled.cols()))
            * std::max(1.0, maximum);
        rank = 0;
        for (int i = 0; i < singular.size(); ++i) {
            if (singular[i] > threshold)
                ++rank;
        }
        const int reduced_count = raw_count - rank;
        if (reduced_count < 1) {
            throw std::invalid_argument(
                "NURBS density constraints eliminate every coefficient");
        }
        reduction = svd.matrixV().rightCols(reduced_count);

        const double scaled_residual =
            (scaled * reduction).cwiseAbs().maxCoeff();
        if (!std::isfinite(scaled_residual)
            || scaled_residual > 5.0e-11) {
            throw std::runtime_error(
                "NURBS density continuity reduction is inaccurate");
        }
    }

    const BoundaryGeometry* geometry = nullptr;
    int degree = 0;
    double target_spacing = 0.0;
    std::vector<NurbsDensityContinuity2D> connections;
    NurbsDensityCoordinate2D coordinate =
        NurbsDensityCoordinate2D::PhysicalArclength;
    std::vector<BranchData> branches;
    int raw_count = 0;
    int rank = 0;
    Eigen::MatrixXd constraints;
    Eigen::MatrixXd reduction;
};

NurbsDensitySpace2D::NurbsDensitySpace2D(
    const BoundaryGeometry& geometry,
    int degree,
    double target_spacing,
    std::vector<NurbsDensityContinuity2D> connections,
    NurbsDensityCoordinate2D coordinate)
    : impl_(std::make_shared<Impl>(
          geometry,
          degree,
          target_spacing,
          std::move(connections),
          coordinate))
{}

int NurbsDensitySpace2D::degree() const
{
    return impl_->degree;
}

double NurbsDensitySpace2D::target_spacing() const
{
    return impl_->target_spacing;
}

int NurbsDensitySpace2D::branch_count() const
{
    return static_cast<int>(impl_->branches.size());
}

int NurbsDensitySpace2D::raw_coefficient_count() const
{
    return impl_->raw_count;
}

int NurbsDensitySpace2D::reduced_coefficient_count() const
{
    return static_cast<int>(impl_->reduction.cols());
}

int NurbsDensitySpace2D::constraint_rank() const
{
    return impl_->rank;
}

NurbsDensityCoordinate2D NurbsDensitySpace2D::coordinate() const
{
    return impl_->coordinate;
}

const BoundaryGeometry& NurbsDensitySpace2D::geometry() const
{
    return *impl_->geometry;
}

const std::vector<NurbsDensityContinuity2D>&
NurbsDensitySpace2D::connections() const
{
    return impl_->connections;
}

const NurbsDensityBranchInfo2D&
NurbsDensitySpace2D::branch_info(int branch) const
{
    require_branch(branch, branch_count());
    return impl_->branches[static_cast<std::size_t>(branch)].info;
}

const Eigen::MatrixXd& NurbsDensitySpace2D::constraint_matrix() const
{
    return impl_->constraints;
}

const Eigen::MatrixXd& NurbsDensitySpace2D::reduction_matrix() const
{
    return impl_->reduction;
}

Eigen::VectorXd NurbsDensitySpace2D::expand_reduced_coefficients(
    const Eigen::VectorXd& reduced_coefficients) const
{
    if (reduced_coefficients.size() != reduced_coefficient_count()) {
        throw std::invalid_argument(
            "NURBS density reduced coefficient vector has the wrong size");
    }
    if (!reduced_coefficients.allFinite())
        throw std::invalid_argument(
            "NURBS density reduced coefficients must be finite");
    return impl_->reduction * reduced_coefficients;
}

NurbsDensityEvaluationRows2D NurbsDensitySpace2D::evaluation_rows(
    int branch,
    double parameter) const
{
    const Impl::RawRows raw = impl_->raw_rows(branch, parameter);
    NurbsDensityEvaluationRows2D rows;
    rows.value = raw.value * impl_->reduction;
    rows.arclength_first = raw.first * impl_->reduction;
    rows.arclength_second = raw.second * impl_->reduction;
    return rows;
}

NurbsDensityEvaluationRows2D
NurbsDensitySpace2D::sampled_finite_difference_rows(
    int branch,
    double parameter,
    double difference_step_over_span) const
{
    if (coordinate() != NurbsDensityCoordinate2D::PhysicalArclength) {
        throw std::invalid_argument(
            "sampled density differences require PhysicalArclength coordinates");
    }
    if (!(difference_step_over_span > 0.0)
        || !std::isfinite(difference_step_over_span)) {
        throw std::invalid_argument(
            "sampled density difference spacing must be finite and positive");
    }
    require_branch(branch, branch_count());
    const Impl::BranchData& data =
        impl_->branches[static_cast<std::size_t>(branch)];
    const double parameter_scale = std::max(
        1.0,
        std::abs(data.info.parameter_end - data.info.parameter_start));
    const double parameter_tolerance = 1.0e-12 * parameter_scale;
    if (!std::isfinite(parameter)
        || parameter < data.info.parameter_start - parameter_tolerance
        || parameter > data.info.parameter_end + parameter_tolerance) {
        throw std::invalid_argument(
            "sampled density evaluation parameter is outside its branch");
    }
    parameter = std::clamp(
        parameter, data.info.parameter_start, data.info.parameter_end);
    const double local = impl_->local_arclength(data, parameter);
    const double mean_span = data.info.arclength
        / static_cast<double>(data.info.spline_span_count);
    double step = difference_step_over_span * mean_span;
    const double minimum_step = 64.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, data.info.arclength);
    if (!(step > minimum_step)) {
        throw std::invalid_argument(
            "sampled density difference spacing is too small");
    }

    const auto reduced_value_row = [&](double sample) -> Eigen::RowVectorXd {
        return (impl_->raw_value_row_at_local_arclength(branch, sample)
                * impl_->reduction)
            .eval();
    };

    NurbsDensityEvaluationRows2D rows;
    const Eigen::RowVectorXd center = reduced_value_row(local);
    rows.value = center;
    if (data.info.native_periodic) {
        step = std::min(step, data.info.arclength / 8.0);
        const Eigen::RowVectorXd minus = reduced_value_row(local - step);
        const Eigen::RowVectorXd plus = reduced_value_row(local + step);
        rows.arclength_first = (plus - minus) / (2.0 * step);
        rows.arclength_second =
            (plus - 2.0 * center + minus) / (step * step);
        return rows;
    }

    const double length = data.info.arclength;
    if (local >= step && local + step <= length) {
        const Eigen::RowVectorXd minus = reduced_value_row(local - step);
        const Eigen::RowVectorXd plus = reduced_value_row(local + step);
        rows.arclength_first = (plus - minus) / (2.0 * step);
        rows.arclength_second =
            (plus - 2.0 * center + minus) / (step * step);
        return rows;
    }

    const bool forward = local <= 0.5 * length;
    const double available = forward ? length - local : local;
    step = std::min(step, available / 3.0);
    if (!(step > minimum_step)) {
        throw std::runtime_error(
            "sampled density difference has no usable one-sided branch interval");
    }
    const double direction = forward ? 1.0 : -1.0;
    const Eigen::RowVectorXd sample1 =
        reduced_value_row(local + direction * step);
    const Eigen::RowVectorXd sample2 =
        reduced_value_row(local + direction * 2.0 * step);
    const Eigen::RowVectorXd sample3 =
        reduced_value_row(local + direction * 3.0 * step);
    // Cubic one-sided first derivative and quadratic-accurate one-sided
    // second derivative.  Reversing the sampling direction changes only the
    // sign of the first derivative.
    rows.arclength_first = direction
        * (-11.0 * center + 18.0 * sample1 - 9.0 * sample2
           + 2.0 * sample3)
        / (6.0 * step);
    rows.arclength_second =
        (2.0 * center - 5.0 * sample1 + 4.0 * sample2 - sample3)
        / (step * step);
    return rows;
}

NurbsDensityEvaluationRows2D
NurbsDensitySpace2D::covariant_parameter_finite_difference_rows(
    int branch,
    double parameter,
    double difference_step_over_span) const
{
    if (coordinate()
        != NurbsDensityCoordinate2D::LegacyNurbsParameter) {
        throw std::invalid_argument(
            "covariant parameter differences require NURBS-parameter coordinates");
    }
    if (!(difference_step_over_span > 0.0)
        || !std::isfinite(difference_step_over_span)) {
        throw std::invalid_argument(
            "covariant parameter difference spacing must be finite and positive");
    }
    require_branch(branch, branch_count());
    const Impl::BranchData& data =
        impl_->branches[static_cast<std::size_t>(branch)];
    const double parameter_width =
        data.info.parameter_end - data.info.parameter_start;
    const double parameter_scale = std::max(1.0, parameter_width);
    const double parameter_tolerance = 1.0e-12 * parameter_scale;
    if (!std::isfinite(parameter)
        || parameter < data.info.parameter_start - parameter_tolerance
        || parameter > data.info.parameter_end + parameter_tolerance) {
        throw std::invalid_argument(
            "covariant density evaluation parameter is outside its branch");
    }
    parameter = std::clamp(
        parameter, data.info.parameter_start, data.info.parameter_end);

    const geometry2d::NurbsBoundaryGeometry2D differential =
        impl_->geometry->evaluate_on_span(branch, parameter);
    const double speed = differential.speed;
    const double speed_parameter_derivative =
        differential.parameter_tangent.dot(differential.parameter_second)
        / speed;
    if (!(speed > 0.0) || !std::isfinite(speed)
        || !std::isfinite(speed_parameter_derivative)) {
        throw std::runtime_error(
            "covariant density differences found degenerate NURBS metric data");
    }

    // Keep the user-facing step tied to physical grid refinement even though
    // the samples are taken in xi.  The equal-arclength density break layout
    // makes mean_span the same resolution measure used by the arclength-FD
    // route; division by the exact local J maps it to a parameter increment.
    const double mean_span = data.info.arclength
        / static_cast<double>(data.info.spline_span_count);
    double step = difference_step_over_span * mean_span / speed;
    const double minimum_step = 64.0
        * std::numeric_limits<double>::epsilon() * parameter_scale;
    if (!(step > minimum_step) || !std::isfinite(step)) {
        throw std::invalid_argument(
            "covariant parameter difference spacing is too small or nonfinite");
    }

    const auto reduced_value_row =
        [&](double sample) -> Eigen::RowVectorXd {
        return (impl_->raw_value_row_at_parameter(branch, sample)
                * impl_->reduction)
            .eval();
    };

    NurbsDensityEvaluationRows2D rows;
    const Eigen::RowVectorXd center = reduced_value_row(parameter);
    rows.value = center;
    Eigen::RowVectorXd parameter_first;
    Eigen::RowVectorXd parameter_second;

    double chart_start = data.info.parameter_start;
    double chart_end = data.info.parameter_end;
    for (double chart : data.parameter_chart_breaks) {
        if (parameter >= chart)
            chart_start = chart;
        else {
            chart_end = chart;
            break;
        }
    }

    if (parameter - step >= chart_start
        && parameter + step <= chart_end) {
        const Eigen::RowVectorXd minus =
            reduced_value_row(parameter - step);
        const Eigen::RowVectorXd plus =
            reduced_value_row(parameter + step);
        parameter_first = (plus - minus) / (2.0 * step);
        parameter_second =
            (plus - 2.0 * center + minus) / (step * step);
    } else {
        // A feature connection has one-sided geometry and one-sided density
        // regularity.  Smooth closed seams also use the branch-local limit:
        // wrapping xi would assume an affine parameter transition that a
        // general NURBS representation does not promise.
        const double midpoint = 0.5 * (chart_start + chart_end);
        const bool forward = parameter <= midpoint;
        const double available = forward
            ? chart_end - parameter
            : parameter - chart_start;
        step = std::min(step, available / 3.0);
        // Keep all four samples inside one scalar polynomial piece.  At a
        // chart boundary or closed seam the one-sided finite difference then
        // reproduces the P3/P2 parameter jet exactly, so the physical seam
        // agreement is inherited directly from the reduction constraints.
        double polynomial_available = available;
        if (forward) {
            const auto next = std::upper_bound(
                data.info.parameter_breaks.begin(),
                data.info.parameter_breaks.end(),
                parameter);
            if (next != data.info.parameter_breaks.end())
                polynomial_available = *next - parameter;
        } else {
            const auto current = std::lower_bound(
                data.info.parameter_breaks.begin(),
                data.info.parameter_breaks.end(),
                parameter);
            if (current != data.info.parameter_breaks.begin())
                polynomial_available = parameter - *std::prev(current);
        }
        step = std::min(step, polynomial_available / 3.0);
        if (!(step > minimum_step)) {
            throw std::runtime_error(
                "covariant parameter difference has no usable one-sided branch interval");
        }
        const double direction = forward ? 1.0 : -1.0;
        const Eigen::RowVectorXd sample1 =
            reduced_value_row(parameter + direction * step);
        const Eigen::RowVectorXd sample2 =
            reduced_value_row(parameter + direction * 2.0 * step);
        const Eigen::RowVectorXd sample3 =
            reduced_value_row(parameter + direction * 3.0 * step);
        parameter_first = direction
            * (-11.0 * center + 18.0 * sample1 - 9.0 * sample2
               + 2.0 * sample3)
            / (6.0 * step);
        parameter_second =
            (2.0 * center - 5.0 * sample1 + 4.0 * sample2 - sample3)
            / (step * step);
    }

    const double inverse_speed = 1.0 / speed;
    rows.arclength_first = parameter_first * inverse_speed;
    rows.arclength_second =
        parameter_second * (inverse_speed * inverse_speed)
        - parameter_first
            * (speed_parameter_derivative
               * inverse_speed * inverse_speed * inverse_speed);
    if (!rows.value.allFinite() || !rows.arclength_first.allFinite()
        || !rows.arclength_second.allFinite()) {
        throw std::runtime_error(
            "covariant parameter differences produced a nonfinite physical jet");
    }
    return rows;
}

NurbsDensityJet2D NurbsDensitySpace2D::evaluate(
    int branch,
    double parameter,
    const Eigen::VectorXd& reduced_coefficients) const
{
    if (reduced_coefficients.size() != reduced_coefficient_count()
        || !reduced_coefficients.allFinite()) {
        throw std::invalid_argument(
            "NURBS density evaluation has invalid reduced coefficients");
    }
    const NurbsDensityEvaluationRows2D rows =
        evaluation_rows(branch, parameter);
    NurbsDensityJet2D result;
    result.value = rows.value.dot(reduced_coefficients);
    result.arclength_first =
        rows.arclength_first.dot(reduced_coefficients);
    result.arclength_second =
        rows.arclength_second.dot(reduced_coefficients);
    return result;
}

NurbsDensityJet2D
NurbsDensitySpace2D::evaluate_sampled_finite_difference(
    int branch,
    double parameter,
    const Eigen::VectorXd& reduced_coefficients,
    double difference_step_over_span) const
{
    if (reduced_coefficients.size() != reduced_coefficient_count()
        || !reduced_coefficients.allFinite()) {
        throw std::invalid_argument(
            "sampled density evaluation has invalid reduced coefficients");
    }
    const NurbsDensityEvaluationRows2D rows =
        sampled_finite_difference_rows(
            branch, parameter, difference_step_over_span);
    NurbsDensityJet2D result;
    result.value = rows.value.dot(reduced_coefficients);
    result.arclength_first =
        rows.arclength_first.dot(reduced_coefficients);
    result.arclength_second =
        rows.arclength_second.dot(reduced_coefficients);
    return result;
}

NurbsDensityJet2D
NurbsDensitySpace2D::evaluate_covariant_parameter_finite_difference(
    int branch,
    double parameter,
    const Eigen::VectorXd& reduced_coefficients,
    double difference_step_over_span) const
{
    if (reduced_coefficients.size() != reduced_coefficient_count()
        || !reduced_coefficients.allFinite()) {
        throw std::invalid_argument(
            "covariant density evaluation has invalid reduced coefficients");
    }
    const NurbsDensityEvaluationRows2D rows =
        covariant_parameter_finite_difference_rows(
            branch, parameter, difference_step_over_span);
    NurbsDensityJet2D result;
    result.value = rows.value.dot(reduced_coefficients);
    result.arclength_first =
        rows.arclength_first.dot(reduced_coefficients);
    result.arclength_second =
        rows.arclength_second.dot(reduced_coefficients);
    return result;
}

NurbsDensityJet2D NurbsDensitySpace2D::evaluate(
    NurbsDensityLocation2D location,
    const Eigen::VectorXd& reduced_coefficients) const
{
    return evaluate(
        location.branch, location.parameter, reduced_coefficients);
}

bool NurbsDensitySpace2D::can_resolve_crossing(
    const P2CrossingOwner2D& crossing) const
{
    try {
        (void)resolve_crossing(crossing);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

NurbsDensityLocation2D NurbsDensitySpace2D::resolve_crossing(
    const P2CrossingOwner2D& crossing) const
{
    const BoundaryGeometry& boundary = geometry();
    if (crossing.panel_index < 0
        || crossing.panel_index >= boundary.num_panels()
        || !crossing.crossing_point.allFinite()) {
        throw std::invalid_argument(
            "NURBS density crossing has no valid compatibility panel");
    }
    const int branch = boundary.panel_span(crossing.panel_index);
    NurbsDensityLocation2D location;
    location.branch = branch;

    if (crossing.status == P2CrossingOwnerStatus2D::ExactIntersection
        && std::isfinite(crossing.local_s)
        && crossing.local_s >= -1.0 - 1.0e-10
        && crossing.local_s <= 1.0 + 1.0e-10) {
        location.parameter = boundary.panel_parameter(
            crossing.panel_index, crossing.local_s);
        const Eigen::Vector2d point =
            boundary.curve().evaluate(location.parameter);
        const double scale = std::max(
            {1.0,
             branch_info(branch).arclength,
             crossing.crossing_point.norm()});
        if ((point - crossing.crossing_point).norm() > 1.0e-8 * scale) {
            throw std::invalid_argument(
                "NURBS density exact crossing disagrees with its parameter");
        }
        return location;
    }

    if (crossing.status != P2CrossingOwnerStatus2D::GapFallback
        || !crossing.explicit_gap_intersection) {
        throw std::invalid_argument(
            "NURBS density refuses an unresolved crossing fallback");
    }
    const geometry2d::NurbsSpanProjection2D projection =
        boundary.project_to_span(branch, crossing.crossing_point);
    const double scale = std::max(
        {1.0,
         branch_info(branch).arclength,
         crossing.crossing_point.norm()});
    if (!projection.converged
        || projection.distance > 1.0e-8 * scale) {
        throw std::invalid_argument(
            "NURBS density gap crossing is not on its owning branch");
    }
    location.parameter = projection.parameter;
    return location;
}

NurbsDensityJet2D NurbsDensitySpace2D::evaluate_crossing(
    const P2CrossingOwner2D& crossing,
    const Eigen::VectorXd& reduced_coefficients) const
{
    return evaluate(resolve_crossing(crossing), reduced_coefficients);
}

NurbsDensityJet2D
NurbsDensitySpace2D::evaluate_crossing_sampled_finite_difference(
    const P2CrossingOwner2D& crossing,
    const Eigen::VectorXd& reduced_coefficients,
    double difference_step_over_span) const
{
    const NurbsDensityLocation2D location = resolve_crossing(crossing);
    return evaluate_sampled_finite_difference(
        location.branch,
        location.parameter,
        reduced_coefficients,
        difference_step_over_span);
}

NurbsDensityJet2D
NurbsDensitySpace2D::evaluate_crossing_covariant_parameter_finite_difference(
    const P2CrossingOwner2D& crossing,
    const Eigen::VectorXd& reduced_coefficients,
    double difference_step_over_span) const
{
    const NurbsDensityLocation2D location = resolve_crossing(crossing);
    return evaluate_covariant_parameter_finite_difference(
        location.branch,
        location.parameter,
        reduced_coefficients,
        difference_step_over_span);
}

LaplaceCrossingTraceJet2D
NurbsDensitySpace2D::evaluate_crossing_trace(
    const P2CrossingOwner2D& crossing,
    const Eigen::VectorXd& reduced_coefficients,
    NurbsDensityTraceRole2D role) const
{
    const NurbsDensityJet2D density =
        evaluate_crossing(crossing, reduced_coefficients);
    LaplaceCrossingTraceJet2D trace;
    switch (role) {
    case NurbsDensityTraceRole2D::PhiValueJump:
        trace.value = density.value;
        trace.value_tangent_derivative = density.arclength_first;
        trace.value_tangent_second_derivative =
            density.arclength_second;
        break;
    case NurbsDensityTraceRole2D::PsiNormalJump:
        trace.normal_value = density.value;
        trace.normal_tangent_derivative = density.arclength_first;
        break;
    }
    return trace;
}

Eigen::VectorXd NurbsDensitySpace2D::fit_samples(
    const std::vector<NurbsDensitySample2D>& samples,
    double relative_rank_tolerance) const
{
    if (samples.empty())
        throw std::invalid_argument(
            "NURBS density fit requires at least one sample");
    if (!(relative_rank_tolerance > 0.0)
        || !std::isfinite(relative_rank_tolerance)) {
        throw std::invalid_argument(
            "NURBS density fit rank tolerance must be finite and positive");
    }
    const int reduced = reduced_coefficient_count();
    Eigen::MatrixXd design(samples.size(), reduced);
    Eigen::VectorXd rhs(samples.size());
    for (int row = 0; row < static_cast<int>(samples.size()); ++row) {
        const NurbsDensitySample2D& sample =
            samples[static_cast<std::size_t>(row)];
        if (!std::isfinite(sample.value) || !(sample.weight > 0.0)
            || !std::isfinite(sample.weight)) {
            throw std::invalid_argument(
                "NURBS density fit sample is nonfinite or has invalid weight");
        }
        const double scale = std::sqrt(sample.weight);
        design.row(row) = scale
            * evaluation_rows(sample.location.branch,
                              sample.location.parameter)
                  .value;
        rhs[row] = scale * sample.value;
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
    qr.setThreshold(relative_rank_tolerance);
    if (qr.rank() != reduced) {
        throw std::invalid_argument(
            "NURBS density samples do not determine every reduced coefficient");
    }
    Eigen::VectorXd coefficients = qr.solve(rhs);
    if (!coefficients.allFinite())
        throw std::runtime_error(
            "NURBS density fit produced nonfinite coefficients");
    return coefficients;
}

Eigen::VectorXd NurbsDensitySpace2D::fit_function(
    const Function& function,
    double relative_rank_tolerance) const
{
    if (!function)
        throw std::invalid_argument(
            "NURBS density function fit requires a callable function");
    std::vector<NurbsDensitySample2D> samples;
    if (coordinate()
        == NurbsDensityCoordinate2D::LegacyNurbsParameter) {
        // Chart multiplicities deliberately lower xi continuity before the
        // intrinsic derivative constraints restore physical smoothness. A
        // single Greville sample per raw coefficient is unnecessarily fragile
        // for this constrained space, especially when a fixed P3 datum later
        // supplies rho_ss. Use physical-L2-weighted Gauss oversampling on each
        // actual density span instead. This fit is preprocessing only; GMRES
        // matvecs still evaluate the reduced coefficients directly.
        std::size_t span_total = 0;
        for (const Impl::BranchData& branch : impl_->branches)
            span_total += branch.info.parameter_breaks.size() - 1;
        samples.reserve(span_total * kGaussNodes.size());
        for (const Impl::BranchData& branch : impl_->branches) {
            const std::vector<double>& breaks =
                branch.info.parameter_breaks;
            for (std::size_t span = 1; span < breaks.size(); ++span) {
                const double left = breaks[span - 1];
                const double right = breaks[span];
                const double midpoint = 0.5 * (left + right);
                const double half = 0.5 * (right - left);
                for (std::size_t q = 0; q < kGaussNodes.size(); ++q) {
                    const double parameter =
                        midpoint + half * kGaussNodes[q];
                    const geometry2d::NurbsBoundaryGeometry2D differential =
                        geometry().evaluate_on_span(
                            branch.info.branch, parameter);
                    const double value = function(
                        branch.info.branch,
                        parameter,
                        differential.point);
                    samples.push_back(
                        {{branch.info.branch, parameter},
                         value,
                         kGaussWeights[q] * half * differential.speed});
                }
            }
        }
        return fit_samples(samples, relative_rank_tolerance);
    }

    samples.reserve(static_cast<std::size_t>(raw_coefficient_count()));
    for (const Impl::BranchData& branch : impl_->branches) {
        const std::vector<double>& knots = branch.basis.knots();
        const int count = branch.info.raw_coefficient_count;
        for (int coefficient = 0; coefficient < count; ++coefficient) {
            double spline_sample = 0.0;
            if (degree() == 0) {
                spline_sample = 0.5
                    * (knots[static_cast<std::size_t>(coefficient)]
                       + knots[static_cast<std::size_t>(coefficient + 1)]);
            } else {
                for (int offset = 1; offset <= degree(); ++offset) {
                    spline_sample += knots[static_cast<std::size_t>(
                        coefficient + offset)];
                }
                spline_sample /= static_cast<double>(degree());
            }
            double parameter = spline_sample;
            if (coordinate()
                == NurbsDensityCoordinate2D::PhysicalArclength) {
                if (branch.info.native_periodic) {
                    spline_sample = wrap_periodic_coordinate(
                        spline_sample, branch.info.arclength);
                }
                parameter = impl_->parameter_at_local_arclength(
                    branch, spline_sample);
            }
            const Eigen::Vector2d point =
                geometry().curve().evaluate(parameter);
            const double value = function(
                branch.info.branch, parameter, point);
            samples.push_back(
                {{branch.info.branch, parameter}, value, 1.0});
        }
    }
    return fit_samples(samples, relative_rank_tolerance);
}

Eigen::VectorXd
NurbsDensitySpace2D::fit_parameterized_point_samples(
    const Eigen::VectorXd& values,
    double relative_rank_tolerance) const
{
    if (values.size() != geometry().num_parameterized_points()
        || !values.allFinite()) {
        throw std::invalid_argument(
            "NURBS density interface samples must match parameterized geometry points");
    }
    std::vector<NurbsDensitySample2D> samples;
    samples.reserve(static_cast<std::size_t>(values.size()));
    for (int point = 0; point < values.size(); ++point) {
        samples.push_back(
            {{geometry().point_span(point),
              geometry().point_parameter(point)},
             values[point],
             1.0});
    }
    return fit_samples(samples, relative_rank_tolerance);
}

LaplaceP2CrossingLocalPolynomial2D
build_nurbs_density_local_polynomial_2d(
    const NurbsDensitySpace2D& phi_space,
    const Eigen::VectorXd& phi_coefficients,
    const NurbsDensitySpace2D& psi_space,
    const Eigen::VectorXd& psi_coefficients,
    const P2CrossingOwner2D& crossing,
    double forcing_jump,
    double alpha)
{
    if (&phi_space.geometry() != &psi_space.geometry()) {
        throw std::invalid_argument(
            "NURBS density Cauchy spaces must share one geometry object");
    }
    if (!std::isfinite(forcing_jump) || !std::isfinite(alpha)) {
        throw std::invalid_argument(
            "NURBS density Cauchy closure has nonfinite PDE data");
    }
    const NurbsDensityLocation2D phi_location =
        phi_space.resolve_crossing(crossing);
    const NurbsDensityLocation2D psi_location =
        psi_space.resolve_crossing(crossing);
    const double parameter_scale = std::max(
        1.0, std::abs(phi_location.parameter));
    if (phi_location.branch != psi_location.branch
        || std::abs(phi_location.parameter - psi_location.parameter)
               > 1.0e-12 * parameter_scale) {
        throw std::logic_error(
            "NURBS density Cauchy spaces resolved different crossings");
    }

    const NurbsDensityJet2D phi =
        phi_space.evaluate(phi_location, phi_coefficients);
    const NurbsDensityJet2D psi =
        psi_space.evaluate(psi_location, psi_coefficients);
    LaplaceCrossingTraceJet2D trace;
    trace.value = phi.value;
    trace.value_tangent_derivative = phi.arclength_first;
    trace.value_tangent_second_derivative = phi.arclength_second;
    trace.normal_value = psi.value;
    trace.normal_tangent_derivative = psi.arclength_first;

    const geometry2d::NurbsBoundaryGeometry2D geometry_value =
        phi_space.geometry().evaluate_on_span(
            phi_location.branch, phi_location.parameter);
    LaplaceCrossingGeometryJet2D geometry_jet;
    geometry_jet.center = geometry_value.point;
    geometry_jet.tangent = geometry_value.tangent;
    geometry_jet.normal = geometry_value.normal;
    geometry_jet.curvature = geometry_value.curvature;
    geometry_jet.forcing = forcing_jump;
    return build_laplace_crossing_local_polynomial_from_geometry_trace_jet_2d(
        geometry_jet, trace, alpha);
}

} // namespace kfbim
