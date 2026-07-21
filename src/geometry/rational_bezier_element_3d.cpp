#include "rational_bezier_element_3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kfbim::geometry3d {

namespace {

double midpoint_if_representable(double begin, double end)
{
    const double midpoint = 0.5 * begin + 0.5 * end;
    if (!std::isfinite(midpoint) || midpoint <= begin || midpoint >= end) {
        throw std::invalid_argument(
            "Bezier parameter interval has no representable midpoint");
    }
    return midpoint;
}

std::size_t expected_control_count(const RationalBezierElement3D& element)
{
    if (element.degree_u < 0 || element.degree_v < 0)
        throw std::invalid_argument("Bezier element degree must be nonnegative");
    if (element.degree_u == std::numeric_limits<int>::max()
        || element.degree_v == std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "Bezier element degree is too large for inclusive indexing");
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const auto checked_extent = [maximum](int degree) {
        if (static_cast<std::uintmax_t>(degree)
            >= static_cast<std::uintmax_t>(maximum)) {
            throw std::invalid_argument(
                "Bezier element control count is too large");
        }
        return static_cast<std::size_t>(degree) + std::size_t{1};
    };
    const std::size_t count_u = checked_extent(element.degree_u);
    const std::size_t count_v = checked_extent(element.degree_v);
    if (count_u > maximum / count_v) {
        throw std::invalid_argument(
            "Bezier element control count is too large");
    }
    const std::size_t count = count_u * count_v;
    if (count > maximum / sizeof(Eigen::Vector4d)) {
        throw std::invalid_argument(
            "Bezier element control count is too large");
    }
    return count;
}

void validate_element(const RationalBezierElement3D& element)
{
    if (!std::isfinite(element.parameter_u0)
        || !std::isfinite(element.parameter_u1)
        || !std::isfinite(element.parameter_v0)
        || !std::isfinite(element.parameter_v1)
        || element.parameter_u0 >= element.parameter_u1
        || element.parameter_v0 >= element.parameter_v1) {
        throw std::invalid_argument("Bezier element has a zero parameter span");
    }
    if (element.homogeneous_controls.size()
        != expected_control_count(element)) {
        throw std::invalid_argument("Bezier element has an inconsistent control count");
    }
    for (const Eigen::Vector4d& control : element.homogeneous_controls) {
        if (!control.allFinite() || control.w() <= 0.0) {
            throw std::invalid_argument(
                "Bezier element has a nonpositive homogeneous weight");
        }
    }
}

Eigen::Vector4d evaluate_curve(std::vector<Eigen::Vector4d> controls,
                               double parameter)
{
    for (std::size_t remaining = controls.size(); remaining > 1; --remaining) {
        for (std::size_t i = 0; i + 1 < remaining; ++i) {
            controls[i] = (1.0 - parameter) * controls[i]
                        + parameter * controls[i + 1];
        }
    }
    return controls.front();
}

std::array<std::vector<Eigen::Vector4d>, 2> split_curve(
    std::vector<Eigen::Vector4d> controls)
{
    const std::size_t count = controls.size();
    std::array<std::vector<Eigen::Vector4d>, 2> children{
        std::vector<Eigen::Vector4d>(count),
        std::vector<Eigen::Vector4d>(count)};
    children[0][0] = controls.front();
    children[1][count - 1] = controls.back();
    for (std::size_t level = 1; level < count; ++level) {
        for (std::size_t i = 0; i + level < count; ++i)
            controls[i] = 0.5 * controls[i] + 0.5 * controls[i + 1];
        children[0][level] = controls[0];
        children[1][count - level - 1] = controls[count - level - 1];
    }
    return children;
}

} // namespace

double RationalBezierElement3D::u0() const { return parameter_u0; }
double RationalBezierElement3D::u1() const { return parameter_u1; }
double RationalBezierElement3D::v0() const { return parameter_v0; }
double RationalBezierElement3D::v1() const { return parameter_v1; }

const Eigen::Vector4d& RationalBezierElement3D::control(int i, int j) const
{
    if (i < 0 || i > degree_u || j < 0 || j > degree_v)
        throw std::out_of_range("Bezier control index is out of range");
    if (homogeneous_controls.size() != expected_control_count(*this))
        throw std::invalid_argument("Bezier element has an inconsistent control count");
    return homogeneous_controls[
        static_cast<std::size_t>(i * (degree_v + 1) + j)];
}

Eigen::Vector3d RationalBezierElement3D::evaluate(double u, double v) const
{
    validate_element(*this);
    if (!std::isfinite(u) || !std::isfinite(v)
        || u < parameter_u0 || u > parameter_u1
        || v < parameter_v0 || v > parameter_v1) {
        throw std::invalid_argument("Bezier evaluation parameter is outside its element");
    }
    const double local_u = (u - parameter_u0) / (parameter_u1 - parameter_u0);
    const double local_v = (v - parameter_v0) / (parameter_v1 - parameter_v0);
    std::vector<Eigen::Vector4d> reduced_u(
        static_cast<std::size_t>(degree_u + 1));
    for (int i = 0; i <= degree_u; ++i) {
        std::vector<Eigen::Vector4d> row;
        row.reserve(static_cast<std::size_t>(degree_v + 1));
        for (int j = 0; j <= degree_v; ++j)
            row.push_back(control(i, j));
        reduced_u[static_cast<std::size_t>(i)] = evaluate_curve(row, local_v);
    }
    const Eigen::Vector4d homogeneous = evaluate_curve(reduced_u, local_u);
    if (!homogeneous.allFinite() || homogeneous.w() <= 0.0)
        throw std::runtime_error("Bezier evaluation has a nonpositive weight");
    return homogeneous.head<3>() / homogeneous.w();
}

NurbsAabb3D RationalBezierElement3D::bounds() const
{
    validate_element(*this);
    NurbsAabb3D result;
    result.lower = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::infinity());
    result.upper = Eigen::Vector3d::Constant(
        -std::numeric_limits<double>::infinity());
    for (const Eigen::Vector4d& control : homogeneous_controls) {
        const Eigen::Vector3d projected = control.head<3>() / control.w();
        result.lower = result.lower.cwiseMin(projected);
        result.upper = result.upper.cwiseMax(projected);
    }
    return result;
}

std::array<RationalBezierElement3D, 2>
RationalBezierElement3D::split_u() const
{
    validate_element(*this);
    std::array<RationalBezierElement3D, 2> children{*this, *this};
    const double midpoint = midpoint_if_representable(
        parameter_u0, parameter_u1);
    children[0].parameter_u1 = midpoint;
    children[1].parameter_u0 = midpoint;
    for (int j = 0; j <= degree_v; ++j) {
        std::vector<Eigen::Vector4d> column;
        column.reserve(static_cast<std::size_t>(degree_u + 1));
        for (int i = 0; i <= degree_u; ++i)
            column.push_back(control(i, j));
        const auto split = split_curve(std::move(column));
        for (int i = 0; i <= degree_u; ++i) {
            children[0].homogeneous_controls[
                static_cast<std::size_t>(i * (degree_v + 1) + j)] =
                split[0][static_cast<std::size_t>(i)];
            children[1].homogeneous_controls[
                static_cast<std::size_t>(i * (degree_v + 1) + j)] =
                split[1][static_cast<std::size_t>(i)];
        }
    }
    validate_element(children[0]);
    validate_element(children[1]);
    return children;
}

std::array<RationalBezierElement3D, 2>
RationalBezierElement3D::split_v() const
{
    validate_element(*this);
    std::array<RationalBezierElement3D, 2> children{*this, *this};
    const double midpoint = midpoint_if_representable(
        parameter_v0, parameter_v1);
    children[0].parameter_v1 = midpoint;
    children[1].parameter_v0 = midpoint;
    for (int i = 0; i <= degree_u; ++i) {
        std::vector<Eigen::Vector4d> row;
        row.reserve(static_cast<std::size_t>(degree_v + 1));
        for (int j = 0; j <= degree_v; ++j)
            row.push_back(control(i, j));
        const auto split = split_curve(std::move(row));
        for (int j = 0; j <= degree_v; ++j) {
            children[0].homogeneous_controls[
                static_cast<std::size_t>(i * (degree_v + 1) + j)] =
                split[0][static_cast<std::size_t>(j)];
            children[1].homogeneous_controls[
                static_cast<std::size_t>(i * (degree_v + 1) + j)] =
                split[1][static_cast<std::size_t>(j)];
        }
    }
    validate_element(children[0]);
    validate_element(children[1]);
    return children;
}

} // namespace kfbim::geometry3d
