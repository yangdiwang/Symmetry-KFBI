#include "native_nurbs_exact_geometry_3d.hpp"

#include <CGAL/number_utils.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kfbim::geometry3d {
namespace {

using Rational = CGAL::Gmpq;
using HomogeneousPoint = ExactNativeHomogeneousPoint3D;
using HomogeneousNet = std::vector<std::vector<HomogeneousPoint>>;

Rational exact_finite_double(double value)
{
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "Exact native NURBS geometry requires finite original doubles");
    }
    return Rational(value);
}

void validate_axis(const std::vector<double>& knots, int degree,
                   std::size_t control_count)
{
    if (degree < 0 || control_count <= static_cast<std::size_t>(degree)
        || control_count > static_cast<std::size_t>(
               std::numeric_limits<int>::max())
        || knots.size() != control_count + static_cast<std::size_t>(degree) + 1) {
        throw std::invalid_argument(
            "Exact native Bezier extraction has inconsistent degree/control/knots");
    }
    for (std::size_t i = 0; i < knots.size(); ++i) {
        if (!std::isfinite(knots[i]) || (i > 0 && knots[i] < knots[i - 1])) {
            throw std::invalid_argument(
                "Exact native Bezier extraction requires finite nondecreasing knots");
        }
    }
    const double start = knots[static_cast<std::size_t>(degree)];
    const double end = knots[control_count];
    if (!(start < end)) {
        throw std::invalid_argument(
            "Exact native Bezier extraction has an empty active domain");
    }
    for (int i = 0; i <= degree; ++i) {
        if (knots[static_cast<std::size_t>(i)] != start
            || knots[knots.size() - 1 - static_cast<std::size_t>(i)] != end) {
            throw std::invalid_argument(
                "Exact native Bezier extraction does not support unclamped active endpoints");
        }
    }
    for (std::size_t begin = 0; begin < knots.size();) {
        std::size_t end_index = begin + 1;
        while (end_index < knots.size() && knots[end_index] == knots[begin])
            ++end_index;
        if (end_index - begin > static_cast<std::size_t>(degree) + 1) {
            throw std::invalid_argument(
                "Exact native Bezier extraction has an excessive knot multiplicity");
        }
        begin = end_index;
    }
}

HomogeneousPoint interpolate(const HomogeneousPoint& left,
                             const HomogeneousPoint& right,
                             const Rational& parameter)
{
    HomogeneousPoint result;
    const Rational complement = Rational(1) - parameter;
    for (int coordinate = 0; coordinate < 4; ++coordinate) {
        result[coordinate] = complement * left[coordinate]
                           + parameter * right[coordinate];
    }
    return result;
}

void insert_knot_u_once(HomogeneousNet& net, std::vector<double>& knots,
                        int degree, double knot)
{
    const int last_control = static_cast<int>(net.size()) - 1;
    const int span = static_cast<int>(
        std::upper_bound(knots.begin(), knots.end(), knot) - knots.begin()) - 1;
    const int multiplicity = static_cast<int>(
        std::count(knots.begin(), knots.end(), knot));
    if (span - degree < 0 || span - degree > last_control
        || span - multiplicity < 0 || span - multiplicity > last_control
        || multiplicity >= degree) {
        throw std::invalid_argument(
            "Exact native Bezier knot insertion is outside its active interior");
    }
    HomogeneousNet refined(net.size() + 1,
                           std::vector<HomogeneousPoint>(net.front().size()));
    for (int i = 0; i <= span - degree; ++i)
        refined[static_cast<std::size_t>(i)] = net[static_cast<std::size_t>(i)];
    for (int i = span - multiplicity; i <= last_control; ++i)
        refined[static_cast<std::size_t>(i + 1)] = net[static_cast<std::size_t>(i)];

    const Rational exact_knot = exact_finite_double(knot);
    for (int i = span - degree + 1; i <= span - multiplicity; ++i) {
        // Convert each original knot before subtracting or dividing. Doing
        // the subtraction in double would change the native input geometry.
        const Rational left = exact_finite_double(knots[static_cast<std::size_t>(i)]);
        const Rational denominator =
            exact_finite_double(knots[static_cast<std::size_t>(i + degree)]) - left;
        if (denominator <= 0) {
            throw std::invalid_argument(
                "Exact native Bezier knot insertion has a nonpositive denominator");
        }
        const Rational alpha = (exact_knot - left) / denominator;
        if (alpha < 0 || alpha > 1) {
            throw std::invalid_argument(
                "Exact native Bezier knot insertion is not a convex combination");
        }
        for (std::size_t j = 0; j < net.front().size(); ++j) {
            refined[static_cast<std::size_t>(i)][j] = interpolate(
                net[static_cast<std::size_t>(i - 1)][j],
                net[static_cast<std::size_t>(i)][j], alpha);
        }
    }
    knots.insert(knots.begin() + span + 1, knot);
    net = std::move(refined);
}

void refine_interior_knots_u(HomogeneousNet& net, std::vector<double>& knots,
                             int degree)
{
    if (degree == 0)
        return;
    const double start = knots[static_cast<std::size_t>(degree)];
    const double end = knots[net.size()];
    std::vector<double> interior_knots;
    for (const double knot : knots) {
        if (start < knot && knot < end
            && (interior_knots.empty() || knot != interior_knots.back())) {
            interior_knots.push_back(knot);
        }
    }
    for (const double knot : interior_knots) {
        const int multiplicity = static_cast<int>(
            std::count(knots.begin(), knots.end(), knot));
        for (int insertion = multiplicity; insertion < degree; ++insertion)
            insert_knot_u_once(net, knots, degree, knot);
    }
}

HomogeneousNet transpose(const HomogeneousNet& net)
{
    HomogeneousNet result(net.front().size(),
                          std::vector<HomogeneousPoint>(net.size()));
    for (std::size_t i = 0; i < net.size(); ++i) {
        for (std::size_t j = 0; j < net.front().size(); ++j)
            result[j][i] = net[i][j];
    }
    return result;
}

void validate_element(const ExactNativeBezierElement3D& element)
{
    if (element.degree_u < 0 || element.degree_v < 0
        || !std::isfinite(element.parameter_u0)
        || !std::isfinite(element.parameter_u1)
        || !std::isfinite(element.parameter_v0)
        || !std::isfinite(element.parameter_v1)
        || !(element.parameter_u0 < element.parameter_u1)
        || !(element.parameter_v0 < element.parameter_v1)) {
        throw std::invalid_argument("Invalid exact native Bezier element domain/degree");
    }
    const std::size_t count_u = static_cast<std::size_t>(element.degree_u) + 1;
    const std::size_t count_v = static_cast<std::size_t>(element.degree_v) + 1;
    if (count_u > std::numeric_limits<std::size_t>::max() / count_v
        || element.controls.size() != count_u * count_v) {
        throw std::invalid_argument("Invalid exact native Bezier element control count");
    }
    for (const auto& control : element.controls) {
        if (control[3] <= 0) {
            throw std::invalid_argument(
                "Exact native Bezier geometry requires positive homogeneous weights");
        }
    }
}

HomogeneousPoint de_casteljau(std::vector<HomogeneousPoint> controls,
                              const Rational& parameter)
{
    for (std::size_t count = controls.size(); count > 1; --count) {
        for (std::size_t i = 0; i + 1 < count; ++i)
            controls[i] = interpolate(controls[i], controls[i + 1], parameter);
    }
    return controls.front();
}

} // namespace

NurbsAabb3D ExactNativeBezierElement3D::bounds() const
{
    validate_element(*this);
    NurbsAabb3D result;
    result.lower.setConstant(std::numeric_limits<double>::infinity());
    result.upper.setConstant(-std::numeric_limits<double>::infinity());
    for (const auto& control : controls) {
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            const Rational cartesian = control[coordinate] / control[3];
            const auto interval = CGAL::to_interval(cartesian);
            if (std::isnan(interval.first) || std::isnan(interval.second)
                || interval.first > interval.second) {
                throw std::runtime_error("Exact native Bezier bounds conversion failed");
            }
            result.lower[coordinate] = std::min(result.lower[coordinate], interval.first);
            result.upper[coordinate] = std::max(result.upper[coordinate], interval.second);
        }
    }
    return result;
}

std::vector<ExactNativeBezierElement3D>
extract_exact_native_bezier_elements_3d(const NurbsSurfaceModel3D& model)
{
    std::vector<ExactNativeBezierElement3D> elements;
    for (int patch_index = 0; patch_index < model.num_patches(); ++patch_index) {
        const auto& patch = model.patch(patch_index);
        const auto& points = patch.control_net();
        const auto& weights = patch.weights();
        if (points.empty() || points.front().empty() || weights.size() != points.size()) {
            throw std::invalid_argument("Exact native Bezier extraction has an empty/inconsistent net");
        }
        const int degree_u = patch.basis_u().degree();
        const int degree_v = patch.basis_v().degree();
        std::vector<double> knots_u = patch.basis_u().knots();
        std::vector<double> knots_v = patch.basis_v().knots();
        validate_axis(knots_u, degree_u, points.size());
        validate_axis(knots_v, degree_v, points.front().size());
        HomogeneousNet net(points.size(),
                           std::vector<HomogeneousPoint>(points.front().size()));
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (points[i].size() != points.front().size()
                || weights[i].size() != points.front().size()) {
                throw std::invalid_argument("Exact native Bezier extraction has a ragged control net");
            }
            for (std::size_t j = 0; j < points[i].size(); ++j) {
                const Rational weight = exact_finite_double(weights[i][j]);
                if (weight <= 0) {
                    throw std::invalid_argument("Exact native Bezier extraction requires positive weights");
                }
                for (int coordinate = 0; coordinate < 3; ++coordinate) {
                    net[i][j][coordinate] =
                        weight * exact_finite_double(points[i][j][coordinate]);
                }
                net[i][j][3] = weight;
            }
        }
        refine_interior_knots_u(net, knots_u, degree_u);
        auto transposed = transpose(net);
        refine_interior_knots_u(transposed, knots_v, degree_v);
        net = transpose(transposed);
        validate_axis(knots_u, degree_u, net.size());
        validate_axis(knots_v, degree_v, net.front().size());

        for (int span_u = degree_u; span_u < static_cast<int>(net.size()); ++span_u) {
            if (!(knots_u[static_cast<std::size_t>(span_u)]
                  < knots_u[static_cast<std::size_t>(span_u + 1)]))
                continue;
            for (int span_v = degree_v;
                 span_v < static_cast<int>(net.front().size()); ++span_v) {
                if (!(knots_v[static_cast<std::size_t>(span_v)]
                      < knots_v[static_cast<std::size_t>(span_v + 1)]))
                    continue;
                ExactNativeBezierElement3D element;
                element.patch_index = patch_index;
                element.component = model.patch_component(patch_index);
                element.degree_u = degree_u;
                element.degree_v = degree_v;
                element.parameter_u0 = knots_u[static_cast<std::size_t>(span_u)];
                element.parameter_u1 = knots_u[static_cast<std::size_t>(span_u + 1)];
                element.parameter_v0 = knots_v[static_cast<std::size_t>(span_v)];
                element.parameter_v1 = knots_v[static_cast<std::size_t>(span_v + 1)];
                for (int i = span_u - degree_u; i <= span_u; ++i) {
                    for (int j = span_v - degree_v; j <= span_v; ++j)
                        element.controls.push_back(net[static_cast<std::size_t>(i)]
                                                       [static_cast<std::size_t>(j)]);
                }
                (void)element.bounds();
                elements.push_back(std::move(element));
            }
        }
    }
    return elements;
}

ExactNativeHomogeneousPoint3D evaluate_native_bezier_homogeneous(
    const ExactNativeBezierElement3D& element, double u, double v)
{
    return evaluate_native_bezier_homogeneous(
        element, exact_finite_double(u), exact_finite_double(v));
}

ExactNativeHomogeneousPoint3D evaluate_native_bezier_homogeneous(
    const ExactNativeBezierElement3D& element,
    const CGAL::Gmpq& u, const CGAL::Gmpq& v)
{
    validate_element(element);
    const Rational u0 = exact_finite_double(element.parameter_u0);
    const Rational u1 = exact_finite_double(element.parameter_u1);
    const Rational v0 = exact_finite_double(element.parameter_v0);
    const Rational v1 = exact_finite_double(element.parameter_v1);
    if (u < u0 || u > u1 || v < v0 || v > v1)
        throw std::out_of_range("Exact native Bezier parameter lies outside its element");
    const Rational local_u = (u - u0) / (u1 - u0);
    const Rational local_v = (v - v0) / (v1 - v0);
    const std::size_t count_u = static_cast<std::size_t>(element.degree_u) + 1;
    const std::size_t count_v = static_cast<std::size_t>(element.degree_v) + 1;
    std::vector<HomogeneousPoint> u_controls(count_u);
    for (std::size_t i = 0; i < count_u; ++i) {
        std::vector<HomogeneousPoint> v_controls;
        v_controls.reserve(count_v);
        for (std::size_t j = 0; j < count_v; ++j)
            v_controls.push_back(element.controls[i * count_v + j]);
        u_controls[i] = de_casteljau(std::move(v_controls), local_v);
    }
    return de_casteljau(std::move(u_controls), local_u);
}

} // namespace kfbim::geometry3d
