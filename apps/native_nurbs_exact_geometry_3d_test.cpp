#include "src/geometry/native_nurbs_exact_geometry_3d.hpp"

#include <CGAL/number_utils.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Rational = CGAL::Gmpq;
using kfbim::geometry::NurbsBasis1D;
using kfbim::geometry3d::ExactNativeBezierElement3D;
using kfbim::geometry3d::ExactNativeHomogeneousPoint3D;
using kfbim::geometry3d::NurbsAabb3D;
using kfbim::geometry3d::NurbsSurfaceModel3D;
using kfbim::geometry3d::NurbsSurfacePatch3D;
using kfbim::geometry3d::evaluate_native_bezier_homogeneous;
using kfbim::geometry3d::extract_exact_native_bezier_elements_3d;
using HomogeneousPoint = ExactNativeHomogeneousPoint3D;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class Exception, class Function>
void require_throws(Function&& function, const std::string& message)
{
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }
    require(caught, message);
}

Rational fraction(int numerator, int denominator)
{
    return Rational(numerator) / Rational(denominator);
}

// Independent oracle: direct de Boor evaluation of the ORIGINAL B-spline
// controls and knots. It performs neither knot insertion nor Bezier extraction.
HomogeneousPoint exact_de_boor_curve(
    const std::vector<HomogeneousPoint>& controls,
    const std::vector<double>& knots, int degree, const Rational& parameter)
{
    const int last = static_cast<int>(controls.size()) - 1;
    require(parameter >= Rational(knots[static_cast<std::size_t>(degree)])
                && parameter <= Rational(knots[static_cast<std::size_t>(last + 1)]),
            "de Boor oracle parameter must be in the native active domain");
    int span = last;
    if (parameter != Rational(knots[static_cast<std::size_t>(last + 1)])) {
        span = -1;
        for (int candidate = degree; candidate <= last; ++candidate) {
            if (Rational(knots[static_cast<std::size_t>(candidate)]) <= parameter
                && parameter < Rational(knots[static_cast<std::size_t>(candidate + 1)])) {
                span = candidate;
                break;
            }
        }
    }
    require(span >= degree, "de Boor oracle must find a nonempty knot span");
    std::vector<HomogeneousPoint> work(static_cast<std::size_t>(degree + 1));
    for (int j = 0; j <= degree; ++j)
        work[static_cast<std::size_t>(j)] =
            controls[static_cast<std::size_t>(span - degree + j)];
    for (int recursion = 1; recursion <= degree; ++recursion) {
        for (int j = degree; j >= recursion; --j) {
            const Rational left(knots[static_cast<std::size_t>(span - degree + j)]);
            const Rational right(knots[static_cast<std::size_t>(span + 1 + j - recursion)]);
            require(right > left, "de Boor oracle denominator must be positive");
            const Rational alpha = (parameter - left) / (right - left);
            for (int coordinate = 0; coordinate < 4; ++coordinate) {
                work[static_cast<std::size_t>(j)][coordinate] =
                    (Rational(1) - alpha) * work[static_cast<std::size_t>(j - 1)][coordinate]
                    + alpha * work[static_cast<std::size_t>(j)][coordinate];
            }
        }
    }
    return work[static_cast<std::size_t>(degree)];
}

HomogeneousPoint exact_de_boor_surface(const NurbsSurfacePatch3D& patch,
                                       const Rational& u, const Rational& v)
{
    const auto& points = patch.control_net();
    const auto& weights = patch.weights();
    std::vector<HomogeneousPoint> u_controls(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        std::vector<HomogeneousPoint> v_controls(points[i].size());
        for (std::size_t j = 0; j < points[i].size(); ++j) {
            const Rational weight(weights[i][j]);
            for (int coordinate = 0; coordinate < 3; ++coordinate)
                v_controls[j][coordinate] = Rational(points[i][j][coordinate]) * weight;
            v_controls[j][3] = weight;
        }
        u_controls[i] = exact_de_boor_curve(
            v_controls, patch.basis_v().knots(), patch.basis_v().degree(), v);
    }
    return exact_de_boor_curve(
        u_controls, patch.basis_u().knots(), patch.basis_u().degree(), u);
}

NurbsSurfacePatch3D make_weighted_patch(int degree_u, std::vector<double> knots_u,
                                       int degree_v, std::vector<double> knots_v)
{
    NurbsBasis1D basis_u(degree_u, std::move(knots_u));
    NurbsBasis1D basis_v(degree_v, std::move(knots_v));
    const int nu = basis_u.num_basis_functions();
    const int nv = basis_v.num_basis_functions();
    std::vector<std::vector<Eigen::Vector3d>> points(
        static_cast<std::size_t>(nu),
        std::vector<Eigen::Vector3d>(static_cast<std::size_t>(nv)));
    std::vector<std::vector<double>> weights(
        static_cast<std::size_t>(nu),
        std::vector<double>(static_cast<std::size_t>(nv)));
    for (int i = 0; i < nu; ++i) {
        for (int j = 0; j < nv; ++j) {
            points[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = {
                0.1 + 0.31 * i - 0.07 * j,
                -0.2 + 0.13 * j + 0.03 * i * i,
                0.05 * (i + 1) * (j + 1) - 0.11 * j};
            weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                0.2 + 0.17 * (1 + (3 * i + 2 * j) % 7);
        }
    }
    // This pair detects accidental conversion of a rounded double product.
    points[0][0].x() = 0.1;
    weights[0][0] = 0.3;
    return {std::move(basis_u), std::move(basis_v),
            std::move(points), std::move(weights)};
}

void require_in_bounds(const NurbsAabb3D& bounds, const HomogeneousPoint& value)
{
    require(value[3] > 0, "exact surface evaluation must have positive weight");
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        require(std::isfinite(bounds.lower[coordinate])
                    && std::isfinite(bounds.upper[coordinate]),
                "finite test fixture must have finite enclosing bounds");
        const Rational cartesian = value[coordinate] / value[3];
        require(Rational(bounds.lower[coordinate]) <= cartesian
                    && cartesian <= Rational(bounds.upper[coordinate]),
                "outward-rounded bounds must contain the exact Cartesian value");
    }
}

void test_exact_original_products_and_global_parameter_contract()
{
    const auto patch = make_weighted_patch(
        2, {0.1, 0.1, 0.1, 1.4, 1.4, 1.4},
        1, {-0.7, -0.7, 2.1, 2.1});
    const NurbsSurfaceModel3D model({patch}, {0}, {});
    const auto elements = extract_exact_native_bezier_elements_3d(model);
    require(elements.size() == 1, "single-span tensor patch yields one Bezier element");
    const auto& element = elements.front();
    const Rational exact_product = Rational(0.1) * Rational(0.3);
    require(exact_product != Rational(0.1 * 0.3),
            "product fixture must distinguish exact arithmetic from rounded multiplication");
    require(element.controls.front()[0] == exact_product,
            "original coordinates and weights must be converted BEFORE multiplication");
    require(element.controls.front()[3] == Rational(0.3),
            "original weight must be converted exactly");
    require(evaluate_native_bezier_homogeneous(element, 0.1, -0.7)
                == element.controls.front(),
            "source lower parameters must select the first homogeneous control");
    require(evaluate_native_bezier_homogeneous(element, 1.4, 2.1)
                == element.controls.back(),
            "source upper parameters must select the last homogeneous control");
    const double u = 0.47;
    const double v = 0.36;
    require(evaluate_native_bezier_homogeneous(element, u, v)
                == exact_de_boor_surface(patch, Rational(u), Rational(v)),
            "double parameter API must evaluate in original nonunit parameter coordinates");
    require(evaluate_native_bezier_homogeneous(element, u, v)
                == evaluate_native_bezier_homogeneous(element, Rational(u), Rational(v)),
            "double and exact-parameter APIs must agree without rounding the normalization");
}

void test_nonuniform_tensor_knot_insertion_against_exact_de_boor()
{
    // U has cubic interior multiplicities 1, 2, 3; V has quadratic
    // multiplicities 1, 2. Both axes therefore require genuine knot insertion.
    const auto patch = make_weighted_patch(
        3, {-0.75, -0.75, -0.75, -0.75, -0.11, 0.23, 0.23,
            0.61, 0.61, 0.61, 1.37, 1.37, 1.37, 1.37},
        2, {0.13, 0.13, 0.13, 0.37, 0.81, 0.81, 1.91, 1.91, 1.91});
    const NurbsSurfaceModel3D model({patch}, {0}, {});
    const auto elements = extract_exact_native_bezier_elements_3d(model);
    require(elements.size() == 12, "four U spans and three V spans yield twelve elements");
    const std::array<Rational, 5> fractions = {
        Rational(0), fraction(1, 7), fraction(1, 2), fraction(6, 7), Rational(1)};
    for (const auto& element : elements) {
        require(element.degree_u == 3 && element.degree_v == 2,
                "extraction must preserve both source degrees");
        require(element.patch_index == 0 && element.component == 0,
                "extraction must preserve source metadata");
        require(element.controls.size() == 12, "tensor control count must match the degrees");
        const auto bounds = element.bounds();
        for (const auto& control : element.controls)
            require_in_bounds(bounds, control);
        for (const auto& fu : fractions) {
            for (const auto& fv : fractions) {
                const Rational u = Rational(element.parameter_u0)
                    + fu * (Rational(element.parameter_u1) - Rational(element.parameter_u0));
                const Rational v = Rational(element.parameter_v0)
                    + fv * (Rational(element.parameter_v1) - Rational(element.parameter_v0));
                const auto actual = evaluate_native_bezier_homogeneous(element, u, v);
                require(actual == exact_de_boor_surface(patch, u, v),
                        "two-axis nonuniform knot insertion must preserve the original homogeneous spline EXACTLY");
                require_in_bounds(bounds, actual);
            }
        }
        const std::array<double, 4> parameters_u = {
            element.parameter_u0,
            std::nextafter(element.parameter_u0, element.parameter_u1),
            std::nextafter(element.parameter_u1, element.parameter_u0),
            element.parameter_u1};
        const std::array<double, 4> parameters_v = {
            element.parameter_v0,
            std::nextafter(element.parameter_v0, element.parameter_v1),
            std::nextafter(element.parameter_v1, element.parameter_v0),
            element.parameter_v1};
        for (double u : parameters_u) {
            for (double v : parameters_v) {
                require(evaluate_native_bezier_homogeneous(element, u, v)
                            == exact_de_boor_surface(patch, Rational(u), Rational(v)),
                        "endpoint and adjacent-IEEE-value evaluations must match the exact native spline");
            }
        }
        // The floating evaluator is only an additional approximate check;
        // it is not the oracle for any exact-equality assertion above.
        const double u = (element.parameter_u0 + element.parameter_u1) / 2.0;
        const double v = (element.parameter_v0 + element.parameter_v1) / 2.0;
        const auto exact = evaluate_native_bezier_homogeneous(element, u, v);
        const auto approximate = patch.evaluate(u, v);
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            require(std::abs(CGAL::to_double(exact[coordinate] / exact[3])
                             - approximate[coordinate]) <= 2e-12,
                    "ordinary double evaluation must agree as a numerical reference");
        }
    }
}

void test_analytic_plane_and_paraboloid_and_patch_metadata()
{
    std::vector<std::vector<Eigen::Vector3d>> plane(2, std::vector<Eigen::Vector3d>(2));
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            const double u = i == 0 ? 2.0 : 5.0;
            const double v = j == 0 ? -3.0 : 1.0;
            plane[i][j] = {u, v, 2.0 * u + 3.0 * v + 7.0};
        }
    }
    NurbsSurfacePatch3D plane_patch(
        NurbsBasis1D(1, std::vector<double>{2, 2, 5, 5}),
        NurbsBasis1D(1, std::vector<double>{-3, -3, 1, 1}), plane,
        std::vector<std::vector<double>>(2, std::vector<double>(2, 1.0)));
    std::vector<std::vector<Eigen::Vector3d>> paraboloid(
        3, std::vector<Eigen::Vector3d>(3));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            paraboloid[i][j] = {0.5 * i, 0.5 * j,
                                (i == 2 ? 1.0 : 0.0) + (j == 2 ? 1.0 : 0.0)};
        }
    }
    NurbsSurfacePatch3D paraboloid_patch(
        NurbsBasis1D(2, std::vector<double>{0, 0, 0, 1, 1, 1}),
        NurbsBasis1D(2, std::vector<double>{0, 0, 0, 1, 1, 1}), paraboloid,
        std::vector<std::vector<double>>(3, std::vector<double>(3, 1.0)));
    const NurbsSurfaceModel3D model({plane_patch, paraboloid_patch}, {0, 1}, {});
    const auto elements = extract_exact_native_bezier_elements_3d(model);
    require(elements.size() == 2 && elements[0].patch_index == 0
                && elements[0].component == 0 && elements[1].patch_index == 1
                && elements[1].component == 1,
            "elements must retain patch indices and separate component IDs");
    const Rational u = Rational(2) + fraction(3, 7);
    const Rational v = Rational(-3) + fraction(8, 5);
    const auto planar = evaluate_native_bezier_homogeneous(elements[0], u, v);
    require(planar == HomogeneousPoint{u, v, Rational(2) * u + Rational(3) * v + Rational(7), Rational(1)},
            "nonunit-domain plane must match its analytic exact expression without changing orientation");
    const Rational s = fraction(2, 7);
    const Rational t = fraction(3, 11);
    const auto curved = evaluate_native_bezier_homogeneous(elements[1], s, t);
    require(curved == HomogeneousPoint{s, t, s * s + t * t, Rational(1)},
            "quadratic paraboloid must match its analytic exact expression");
}

void test_directed_bounds_and_constant_degree()
{
    ExactNativeBezierElement3D rational_point;
    rational_point.controls = {{Rational(1), Rational(1), Rational(-1), Rational(3)}};
    auto bounds = rational_point.bounds();
    require(Rational(bounds.lower.x()) < fraction(1, 3)
                && fraction(1, 3) < Rational(bounds.upper.x()),
            "one-third coordinate requires genuinely outward-rounded, not nearest-only bounds");
    require(Rational(bounds.lower.z()) < -fraction(1, 3)
                && -fraction(1, 3) < Rational(bounds.upper.z()),
            "negative nonrepresentable coordinates also require outward rounding");
    require_in_bounds(bounds, evaluate_native_bezier_homogeneous(rational_point, 0.13, 0.91));
    rational_point.controls[0][0] = Rational(10);
    bounds = rational_point.bounds();
    require_in_bounds(bounds, rational_point.controls[0]);
    require(bounds.lower.x() > 3.0, "public control changes must not reuse a stale hidden bound");

    const auto constant_patch = make_weighted_patch(0, {2, 5}, 0, {-3, 1});
    const auto elements = extract_exact_native_bezier_elements_3d(
        NurbsSurfaceModel3D({constant_patch}, {0}, {}));
    require(elements.size() == 1 && elements[0].controls.size() == 1,
            "a degree-zero tensor spline extracts to one constant control");
    require(evaluate_native_bezier_homogeneous(elements[0], 3.0, -1.0)
                == exact_de_boor_surface(constant_patch, Rational(3), Rational(-1)),
            "degree-zero extraction must agree exactly with the original constant spline");
}

void test_rejections_without_parameter_clamping()
{
    const auto patch = make_weighted_patch(1, {2, 2, 5, 5}, 1, {-3, -3, 1, 1});
    const auto elements = extract_exact_native_bezier_elements_3d(
        NurbsSurfaceModel3D({patch}, {0}, {}));
    const auto& element = elements.front();
    const double infinity = std::numeric_limits<double>::infinity();
    require_throws<std::out_of_range>([&] {
        (void)evaluate_native_bezier_homogeneous(element, std::nextafter(2.0, -infinity), 0.0);
    }, "a one-ULP exterior U parameter must be rejected, not clamped");
    require_throws<std::out_of_range>([&] {
        (void)evaluate_native_bezier_homogeneous(element, 3.0, std::nextafter(1.0, infinity));
    }, "a one-ULP exterior V parameter must be rejected, not clamped");
    require_throws<std::out_of_range>([&] {
        (void)evaluate_native_bezier_homogeneous(element, fraction(1, 3), Rational(0));
    }, "the exact-parameter overload must reject parameters outside the source box");
    require_throws<std::invalid_argument>([&] {
        (void)evaluate_native_bezier_homogeneous(element, infinity, 0.0);
    }, "infinite parameters must be rejected before Gmpq conversion");
    require_throws<std::invalid_argument>([&] {
        (void)evaluate_native_bezier_homogeneous(
            element, 3.0, std::numeric_limits<double>::quiet_NaN());
    }, "NaN parameters must be rejected before Gmpq conversion");
    auto invalid = element;
    invalid.controls[0][3] = Rational(0);
    require_throws<std::invalid_argument>([&] { (void)invalid.bounds(); },
                                          "zero weights must not receive certified bounds");
    invalid.controls[0][3] = Rational(-1);
    require_throws<std::invalid_argument>([&] {
        (void)evaluate_native_bezier_homogeneous(invalid, 3.0, 0.0);
    }, "negative weights must not enter exact evaluation");
    invalid = element;
    invalid.controls.pop_back();
    require_throws<std::invalid_argument>([&] { (void)invalid.bounds(); },
                                          "incomplete tensor control nets must be rejected");
    invalid = element;
    invalid.parameter_u1 = invalid.parameter_u0;
    require_throws<std::invalid_argument>([&] { (void)invalid.bounds(); },
                                          "degenerate element domains must be rejected");

    // Construction itself is valid in the existing native model. Extraction
    // must explicitly reject its unsupported unclamped active endpoints.
    const auto unclamped_u = make_weighted_patch(
        2, {-1, -0.5, 0, 0.3, 0.7, 1, 1.5, 2}, 1, {0, 0, 1, 1});
    const NurbsSurfaceModel3D model_u({unclamped_u}, {0}, {});
    require_throws<std::invalid_argument>([&] {
        (void)extract_exact_native_bezier_elements_3d(model_u);
    }, "unclamped U endpoints must be rejected rather than silently certified");
    const auto unclamped_v = make_weighted_patch(
        1, {0, 0, 1, 1}, 2, {-1, -0.5, 0, 0.3, 0.7, 1, 1.5, 2});
    const NurbsSurfaceModel3D model_v({unclamped_v}, {0}, {});
    require_throws<std::invalid_argument>([&] {
        (void)extract_exact_native_bezier_elements_3d(model_v);
    }, "unclamped V endpoints must be rejected rather than silently certified");
}

} // namespace

int main()
{
    try {
        test_exact_original_products_and_global_parameter_contract();
        test_nonuniform_tensor_knot_insertion_against_exact_de_boor();
        test_analytic_plane_and_paraboloid_and_patch_metadata();
        test_directed_bounds_and_constant_degree();
        test_rejections_without_parameter_clamping();
        std::cout << "native_nurbs_exact_geometry_3d_test: all checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native_nurbs_exact_geometry_3d_test failed: " << error.what() << '\n';
        return 1;
    }
}
