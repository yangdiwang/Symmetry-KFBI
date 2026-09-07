#include "src/support/density/native_density_resolution_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {

using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::NativeDensityResolution3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::choose_native_density_resolution_3d;
using kfbim::app3d::make_native_nurbs_surface_3d;

constexpr double kBoxSide = 3.0;
constexpr double kTargetCrossingsPerCell = 14.0;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_near(double actual,
                  double expected,
                  double tolerance,
                  const std::string& message)
{
    require(std::abs(actual - expected) <= tolerance, message);
}

template <class Function>
void require_throws_invalid_argument(Function&& function,
                                     const std::string& message)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void check_closed_form(const NativeDensityResolution3D& result)
{
    require(result.patch_count > 0, "resolution retains a positive patch count");
    require(result.patch_orientation_integrals.size()
                == static_cast<std::size_t>(result.patch_count),
            "resolution records one orientation integral per patch");
    const double patch_sum = std::accumulate(
        result.patch_orientation_integrals.begin(),
        result.patch_orientation_integrals.end(), 0.0);
    require_near(patch_sum, result.orientation_integral,
                 2.0e-14 * result.orientation_integral,
                 "patch diagnostics sum to the surface integral");
    require_near(
        result.predicted_crossings,
        result.orientation_integral
            / (result.cartesian_spacing * result.cartesian_spacing),
        2.0e-14 * result.predicted_crossings,
        "predicted crossings use the direct orientation formula");
    const double expected_continuous = std::sqrt(
        result.orientation_integral
        / (result.patch_count * result.target_crossings_per_cell
           * result.cartesian_spacing * result.cartesian_spacing));
    require_near(result.continuous_elements_per_direction,
                 expected_continuous,
                 2.0e-14 * expected_continuous,
                 "continuous element count uses the direct formula");
    const int expected_elements = std::max(
        1,
        static_cast<int>(std::floor(std::nextafter(
            expected_continuous, std::numeric_limits<double>::infinity()))));
    require(result.elements_per_direction == expected_elements,
            "integer element count is a single closed-form floor");
    require(result.coefficients_per_direction == expected_elements + 3,
            "cubic coefficient count is element count plus three");
}

void check_levels(GeometryKind3D kind,
                  double expected_orientation_integral,
                  const std::array<int, 3>& actual_crossings,
                  const std::array<int, 3>& expected_coefficients)
{
    const NativeNurbsSurface3D surface = make_native_nurbs_surface_3d(kind);
    const std::array<int, 3> levels{{32, 64, 128}};
    int previous_coefficients = 0;
    for (std::size_t level = 0; level < levels.size(); ++level) {
        const double h = kBoxSide / static_cast<double>(levels[level]);
        const NativeDensityResolution3D result =
            choose_native_density_resolution_3d(surface, h);
        check_closed_form(result);
        require_near(result.orientation_integral,
                     expected_orientation_integral,
                     2.0e-6 * expected_orientation_integral,
                     "five-point Gauss orientation integral matches geometry");
        require(result.coefficients_per_direction
                    == expected_coefficients[level],
                "native automatic coefficient count");
        require(result.coefficients_per_direction > previous_coefficients,
                "native coefficient count grows monotonically with refinement");
        previous_coefficients = result.coefficients_per_direction;

        const double actual = static_cast<double>(actual_crossings[level]);
        const double ratio = result.predicted_crossings / actual;
        require(ratio > 0.90 && ratio < 1.10,
                "orientation prediction and measured crossing count are same scale");
    }
}

void test_cylinder_and_l_prism_levels()
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double cylinder_orientation =
        8.0 * (0.55 + 0.25) * 1.30
        + 2.0 * pi * (0.55 * 0.55 - 0.25 * 0.25);
    check_levels(GeometryKind3D::HollowCylinder,
                 cylinder_orientation,
                 {{1174, 4544, 17812}},
                 {{5, 7, 11}});

    // Every L-prism face is axis aligned, hence ||n||_1 = 1 and the
    // orientation integral equals its exact surface area.
    const double l_prism_orientation =
        2.0 * 3.0 * 0.60 * 0.60 + 8.0 * 0.60 * 1.30;
    check_levels(GeometryKind3D::LPrism,
                 l_prism_orientation,
                 {{982, 3926, 15122}},
                 {{5, 7, 12}});
}

void test_patch_orientation_diagnostics()
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const NativeDensityResolution3D result =
        choose_native_density_resolution_3d(cylinder, kBoxSide / 32.0);
    require(result.patch_orientation_integrals.size() == 16,
            "cylinder records all sixteen patch integrals");
    for (int patch = 0; patch < 4; ++patch) {
        require_near(result.patch_orientation_integrals[
                         static_cast<std::size_t>(patch)],
                     2.0 * 0.55 * 1.30,
                     3.0e-6,
                     "outer-wall quarter orientation integral");
        require_near(result.patch_orientation_integrals[
                         static_cast<std::size_t>(4 + patch)],
                     2.0 * 0.25 * 1.30,
                     3.0e-6,
                     "inner-wall quarter orientation integral");
        const double annulus_quarter =
            0.25 * pi * (0.55 * 0.55 - 0.25 * 0.25);
        require_near(result.patch_orientation_integrals[
                         static_cast<std::size_t>(8 + patch)],
                     annulus_quarter,
                     3.0e-7,
                     "top-annulus quarter orientation integral");
        require_near(result.patch_orientation_integrals[
                         static_cast<std::size_t>(12 + patch)],
                     annulus_quarter,
                     3.0e-7,
                     "bottom-annulus quarter orientation integral");
    }
}

void test_target_and_input_contracts()
{
    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const double h = kBoxSide / 64.0;
    const NativeDensityResolution3D default_target =
        choose_native_density_resolution_3d(cylinder, h);
    const NativeDensityResolution3D explicit_target =
        choose_native_density_resolution_3d(
            cylinder, h, kTargetCrossingsPerCell);
    require(default_target.coefficients_per_direction
                == explicit_target.coefficients_per_direction
                && default_target.predicted_crossings
                       == explicit_target.predicted_crossings,
            "default crossing target is fourteen");

    const NativeDensityResolution3D larger_target =
        choose_native_density_resolution_3d(cylinder, h, 28.0);
    require(larger_target.coefficients_per_direction
                <= default_target.coefficients_per_direction,
            "larger crossing target cannot request more density cells");

    require_throws_invalid_argument(
        [&] { (void)choose_native_density_resolution_3d(cylinder, 0.0); },
        "zero Cartesian spacing is rejected");
    require_throws_invalid_argument(
        [&] {
            (void)choose_native_density_resolution_3d(
                cylinder, h, std::numeric_limits<double>::infinity());
        },
        "non-finite crossing target is rejected");
}

} // namespace

int main()
{
    try {
        test_cylinder_and_l_prism_levels();
        test_patch_orientation_diagnostics();
        test_target_and_input_contracts();
        std::cout << "native density resolution tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native density resolution test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
