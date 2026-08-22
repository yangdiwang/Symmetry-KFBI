#pragma once

#include "native_nurbs_surface_3d.hpp"

#include <vector>

namespace kfbim::app3d {

// Geometry-only estimate for a uniform tensor density resolution on every
// native NURBS patch.  The orientation integral predicts the number of
// coordinate-aligned Cartesian edges intersected by the surface:
//
//   predicted_crossings = integral_Gamma ||n||_1 dS / h^2.
//
// The estimate is closed form once that integral is known; no candidate
// density space is constructed or evaluated.
struct NativeDensityResolution3D {
    int patch_count = 0;
    double cartesian_spacing = 0.0;
    double target_crossings_per_cell = 14.0;

    // One entry per NativeNurbsSurface3D::patches item, in the same order.
    std::vector<double> patch_orientation_integrals;
    double orientation_integral = 0.0;
    double predicted_crossings = 0.0;
    double continuous_elements_per_direction = 0.0;
    int elements_per_direction = 1;
    int coefficients_per_direction = 4;
};

// Uses tensor-product five-point Gauss quadrature on every non-empty knot
// span of every patch.  Cubic open-uniform density splines then use
// coefficients_per_direction = elements_per_direction + 3.
[[nodiscard]] NativeDensityResolution3D
choose_native_density_resolution_3d(
    const NativeNurbsSurface3D& surface,
    double cartesian_spacing,
    double target_crossings_per_cell = 14.0);

} // namespace kfbim::app3d
