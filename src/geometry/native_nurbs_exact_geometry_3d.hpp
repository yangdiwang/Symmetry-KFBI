#pragma once

#include "nurbs_surface_model_3d.hpp"

#include <CGAL/Gmpq.h>

#include <array>
#include <vector>

namespace kfbim::geometry3d {

using ExactNativeHomogeneousPoint3D = std::array<CGAL::Gmpq, 4>;

// The homogeneous control net is exact for the original IEEE-double NURBS
// data, not for a previously rounded floating-point Bezier extraction.
// Controls use u-major ordering: controls[i * (degree_v + 1) + j].
// Parameter orientation is unchanged from the source patch.
struct ExactNativeBezierElement3D {
    int patch_index = -1;
    int component = -1;
    int degree_u = 0;
    int degree_v = 0;
    double parameter_u0 = 0.0;
    double parameter_u1 = 1.0;
    double parameter_v0 = 0.0;
    double parameter_v1 = 1.0;
    std::vector<ExactNativeHomogeneousPoint3D> controls;

    // Outward-rounded enclosure of the Cartesian control hull. Positive
    // homogeneous weights make this a conservative surface enclosure.
    NurbsAabb3D bounds() const;
};

// Requires finite original data, positive weights, and clamped active knot
// endpoints. General nonuniform interior knots and multiplicities are allowed.
// Unsupported unclamped endpoints are rejected, never silently certified.
std::vector<ExactNativeBezierElement3D>
extract_exact_native_bezier_elements_3d(const NurbsSurfaceModel3D& model);

// Evaluate at source-patch parameters (not unit element-local parameters).
// The double overload first converts each argument exactly to Gmpq. Neither
// overload clamps or rounds the parameter to the element's parameter box.
ExactNativeHomogeneousPoint3D evaluate_native_bezier_homogeneous(
    const ExactNativeBezierElement3D& element, double u, double v);

ExactNativeHomogeneousPoint3D evaluate_native_bezier_homogeneous(
    const ExactNativeBezierElement3D& element,
    const CGAL::Gmpq& u, const CGAL::Gmpq& v);

} // namespace kfbim::geometry3d
