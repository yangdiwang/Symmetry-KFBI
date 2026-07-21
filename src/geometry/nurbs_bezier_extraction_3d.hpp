#pragma once

#include "nurbs_surface_model_3d.hpp"
#include "rational_bezier_element_3d.hpp"

#include <vector>

namespace kfbim::geometry3d {

std::vector<RationalBezierElement3D>
extract_rational_bezier_elements_3d(const NurbsSurfaceModel3D& model);

} // namespace kfbim::geometry3d
