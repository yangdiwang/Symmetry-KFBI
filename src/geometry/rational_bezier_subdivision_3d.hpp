#pragma once

#include "rational_bezier_element_3d.hpp"

#include <vector>

namespace kfbim::geometry3d {

std::vector<RationalBezierElement3D>
subdivide_rational_bezier_elements_to_extent_3d(
    const std::vector<RationalBezierElement3D>& elements,
    double maximum_extent);

} // namespace kfbim::geometry3d
