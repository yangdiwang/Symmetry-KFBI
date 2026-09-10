#pragma once
#include "src/geometry/nurbs_cartesian_domain_3d.hpp"

namespace kfbim::geometry3d {
// A whole positive-axis grid line can be split only if its full root set is
// certified, transverse, and separated from every grid node. Otherwise the
// caller must use the original per-edge certified query, not round an event
// into an arbitrary neighboring cell. No roots are merged or dropped here.
bool certified_grid_line_can_partition_3d(
    const CartesianGrid3D& grid, int axis,
    const NurbsCartesianEdgeIntersections3D& line, double geometry_tolerance);

// Requires a line accepted by certified_grid_line_can_partition_3d. Keeps all
// physical roots/owners on this edge and rebuilds its component parity.
NurbsCartesianEdgeIntersections3D extract_grid_line_edge_3d(
    const NurbsCartesianEdgeQuery3D& edge,
    const std::vector<NurbsSurfaceCrossing3D>& line_crossings);
} // namespace kfbim::geometry3d
