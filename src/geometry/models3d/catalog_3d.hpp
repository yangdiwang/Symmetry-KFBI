#pragma once
#include "src/geometry/nurbs_surface_model_3d.hpp"
#include <string>
#include <vector>

namespace kfbim::geometry3d {
struct GeometryModelInfo3D {
    std::string id;
    std::string description;
};

// Geometry discovery only. Solver/chart support is decided by the case adapter.
// Explicit family IDs preserve distinct historical patch layouts and dimensions.
const std::vector<GeometryModelInfo3D>& available_geometry_models_3d();
NurbsSurfaceModel3D make_geometry_model_3d(const std::string& id);
} // namespace kfbim::geometry3d
