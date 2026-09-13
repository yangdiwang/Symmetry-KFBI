#pragma once

#include "src/geometry/models3d/native_surface_3d.hpp"

namespace kfbim::app3d::native_detail {

NativeNurbsSurface3D make_default_torus_3d();
NativeNurbsSurface3D make_hollow_cylinder_3d();
NativeNurbsSurface3D make_l_prism_3d();
NativeNurbsSurface3D make_u_prism_3d();

} // namespace kfbim::app3d::native_detail
