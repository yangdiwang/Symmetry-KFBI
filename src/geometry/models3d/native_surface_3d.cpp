#include "native_surface_3d.hpp"

#include "native/models_3d.hpp"

#include <stdexcept>

namespace kfbim::app3d {

geometry3d::NurbsSurfaceModel3D NativeNurbsSurface3D::geometry_model() const
{
    return {patches, patch_components, geometric_connections};
}

NativeNurbsSurface3D make_native_nurbs_surface_3d(GeometryKind3D kind)
{
    switch (kind) {
    case GeometryKind3D::Torus:
        return native_detail::make_default_torus_3d();
    case GeometryKind3D::HollowCylinder:
        return native_detail::make_hollow_cylinder_3d();
    case GeometryKind3D::LPrism:
        return native_detail::make_l_prism_3d();
    case GeometryKind3D::UPrism:
        return native_detail::make_u_prism_3d();
    }
    throw std::invalid_argument("unknown native 3D geometry kind");
}

} // namespace kfbim::app3d
