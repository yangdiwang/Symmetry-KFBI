#include "bicubic_nurbs_models_3d.hpp"
#include "bicubic_nurbs/construction.hpp"
#include <stdexcept>
#include <algorithm>

namespace kfbim::app3d {
std::array<Eigen::Vector2d,2> bicubic_nurbs_model_parameters::cap_corners(int q,bool top)
{
    const std::array<Eigen::Vector2d,4> corners{{
        {half_square,-half_square},{half_square,half_square},
        {-half_square,half_square},{-half_square,-half_square}}};
    std::array<Eigen::Vector2d,2> result{{corners[q],corners[(q+1)%4]}};
    if(!top) std::swap(result[0],result[1]);
    return result;
}

BicubicNurbsModel3D make_bicubic_nurbs_model_3d(BicubicNurbsShape3D shape)
{
    BicubicNurbsModel3D geometry;
    switch (shape) {
    case BicubicNurbsShape3D::Box:
        bicubic_nurbs_detail::box(geometry);
        geometry.surface.name="bicubic_nurbs_box";
        break;
    case BicubicNurbsShape3D::SolidCylinder:
        bicubic_nurbs_detail::solid_cylinder(geometry);
        geometry.surface.name="bicubic_nurbs_solid_cylinder";
        break;
    case BicubicNurbsShape3D::LPrism:
        bicubic_nurbs_detail::l_prism(geometry);
        geometry.surface.name="bicubic_nurbs_l_prism";
        break;
    case BicubicNurbsShape3D::UPrism:
        bicubic_nurbs_detail::u_prism(geometry);
        geometry.surface.name="bicubic_nurbs_u_prism";
        break;
    default:
        throw std::invalid_argument("unknown bicubic NURBS shape");
    }
    geometry.surface.description="Exact bicubic NURBS surface with per-patch analysis charts";
    bicubic_nurbs_detail::connect_complete_edges(geometry);
    return geometry;
}
} // namespace kfbim::app3d
