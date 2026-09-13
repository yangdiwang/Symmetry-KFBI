#include "catalog_3d.hpp"
#include "native_surface_3d.hpp"
#include "bicubic_nurbs_models_3d.hpp"
#include "industrial/models_3d.hpp"
#include <stdexcept>

namespace kfbim::geometry3d {
namespace {
struct Entry {
    GeometryModelInfo3D info;
    NurbsSurfaceModel3D (*build)();
};

const std::vector<Entry>& entries()
{
    using namespace app3d;
    static const std::vector<Entry> models{
        {{"native/torus", "16-patch rational-quadratic NURBS torus"},
         [] { return make_native_nurbs_surface_3d(GeometryKind3D::Torus).geometry_model(); }},
        {{"native/hollow_cylinder", "Hollow NURBS cylinder with annular end caps"},
         [] { return make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder).geometry_model(); }},
        {{"native/l_prism", "12-patch bilinear NURBS L prism with interval edge connections"},
         [] { return make_native_nurbs_surface_3d(GeometryKind3D::LPrism).geometry_model(); }},
        {{"native/u_prism", "18-patch bilinear NURBS U prism"},
         [] { return make_native_nurbs_surface_3d(GeometryKind3D::UPrism).geometry_model(); }},
        {{"bicubic_nurbs/box", "6-patch bicubic NURBS box"},
         [] { return make_bicubic_nurbs_model_3d(BicubicNurbsShape3D::Box).surface.geometry_model(); }},
        {{"bicubic_nurbs/l_prism", "14-patch bicubic NURBS L prism"},
         [] { return make_bicubic_nurbs_model_3d(BicubicNurbsShape3D::LPrism).surface.geometry_model(); }},
        {{"bicubic_nurbs/u_prism", "22-patch bicubic NURBS U prism"},
         [] { return make_bicubic_nurbs_model_3d(BicubicNurbsShape3D::UPrism).surface.geometry_model(); }},
        {{"bicubic_nurbs/solid_cylinder", "14-patch solid rational NURBS cylinder, radius 0.54, angular analysis charts"},
         [] { return make_bicubic_nurbs_model_3d(BicubicNurbsShape3D::SolidCylinder).surface.geometry_model(); }},
        {{"industrial/sleeve", "NURBS sleeve with fillets and shoulder"},
         [] { return make_industrial_sleeve_3d().geometry_model(); }},
        {{"industrial/u_bracket", "NURBS bracket with ear through-holes"},
         [] { return make_industrial_u_bracket_3d().geometry_model(); }},
        {{"industrial/flange", "NURBS flange with bolt holes"},
         [] { return make_industrial_flange_3d().geometry_model(); }},
        {{"industrial/impeller", "NURBS impeller with curved vanes"},
         [] { return make_industrial_impeller_3d().geometry_model(); }}
    };
    return models;
}
} // namespace

const std::vector<GeometryModelInfo3D>& available_geometry_models_3d()
{
    static const auto models=[] {
        std::vector<GeometryModelInfo3D> result;
        for (const auto& entry : entries()) result.push_back(entry.info);
        return result;
    }();
    return models;
}

NurbsSurfaceModel3D make_geometry_model_3d(const std::string& id)
{
    for (const auto& entry : entries())
        if (entry.info.id==id) return entry.build();
    throw std::invalid_argument("unknown geometry model: "+id);
}
} // namespace kfbim::geometry3d
