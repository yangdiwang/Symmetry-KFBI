#pragma once

#include "src/geometry/models3d/native_surface_3d.hpp"
#include "src/geometry/nurbs_basis.hpp"

#include <Eigen/Dense>

#include <array>
#include <string>

namespace kfbim::app3d::native_detail {

inline constexpr double kPi = 3.141592653589793238462643383279502884;

struct RationalQuarterArc2D {
    std::array<Eigen::Vector2d, 3> controls;
    std::array<double, 3> weights;
};

RationalQuarterArc2D quarter_arc(double angle0);
geometry::NurbsBasis1D quadratic_unit_basis();
geometry::NurbsBasis1D linear_unit_basis();

void append_patch(NativeNurbsSurface3D& surface,
                  std::string name,
                  geometry3d::NurbsSurfacePatch3D patch);

void connect_smooth(NativeNurbsSurface3D& surface,
                    int patch_a,
                    PatchEdge3D edge_a,
                    int patch_b,
                    PatchEdge3D edge_b,
                    bool reversed);

void connect_feature(NativeNurbsSurface3D& surface,
                     int patch_a,
                     PatchEdge3D edge_a,
                     double begin_a,
                     double end_a,
                     int patch_b,
                     PatchEdge3D edge_b,
                     double begin_b,
                     double end_b,
                     bool reversed);

void append_prism_top(NativeNurbsSurface3D& surface,
                      const std::string& name,
                      double xmin,
                      double xmax,
                      double ymin,
                      double ymax,
                      double z);

void append_prism_bottom(NativeNurbsSurface3D& surface,
                         const std::string& name,
                         double xmin,
                         double xmax,
                         double ymin,
                         double ymax,
                         double z);

} // namespace kfbim::app3d::native_detail
