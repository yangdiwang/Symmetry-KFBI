#pragma once
#include "../bicubic_nurbs_models_3d.hpp"

namespace kfbim::app3d::bicubic_nurbs_detail {
using bicubic_nurbs_model_parameters::pi;
using bicubic_nurbs_model_parameters::radius;
using bicubic_nurbs_model_parameters::half_square;
using bicubic_nurbs_model_parameters::z0;
using bicubic_nurbs_model_parameters::z1;
using bicubic_nurbs_model_parameters::cap_corners;
using Vector = Eigen::Vector3d;
using HNet = std::vector<std::vector<Eigen::Vector4d>>;
using Patch = geometry3d::NurbsSurfacePatch3D;
using Edge = geometry3d::NurbsPatchEdge3D;
struct PrismCell { std::string name; double x0, x1, y0, y1; };

Patch bicubic(HNet net);
void append(BicubicNurbsModel3D&, const std::string&, Patch, SurfaceAnalysisChart3D);
void affine(BicubicNurbsModel3D&, const std::string&, const Vector&, const Vector&, const Vector&);
void extrude_prism(BicubicNurbsModel3D&, const std::vector<Eigen::Vector2d>&,
                   const std::vector<PrismCell>&);
void connect_complete_edges(BicubicNurbsModel3D&);
void box(BicubicNurbsModel3D&);
void l_prism(BicubicNurbsModel3D&);
void u_prism(BicubicNurbsModel3D&);
void solid_cylinder(BicubicNurbsModel3D&);
} // namespace kfbim::app3d::bicubic_nurbs_detail
