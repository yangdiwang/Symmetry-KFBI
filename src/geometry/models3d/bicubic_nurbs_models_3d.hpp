#pragma once
#include "native_surface_3d.hpp"

namespace kfbim::app3d {
// Physical dimensions and analysis-chart parameters of these fixed models.
// The cylinder's angular analysis chart differs from its rational NURBS chart.
namespace bicubic_nurbs_model_parameters {
inline constexpr double pi = 3.141592653589793238462643383279502884;
inline constexpr double radius = 0.54, half_square = 0.22, z0 = -0.5, z1 = 0.5;
std::array<Eigen::Vector2d,2> cap_corners(int quarter, bool top);
}

enum class SurfaceAnalysisChartKind3D { Affine, CylinderSide, CylinderCapRing };

enum class BicubicNurbsShape3D { Box, SolidCylinder, LPrism, UPrism };

struct SurfaceAnalysisChart3D {
    SurfaceAnalysisChartKind3D kind = SurfaceAnalysisChartKind3D::Affine;
    bool planar = true;
    // Canonical native physical sheet: minimum patch ID in its G1 component.
    int sheet = -1;
    double Lu = 0.0;
    double Lv = 0.0;
    double density_length = 0.0;
    Eigen::Vector3d local_origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d du = Eigen::Vector3d::Zero();
    Eigen::Vector3d dv = Eigen::Vector3d::Zero();
    int quarter = -1;
    bool top = true;
};

// Geometry-only result in body coordinates. Analysis metadata is ordered by
// the same stable patch IDs as surface.patches; no PDE or density is constructed.
struct BicubicNurbsModel3D {
    NativeNurbsSurface3D surface;
    std::vector<SurfaceAnalysisChart3D> analysis_patches;
    bool native_parameters_match_analysis = true;
};

BicubicNurbsModel3D make_bicubic_nurbs_model_3d(BicubicNurbsShape3D shape);
} // namespace kfbim::app3d
