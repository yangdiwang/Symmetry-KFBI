#pragma once

#include "native_nurbs_surface_3d.hpp"

#include <Eigen/Dense>

#include <memory>
#include <vector>

namespace kfbim::app3d {

enum class GeometricFeatureTraceKind3D {
    Regular,
    Edge,
    Vertex
};

struct GeometricFeatureTraceMap3D {
    GeometricFeatureTraceKind3D kind =
        GeometricFeatureTraceKind3D::Regular;
    std::vector<int> value_ids;
    Eigen::MatrixXd harmonic_map;
    double condition = 0.0;
    double constraint_residual = 0.0;
    int feature_count = 0;
    int sheet_count = 0;

    [[nodiscard]] bool active() const noexcept
    {
        return kind != GeometricFeatureTraceKind3D::Regular;
    }
};

struct GeometricFeatureTraceSummary3D {
    int feature_segments = 0;
    int feature_vertices = 0;
    int active_maps = 0;
    int edge_maps = 0;
    int vertex_maps = 0;
    int maximum_features_per_map = 0;
    int maximum_sheets_per_map = 0;
    double maximum_constraint_residual = 0.0;
};

struct GeometricFeatureTraceOptions3D {
    double activation_radius_over_h = 4.0;
    double vertex_radius_over_h = 3.0;
    int polyline_segments = 16;
    int minimum_samples_per_sheet = 24;
};

class GeometricFeatureTraceFit3D {
public:
    GeometricFeatureTraceFit3D(
        const NativeNurbsSurface3D& surface,
        const SurfaceDofCloud3D& cloud,
        double h,
        int degree,
        GeometricFeatureTraceOptions3D options = {});

    [[nodiscard]] int degree() const noexcept;
    [[nodiscard]] int harmonic_dimension() const noexcept;
    [[nodiscard]] bool active(int center_dof) const;
    [[nodiscard]] const GeometricFeatureTraceMap3D& map(
        int center_dof) const;
    [[nodiscard]] Eigen::VectorXd coefficients(
        int center_dof,
        const Eigen::VectorXd& value_jump) const;
    [[nodiscard]] std::vector<double> condition_values() const;
    [[nodiscard]] const GeometricFeatureTraceSummary3D& summary() const
        noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace kfbim::app3d
