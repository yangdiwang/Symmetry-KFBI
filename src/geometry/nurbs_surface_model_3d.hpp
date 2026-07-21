#pragma once

#include "nurbs_surface_3d.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <vector>

namespace kfbim::geometry3d {

enum class NurbsPatchEdge3D { UMin, UMax, VMin, VMax };

struct NurbsPatchEdgeInterval3D {
    int patch = -1;
    NurbsPatchEdge3D edge = NurbsPatchEdge3D::UMin;
    double begin = 0.0;
    double end = 0.0;
};

struct NurbsPatchEdgeConnection3D {
    NurbsPatchEdgeInterval3D first;
    NurbsPatchEdgeInterval3D second;
    bool reversed = false;
    bool g1 = false;
};

struct NurbsSurfaceTopologyDiagnostics3D {
    int patch_count = 0;
    int component_count = 0;
    int connection_count = 0;
    int uncovered_interval_count = 0;
    int multiply_covered_interval_count = 0;
    int position_mismatch_count = 0;
    int orientation_mismatch_count = 0;
    int g1_normal_mismatch_count = 0;
};

struct NurbsAabb3D {
    Eigen::Vector3d lower = Eigen::Vector3d::Zero();
    Eigen::Vector3d upper = Eigen::Vector3d::Zero();

    bool overlaps(const NurbsAabb3D& other, double tolerance = 0.0) const;
    bool contains(const Eigen::Vector3d& point,
                  double tolerance = 0.0) const;
    double max_extent() const;
    double diameter() const;
};

class NurbsSurfaceModel3D {
public:
    NurbsSurfaceModel3D(
        std::vector<NurbsSurfacePatch3D> patches,
        std::vector<int> patch_components,
        std::vector<NurbsPatchEdgeConnection3D> connections);

    int num_patches() const;
    int num_components() const;
    const NurbsSurfacePatch3D& patch(int patch) const;
    int patch_component(int patch) const;
    const std::vector<NurbsSurfacePatch3D>& patches() const;
    const std::vector<NurbsPatchEdgeConnection3D>& connections() const;
    NurbsSurfaceTopologyDiagnostics3D validate_closed(
        double relative_tolerance = 1e-10) const;
    NurbsAabb3D control_bounds() const;

private:
    std::vector<NurbsSurfacePatch3D> patches_;
    std::vector<int> patch_components_;
    std::vector<NurbsPatchEdgeConnection3D> connections_;
};

NurbsPatchEdgeInterval3D full_patch_edge_interval(
    const NurbsSurfacePatch3D& patch,
    int patch_index,
    NurbsPatchEdge3D edge);

} // namespace kfbim::geometry3d
