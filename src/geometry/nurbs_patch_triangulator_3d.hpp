#pragma once

#include "edge_patch_3d.hpp"
#include "nurbs_surface_3d.hpp"
#include "src/interface/interface_3d.hpp"

#include <Eigen/Dense>

#include <array>
#include <vector>

namespace kfbim::geometry3d {

struct NurbsPatchEdgeOffset3D {
    double knot_offset = 0.0;
    double physical_edge_length = 0.0;
    int    sample_count = 0;
    double min_cross_speed = 0.0;
    double max_cross_speed = 0.0;
};

struct NurbsPatchEdgeOffsets3D {
    NurbsPatchEdgeOffset3D u_min;
    NurbsPatchEdgeOffset3D u_max;
    NurbsPatchEdgeOffset3D v_min;
    NurbsPatchEdgeOffset3D v_max;
};

struct NurbsPatchTriangulatorOptions3D {
    double edge_buffer_factor = 0.55;
    double edge_sample_step_factor = 1.0;
    int    min_edge_samples = 8;
    int    edge_length_quadrature_samples = 64;
    double max_param_offset_fraction = 0.20;

    double H_factor = 4.0;
    double max_edge_over_H = 1.18;
    double metric_ratio_tolerance = 1.55;
    int    max_depth = 8;

    // Coincident C1 patch edges are smooth parameterization seams, not
    // geometric features.  They are left unbuffered and their P2 nodes are
    // welded. Non-smooth/unmatched edges retain the physical edge buffer only
    // on the correction/collocation tessellation, so no iteration DOF is
    // placed on a geometric edge or vertex. The separate geometry_interface
    // is always generated on the complete, unbuffered NURBS patch domains.
    bool   detect_smooth_seams = true;
    double edge_match_tolerance = 1.0e-10;
    double smooth_normal_cosine = 0.999999;

    // Optional fixed-physical-radius artificial cylinders around geometric
    // feature edges. This is independent of edge_buffer_factor and h.
    EdgePatchConfig3D edge_patch;
};

struct NurbsPatchGeometricEdge3D {
    Eigen::Vector3d start = Eigen::Vector3d::Zero();
    Eigen::Vector3d end = Eigen::Vector3d::Zero();
    std::array<int, 2> incident_patches{{-1, -1}};
    std::array<int, 2> incident_edge_samples{{0, 0}};
};

struct NurbsParamTriangle3D {
    int patch_index = 0;
    std::array<Eigen::Vector2d, 3> uv;
};

struct NurbsPatchTriangulationSummary3D {
    int num_patches = 0;
    int num_rectangles = 0;
    int num_splits = 0;
    int num_triangles = 0;
    int num_points = 0;
    int num_smooth_seams = 0;
    int num_feature_edges = 0;
    int num_feature_vertices = 0;
    double target_H = 0.0;
    double max_physical_edge_over_H = 0.0;
    double max_metric_ratio = 0.0;
};

struct NurbsPatchTriangulation3D {
    // Correction/collocation surface. Feature-edge buffers are applied here,
    // so no GMRES interface DOF is placed on a non-smooth edge or vertex.
    kfbim::Interface3D interface;
    std::vector<NurbsParamTriangle3D> triangles;

    // Complete, unbuffered geometric surface used for domain labeling and
    // Cartesian-edge crossing queries. It remains seamless at feature edges
    // and is deliberately independent of the collocation-DOF trimming above.
    kfbim::Interface3D geometry_interface;
    std::vector<NurbsParamTriangle3D> geometry_triangles;

    std::vector<NurbsPatchEdgeOffsets3D> edge_offsets;
    std::vector<NurbsPatchGeometricEdge3D> feature_edges;
    std::vector<EdgePatch3D> edge_patches;
    std::vector<Eigen::Vector3d> feature_vertices;
    NurbsPatchTriangulationSummary3D summary;
};

[[nodiscard]] NurbsPatchEdgeOffsets3D compute_nurbs_patch_edge_offsets_3d(
    const NurbsSurfacePatch3D& patch,
    double h,
    const NurbsPatchTriangulatorOptions3D& options = {});

[[nodiscard]] NurbsPatchTriangulation3D triangulate_nurbs_surface_patches_3d(
    const std::vector<NurbsSurfacePatch3D>& patches,
    double h,
    const NurbsPatchTriangulatorOptions3D& options = {});

} // namespace kfbim::geometry3d
