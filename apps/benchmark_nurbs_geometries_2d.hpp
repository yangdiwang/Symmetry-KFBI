#pragma once

#include "src/geometry/nurbs_boundary_2d.hpp"
#include "src/interface/interface_2d.hpp"

#include <string>
#include <vector>

namespace kfbim::app2d {

enum class BenchmarkNurbsGeometryKind2D {
    Circle,
    Ellipse,
    Flower,
    Heart,
    LShape
};

enum class BenchmarkGeometryContinuity2D {
    G2,
    G1,
    G0
};

// A location at which the numerical density space must recognize a loss of
// geometric G2 continuity.  Interface2D only distinguishes smooth points from
// feature points; this record retains the stronger G1/G0 classification needed
// when the coefficient spaces are assembled later.
struct BenchmarkNurbsFeature2D {
    int point = -1;
    int span_minus = -1;
    int span_plus = -1;
    double parameter = 0.0;
    BenchmarkGeometryContinuity2D continuity =
        BenchmarkGeometryContinuity2D::G0;
};

struct BenchmarkNurbsGeometry2D {
    BenchmarkNurbsGeometryKind2D kind;
    std::string name;
    Interface2D interface;

    // Geometry-attached exterior-trace candidates.  These are the exact-NURBS
    // parameter-midpoint of every compatibility panel.  Feature points and
    // compatibility endpoints are deliberately absent.
    std::vector<int> active_trace_points;
    std::vector<BenchmarkNurbsFeature2D> features;

    double target_panel_length = 0.0;
    double maximum_panel_length = 0.0;
};

const char* benchmark_nurbs_geometry_name_2d(
    BenchmarkNurbsGeometryKind2D kind) noexcept;

BenchmarkNurbsGeometryKind2D parse_benchmark_nurbs_geometry_2d(
    const std::string& name);

// Build compatibility P2 panels from a Cartesian-grid scale.  The
// authoritative NURBS curve is independent of h; h only controls the number
// of compatibility cells and trace candidates.
BenchmarkNurbsGeometry2D make_benchmark_nurbs_geometry_2d(
    BenchmarkNurbsGeometryKind2D kind,
    double grid_spacing,
    double panel_length_over_h = 1.5);

// Direct physical panel-length form, useful for geometry-only tests and for
// callers whose bulk grid has not yet been constructed.
BenchmarkNurbsGeometry2D make_benchmark_nurbs_geometry_for_panel_length_2d(
    BenchmarkNurbsGeometryKind2D kind,
    double target_panel_length);

const geometry2d::NurbsBoundaryPanelGeometry2D&
benchmark_nurbs_provider_2d(const BenchmarkNurbsGeometry2D& geometry);

} // namespace kfbim::app2d
