#pragma once

#include <array>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "nurbs_curve_2d.hpp"
#include "panel_geometry_2d.hpp"

namespace kfbim::geometry2d {

// Exact one-sided differential geometry on a NURBS smooth branch. A branch
// may contain several ordinary smooth knot spans; a repeated knot that is a
// physical corner separates two branches. The branch is part of the identity
// because a corner has two valid one-sided tangent/normal limits at the same
// global parameter.
struct NurbsBoundaryGeometry2D {
    int span = -1;
    double parameter = 0.0;
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d parameter_tangent = Eigen::Vector2d::Zero();
    Eigen::Vector2d parameter_second = Eigen::Vector2d::Zero();
    Eigen::Vector2d tangent = Eigen::Vector2d::UnitX();
    Eigen::Vector2d normal = Eigen::Vector2d::UnitY();
    double speed = 0.0;
    double curvature = 0.0;
};

struct NurbsSpanProjection2D {
    int span = -1;
    double parameter = 0.0;
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    double distance = 0.0;
    bool converged = false;
};

// Authoritative NURBS geometry with an exact-geometry adapter for the legacy
// P2 computational cells.  panel_parameters merely tell a compatibility cell
// which subinterval of the NURBS curve it covers; all point, tangent, normal,
// curvature and crossing projection queries are evaluated on the NURBS curve.
// Density reconstruction independently uses point_parameters and the full
// smooth-branch intervals, including short corner intervals not covered by a
// compatibility panel. For a smooth closed curve, pass one branch covering
// the whole parameter domain; NSP-CJ then constructs periodic P3/P2 spaces.
class NurbsBoundaryPanelGeometry2D final : public IPanelGeometry2D {
public:
    NurbsBoundaryPanelGeometry2D(
        NurbsCurve2D curve,
        std::vector<double> span_breaks,
        std::vector<int> panel_spans,
        std::vector<std::array<double, 2>> panel_parameters,
        std::vector<int> point_spans,
        std::vector<double> point_parameters,
        bool outward_normal_is_right = true,
        bool closed = true);

    int num_panels() const override;
    Eigen::Vector2d point(int panel, double local_s) const override;
    Eigen::Vector2d tangent(int panel, double local_s) const override;
    Eigen::Vector2d second_derivative(
        int panel,
        double local_s) const override;
    Eigen::Vector2d normal(int panel, double local_s) const override;

    const NurbsCurve2D& curve() const noexcept { return curve_; }
    int num_spans() const noexcept
    {
        return static_cast<int>(span_breaks_.size()) - 1;
    }
    int num_parameterized_points() const noexcept
    {
        return static_cast<int>(point_parameters_.size());
    }
    bool closed() const noexcept { return closed_; }

    std::pair<double, double> span_interval(int span) const;
    int panel_span(int panel) const;
    double panel_parameter(int panel, double local_s) const;
    int point_span(int point) const;
    double point_parameter(int point) const;

    NurbsBoundaryGeometry2D evaluate_on_span(
        int span,
        double parameter) const;
    NurbsSpanProjection2D project_to_span(
        int span,
        const Eigen::Vector2d& query) const;

private:
    NurbsCurve2D curve_;
    std::vector<double> span_breaks_;
    std::vector<int> panel_spans_;
    std::vector<std::array<double, 2>> panel_parameters_;
    std::vector<int> point_spans_;
    std::vector<double> point_parameters_;
    bool outward_normal_is_right_ = true;
    bool closed_ = true;

    void validate() const;
    double one_sided_parameter(int span, double parameter) const;
};

} // namespace kfbim::geometry2d
