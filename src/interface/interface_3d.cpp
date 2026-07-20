#include "interface_3d.hpp"
#include <stdexcept>
#include <utility>

namespace kfbim {

namespace {

Eigen::MatrixXi make_panel_major_connectivity(int num_points,
                                              int points_per_panel,
                                              int num_panels)
{
    if (points_per_panel <= 0)
        throw std::invalid_argument("points_per_panel must be positive");
    if (num_points % points_per_panel != 0)
        throw std::invalid_argument("num_points must be divisible by points_per_panel");
    if (num_points / points_per_panel != num_panels)
        throw std::invalid_argument(
            "panel-major Interface3D requires num_points / points_per_panel == num_panels");

    Eigen::MatrixXi panel_point_indices(num_panels, points_per_panel);
    for (int p = 0; p < num_panels; ++p)
        for (int q = 0; q < points_per_panel; ++q)
            panel_point_indices(p, q) = p * points_per_panel + q;
    return panel_point_indices;
}

void validate_common(const Eigen::MatrixX3d& points,
                     const Eigen::MatrixX3d& normals,
                     const Eigen::VectorXd&  weights,
                     int                     points_per_panel,
                     const Eigen::MatrixXi&  panel_point_indices,
                     const Eigen::VectorXi&  panel_components,
                     int                     num_panels,
                     PanelNodeLayout3D       panel_node_layout)
{
    if (points_per_panel <= 0)
        throw std::invalid_argument("points_per_panel must be positive");
    if (normals.rows() != points.rows())
        throw std::invalid_argument("normals row count must equal points row count");
    if (weights.size() != points.rows())
        throw std::invalid_argument("weights size must equal num_points");
    if (panel_point_indices.rows() != num_panels)
        throw std::invalid_argument("panel_point_indices row count must equal num_panels");
    if (panel_point_indices.cols() != points_per_panel)
        throw std::invalid_argument(
            "panel_point_indices column count must equal points_per_panel");
    if (panel_components.size() != num_panels)
        throw std::invalid_argument("panel_components size must equal num_panels");
    for (int p = 0; p < panel_point_indices.rows(); ++p) {
        for (int q = 0; q < panel_point_indices.cols(); ++q) {
            const int idx = panel_point_indices(p, q);
            if (idx < 0 || idx >= points.rows())
                throw std::invalid_argument(
                    "panel_point_indices contains an out-of-range point index");
        }
    }
    if (panel_node_layout == PanelNodeLayout3D::QuadraticLagrange
        && points_per_panel != 6) {
        throw std::invalid_argument(
            "QuadraticLagrange Interface3D requires 6 points per panel");
    }
}

} // namespace

Interface3D::Interface3D(Eigen::MatrixX3d vertices,
                         Eigen::MatrixX3i panels,
                         Eigen::MatrixX3d points,
                         Eigen::MatrixX3d normals,
                         Eigen::VectorXd  weights,
                         int              points_per_panel,
                         Eigen::VectorXi  panel_components,
                         PanelNodeLayout3D panel_node_layout)
    : vertices_(std::move(vertices))
    , panels_(std::move(panels))
    , points_(std::move(points))
    , normals_(std::move(normals))
    , weights_(std::move(weights))
    , points_per_panel_(points_per_panel)
    , panel_point_indices_(make_panel_major_connectivity(
          static_cast<int>(points_.rows()),
          points_per_panel,
          static_cast<int>(panels_.rows())))
    , panel_components_(std::move(panel_components))
    , panel_smooth_groups_(panel_components_)
    , panel_node_layout_(panel_node_layout)
{
    validate_common(points_,
                    normals_,
                    weights_,
                    points_per_panel_,
                    panel_point_indices_,
                    panel_components_,
                    num_panels(),
                    panel_node_layout_);
}

Interface3D::Interface3D(Eigen::MatrixX3d vertices,
                         Eigen::MatrixX3i panels,
                         Eigen::MatrixX3d points,
                         Eigen::MatrixX3d normals,
                         Eigen::VectorXd  weights,
                         int              points_per_panel,
                         Eigen::MatrixXi  panel_point_indices,
                         Eigen::VectorXi  panel_components,
                         PanelNodeLayout3D panel_node_layout)
    : vertices_(std::move(vertices))
    , panels_(std::move(panels))
    , points_(std::move(points))
    , normals_(std::move(normals))
    , weights_(std::move(weights))
    , points_per_panel_(points_per_panel)
    , panel_point_indices_(std::move(panel_point_indices))
    , panel_components_(std::move(panel_components))
    , panel_smooth_groups_(panel_components_)
    , panel_node_layout_(panel_node_layout)
{
    validate_common(points_,
                    normals_,
                    weights_,
                    points_per_panel_,
                    panel_point_indices_,
                    panel_components_,
                    num_panels(),
                    panel_node_layout_);
}

Interface3D::Interface3D(Eigen::MatrixX3d vertices,
                         Eigen::MatrixX3i panels,
                         Eigen::MatrixX3d points,
                         Eigen::MatrixX3d normals,
                         Eigen::VectorXd  weights,
                         int              points_per_panel,
                         Eigen::MatrixXi  panel_point_indices,
                         Eigen::VectorXi  panel_components,
                         Eigen::VectorXi  panel_smooth_groups,
                         PanelNodeLayout3D panel_node_layout)
    : vertices_(std::move(vertices))
    , panels_(std::move(panels))
    , points_(std::move(points))
    , normals_(std::move(normals))
    , weights_(std::move(weights))
    , points_per_panel_(points_per_panel)
    , panel_point_indices_(std::move(panel_point_indices))
    , panel_components_(std::move(panel_components))
    , panel_smooth_groups_(std::move(panel_smooth_groups))
    , panel_node_layout_(panel_node_layout)
{
    validate_common(points_,
                    normals_,
                    weights_,
                    points_per_panel_,
                    panel_point_indices_,
                    panel_components_,
                    num_panels(),
                    panel_node_layout_);
    if (panel_smooth_groups_.size() != num_panels())
        throw std::invalid_argument(
            "panel_smooth_groups size must equal num_panels");
}

} // namespace kfbim
