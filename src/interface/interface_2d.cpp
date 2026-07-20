#include "interface_2d.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace kfbim {

namespace {

constexpr double kTwoPi = 6.28318530717958647692;

Eigen::MatrixXi make_panel_major_connectivity(int num_points, int points_per_panel)
{
    if (points_per_panel <= 0)
        throw std::invalid_argument("points_per_panel must be positive");
    if (num_points % points_per_panel != 0)
        throw std::invalid_argument("num_points must be divisible by points_per_panel");

    const int num_panels = num_points / points_per_panel;
    Eigen::MatrixXi panel_point_indices(num_panels, points_per_panel);
    for (int p = 0; p < num_panels; ++p)
        for (int q = 0; q < points_per_panel; ++q)
            panel_point_indices(p, q) = p * points_per_panel + q;
    return panel_point_indices;
}

Eigen::Vector2d normalize_or_zero(const Eigen::Vector2d& v)
{
    const double len = v.norm();
    if (len <= 1.0e-14)
        return Eigen::Vector2d::Zero();
    return v / len;
}

Eigen::Vector2d tangent_from_normal(const Eigen::Vector2d& normal)
{
    const Eigen::Vector2d n = normalize_or_zero(normal);
    if (n.squaredNorm() <= 0.0)
        return {1.0, 0.0};
    return {-n[1], n[0]};
}

void p2_shape_deriv(double s, double dN[3])
{
    dN[0] = s - 0.5;
    dN[1] = -2.0 * s;
    dN[2] = s + 0.5;
}

} // namespace

Interface2D::Interface2D(Eigen::MatrixX2d points,
                         Eigen::MatrixX2d normals,
                         Eigen::VectorXd  weights,
                         int              points_per_panel,
                         Eigen::VectorXi  panel_components,
                         PanelNodeLayout2D panel_node_layout)
    : points_(std::move(points))
    , normals_(std::move(normals))
    , weights_(std::move(weights))
    , points_per_panel_(points_per_panel)
    , panel_point_indices_(make_panel_major_connectivity(static_cast<int>(points_.rows()),
                                                         points_per_panel))
    , panel_components_(std::move(panel_components))
    , panel_node_layout_(panel_node_layout)
{
    validate_topology();
    initialize_default_panel_side_geometry();
    initialize_default_corner_metadata();
    validate();
}

Interface2D::Interface2D(Eigen::MatrixX2d points,
                         Eigen::MatrixX2d normals,
                         Eigen::VectorXd  weights,
                         int              points_per_panel,
                         Eigen::MatrixXi  panel_point_indices,
                         Eigen::VectorXi  panel_components,
                         PanelNodeLayout2D panel_node_layout)
    : points_(std::move(points))
    , normals_(std::move(normals))
    , weights_(std::move(weights))
    , points_per_panel_(points_per_panel)
    , panel_point_indices_(std::move(panel_point_indices))
    , panel_components_(std::move(panel_components))
    , panel_node_layout_(panel_node_layout)
{
    validate_topology();
    initialize_default_panel_side_geometry();
    initialize_default_corner_metadata();
    validate();
}

Interface2D::Interface2D(Eigen::MatrixX2d points,
                         Eigen::MatrixX2d normals,
                         Eigen::VectorXd  weights,
                         int              points_per_panel,
                         Eigen::MatrixXi  panel_point_indices,
                         Eigen::VectorXi  panel_components,
                         PanelSideGeometry2D panel_side_geometry,
                         std::vector<InterfacePointKind2D> point_kind,
                         std::vector<int> corner_index_by_point,
                         std::vector<CornerData2D> corners,
                         PanelNodeLayout2D panel_node_layout,
                         std::vector<CornerPatch2D> corner_patches)
    : points_(std::move(points))
    , normals_(std::move(normals))
    , weights_(std::move(weights))
    , points_per_panel_(points_per_panel)
    , panel_point_indices_(std::move(panel_point_indices))
    , panel_components_(std::move(panel_components))
    , panel_normals_(std::move(panel_side_geometry.point_normals))
    , panel_tangents_(std::move(panel_side_geometry.point_tangents))
    , point_kind_(std::move(point_kind))
    , corner_index_by_point_(std::move(corner_index_by_point))
    , corners_(std::move(corners))
    , corner_patches_(std::move(corner_patches))
    , panel_node_layout_(panel_node_layout)
{
    validate();
}

void Interface2D::initialize_default_panel_side_geometry()
{
    const int n_panel_rows = static_cast<int>(panel_point_indices_.rows())
                             * points_per_panel_;
    panel_normals_.resize(n_panel_rows, 2);
    panel_tangents_.resize(n_panel_rows, 2);

    for (int p = 0; p < panel_point_indices_.rows(); ++p) {
        for (int q = 0; q < points_per_panel_; ++q) {
            const int point = panel_point_indices_(p, q);
            const int row = panel_local_row(p, q);
            panel_normals_.row(row) = normals_.row(point);

            Eigen::Vector2d tangent = Eigen::Vector2d::Zero();
            if (panel_node_layout_ == PanelNodeLayout2D::QuadraticLagrange
                && points_per_panel_ == 3) {
                constexpr double s_nodes[3] = {-1.0, 0.0, 1.0};
                double dN[3];
                p2_shape_deriv(s_nodes[q], dN);
                for (int a = 0; a < 3; ++a) {
                    const int point_a = panel_point_indices_(p, a);
                    tangent += dN[a] * points_.row(point_a).transpose();
                }
            } else if (points_per_panel_ > 1) {
                const int left_q = q == 0 ? q : q - 1;
                const int right_q = q == points_per_panel_ - 1 ? q : q + 1;
                const int left = panel_point_indices_(p, left_q);
                const int right = panel_point_indices_(p, right_q);
                tangent = points_.row(right).transpose()
                          - points_.row(left).transpose();
            }

            tangent = normalize_or_zero(tangent);
            if (tangent.squaredNorm() <= 0.0)
                tangent = tangent_from_normal(normals_.row(point).transpose());
            panel_tangents_.row(row) = tangent.transpose();
        }
    }
}

void Interface2D::initialize_default_corner_metadata()
{
    point_kind_.assign(static_cast<std::size_t>(points_.rows()),
                       InterfacePointKind2D::Smooth);
    corner_index_by_point_.assign(static_cast<std::size_t>(points_.rows()), -1);
    corners_.clear();
}

void Interface2D::validate_topology() const
{
    if (points_per_panel_ <= 0)
        throw std::invalid_argument("points_per_panel must be positive");
    if (normals_.rows() != points_.rows())
        throw std::invalid_argument("normals row count must equal points row count");
    if (points_.cols() != 2)
        throw std::invalid_argument("points must have exactly two columns");
    if (normals_.cols() != 2)
        throw std::invalid_argument("normals must have exactly two columns");
    if (weights_.size() != points_.rows())
        throw std::invalid_argument("weights size must equal num_points");
    if (panel_point_indices_.cols() != points_per_panel_)
        throw std::invalid_argument("panel_point_indices column count must equal points_per_panel");
    if (panel_point_indices_.rows() <= 0)
        throw std::invalid_argument("panel_point_indices must contain at least one panel");
    for (int p = 0; p < panel_point_indices_.rows(); ++p) {
        for (int q = 0; q < panel_point_indices_.cols(); ++q) {
            const int idx = panel_point_indices_(p, q);
            if (idx < 0 || idx >= points_.rows())
                throw std::invalid_argument("panel_point_indices contains an out-of-range point index");
        }
    }
    if (panel_components_.size() != num_panels())
        throw std::invalid_argument("panel_components size must equal num_panels");
}

void Interface2D::validate() const
{
    validate_topology();

    const int n_panel_rows = num_panels() * points_per_panel_;
    if (panel_normals_.rows() != n_panel_rows)
        throw std::invalid_argument("panel_normals row count must equal num_panels * points_per_panel");
    if (panel_normals_.cols() != 2)
        throw std::invalid_argument("panel_normals must have exactly two columns");
    if (panel_tangents_.rows() != n_panel_rows)
        throw std::invalid_argument("panel_tangents row count must equal num_panels * points_per_panel");
    if (panel_tangents_.cols() != 2)
        throw std::invalid_argument("panel_tangents must have exactly two columns");
    if (point_kind_.size() != static_cast<std::size_t>(num_points()))
        throw std::invalid_argument("point_kind size must equal num_points");
    if (corner_index_by_point_.size() != static_cast<std::size_t>(num_points()))
        throw std::invalid_argument("corner_index_by_point size must equal num_points");

    for (int q = 0; q < num_points(); ++q) {
        const int corner_id = corner_index_by_point_[static_cast<std::size_t>(q)];
        if (point_kind_[static_cast<std::size_t>(q)] == InterfacePointKind2D::Smooth) {
            if (corner_id != -1)
                throw std::invalid_argument("smooth points must have corner_index_by_point == -1");
        } else {
            if (corner_id < 0 || corner_id >= static_cast<int>(corners_.size()))
                throw std::invalid_argument("corner point has an invalid corner_index_by_point");
        }
    }

    for (int c = 0; c < static_cast<int>(corners_.size()); ++c) {
        const CornerData2D& corner = corners_[static_cast<std::size_t>(c)];
        if (corner.point < 0 || corner.point >= num_points())
            throw std::invalid_argument("corner point is out of range");
        if (corner.prev_panel < 0 || corner.prev_panel >= num_panels())
            throw std::invalid_argument("corner prev_panel is out of range");
        if (corner.next_panel < 0 || corner.next_panel >= num_panels())
            throw std::invalid_argument("corner next_panel is out of range");
        if (point_kind_[static_cast<std::size_t>(corner.point)]
            != InterfacePointKind2D::Corner)
            throw std::invalid_argument("corner data references a non-corner point");
        if (corner_index_by_point_[static_cast<std::size_t>(corner.point)] != c)
            throw std::invalid_argument("corner_index_by_point does not match corner list order");
    }

    for (const CornerPatch2D& patch : corner_patches_) {
        if (patch.corner < 0 || patch.corner >= static_cast<int>(corners_.size()))
            throw std::invalid_argument("corner patch references an invalid corner");
        if (patch.point < 0 || patch.point >= num_points())
            throw std::invalid_argument("corner patch point is out of range");
        if (patch.point != corners_[static_cast<std::size_t>(patch.corner)].point)
            throw std::invalid_argument("corner patch point does not match its corner");
        if (patch.shape == CornerPatchShape2D::Circle) {
            if (!(patch.opening_angle > 0.0
                  && patch.opening_angle <= kTwoPi + 1.0e-12)) {
                throw std::invalid_argument(
                    "circle corner patch opening_angle must be in (0, 2*pi]");
            }
        } else if (!(patch.opening_angle > 0.0
                     && patch.opening_angle < kTwoPi)) {
            throw std::invalid_argument(
                "corner patch opening_angle must be in (0, 2*pi)");
        }
        if (!(patch.radius > 0.0))
            throw std::invalid_argument("corner patch radius must be positive");
        if (patch.ray_minus.norm() <= 1.0e-14 || patch.ray_plus.norm() <= 1.0e-14)
            throw std::invalid_argument("corner patch rays must be nonzero");
    }
}

} // namespace kfbim
