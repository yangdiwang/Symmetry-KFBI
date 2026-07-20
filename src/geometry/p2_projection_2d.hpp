#pragma once

#include <vector>

#include <Eigen/Dense>

namespace kfbim {

class GridPair2D;

struct CurveProjection2D {
    int             grid_node = -1;
    int             panel = -1;
    int             component = -1;
    double          local_s = 0.0;
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    double          signed_distance = 0.0;
    double          distance = 0.0;
    double          tangential_residual = 0.0;
    int             iterations = 0;
    bool            converged = false;
};

class NarrowBandProjection2D {
public:
    double radius() const { return radius_; }
    const std::vector<int>& nodes() const { return nodes_; }
    const std::vector<CurveProjection2D>& projections() const { return projections_; }

    bool has_projection(int bulk_node_idx) const;
    const CurveProjection2D& projection(int bulk_node_idx) const;

private:
    friend NarrowBandProjection2D project_p2_near_interface_nodes_2d(
        const GridPair2D& grid_pair,
        double radius);
    friend NarrowBandProjection2D project_p2_grid_nodes_to_interface_2d(
        const GridPair2D& grid_pair,
        const std::vector<int>& bulk_node_indices);

    double radius_ = 0.0;
    std::vector<int> nodes_;
    std::vector<CurveProjection2D> projections_;
    std::vector<int> projection_index_by_grid_node_;
};

NarrowBandProjection2D project_p2_near_interface_nodes_2d(
    const GridPair2D& grid_pair,
    double radius);

NarrowBandProjection2D project_p2_grid_nodes_to_interface_2d(
    const GridPair2D& grid_pair,
    const std::vector<int>& bulk_node_indices);

} // namespace kfbim
