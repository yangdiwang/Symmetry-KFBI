#pragma once

#include <vector>

#include <Eigen/Dense>

namespace kfbim {

class GridPair3D;

struct SurfaceProjection3D {
    int             grid_node = -1;
    int             panel = -1;
    int             component = -1;
    Eigen::Vector3d barycentric = Eigen::Vector3d::Zero();
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double          signed_distance = 0.0;
    double          distance = 0.0;
    double          tangential_residual = 0.0;
    int             iterations = 0;
    bool            converged = false;
};

class NarrowBandProjection3D {
public:
    double radius() const { return radius_; }
    const std::vector<int>& nodes() const { return nodes_; }
    const std::vector<SurfaceProjection3D>& projections() const { return projections_; }

    bool has_projection(int bulk_node_idx) const;
    const SurfaceProjection3D& projection(int bulk_node_idx) const;

private:
    friend NarrowBandProjection3D project_p2_near_interface_nodes_3d(
        const GridPair3D& grid_pair,
        double radius);
    friend NarrowBandProjection3D project_p2_grid_nodes_to_interface_3d(
        const GridPair3D& grid_pair,
        const std::vector<int>& bulk_node_indices);

    double radius_ = 0.0;
    std::vector<int> nodes_;
    std::vector<SurfaceProjection3D> projections_;
    std::vector<int> projection_index_by_grid_node_;
};

NarrowBandProjection3D project_p2_near_interface_nodes_3d(
    const GridPair3D& grid_pair,
    double radius);

NarrowBandProjection3D project_p2_grid_nodes_to_interface_3d(
    const GridPair3D& grid_pair,
    const std::vector<int>& bulk_node_indices);

} // namespace kfbim
