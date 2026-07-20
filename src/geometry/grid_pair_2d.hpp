#pragma once

#include <memory>
#include <vector>
#include "../grid/cartesian_grid_2d.hpp"
#include "../interface/interface_2d.hpp"
#include "p2_projection_2d.hpp"

namespace kfbim {

class GridPair2D {
public:
    GridPair2D(const CartesianGrid2D& grid, const Interface2D& interface);
    ~GridPair2D();
    GridPair2D(const GridPair2D&)            = delete;
    GridPair2D& operator=(const GridPair2D&) = delete;

    // interface point index → nearest bulk node
    int closest_bulk_node(int interface_pt_idx) const;

    // bulk node index → nearest interface point (lazy compatibility query)
    int closest_interface_point(int bulk_node_idx) const;

    // bulk node index → nearest active P2 expansion center. Available only for
    // QuadraticLagrange 3-point panels.
    int nearest_p2_expansion_center(int bulk_node_idx) const;
    int nearest_p2_expansion_center_between(int bulk_node_a,
                                            int bulk_node_b) const;

    // domain label: 0 = Ω⁻ (exterior), 1,2,... = Ω⁺ (interior) of each component
    int domain_label(int bulk_node_idx) const;

    // corner patch label: -1 = outside all active corner patches, otherwise patch id
    int corner_patch_label(int bulk_node_idx) const;
    bool is_in_corner_patch(int bulk_node_idx) const;
    double corner_patch_radial_signed_distance(int bulk_node_idx) const;

    bool is_near_interface(int bulk_node_idx, double radius) const;

    // all bulk node indices within radius of active geometry samples
    // (P2 expansion centers for QuadraticLagrange panels)
    std::vector<int> near_interface_nodes(double radius) const;

    // all interface point indices within radius of a given bulk node
    // (may span multiple components; used by Corrector to accumulate all contributions)
    std::vector<int> near_interface_points(int bulk_node_idx, double radius) const;

    NarrowBandProjection2D project_near_interface_nodes(double radius) const;
    NarrowBandProjection2D project_grid_nodes_to_interface(
        const std::vector<int>& bulk_node_indices) const;

    const CartesianGrid2D& grid()      const { return grid_; }
    const Interface2D&     interface() const { return interface_; }

private:
    const CartesianGrid2D& grid_;
    const Interface2D&     interface_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfbim
