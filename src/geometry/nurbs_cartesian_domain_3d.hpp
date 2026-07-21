#pragma once

#include "nurbs_surface_intersector_3d.hpp"
#include "../grid/cartesian_grid_3d.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace kfbim::geometry3d {

struct NurbsCartesianDomainOptions3D {
    bool use_triangle_seeds = true;
};

struct NurbsCartesianDomainDiagnostics3D {
    int nurbs_patch_count = 0;
    int bezier_element_count = 0;
    int acceleration_leaf_count = 0;
    std::size_t candidate_grid_edge_count = 0;
    std::array<std::size_t, 3> barrier_edge_counts{{0, 0, 0}};
    int grid_component_count = 0;
    int box_exterior_component_count = 0;
    int representative_query_count = 0;
    std::vector<std::size_t> component_sizes;
    NurbsSurfaceIntersectionDiagnostics3D intersections;
    double maximum_root_residual = 0.0;
};

class NurbsCartesianDomain3D {
public:
    NurbsCartesianDomain3D(
        const CartesianGrid3D& grid,
        NurbsSurfaceModel3D model,
        NurbsCartesianDomainOptions3D options = {});

    int label(int node) const;
    bool has_barrier_between(int node_a, int node_b) const;
    const NurbsSurfaceCrossing3D& crossing_between(
        int node_a, int node_b) const;
    const std::vector<int>& labels() const;
    const NurbsCartesianDomainDiagnostics3D& diagnostics() const;
    const NurbsAabb3D& surface_bounds() const;
    double geometry_tolerance() const;
    bool is_compatible_grid(const CartesianGrid3D& grid) const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace kfbim::geometry3d
