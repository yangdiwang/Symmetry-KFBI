#pragma once

#include "nurbs_surface_intersector_3d.hpp"
#include "../grid/cartesian_grid_3d.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace kfbim::geometry3d {

class NurbsCartesianDomain3D;

class NurbsSurfaceCrossingRange3D {
public:
    NurbsSurfaceCrossingRange3D() noexcept = default;

    const NurbsSurfaceCrossing3D* begin() const noexcept;
    const NurbsSurfaceCrossing3D* end() const noexcept;
    std::size_t size() const noexcept;
    const NurbsSurfaceCrossing3D& operator[](std::size_t index) const;

private:
    friend class NurbsCartesianDomain3D;
    NurbsSurfaceCrossingRange3D(
        const NurbsSurfaceCrossing3D* data, std::size_t size) noexcept;

    const NurbsSurfaceCrossing3D* data_ = nullptr;
    std::size_t size_ = 0;
};

struct NurbsCartesianDomainOptions3D {
    bool use_triangle_seeds = true;
};
struct NurbsCartesianEdgeClassification3D {
    bool queried = false;
    bool has_confirmed_interface = false;
    bool changes_component_membership = false;
    bool root_count_known = true;
    bool parity_known_from_roots = true;
    bool has_near_tangent_candidate = false;
    bool used_targeted_retry = false;
    bool correction_safe = false;
    std::size_t confirmed_crossing_count = 0;
    std::size_t ambiguous_cluster_count = 0;
    int confirmed_transverse_count = 0;
};

struct NurbsCartesianDomainDiagnostics3D {
    int nurbs_patch_count = 0;
    int bezier_element_count = 0;
    int acceleration_leaf_count = 0;
    std::size_t candidate_grid_edge_count = 0;
    double maximum_query_element_extent = 0.0;
    std::array<std::size_t, 3> barrier_edge_counts{{0, 0, 0}};
    std::array<std::size_t, 3> interface_edge_counts{{0, 0, 0}};
    std::size_t multi_crossing_edge_count = 0;
    std::size_t even_parity_interface_edge_count = 0;
    std::size_t odd_parity_interface_edge_count = 0;
    std::size_t ambiguous_parity_edge_count = 0;
    std::size_t endpoint_parity_fallback_count = 0;
    std::size_t endpoint_classification_query_count = 0;
    std::size_t component_parity_toggle_count = 0;
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
    bool has_interface_between(int node_a, int node_b) const;
    NurbsSurfaceCrossingRange3D crossings_between(
        int node_a, int node_b) const;
    const NurbsSurfaceCrossing3D& crossing_between(
        int node_a, int node_b) const;
    NurbsCartesianEdgeClassification3D edge_classification_between(
        int node_a, int node_b) const;
    const NurbsSurfaceCrossing3D& correction_crossing_between(
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
