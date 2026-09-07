#pragma once

#include "nurbs_surface_model_3d.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace kfbim::geometry3d {

enum class NurbsSurfaceClosestPointStatus3D { Bounded, BudgetExceeded, Unsupported };

struct NurbsSurfaceClosestPointOptions3D {
    // Zero selects a model-scale default: 1e-9 for distance, 1e-7 for location.
    double distance_tolerance = 0.0;
    double localization_tolerance = 0.0;
    std::size_t max_boxes = 2048;
    int max_subdivision_depth = 64;
    int max_local_iterations = 24;
    double tie_distance_tolerance = 0.0;
    double separation_tolerance = 0.0;
};

struct NurbsSurfaceClosestPointCandidate3D {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    bool normal_valid = false;
    int patch_index = -1;
    int component = -1;
    double u = 0.0;
    double v = 0.0;
    double distance_upper = std::numeric_limits<double>::infinity();
};

struct NurbsSurfaceClosestPointStatistics3D {
    std::size_t boxes_visited = 0;
    std::size_t subdivisions = 0;
    std::size_t local_iterations = 0;
    std::size_t point_evaluations = 0;
};

// Bounded certifies only the global distance gap. Neither Bounded nor localized
// proves mathematical uniqueness. Localized encloses all remaining possible
// minimizers within localization_radius_upper of the returned representative.
struct NurbsSurfaceClosestPointResult3D : NurbsSurfaceClosestPointCandidate3D {
    NurbsSurfaceClosestPointStatus3D status = NurbsSurfaceClosestPointStatus3D::Unsupported;
    double distance_lower = 0.0;
    double localization_radius_upper = std::numeric_limits<double>::infinity();
    bool localized = false;
    // Observed separated near-equal candidates; false does not prove uniqueness.
    bool separated_near_ties = false;
    std::vector<NurbsSurfaceClosestPointCandidate3D> candidates;
    NurbsSurfaceClosestPointStatistics3D stats;
};

class NurbsSurfaceClosestPointWorkspace3D {
public:
    NurbsSurfaceClosestPointWorkspace3D();
    ~NurbsSurfaceClosestPointWorkspace3D();
    NurbsSurfaceClosestPointWorkspace3D(NurbsSurfaceClosestPointWorkspace3D&&) noexcept;
    NurbsSurfaceClosestPointWorkspace3D& operator=(NurbsSurfaceClosestPointWorkspace3D&&) noexcept;
    NurbsSurfaceClosestPointWorkspace3D(const NurbsSurfaceClosestPointWorkspace3D&) = delete;
    NurbsSurfaceClosestPointWorkspace3D& operator=(const NurbsSurfaceClosestPointWorkspace3D&) = delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class NurbsSurfaceClosestPointIndex3D;
};

// Owns an immutable model, original homogeneous base nets and a conservative
// BVH. Exact extraction supports positive-weight clamped native NURBS patches.
// Unsupported representations return Unsupported, not a local-only success.
class NurbsSurfaceClosestPointIndex3D {
public:
    explicit NurbsSurfaceClosestPointIndex3D(const NurbsSurfaceModel3D& model);
    NurbsSurfaceClosestPointResult3D query(
        const Eigen::Vector3d& point,
        const NurbsSurfaceClosestPointOptions3D& options,
        NurbsSurfaceClosestPointWorkspace3D& workspace) const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace kfbim::geometry3d
