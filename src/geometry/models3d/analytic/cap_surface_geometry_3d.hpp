#pragma once

#include <Eigen/Core>

namespace kfbim::geometry3d {

// Analytic images of the shared reference-sphere cap atlas.  The patch
// parameterization is independent of the density basis that may later be
// placed on the surface.
enum class AnalyticCapShape3D {
    Ellipsoid,
    Flower
};

struct AnalyticCapGeometryOptions3D {
    AnalyticCapShape3D shape = AnalyticCapShape3D::Ellipsoid;

    // Reference-sphere atlas.  The central square is [-a,a]^2 and the polar
    // ring terminates on the circle rho=polar_radius.
    double square_half_width = 0.35;
    double polar_radius = 0.72;

    Eigen::Vector3d ellipsoid_axes = Eigen::Vector3d(1.20, 0.90, 0.72);
    double flower_epsilon = 0.16;
    double flower_eta = 0.035;

    // x_world = center + rotation * (x_reference - center) + translation.
    Eigen::Matrix3d rigid_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d rigid_center = Eigen::Vector3d::Zero();
    Eigen::Vector3d rigid_translation = Eigen::Vector3d::Zero();
};

struct AnalyticCapSurfaceEvaluation3D {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 2> tangents =
        Eigen::Matrix<double, 3, 2>::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double area_element = 0.0;
};

struct AnalyticCapPatchLocation3D {
    int patch = -1;
    double u = 0.0;
    double v = 0.0;
    bool valid = false;
};

class AnalyticCapGeometry3D {
public:
    explicit AnalyticCapGeometry3D(
        AnalyticCapGeometryOptions3D options = {});

    const AnalyticCapGeometryOptions3D& options() const noexcept
    {
        return options_;
    }

    // Stable atlas ordering: north center, four north rings, four belts,
    // south center, and four south rings (14 patches total).
    int patch_count() const noexcept { return 14; }

    AnalyticCapSurfaceEvaluation3D evaluate(
        int patch, double u, double v) const;

    double level_set(const Eigen::Vector3d& point) const;
    Eigen::Vector3d level_set_gradient(const Eigen::Vector3d& point) const;
    Eigen::Vector3d outward_normal(const Eigen::Vector3d& point) const;
    bool inside(const Eigen::Vector3d& point) const;

    // The supplied direction belongs to the untransformed reference sphere.
    Eigen::Vector3d surface_point_from_direction(
        const Eigen::Vector3d& direction) const;

    AnalyticCapPatchLocation3D locate(
        const Eigen::Vector3d& point) const;

private:
    AnalyticCapGeometryOptions3D options_;

    bool has_nontrivial_rigid_map() const noexcept;
    Eigen::Vector3d forward_rigid_point(
        const Eigen::Vector3d& reference) const;
    Eigen::Vector3d inverse_rigid_point(
        const Eigen::Vector3d& world) const;
    Eigen::Vector3d forward_rigid_vector(
        const Eigen::Vector3d& reference) const;
    double flower_radius(const Eigen::Vector3d& direction) const;
    Eigen::Vector3d flower_radius_gradient(
        const Eigen::Vector3d& direction) const;
};

} // namespace kfbim::geometry3d
