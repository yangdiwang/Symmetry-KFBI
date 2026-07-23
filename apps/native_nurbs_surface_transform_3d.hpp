#pragma once

#include "native_nurbs_surface_3d.hpp"

#include <Eigen/Dense>

namespace kfbim::app3d {

class RigidTransform3D {
public:
    RigidTransform3D();
    RigidTransform3D(Eigen::Matrix3d rotation,
                     Eigen::Vector3d center,
                     Eigen::Vector3d translation);

    static RigidTransform3D from_axis_angle(
        const Eigen::Vector3d& axis,
        double angle_radians,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& translation);

    Eigen::Vector3d forward_point(const Eigen::Vector3d& point) const;
    Eigen::Vector3d inverse_point(const Eigen::Vector3d& point) const;
    Eigen::Vector3d forward_vector(const Eigen::Vector3d& vector) const;

    const Eigen::Matrix3d& rotation() const noexcept;
    const Eigen::Vector3d& center() const noexcept;
    const Eigen::Vector3d& translation() const noexcept;

private:
    Eigen::Matrix3d rotation_;
    Eigen::Vector3d center_;
    Eigen::Vector3d translation_;
};

NativeNurbsSurface3D transform_native_nurbs_surface_3d(
    const NativeNurbsSurface3D& source,
    const RigidTransform3D& transform);

} // namespace kfbim::app3d