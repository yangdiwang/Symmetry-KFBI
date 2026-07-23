#include "native_nurbs_surface_transform_3d.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d {

namespace {

constexpr double kRigidTransformTolerance = 1.0e-12;

void validate_rigid_transform(const Eigen::Matrix3d& rotation,
                              const Eigen::Vector3d& center,
                              const Eigen::Vector3d& translation)
{
    if (!rotation.allFinite() || !center.allFinite() || !translation.allFinite()) {
        throw std::invalid_argument(
            "rigid transform rotation, center, and translation must be finite");
    }
    const Eigen::Matrix3d orthogonality_error =
        rotation.transpose() * rotation - Eigen::Matrix3d::Identity();
    if (orthogonality_error.cwiseAbs().maxCoeff() > kRigidTransformTolerance) {
        throw std::invalid_argument("rigid transform rotation must be orthogonal");
    }
    if (std::abs(rotation.determinant() - 1.0) > kRigidTransformTolerance) {
        throw std::invalid_argument(
            "rigid transform rotation must have determinant one");
    }
}

} // namespace

RigidTransform3D::RigidTransform3D()
    : rotation_(Eigen::Matrix3d::Identity())
    , center_(Eigen::Vector3d::Zero())
    , translation_(Eigen::Vector3d::Zero())
{}

RigidTransform3D::RigidTransform3D(Eigen::Matrix3d rotation,
                                   Eigen::Vector3d center,
                                   Eigen::Vector3d translation)
    : rotation_(std::move(rotation))
    , center_(std::move(center))
    , translation_(std::move(translation))
{
    validate_rigid_transform(rotation_, center_, translation_);
}

RigidTransform3D RigidTransform3D::from_axis_angle(
    const Eigen::Vector3d& axis,
    double angle_radians,
    const Eigen::Vector3d& center,
    const Eigen::Vector3d& translation)
{
    const double axis_norm = axis.norm();
    if (!axis.allFinite() || !std::isfinite(axis_norm) || axis_norm == 0.0) {
        throw std::invalid_argument("rigid transform axis must be finite and nonzero");
    }
    if (!std::isfinite(angle_radians))
        throw std::invalid_argument("rigid transform angle must be finite");
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(angle_radians, axis / axis_norm).toRotationMatrix();
    return RigidTransform3D(rotation, center, translation);
}

Eigen::Vector3d RigidTransform3D::forward_point(const Eigen::Vector3d& point) const
{
    return center_ + rotation_ * (point - center_) + translation_;
}

Eigen::Vector3d RigidTransform3D::inverse_point(const Eigen::Vector3d& point) const
{
    return center_ + rotation_.transpose() * (point - center_ - translation_);
}

Eigen::Vector3d RigidTransform3D::forward_vector(const Eigen::Vector3d& vector) const
{
    return rotation_ * vector;
}

const Eigen::Matrix3d& RigidTransform3D::rotation() const noexcept
{
    return rotation_;
}

const Eigen::Vector3d& RigidTransform3D::center() const noexcept
{
    return center_;
}

const Eigen::Vector3d& RigidTransform3D::translation() const noexcept
{
    return translation_;
}

NativeNurbsSurface3D transform_native_nurbs_surface_3d(
    const NativeNurbsSurface3D& source,
    const RigidTransform3D& transform)
{
    if (!source.exact_inside)
        throw std::invalid_argument("native NURBS surface exact-inside predicate is empty");

    NativeNurbsSurface3D result = source;
    result.patches.clear();
    result.patches.reserve(source.patches.size());
    for (const auto& source_patch : source.patches) {
        auto control_net = source_patch.control_net();
        for (auto& control_row : control_net) {
            for (Eigen::Vector3d& control_point : control_row)
                control_point = transform.forward_point(control_point);
        }
        result.patches.emplace_back(source_patch.basis_u(),
                                    source_patch.basis_v(),
                                    std::move(control_net),
                                    source_patch.weights());
    }
    const auto source_predicate = source.exact_inside;
    result.exact_inside = [source_predicate, transform](const Eigen::Vector3d& point) {
        return source_predicate(transform.inverse_point(point));
    };
    return result;
}

} // namespace kfbim::app3d