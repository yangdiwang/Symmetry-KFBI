#include "dirichlet_rigid_transform_study_3d.hpp"

#include <cmath>
#include <utility>

namespace kfbim::app3d {

namespace {

constexpr double kDegreesToRadians =
    3.141592653589793238462643383279502884 / 180.0;

const Eigen::Vector3d kRotationAxis =
    Eigen::Vector3d(1.0, 2.0, 3.0).normalized();
const Eigen::Vector3d kRotationCenter(0.07, -0.07, 0.02);
const Eigen::Vector3d kTranslationX(0.137, 0.0, 0.0);
const Eigen::Vector3d kTranslationY(0.0, -0.083, 0.0);
const Eigen::Vector3d kTranslationZ(0.0, 0.0, 0.061);
const Eigen::Vector3d kTranslationXYZ1(0.137, -0.083, 0.061);
const Eigen::Vector3d kTranslationXYZ2(-0.109, 0.151, -0.047);

DirichletRigidStudyCase3D make_translation_case(
    std::string id,
    const Eigen::Vector3d& translation)
{
    return {std::move(id),
            RigidTransform3D(Eigen::Matrix3d::Identity(), kRotationCenter,
                             translation),
            Eigen::Vector3d::Zero(),
            0.0};
}

DirichletRigidStudyCase3D make_rotation_case(
    std::string id,
    const Eigen::Vector3d& translation)
{
    return {std::move(id),
            RigidTransform3D::from_axis_angle(
                kRotationAxis, 17.0 * kDegreesToRadians, kRotationCenter,
                translation),
            kRotationAxis,
            17.0};
}

} // namespace

std::vector<DirichletRigidStudyCase3D>
make_l_prism_dirichlet_rigid_study_cases_3d()
{
    return {
        make_translation_case("baseline", Eigen::Vector3d::Zero()),
        make_translation_case("tx_p0137", kTranslationX),
        make_translation_case("ty_m0083", kTranslationY),
        make_translation_case("tz_p0061", kTranslationZ),
        make_translation_case("t_xyz_1", kTranslationXYZ1),
        make_translation_case("t_xyz_2", kTranslationXYZ2),
        make_rotation_case("rot_axis123_17deg", Eigen::Vector3d::Zero()),
        make_rotation_case("rot_axis123_17deg_t_xyz_1", kTranslationXYZ1),
    };
}

double manufactured_harmonic_value_3d(const Eigen::Vector3d& point)
{
    return std::exp(0.35 * point.x()) * std::cos(0.21 * point.y()) *
           std::cos(0.28 * point.z());
}

Eigen::Vector3d manufactured_harmonic_gradient_3d(const Eigen::Vector3d& point)
{
    const double exponential = std::exp(0.35 * point.x());
    const double cos_y = std::cos(0.21 * point.y());
    const double cos_z = std::cos(0.28 * point.z());
    return {0.35 * exponential * cos_y * cos_z,
            -0.21 * exponential * std::sin(0.21 * point.y()) * cos_z,
            -0.28 * exponential * cos_y * std::sin(0.28 * point.z())};
}

double transformed_manufactured_harmonic_value_3d(
    const RigidTransform3D& transform,
    const Eigen::Vector3d& point)
{
    return manufactured_harmonic_value_3d(transform.inverse_point(point));
}

Eigen::Vector3d transformed_manufactured_harmonic_gradient_3d(
    const RigidTransform3D& transform,
    const Eigen::Vector3d& point)
{
    const Eigen::Vector3d source = transform.inverse_point(point);
    return transform.forward_vector(manufactured_harmonic_gradient_3d(source));
}

} // namespace kfbim::app3d
