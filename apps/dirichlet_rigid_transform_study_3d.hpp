#pragma once

#include "native_nurbs_surface_transform_3d.hpp"

#include <string>
#include <vector>

namespace kfbim::app3d {

struct DirichletRigidStudyCase3D {
    std::string id;
    RigidTransform3D transform;
    Eigen::Vector3d rotation_axis;
    double rotation_angle_degrees = 0.0;
};

std::vector<DirichletRigidStudyCase3D>
make_l_prism_dirichlet_rigid_study_cases_3d();

double manufactured_harmonic_value_3d(const Eigen::Vector3d& point);
Eigen::Vector3d manufactured_harmonic_gradient_3d(
    const Eigen::Vector3d& point);
double transformed_manufactured_harmonic_value_3d(
    const RigidTransform3D& transform,
    const Eigen::Vector3d& point);
Eigen::Vector3d transformed_manufactured_harmonic_gradient_3d(
    const RigidTransform3D& transform,
    const Eigen::Vector3d& point);

} // namespace kfbim::app3d
