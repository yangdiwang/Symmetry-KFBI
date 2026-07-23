#include "dirichlet_rigid_transform_study_3d.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kfbim::app3d::DirichletRigidStudyCase3D;
using kfbim::app3d::RigidStudyCriterionStatus3D;
using kfbim::app3d::combine_rigid_study_criteria_3d;
using kfbim::app3d::make_l_prism_dirichlet_rigid_study_cases_3d;
using kfbim::app3d::manufactured_harmonic_gradient_3d;
using kfbim::app3d::manufactured_harmonic_value_3d;
using kfbim::app3d::transformed_manufactured_harmonic_gradient_3d;
using kfbim::app3d::transformed_manufactured_harmonic_value_3d;

constexpr double kDegreesToRadians =
    3.141592653589793238462643383279502884 / 180.0;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_near(const Eigen::Vector3d& actual,
                  const Eigen::Vector3d& expected,
                  double tolerance,
                  const std::string& message)
{
    require((actual - expected).norm() <= tolerance, message);
}

void test_catalog_metadata()
{
    const std::vector<DirichletRigidStudyCase3D> cases =
        make_l_prism_dirichlet_rigid_study_cases_3d();
    const std::vector<std::string> expected_ids = {
        "baseline", "tx_p0137", "ty_m0083", "tz_p0061", "t_xyz_1",
        "t_xyz_2", "rot_axis123_17deg", "rot_axis123_17deg_t_xyz_1"};
    require(cases.size() == expected_ids.size(), "study case count");

    const Eigen::Vector3d center(0.07, -0.07, 0.02);
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, 3.0).normalized();
    const std::vector<Eigen::Vector3d> expected_translations = {
        {0.0, 0.0, 0.0}, {0.137, 0.0, 0.0}, {0.0, -0.083, 0.0},
        {0.0, 0.0, 0.061}, {0.137, -0.083, 0.061},
        {-0.109, 0.151, -0.047}, {0.0, 0.0, 0.0}, {0.137, -0.083, 0.061}};

    for (std::size_t index = 0; index < cases.size(); ++index) {
        const auto& item = cases[index];
        require(item.id == expected_ids[index], "study case ID");
        require_near(item.transform.center(), center, 0.0, "rotation center");
        require_near(item.transform.translation(), expected_translations[index], 0.0,
                     "study translation");
        if (index < 6) {
            require_near(item.rotation_axis, Eigen::Vector3d::Zero(), 0.0,
                         "identity rotation axis");
            require(item.rotation_angle_degrees == 0.0, "identity rotation angle");
        } else {
            require_near(item.rotation_axis, axis, 2.0e-14, "normalized rotation axis");
            require(std::abs(item.rotation_angle_degrees - 17.0) < 2.0e-14,
                    "rotation angle");
            require_near(item.transform.rotation() * axis, axis, 2.0e-14,
                         "rotation axis is preserved");
            require(std::abs(item.transform.rotation().trace() -
                             (1.0 + 2.0 * std::cos(17.0 * kDegreesToRadians))) < 2.0e-14,
                    "rotation angle matches transform");
        }
    }
}

void test_transformed_harmonic_data_are_covariant()
{
    const std::vector<DirichletRigidStudyCase3D> cases =
        make_l_prism_dirichlet_rigid_study_cases_3d();
    const std::vector<Eigen::Vector3d> sample_points = {
        {0.0, 0.0, 0.0}, {0.13, -0.29, 0.41}, {-0.37, 0.23, -0.19}};

    for (const auto& item : cases) {
        for (const Eigen::Vector3d& point : sample_points) {
            const Eigen::Vector3d moved = item.transform.forward_point(point);
            require(std::abs(transformed_manufactured_harmonic_value_3d(
                                 item.transform, moved) -
                             manufactured_harmonic_value_3d(point)) < 2.0e-14,
                    "transformed harmonic value is not covariant");
            require((transformed_manufactured_harmonic_gradient_3d(
                         item.transform, moved) -
                     item.transform.forward_vector(
                         manufactured_harmonic_gradient_3d(point))).norm() < 2.0e-14,
                    "transformed harmonic gradient is not covariant");
        }

    }
}
void test_acceptance_criterion_combination()
{
    using Status = RigidStudyCriterionStatus3D;
    require(combine_rigid_study_criteria_3d(
                {Status::Pass, Status::Fail, Status::NotEvaluated},
                true, true) == Status::Fail,
            "failure dominates rigid-study criteria");
    require(combine_rigid_study_criteria_3d(
                {Status::Pass, Status::NotEvaluated},
                false, true) == Status::NotEvaluated,
            "empty rigid-study result is not evaluated");
    require(combine_rigid_study_criteria_3d(
                {Status::Pass, Status::Pass}, true, true) == Status::Pass,
            "complete passing rigid-study criteria pass");
    require(combine_rigid_study_criteria_3d(
                {Status::Pass, Status::NotEvaluated},
                true, true) == Status::NotEvaluated,
            "incomplete acceptance remains not evaluated");
    require(combine_rigid_study_criteria_3d(
                {Status::Pass, Status::NotEvaluated},
                true, false) == Status::Pass,
            "smoke-study unavailable criteria are neutral");
}



} // namespace

int main()
{
    try {
        test_acceptance_criterion_combination();
        test_catalog_metadata();
        test_transformed_harmonic_data_are_covariant();
        std::cout << "3D Dirichlet rigid-transform study tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "3D Dirichlet rigid-transform study test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
