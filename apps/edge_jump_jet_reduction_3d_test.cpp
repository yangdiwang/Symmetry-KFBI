#include "edge_jump_jet_reduction_3d.hpp"

#include <Eigen/LU>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using kfbim::app3d::EdgeJumpJetAffineReduction3D;
using kfbim::app3d::make_edge_jump_jet_compatibility_3d;
using kfbim::app3d::project_edge_jump_jet_targets_to_mean_zero_range_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_near(double actual,
                  double expected,
                  double tolerance,
                  const std::string& message)
{
    require(std::abs(actual - expected) <= tolerance, message);
}

void test_two_normal_compatibility()
{
    const auto relation = make_edge_jump_jet_compatibility_3d(
        Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), 2.0, 3.0);
    require_near(relation.normal_dot, 0.0, 1.0e-15,
                 "orthogonal edge normal dot");
    require_near(relation.first_conormal_derivative, 3.0, 1.0e-15,
                 "first conormal target");
    require_near(relation.second_conormal_derivative, 2.0, 1.0e-15,
                 "second conormal target");
    const Eigen::Vector3d first_gradient =
        2.0 * Eigen::Vector3d::UnitX()
        + relation.first_conormal_derivative * relation.first_conormal;
    const Eigen::Vector3d second_gradient =
        3.0 * Eigen::Vector3d::UnitY()
        + relation.second_conormal_derivative * relation.second_conormal;
    require((first_gradient - second_gradient).norm() <= 2.0e-15,
            "two independent conormal targets recover one jump gradient");

    bool threw = false;
    try {
        (void)make_edge_jump_jet_compatibility_3d(
            Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitX(), 1.0, 1.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "G1 seam is rejected by the sharp-edge formula");
}

EdgeJumpJetAffineReduction3D make_reduction()
{
    Eigen::MatrixXd edge(2, 5);
    edge << 1.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, -1.0, 0.0;
    Eigen::Vector2d data;
    data << 2.0, -1.0;
    Eigen::Vector2d lambda;
    lambda << -0.5, 0.25;
    const Eigen::VectorXd mean = Eigen::VectorXd::Ones(5) / 5.0;
    return EdgeJumpJetAffineReduction3D(
        edge, data, lambda, mean, 0.0);
}

void test_affine_constraints_and_intrinsic_mean()
{
    const EdgeJumpJetAffineReduction3D reduction = make_reduction();
    require(reduction.full_size() == 5, "full coordinate count");
    require(reduction.constraint_rank() == 3, "edge plus mean rank");
    require(reduction.reduced_size() == 2, "intrinsic reduced count");
    require(reduction.data_constraint_residual() <= 2.0e-14,
            "data lift satisfies affine constraints");
    require(reduction.lambda_constraint_residual() <= 2.0e-14,
            "lambda lift satisfies affine constraints");
    require(reduction.nullspace_constraint_residual() <= 2.0e-14,
            "homogeneous basis satisfies edge and mean constraints");
    require(reduction.nullspace_orthogonality_residual() <= 2.0e-14,
            "homogeneous basis is orthonormal");

    Eigen::Vector2d reduced;
    reduced << 0.37, -0.81;
    const double multiplier = 0.7;
    const Eigen::VectorXd full = reduction.expand(reduced, multiplier);
    require_near(full[0] - full[1], 2.0 - 0.5 * multiplier,
                 3.0e-14, "first lambda-coupled edge constraint");
    require_near(full[2] - full[3], -1.0 + 0.25 * multiplier,
                 3.0e-14, "second lambda-coupled edge constraint");
    require_near(full.mean(), 0.0, 3.0e-14,
                 "mean is intrinsic for every reduced vector and lambda");
}

void test_projected_operator_and_lambda_recovery()
{
    const EdgeJumpJetAffineReduction3D reduction = make_reduction();
    Eigen::Matrix<double, 5, 5> matrix;
    matrix <<
        3.0,  0.2, -0.1,  0.0,  0.1,
       -0.3,  2.8,  0.4, -0.2,  0.0,
        0.2, -0.1,  3.4,  0.3, -0.2,
        0.0,  0.2, -0.4,  2.6,  0.1,
       -0.1,  0.0,  0.2, -0.3,  3.2;
    Eigen::VectorXd raw_border(5);
    raw_border << 0.8, -0.1, 0.3, 0.5, -0.4;
    const Eigen::VectorXd effective_border = reduction.effective_border(
        raw_border, matrix * reduction.lambda_lift());

    Eigen::Vector2d expected_reduced;
    expected_reduced << -0.42, 0.93;
    const double expected_lambda = -0.36;
    const Eigen::VectorXd expected_full =
        reduction.expand(expected_reduced, expected_lambda);
    const Eigen::VectorXd rhs =
        matrix * expected_full + raw_border * expected_lambda;
    const Eigen::VectorXd shifted_rhs =
        rhs - matrix * reduction.data_lift();

    Eigen::MatrixXd reduced_matrix(
        reduction.reduced_size(), reduction.reduced_size());
    for (int column = 0; column < reduction.reduced_size(); ++column) {
        reduced_matrix.col(column) = reduction.project_and_test(
            matrix * reduction.homogeneous_basis().col(column),
            effective_border);
    }
    const Eigen::VectorXd reduced_rhs = reduction.project_and_test(
        shifted_rhs, effective_border);
    const Eigen::VectorXd solved_reduced =
        reduced_matrix.fullPivLu().solve(reduced_rhs);
    require((solved_reduced - expected_reduced).norm() <= 2.0e-13,
            "projected intrinsic system recovers reduced coordinates");

    const Eigen::VectorXd homogeneous_image =
        matrix * reduction.homogeneous_coordinates(solved_reduced);
    const double solved_lambda = reduction.recover_lambda(
        shifted_rhs - homogeneous_image, effective_border);
    require_near(solved_lambda, expected_lambda, 2.0e-13,
                 "effective border recovers operator-flux multiplier");
    const Eigen::VectorXd solved_full =
        reduction.expand(solved_reduced, solved_lambda);
    require((matrix * solved_full + raw_border * solved_lambda - rhs).norm()
                <= 2.0e-12,
            "reconstructed affine solution closes the unreduced equation");
}

void test_inconsistent_redundant_constraints_are_rejected()
{
    Eigen::MatrixXd edge(2, 3);
    edge << 1.0, 0.0, 0.0,
            2.0, 0.0, 0.0;
    Eigen::Vector2d data;
    data << 1.0, 3.0;
    const Eigen::Vector2d lambda = Eigen::Vector2d::Zero();
    const Eigen::Vector3d mean = Eigen::Vector3d::Ones();
    bool threw = false;
    try {
        (void)EdgeJumpJetAffineReduction3D(
            edge, data, lambda, mean, 0.0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "inconsistent redundant edge data are rejected");
}

void test_inconsistent_targets_are_projected_with_exact_mean()
{
    Eigen::MatrixXd edge(3, 4);
    edge << 1.0, -1.0, 0.0, 0.0,
            2.0, -2.0, 0.0, 0.0,
            0.0, 0.0, 1.0, -1.0;
    Eigen::MatrixXd targets(3, 2);
    targets << 1.0, -0.2,
               3.0, -0.1,
              -0.5,  0.7;
    const Eigen::Vector4d mean = Eigen::Vector4d::Ones() / 4.0;
    const auto projection =
        project_edge_jump_jet_targets_to_mean_zero_range_3d(
            edge, targets, mean);

    require(projection.normalized_projection_linf > 1.0e-2,
            "inconsistent dependent targets have a visible projection");
    require_near(projection.targets(1, 0),
                 2.0 * projection.targets(0, 0), 2.0e-14,
                 "projected data target respects the dependent row");
    require_near(projection.targets(1, 1),
                 2.0 * projection.targets(0, 1), 2.0e-14,
                 "projected lambda target respects the dependent row");
    require((edge * projection.lifts - projection.targets).norm()
                <= 3.0e-14,
            "projected targets have explicit coordinate lifts");
    require((mean.transpose() * projection.lifts).norm() <= 3.0e-14,
            "compatible target lifts are intrinsically mean zero");
    require_near(projection.mean_null_denominator, 1.0, 2.0e-14,
                 "constant mode supplies a stable mean-null direction");
    require(projection.mean_null_residual_linf <= 2.0e-14,
            "mean-null direction is in the derivative nullspace");
    require(projection.mean_null_identity_residual <= 2.0e-14,
            "mean-null projection identity holds");
    require(projection.lift_mean_residual_linf <= 2.0e-14,
            "reported lift mean audit closes");

    const EdgeJumpJetAffineReduction3D reduction(
        edge, projection.targets.col(0), projection.targets.col(1),
        mean, 0.0);
    require(reduction.data_constraint_residual() <= 3.0e-14,
            "projected data target is accepted by affine reduction");
    require(reduction.lambda_constraint_residual() <= 3.0e-14,
            "projected lambda target is accepted by affine reduction");
}

} // namespace

int main()
{
    try {
        test_two_normal_compatibility();
        test_affine_constraints_and_intrinsic_mean();
        test_projected_operator_and_lambda_recovery();
        test_inconsistent_redundant_constraints_are_rejected();
        test_inconsistent_targets_are_projected_with_exact_mean();
        std::cout << "edge jump-jet affine reduction tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "edge jump-jet affine reduction test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
