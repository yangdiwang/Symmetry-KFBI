#include "src/support/trace/harmonic_trace_correction_3d.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kfbim::app3d::HarmonicTraceCorrectionTermInput3D;
using kfbim::app3d::TraceCorrectionOwnerMode3D;
using kfbim::app3d::apply_harmonic_trace_correction_3d;

void require_near(double actual,
                  double expected,
                  double tolerance,
                  const std::string& message)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected)
            + ", got " + std::to_string(actual));
    }
}

template <class Function>
void require_throws(Function&& function, const std::string& message)
{
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

Eigen::MatrixXd reference_coefficients()
{
    Eigen::MatrixXd coefficients(3, 2);
    coefficients << 2.0, 3.0,
                    5.0, 7.0,
                   11.0, 13.0;
    return coefficients;
}

void test_center_route_uses_center_dof()
{
    const Eigen::MatrixXd coefficients = reference_coefficients();
    Eigen::VectorXd legacy(2);
    legacy << 17.0, 19.0;
    const double value = apply_harmonic_trace_correction_3d(
        1, coefficients, legacy, {},
        TraceCorrectionOwnerMode3D::CenterDof);
    require_near(value, 218.0, 0.0,
                 "center route did not use the center coefficient row");
}

void test_crossing_route_uses_each_foreign_owner()
{
    const Eigen::MatrixXd coefficients = reference_coefficients();
    Eigen::VectorXd legacy = Eigen::VectorXd::Zero(2);
    std::vector<HarmonicTraceCorrectionTermInput3D> terms(2);
    terms[0].owner_dof = 2;
    terms[0].evaluation = Eigen::Vector2d(1.0, 2.0);
    terms[1].owner_dof = 0;
    terms[1].evaluation = Eigen::Vector2d(-3.0, 4.0);

    const double value = apply_harmonic_trace_correction_3d(
        1, coefficients, legacy, terms,
        TraceCorrectionOwnerMode3D::CrossingOwner);
    require_near(value, 43.0, 0.0,
                 "crossing route did not use selected owner rows");
}

void test_crossing_route_preserves_term_order()
{
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Ones(3, 1);
    Eigen::VectorXd legacy = Eigen::VectorXd::Zero(1);
    std::vector<HarmonicTraceCorrectionTermInput3D> terms(3);
    terms[0].owner_dof = 0;
    terms[0].evaluation = Eigen::VectorXd::Constant(1, 1.0e16);
    terms[1].owner_dof = 1;
    terms[1].evaluation = Eigen::VectorXd::Constant(1, -1.0e16);
    terms[2].owner_dof = 2;
    terms[2].evaluation = Eigen::VectorXd::Constant(1, 1.0);

    const double value = apply_harmonic_trace_correction_3d(
        0, coefficients, legacy, terms,
        TraceCorrectionOwnerMode3D::CrossingOwner);
    require_near(value, 1.0, 0.0,
                 "crossing route reordered correction terms");
}

void test_crossing_route_accepts_no_wrong_side_terms()
{
    const Eigen::MatrixXd coefficients = reference_coefficients();
    Eigen::VectorXd legacy = Eigen::VectorXd::Ones(2);
    const double value = apply_harmonic_trace_correction_3d(
        0, coefficients, legacy, {},
        TraceCorrectionOwnerMode3D::CrossingOwner);
    require_near(value, 0.0, 0.0,
                 "empty crossing correction was not zero");
}

void test_invalid_owner_is_rejected()
{
    const Eigen::MatrixXd coefficients = reference_coefficients();
    Eigen::VectorXd legacy = Eigen::VectorXd::Zero(2);
    HarmonicTraceCorrectionTermInput3D term;
    term.owner_dof = 3;
    term.evaluation = Eigen::Vector2d::Ones();
    require_throws(
        [&] {
            (void)apply_harmonic_trace_correction_3d(
                0, coefficients, legacy, {term},
                TraceCorrectionOwnerMode3D::CrossingOwner);
        },
        "out-of-range crossing owner was accepted");
}

} // namespace

int main()
{
    try {
        test_center_route_uses_center_dof();
        test_crossing_route_uses_each_foreign_owner();
        test_crossing_route_preserves_term_order();
        test_crossing_route_accepts_no_wrong_side_terms();
        test_invalid_owner_is_rejected();
        std::cout << "harmonic trace correction 3D tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "harmonic trace correction 3D test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
