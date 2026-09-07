#include "src/support/trace/mean_free_trace_mass_coordinates_3d.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using kfbim::app3d::MeanFreeTraceMassCoordinates3D;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void test_general_direction()
{
    Eigen::Vector4d mean;
    mean << 2.0, -1.0, 3.0, 0.5;
    const MeanFreeTraceMassCoordinates3D coordinates(mean);
    require(coordinates.source_coordinate_count() == 4,
            "source coordinate count");
    require(coordinates.reduced_coordinate_count() == 3,
            "reduced coordinate count");
    require(coordinates.diagnostics().pivot_coordinate == 2,
            "maximum-moment pivot");

    Eigen::MatrixXd q(4, 3);
    for (int column = 0; column < 3; ++column) {
        Eigen::Vector3d unit = Eigen::Vector3d::Zero();
        unit[column] = 1.0;
        q.col(column) = coordinates.lift(unit);
    }
    require((q.transpose() * q - Eigen::Matrix3d::Identity())
                    .cwiseAbs().maxCoeff()
                < 2.0e-14,
            "implicit Q is orthonormal");
    require((mean.transpose() * q).cwiseAbs().maxCoeff() < 2.0e-14,
            "implicit Q is mean-free");

    Eigen::Vector3d reduced;
    reduced << 0.3, -0.7, 1.1;
    require((coordinates.restrict(coordinates.lift(reduced)) - reduced)
                    .lpNorm<Eigen::Infinity>()
                < 2.0e-14,
            "Q^T Q round trip");

    Eigen::Vector4d full;
    full << -0.2, 0.4, 1.3, -0.8;
    require(std::abs(coordinates.lift(reduced).dot(full)
                     - reduced.dot(coordinates.restrict(full)))
                < 2.0e-14,
            "lift and restriction are Euclidean adjoints");
    const Eigen::Matrix4d exact_projector = Eigen::Matrix4d::Identity()
        - mean * mean.transpose() / mean.squaredNorm();
    require((coordinates.project(full) - exact_projector * full)
                    .lpNorm<Eigen::Infinity>()
                < 3.0e-14,
            "Q Q^T is the mean-free projector");

    const MeanFreeTraceMassCoordinates3D scaled_coordinates(-7.0 * mean);
    require((scaled_coordinates.project(full) - coordinates.project(full))
                    .lpNorm<Eigen::Infinity>()
                < 2.0e-14,
            "mean-free projector is invariant under dual scaling");
}

void test_single_supported_moment()
{
    Eigen::VectorXd mean(3);
    mean << -4.0, 0.0, 0.0;
    const MeanFreeTraceMassCoordinates3D coordinates(mean);
    Eigen::Vector2d reduced;
    reduced << 0.25, -0.75;
    const Eigen::Vector3d full = coordinates.lift(reduced);
    require(full[0] == 0.0 && full.tail<2>() == reduced,
            "single-supported moment gives coordinate embedding");
    require(coordinates.diagnostics().coordinate_map_condition == 1.0,
            "single-supported coordinate condition");
}

void test_two_coordinate_direction()
{
    Eigen::Vector2d mean;
    mean << 2.0, -5.0;
    const MeanFreeTraceMassCoordinates3D coordinates(mean);
    require(coordinates.reduced_coordinate_count() == 1,
            "two source coordinates give one final coordinate");
    require(coordinates.diagnostics().coordinate_map_condition == 1.0,
            "a one-column coordinate map has unit condition number");
    require(std::abs(mean.dot(coordinates.lift(Eigen::VectorXd::Ones(1))))
                < 2.0e-14,
            "two-coordinate lift is mean-free");
}

void test_input_validation()
{
    bool caught = false;
    try {
        MeanFreeTraceMassCoordinates3D invalid(Eigen::VectorXd::Ones(1));
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "one-coordinate input is rejected");

    caught = false;
    try {
        MeanFreeTraceMassCoordinates3D invalid(Eigen::VectorXd::Zero(3));
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "zero mean dual is rejected");

    caught = false;
    try {
        Eigen::VectorXd invalid_mean = Eigen::VectorXd::Ones(3);
        invalid_mean[1] = std::numeric_limits<double>::quiet_NaN();
        MeanFreeTraceMassCoordinates3D invalid(invalid_mean);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught, "non-finite mean dual is rejected");
}

} // namespace

int main()
{
    try {
        test_general_direction();
        test_single_supported_moment();
        test_two_coordinate_direction();
        test_input_validation();
        std::cout << "mean_free_trace_mass_coordinates_3d_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mean_free_trace_mass_coordinates_3d_test: "
                  << error.what() << '\n';
        return 1;
    }
}
