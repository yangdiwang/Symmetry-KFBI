#include "topology_mean_free_reduction_3d.hpp"

#include <Eigen/SVD>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kfbim::app3d::SparseMatrixCSR3D;
using kfbim::app3d::TopologyMeanFreeReduction3D;
using kfbim::app3d::eliminate_topology_mean_3d;

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

SparseMatrixCSR3D sparse_matrix(
    Eigen::Index rows,
    Eigen::Index columns,
    std::initializer_list<Eigen::Triplet<double>> entries)
{
    SparseMatrixCSR3D result(rows, columns);
    result.setFromTriplets(entries.begin(), entries.end());
    result.makeCompressed();
    return result;
}

void test_affine_mean_elimination_and_source_coordinate_map()
{
    // The input affine set also satisfies C c=d and C G=0.  The global mean
    // operation must preserve those pre-existing topology invariants.
    const SparseMatrixCSR3D C = sparse_matrix(
        2, 6, {{0, 0, 1.0}, {0, 1, -1.0},
               {1, 2, 1.0}, {1, 3, 1.0}, {1, 4, -1.0}});
    Eigen::VectorXd particular(6);
    particular << 1.0, 1.0, 0.5, -0.25, 0.25, 2.0;
    const Eigen::VectorXd d = C * particular;
    const SparseMatrixCSR3D homogeneous = sparse_matrix(
        6, 4,
        {{0, 0, 1.0}, {1, 0, 1.0},
         {2, 1, 1.0}, {3, 1, -1.0},
         {2, 2, 1.0}, {4, 2, 1.0},
         {5, 3, 2.0}});
    require((C * homogeneous).norm() < 1.0e-15,
            "test homogeneous map satisfies the topology rows");

    Eigen::VectorXd mean_dual(6);
    mean_dual << 0.05, 0.15, 0.20, 0.10, 0.30, 0.20;
    const TopologyMeanFreeReduction3D result =
        eliminate_topology_mean_3d(
            particular, homogeneous, mean_dual);

    require(result.source_coordinate_count() == 4
                && result.reduced_coordinate_count() == 3,
            "one final global mean degree is removed");
    require(result.diagnostics().coordinate_rank == 3,
            "sparse coordinate map reports its identity-minor rank");
    require_near(mean_dual.dot(result.particular()), 0.0, 3.0e-15,
                 "adjusted affine particular has zero mean");
    require((result.homogeneous().transpose() * mean_dual)
                    .lpNorm<Eigen::Infinity>()
                < 3.0e-15,
            "every final homogeneous density direction has zero mean");
    require((C * result.particular() - d).norm() < 2.0e-15,
            "mean adjustment preserves the affine topology target");
    require((C * result.homogeneous()).norm() < 2.0e-15,
            "mean-free directions preserve homogeneous topology rows");

    Eigen::Vector3d reduced;
    reduced << 0.7, -1.1, 0.4;
    const Eigen::VectorXd source =
        result.coordinate_particular()
        + result.coordinate_homogeneous() * reduced;
    const Eigen::VectorXd expected = particular + homogeneous * source;
    require((result.lift(reduced) - expected).norm() < 2.0e-15,
            "source coordinate map reproduces the final affine lift");
    require_near(mean_dual.dot(result.lift(reduced)), 0.0, 4.0e-15,
                 "all lifted final densities satisfy the mean constraint");

    const Eigen::MatrixXd dense_coordinate(
        result.coordinate_homogeneous());
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        dense_coordinate, Eigen::ComputeThinU | Eigen::ComputeThinV);
    require(svd.rank() == 3,
            "coordinate map has the claimed numerical rank");
    const double condition = svd.singularValues()[0]
        / svd.singularValues()[2];
    require_near(condition,
                 result.diagnostics().coordinate_map_condition,
                 2.0e-14,
                 "closed-form sparse coordinate condition is exact");
    require(result.diagnostics().maximum_elimination_multiplier <= 1.0,
            "maximum-moment pivot bounds every elimination multiplier");
}

void test_nonzero_target_and_single_coordinate_exhaustion()
{
    Eigen::Vector3d particular;
    particular << 2.0, -1.0, 0.5;
    const SparseMatrixCSR3D homogeneous = sparse_matrix(
        3, 1, {{0, 0, 1.0}, {1, 0, -2.0}, {2, 0, 0.5}});
    Eigen::Vector3d mean_dual;
    mean_dual << 0.2, 0.3, 0.5;
    constexpr double target = -0.75;
    const TopologyMeanFreeReduction3D result =
        eliminate_topology_mean_3d(
            particular, homogeneous, mean_dual, target);
    require(result.reduced_coordinate_count() == 0
                && result.homogeneous().cols() == 0
                && result.coordinate_homogeneous().cols() == 0,
            "one observable source coordinate may be fully exhausted");
    require_near(mean_dual.dot(result.particular()), target, 2.0e-15,
                 "single-coordinate correction reaches a nonzero target");
    const Eigen::VectorXd empty(0);
    require((result.lift(empty) - result.particular()).norm() == 0.0,
            "zero-dimensional affine lift returns its unique particular");
}

void test_one_column_output_condition_is_one()
{
    const Eigen::Vector3d particular = Eigen::Vector3d::Zero();
    const SparseMatrixCSR3D homogeneous = sparse_matrix(
        3, 2, {{0, 0, 2.0}, {1, 1, 3.0}});
    Eigen::Vector3d mean_dual;
    mean_dual << 0.5, 0.25, 0.25;
    const TopologyMeanFreeReduction3D result =
        eliminate_topology_mean_3d(
            particular, homogeneous, mean_dual);
    require(result.reduced_coordinate_count() == 1,
            "two source coordinates leave one mean-free coordinate");
    require_near(result.diagnostics().coordinate_map_condition,
                 1.0, 0.0,
                 "a nonzero one-column map has condition one");
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        Eigen::MatrixXd(result.coordinate_homogeneous()),
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    require(svd.singularValues().size() == 1,
            "one-column condition oracle has one singular value");
}

void test_maximum_moment_pivot_is_deterministic()
{
    const SparseMatrixCSR3D homogeneous = sparse_matrix(
        4, 3,
        {{0, 0, 100.0}, {1, 0, -100.0},
         {0, 1, 1.0}, {2, 1, 2.0},
         {1, 2, 1.0}, {3, 2, 5.0}});
    const Eigen::Vector4d particular = Eigen::Vector4d::Ones();
    Eigen::Vector4d mean_dual;
    mean_dual << 0.25, 0.25, 0.25, 0.25;
    const TopologyMeanFreeReduction3D result =
        eliminate_topology_mean_3d(
            particular, homogeneous, mean_dual);
    // Moments are (0, 0.75, 1.5), so coordinate two is the unique pivot.
    require(result.diagnostics().pivot_coordinate == 2,
            "largest absolute mean moment selects the pivot");
    require_near(result.diagnostics().pivot_moment, 1.5, 1.0e-15,
                 "pivot diagnostic retains its signed mean moment");
}

void test_unobservable_and_invalid_inputs_are_explicit()
{
    const Eigen::Vector3d particular = Eigen::Vector3d::Zero();
    const SparseMatrixCSR3D homogeneous = sparse_matrix(
        3, 2, {{0, 0, 1.0}, {1, 0, -1.0},
               {1, 1, 1.0}, {2, 1, -1.0}});
    const Eigen::Vector3d constant_mean = Eigen::Vector3d::Ones();
    bool unobservable_threw = false;
    try {
        (void)eliminate_topology_mean_3d(
            particular, homogeneous, constant_mean);
    } catch (const std::runtime_error&) {
        unobservable_threw = true;
    }
    require(unobservable_threw,
            "an unobservable mean is not silently reported as rank-one");

    SparseMatrixCSR3D no_coordinates(3, 0);
    bool empty_threw = false;
    try {
        (void)eliminate_topology_mean_3d(
            particular, no_coordinates, constant_mean);
    } catch (const std::invalid_argument&) {
        empty_threw = true;
    }
    require(empty_threw,
            "a fully constrained input cannot eliminate another coordinate");

    Eigen::Vector3d nonfinite = constant_mean;
    nonfinite[1] = std::numeric_limits<double>::quiet_NaN();
    bool nonfinite_threw = false;
    try {
        (void)eliminate_topology_mean_3d(
            particular, homogeneous, nonfinite);
    } catch (const std::invalid_argument&) {
        nonfinite_threw = true;
    }
    require(nonfinite_threw,
            "non-finite mean data is rejected before elimination");
}

} // namespace

int main()
{
    try {
        test_affine_mean_elimination_and_source_coordinate_map();
        test_nonzero_target_and_single_coordinate_exhaustion();
        test_one_column_output_condition_is_one();
        test_maximum_moment_pivot_is_deterministic();
        test_unobservable_and_invalid_inputs_are_explicit();
        std::cout << "topology_mean_free_reduction_3d_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "topology_mean_free_reduction_3d_test: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
