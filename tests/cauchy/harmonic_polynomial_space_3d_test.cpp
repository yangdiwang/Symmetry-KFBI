#include "src/support/cauchy/harmonic_polynomial_space_3d.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using kfbim::app3d::HarmonicPolynomialSpace3D;
using kfbim::app3d::svd_pseudoinverse_3d;
using kfbim::app3d::svd_pseudoinverse_from_decomposition_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class Function>
void require_throws(Function&& function, const std::string& message)
{
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

void test_dimensions()
{
    require(HarmonicPolynomialSpace3D(1).dimension() == 4,
            "linear harmonic dimension");
    require(HarmonicPolynomialSpace3D(2).dimension() == 9,
            "quadratic harmonic dimension");
    require(HarmonicPolynomialSpace3D(3).dimension() == 16,
            "cubic harmonic dimension");
    require(HarmonicPolynomialSpace3D(4).dimension() == 25,
            "quartic harmonic dimension");
    require(HarmonicPolynomialSpace3D(3).degree() == 3,
            "harmonic degree accessor");
    require_throws(
        [] { (void)HarmonicPolynomialSpace3D(0); },
        "non-positive degree is rejected");
}

void test_basis_is_harmonic()
{
    for (int degree : {3, 4}) {
        const HarmonicPolynomialSpace3D space(degree);
        const Eigen::Vector3d point(0.17, -0.23, 0.31);
        const double epsilon = 1.0e-4;
        const double inverse_epsilon_sq = 1.0 / (epsilon * epsilon);
        for (int coefficient = 0;
             coefficient < space.dimension();
             ++coefficient) {
            const auto value = [&](const Eigen::Vector3d& x) {
                return space.basis(x.x(), x.y(), x.z())[coefficient];
            };
            double laplacian = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                Eigen::Vector3d plus = point;
                Eigen::Vector3d minus = point;
                plus[axis] += epsilon;
                minus[axis] -= epsilon;
                laplacian +=
                    (value(plus) - 2.0 * value(point) + value(minus))
                    * inverse_epsilon_sq;
            }
            require(std::abs(laplacian) < 3.0e-6,
                    "harmonic basis has nonzero Laplacian");
        }
    }
}

void test_gradient_matches_centered_difference()
{
    for (int degree : {3, 4}) {
        const HarmonicPolynomialSpace3D space(degree);
        const Eigen::Vector3d point(-0.19, 0.13, 0.27);
        const Eigen::MatrixXd gradient =
            space.gradient(point.x(), point.y(), point.z());
        const double epsilon = 1.0e-6;
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3d plus = point;
            Eigen::Vector3d minus = point;
            plus[axis] += epsilon;
            minus[axis] -= epsilon;
            const Eigen::VectorXd finite_difference =
                (space.basis(plus.x(), plus.y(), plus.z())
                 - space.basis(minus.x(), minus.y(), minus.z()))
                / (2.0 * epsilon);
            require(
                (gradient.row(axis).transpose() - finite_difference).norm()
                    < 3.0e-9,
                "harmonic basis gradient mismatch");
        }
    }
}

void test_pseudoinverse()
{
    Eigen::MatrixXd matrix(5, 3);
    matrix << 1.0, 0.2, -0.4,
              0.3, 1.1, 0.5,
              -0.2, 0.7, 1.3,
              0.9, -0.6, 0.8,
              1.2, 0.4, 0.1;
    const Eigen::MatrixXd inverse =
        svd_pseudoinverse_3d(matrix, 1.0e-13);
    require((inverse * matrix - Eigen::Matrix3d::Identity()).norm()
                < 2.0e-13,
            "SVD pseudoinverse is not a left inverse");
    require_throws(
        [] {
            (void)svd_pseudoinverse_3d(Eigen::MatrixXd(0, 0), 1.0e-13);
        },
        "empty pseudoinverse input is rejected");
}

void test_reused_pseudoinverse()
{
    for (const bool wide : {false, true}) {
        Eigen::MatrixXd matrix(5, 3);
        matrix << 1.0, 0.2, -0.4,
                  0.3, 1.1, 0.5,
                  -0.2, 0.7, 1.3,
                  0.9, -0.6, 0.8,
                  1.2, 0.4, 0.1;
        if (wide)
            matrix = matrix.transpose().eval();
        const Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(
            matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::MatrixXd reference = svd_pseudoinverse_3d(matrix, 3.0e-12);
        const Eigen::MatrixXd reused =
            svd_pseudoinverse_from_decomposition_3d(decomposition, 3.0e-12);
        require((reference.array() == reused.array()).all(),
                "reusing a decomposition changed pseudoinverse entries");
        const Eigen::JacobiSVD<Eigen::MatrixXd> full_decomposition(
            matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::MatrixXd full_reused =
            svd_pseudoinverse_from_decomposition_3d(full_decomposition, 3.0e-12);
        require(full_reused.rows() == matrix.cols()
                    && full_reused.cols() == matrix.rows()
                    && (full_reused - reference).norm() < 2.0e-13,
                "full rectangular decomposition reuse changed pseudoinverse");
        const Eigen::JacobiSVD<Eigen::MatrixXd> values_only(matrix);
        require_throws([&] {
            (void)svd_pseudoinverse_from_decomposition_3d(values_only, 3.0e-12);
        }, "pseudoinverse reuse must reject a values-only decomposition");
    }
    Eigen::MatrixXd diagonal = Eigen::MatrixXd::Zero(4, 4);
    diagonal.diagonal() << 1.0, 1.0e-8, 1.0e-13, 0.0;
    const Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(
        diagonal, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::MatrixXd reused =
        svd_pseudoinverse_from_decomposition_3d(decomposition, 3.0e-12);
    require(reused(0, 0) == 1.0 && reused(1, 1) == 1.0e8
                && reused(2, 2) == 0.0 && reused(3, 3) == 0.0,
            "reused decomposition must preserve the relative rank cutoff");
}

} // namespace

int main()
{
    try {
        test_dimensions();
        test_basis_is_harmonic();
        test_gradient_matches_centered_difference();
        test_pseudoinverse();
        test_reused_pseudoinverse();
        std::cout << "3D harmonic polynomial space tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "3D harmonic polynomial space test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
