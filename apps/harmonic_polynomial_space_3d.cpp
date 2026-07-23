#include "harmonic_polynomial_space_3d.hpp"

#include <Eigen/SVD>

#include <array>
#include <map>
#include <stdexcept>

namespace kfbim::app3d {
namespace {

double integer_power(double value, int power)
{
    double result = 1.0;
    for (int i = 0; i < power; ++i)
        result *= value;
    return result;
}

} // namespace

HarmonicPolynomialSpace3D::HarmonicPolynomialSpace3D(int degree)
    : degree_(degree)
{
    if (degree_ < 1) {
        throw std::invalid_argument(
            "harmonic polynomial degree must be positive");
    }
    powers_ = powers_through_degree(degree_);
    const std::vector<PolynomialPower3D> lower =
        powers_through_degree(degree_ - 2);
    std::map<std::array<int, 3>, int> lower_index;
    for (int i = 0; i < static_cast<int>(lower.size()); ++i) {
        lower_index[{lower[static_cast<std::size_t>(i)].x,
                     lower[static_cast<std::size_t>(i)].y,
                     lower[static_cast<std::size_t>(i)].z}] = i;
    }

    Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(
        static_cast<int>(lower.size()), static_cast<int>(powers_.size()));
    for (int col = 0; col < static_cast<int>(powers_.size()); ++col) {
        const PolynomialPower3D p = powers_[static_cast<std::size_t>(col)];
        if (p.x >= 2) {
            laplacian(lower_index.at({p.x - 2, p.y, p.z}), col)
                += static_cast<double>(p.x * (p.x - 1));
        }
        if (p.y >= 2) {
            laplacian(lower_index.at({p.x, p.y - 2, p.z}), col)
                += static_cast<double>(p.y * (p.y - 1));
        }
        if (p.z >= 2) {
            laplacian(lower_index.at({p.x, p.y, p.z - 2}), col)
                += static_cast<double>(p.z * (p.z - 1));
        }
    }

    const int expected_dimension = (degree_ + 1) * (degree_ + 1);
    if (laplacian.rows() == 0) {
        if (static_cast<int>(powers_.size()) != expected_dimension)
            throw std::runtime_error("unexpected linear harmonic dimension");
        transform_ = Eigen::MatrixXd::Identity(
            static_cast<int>(powers_.size()), expected_dimension);
        return;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        laplacian, Eigen::ComputeFullV);
    if (static_cast<int>(powers_.size()) - laplacian.rows()
        != expected_dimension) {
        throw std::runtime_error(
            "unexpected 3D harmonic-space dimension");
    }
    transform_ = svd.matrixV().rightCols(expected_dimension);
    if ((laplacian * transform_).norm() > 1.0e-10) {
        throw std::runtime_error(
            "failed to construct a harmonic polynomial basis");
    }
}

int HarmonicPolynomialSpace3D::degree() const noexcept
{
    return degree_;
}

int HarmonicPolynomialSpace3D::dimension() const noexcept
{
    return static_cast<int>(transform_.cols());
}

Eigen::VectorXd HarmonicPolynomialSpace3D::basis(
    double x, double y, double z) const
{
    Eigen::VectorXd monomials(static_cast<int>(powers_.size()));
    for (int i = 0; i < monomials.size(); ++i) {
        const PolynomialPower3D p = powers_[static_cast<std::size_t>(i)];
        monomials[i] = integer_power(x, p.x)
                     * integer_power(y, p.y)
                     * integer_power(z, p.z);
    }
    return transform_.transpose() * monomials;
}

Eigen::MatrixXd HarmonicPolynomialSpace3D::gradient(
    double x, double y, double z) const
{
    Eigen::MatrixXd monomial_gradient = Eigen::MatrixXd::Zero(
        3, static_cast<int>(powers_.size()));
    for (int i = 0; i < static_cast<int>(powers_.size()); ++i) {
        const PolynomialPower3D p = powers_[static_cast<std::size_t>(i)];
        if (p.x > 0) {
            monomial_gradient(0, i) = static_cast<double>(p.x)
                * integer_power(x, p.x - 1)
                * integer_power(y, p.y)
                * integer_power(z, p.z);
        }
        if (p.y > 0) {
            monomial_gradient(1, i) = static_cast<double>(p.y)
                * integer_power(x, p.x)
                * integer_power(y, p.y - 1)
                * integer_power(z, p.z);
        }
        if (p.z > 0) {
            monomial_gradient(2, i) = static_cast<double>(p.z)
                * integer_power(x, p.x)
                * integer_power(y, p.y)
                * integer_power(z, p.z - 1);
        }
    }
    return monomial_gradient * transform_;
}

std::vector<HarmonicPolynomialSpace3D::PolynomialPower3D>
HarmonicPolynomialSpace3D::powers_through_degree(int degree)
{
    std::vector<PolynomialPower3D> result;
    if (degree < 0)
        return result;
    for (int total = 0; total <= degree; ++total) {
        for (int px = 0; px <= total; ++px) {
            for (int py = 0; py <= total - px; ++py)
                result.push_back({px, py, total - px - py});
        }
    }
    return result;
}

Eigen::MatrixXd svd_pseudoinverse_3d(
    const Eigen::MatrixXd& matrix,
    double relative_cutoff)
{
    if (matrix.rows() == 0 || matrix.cols() == 0) {
        throw std::runtime_error(
            "cannot invert an empty Cauchy design matrix");
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular = svd.singularValues();
    if (singular.size() == 0 || !(singular[0] > 0.0)) {
        throw std::runtime_error(
            "cannot invert an empty Cauchy design matrix");
    }
    Eigen::VectorXd inverse = singular;
    const double cutoff = relative_cutoff * singular[0];
    for (int i = 0; i < inverse.size(); ++i)
        inverse[i] = singular[i] > cutoff ? 1.0 / singular[i] : 0.0;
    return svd.matrixV() * inverse.asDiagonal() * svd.matrixU().transpose();
}

} // namespace kfbim::app3d
