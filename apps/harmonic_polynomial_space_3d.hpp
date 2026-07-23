#pragma once

#include <Eigen/Dense>

#include <vector>

namespace kfbim::app3d {

class HarmonicPolynomialSpace3D {
public:
    explicit HarmonicPolynomialSpace3D(int degree);

    [[nodiscard]] int degree() const noexcept;
    [[nodiscard]] int dimension() const noexcept;
    [[nodiscard]] Eigen::VectorXd basis(
        double x, double y, double z) const;
    [[nodiscard]] Eigen::MatrixXd gradient(
        double x, double y, double z) const;

private:
    struct PolynomialPower3D {
        int x = 0;
        int y = 0;
        int z = 0;
    };

    static std::vector<PolynomialPower3D> powers_through_degree(int degree);

    int degree_ = 1;
    std::vector<PolynomialPower3D> powers_;
    Eigen::MatrixXd transform_;
};

[[nodiscard]] Eigen::MatrixXd svd_pseudoinverse_3d(
    const Eigen::MatrixXd& matrix,
    double relative_cutoff);

} // namespace kfbim::app3d
