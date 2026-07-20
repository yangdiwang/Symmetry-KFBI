#pragma once

#include <vector>

#include <Eigen/Dense>

namespace kfbim {

// Geometry-only input for one face branch adjacent to a three-dimensional
// feature edge.  rho is the positive, in-face distance to the edge and s is
// an arc-length-like edge coordinate.  Samples from different smooth surface
// groups must not be mixed in one fit.
struct EdgeSingularSample3D {
    double rho = 0.0;
    double s = 0.0;
    double weight = 1.0;
    // Optional caller-owned interface DOF index.  The algebraic fitter does
    // not dereference it, but retaining it beside the geometry makes it
    // straightforward to scatter the returned local weights into a global
    // sparse KFBI surface row.
    int point_index = -1;
};

// Local enriched model in dimensionless coordinates
//
//   R = rho / rho_scale,  T = (s - s0) / s_scale.
//
// It contains every regular monomial R^i T^j with i+j <=
// regular_total_degree and, for every lambda in exponents, the edge terms
// R^lambda T^j, 0 <= j <= strength_degree.  The latter are a local Taylor
// representation of a_lambda(s) rho^lambda and therefore allow a varying
// singular strength without requiring samples at one common edge station.
struct EdgeSingularFitOptions3D {
    int regular_total_degree = 2;
    int strength_degree = 2;
    std::vector<double> exponents{2.0 / 3.0, 4.0 / 3.0};
    double rho_scale = 1.0;
    double s_scale = 1.0;
    double svd_relative_tolerance = 1.0e-12;
};

struct EdgeSingularFitDiagnostics3D {
    int sample_count = 0;
    int basis_size = 0;
    int numerical_rank = 0;
    double scaled_condition_number = 0.0;
    double reproduction_residual = 0.0;
};

// Precomputed linear fit/differentiation operator for one edge-near target.
// The constructor depends only on geometry.  Consequently all methods that
// consume sample_values are fixed linear maps and are safe to use inside a
// matrix-free GMRES operator.
class EdgeSingularFit3D {
public:
    EdgeSingularFit3D(std::vector<EdgeSingularSample3D> samples,
                      double edge_station,
                      EdgeSingularFitOptions3D options = {});

    int sample_count() const;
    int basis_size() const;
    double edge_station() const;
    const std::vector<EdgeSingularSample3D>& samples() const;
    const EdgeSingularFitOptions3D& options() const;
    const EdgeSingularFitDiagnostics3D& diagnostics() const;

    Eigen::VectorXd fit_coefficients(
        const Eigen::Ref<const Eigen::VectorXd>& sample_values) const;

    // Fixed sample weights for evaluation at (rho, s).  The derivative
    // methods are physical derivatives, including rho_scale^{-2} or
    // s_scale^{-2}.  flat_surface_laplacian is appropriate for a planar prism
    // face.  A curved NURBS caller can assemble its metric-dependent
    // Laplace--Beltrami row from derivative_weights(), without refitting.
    Eigen::VectorXd value_weights(double rho, double s) const;
    Eigen::VectorXd derivative_weights(double rho,
                                       double s,
                                       int rho_derivative_order,
                                       int s_derivative_order) const;
    Eigen::VectorXd radial_second_derivative_weights(double rho,
                                                      double s) const;
    Eigen::VectorXd tangential_second_derivative_weights(double rho,
                                                          double s) const;
    Eigen::VectorXd flat_surface_laplacian_weights(double rho,
                                                   double s) const;

    double evaluate(const Eigen::Ref<const Eigen::VectorXd>& weights,
                    const Eigen::Ref<const Eigen::VectorXd>& sample_values) const;

    // Recover d^order a_lambda/ds^order at the expansion station s0 from
    //
    //   a_lambda(s) rho^lambda.
    //
    // This is optional diagnostic/output information; spread can use the
    // derivative weights above without explicitly materializing a_lambda.
    Eigen::VectorXd singular_strength_weights(int exponent_index,
                                               int derivative_order = 0) const;
    double singular_strength(
        const Eigen::Ref<const Eigen::VectorXd>& sample_values,
        int exponent_index,
        int derivative_order = 0) const;

private:
    struct BasisTerm {
        double rho_power = 0.0;
        int s_power = 0;
        int exponent_index = -1;
    };

    Eigen::VectorXd basis_functional(double rho,
                                     double s,
                                     int rho_derivative_order,
                                     int s_derivative_order) const;
    Eigen::VectorXd sample_weights_from_functional(
        const Eigen::Ref<const Eigen::VectorXd>& functional) const;
    void require_sample_values(
        const Eigen::Ref<const Eigen::VectorXd>& sample_values) const;

    std::vector<EdgeSingularSample3D> samples_;
    double edge_station_ = 0.0;
    EdgeSingularFitOptions3D options_;
    std::vector<BasisTerm> basis_terms_;
    Eigen::MatrixXd coefficient_operator_;
    EdgeSingularFitDiagnostics3D diagnostics_;
};

} // namespace kfbim
