#pragma once

#include <vector>

#include <Eigen/Dense>

namespace kfbim {

enum class EdgeSingularRegularBasis3D {
    // Python all-edge extractor: {1, R, S, R*S} independently on each face.
    PythonFourTerm,
    // Complete monomials R^i S^j with i+j <= regular_total_degree.
    TotalDegree
};

// One surface sample near a feature edge. rho is the positive in-face
// distance to the edge, s is an arc-length-like edge coordinate, and psi is
// the polar angle in the plane normal to the edge. regular_group identifies
// the incident smooth face: every group gets an independent regular
// polynomial, while all groups share the same edge-strength B-splines.
struct EdgeSingularBSplineSample3D {
    double rho = 0.0;
    double s = 0.0;
    double psi = 0.0;
    double weight = 1.0;
    int regular_group = 0;
    int point_index = -1;
    // The Python artificial-edge construction uses a common +/- trace sign
    // in addition to cos(lambda*psi). Keep it per sample so interior and
    // exterior lifting conventions can use the same fitter.
    double singular_sign = 1.0;
};

// Global-in-s edge model
//
//   mu_g(rho,s,psi) = p_g(R,S)
//       + sum_m C_m(s) R^lambda_m cos(lambda_m psi),
//   C_m(s) = sum_j c_mj B_j(s),  R = rho/rho_scale.
//
// Each p_g is an independent total-degree regular polynomial for one
// incident smooth face. B_j are clamped, uniformly-spaced B-splines on
// [s_min,s_max]. The physical coefficient multiplying rho^lambda is
// a_m(s)=C_m(s)/rho_scale^lambda.
struct EdgeSingularBSplineFitOptions3D {
    EdgeSingularRegularBasis3D regular_basis =
        EdgeSingularRegularBasis3D::PythonFourTerm;
    int regular_total_degree = 2;
    int spline_degree = 3;
    // Set spline_elements directly, or leave it zero and provide a positive
    // spline_spacing. In the latter case ceil(edge_length/spacing) is used.
    int spline_elements = 0;
    double spline_spacing = 0.0;
    double s_min = 0.0;
    double s_max = 1.0;
    double rho_scale = 1.0;
    std::vector<double> exponents{2.0 / 3.0, 4.0 / 3.0};
    // Tikhonov penalty on singular coefficients and a second-difference
    // penalty along each B-spline coefficient block. Both act after design
    // column scaling, matching the Python all-edge construction.
    double coefficient_ridge = 0.0;
    double smoothness_penalty = 0.0;
    // Match notes/kfbi3d_l_prism_all_edges_all_vertices.py: discard singular
    // directions below rcond*sigma_max and use sigma/(sigma^2+ridge).
    double svd_relative_tolerance = 1.0e-7;
    double spectral_ridge = 1.0e-8;
    // The Python extractor keeps the resolved subspace instead of rejecting a
    // locally rank-deficient edge. Set this for legacy exact-recovery tests.
    bool require_full_rank = false;
};

struct EdgeSingularBSplineFitDiagnostics3D {
    int sample_count = 0;
    int regular_group_count = 0;
    int regular_basis_size = 0;
    int spline_basis_size = 0;
    int singular_basis_size = 0;
    int total_basis_size = 0;
    int active_basis_size = 0;
    int numerical_rank = 0;
    double scaled_data_condition_number = 0.0;
    double augmented_condition_number = 0.0;
    // This is near roundoff only for an unfiltered full-rank pseudoinverse.
    double data_reproduction_residual = 0.0;
};

// Geometry-dependent, fixed linear extractor for a complete edge-strength
// function. It is safe inside a matrix-free GMRES operator: construction may
// use SVD, but every subsequent fit/evaluation is a matrix-vector product.
class EdgeSingularBSplineFit3D {
public:
    EdgeSingularBSplineFit3D(
        std::vector<EdgeSingularBSplineSample3D> samples,
        EdgeSingularBSplineFitOptions3D options);

    int sample_count() const;
    int basis_size() const;
    int regular_group_count() const;
    int regular_basis_size_per_group() const;
    int spline_basis_size() const;
    int singular_coefficient_count() const;
    const std::vector<double>& knots() const;
    const std::vector<EdgeSingularBSplineSample3D>& samples() const;
    const EdgeSingularBSplineFitOptions3D& options() const;
    const EdgeSingularBSplineFitDiagnostics3D& diagnostics() const;

    Eigen::VectorXd fit_coefficients(
        const Eigen::Ref<const Eigen::VectorXd>& sample_values) const;
    Eigen::VectorXd singular_coefficients(
        const Eigen::Ref<const Eigen::VectorXd>& sample_values) const;

    // C_m^(d)(s), where the fitted field uses C_m(s)*(rho/rho_scale)^lambda.
    Eigen::VectorXd normalized_strength_weights(
        double s,
        int exponent_index,
        int derivative_order = 0) const;
    double normalized_strength(
        const Eigen::Ref<const Eigen::VectorXd>& sample_values,
        double s,
        int exponent_index,
        int derivative_order = 0) const;

    // Physical a_m^(d)(s) multiplying rho^lambda.
    Eigen::VectorXd singular_strength_weights(
        double s,
        int exponent_index,
        int derivative_order = 0) const;
    double singular_strength(
        const Eigen::Ref<const Eigen::VectorXd>& sample_values,
        double s,
        int exponent_index,
        int derivative_order = 0) const;
    // Fast path for repeated field evaluation after fitting once.  The input
    // is singular_coefficients(sample_values), ordered by exponent and then
    // by B-spline basis index.
    double singular_strength_from_coefficients(
        const Eigen::Ref<const Eigen::VectorXd>& singular_coefficients,
        double s,
        int exponent_index,
        int derivative_order = 0) const;

    // Singular part only; the independently fitted regular face background
    // is deliberately excluded from these functionals.
    Eigen::VectorXd singular_value_weights(double rho,
                                           double s,
                                           double psi,
                                           double singular_sign = 1.0) const;
    Eigen::VectorXd singular_radial_second_derivative_weights(
        double rho,
        double s,
        double psi,
        double singular_sign = 1.0) const;
    Eigen::VectorXd singular_tangential_second_derivative_weights(
        double rho,
        double s,
        double psi,
        double singular_sign = 1.0) const;
    Eigen::VectorXd singular_flat_face_laplacian_weights(
        double rho,
        double s,
        double psi,
        double singular_sign = 1.0) const;
    // For the full wedge harmonic r^lambda cos(lambda psi), the transverse
    // polar Laplacian cancels; Delta_3D leaves C_m''(s) R^lambda cos(...).
    Eigen::VectorXd singular_volume_residual_weights(
        double rho,
        double s,
        double psi,
        double singular_sign = 1.0) const;

    double evaluate(const Eigen::Ref<const Eigen::VectorXd>& weights,
                    const Eigen::Ref<const Eigen::VectorXd>& sample_values) const;

private:
    struct RegularTerm {
        int rho_power = 0;
        int s_power = 0;
    };

    Eigen::VectorXd bspline_values(double s, int derivative_order) const;
    double bspline_basis_value(int index,
                               int degree,
                               int derivative_order,
                               double s) const;
    Eigen::VectorXd sample_weights_from_coefficient_functional(
        const Eigen::Ref<const Eigen::VectorXd>& functional) const;
    Eigen::VectorXd singular_coefficient_functional(
        double rho,
        double s,
        double psi,
        int rho_derivative_order,
        int s_derivative_order,
        double singular_sign) const;
    int singular_basis_index(int exponent_index, int spline_index) const;
    void require_sample_values(
        const Eigen::Ref<const Eigen::VectorXd>& sample_values) const;

    std::vector<EdgeSingularBSplineSample3D> samples_;
    EdgeSingularBSplineFitOptions3D options_;
    int regular_group_count_ = 0;
    std::vector<RegularTerm> regular_terms_;
    std::vector<double> knots_;
    int spline_basis_size_ = 0;
    int regular_basis_size_ = 0;
    int singular_basis_offset_ = 0;
    Eigen::MatrixXd coefficient_operator_;
    EdgeSingularBSplineFitDiagnostics3D diagnostics_;
};

} // namespace kfbim
