#include "src/support/cauchy/direct_coefficient_cauchy_3d.hpp"
#include "src/support/geometry/native_nurbs_surface_3d.hpp"
#include "src/support/geometry/native_nurbs_surface_transform_3d.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace kfbim::app3d;

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
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual)
            + " expected=" + std::to_string(expected)
            + " tolerance=" + std::to_string(tolerance));
    }
}

NativeNurbsDensitySpace3D make_density()
{
    NativeNurbsDensityOptions3D options;
    options.field = NativeDensityField3D::ValueTrace;
    options.reduction_backend = NativeDensityReductionBackend3D::BaseOnly;
    options.coefficients_per_direction = 7;
    return NativeNurbsDensitySpace3D(
        make_native_nurbs_surface_3d(GeometryKind3D::Torus), options);
}

NativeNurbsDensitySpace3D make_normal_density()
{
    NativeNurbsDensityOptions3D options;
    options.field = NativeDensityField3D::NormalTrace;
    options.reduction_backend = NativeDensityReductionBackend3D::BaseOnly;
    options.coefficients_per_direction = 7;
    return NativeNurbsDensitySpace3D(
        make_native_nurbs_surface_3d(GeometryKind3D::Torus), options);
}

void test_analytic_geometry_parameter_jet(
    const NativeNurbsDensitySpace3D& density)
{
    constexpr int patch = 0;
    constexpr double u = 0.37;
    constexpr double v = 0.41;
    constexpr double epsilon = 2.0e-4;
    const NativeSurfaceParameterJet3D jet =
        native_surface_parameter_jet_3d(density, patch, u, v);
    auto point = [&](double a, double b) {
        return density.geometry(patch, a, b).point;
    };
    const Eigen::Vector3d center = point(u, v);
    const Eigen::Vector3d du =
        (point(u + epsilon, v) - point(u - epsilon, v))
        / (2.0 * epsilon);
    const Eigen::Vector3d dv =
        (point(u, v + epsilon) - point(u, v - epsilon))
        / (2.0 * epsilon);
    const Eigen::Vector3d duu =
        (point(u + epsilon, v) - 2.0 * center
         + point(u - epsilon, v))
        / (epsilon * epsilon);
    const Eigen::Vector3d dvv =
        (point(u, v + epsilon) - 2.0 * center
         + point(u, v - epsilon))
        / (epsilon * epsilon);
    const Eigen::Vector3d duv =
        (point(u + epsilon, v + epsilon)
         - point(u + epsilon, v - epsilon)
         - point(u - epsilon, v + epsilon)
         + point(u - epsilon, v - epsilon))
        / (4.0 * epsilon * epsilon);
    require((jet.point - center).norm() < 3.0e-13,
            "analytic geometry jet point mismatch");
    require((jet.x_u - du).norm() < 2.0e-6,
            "analytic geometry x_u mismatch");
    require((jet.x_v - dv).norm() < 2.0e-6,
            "analytic geometry x_v mismatch");
    require((jet.x_uu - duu).norm() < 2.0e-5,
            "analytic geometry x_uu mismatch");
    require((jet.x_uv - duv).norm() < 2.0e-5,
            "analytic geometry x_uv mismatch");
    require((jet.x_vv - dvv).norm() < 2.0e-5,
            "analytic geometry x_vv mismatch");
    const NativeSurfaceNormalParameterJet3D normal_jet =
        surface_normal_parameter_jet_3d(jet);
    const Eigen::Vector3d normal_u =
        (density.geometry(patch, u + epsilon, v).normal
         - density.geometry(patch, u - epsilon, v).normal)
        / (2.0 * epsilon);
    const Eigen::Vector3d normal_v =
        (density.geometry(patch, u, v + epsilon).normal
         - density.geometry(patch, u, v - epsilon).normal)
        / (2.0 * epsilon);
    require((normal_jet.normal_u - normal_u).norm() < 2.0e-6,
            "analytic surface normal u derivative mismatch");
    require((normal_jet.normal_v - normal_v).norm() < 2.0e-6,
            "analytic surface normal v derivative mismatch");
}

void test_second_order_inverse_chain_rule(
    const NativeNurbsDensitySpace3D& density)
{
    constexpr int patch = 0;
    constexpr double u = 0.37;
    constexpr double v = 0.41;
    const NativeSurfaceParameterJet3D geometry =
        native_surface_parameter_jet_3d(density, patch, u, v);
    const LocalOrthonormalFrame3D frame =
        make_local_orthonormal_frame_3d(
            geometry.normal, geometry.x_u + 0.31 * geometry.x_v);
    const auto transform = parameter_to_cauchy_value_jet_matrix_3d(
        geometry, frame);
    const TangentGraphHessian3D graph =
        tangent_graph_hessian_from_parameter_jet_3d(geometry, frame);

    const double constant = 0.73;
    const Eigen::Vector3d gradient(0.8, -1.1, 0.45);
    ValueJet3D parameter_jet;
    parameter_jet <<
        constant + gradient.dot(geometry.point),
        gradient.dot(geometry.x_u),
        gradient.dot(geometry.x_v),
        gradient.dot(geometry.x_uu),
        gradient.dot(geometry.x_uv),
        gradient.dot(geometry.x_vv);
    const ValueJet3D actual = transform * parameter_jet;
    ValueJet3D expected;
    const double normal_gradient = gradient.dot(frame.normal);
    expected << constant + gradient.dot(geometry.point),
        gradient.dot(frame.tangent1),
        gradient.dot(frame.tangent2),
        normal_gradient * graph.h11,
        normal_gradient * graph.h12,
        normal_gradient * graph.h22;
    require((actual - expected).cwiseAbs().maxCoeff() < 3.0e-11,
            "second-order inverse parameter chain rule mismatch");
}

void test_known_dirichlet_analytic_jet(
    const NativeNurbsDensitySpace3D& density)
{
    constexpr int patch = 0;
    constexpr double u = 0.37;
    constexpr double v = 0.41;
    const NativeSurfaceParameterJet3D geometry =
        native_surface_parameter_jet_3d(density, patch, u, v);
    const LocalOrthonormalFrame3D frame =
        make_local_orthonormal_frame_3d(
            geometry.normal, geometry.x_u + 0.29 * geometry.x_v);
    const TangentGraphHessian3D graph =
        tangent_graph_hessian_from_parameter_jet_3d(geometry, frame);

    KnownDirichletValueGradientHessian3D known;
    known.value = -0.47;
    known.ambient_gradient = Eigen::Vector3d(0.83, -0.31, 1.07);
    known.ambient_hessian <<
        0.52, -0.17, 0.11,
        -0.17, -0.38, 0.26,
        0.11, 0.26, 0.71;
    const ValueJet3D actual =
        known_dirichlet_jet_from_ambient_derivatives_3d(
            known, geometry, frame);
    const double normal_gradient = known.ambient_gradient.dot(frame.normal);
    ValueJet3D expected;
    expected << known.value,
        known.ambient_gradient.dot(frame.tangent1),
        known.ambient_gradient.dot(frame.tangent2),
        frame.tangent1.dot(known.ambient_hessian * frame.tangent1)
            + normal_gradient * graph.h11,
        frame.tangent1.dot(known.ambient_hessian * frame.tangent2)
            + normal_gradient * graph.h12,
        frame.tangent2.dot(known.ambient_hessian * frame.tangent2)
            + normal_gradient * graph.h22;
    require((actual - expected).cwiseAbs().maxCoeff() < 4.0e-11,
            "analytic known Dirichlet surface jet mismatch");

    bool callback_called = false;
    const KnownDirichletValueGradientHessianCallback3D callback =
        [&](int callback_patch,
            double callback_u,
            double callback_v,
            const Eigen::Vector3d& point,
            const Eigen::Vector3d& normal) {
            callback_called = true;
            require(callback_patch == patch,
                    "known Dirichlet callback patch mismatch");
            require_near(callback_u, u, 0.0,
                         "known Dirichlet callback u");
            require_near(callback_v, v, 0.0,
                         "known Dirichlet callback v");
            require((point - geometry.point).norm() < 1.0e-15,
                    "known Dirichlet callback point mismatch");
            require((normal - geometry.normal).norm() < 1.0e-15,
                    "known Dirichlet callback normal mismatch");
            return known;
        };
    const ValueJet3D evaluated = evaluate_known_dirichlet_jet_3d(
        callback, patch, u, v, geometry, frame);
    require(callback_called, "known Dirichlet callback was not evaluated");
    require((evaluated - actual).cwiseAbs().maxCoeff() < 1.0e-14,
            "known Dirichlet callback jet mismatch");

    const Eigen::Vector3d target = geometry.point
        - 0.013 * frame.tangent1 + 0.019 * frame.tangent2
        - 0.006 * frame.normal;
    const CauchyPolynomialWeights3D weights =
        cauchy_polynomial_weights_3d(
            geometry.point, frame, graph, target);
    require_near(weights.w0.dot(evaluated),
                 weights.apply_value_jet(actual), 2.0e-14,
                 "precomposed known Dirichlet Cauchy value row");
}

void test_direct_density_stencils(
    const NativeNurbsDensitySpace3D& density)
{
    constexpr int patch = 0;
    constexpr double u = 0.37;
    constexpr double v = 0.41;
    const NativeSurfaceParameterJet3D geometry =
        native_surface_parameter_jet_3d(density, patch, u, v);
    const LocalOrthonormalFrame3D frame =
        make_local_orthonormal_frame_3d(
            geometry.normal, geometry.x_u + 0.17 * geometry.x_v);
    const DirectCoefficientValueJetPlan3D plan =
        build_direct_coefficient_value_jet_plan_3d(
            density, patch, u, v, frame);

    Eigen::VectorXd coefficients(density.c0_coefficient_count());
    for (Eigen::Index q = 0; q < coefficients.size(); ++q)
        coefficients[q] = std::sin(0.37 * static_cast<double>(q + 1));
    const ValueJet3D direct = plan.evaluate(coefficients);

    ValueJet3D parameter;
    const std::array<std::array<int, 2>, 6> derivatives{{
        {{0, 0}}, {{1, 0}}, {{0, 1}},
        {{2, 0}}, {{1, 1}}, {{0, 2}}}};
    for (int q = 0; q < 6; ++q) {
        parameter[q] = density.c0_parameter_derivative_row(
            patch, u, v,
            derivatives[static_cast<std::size_t>(q)][0],
            derivatives[static_cast<std::size_t>(q)][1]).dot(coefficients);
    }
    const ValueJet3D dense_reference =
        parameter_to_cauchy_value_jet_matrix_3d(geometry, frame)
        * parameter;
    require((direct - dense_reference).cwiseAbs().maxCoeff() < 2.0e-12,
            "sparse direct coefficient jet differs from dense chain rule");
    for (const NativeDensityC0Stencil3D& row : plan.cauchy_rows)
        require(row.count <= 16, "direct jet row lost local cubic support");

    constexpr double epsilon_first = 2.0e-6;
    constexpr double epsilon_second = 2.0e-4;
    auto value = [&](double a, double b) {
        return density.evaluate_c0(patch, a, b, coefficients);
    };
    const double center = value(u, v);
    require_near(parameter[0], center, 2.0e-13,
                 "density parameter value");
    require_near(parameter[1],
                 (value(u + epsilon_first, v)
                  - value(u - epsilon_first, v))
                     / (2.0 * epsilon_first),
                 2.0e-8, "density parameter u derivative");
    require_near(parameter[2],
                 (value(u, v + epsilon_first)
                  - value(u, v - epsilon_first))
                     / (2.0 * epsilon_first),
                 2.0e-8, "density parameter v derivative");
    require_near(parameter[3],
                 (value(u + epsilon_second, v) - 2.0 * center
                  + value(u - epsilon_second, v))
                     / (epsilon_second * epsilon_second),
                 1.0e-5, "density parameter uu derivative");
    require_near(parameter[4],
                 (value(u + epsilon_second, v + epsilon_second)
                  - value(u + epsilon_second, v - epsilon_second)
                  - value(u - epsilon_second, v + epsilon_second)
                  + value(u - epsilon_second, v - epsilon_second))
                     / (4.0 * epsilon_second * epsilon_second),
                 1.0e-5, "density parameter uv derivative");
    require_near(parameter[5],
                 (value(u, v + epsilon_second) - 2.0 * center
                  + value(u, v - epsilon_second))
                     / (epsilon_second * epsilon_second),
                 1.0e-5, "density parameter vv derivative");

    const Eigen::Vector3d target = geometry.point
        + 0.017 * frame.tangent1 - 0.011 * frame.tangent2
        + 0.008 * frame.normal;
    const CauchyPolynomialWeights3D weights =
        cauchy_polynomial_weights_3d(
            geometry.point, frame, plan.graph_hessian, target);
    const NativeDensityC0Stencil3D composed =
        plan.compose_value_row(weights.w0);
    require_near(composed.dot(coefficients),
                 weights.apply_value_jet(direct),
                 3.0e-12,
                 "precomposed direct Cauchy coefficient row");

    const KnownNeumannValueGradient3D known{
        1.25, Eigen::Vector3d(0.4, -0.2, 0.9)};
    const NormalJet3D known_jet =
        known_neumann_jet_from_ambient_gradient_3d(known, frame);
    require_near(known_jet[0], known.value, 0.0,
                 "known Neumann value");
    require_near(known_jet[1], known.ambient_gradient.dot(frame.tangent1),
                 2.0e-15, "known Neumann s derivative");
    require_near(known_jet[2], known.ambient_gradient.dot(frame.tangent2),
                 2.0e-15, "known Neumann t derivative");
    const DirectCoefficientCauchyRow3D complete =
        compose_direct_coefficient_cauchy_row_3d(plan, weights, known_jet);
    require_near(complete.unknown_value_row.dot(coefficients)
                     + complete.known_normal_offset,
                 weights.apply_value_jet(direct)
                     + weights.apply_normal_jet(known_jet),
                 4.0e-12,
                 "complete direct coefficient Cauchy row");
}

void test_direct_normal_density_stencils()
{
    NativeNurbsDensitySpace3D density = make_normal_density();
    constexpr int patch = 0;
    constexpr double u = 0.37;
    constexpr double v = 0.41;
    const NativeSurfaceParameterJet3D geometry =
        native_surface_parameter_jet_3d(density, patch, u, v);
    const LocalOrthonormalFrame3D frame =
        make_local_orthonormal_frame_3d(
            geometry.normal, geometry.x_u + 0.23 * geometry.x_v);
    const DirectCoefficientNormalJetPlan3D plan =
        build_direct_coefficient_normal_jet_plan_3d(
            density, patch, u, v, frame);
    Eigen::VectorXd coefficients(density.c0_coefficient_count());
    for (Eigen::Index q = 0; q < coefficients.size(); ++q)
        coefficients[q] = std::cos(0.29 * static_cast<double>(q + 2));

    Eigen::Vector3d parameter;
    parameter <<
        density.c0_parameter_derivative_row(
            patch, u, v, 0, 0).dot(coefficients),
        density.c0_parameter_derivative_row(
            patch, u, v, 1, 0).dot(coefficients),
        density.c0_parameter_derivative_row(
            patch, u, v, 0, 1).dot(coefficients);
    Eigen::Matrix2d jacobian;
    jacobian << frame.tangent1.dot(geometry.x_u),
        frame.tangent1.dot(geometry.x_v),
        frame.tangent2.dot(geometry.x_u),
        frame.tangent2.dot(geometry.x_v);
    NormalJet3D expected;
    expected[0] = parameter[0];
    expected.tail<2>() = jacobian.inverse().transpose()
        * parameter.tail<2>();
    const NormalJet3D actual = plan.evaluate(coefficients);
    require((actual - expected).cwiseAbs().maxCoeff() < 2.0e-12,
            "sparse direct normal coefficient jet differs from dense chain rule");
    for (const NativeDensityC0Stencil3D& row : plan.cauchy_rows)
        require(row.count <= 16, "direct normal jet lost local cubic support");

    NormalCauchyWeightRow3D weights;
    weights << 0.17, -0.031, 0.052;
    const NativeDensityC0Stencil3D composed =
        plan.compose_normal_row(weights);
    require_near(composed.dot(coefficients), weights.dot(actual), 3.0e-12,
                 "precomposed direct normal Cauchy coefficient row");
}

void require_identical_stencil(const NativeDensityC0Stencil3D& actual,
                               const NativeDensityC0Stencil3D& expected,
                               const std::string& context)
{
    require(actual.count == expected.count, context + " support size");
    for (int entry = 0; entry < actual.count; ++entry) {
        const auto index = static_cast<std::size_t>(entry);
        require(actual.indices[index] == expected.indices[index],
                context + " support index");
        require_near(actual.weights[index], expected.weights[index], 0.0,
                     context + " coefficient weight");
    }
}

template <typename Plan>
void require_identical_plan(const Plan& actual,
                           const Plan& expected,
                           const std::string& context)
{
    require(actual.patch == expected.patch, context + " patch");
    require_near(actual.u, expected.u, 0.0, context + " parameter u");
    require_near(actual.v, expected.v, 0.0, context + " parameter v");
    require((actual.frame.normal - expected.frame.normal).norm() == 0.0
                && (actual.frame.tangent1 - expected.frame.tangent1).norm()
                       == 0.0
                && (actual.frame.tangent2 - expected.frame.tangent2).norm()
                       == 0.0,
            context + " frame");
    require_near(actual.graph_hessian.h11, expected.graph_hessian.h11, 0.0,
                 context + " graph h11");
    require_near(actual.graph_hessian.h12, expected.graph_hessian.h12, 0.0,
                 context + " graph h12");
    require_near(actual.graph_hessian.h22, expected.graph_hessian.h22, 0.0,
                 context + " graph h22");
    require_near(actual.diagnostics.parameter_to_tangent_determinant,
                 expected.diagnostics.parameter_to_tangent_determinant, 0.0,
                 context + " determinant diagnostic");
    require_near(actual.diagnostics.parameter_to_tangent_condition,
                 expected.diagnostics.parameter_to_tangent_condition, 0.0,
                 context + " condition diagnostic");
    require_near(actual.diagnostics.tangent_plane_residual,
                 expected.diagnostics.tangent_plane_residual, 0.0,
                 context + " tangent-plane diagnostic");
    for (std::size_t row = 0; row < actual.cauchy_rows.size(); ++row) {
        require_identical_stencil(actual.cauchy_rows[row],
                                  expected.cauchy_rows[row], context);
    }
}

void test_precomputed_geometry_overloads()
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    const RigidTransform3D transform = RigidTransform3D::from_axis_angle(
        {1.0, 2.0, 3.0}, 17.0 * pi / 180.0,
        {0.07, -0.07, 0.02}, {0.137, -0.083, 0.061});
    const std::array<std::array<double, 2>, 3> parameters{{
        {{0.0, 0.0}}, {{0.37, 0.41}}, {{1.0, 0.63}}}};
    for (const GeometryKind3D kind :
         {GeometryKind3D::LPrism, GeometryKind3D::HollowCylinder}) {
        const NativeNurbsSurface3D source = make_native_nurbs_surface_3d(kind);
        const std::array<NativeNurbsSurface3D, 2> surfaces{{
            source, transform_native_nurbs_surface_3d(source, transform)}};
        for (const NativeNurbsSurface3D& surface : surfaces) {
            for (const NativeDensityField3D field :
                 {NativeDensityField3D::ValueTrace,
                  NativeDensityField3D::NormalTrace}) {
                NativeNurbsDensityOptions3D options;
                options.field = field;
                options.reduction_backend =
                    NativeDensityReductionBackend3D::BaseOnly;
                options.coefficients_per_direction = 7;
                const NativeNurbsDensitySpace3D density(surface, options);
                for (int patch = 0; patch < density.patch_count(); ++patch) {
                    for (const auto& uv : parameters) {
                        const double u = uv[0];
                        const double v = uv[1];
                        const NativeSurfaceParameterJet3D geometry =
                            native_surface_parameter_jet_3d(
                                density, patch, u, v);
                        const LocalOrthonormalFrame3D frame =
                            make_local_orthonormal_frame_3d(
                                geometry.normal,
                                geometry.x_u + 0.23 * geometry.x_v);
                        const Eigen::Vector3d target = geometry.point
                            + 0.017 * frame.tangent1 - 0.011 * frame.tangent2
                            + 0.008 * frame.normal;
                        const std::string context = surface.name + " patch "
                            + std::to_string(patch)
                            + (field == NativeDensityField3D::ValueTrace
                                   ? " value geometry reuse"
                                   : " normal geometry reuse");
                        if (field == NativeDensityField3D::ValueTrace) {
                            const auto expected =
                                build_direct_coefficient_value_jet_plan_3d(
                                    density, patch, u, v, frame);
                            const auto actual =
                                build_direct_coefficient_value_jet_plan_3d(
                                    density, patch, u, v, geometry, frame);
                            require_identical_plan(actual, expected, context);
                            const auto weights = cauchy_polynomial_weights_3d(
                                geometry.point, frame,
                                expected.graph_hessian, target);
                            require_identical_stencil(
                                actual.compose_value_row(weights.w0),
                                expected.compose_value_row(weights.w0),
                                context + " composed row");
                        } else {
                            const auto expected =
                                build_direct_coefficient_normal_jet_plan_3d(
                                    density, patch, u, v, frame);
                            const auto actual =
                                build_direct_coefficient_normal_jet_plan_3d(
                                    density, patch, u, v, geometry, frame);
                            require_identical_plan(actual, expected, context);
                            const auto weights = cauchy_polynomial_weights_3d(
                                geometry.point, frame,
                                expected.graph_hessian, target);
                            require_identical_stencil(
                                actual.compose_normal_row(weights.w1),
                                expected.compose_normal_row(weights.w1),
                                context + " composed row");
                        }
                    }
                }
            }
        }
    }
}

} // namespace

int main()
{
    try {
        const NativeNurbsDensitySpace3D density = make_density();
        test_analytic_geometry_parameter_jet(density);
        test_second_order_inverse_chain_rule(density);
        test_known_dirichlet_analytic_jet(density);
        test_direct_density_stencils(density);
        test_direct_normal_density_stencils();
        test_precomputed_geometry_overloads();
        std::cout << "direct_coefficient_cauchy_3d_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "direct_coefficient_cauchy_3d_test: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
