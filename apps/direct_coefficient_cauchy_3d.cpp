#include "direct_coefficient_cauchy_3d.hpp"

#include <Eigen/LU>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfbim::app3d {
namespace {

constexpr int value_jet_size = 6;

void require_normalized_parameter(double value, const char* name)
{
    if (!std::isfinite(value) || value < -1.0e-12 || value > 1.0 + 1.0e-12)
        throw std::out_of_range(std::string(name) + " lies outside [0,1]");
}

NativeDensityC0Stencil3D linear_combination(
    const std::array<NativeDensityC0Stencil3D, value_jet_size>& rows,
    const Eigen::Matrix<double, 1, value_jet_size>& weights)
{
    NativeDensityC0Stencil3D result;
    for (int derivative = 0; derivative < value_jet_size; ++derivative) {
        const double factor = weights[derivative];
        if (factor == 0.0)
            continue;
        const NativeDensityC0Stencil3D& source =
            rows[static_cast<std::size_t>(derivative)];
        for (int q = 0; q < source.count; ++q) {
            const int index = source.indices[static_cast<std::size_t>(q)];
            int slot = 0;
            while (slot < result.count
                   && result.indices[static_cast<std::size_t>(slot)] != index) {
                ++slot;
            }
            if (slot == result.count) {
                if (result.count
                    >= static_cast<int>(result.indices.size())) {
                    throw std::runtime_error(
                        "direct coefficient Cauchy row exceeded cubic support");
                }
                result.indices[static_cast<std::size_t>(slot)] = index;
                ++result.count;
            }
            result.weights[static_cast<std::size_t>(slot)] +=
                factor * source.weights[static_cast<std::size_t>(q)];
        }
    }
    return result;
}

NativeDensityC0Stencil3D linear_combination(
    const std::array<NativeDensityC0Stencil3D, 3>& rows,
    const NormalCauchyWeightRow3D& weights)
{
    NativeDensityC0Stencil3D result;
    for (int derivative = 0; derivative < 3; ++derivative) {
        const double factor = weights[derivative];
        if (factor == 0.0)
            continue;
        const NativeDensityC0Stencil3D& source =
            rows[static_cast<std::size_t>(derivative)];
        for (int q = 0; q < source.count; ++q) {
            const int index = source.indices[static_cast<std::size_t>(q)];
            int slot = 0;
            while (slot < result.count
                   && result.indices[static_cast<std::size_t>(slot)] != index) {
                ++slot;
            }
            if (slot == result.count) {
                if (result.count
                    >= static_cast<int>(result.indices.size())) {
                    throw std::runtime_error(
                        "direct coefficient normal row exceeded cubic support");
                }
                result.indices[static_cast<std::size_t>(slot)] = index;
                ++result.count;
            }
            result.weights[static_cast<std::size_t>(slot)] +=
                factor * source.weights[static_cast<std::size_t>(q)];
        }
    }
    return result;
}

Eigen::Matrix2d parameter_to_tangent_jacobian(
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame)
{
    Eigen::Matrix2d jacobian;
    jacobian << frame.tangent1.dot(geometry.x_u),
        frame.tangent1.dot(geometry.x_v),
        frame.tangent2.dot(geometry.x_u),
        frame.tangent2.dot(geometry.x_v);
    if (!jacobian.allFinite())
        throw std::runtime_error("parameter-to-tangent Jacobian is not finite");
    const Eigen::JacobiSVD<Eigen::Matrix2d> svd(jacobian);
    const Eigen::Vector2d singular = svd.singularValues();
    if (!(singular[0] > 0.0)
        || !(singular[1] > 128.0 * std::numeric_limits<double>::epsilon()
                                  * singular[0])) {
        throw std::runtime_error("parameter-to-tangent Jacobian is singular");
    }
    return jacobian;
}

std::array<Eigen::Vector3d, 3> rational_surface_numerator_derivatives(
    const geometry3d::NurbsSurfacePatch3D& patch,
    double u,
    double v,
    int derivative_u,
    int derivative_v,
    double& denominator)
{
    const auto& basis_u = patch.basis_u();
    const auto& basis_v = patch.basis_v();
    const auto active_u = basis_u.active_basis_indices(u);
    const auto active_v = basis_v.active_basis_indices(v);
    auto basis_values = [](const geometry::NurbsBasis1D& basis,
                           double parameter,
                           int derivative) {
        if (derivative == 0)
            return basis.evaluate_nonzero(parameter);
        if (derivative == 1)
            return basis.evaluate_nonzero_first_derivatives(parameter);
        return basis.evaluate_nonzero_second_derivatives(parameter);
    };
    const std::vector<double> values_u =
        basis_values(basis_u, u, derivative_u);
    const std::vector<double> values_v =
        basis_values(basis_v, v, derivative_v);

    Eigen::Vector3d numerator = Eigen::Vector3d::Zero();
    denominator = 0.0;
    const auto& control = patch.control_net();
    const auto& weights = patch.weights();
    for (std::size_t a = 0; a < active_u.size(); ++a) {
        const int i = active_u[a];
        for (std::size_t b = 0; b < active_v.size(); ++b) {
            const int j = active_v[b];
            const double basis = values_u[a] * values_v[b]
                               * weights[static_cast<std::size_t>(i)]
                                        [static_cast<std::size_t>(j)];
            numerator += basis
                * control[static_cast<std::size_t>(i)]
                         [static_cast<std::size_t>(j)];
            denominator += basis;
        }
    }
    return {numerator, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
}

} // namespace

NativeSurfaceParameterJet3D native_surface_parameter_jet_3d(
    const NativeNurbsDensitySpace3D& density,
    int patch_index,
    double u,
    double v)
{
    require_normalized_parameter(u, "normalized u parameter");
    require_normalized_parameter(v, "normalized v parameter");
    if (patch_index < 0 || patch_index >= density.patch_count())
        throw std::out_of_range("surface parameter jet patch is out of range");
    u = std::clamp(u, 0.0, 1.0);
    v = std::clamp(v, 0.0, 1.0);

    const auto& patch = density.surface().patches[
        static_cast<std::size_t>(patch_index)];
    const double scale_u = patch.domain_end_u() - patch.domain_start_u();
    const double scale_v = patch.domain_end_v() - patch.domain_start_v();
    if (!(scale_u > 0.0) || !(scale_v > 0.0))
        throw std::runtime_error("surface parameter jet has a degenerate domain");
    const double native_u = patch.domain_start_u() + scale_u * u;
    const double native_v = patch.domain_start_v() + scale_v * v;

    Eigen::Vector3d numerator[3][3];
    double denominator[3][3]{};
    for (int du = 0; du <= 2; ++du) {
        for (int dv = 0; dv <= 2 - du; ++dv) {
            numerator[du][dv] = rational_surface_numerator_derivatives(
                patch, native_u, native_v, du, dv,
                denominator[du][dv])[0];
        }
    }
    const double weight = denominator[0][0];
    if (!(std::abs(weight) > 64.0 * std::numeric_limits<double>::epsilon())
        || !std::isfinite(weight)) {
        throw std::runtime_error("NURBS surface homogeneous weight is singular");
    }

    NativeSurfaceParameterJet3D result;
    result.point = numerator[0][0] / weight;
    const Eigen::Vector3d native_xu =
        (numerator[1][0] - denominator[1][0] * result.point) / weight;
    const Eigen::Vector3d native_xv =
        (numerator[0][1] - denominator[0][1] * result.point) / weight;
    const Eigen::Vector3d native_xuu =
        (numerator[2][0] - denominator[2][0] * result.point
         - 2.0 * denominator[1][0] * native_xu) / weight;
    const Eigen::Vector3d native_xuv =
        (numerator[1][1] - denominator[1][1] * result.point
         - denominator[1][0] * native_xv
         - denominator[0][1] * native_xu) / weight;
    const Eigen::Vector3d native_xvv =
        (numerator[0][2] - denominator[0][2] * result.point
         - 2.0 * denominator[0][1] * native_xv) / weight;
    result.x_u = scale_u * native_xu;
    result.x_v = scale_v * native_xv;
    result.x_uu = scale_u * scale_u * native_xuu;
    result.x_uv = scale_u * scale_v * native_xuv;
    result.x_vv = scale_v * scale_v * native_xvv;
    const Eigen::Vector3d cross = result.x_u.cross(result.x_v);
    if (!(cross.norm() > 0.0) || !cross.allFinite())
        throw std::runtime_error("surface parameter jet is degenerate");
    result.normal = cross.normalized();
    if (!result.point.allFinite() || !result.x_u.allFinite()
        || !result.x_v.allFinite() || !result.x_uu.allFinite()
        || !result.x_uv.allFinite() || !result.x_vv.allFinite()) {
        throw std::runtime_error("surface parameter jet is not finite");
    }
    return result;
}

NativeSurfaceNormalParameterJet3D surface_normal_parameter_jet_3d(
    const NativeSurfaceParameterJet3D& geometry)
{
    const Eigen::Vector3d cross = geometry.x_u.cross(geometry.x_v);
    const double area = cross.norm();
    if (!(area > 0.0) || !std::isfinite(area))
        throw std::runtime_error("surface normal jet has degenerate tangents");
    NativeSurfaceNormalParameterJet3D result;
    result.normal = cross / area;
    const Eigen::Matrix3d tangent_projection = Eigen::Matrix3d::Identity()
        - result.normal * result.normal.transpose();
    const Eigen::Vector3d cross_u = geometry.x_uu.cross(geometry.x_v)
        + geometry.x_u.cross(geometry.x_uv);
    const Eigen::Vector3d cross_v = geometry.x_uv.cross(geometry.x_v)
        + geometry.x_u.cross(geometry.x_vv);
    result.normal_u = tangent_projection * cross_u / area;
    result.normal_v = tangent_projection * cross_v / area;
    if (!result.normal_u.allFinite() || !result.normal_v.allFinite())
        throw std::runtime_error("surface normal parameter jet is not finite");
    return result;
}

Eigen::Vector2d parameter_gradient_to_tangent_gradient_3d(
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame,
    const Eigen::Vector2d& parameter_gradient)
{
    validate_local_orthonormal_frame_3d(frame);
    if (!parameter_gradient.allFinite())
        throw std::invalid_argument("parameter gradient is not finite");
    return parameter_to_tangent_jacobian(geometry, frame)
        .inverse().transpose() * parameter_gradient;
}

Eigen::Matrix<double, 6, 6> parameter_to_cauchy_value_jet_matrix_3d(
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame)
{
    validate_local_orthonormal_frame_3d(frame);
    const Eigen::Matrix2d jacobian =
        parameter_to_tangent_jacobian(geometry, frame);
    const Eigen::Matrix2d inverse = jacobian.inverse();
    Eigen::Matrix2d hessian_s;
    hessian_s << frame.tangent1.dot(geometry.x_uu),
        frame.tangent1.dot(geometry.x_uv),
        frame.tangent1.dot(geometry.x_uv),
        frame.tangent1.dot(geometry.x_vv);
    Eigen::Matrix2d hessian_t;
    hessian_t << frame.tangent2.dot(geometry.x_uu),
        frame.tangent2.dot(geometry.x_uv),
        frame.tangent2.dot(geometry.x_uv),
        frame.tangent2.dot(geometry.x_vv);

    Eigen::Matrix<double, 6, 6> transform =
        Eigen::Matrix<double, 6, 6>::Zero();
    for (int column = 0; column < 6; ++column) {
        const double value = column == 0 ? 1.0 : 0.0;
        Eigen::Vector2d gradient_q = Eigen::Vector2d::Zero();
        if (column == 1)
            gradient_q.x() = 1.0;
        if (column == 2)
            gradient_q.y() = 1.0;
        Eigen::Matrix2d hessian_q = Eigen::Matrix2d::Zero();
        if (column == 3)
            hessian_q(0, 0) = 1.0;
        if (column == 4)
            hessian_q(0, 1) = hessian_q(1, 0) = 1.0;
        if (column == 5)
            hessian_q(1, 1) = 1.0;

        const Eigen::Vector2d gradient_y = inverse.transpose() * gradient_q;
        const Eigen::Matrix2d hessian_y = inverse.transpose()
            * (hessian_q - gradient_y.x() * hessian_s
                         - gradient_y.y() * hessian_t)
            * inverse;
        transform(0, column) = value;
        transform(1, column) = gradient_y.x();
        transform(2, column) = gradient_y.y();
        transform(3, column) = hessian_y(0, 0);
        transform(4, column) = hessian_y(0, 1);
        transform(5, column) = hessian_y(1, 1);
    }
    if (!transform.allFinite())
        throw std::runtime_error("parameter-to-Cauchy jet map is not finite");
    return transform;
}

TangentGraphHessian3D tangent_graph_hessian_from_parameter_jet_3d(
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame)
{
    validate_local_orthonormal_frame_3d(frame);
    const Eigen::Matrix2d inverse =
        parameter_to_tangent_jacobian(geometry, frame).inverse();
    Eigen::Matrix2d normal_parameter_hessian;
    normal_parameter_hessian << frame.normal.dot(geometry.x_uu),
        frame.normal.dot(geometry.x_uv),
        frame.normal.dot(geometry.x_uv),
        frame.normal.dot(geometry.x_vv);
    const Eigen::Matrix2d graph_hessian = inverse.transpose()
        * normal_parameter_hessian * inverse;
    TangentGraphHessian3D result;
    result.h11 = graph_hessian(0, 0);
    result.h12 = graph_hessian(0, 1);
    result.h22 = graph_hessian(1, 1);
    if (!result.all_finite())
        throw std::runtime_error("analytic tangent graph Hessian is not finite");
    return result;
}

ValueJet3D DirectCoefficientValueJetPlan3D::evaluate(
    const Eigen::Ref<const Eigen::VectorXd>& c0_coefficients) const
{
    ValueJet3D result;
    for (int row = 0; row < result.size(); ++row)
        result[row] = cauchy_rows[static_cast<std::size_t>(row)]
                          .dot(c0_coefficients);
    if (!result.allFinite())
        throw std::runtime_error("direct coefficient value jet is not finite");
    return result;
}

NativeDensityC0Stencil3D DirectCoefficientValueJetPlan3D::compose_value_row(
    const ValueCauchyWeightRow3D& cauchy_weights) const
{
    if (!cauchy_weights.allFinite())
        throw std::invalid_argument("Cauchy value weights are not finite");
    return linear_combination(cauchy_rows, cauchy_weights);
}

DirectCoefficientValueJetPlan3D
build_direct_coefficient_value_jet_plan_3d(
    const NativeNurbsDensitySpace3D& density,
    int patch,
    double u,
    double v,
    const LocalOrthonormalFrame3D& frame)
{
    const NativeSurfaceParameterJet3D geometry =
        native_surface_parameter_jet_3d(density, patch, u, v);
    return build_direct_coefficient_value_jet_plan_3d(
        density, patch, u, v, geometry, frame);
}

DirectCoefficientValueJetPlan3D
build_direct_coefficient_value_jet_plan_3d(
    const NativeNurbsDensitySpace3D& density,
    int patch,
    double u,
    double v,
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame)
{
    const Eigen::Matrix<double, 6, 6> transform =
        parameter_to_cauchy_value_jet_matrix_3d(geometry, frame);
    const std::array<NativeDensityC0Stencil3D, 6> parameter_rows =
        density.c0_parameter_jet_stencils(patch, u, v);

    DirectCoefficientValueJetPlan3D result;
    result.patch = patch;
    result.u = u;
    result.v = v;
    result.frame = frame;
    result.graph_hessian =
        tangent_graph_hessian_from_parameter_jet_3d(geometry, frame);
    for (int row = 0; row < 6; ++row)
        result.cauchy_rows[static_cast<std::size_t>(row)] =
            linear_combination(parameter_rows, transform.row(row));

    const Eigen::Matrix2d jacobian =
        parameter_to_tangent_jacobian(geometry, frame);
    const Eigen::JacobiSVD<Eigen::Matrix2d> svd(jacobian);
    result.diagnostics.parameter_to_tangent_determinant =
        jacobian.determinant();
    result.diagnostics.parameter_to_tangent_condition =
        svd.singularValues()[0] / svd.singularValues()[1];
    result.diagnostics.tangent_plane_residual = std::max(
        std::abs(frame.normal.dot(geometry.x_u)),
        std::abs(frame.normal.dot(geometry.x_v)));
    return result;
}

NormalJet3D DirectCoefficientNormalJetPlan3D::evaluate(
    const Eigen::Ref<const Eigen::VectorXd>& c0_coefficients) const
{
    NormalJet3D result;
    for (int row = 0; row < result.size(); ++row)
        result[row] = cauchy_rows[static_cast<std::size_t>(row)]
                          .dot(c0_coefficients);
    if (!result.allFinite())
        throw std::runtime_error("direct coefficient normal jet is not finite");
    return result;
}

NativeDensityC0Stencil3D
DirectCoefficientNormalJetPlan3D::compose_normal_row(
    const NormalCauchyWeightRow3D& cauchy_weights) const
{
    if (!cauchy_weights.allFinite())
        throw std::invalid_argument("Cauchy normal weights are not finite");
    return linear_combination(cauchy_rows, cauchy_weights);
}

DirectCoefficientNormalJetPlan3D
build_direct_coefficient_normal_jet_plan_3d(
    const NativeNurbsDensitySpace3D& density,
    int patch,
    double u,
    double v,
    const LocalOrthonormalFrame3D& frame)
{
    const NativeSurfaceParameterJet3D geometry =
        native_surface_parameter_jet_3d(density, patch, u, v);
    return build_direct_coefficient_normal_jet_plan_3d(
        density, patch, u, v, geometry, frame);
}

DirectCoefficientNormalJetPlan3D
build_direct_coefficient_normal_jet_plan_3d(
    const NativeNurbsDensitySpace3D& density,
    int patch,
    double u,
    double v,
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame)
{
    const Eigen::Matrix2d jacobian =
        parameter_to_tangent_jacobian(geometry, frame);
    const Eigen::Matrix2d inverse = jacobian.inverse();
    Eigen::Matrix3d transform = Eigen::Matrix3d::Zero();
    transform(0, 0) = 1.0;
    transform.block<2, 2>(1, 1) = inverse.transpose();

    const std::array<std::array<int, 2>, 3> derivatives{{
        {{0, 0}}, {{1, 0}}, {{0, 1}}}};
    std::array<NativeDensityC0Stencil3D, 3> parameter_rows;
    for (int q = 0; q < 3; ++q) {
        parameter_rows[static_cast<std::size_t>(q)] =
            density.c0_parameter_derivative_stencil(
                patch, u, v,
                derivatives[static_cast<std::size_t>(q)][0],
                derivatives[static_cast<std::size_t>(q)][1]);
    }

    DirectCoefficientNormalJetPlan3D result;
    result.patch = patch;
    result.u = u;
    result.v = v;
    result.frame = frame;
    result.graph_hessian =
        tangent_graph_hessian_from_parameter_jet_3d(geometry, frame);
    for (int row = 0; row < 3; ++row) {
        const NormalCauchyWeightRow3D weights = transform.row(row);
        result.cauchy_rows[static_cast<std::size_t>(row)] =
            linear_combination(parameter_rows, weights);
    }
    const Eigen::JacobiSVD<Eigen::Matrix2d> svd(jacobian);
    result.diagnostics.parameter_to_tangent_determinant =
        jacobian.determinant();
    result.diagnostics.parameter_to_tangent_condition =
        svd.singularValues()[0] / svd.singularValues()[1];
    result.diagnostics.tangent_plane_residual = std::max(
        std::abs(frame.normal.dot(geometry.x_u)),
        std::abs(frame.normal.dot(geometry.x_v)));
    return result;
}

ValueJet3D known_dirichlet_jet_from_ambient_derivatives_3d(
    const KnownDirichletValueGradientHessian3D& data,
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame)
{
    validate_local_orthonormal_frame_3d(frame);
    if (!std::isfinite(data.value) || !data.ambient_gradient.allFinite()
        || !data.ambient_hessian.allFinite()) {
        throw std::invalid_argument(
            "known Dirichlet value/gradient/Hessian is not finite");
    }
    ValueJet3D parameter_jet;
    parameter_jet << data.value,
        data.ambient_gradient.dot(geometry.x_u),
        data.ambient_gradient.dot(geometry.x_v),
        geometry.x_u.dot(data.ambient_hessian * geometry.x_u)
            + data.ambient_gradient.dot(geometry.x_uu),
        geometry.x_u.dot(data.ambient_hessian * geometry.x_v)
            + data.ambient_gradient.dot(geometry.x_uv),
        geometry.x_v.dot(data.ambient_hessian * geometry.x_v)
            + data.ambient_gradient.dot(geometry.x_vv);
    if (!parameter_jet.allFinite()) {
        throw std::invalid_argument(
            "known Dirichlet surface parameter jet is not finite");
    }
    return parameter_to_cauchy_value_jet_matrix_3d(geometry, frame)
        * parameter_jet;
}

ValueJet3D evaluate_known_dirichlet_jet_3d(
    const KnownDirichletValueGradientHessianCallback3D& callback,
    int patch,
    double u,
    double v,
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame)
{
    if (!callback)
        throw std::invalid_argument("known Dirichlet jet callback is empty");
    return known_dirichlet_jet_from_ambient_derivatives_3d(
        callback(patch, u, v, geometry.point, geometry.normal),
        geometry, frame);
}

NormalJet3D known_neumann_jet_from_ambient_gradient_3d(
    const KnownNeumannValueGradient3D& data,
    const LocalOrthonormalFrame3D& frame)
{
    validate_local_orthonormal_frame_3d(frame);
    if (!std::isfinite(data.value) || !data.ambient_gradient.allFinite())
        throw std::invalid_argument("known Neumann value/gradient is not finite");
    NormalJet3D result;
    result << data.value,
        data.ambient_gradient.dot(frame.tangent1),
        data.ambient_gradient.dot(frame.tangent2);
    return result;
}

NormalJet3D evaluate_known_neumann_jet_3d(
    const KnownNeumannValueGradientCallback3D& callback,
    int patch,
    double u,
    double v,
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame)
{
    if (!callback)
        throw std::invalid_argument("known Neumann jet callback is empty");
    return known_neumann_jet_from_ambient_gradient_3d(
        callback(patch, u, v, geometry.point, geometry.normal), frame);
}

DirectCoefficientCauchyRow3D compose_direct_coefficient_cauchy_row_3d(
    const DirectCoefficientValueJetPlan3D& value_plan,
    const CauchyPolynomialWeights3D& weights,
    const NormalJet3D& known_normal_jet)
{
    if (!known_normal_jet.allFinite())
        throw std::invalid_argument("known normal jet is not finite");
    DirectCoefficientCauchyRow3D result;
    result.unknown_value_row = value_plan.compose_value_row(weights.w0);
    result.known_normal_offset = weights.apply_normal_jet(known_normal_jet);
    return result;
}

} // namespace kfbim::app3d
