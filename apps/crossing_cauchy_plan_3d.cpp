#include "crossing_cauchy_plan_3d.hpp"

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace kfbim::app3d {
namespace {

constexpr int kValueJetDimension = 6;
constexpr int kNormalJetDimension = 3;

void validate_center(const Eigen::Vector3d& center)
{
    if (!center.allFinite())
        throw std::invalid_argument("Cauchy crossing center must be finite");
}

void validate_graph_hessian(
    const TangentGraphHessian3D& graph_hessian)
{
    if (!graph_hessian.all_finite()) {
        throw std::invalid_argument(
            "Cauchy tangent-graph Hessian must be finite");
    }
}

Eigen::Vector3d fallback_tangent_axis(const Eigen::Vector3d& normal)
{
    const std::array<Eigen::Vector3d, 3> axes{{
        Eigen::Vector3d::UnitX(),
        Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitZ()}};
    int index = 0;
    double alignment = std::abs(normal.dot(axes[0]));
    for (int i = 1; i < static_cast<int>(axes.size()); ++i) {
        const double candidate = std::abs(normal.dot(axes[i]));
        if (candidate < alignment) {
            alignment = candidate;
            index = i;
        }
    }
    return axes[static_cast<std::size_t>(index)];
}

SurfacePointMatrix3D point_matrix(
    const std::vector<Eigen::Vector3d>& points)
{
    SurfacePointMatrix3D result(static_cast<int>(points.size()), 3);
    for (int row = 0; row < result.rows(); ++row)
        result.row(row) = points[static_cast<std::size_t>(row)].transpose();
    return result;
}

Eigen::MatrixXd taylor_design(
    const TangentCoordinateMatrix3D& coordinates,
    int dimension,
    double scale)
{
    Eigen::MatrixXd design(coordinates.rows(), dimension);
    for (int row = 0; row < coordinates.rows(); ++row) {
        const double s = coordinates(row, 0) / scale;
        const double t = coordinates(row, 1) / scale;
        design(row, 0) = 1.0;
        design(row, 1) = s;
        design(row, 2) = t;
        if (dimension == kValueJetDimension) {
            design(row, 3) = 0.5 * s * s;
            design(row, 4) = s * t;
            design(row, 5) = 0.5 * t * t;
        }
    }
    return design;
}

Eigen::MatrixXd physical_taylor_design(
    const TangentCoordinateMatrix3D& coordinates,
    int dimension)
{
    Eigen::MatrixXd design(coordinates.rows(), dimension);
    for (int row = 0; row < coordinates.rows(); ++row) {
        const double s = coordinates(row, 0);
        const double t = coordinates(row, 1);
        design(row, 0) = 1.0;
        design(row, 1) = s;
        design(row, 2) = t;
        if (dimension == kValueJetDimension) {
            design(row, 3) = 0.5 * s * s;
            design(row, 4) = s * t;
            design(row, 5) = 0.5 * t * t;
        }
    }
    return design;
}

double choose_coordinate_scale(
    const TangentCoordinateMatrix3D& coordinates,
    double requested_scale)
{
    if (std::isfinite(requested_scale) && requested_scale > 0.0)
        return requested_scale;
    if (!std::isfinite(requested_scale)) {
        throw std::invalid_argument(
            "jet-recovery coordinate scale must be finite");
    }

    double scale = 0.0;
    for (int row = 0; row < coordinates.rows(); ++row)
        scale = std::max(scale, coordinates.row(row).norm());
    if (!std::isfinite(scale) || !(scale > 0.0)) {
        throw std::invalid_argument(
            "jet recovery needs a positive tangent sample radius");
    }
    return scale;
}

Eigen::VectorXd normalized_sqrt_weights(
    int sample_count,
    const Eigen::VectorXd& requested_weights)
{
    if (requested_weights.size() == 0)
        return Eigen::VectorXd::Ones(sample_count);
    if (requested_weights.size() != sample_count) {
        throw std::invalid_argument(
            "jet-recovery weight count does not match sample count");
    }
    if (!requested_weights.allFinite()
        || (requested_weights.array() <= 0.0).any()) {
        throw std::invalid_argument(
            "jet-recovery weights must be finite and strictly positive");
    }
    const double maximum = requested_weights.maxCoeff();
    return (requested_weights.array() / maximum).sqrt().matrix();
}

JetRecoveryMatrix3D build_recovery(
    const TangentCoordinateMatrix3D& coordinates,
    int dimension,
    const char* jet_name,
    const JetRecoveryOptions3D& options)
{
    if (coordinates.rows() < dimension) {
        std::ostringstream message;
        message << jet_name << " jet recovery needs at least " << dimension
                << " samples, but received " << coordinates.rows();
        throw std::invalid_argument(message.str());
    }
    if (!coordinates.allFinite()) {
        throw std::invalid_argument(
            std::string(jet_name)
            + " jet-recovery tangent coordinates must be finite");
    }
    if (!std::isfinite(options.relative_singular_tolerance)
        || !(options.relative_singular_tolerance > 0.0)
        || !(options.relative_singular_tolerance < 1.0)) {
        throw std::invalid_argument(
            "jet-recovery relative singular tolerance must be in (0,1)");
    }

    const double scale = choose_coordinate_scale(
        coordinates, options.coordinate_scale);
    const Eigen::VectorXd sqrt_weights = normalized_sqrt_weights(
        coordinates.rows(), options.sample_weights);
    const Eigen::MatrixXd design = taylor_design(
        coordinates, dimension, scale);
    const Eigen::MatrixXd weighted_design =
        sqrt_weights.asDiagonal() * design;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        weighted_design, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular_values = svd.singularValues();
    if (singular_values.size() != dimension
        || !singular_values.allFinite()
        || !(singular_values[0] > 0.0)) {
        throw std::runtime_error(
            std::string(jet_name) + " jet recovery SVD failed");
    }

    const double cutoff = options.relative_singular_tolerance
        * singular_values[0];
    Eigen::VectorXd inverse = Eigen::VectorXd::Zero(dimension);
    int numerical_rank = 0;
    double smallest_retained = std::numeric_limits<double>::infinity();
    for (int i = 0; i < dimension; ++i) {
        if (singular_values[i] > cutoff) {
            inverse[i] = 1.0 / singular_values[i];
            smallest_retained = singular_values[i];
            ++numerical_rank;
        }
    }

    JetRecoveryMatrix3D result;
    JetRecoveryDiagnostics3D& diagnostics = result.diagnostics;
    diagnostics.sample_count = coordinates.rows();
    diagnostics.jet_dimension = dimension;
    diagnostics.numerical_rank = numerical_rank;
    diagnostics.coordinate_scale = scale;
    diagnostics.relative_singular_tolerance =
        options.relative_singular_tolerance;
    diagnostics.largest_singular_value = singular_values[0];
    diagnostics.smallest_singular_value =
        singular_values[singular_values.size() - 1];
    diagnostics.smallest_retained_singular_value =
        numerical_rank == 0 ? 0.0 : smallest_retained;
    diagnostics.condition_number = numerical_rank == dimension
        ? singular_values[0] / singular_values[dimension - 1]
        : std::numeric_limits<double>::infinity();
    diagnostics.retained_condition_number = numerical_rank == 0
        ? std::numeric_limits<double>::infinity()
        : singular_values[0] / smallest_retained;
    diagnostics.singular_values = singular_values;

    if (options.require_full_rank && numerical_rank != dimension) {
        std::ostringstream message;
        message << jet_name << " jet sample geometry is rank deficient (rank "
                << numerical_rank << " of " << dimension
                << ", sigma_min/sigma_max = "
                << singular_values[dimension - 1] / singular_values[0]
                << ')';
        throw std::runtime_error(message.str());
    }

    const Eigen::MatrixXd dimensionless_recovery =
        svd.matrixV() * inverse.asDiagonal() * svd.matrixU().transpose()
        * sqrt_weights.asDiagonal();
    Eigen::VectorXd derivative_scale = Eigen::VectorXd::Ones(dimension);
    derivative_scale[1] = 1.0 / scale;
    derivative_scale[2] = 1.0 / scale;
    if (dimension == kValueJetDimension) {
        const double inverse_scale_squared = 1.0 / (scale * scale);
        derivative_scale[3] = inverse_scale_squared;
        derivative_scale[4] = inverse_scale_squared;
        derivative_scale[5] = inverse_scale_squared;
    }
    result.matrix = derivative_scale.asDiagonal()
        * dimensionless_recovery;
    if (!result.matrix.allFinite()) {
        throw std::runtime_error(
            std::string(jet_name)
            + " jet recovery produced non-finite coefficients");
    }

    const Eigen::MatrixXd identity_error =
        result.matrix * physical_taylor_design(coordinates, dimension)
        - Eigen::MatrixXd::Identity(dimension, dimension);
    diagnostics.polynomial_reproduction_error =
        identity_error.cwiseAbs().maxCoeff();
    return result;
}

} // namespace

LocalOrthonormalFrame3D
make_local_orthonormal_frame_3d(const Eigen::Vector3d& normal)
{
    if (!normal.allFinite() || !(normal.norm() > 0.0)) {
        throw std::invalid_argument(
            "local frame normal must be finite and nonzero");
    }
    const Eigen::Vector3d unit_normal = normal.normalized();
    return make_local_orthonormal_frame_3d(
        unit_normal, fallback_tangent_axis(unit_normal));
}

LocalOrthonormalFrame3D
make_local_orthonormal_frame_3d(
    const Eigen::Vector3d& normal,
    const Eigen::Vector3d& tangent1_hint)
{
    if (!normal.allFinite() || !(normal.norm() > 0.0)) {
        throw std::invalid_argument(
            "local frame normal must be finite and nonzero");
    }
    if (!tangent1_hint.allFinite()) {
        throw std::invalid_argument("local frame tangent hint must be finite");
    }

    LocalOrthonormalFrame3D result;
    result.normal = normal.normalized();
    Eigen::Vector3d tangent = tangent1_hint
        - tangent1_hint.dot(result.normal) * result.normal;
    const double hint_reference = std::max(1.0, tangent1_hint.norm());
    if (!(tangent.norm() > 64.0 * std::numeric_limits<double>::epsilon()
                               * hint_reference)) {
        const Eigen::Vector3d fallback = fallback_tangent_axis(result.normal);
        tangent = fallback - fallback.dot(result.normal) * result.normal;
    }
    result.tangent1 = tangent.normalized();
    result.tangent2 = result.normal.cross(result.tangent1).normalized();
    validate_local_orthonormal_frame_3d(result);
    return result;
}

bool is_local_orthonormal_frame_3d(
    const LocalOrthonormalFrame3D& frame,
    double tolerance)
{
    if (!std::isfinite(tolerance) || tolerance < 0.0
        || !frame.tangent1.allFinite() || !frame.tangent2.allFinite()
        || !frame.normal.allFinite()) {
        return false;
    }
    const double norm_error = std::max({
        std::abs(frame.tangent1.norm() - 1.0),
        std::abs(frame.tangent2.norm() - 1.0),
        std::abs(frame.normal.norm() - 1.0)});
    const double orthogonality_error = std::max({
        std::abs(frame.tangent1.dot(frame.tangent2)),
        std::abs(frame.tangent1.dot(frame.normal)),
        std::abs(frame.tangent2.dot(frame.normal))});
    const double orientation =
        frame.tangent1.cross(frame.tangent2).dot(frame.normal);
    return norm_error <= tolerance && orthogonality_error <= tolerance
        && std::abs(orientation - 1.0) <= tolerance;
}

void validate_local_orthonormal_frame_3d(
    const LocalOrthonormalFrame3D& frame,
    double tolerance)
{
    if (!is_local_orthonormal_frame_3d(frame, tolerance)) {
        throw std::invalid_argument(
            "Cauchy frame must be finite, right-handed and orthonormal");
    }
}

Eigen::Matrix2d TangentGraphHessian3D::matrix() const
{
    Eigen::Matrix2d result;
    result << h11, h12, h12, h22;
    return result;
}

bool TangentGraphHessian3D::all_finite() const
{
    return std::isfinite(h11) && std::isfinite(h12) && std::isfinite(h22);
}

double CauchyPolynomialWeights3D::apply_value_jet(
    const ValueJet3D& jet) const
{
    return w0.dot(jet);
}

double CauchyPolynomialWeights3D::apply_normal_jet(
    const NormalJet3D& jet) const
{
    return w1.dot(jet);
}

CauchyPolynomialWeights3D pweights_general_3d(
    const Eigen::Vector3d& displacement,
    const LocalOrthonormalFrame3D& frame,
    const TangentGraphHessian3D& graph_hessian)
{
    if (!displacement.allFinite()) {
        throw std::invalid_argument(
            "Cauchy target displacement must be finite");
    }
    validate_local_orthonormal_frame_3d(frame);
    validate_graph_hessian(graph_hessian);

    CauchyPolynomialWeights3D result;
    const double s = displacement.dot(frame.tangent1);
    const double t = displacement.dot(frame.tangent2);
    const double r = displacement.dot(frame.normal);
    const double h11 = graph_hessian.h11;
    const double h12 = graph_hessian.h12;
    const double h22 = graph_hessian.h22;
    result.local_displacement = {s, t, r};

    result.w0 << 1.0,
        s + h11 * s * r + h12 * t * r,
        t + h12 * s * r + h22 * t * r,
        0.5 * (s * s - r * r),
        s * t,
        0.5 * (t * t - r * r);
    result.w1 <<
        r - 0.5 * h11 * s * s - h12 * s * t - 0.5 * h22 * t * t
            + 0.5 * (h11 + h22) * r * r,
        s * r,
        t * r;
    return result;
}

CauchyPolynomialWeights3D cauchy_polynomial_weights_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const TangentGraphHessian3D& graph_hessian,
    const Eigen::Vector3d& target)
{
    validate_center(center);
    if (!target.allFinite())
        throw std::invalid_argument("Cauchy target must be finite");
    return pweights_general_3d(
        target - center, frame, graph_hessian);
}

bool JetRecoveryDiagnostics3D::full_rank() const noexcept
{
    return jet_dimension > 0 && numerical_rank == jet_dimension;
}

Eigen::VectorXd JetRecoveryMatrix3D::recover(
    const Eigen::VectorXd& sample_values) const
{
    if (sample_values.size() != matrix.cols()) {
        throw std::invalid_argument(
            "jet-recovery sample value count does not match recovery matrix");
    }
    if (!sample_values.allFinite()) {
        throw std::invalid_argument(
            "jet-recovery sample values must be finite");
    }
    return matrix * sample_values;
}

TangentCoordinateMatrix3D centered_tangent_coordinates_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const SurfacePointMatrix3D& sample_points)
{
    validate_center(center);
    validate_local_orthonormal_frame_3d(frame);
    if (!sample_points.allFinite()) {
        throw std::invalid_argument(
            "surface sample points must be finite");
    }

    TangentCoordinateMatrix3D result(sample_points.rows(), 2);
    for (int row = 0; row < sample_points.rows(); ++row) {
        const Eigen::Vector3d displacement =
            sample_points.row(row).transpose() - center;
        result(row, 0) = displacement.dot(frame.tangent1);
        result(row, 1) = displacement.dot(frame.tangent2);
    }
    return result;
}

TangentCoordinateMatrix3D centered_tangent_coordinates_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const std::vector<Eigen::Vector3d>& sample_points)
{
    return centered_tangent_coordinates_3d(
        center, frame, point_matrix(sample_points));
}

JetRecoveryMatrix3D build_value_jet_recovery_3d(
    const TangentCoordinateMatrix3D& centered_tangent_coordinates,
    const JetRecoveryOptions3D& options)
{
    return build_recovery(
        centered_tangent_coordinates,
        kValueJetDimension,
        "value",
        options);
}

JetRecoveryMatrix3D build_value_jet_recovery_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const SurfacePointMatrix3D& sample_points,
    const JetRecoveryOptions3D& options)
{
    return build_value_jet_recovery_3d(
        centered_tangent_coordinates_3d(center, frame, sample_points),
        options);
}

JetRecoveryMatrix3D build_value_jet_recovery_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const std::vector<Eigen::Vector3d>& sample_points,
    const JetRecoveryOptions3D& options)
{
    return build_value_jet_recovery_3d(
        centered_tangent_coordinates_3d(center, frame, sample_points),
        options);
}

JetRecoveryMatrix3D build_normal_jet_recovery_3d(
    const TangentCoordinateMatrix3D& centered_tangent_coordinates,
    const JetRecoveryOptions3D& options)
{
    return build_recovery(
        centered_tangent_coordinates,
        kNormalJetDimension,
        "normal",
        options);
}

JetRecoveryMatrix3D build_normal_jet_recovery_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const SurfacePointMatrix3D& sample_points,
    const JetRecoveryOptions3D& options)
{
    return build_normal_jet_recovery_3d(
        centered_tangent_coordinates_3d(center, frame, sample_points),
        options);
}

JetRecoveryMatrix3D build_normal_jet_recovery_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const std::vector<Eigen::Vector3d>& sample_points,
    const JetRecoveryOptions3D& options)
{
    return build_normal_jet_recovery_3d(
        centered_tangent_coordinates_3d(center, frame, sample_points),
        options);
}

CauchyPolynomialWeights3D CrossingCauchyPlan3D::weights_at(
    const Eigen::Vector3d& target) const
{
    return cauchy_polynomial_weights_3d(
        center, frame, graph_hessian, target);
}

CrossingCauchyPlan3D build_crossing_cauchy_plan_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const TangentGraphHessian3D& graph_hessian,
    const SurfacePointMatrix3D& value_sample_points,
    const SurfacePointMatrix3D& normal_sample_points,
    const JetRecoveryOptions3D& value_options,
    const JetRecoveryOptions3D& normal_options)
{
    validate_center(center);
    validate_local_orthonormal_frame_3d(frame);
    validate_graph_hessian(graph_hessian);

    CrossingCauchyPlan3D result;
    result.center = center;
    result.frame = frame;
    result.graph_hessian = graph_hessian;
    result.value_jet_recovery = build_value_jet_recovery_3d(
        center, frame, value_sample_points, value_options);
    result.normal_jet_recovery = build_normal_jet_recovery_3d(
        center, frame, normal_sample_points, normal_options);
    return result;
}

CrossingCauchyPlan3D build_crossing_cauchy_plan_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const TangentGraphHessian3D& graph_hessian,
    const std::vector<Eigen::Vector3d>& value_sample_points,
    const std::vector<Eigen::Vector3d>& normal_sample_points,
    const JetRecoveryOptions3D& value_options,
    const JetRecoveryOptions3D& normal_options)
{
    return build_crossing_cauchy_plan_3d(
        center,
        frame,
        graph_hessian,
        point_matrix(value_sample_points),
        point_matrix(normal_sample_points),
        value_options,
        normal_options);
}

} // namespace kfbim::app3d
