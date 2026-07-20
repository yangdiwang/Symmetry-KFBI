#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>
#include <string>

namespace kfbim::geometry {

inline constexpr double kNurbsGeometryTolerance = 1.0e-12;
inline constexpr double kPi = 3.141592653589793238462643383279502884;

enum class NurbsProjectionStatus {
    NotStarted,
    ConvergedBracketSmall,
    ConvergedGradientSmall,
    ConvergedStepSmall,
    ConvergedLineSearchStepSmall,
    ConvergedCoordinateSearchStalled,
    DegenerateNormalSystem,
    NonFiniteStep,
    MaxIterationsReached
};

template <int Dim>
struct NurbsProjectionResult {
    using Vector = Eigen::Matrix<double, Dim, 1>;

    double parameter = 0.0;
    double u = 0.0;
    double v = 0.0;
    Vector point = Vector::Zero();
    double distance = 0.0;
    bool converged = false;
    int iterations = 0;
    NurbsProjectionStatus status = NurbsProjectionStatus::NotStarted;
    double residual_norm = 0.0;
    double gradient_norm = 0.0;
    double step_norm = 0.0;
    double normal_matrix_determinant = 0.0;
    int coarse_samples = 0;
    int line_search_iterations = 0;
    bool used_coordinate_search = false;
    std::string message;
};

using NurbsProjectionResult2D = NurbsProjectionResult<2>;
using NurbsProjectionResult3D = NurbsProjectionResult<3>;

template <int Dim>
inline void require_finite(const Eigen::Matrix<double, Dim, 1>& v,
                           const char*                         name)
{
    if (!v.allFinite())
        throw std::invalid_argument(std::string(name) + " must be finite");
}

inline void require_finite(double value, const char* name)
{
    if (!std::isfinite(value))
        throw std::invalid_argument(std::string(name) + " must be finite");
}

template <int Dim>
inline Eigen::Matrix<double, Dim, 1> normalize_safe(
    const Eigen::Matrix<double, Dim, 1>& v,
    double tolerance = kNurbsGeometryTolerance)
{
    require_finite<Dim>(v, "vector");
    const double norm = v.norm();
    if (!std::isfinite(norm) || norm <= tolerance)
        throw std::invalid_argument("cannot normalize a zero or near-zero vector");
    return v / norm;
}

inline double wrap_angle_positive(double theta)
{
    require_finite(theta, "theta");
    constexpr double two_pi = 2.0 * kPi;
    double wrapped = std::fmod(theta, two_pi);
    if (wrapped < 0.0)
        wrapped += two_pi;
    if (wrapped >= two_pi)
        wrapped = 0.0;
    return wrapped;
}

} // namespace kfbim::geometry
