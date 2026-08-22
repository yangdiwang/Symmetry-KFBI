#include "native_density_resolution_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfbim::app3d {

namespace {

constexpr std::array<double, 5> kGaussNodes{{
    -0.906179845938663992797626878299,
    -0.538469310105683091036314420700,
     0.0,
     0.538469310105683091036314420700,
     0.906179845938663992797626878299}};

constexpr std::array<double, 5> kGaussWeights{{
    0.236926885056189087514264040720,
    0.478628670499366468041291514836,
    0.568888888888888888888888888889,
    0.478628670499366468041291514836,
    0.236926885056189087514264040720}};

using ParameterSpan = std::pair<double, double>;

std::vector<ParameterSpan> nonempty_spans(
    const geometry::NurbsBasis1D& basis)
{
    std::vector<ParameterSpan> spans;
    const std::vector<double>& knots = basis.knots();
    const double domain_start = basis.domain_start();
    const double domain_end = basis.domain_end();
    spans.reserve(knots.size());
    for (std::size_t i = 0; i + 1 < knots.size(); ++i) {
        const double begin = std::max(domain_start, knots[i]);
        const double end = std::min(domain_end, knots[i + 1]);
        if (end > begin)
            spans.emplace_back(begin, end);
    }
    if (spans.empty())
        throw std::runtime_error("Native NURBS patch has no non-empty knot span");
    return spans;
}

double patch_orientation_integral(
    const geometry3d::NurbsSurfacePatch3D& patch)
{
    const std::vector<ParameterSpan> u_spans = nonempty_spans(patch.basis_u());
    const std::vector<ParameterSpan> v_spans = nonempty_spans(patch.basis_v());
    double integral = 0.0;
    for (const ParameterSpan& u_span : u_spans) {
        const double u_midpoint = 0.5 * (u_span.first + u_span.second);
        const double u_half_width = 0.5 * (u_span.second - u_span.first);
        for (const ParameterSpan& v_span : v_spans) {
            const double v_midpoint = 0.5 * (v_span.first + v_span.second);
            const double v_half_width = 0.5 * (v_span.second - v_span.first);
            for (std::size_t gu = 0; gu < kGaussNodes.size(); ++gu) {
                const double u = u_midpoint + u_half_width * kGaussNodes[gu];
                for (std::size_t gv = 0; gv < kGaussNodes.size(); ++gv) {
                    const double v =
                        v_midpoint + v_half_width * kGaussNodes[gv];
                    const geometry3d::NurbsSurfaceDerivatives3D derivatives =
                        patch.evaluate_with_derivatives(u, v);
                    const Eigen::Vector3d oriented_area =
                        derivatives.du.cross(derivatives.dv);
                    if (!oriented_area.allFinite()
                        || !(oriented_area.norm() > 0.0)) {
                        throw std::runtime_error(
                            "Native NURBS orientation quadrature encountered "
                            "a degenerate Jacobian");
                    }
                    // ||n||_1 dS = ||X_u x X_v||_1 du dv.
                    const double integrand = oriented_area.cwiseAbs().sum();
                    integral += u_half_width * v_half_width
                              * kGaussWeights[gu] * kGaussWeights[gv]
                              * integrand;
                }
            }
        }
    }
    if (!std::isfinite(integral) || !(integral > 0.0))
        throw std::runtime_error("Native NURBS patch orientation integral is invalid");
    return integral;
}

} // namespace

NativeDensityResolution3D choose_native_density_resolution_3d(
    const NativeNurbsSurface3D& surface,
    double cartesian_spacing,
    double target_crossings_per_cell)
{
    if (surface.patches.empty())
        throw std::invalid_argument("Native density resolution requires patches");
    if (!std::isfinite(cartesian_spacing) || !(cartesian_spacing > 0.0)) {
        throw std::invalid_argument(
            "Native density resolution requires positive Cartesian spacing");
    }
    if (!std::isfinite(target_crossings_per_cell)
        || !(target_crossings_per_cell > 0.0)) {
        throw std::invalid_argument(
            "Native density resolution requires a positive crossing target");
    }

    NativeDensityResolution3D result;
    result.patch_count = static_cast<int>(surface.patches.size());
    result.cartesian_spacing = cartesian_spacing;
    result.target_crossings_per_cell = target_crossings_per_cell;
    result.patch_orientation_integrals.reserve(surface.patches.size());
    for (const geometry3d::NurbsSurfacePatch3D& patch : surface.patches) {
        const double patch_integral = patch_orientation_integral(patch);
        result.patch_orientation_integrals.push_back(patch_integral);
        result.orientation_integral += patch_integral;
    }
    if (!std::isfinite(result.orientation_integral)
        || !(result.orientation_integral > 0.0)) {
        throw std::runtime_error(
            "Native NURBS surface orientation integral is invalid");
    }

    const double inverse_h_squared =
        1.0 / (cartesian_spacing * cartesian_spacing);
    result.predicted_crossings =
        result.orientation_integral * inverse_h_squared;
    result.continuous_elements_per_direction = std::sqrt(
        result.orientation_integral
        / (static_cast<double>(result.patch_count)
           * target_crossings_per_cell
           * cartesian_spacing * cartesian_spacing));
    if (!std::isfinite(result.predicted_crossings)
        || !std::isfinite(result.continuous_elements_per_direction)) {
        throw std::overflow_error("Native density resolution estimate overflowed");
    }

    const double threshold_safe_elements = std::nextafter(
        result.continuous_elements_per_direction,
        std::numeric_limits<double>::infinity());
    if (threshold_safe_elements
        > static_cast<double>(std::numeric_limits<int>::max() - 3)) {
        throw std::overflow_error("Native density resolution is too large");
    }
    result.elements_per_direction = std::max(
        1, static_cast<int>(std::floor(threshold_safe_elements)));
    result.coefficients_per_direction = result.elements_per_direction + 3;
    return result;
}

} // namespace kfbim::app3d
