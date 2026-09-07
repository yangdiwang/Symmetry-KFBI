#include "nurbs_patch_polar_evaluator_3d.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace kfbim::geometry3d {
namespace {

// Patch evaluation first clamps to the domain, then basis.checked_parameter
// absorbs the tolerance-sized endpoint neighborhoods into the exact endpoints.
double normalized_parameter(const geometry::NurbsBasis1D& basis, double value)
{
    geometry::require_finite(value, "surface parameter");
    value = std::clamp(value, basis.domain_start(), basis.domain_end());
    if (value <= basis.domain_start() + basis.tolerance())
        return basis.domain_start();
    if (value >= basis.domain_end() - basis.tolerance())
        return basis.domain_end();
    return value;
}

int original_span(const geometry::NurbsBasis1D& basis, double begin, double end)
{
    const auto& knots = basis.knots();
    const auto upper = std::upper_bound(knots.begin(), knots.end(), begin);
    const int index = static_cast<int>(upper - knots.begin()) - 1;
    if (index < basis.degree() || index >= basis.num_basis_functions()
        || knots[static_cast<std::size_t>(index)] != begin
        || knots[static_cast<std::size_t>(index + 1)] != end)
        return -1;
    // The original basis uses divide_or_zero for denominators <= tolerance.
    // A mathematical Bezier evaluator is not equivalent on these tiny spans.
    if (!(end - begin > basis.tolerance()) || !std::isfinite(end - begin))
        return -1;
    return index;
}

// Leave the final two points intact so their difference also gives the first
// derivative. Controls are homogeneous points; derivatives may have any weight.
void reduce_to_pair(Eigen::Vector4d* controls, int degree, double parameter)
{
    for (int remaining = degree; remaining > 1; --remaining) {
        for (int i = 0; i < remaining; ++i)
            controls[i] = (1.0 - parameter) * controls[i]
                        + parameter * controls[i + 1];
    }
}

} // namespace

NurbsPatchPolarEvaluator3D::NurbsPatchPolarEvaluator3D(
    const NurbsSurfacePatch3D& patch,
    std::vector<RationalBezierElement3D> base_elements)
    : base_elements_(std::move(base_elements))
    , span_count_u_(patch.basis_u().num_basis_functions())
    , span_count_v_(patch.basis_v().num_basis_functions())
{
    span_elements_.assign(static_cast<std::size_t>(span_count_u_)
                         * static_cast<std::size_t>(span_count_v_), -1);
    for (std::size_t index = 0; index < base_elements_.size(); ++index) {
        const auto& element = base_elements_[index];
        if (element.degree_u != patch.basis_u().degree()
            || element.degree_v != patch.basis_v().degree()) continue;
        const int su = original_span(patch.basis_u(), element.u0(), element.u1());
        const int sv = original_span(patch.basis_v(), element.v0(), element.v1());
        if (su < 0 || sv < 0) continue;
        if (element.homogeneous_controls.size()
            != (static_cast<std::size_t>(element.degree_u) + 1)
             * (static_cast<std::size_t>(element.degree_v) + 1)) continue;
        bool finite_positive = true;
        for (const auto& control : element.homogeneous_controls)
            finite_positive = finite_positive && control.allFinite() && control.w() > 0;
        if (!finite_positive) continue;
        int& mapped = span_elements_[static_cast<std::size_t>(su) * span_count_v_ + sv];
        // Duplicate base spans are ambiguous; retain the original fallback.
        mapped = mapped == -1 ? static_cast<int>(index) : -2;
    }
}

NurbsSurfaceDerivatives3D NurbsPatchPolarEvaluator3D::evaluate_with_derivatives(
    const NurbsSurfacePatch3D& patch, double u, double v,
    NurbsPolarEvaluationWorkspace3D& workspace) const
{
    const double parameter_u = normalized_parameter(patch.basis_u(), u);
    const double parameter_v = normalized_parameter(patch.basis_v(), v);
    const int su = patch.basis_u().find_span(parameter_u);
    const int sv = patch.basis_v().find_span(parameter_v);
    if (su < 0 || sv < 0 || su >= span_count_u_ || sv >= span_count_v_)
        return patch.evaluate_with_derivatives(u, v);
    const int element_index = span_elements_[static_cast<std::size_t>(su) * span_count_v_ + sv];
    if (element_index < 0) return patch.evaluate_with_derivatives(u, v);
    const auto& element = base_elements_[static_cast<std::size_t>(element_index)];
    const int p = element.degree_u, q = element.degree_v;
    const double span_u = element.u1() - element.u0();
    const double span_v = element.v1() - element.v0();
    const double local_u = (parameter_u - element.u0()) / span_u;
    const double local_v = (parameter_v - element.v0()) / span_v;
    if (!(local_u >= 0 && local_u <= 1 && local_v >= 0 && local_v <= 1))
        return patch.evaluate_with_derivatives(u, v);
    const std::size_t net_size = element.homogeneous_controls.size();
    const std::size_t scratch_size = net_size + 2 * (static_cast<std::size_t>(p) + 1);
    Eigen::Vector4d* scratch;
    if (p <= NurbsPolarEvaluationWorkspace3D::fixed_degree
        && q <= NurbsPolarEvaluationWorkspace3D::fixed_degree) {
        scratch = workspace.fixed_.data();
    } else {
        workspace.dynamic_.resize(scratch_size);
        scratch = workspace.dynamic_.data();
    }
    std::copy(element.homogeneous_controls.begin(), element.homogeneous_controls.end(), scratch);
    Eigen::Vector4d* values = scratch + net_size;
    Eigen::Vector4d* partial_v = values + p + 1;
    for (int i = 0; i <= p; ++i) {
        Eigen::Vector4d* row = scratch + static_cast<std::size_t>(i) * (q + 1);
        reduce_to_pair(row, q, local_v);
        if (q == 0) {
            values[i] = row[0];
            partial_v[i].setZero();
        } else {
            partial_v[i] = (static_cast<double>(q) / span_v) * (row[1] - row[0]);
            values[i] = (1.0 - local_v) * row[0] + local_v * row[1];
        }
    }
    reduce_to_pair(values, p, local_u);
    reduce_to_pair(partial_v, p, local_u);
    Eigen::Vector4d h = values[0];
    Eigen::Vector4d hu = Eigen::Vector4d::Zero();
    Eigen::Vector4d hv = partial_v[0];
    if (p > 0) {
        hu = (static_cast<double>(p) / span_u) * (values[1] - values[0]);
        h = (1.0 - local_u) * values[0] + local_u * values[1];
        hv = (1.0 - local_u) * partial_v[0] + local_u * partial_v[1];
    }
    if (!h.allFinite() || !hu.allFinite() || !hv.allFinite()
        || h.w() <= std::max(patch.basis_u().tolerance(), patch.basis_v().tolerance()))
        return patch.evaluate_with_derivatives(u, v);
    NurbsSurfaceDerivatives3D result;
    result.point = h.head<3>() / h.w();
    result.du = (hu.head<3>() - result.point * hu.w()) / h.w();
    result.dv = (hv.head<3>() - result.point * hv.w()) / h.w();
    if (!result.point.allFinite() || !result.du.allFinite() || !result.dv.allFinite())
        return patch.evaluate_with_derivatives(u, v);
    return result;
}

} // namespace kfbim::geometry3d
