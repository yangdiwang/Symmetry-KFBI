#include "nurbs_bezier_extraction_3d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfbim::geometry3d {

namespace {

using HomogeneousNet = std::vector<std::vector<Eigen::Vector4d>>;

void validate_net(const HomogeneousNet& net,
                  int degree,
                  const std::vector<double>& knots)
{
    if (net.empty() || knots.size() != net.size()
            + static_cast<std::size_t>(degree) + 1) {
        throw std::invalid_argument(
            "Bezier extraction has an inconsistent control count");
    }
    const std::size_t column_count = net.front().size();
    if (column_count == 0)
        throw std::invalid_argument("Bezier extraction has an empty control net");
    for (const auto& row : net) {
        if (row.size() != column_count) {
            throw std::invalid_argument(
                "Bezier extraction has an inconsistent control count");
        }
    }
}

void insert_knot_u_once(HomogeneousNet& net,
                        std::vector<double>& knots,
                        int degree,
                        double knot)
{
    validate_net(net, degree, knots);
    const int old_last_control = static_cast<int>(net.size()) - 1;
    const int span = static_cast<int>(
        std::distance(knots.begin(), std::upper_bound(knots.begin(), knots.end(), knot))) - 1;
    const int multiplicity = static_cast<int>(std::count(knots.begin(), knots.end(), knot));
    const int unchanged_prefix_end = span - degree;
    const int shifted_suffix_begin = span - multiplicity;
    if (unchanged_prefix_end < 0
        || unchanged_prefix_end > old_last_control
        || shifted_suffix_begin < 0
        || shifted_suffix_begin > old_last_control) {
        throw std::invalid_argument("Bezier extraction knot lies outside its domain");
    }

    HomogeneousNet refined(
        net.size() + 1,
        std::vector<Eigen::Vector4d>(net.front().size(), Eigen::Vector4d::Zero()));
    for (int i = 0; i <= span - degree; ++i)
        refined[static_cast<std::size_t>(i)] = net[static_cast<std::size_t>(i)];
    for (int i = span - multiplicity; i <= old_last_control; ++i) {
        refined[static_cast<std::size_t>(i + 1)] =
            net[static_cast<std::size_t>(i)];
    }
    for (int i = span - degree + 1; i <= span - multiplicity; ++i) {
        const double denominator =
            knots[static_cast<std::size_t>(i + degree)]
          - knots[static_cast<std::size_t>(i)];
        if (!(denominator > 0.0)) {
            throw std::invalid_argument(
                "Bezier extraction encountered a zero knot-insertion denominator");
        }
        const double alpha =
            (knot - knots[static_cast<std::size_t>(i)]) / denominator;
        for (std::size_t j = 0; j < net.front().size(); ++j) {
            refined[static_cast<std::size_t>(i)][j] =
                alpha * net[static_cast<std::size_t>(i)][j]
              + (1.0 - alpha) * net[static_cast<std::size_t>(i - 1)][j];
        }
    }
    knots.insert(knots.begin() + span + 1, knot);
    net = std::move(refined);
}

HomogeneousNet transpose(const HomogeneousNet& net)
{
    if (net.empty() || net.front().empty())
        throw std::invalid_argument("Bezier extraction has an empty control net");
    HomogeneousNet result(
        net.front().size(),
        std::vector<Eigen::Vector4d>(net.size(), Eigen::Vector4d::Zero()));
    for (std::size_t i = 0; i < net.size(); ++i) {
        if (net[i].size() != net.front().size()) {
            throw std::invalid_argument(
                "Bezier extraction has an inconsistent control count");
        }
        for (std::size_t j = 0; j < net[i].size(); ++j)
            result[j][i] = net[i][j];
    }
    return result;
}

void refine_active_knots_u(HomogeneousNet& net,
                           std::vector<double>& knots,
                           int degree,
                           double domain_start,
                           double domain_end)
{
    if (degree == 0)
        return;
    std::vector<double> active_knots;
    for (const double knot : knots) {
        if (knot >= domain_start && knot <= domain_end
            && (active_knots.empty() || knot != active_knots.back())) {
            active_knots.push_back(knot);
        }
    }
    for (const double knot : active_knots) {
        const int multiplicity =
            static_cast<int>(std::count(knots.begin(), knots.end(), knot));
        const int target_multiplicity =
            (knot == domain_start || knot == domain_end) ? degree + 1 : degree;
        for (int insertion = multiplicity; insertion < target_multiplicity;
             ++insertion) {
            insert_knot_u_once(net, knots, degree, knot);
        }
    }
}

std::vector<int> nonzero_spans(const std::vector<double>& knots,
                               int degree,
                               int control_count)
{
    std::vector<int> spans;
    for (int span = degree; span < control_count; ++span) {
        if (knots[static_cast<std::size_t>(span)]
            < knots[static_cast<std::size_t>(span + 1)]) {
            spans.push_back(span);
        }
    }
    return spans;
}

} // namespace

std::vector<RationalBezierElement3D>
extract_rational_bezier_elements_3d(const NurbsSurfaceModel3D& model)
{
    std::vector<RationalBezierElement3D> elements;
    for (int patch_index = 0; patch_index < model.num_patches(); ++patch_index) {
        const NurbsSurfacePatch3D& patch = model.patch(patch_index);
        const int degree_u = patch.basis_u().degree();
        const int degree_v = patch.basis_v().degree();
        const auto& points = patch.control_net();
        const auto& weights = patch.weights();
        HomogeneousNet net(
            points.size(),
            std::vector<Eigen::Vector4d>(points.front().size()));
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (points[i].size() != points.front().size()
                || weights[i].size() != points.front().size()) {
                throw std::invalid_argument(
                    "Bezier extraction has an inconsistent control count");
            }
            for (std::size_t j = 0; j < points[i].size(); ++j) {
                const double weight = weights[i][j];
                if (!std::isfinite(weight) || weight <= 0.0) {
                    throw std::invalid_argument(
                        "Bezier extraction has a nonpositive homogeneous weight");
                }
                net[i][j].head<3>() = weight * points[i][j];
                net[i][j].w() = weight;
            }
        }

        std::vector<double> knots_u = patch.basis_u().knots();
        std::vector<double> knots_v = patch.basis_v().knots();
        refine_active_knots_u(
            net, knots_u, degree_u,
            patch.domain_start_u(), patch.domain_end_u());
        HomogeneousNet transposed = transpose(net);
        refine_active_knots_u(
            transposed, knots_v, degree_v,
            patch.domain_start_v(), patch.domain_end_v());
        net = transpose(transposed);
        validate_net(net, degree_u, knots_u);
        if (knots_v.size() != net.front().size()
                + static_cast<std::size_t>(degree_v) + 1) {
            throw std::invalid_argument(
                "Bezier extraction has an inconsistent control count");
        }

        const auto spans_u = nonzero_spans(
            knots_u, degree_u, static_cast<int>(net.size()));
        const auto spans_v = nonzero_spans(
            knots_v, degree_v, static_cast<int>(net.front().size()));
        for (const int span_u : spans_u) {
            for (const int span_v : spans_v) {
                RationalBezierElement3D element;
                element.patch_index = patch_index;
                element.component = model.patch_component(patch_index);
                element.degree_u = degree_u;
                element.degree_v = degree_v;
                element.parameter_u0 = knots_u[static_cast<std::size_t>(span_u)];
                element.parameter_u1 = knots_u[static_cast<std::size_t>(span_u + 1)];
                element.parameter_v0 = knots_v[static_cast<std::size_t>(span_v)];
                element.parameter_v1 = knots_v[static_cast<std::size_t>(span_v + 1)];
                for (int i = span_u - degree_u; i <= span_u; ++i) {
                    for (int j = span_v - degree_v; j <= span_v; ++j) {
                        element.homogeneous_controls.push_back(
                            net[static_cast<std::size_t>(i)]
                               [static_cast<std::size_t>(j)]);
                    }
                }
                (void)element.bounds();
                elements.push_back(std::move(element));
            }
        }
    }
    return elements;
}

} // namespace kfbim::geometry3d
