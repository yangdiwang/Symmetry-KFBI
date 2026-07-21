#include "rational_bezier_surface_3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kfbim::geometry3d {

namespace {

using HomogeneousNet = std::vector<std::vector<Eigen::Vector4d>>;

double midpoint_if_representable(double begin, double end)
{
    const double midpoint = 0.5 * begin + 0.5 * end;
    if (!std::isfinite(midpoint) || midpoint <= begin || midpoint >= end) {
        throw std::invalid_argument(
            "Bezier parameter interval has no representable midpoint");
    }
    return midpoint;
}

bool has_representable_midpoint(double begin, double end)
{
    const double midpoint = 0.5 * begin + 0.5 * end;
    return std::isfinite(midpoint) && midpoint > begin && midpoint < end;
}

std::size_t expected_control_count(const RationalBezierElement3D& element)
{
    if (element.degree_u < 0 || element.degree_v < 0)
        throw std::invalid_argument("Bezier element degree must be nonnegative");
    if (element.degree_u == std::numeric_limits<int>::max()
        || element.degree_v == std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "Bezier element degree is too large for inclusive indexing");
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const auto checked_extent = [maximum](int degree) {
        if (static_cast<std::uintmax_t>(degree)
            >= static_cast<std::uintmax_t>(maximum)) {
            throw std::invalid_argument(
                "Bezier element control count is too large");
        }
        return static_cast<std::size_t>(degree) + std::size_t{1};
    };
    const std::size_t count_u = checked_extent(element.degree_u);
    const std::size_t count_v = checked_extent(element.degree_v);
    if (count_u > maximum / count_v) {
        throw std::invalid_argument(
            "Bezier element control count is too large");
    }
    const std::size_t count = count_u * count_v;
    if (count > maximum / sizeof(Eigen::Vector4d)) {
        throw std::invalid_argument(
            "Bezier element control count is too large");
    }
    return count;
}

void validate_element(const RationalBezierElement3D& element)
{
    if (!std::isfinite(element.parameter_u0)
        || !std::isfinite(element.parameter_u1)
        || !std::isfinite(element.parameter_v0)
        || !std::isfinite(element.parameter_v1)
        || element.parameter_u0 >= element.parameter_u1
        || element.parameter_v0 >= element.parameter_v1) {
        throw std::invalid_argument("Bezier element has a zero parameter span");
    }
    if (element.homogeneous_controls.size()
        != expected_control_count(element)) {
        throw std::invalid_argument("Bezier element has an inconsistent control count");
    }
    for (const Eigen::Vector4d& control : element.homogeneous_controls) {
        if (!control.allFinite() || control.w() <= 0.0) {
            throw std::invalid_argument(
                "Bezier element has a nonpositive homogeneous weight");
        }
    }
}

Eigen::Vector4d evaluate_curve(std::vector<Eigen::Vector4d> controls,
                               double parameter)
{
    for (std::size_t remaining = controls.size(); remaining > 1; --remaining) {
        for (std::size_t i = 0; i + 1 < remaining; ++i) {
            controls[i] = (1.0 - parameter) * controls[i]
                        + parameter * controls[i + 1];
        }
    }
    return controls.front();
}

std::array<std::vector<Eigen::Vector4d>, 2> split_curve(
    std::vector<Eigen::Vector4d> controls)
{
    const std::size_t count = controls.size();
    std::array<std::vector<Eigen::Vector4d>, 2> children{
        std::vector<Eigen::Vector4d>(count),
        std::vector<Eigen::Vector4d>(count)};
    children[0][0] = controls.front();
    children[1][count - 1] = controls.back();
    for (std::size_t level = 1; level < count; ++level) {
        for (std::size_t i = 0; i + level < count; ++i)
            controls[i] = 0.5 * controls[i] + 0.5 * controls[i + 1];
        children[0][level] = controls[0];
        children[1][count - level - 1] = controls[count - level - 1];
    }
    return children;
}

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
    if (multiplicity > degree) {
        throw std::invalid_argument(
            "Bezier extraction knot insertion exceeds its supported multiplicity");
    }
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
        if (knot > domain_start && knot < domain_end
            && multiplicity > degree) {
            throw std::invalid_argument(
                "Bezier extraction interior knot multiplicity exceeds degree "
                + std::to_string(degree) + " at knot "
                + std::to_string(knot));
        }
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

double projected_control_variation(const RationalBezierElement3D& element,
                                   bool along_u)
{
    double maximum = 0.0;
    if (along_u) {
        for (int j = 0; j <= element.degree_v; ++j) {
            double variation = 0.0;
            for (int i = 1; i <= element.degree_u; ++i) {
                const Eigen::Vector4d& previous = element.control(i - 1, j);
                const Eigen::Vector4d& current = element.control(i, j);
                variation += (current.head<3>() / current.w()
                            - previous.head<3>() / previous.w()).norm();
            }
            maximum = std::max(maximum, variation);
        }
    } else {
        for (int i = 0; i <= element.degree_u; ++i) {
            double variation = 0.0;
            for (int j = 1; j <= element.degree_v; ++j) {
                const Eigen::Vector4d& previous = element.control(i, j - 1);
                const Eigen::Vector4d& current = element.control(i, j);
                variation += (current.head<3>() / current.w()
                            - previous.head<3>() / previous.w()).norm();
            }
            maximum = std::max(maximum, variation);
        }
    }
    return maximum;
}

} // namespace

double RationalBezierElement3D::u0() const { return parameter_u0; }
double RationalBezierElement3D::u1() const { return parameter_u1; }
double RationalBezierElement3D::v0() const { return parameter_v0; }
double RationalBezierElement3D::v1() const { return parameter_v1; }

const Eigen::Vector4d& RationalBezierElement3D::control(int i, int j) const
{
    if (i < 0 || i > degree_u || j < 0 || j > degree_v)
        throw std::out_of_range("Bezier control index is out of range");
    if (homogeneous_controls.size() != expected_control_count(*this))
        throw std::invalid_argument("Bezier element has an inconsistent control count");
    return homogeneous_controls[
        static_cast<std::size_t>(i * (degree_v + 1) + j)];
}

Eigen::Vector3d RationalBezierElement3D::evaluate(double u, double v) const
{
    validate_element(*this);
    if (!std::isfinite(u) || !std::isfinite(v)
        || u < parameter_u0 || u > parameter_u1
        || v < parameter_v0 || v > parameter_v1) {
        throw std::invalid_argument("Bezier evaluation parameter is outside its element");
    }
    const double local_u = (u - parameter_u0) / (parameter_u1 - parameter_u0);
    const double local_v = (v - parameter_v0) / (parameter_v1 - parameter_v0);
    std::vector<Eigen::Vector4d> reduced_u(
        static_cast<std::size_t>(degree_u + 1));
    for (int i = 0; i <= degree_u; ++i) {
        std::vector<Eigen::Vector4d> row;
        row.reserve(static_cast<std::size_t>(degree_v + 1));
        for (int j = 0; j <= degree_v; ++j)
            row.push_back(control(i, j));
        reduced_u[static_cast<std::size_t>(i)] = evaluate_curve(row, local_v);
    }
    const Eigen::Vector4d homogeneous = evaluate_curve(reduced_u, local_u);
    if (!homogeneous.allFinite() || homogeneous.w() <= 0.0)
        throw std::runtime_error("Bezier evaluation has a nonpositive weight");
    return homogeneous.head<3>() / homogeneous.w();
}

NurbsAabb3D RationalBezierElement3D::bounds() const
{
    validate_element(*this);
    NurbsAabb3D result;
    result.lower = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::infinity());
    result.upper = Eigen::Vector3d::Constant(
        -std::numeric_limits<double>::infinity());
    for (const Eigen::Vector4d& control : homogeneous_controls) {
        const Eigen::Vector3d projected = control.head<3>() / control.w();
        result.lower = result.lower.cwiseMin(projected);
        result.upper = result.upper.cwiseMax(projected);
    }
    return result;
}

std::array<RationalBezierElement3D, 2>
RationalBezierElement3D::split_u() const
{
    validate_element(*this);
    std::array<RationalBezierElement3D, 2> children{*this, *this};
    const double midpoint = midpoint_if_representable(
        parameter_u0, parameter_u1);
    children[0].parameter_u1 = midpoint;
    children[1].parameter_u0 = midpoint;
    for (int j = 0; j <= degree_v; ++j) {
        std::vector<Eigen::Vector4d> column;
        column.reserve(static_cast<std::size_t>(degree_u + 1));
        for (int i = 0; i <= degree_u; ++i)
            column.push_back(control(i, j));
        const auto split = split_curve(std::move(column));
        for (int i = 0; i <= degree_u; ++i) {
            children[0].homogeneous_controls[
                static_cast<std::size_t>(i * (degree_v + 1) + j)] =
                split[0][static_cast<std::size_t>(i)];
            children[1].homogeneous_controls[
                static_cast<std::size_t>(i * (degree_v + 1) + j)] =
                split[1][static_cast<std::size_t>(i)];
        }
    }
    validate_element(children[0]);
    validate_element(children[1]);
    return children;
}

std::array<RationalBezierElement3D, 2>
RationalBezierElement3D::split_v() const
{
    validate_element(*this);
    std::array<RationalBezierElement3D, 2> children{*this, *this};
    const double midpoint = midpoint_if_representable(
        parameter_v0, parameter_v1);
    children[0].parameter_v1 = midpoint;
    children[1].parameter_v0 = midpoint;
    for (int i = 0; i <= degree_u; ++i) {
        std::vector<Eigen::Vector4d> row;
        row.reserve(static_cast<std::size_t>(degree_v + 1));
        for (int j = 0; j <= degree_v; ++j)
            row.push_back(control(i, j));
        const auto split = split_curve(std::move(row));
        for (int j = 0; j <= degree_v; ++j) {
            children[0].homogeneous_controls[
                static_cast<std::size_t>(i * (degree_v + 1) + j)] =
                split[0][static_cast<std::size_t>(j)];
            children[1].homogeneous_controls[
                static_cast<std::size_t>(i * (degree_v + 1) + j)] =
                split[1][static_cast<std::size_t>(j)];
        }
    }
    validate_element(children[0]);
    validate_element(children[1]);
    return children;
}

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
                validate_element(element);
                elements.push_back(std::move(element));
            }
        }
    }
    return elements;
}

std::vector<RationalBezierElement3D>
subdivide_rational_bezier_elements_to_extent_3d(
    const std::vector<RationalBezierElement3D>& elements,
    double maximum_extent)
{
    if (!std::isfinite(maximum_extent) || maximum_extent <= 0.0) {
        throw std::invalid_argument(
            "Bezier acceleration maximum extent must be positive and finite");
    }
    struct PendingElement {
        RationalBezierElement3D element;
        int depth = 0;
    };
    std::vector<PendingElement> pending;
    pending.reserve(elements.size());
    for (const auto& element : elements) {
        validate_element(element);
        pending.push_back({element, 0});
    }

    std::vector<RationalBezierElement3D> leaves;
    while (!pending.empty()) {
        PendingElement current = std::move(pending.back());
        pending.pop_back();
        if (current.element.bounds().max_extent() <= maximum_extent) {
            leaves.push_back(std::move(current.element));
            continue;
        }
        if (current.depth >= 48) {
            throw std::runtime_error(
                "Bezier acceleration subdivision exceeded depth 48");
        }
        const bool split_along_u =
            projected_control_variation(current.element, true)
            >= projected_control_variation(current.element, false);
        if ((split_along_u
                && !has_representable_midpoint(
                    current.element.u0(), current.element.u1()))
            || (!split_along_u
                && !has_representable_midpoint(
                    current.element.v0(), current.element.v1()))) {
            throw std::runtime_error(
                "Bezier acceleration subdivision exceeded depth 48");
        }
        const auto children = split_along_u ? current.element.split_u()
                                            : current.element.split_v();
        pending.push_back({children[1], current.depth + 1});
        pending.push_back({children[0], current.depth + 1});
    }
    return leaves;
}

} // namespace kfbim::geometry3d
