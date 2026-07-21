#include "rational_bezier_subdivision_3d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfbim::geometry3d {

namespace {

bool has_representable_midpoint(double begin, double end)
{
    const double midpoint = 0.5 * begin + 0.5 * end;
    return std::isfinite(midpoint) && midpoint > begin && midpoint < end;
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
        (void)element.bounds();
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
