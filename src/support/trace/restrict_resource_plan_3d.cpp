#include "src/support/trace/restrict_resource_plan_3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace kfbim::app3d {
namespace {

class AnchorTree {
public:
    struct Match {
        int index = -1;
        double squared_distance = std::numeric_limits<double>::infinity();
    };

    AnchorTree(const std::vector<RestrictResourceAnchor3D>& anchors,
               std::vector<int> indices)
        : anchors_(anchors)
    {
        nodes_.reserve(indices.size());
        root_ = build(indices, 0, static_cast<int>(indices.size()), 0);
    }

    Match nearest(const Eigen::Vector3d& point, int excluded_sheet = -1) const
    {
        Match result;
        search(root_, point, excluded_sheet, result);
        return result;
    }

private:
    struct Node { int anchor; int axis; int left; int right; int uniform_sheet; };
    const std::vector<RestrictResourceAnchor3D>& anchors_;
    std::vector<Node> nodes_;
    int root_ = -1;

    int build(std::vector<int>& ids, int begin, int end, int depth)
    {
        if (begin == end)
            return -1;
        const int axis = depth % 3;
        const int middle = begin + (end - begin) / 2;
        std::nth_element(ids.begin() + begin, ids.begin() + middle,
                         ids.begin() + end, [&](int a, int b) {
            const double x = anchors_[static_cast<std::size_t>(a)].point[axis];
            const double y = anchors_[static_cast<std::size_t>(b)].point[axis];
            return x < y || (x == y && a < b);
        });
        const int index = static_cast<int>(nodes_.size());
        const int anchor = ids[static_cast<std::size_t>(middle)];
        nodes_.push_back({anchor, axis, -1, -1,
                         anchors_[static_cast<std::size_t>(anchor)].sheet_id});
        const int left = build(ids, begin, middle, depth + 1);
        const int right = build(ids, middle + 1, end, depth + 1);
        auto& node = nodes_[static_cast<std::size_t>(index)];
        node.left = left;
        node.right = right;
        for (int child : {left, right}) {
            if (child >= 0 && nodes_[static_cast<std::size_t>(child)].uniform_sheet
                                  != node.uniform_sheet)
                node.uniform_sheet = -1;
        }
        return index;
    }

    void search(int index, const Eigen::Vector3d& point, int excluded_sheet,
                Match& result) const
    {
        if (index < 0)
            return;
        const auto& node = nodes_[static_cast<std::size_t>(index)];
        if (excluded_sheet >= 0 && node.uniform_sheet == excluded_sheet)
            return;
        const auto& anchor = anchors_[static_cast<std::size_t>(node.anchor)];
        const double squared = (anchor.point - point).squaredNorm();
        if (anchor.sheet_id != excluded_sheet
            && (squared < result.squared_distance
                || (squared == result.squared_distance
                    && (result.index < 0 || node.anchor < result.index)))) {
            result.index = node.anchor;
            result.squared_distance = squared;
        }
        const double delta = point[node.axis] - anchor.point[node.axis];
        const int near = delta <= 0.0 ? node.left : node.right;
        const int far = delta <= 0.0 ? node.right : node.left;
        search(near, point, excluded_sheet, result);
        // Equality is required: a tied first-input anchor can be in either half.
        if (delta * delta <= result.squared_distance)
            search(far, point, excluded_sheet, result);
    }
};

using SheetTrees = std::map<int, AnchorTree>;

SheetTrees make_sheet_trees(const std::vector<RestrictResourceAnchor3D>& anchors,
                           bool is_spread)
{
    std::map<int, std::vector<int>> indices;
    for (int i = 0; i < static_cast<int>(anchors.size()); ++i) {
        const auto& anchor = anchors[static_cast<std::size_t>(i)];
        if (!anchor.point.allFinite() || !std::isfinite(anchor.u)
            || !std::isfinite(anchor.v) || anchor.patch_id < 0
            || anchor.sheet_id < 0 || (is_spread && anchor.crossing_id < 0)) {
            throw std::invalid_argument("restrict resource anchor is malformed");
        }
        indices[anchor.sheet_id].push_back(i);
    }
    SheetTrees result;
    for (auto& entry : indices)
        result.emplace(std::piecewise_construct,
                       std::forward_as_tuple(entry.first),
                       std::forward_as_tuple(anchors, std::move(entry.second)));
    return result;
}

struct SharedAxis {
    int start = 0;
    Eigen::Matrix<double, 3, Eigen::Dynamic> weights;
};

SharedAxis shared_axis(const std::array<Eigen::Vector3d, 3>& samples,
                       int axis, double origin, double spacing,
                       int dimension, int width, bool python_mean_centered,
                       double endpoint_snap)
{
    if (!std::isfinite(origin) || !std::isfinite(spacing) || !(spacing > 0.0)
        || dimension < width || !std::isfinite(endpoint_snap) || endpoint_snap < 0.0)
        throw std::invalid_argument("shared-side cover has invalid grid axis");
    std::array<double, 3> coordinates{};
    const double upper = static_cast<double>(dimension - 1);
    for (int s = 0; s < 3; ++s) {
        double coordinate = (samples[static_cast<std::size_t>(s)][axis] - origin)
                          / spacing;
        const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
                               * std::max({1.0, std::abs(coordinate), upper});
        if (!std::isfinite(coordinate) || coordinate < -tolerance
            || coordinate > upper + tolerance)
            throw std::out_of_range("shared-side cover sample lies outside node box");
        coordinates[static_cast<std::size_t>(s)] = std::clamp(coordinate, 0.0, upper);
    }
    const auto range = std::minmax_element(coordinates.begin(), coordinates.end());
    const double low = *range.first;
    const double high = *range.second;
    const double center = python_mean_centered
        ? (coordinates[0] + coordinates[1] + coordinates[2]) / 3.0
        : 0.5 * (low + high);
    // Python min(range(...), key=distance) selects the smaller start on ties.
    const double ideal = center - 0.5 * static_cast<double>(width - 1);
    const int preferred = python_mean_centered
        ? static_cast<int>(std::ceil(ideal - 0.5))
        : static_cast<int>(std::floor(ideal + 0.5));
    // Keep the old zero-snap arithmetic untouched. The 93 opt-in follows
    // floor(min+eps)/ceil(max-eps), in normalized grid coordinates, exactly
    // as Grid.cover3/cover4 and curved Grid.cover define their legal range.
    const int enclosing_low = endpoint_snap==0.0
        ? std::max(0, static_cast<int>(std::ceil(high - width + 1)))
        : std::max(0, static_cast<int>(std::ceil(high-endpoint_snap))-(width-1));
    const int enclosing_high = std::min(dimension-width,
        static_cast<int>(std::floor(low+endpoint_snap)));
    SharedAxis result;
    int fallback=preferred;
    if(python_mean_centered){
        const int lower=static_cast<int>(std::floor(center));
        const double fraction=center-lower;
        const int rounded=lower+((fraction>0.5 || (fraction==0.5 && lower%2!=0))?1:0);
        fallback=rounded-width/2; // Python round(mean), ties to even.
    }
    result.start = enclosing_low <= enclosing_high
        ? std::clamp(preferred, enclosing_low, enclosing_high)
        : std::clamp(fallback, 0, dimension - width);
    result.weights = Eigen::Matrix<double, 3, Eigen::Dynamic>::Ones(3, width);
    for (int s = 0; s < 3; ++s) {
        const double local = coordinates[static_cast<std::size_t>(s)] - result.start;
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < width; ++j) {
                if (i != j)
                    result.weights(s, i) *= (local - j) / static_cast<double>(i - j);
            }
        }
    }
    return result;
}

} // namespace

RestrictResourcePlan3D build_restrict_resource_plan_3d(
    const std::vector<RestrictResourceAnchor3D>& trace_anchors,
    const std::vector<RestrictResourceAnchor3D>& spread_anchors,
    const std::vector<RestrictSupportVisit3D>& visits,
    double h, RestrictResourceMode3D mode)
{
    if (!std::isfinite(h) || !(h > 0.0))
        throw std::invalid_argument("restrict resource plan requires positive spacing");
    if (mode != RestrictResourceMode3D::ReferencePerVisit
        && mode != RestrictResourceMode3D::ReuseBySheetNode
        && mode != RestrictResourceMode3D::GuardedReuseBySheetNode)
        throw std::invalid_argument("unknown restrict resource plan mode");
    const auto trace_trees = make_sheet_trees(trace_anchors, false);
    const auto spread_trees = make_sheet_trees(spread_anchors, true);
    std::vector<int> all_ids(trace_anchors.size());
    std::iota(all_ids.begin(), all_ids.end(), 0);
    const AnchorTree all_trace_tree(trace_anchors, std::move(all_ids));

    RestrictResourcePlan3D plan;
    plan.mode = mode;
    plan.visit_to_request.assign(visits.size(), -1);
    plan.counts.support_visits = visits.size();
    std::map<std::pair<int, int>, int> request_index;
    std::map<int, std::pair<Eigen::Vector3d, bool>> grid_nodes;
    std::set<int> used_trace, used_spread, used_crossings;
    for (std::size_t i = 0; i < visits.size(); ++i) {
        const auto& visit = visits[i];
        if (visit.trace_center_id < 0
            || visit.trace_center_id >= static_cast<int>(trace_anchors.size())
            || visit.grid_full_id < 0 || !visit.grid_point.allFinite())
            throw std::invalid_argument("restrict support visit is malformed");
        const auto inserted = grid_nodes.emplace(visit.grid_full_id,
            std::make_pair(visit.grid_point, visit.actual_inside));
        if (!inserted.second
            && (!(inserted.first->second.first.array() == visit.grid_point.array()).all()
                || inserted.first->second.second != visit.actual_inside))
            throw std::invalid_argument("full grid ID has inconsistent coordinates or side");
        if (visit.desired_inside == visit.actual_inside)
            continue;
        ++plan.counts.wrong_side_visits;
        const int sheet = trace_anchors[static_cast<std::size_t>(visit.trace_center_id)].sheet_id;
        const auto key = std::make_pair(sheet, visit.grid_full_id);
        if (mode != RestrictResourceMode3D::ReferencePerVisit) {
            const auto found = request_index.find(key);
            if (found != request_index.end()) {
                plan.visit_to_request[i] = found->second;
                continue;
            }
        }
        const auto nearest_trace = trace_trees.at(sheet).nearest(visit.grid_point);
        RestrictResourceRequest3D request;
        request.sheet_id = sheet;
        request.grid_full_id = visit.grid_full_id;
        request.grid_point = visit.grid_point;
        request.anchor_index = nearest_trace.index;
        request.nearest_trace_distance = std::sqrt(nearest_trace.squared_distance);
        double selected_squared = nearest_trace.squared_distance;
        if (request.nearest_trace_distance > 1.5 * h) {
            const auto found = spread_trees.find(sheet);
            if (found != spread_trees.end()) {
                const auto crossing = found->second.nearest(visit.grid_point);
                if (crossing.index >= 0 && crossing.squared_distance < selected_squared) {
                    request.anchor_source = RestrictAnchorSource3D::SpreadCrossing;
                    request.anchor_index = crossing.index;
                    selected_squared = crossing.squared_distance;
                }
            }
        }
        request.distance = std::sqrt(selected_squared);
        request.too_far = request.distance > 2.25 * h;
        const auto other = all_trace_tree.nearest(visit.grid_point, sheet);
        request.competing_sheet = other.index >= 0
            && std::sqrt(other.squared_distance) < request.distance - 0.15 * h;
        request.fallback_required = mode == RestrictResourceMode3D::GuardedReuseBySheetNode
            && (request.too_far || request.competing_sheet);
        if (request.anchor_source == RestrictAnchorSource3D::Trace) {
            ++plan.counts.trace_requests;
            used_trace.insert(request.anchor_index);
        } else {
            ++plan.counts.spread_requests;
            used_spread.insert(request.anchor_index);
            const int crossing = spread_anchors[static_cast<std::size_t>(request.anchor_index)].crossing_id;
            if (used_crossings.insert(crossing).second)
                plan.required_crossing_ids.push_back(crossing);
        }
        plan.counts.too_far_requests += request.too_far ? 1 : 0;
        plan.counts.competing_sheet_requests += request.competing_sheet ? 1 : 0;
        plan.counts.fallback_requests += request.fallback_required ? 1 : 0;
        const int index = static_cast<int>(plan.requests.size());
        plan.visit_to_request[i] = index;
        request_index.emplace(key, index);
        plan.requests.push_back(request);
    }
    plan.counts.requests = plan.requests.size();
    plan.counts.unique_trace_centers = used_trace.size();
    plan.counts.unique_spread_centers = used_spread.size();
    return plan;
}

SharedSideCoverRestrictStencil3D build_shared_side_cover_restrict_stencil_3d(
    const CartesianGrid3D& grid, const Eigen::Vector3d& trace_point,
    const Eigen::Vector3d& outward_normal, TensorProductCoverKind3D kind,
    bool python_mean_centered_cover, double cover_endpoint_snap)
{
    if (grid.layout() != DofLayout3D::Node || !trace_point.allFinite()
        || !outward_normal.allFinite() || std::abs(outward_normal.norm() - 1.0) > 1.0e-10)
        throw std::invalid_argument("shared-side cover requires node grid and unit normal");
    const int width = tensor_product_cover_width_3d(kind);
    if (width != 3 && width != 4)
        throw std::invalid_argument("shared-side cover received unknown cover kind");
    const auto spacing = grid.spacing();
    const auto origin = grid.origin();
    const auto dimensions = grid.dof_dims();
    const double h = std::min({spacing[0], spacing[1], spacing[2]});
    if (!std::isfinite(h) || !(h > 0.0))
        throw std::invalid_argument("shared-side cover requires positive grid spacing");
    constexpr std::array<double, 6> rho{{-1.5, -0.75, -0.5, 0.5, 0.75, 1.5}};
    // The symmetric Vandermonde splits into two 2x2 Gram blocks.  This is its
    // exact algebraic least-squares pseudoinverse; no per-centre SVD is needed.
    double s2 = 0.0, s4 = 0.0, s6 = 0.0;
    for (double r : rho) {
        s2 += r * r;
        s4 += r * r * r * r;
        s6 += r * r * r * r * r * r;
    }
    const double even_determinant = 6.0 * s4 - s2 * s2;
    const double odd_determinant = s2 * s6 - s4 * s4;
    SharedSideCoverRestrictStencil3D result;
    result.kind = kind;
    result.h = h;
    for (int s = 0; s < 6; ++s) {
        const double r = rho[static_cast<std::size_t>(s)];
        result.cubic_pseudoinverse(0, s) = (s4 - s2 * r * r) / even_determinant;
        result.cubic_pseudoinverse(1, s) = (s6 * r - s4 * r * r * r) / odd_determinant;
        result.cubic_pseudoinverse(2, s) = (6.0 * r * r - s2) / even_determinant;
        result.cubic_pseudoinverse(3, s) = (s2 * r * r * r - s4 * r) / odd_determinant;
    }
    for (int side = 0; side < 2; ++side) {
        auto& output = result.sides[static_cast<std::size_t>(side)];
        output.desired_inside = side == 0;
        for (int s = 0; s < 3; ++s) {
            const int global = side * 3 + s;
            output.signed_rho[static_cast<std::size_t>(s)] = rho[static_cast<std::size_t>(global)];
            output.sample_points[static_cast<std::size_t>(s)] =
                trace_point + h * rho[static_cast<std::size_t>(global)] * outward_normal;
            output.value_recovery[s] = result.cubic_pseudoinverse(0, global);
            output.normal_recovery[s] = result.cubic_pseudoinverse(1, global) / h;
        }
        std::array<SharedAxis, 3> axes;
        for (int d = 0; d < 3; ++d)
            axes[static_cast<std::size_t>(d)] = shared_axis(output.sample_points, d,
                origin[static_cast<std::size_t>(d)], spacing[static_cast<std::size_t>(d)],
                dimensions[static_cast<std::size_t>(d)], width, python_mean_centered_cover,
                cover_endpoint_snap);
        const int count = width * width * width;
        output.grid_ids.reserve(static_cast<std::size_t>(count));
        output.sampling_weights.resize(3, count);
        int slot = 0;
        for (int k = 0; k < width; ++k) {
            for (int j = 0; j < width; ++j) {
                for (int i = 0; i < width; ++i, ++slot) {
                    output.grid_ids.push_back(grid.index(axes[0].start + i,
                        axes[1].start + j, axes[2].start + k));
                    for (int s = 0; s < 3; ++s)
                        output.sampling_weights(s, slot) = axes[0].weights(s, i)
                            * axes[1].weights(s, j) * axes[2].weights(s, k);
                }
            }
        }
        output.value_weights = output.value_recovery * output.sampling_weights;
        output.normal_weights = output.normal_recovery * output.sampling_weights;
        if (!output.sampling_weights.allFinite() || !output.value_weights.allFinite()
            || !output.normal_weights.allFinite())
            throw std::runtime_error("shared-side cover generated non-finite weights");
    }
    return result;
}

} // namespace kfbim::app3d
