#include "nurbs_surface_intersector_3d.hpp"

#include "nurbs_bezier_extraction_3d.hpp"
#include "nurbs_bezier_intersection_3d.hpp"
#include "rational_bezier_subdivision_3d.hpp"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kfbim::geometry3d {

namespace {

using ExactKernel = CGAL::Exact_predicates_exact_constructions_kernel;
using ExactScalar = ExactKernel::FT;
constexpr int kFeatureProbeSubdivisionDepth = 16;


NurbsSurfaceCrossing3D as_surface_crossing(
    const NurbsElementRoot3D& root)
{
    return {root.patch_index,
            root.component,
            root.u,
            root.v,
            root.t,
            root.point,
            root.normal,
            root.residual,
            root.transversality};
}


NurbsAabb3D united_bounds(const NurbsAabb3D& first,
                          const NurbsAabb3D& second)
{
    return {first.lower.cwiseMin(second.lower),
            first.upper.cwiseMax(second.upper)};
}

NurbsAabb3D segment_bounds(const Eigen::Vector3d& start,
                           const Eigen::Vector3d& end)
{
    return {start.cwiseMin(end), start.cwiseMax(end)};
}

double parameter_tolerance(const NurbsSurfacePatch3D& patch)
{
    const double scale = std::max({
        1.0,
        patch.domain_end_u() - patch.domain_start_u(),
        patch.domain_end_v() - patch.domain_start_v()});
    return 16.0e-12 * scale;
}

double segment_parameter_tolerance(double geometry_tolerance,
                                   double segment_length)
{
    return std::max(16.0e-12,
                    16.0 * geometry_tolerance / segment_length);
}

bool same_patch_root(const NurbsSurfaceCrossing3D& first,
                     const NurbsSurfaceCrossing3D& second,
                     const NurbsSurfaceModel3D& model,
                     double geometry_tolerance,
                     double segment_length)
{
    if (first.patch_index != second.patch_index)
        return false;
    const double uv_tolerance = parameter_tolerance(
        model.patch(first.patch_index));
    return std::abs(first.u - second.u) <= uv_tolerance
        && std::abs(first.v - second.v) <= uv_tolerance
        && std::abs(first.edge_parameter - second.edge_parameter)
            <= segment_parameter_tolerance(
                geometry_tolerance, segment_length)
        && (first.point - second.point).norm()
            <= 16.0 * geometry_tolerance;
}

bool coincident_root(const NurbsSurfaceCrossing3D& first,
                     const NurbsSurfaceCrossing3D& second,
                     double geometry_tolerance,
                     double segment_length)
{
    return std::abs(first.edge_parameter - second.edge_parameter)
            <= segment_parameter_tolerance(
                geometry_tolerance, segment_length)
        && (first.point - second.point).norm()
            <= 16.0 * geometry_tolerance;
}

bool root_on_interval(const NurbsSurfaceCrossing3D& root,
                      const NurbsPatchEdgeInterval3D& interval,
                      const NurbsSurfaceModel3D& model,
                      double* along_edge_parameter = nullptr)
{
    if (root.patch_index != interval.patch)
        return false;
    const NurbsSurfacePatch3D& patch = model.patch(interval.patch);
    const double tolerance = parameter_tolerance(patch);
    double fixed = 0.0;
    double boundary = 0.0;
    double parameter = 0.0;
    switch (interval.edge) {
    case NurbsPatchEdge3D::UMin:
        fixed = root.u;
        boundary = patch.domain_start_u();
        parameter = root.v;
        break;
    case NurbsPatchEdge3D::UMax:
        fixed = root.u;
        boundary = patch.domain_end_u();
        parameter = root.v;
        break;
    case NurbsPatchEdge3D::VMin:
        fixed = root.v;
        boundary = patch.domain_start_v();
        parameter = root.u;
        break;
    case NurbsPatchEdge3D::VMax:
        fixed = root.v;
        boundary = patch.domain_end_v();
        parameter = root.u;
        break;
    }
    if (std::abs(fixed - boundary) > tolerance
        || parameter < interval.begin - tolerance
        || parameter > interval.end + tolerance) {
        return false;
    }
    if (along_edge_parameter != nullptr)
        *along_edge_parameter = parameter;
    return true;
}

bool root_on_non_g1_feature(
    const NurbsSurfaceCrossing3D& root,
    const NurbsSurfaceModel3D& model)
{
    return std::any_of(
        model.connections().begin(), model.connections().end(),
        [&](const NurbsPatchEdgeConnection3D& connection) {
            return !connection.g1
                && (root_on_interval(root, connection.first, model)
                    || root_on_interval(root, connection.second, model));
        });
}

bool element_touches_interval(
    const RationalBezierElement3D& element,
    const NurbsPatchEdgeInterval3D& interval,
    const NurbsSurfaceModel3D& model)
{
    if (element.patch_index != interval.patch)
        return false;
    const NurbsSurfacePatch3D& patch = model.patch(interval.patch);
    const double tolerance = parameter_tolerance(patch);
    const auto overlaps = [&](double first_begin, double first_end) {
        return first_end >= interval.begin - tolerance
            && first_begin <= interval.end + tolerance;
    };
    switch (interval.edge) {
    case NurbsPatchEdge3D::UMin:
        return std::abs(element.u0() - patch.domain_start_u()) <= tolerance
            && overlaps(element.v0(), element.v1());
    case NurbsPatchEdge3D::UMax:
        return std::abs(element.u1() - patch.domain_end_u()) <= tolerance
            && overlaps(element.v0(), element.v1());
    case NurbsPatchEdge3D::VMin:
        return std::abs(element.v0() - patch.domain_start_v()) <= tolerance
            && overlaps(element.u0(), element.u1());
    case NurbsPatchEdge3D::VMax:
        return std::abs(element.v1() - patch.domain_end_v()) <= tolerance
            && overlaps(element.u0(), element.u1());
    }
    return false;
}

bool element_touches_non_g1_feature(
    const RationalBezierElement3D& element,
    const NurbsSurfaceModel3D& model)
{
    return std::any_of(
        model.connections().begin(), model.connections().end(),
        [&](const NurbsPatchEdgeConnection3D& connection) {
            return !connection.g1
                && (element_touches_interval(
                        element, connection.first, model)
                    || element_touches_interval(
                        element, connection.second, model));
        });
}

double mapped_connection_parameter(
    const NurbsPatchEdgeConnection3D& connection,
    double first_parameter)
{
    const double fraction =
        (first_parameter - connection.first.begin)
        / (connection.first.end - connection.first.begin);
    if (connection.reversed) {
        return connection.second.end
            - fraction * (connection.second.end - connection.second.begin);
    }
    return connection.second.begin
        + fraction * (connection.second.end - connection.second.begin);
}

bool roots_match_connection(const NurbsSurfaceCrossing3D& first_root,
                            const NurbsSurfaceCrossing3D& second_root,
                            const NurbsPatchEdgeConnection3D& connection,
                            const NurbsSurfaceModel3D& model)
{
    const auto matches_orientation =
        [&](const NurbsSurfaceCrossing3D& first,
            const NurbsSurfaceCrossing3D& second) {
            double first_parameter = 0.0;
            double second_parameter = 0.0;
            if (!root_on_interval(
                    first, connection.first, model, &first_parameter)
                || !root_on_interval(
                    second, connection.second, model,
                    &second_parameter)) {
                return false;
            }
            const double tolerance = std::max(
                parameter_tolerance(model.patch(first.patch_index)),
                parameter_tolerance(model.patch(second.patch_index)));
            return std::abs(mapped_connection_parameter(
                                connection, first_parameter)
                            - second_parameter)
                <= tolerance;
        };
    return matches_orientation(first_root, second_root)
        || matches_orientation(second_root, first_root);
}

std::optional<NurbsSurfaceCrossing3D> find_cartesian_tangent_witness(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end,
    double geometry_tolerance)
{
    const Eigen::Vector3d delta = end - start;
    const double length = delta.norm();
    const double squared_length = delta.squaredNorm();
    std::vector<double> seeds{0.0, 0.25, 0.5, 0.75, 1.0};
    for (const Eigen::Vector4d& control : element.homogeneous_controls) {
        const Eigen::Vector3d point = control.head<3>() / control.w();
        seeds.push_back(std::clamp(
            (point - start).dot(delta) / squared_length, 0.0, 1.0));
    }
    const Eigen::Vector3d element_center = element.evaluate(
        0.5 * (element.u0() + element.u1()),
        0.5 * (element.v0() + element.v1()));
    seeds.push_back(std::clamp(
        (element_center - start).dot(delta) / squared_length, 0.0, 1.0));

    for (double t : seeds) {
        NurbsSurfacePatch3D::ProjectionResult projection;
        bool valid_projection = false;
        for (int iteration = 0; iteration < 32; ++iteration) {
            try {
                projection = patch.project(start + t * delta);
            } catch (const std::exception&) {
                valid_projection = false;
                break;
            }
            if (!projection.point.allFinite()) {
                valid_projection = false;
                break;
            }
            valid_projection = true;
            const double next_t = std::clamp(
                (projection.point - start).dot(delta) / squared_length,
                0.0, 1.0);
            if (std::abs(next_t - t)
                <= segment_parameter_tolerance(
                    geometry_tolerance, length)) {
                t = next_t;
                break;
            }
            t = next_t;
        }
        if (!valid_projection)
            continue;
        try {
            projection = patch.project(start + t * delta);
        } catch (const std::exception&) {
            continue;
        }
        const double uv_tolerance = parameter_tolerance(patch);
        if (projection.u < element.u0() - uv_tolerance
            || projection.u > element.u1() + uv_tolerance
            || projection.v < element.v0() - uv_tolerance
            || projection.v > element.v1() + uv_tolerance) {
            continue;
        }
        const NurbsSurfaceDerivatives3D derivatives =
            patch.evaluate_with_derivatives(projection.u, projection.v);
        const Eigen::Vector3d residual_vector =
            derivatives.point - (start + t * delta);
        const double residual = residual_vector.norm();
        const Eigen::Vector3d cross = derivatives.du.cross(derivatives.dv);
        const double normal_norm = cross.norm();
        if (!std::isfinite(residual) || residual > geometry_tolerance
            || !std::isfinite(normal_norm) || normal_norm == 0.0) {
            continue;
        }
        const Eigen::Vector3d normal = cross / normal_norm;
        const double transversality =
            std::abs(normal.dot(delta / length));
        if (transversality <= 1e-10) {
            return NurbsSurfaceCrossing3D{
                element.patch_index,
                element.component,
                projection.u,
                projection.v,
                t,
                derivatives.point,
                normal,
                residual,
                transversality};
        }
    }
    return std::nullopt;
}

struct CartesianRulingCertificate {
    bool curve_along_u = true;
    int ruling_axis = -1;
    int support_axis = -1;
};

std::optional<CartesianRulingCertificate>
certify_single_root_cartesian_ruling(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end,
    int edge_axis,
    double geometry_tolerance)
{
    if (edge_axis < 0 || edge_axis >= 3)
        return std::nullopt;
    for (int axis = 0; axis < 3; ++axis) {
        if (axis != edge_axis && start[axis] != end[axis])
            return std::nullopt;
    }
    if (start[edge_axis] == end[edge_axis])
        return std::nullopt;

    const bool curve_along_u = element.degree_u == 2
        && element.degree_v == 1;
    const bool curve_along_v = element.degree_u == 1
        && element.degree_v == 2;
    if (!curve_along_u && !curve_along_v)
        return std::nullopt;
    if (patch.basis_u().degree() != element.degree_u
        || patch.basis_v().degree() != element.degree_v
        || patch.basis_u().num_basis_functions()
                != element.degree_u + 1
        || patch.basis_v().num_basis_functions()
                != element.degree_v + 1
        || element.u0() != patch.domain_start_u()
        || element.u1() != patch.domain_end_u()
        || element.v0() != patch.domain_start_v()
        || element.v1() != patch.domain_end_v()) {
        return std::nullopt;
    }

    const auto point = [&](int curved, int ruling) -> const Eigen::Vector3d& {
        const auto& controls = patch.control_net();
        return curve_along_u
            ? controls[static_cast<std::size_t>(curved)]
                      [static_cast<std::size_t>(ruling)]
            : controls[static_cast<std::size_t>(ruling)]
                      [static_cast<std::size_t>(curved)];
    };
    const auto weight = [&](int curved, int ruling) {
        const auto& weights = patch.weights();
        return curve_along_u
            ? weights[static_cast<std::size_t>(curved)]
                     [static_cast<std::size_t>(ruling)]
            : weights[static_cast<std::size_t>(ruling)]
                     [static_cast<std::size_t>(curved)];
    };

    for (int curved = 0; curved < 3; ++curved) {
        const double first_weight = weight(curved, 0);
        if (!std::isfinite(first_weight) || first_weight <= 0.0
            || weight(curved, 1) != first_weight) {
            return std::nullopt;
        }
    }

    int ruling_axis = -1;
    for (int candidate_axis = 0; candidate_axis < 3; ++candidate_axis) {
        if (candidate_axis == edge_axis)
            continue;
        bool matches = point(0, 0)[candidate_axis]
            != point(0, 1)[candidate_axis];
        for (int curved = 0; matches && curved < 3; ++curved) {
            for (int axis = 0; axis < 3; ++axis) {
                if (axis != candidate_axis
                    && point(curved, 0)[axis]
                        != point(curved, 1)[axis]) {
                    matches = false;
                }
            }
            for (int ruling = 0; matches && ruling < 2; ++ruling) {
                if (point(curved, ruling)[candidate_axis]
                    != point(0, ruling)[candidate_axis]) {
                    matches = false;
                }
            }
        }
        if (!matches)
            continue;
        if (ruling_axis >= 0)
            return std::nullopt;
        ruling_axis = candidate_axis;
    }
    if (ruling_axis < 0)
        return std::nullopt;

    const double ruling_begin = point(0, 0)[ruling_axis];
    const double ruling_end = point(0, 1)[ruling_axis];
    const double fixed_ruling = start[ruling_axis];
    if (fixed_ruling < std::min(ruling_begin, ruling_end)
                               - geometry_tolerance
        || fixed_ruling > std::max(ruling_begin, ruling_end)
                                + geometry_tolerance) {
        return std::nullopt;
    }

    int support_axis = -1;
    for (int axis = 0; axis < 3; ++axis) {
        if (axis != edge_axis && axis != ruling_axis)
            support_axis = axis;
    }
    if (support_axis < 0)
        return std::nullopt;

    std::array<CGAL::Sign, 3> signs{};
    int nonzero_count = 0;
    int variations = 0;
    CGAL::Sign previous = CGAL::ZERO;
    for (int curved = 0; curved < 3; ++curved) {
        const ExactScalar coefficient =
            ExactScalar(weight(curved, 0))
            * (ExactScalar(point(curved, 0)[support_axis])
               - ExactScalar(start[support_axis]));
        signs[static_cast<std::size_t>(curved)] = CGAL::sign(coefficient);
        const CGAL::Sign sign = signs[static_cast<std::size_t>(curved)];
        if (sign == CGAL::ZERO)
            continue;
        ++nonzero_count;
        if (previous != CGAL::ZERO && sign != previous)
            ++variations;
        previous = sign;
    }
    if (nonzero_count == 0)
        return std::nullopt;
    const int endpoint_roots =
        (signs.front() == CGAL::ZERO ? 1 : 0)
        + (signs.back() == CGAL::ZERO ? 1 : 0);
    if (endpoint_roots + variations > 1)
        return std::nullopt;

    return CartesianRulingCertificate{
        curve_along_u, ruling_axis, support_axis};
}

std::optional<NurbsSurfaceCrossing3D>
certified_complete_cartesian_tangent(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const NurbsElementIntersectionResult3D* partial,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end,
    int edge_axis,
    double geometry_tolerance)
{
    if ((partial != nullptr && partial->overlap_detected)
        || !certify_single_root_cartesian_ruling(
            element, patch, start, end, edge_axis,
            geometry_tolerance)) {
        return std::nullopt;
    }
    // This numerical witness is considered only after the exact Bernstein
    // certificate above has bounded the Cartesian query to one root.
    const auto tangent = find_cartesian_tangent_witness(
        element, patch, start, end, geometry_tolerance);
    if (!tangent)
        return std::nullopt;

    const double segment_length = (end - start).norm();
    const double edge_tolerance = segment_parameter_tolerance(
        geometry_tolerance, segment_length);
    if (partial != nullptr) {
        for (const NurbsElementRoot3D& root : partial->roots) {
            if (std::abs(root.t - tangent->edge_parameter) > edge_tolerance
                || (root.point - tangent->point).norm()
                    > 16.0 * geometry_tolerance) {
                return std::nullopt;
            }
        }
    }
    return tangent;
}

class FeatureEdgeAlignment : public std::runtime_error {
public:
    explicit FeatureEdgeAlignment(
        std::vector<NurbsSurfaceCrossing3D> roots)
        : std::runtime_error("non-G1 feature-edge alignment")
        , roots_(std::move(roots))
    {
    }

    const std::vector<NurbsSurfaceCrossing3D>& roots() const
    {
        return roots_;
    }

private:
    std::vector<NurbsSurfaceCrossing3D> roots_;
};

class UnrelatedPatchCoincidence : public std::runtime_error {
public:
    explicit UnrelatedPatchCoincidence(
        std::vector<NurbsSurfaceCrossing3D> roots)
        : std::runtime_error(
            "coincident roots on unrelated NURBS patches")
        , roots_(std::move(roots))
    {
    }

    const std::vector<NurbsSurfaceCrossing3D>& roots() const
    {
        return roots_;
    }

private:
    std::vector<NurbsSurfaceCrossing3D> roots_;
};

class DisjointSets {
public:
    explicit DisjointSets(int size)
        : parents_(static_cast<std::size_t>(size))
    {
        std::iota(parents_.begin(), parents_.end(), 0);
    }

    int find(int item)
    {
        int& parent = parents_[static_cast<std::size_t>(item)];
        if (parent != item)
            parent = find(parent);
        return parent;
    }

    void unite(int first, int second)
    {
        first = find(first);
        second = find(second);
        if (first == second)
            return;
        if (first > second)
            std::swap(first, second);
        parents_[static_cast<std::size_t>(second)] = first;
    }

private:
    std::vector<int> parents_;
};

std::vector<NurbsSurfaceCrossing3D> canonicalize_roots(
    std::vector<NurbsSurfaceCrossing3D> roots,
    const NurbsSurfaceModel3D& model,
    double geometry_tolerance,
    double segment_length,
    NurbsSurfaceIntersectionDiagnostics3D& diagnostics)
{
    std::sort(roots.begin(), roots.end(),
              [](const auto& first, const auto& second) {
                  if (first.patch_index != second.patch_index)
                      return first.patch_index < second.patch_index;
                  if (first.u != second.u)
                      return first.u < second.u;
                  if (first.v != second.v)
                      return first.v < second.v;
                  return first.edge_parameter < second.edge_parameter;
              });

    std::vector<NurbsSurfaceCrossing3D> patch_unique;
    for (const auto& root : roots) {
        const auto duplicate = std::find_if(
            patch_unique.begin(), patch_unique.end(),
            [&](const auto& existing) {
                return same_patch_root(existing, root, model,
                                       geometry_tolerance, segment_length);
            });
        if (duplicate == patch_unique.end()) {
            patch_unique.push_back(root);
        } else {
            ++diagnostics.same_patch_deduplications;
            const double minimum_transversality = std::min(
                duplicate->transversality, root.transversality);
            if (root.residual < duplicate->residual)
                *duplicate = root;
            duplicate->transversality = minimum_transversality;
        }
    }

    for (const auto& root : patch_unique) {
        for (const auto& connection : model.connections()) {
            if (!connection.g1
                && (root_on_interval(root, connection.first, model)
                    || root_on_interval(root, connection.second, model))) {
                throw FeatureEdgeAlignment(patch_unique);
            }
        }
    }

    DisjointSets seams(static_cast<int>(patch_unique.size()));
    for (int first = 0; first < static_cast<int>(patch_unique.size()); ++first) {
        for (int second = first + 1;
             second < static_cast<int>(patch_unique.size()); ++second) {
            if (!coincident_root(
                    patch_unique[static_cast<std::size_t>(first)],
                    patch_unique[static_cast<std::size_t>(second)],
                    geometry_tolerance, segment_length)) {
                continue;
            }
            const bool declared_smooth_seam = std::any_of(
                model.connections().begin(), model.connections().end(),
                [&](const auto& connection) {
                    return connection.g1
                        && roots_match_connection(
                            patch_unique[static_cast<std::size_t>(first)],
                            patch_unique[static_cast<std::size_t>(second)],
                            connection, model);
                });
            if (declared_smooth_seam)
                seams.unite(first, second);
        }
    }

    for (int first = 0; first < static_cast<int>(patch_unique.size()); ++first) {
        for (int second = first + 1;
             second < static_cast<int>(patch_unique.size()); ++second) {
            if (patch_unique[static_cast<std::size_t>(first)].patch_index
                    != patch_unique[static_cast<std::size_t>(second)].patch_index
                && coincident_root(
                    patch_unique[static_cast<std::size_t>(first)],
                    patch_unique[static_cast<std::size_t>(second)],
                    geometry_tolerance, segment_length)
                && seams.find(first) != seams.find(second)) {
                throw UnrelatedPatchCoincidence(patch_unique);
            }
        }
    }

    std::vector<NurbsSurfaceCrossing3D> canonical;
    std::vector<int> representative(patch_unique.size(), -1);
    for (int root_index = 0;
         root_index < static_cast<int>(patch_unique.size()); ++root_index) {
        const int group = seams.find(root_index);
        int& output = representative[static_cast<std::size_t>(group)];
        if (output < 0) {
            output = static_cast<int>(canonical.size());
            canonical.push_back(
                patch_unique[static_cast<std::size_t>(root_index)]);
        } else {
            ++diagnostics.seam_deduplications;
            NurbsSurfaceCrossing3D& current =
                canonical[static_cast<std::size_t>(output)];
            const NurbsSurfaceCrossing3D& candidate =
                patch_unique[static_cast<std::size_t>(root_index)];
            const double minimum_transversality = std::min(
                current.transversality, candidate.transversality);
            if (candidate.residual < current.residual)
                current = candidate;
            current.transversality = minimum_transversality;
        }
    }

    std::sort(canonical.begin(), canonical.end(),
              [](const auto& first, const auto& second) {
                  if (first.edge_parameter != second.edge_parameter) {
                      return first.edge_parameter < second.edge_parameter;
                  }
                  if (first.patch_index != second.patch_index)
                      return first.patch_index < second.patch_index;
                  if (first.u != second.u)
                      return first.u < second.u;
                  return first.v < second.v;
              });
    return canonical;
}

std::string cartesian_edge_diagnostic(
    const char* prefix,
    const NurbsCartesianEdgeQuery3D& edge,
    const std::vector<NurbsSurfaceCrossing3D>& roots)
{
    std::ostringstream out;
    out << std::setprecision(17);
    out << prefix << " axis=" << edge.axis
        << " (" << edge.i << ',' << edge.j << ',' << edge.k << ')';
    for (const auto& root : roots) {
        out << " root=(patch=" << root.patch_index
            << ",u=" << root.u << ",v=" << root.v
            << ",t=" << root.edge_parameter
            << ",residual=" << root.residual
            << ",transversality=" << root.transversality << ')';
    }
    return out.str();
}

bool is_retryable_ray_degeneracy(const std::string& message)
{
    return message.find("surface intersects Cartesian node")
                != std::string::npos
        || message.find("surface overlaps Cartesian edge")
                != std::string::npos
        || message.find("non-G1 feature-edge alignment")
                != std::string::npos
        || message.find("tangential Cartesian edge contact")
                != std::string::npos;
}

} // namespace

NurbsSurfaceIntersector3D::NurbsSurfaceIntersector3D(
    NurbsSurfaceModel3D model,
    NurbsSurfaceIntersectorOptions3D options)
    : model_(std::move(model))
    , options_(options)
{
    (void)model_.validate_closed();
    bounds_ = model_.control_bounds();
    geometry_tolerance_ =
        std::max(1e-12 * bounds_.diameter(), 1e-14);
    elements_ = extract_rational_bezier_elements_3d(model_);
    if (options_.bvh_leaf_size <= 0)
        throw std::invalid_argument("NURBS BVH leaf size must be positive");
    element_order_.resize(elements_.size());
    std::iota(element_order_.begin(), element_order_.end(), 0);
    if (!elements_.empty()) {
        bvh_nodes_.reserve(2 * elements_.size());
        bvh_root_ = build_bvh_node(
            0, static_cast<int>(element_order_.size()));
    }
}

const NurbsSurfaceModel3D& NurbsSurfaceIntersector3D::model() const
{
    return model_;
}

const NurbsAabb3D& NurbsSurfaceIntersector3D::bounds() const
{
    return bounds_;
}

double NurbsSurfaceIntersector3D::geometry_tolerance() const
{
    return geometry_tolerance_;
}

int NurbsSurfaceIntersector3D::build_bvh_node(int begin, int end)
{
    NurbsAabb3D node_bounds =
        elements_[static_cast<std::size_t>(
            element_order_[static_cast<std::size_t>(begin)])].bounds();
    for (int item = begin + 1; item < end; ++item) {
        node_bounds = united_bounds(
            node_bounds,
            elements_[static_cast<std::size_t>(
                element_order_[static_cast<std::size_t>(item)])].bounds());
    }

    const int node = static_cast<int>(bvh_nodes_.size());
    bvh_nodes_.push_back({node_bounds, -1, -1, begin, end});
    if (end - begin <= options_.bvh_leaf_size)
        return node;

    Eigen::Index axis = 0;
    (node_bounds.upper - node_bounds.lower).maxCoeff(&axis);
    const int middle = begin + (end - begin) / 2;
    std::nth_element(
        element_order_.begin() + begin,
        element_order_.begin() + middle,
        element_order_.begin() + end,
        [&](int first, int second) {
            const NurbsAabb3D first_bounds =
                elements_[static_cast<std::size_t>(first)].bounds();
            const NurbsAabb3D second_bounds =
                elements_[static_cast<std::size_t>(second)].bounds();
            const double first_center =
                0.5 * (first_bounds.lower[axis] + first_bounds.upper[axis]);
            const double second_center =
                0.5 * (second_bounds.lower[axis] + second_bounds.upper[axis]);
            if (first_center != second_center)
                return first_center < second_center;
            return first < second;
        });
    const int left = build_bvh_node(begin, middle);
    const int right = build_bvh_node(middle, end);
    bvh_nodes_[static_cast<std::size_t>(node)].left = left;
    bvh_nodes_[static_cast<std::size_t>(node)].right = right;
    bvh_nodes_[static_cast<std::size_t>(node)].begin = 0;
    bvh_nodes_[static_cast<std::size_t>(node)].end = 0;
    return node;
}

void NurbsSurfaceIntersector3D::collect_candidate_elements(
    int node,
    const NurbsAabb3D& query_bounds,
    std::vector<int>& candidates) const
{
    if (node < 0)
        return;
    const BvhNode& bvh_node = bvh_nodes_[static_cast<std::size_t>(node)];
    if (!bvh_node.bounds.overlaps(query_bounds, geometry_tolerance_))
        return;
    if (bvh_node.left < 0) {
        for (int item = bvh_node.begin; item < bvh_node.end; ++item) {
            const int element =
                element_order_[static_cast<std::size_t>(item)];
            if (elements_[static_cast<std::size_t>(element)].bounds().overlaps(
                    query_bounds, geometry_tolerance_)) {
                candidates.push_back(element);
            }
        }
        return;
    }
    collect_candidate_elements(bvh_node.left, query_bounds, candidates);
    collect_candidate_elements(bvh_node.right, query_bounds, candidates);
}

NurbsSurfaceIntersectionResult3D
NurbsSurfaceIntersector3D::intersect_segment(
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end) const
{
    return intersect_segment_impl(start, end, nullptr);
}

NurbsSurfaceIntersectionResult3D
NurbsSurfaceIntersector3D::intersect_segment_impl(
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end,
    const NurbsCartesianEdgeQuery3D* cartesian_edge) const
{
    if (!start.allFinite() || !end.allFinite())
        throw std::invalid_argument("NURBS intersection segment must be finite");
    const double segment_length = (end - start).norm();
    if (!std::isfinite(segment_length)
        || segment_length <= geometry_tolerance_) {
        throw std::invalid_argument(
            "NURBS intersection segment endpoints must be distinct");
    }

    std::vector<int> candidates;
    collect_candidate_elements(
        bvh_root_, segment_bounds(start, end), candidates);
    std::sort(candidates.begin(), candidates.end());

    NurbsSurfaceIntersectionResult3D result;
    std::vector<NurbsSurfaceCrossing3D> roots;
    std::vector<NurbsSurfaceCrossing3D> tangential_roots;
    bool unresolved = false;
    const auto accumulate_local =
        [&](const NurbsElementIntersectionResult3D& local) {
            result.diagnostics.triangle_seed_hits +=
                local.diagnostics.triangle_seed_hits;
            result.diagnostics.triangle_seed_misses_recovered +=
                local.diagnostics.roots_recovered_without_triangle_seed;
            result.diagnostics.subdivision_boxes +=
                local.diagnostics.subdivision_boxes;
            result.diagnostics.newton_attempts +=
                local.diagnostics.newton_attempts;
            result.diagnostics.newton_iterations +=
                local.diagnostics.newton_iterations;
            result.diagnostics.unresolved_candidates +=
                local.diagnostics.unresolved_boxes;
            result.overlap_detected =
                result.overlap_detected || local.overlap_detected;
            for (const NurbsElementRoot3D& root : local.roots)
                roots.push_back(as_surface_crossing(root));
        };
    const auto accumulate_probe_work =
        [&](const NurbsElementIntersectionResult3D& probe) {
            result.diagnostics.triangle_seed_hits +=
                probe.diagnostics.triangle_seed_hits;
            result.diagnostics.triangle_seed_misses_recovered +=
                probe.diagnostics.roots_recovered_without_triangle_seed;
            result.diagnostics.subdivision_boxes +=
                probe.diagnostics.subdivision_boxes;
            result.diagnostics.newton_attempts +=
                probe.diagnostics.newton_attempts;
            result.diagnostics.newton_iterations +=
                probe.diagnostics.newton_iterations;
        };


    NurbsElementIntersectionOptions3D local_options;
    local_options.geometry_tolerance = geometry_tolerance_;
    local_options.use_triangle_seed = options_.use_triangle_seeds;
    for (const int candidate : candidates) {
        ++result.diagnostics.candidate_elements;
        const RationalBezierElement3D& element =
            elements_[static_cast<std::size_t>(candidate)];
        if (cartesian_edge == nullptr) {
            const NurbsElementIntersectionResult3D local =
                intersect_nurbs_bezier_element_3d(
                    element, model_.patch(element.patch_index),
                    start, end, local_options);
            accumulate_local(local);
            continue;
        }

        if (cartesian_edge != nullptr) {
            if (const auto tangent = certified_complete_cartesian_tangent(
                    element, model_.patch(element.patch_index), nullptr,
                    start, end, cartesian_edge->axis,
                    geometry_tolerance_)) {
                tangential_roots.push_back(*tangent);
                continue;
            }
        }

        if (element_touches_non_g1_feature(element, model_)) {
            NurbsElementIntersectionOptions3D probe_options = local_options;
            probe_options.max_subdivision_depth =
                kFeatureProbeSubdivisionDepth;
            try {
                const NurbsElementIntersectionResult3D local =
                    intersect_nurbs_bezier_element_3d(
                        element, model_.patch(element.patch_index),
                        start, end, probe_options);
                accumulate_local(local);
                continue;
            } catch (
                const UnresolvedNurbsIntersectionCandidate3D& probe_error) {
                const NurbsElementIntersectionResult3D& partial =
                    probe_error.partial_result();
                const bool feature_root = std::any_of(
                    partial.roots.begin(), partial.roots.end(),
                    [&](const NurbsElementRoot3D& root) {
                        return root_on_non_g1_feature(
                            as_surface_crossing(root), model_);
                    });
                result.overlap_detected =
                    result.overlap_detected || partial.overlap_detected;
                if (feature_root) {
                    for (const NurbsElementRoot3D& root : partial.roots)
                        roots.push_back(as_surface_crossing(root));
                }
                accumulate_probe_work(partial);
            }
        }

        try {
            const NurbsElementIntersectionResult3D local =
                intersect_nurbs_bezier_element_3d(
                    element, model_.patch(element.patch_index),
                    start, end, local_options);
            accumulate_local(local);
        } catch (const UnresolvedNurbsIntersectionCandidate3D& error) {
            const NurbsElementIntersectionResult3D& partial =
                error.partial_result();
            accumulate_local(partial);
            const auto tangent = certified_complete_cartesian_tangent(
                element, model_.patch(element.patch_index),
                &partial, start, end, cartesian_edge->axis,
                geometry_tolerance_);
            if (tangent)
                tangential_roots.push_back(*tangent);
            else
                unresolved = true;
        }
    }

    if (cartesian_edge == nullptr) {
        result.crossings = canonicalize_roots(
            std::move(roots), model_, geometry_tolerance_, segment_length,
            result.diagnostics);
        return result;
    }
    std::vector<NurbsSurfaceCrossing3D> all_roots = roots;
    all_roots.insert(
        all_roots.end(), tangential_roots.begin(), tangential_roots.end());
    if (result.overlap_detected) {
        throw std::runtime_error(cartesian_edge_diagnostic(
            "surface overlaps Cartesian edge", *cartesian_edge, all_roots));
    }
    const double endpoint_tolerance =
        segment_parameter_tolerance(geometry_tolerance_, segment_length);
    if (std::any_of(all_roots.begin(), all_roots.end(), [&](const auto& root) {
            return root.edge_parameter <= endpoint_tolerance
                || root.edge_parameter >= 1.0 - endpoint_tolerance;
        })) {
        throw std::runtime_error(cartesian_edge_diagnostic(
            "surface intersects Cartesian node", *cartesian_edge,
            all_roots));
    }
    if (std::any_of(all_roots.begin(), all_roots.end(), [&](const auto& root) {
            return root_on_non_g1_feature(root, model_);
        })) {
        throw FeatureEdgeAlignment(std::move(all_roots));
    }
    if (unresolved) {
        throw std::runtime_error(cartesian_edge_diagnostic(
            "unresolved conservative NURBS intersection candidate",
            *cartesian_edge, all_roots));
    }

    result.crossings = canonicalize_roots(
        std::move(all_roots), model_, geometry_tolerance_, segment_length,
        result.diagnostics);
    if (!tangential_roots.empty()
        || std::any_of(
            result.crossings.begin(), result.crossings.end(),
            [](const auto& root) {
                return root.transversality <= 1e-10;
            })) {
        throw std::runtime_error(cartesian_edge_diagnostic(
            "tangential Cartesian edge contact", *cartesian_edge,
            result.crossings));
    }
    if (result.crossings.size() > 1) {
        throw std::runtime_error(cartesian_edge_diagnostic(
            "multiple crossings on Cartesian edge", *cartesian_edge,
            result.crossings));
    }
    return result;
}

std::optional<NurbsSurfaceCrossing3D>
NurbsSurfaceIntersector3D::intersect_cartesian_edge(
    const NurbsCartesianEdgeQuery3D& edge,
    NurbsSurfaceIntersectionDiagnostics3D* diagnostics) const
{
    NurbsSurfaceIntersectionResult3D result;
    try {
        result = intersect_segment_impl(
            edge.start, edge.end, &edge);
    } catch (const FeatureEdgeAlignment& alignment) {
        throw std::runtime_error(cartesian_edge_diagnostic(
            "non-G1 feature-edge alignment", edge, alignment.roots()));
    } catch (const UnrelatedPatchCoincidence& coincidence) {
        throw std::runtime_error(cartesian_edge_diagnostic(
            "coincident roots on unrelated NURBS patches", edge,
            coincidence.roots()));
    }
    if (diagnostics != nullptr)
        *diagnostics = result.diagnostics;
    const auto& roots = result.crossings;
    if (roots.empty())
        return std::nullopt;
    return roots.front();
}

std::vector<int> NurbsSurfaceIntersector3D::containing_components(
    const Eigen::Vector3d& point) const
{
    if (!point.allFinite())
        throw std::invalid_argument("NURBS classification point must be finite");
    constexpr std::array<std::array<double, 3>, 3> direction_values{{
        {{1.0, 0.3713906763541037, 0.6947465906068658}},
        {{0.6180339887498948, 1.0, 0.4142135623730950}},
        {{0.2718281828459045, 0.5772156649015329, 1.0}}
    }};

    for (const auto& values : direction_values) {
        const Eigen::Vector3d direction(values[0], values[1], values[2]);
        const Eigen::Vector3d unit_direction = direction.normalized();
        double distance = 0.0;
        for (Eigen::Index axis = 0; axis < 3; ++axis) {
            distance = std::max(
                distance,
                (bounds_.upper[axis] - point[axis]) / unit_direction[axis]);
        }
        distance += std::max(bounds_.diameter(), 1.0);
        if (!std::isfinite(distance))
            throw std::overflow_error("NURBS classification ray is not finite");
        const Eigen::Vector3d end = point + distance * unit_direction;

        try {
            const NurbsSurfaceIntersectionResult3D result =
                intersect_segment(point, end);
            NurbsCartesianEdgeQuery3D ray;
            ray.axis = -1;
            ray.start = point;
            ray.end = end;
            if (result.overlap_detected) {
                throw std::runtime_error(cartesian_edge_diagnostic(
                    "surface overlaps Cartesian edge", ray,
                    result.crossings));
            }
            const double endpoint_tolerance =
                segment_parameter_tolerance(
                    geometry_tolerance_, distance);
            if (std::any_of(
                    result.crossings.begin(), result.crossings.end(),
                    [&](const auto& root) {
                        return root.edge_parameter <= endpoint_tolerance
                            || root.edge_parameter
                                >= 1.0 - endpoint_tolerance;
                    })) {
                throw std::runtime_error(cartesian_edge_diagnostic(
                    "surface intersects Cartesian node", ray,
                    result.crossings));
            }
            if (std::any_of(
                    result.crossings.begin(), result.crossings.end(),
                    [](const auto& root) {
                        return root.transversality <= 1e-10;
                    })) {
                throw std::runtime_error(cartesian_edge_diagnostic(
                    "tangential Cartesian edge contact", ray,
                    result.crossings));
            }

            std::vector<bool> odd(
                static_cast<std::size_t>(model_.num_components()), false);
            for (const auto& crossing : result.crossings) {
                odd[static_cast<std::size_t>(crossing.component)] =
                    !odd[static_cast<std::size_t>(crossing.component)];
            }
            std::vector<int> components;
            for (int component = 0;
                 component < static_cast<int>(odd.size()); ++component) {
                if (odd[static_cast<std::size_t>(component)])
                    components.push_back(component);
            }
            return components;
        } catch (const std::runtime_error& error) {
            if (!is_retryable_ray_degeneracy(error.what()))
                throw;
        }
    }
    throw std::runtime_error(
        "unable to classify point with three deterministic NURBS rays");
}

std::vector<RationalBezierElement3D>
NurbsSurfaceIntersector3D::acceleration_leaves(
    double maximum_extent) const
{
    return subdivide_rational_bezier_elements_to_extent_3d(
        elements_, maximum_extent);
}

} // namespace kfbim::geometry3d
