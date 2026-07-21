#include "nurbs_surface_model_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace kfbim::geometry3d {

namespace {

struct EdgeSample3D {
    Eigen::Vector3d point;
    Eigen::Vector3d positive_parameter_tangent;
    Eigen::Vector3d oriented_boundary_tangent;
};

bool is_u_edge(NurbsPatchEdge3D edge)
{
    return edge == NurbsPatchEdge3D::UMin || edge == NurbsPatchEdge3D::UMax;
}

std::pair<double, double> edge_parameter_domain(
    const NurbsSurfacePatch3D& patch,
    NurbsPatchEdge3D edge)
{
    if (is_u_edge(edge))
        return {patch.domain_start_v(), patch.domain_end_v()};
    return {patch.domain_start_u(), patch.domain_end_u()};
}

EdgeSample3D edge_sample(const NurbsSurfacePatch3D& patch,
                         NurbsPatchEdge3D edge,
                         double parameter)
{
    double u = patch.domain_start_u();
    double v = patch.domain_start_v();
    switch (edge) {
    case NurbsPatchEdge3D::UMin:
        v = parameter;
        break;
    case NurbsPatchEdge3D::UMax:
        u = patch.domain_end_u();
        v = parameter;
        break;
    case NurbsPatchEdge3D::VMin:
        u = parameter;
        break;
    case NurbsPatchEdge3D::VMax:
        u = parameter;
        v = patch.domain_end_v();
        break;
    }
    const auto d = patch.evaluate_with_derivatives(u, v);
    const Eigen::Vector3d positive =
        is_u_edge(edge) ? d.dv : d.du;
    Eigen::Vector3d boundary = positive;
    if (edge == NurbsPatchEdge3D::UMin || edge == NurbsPatchEdge3D::VMax)
        boundary = -boundary;
    return {d.point, positive, boundary};
}

Eigen::Vector3d patch_normal(const NurbsSurfacePatch3D& patch,
                             NurbsPatchEdge3D edge,
                             double parameter)
{
    double u = patch.domain_start_u();
    double v = patch.domain_start_v();
    switch (edge) {
    case NurbsPatchEdge3D::UMin:
        v = parameter;
        break;
    case NurbsPatchEdge3D::UMax:
        u = patch.domain_end_u();
        v = parameter;
        break;
    case NurbsPatchEdge3D::VMin:
        u = parameter;
        break;
    case NurbsPatchEdge3D::VMax:
        u = parameter;
        v = patch.domain_end_v();
        break;
    }
    const auto d = patch.evaluate_with_derivatives(u, v);
    const Eigen::Vector3d normal = d.du.cross(d.dv);
    const double norm = normal.norm();
    if (!std::isfinite(norm) || norm == 0.0)
        return Eigen::Vector3d::Zero();
    return normal / norm;
}

double mapped_parameter(const NurbsPatchEdgeConnection3D& connection,
                        double first_parameter)
{
    const double fraction = (first_parameter - connection.first.begin)
        / (connection.first.end - connection.first.begin);
    if (connection.reversed) {
        return connection.second.end
            - fraction * (connection.second.end - connection.second.begin);
    }
    return connection.second.begin
        + fraction * (connection.second.end - connection.second.begin);
}

void require_valid_patch_index(int patch, int patch_count, const char* context)
{
    if (patch < 0 || patch >= patch_count)
        throw std::out_of_range(std::string(context) + " has invalid patch index");
}

void validate_interval(const NurbsPatchEdgeInterval3D& interval,
                       const std::vector<NurbsSurfacePatch3D>& patches,
                       const char* context)
{
    require_valid_patch_index(interval.patch, static_cast<int>(patches.size()), context);
    if (!std::isfinite(interval.begin) || !std::isfinite(interval.end)
        || interval.begin >= interval.end) {
        throw std::invalid_argument(std::string(context)
                                    + " has non-increasing interval");
    }
    const auto domain = edge_parameter_domain(
        patches[static_cast<std::size_t>(interval.patch)], interval.edge);
    const double scale = std::max({1.0, std::abs(domain.first), std::abs(domain.second)});
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (interval.begin < domain.first - tolerance
        || interval.end > domain.second + tolerance) {
        throw std::out_of_range(std::string(context)
                                + " lies outside its patch-edge parameter domain");
    }
}

int count_components(const std::vector<int>& components)
{
    if (components.empty())
        return 0;
    const int maximum = *std::max_element(components.begin(), components.end());
    if (maximum < 0)
        throw std::invalid_argument("NURBS patch component index is invalid");
    std::vector<bool> present(static_cast<std::size_t>(maximum + 1), false);
    for (const int component : components) {
        if (component < 0)
            throw std::invalid_argument("NURBS patch component index is invalid");
        present[static_cast<std::size_t>(component)] = true;
    }
    if (std::find(present.begin(), present.end(), false) != present.end())
        throw std::invalid_argument("NURBS patch component IDs must be contiguous");
    return maximum + 1;
}

} // namespace

bool NurbsAabb3D::overlaps(const NurbsAabb3D& other, double tolerance) const
{
    if (!std::isfinite(tolerance) || tolerance < 0.0)
        throw std::invalid_argument("NURBS AABB tolerance must be nonnegative and finite");
    return (lower.array() <= other.upper.array() + tolerance).all()
        && (other.lower.array() <= upper.array() + tolerance).all();
}

bool NurbsAabb3D::contains(const Eigen::Vector3d& point, double tolerance) const
{
    if (!std::isfinite(tolerance) || tolerance < 0.0)
        throw std::invalid_argument("NURBS AABB tolerance must be nonnegative and finite");
    return point.allFinite()
        && (point.array() >= lower.array() - tolerance).all()
        && (point.array() <= upper.array() + tolerance).all();
}

double NurbsAabb3D::max_extent() const
{
    return (upper - lower).maxCoeff();
}

double NurbsAabb3D::diameter() const
{
    return (upper - lower).norm();
}

NurbsSurfaceModel3D::NurbsSurfaceModel3D(
    std::vector<NurbsSurfacePatch3D> patches,
    std::vector<int> patch_components,
    std::vector<NurbsPatchEdgeConnection3D> connections)
    : patches_(std::move(patches))
    , patch_components_(std::move(patch_components))
    , connections_(std::move(connections))
{
    if (patch_components_.size() != patches_.size()) {
        throw std::invalid_argument(
            "NURBS patch component count must match patch count");
    }
    (void)count_components(patch_components_);
    for (const auto& connection : connections_) {
        validate_interval(connection.first, patches_, "NURBS connection first edge");
        validate_interval(connection.second, patches_, "NURBS connection second edge");
        if (patch_components_[static_cast<std::size_t>(connection.first.patch)]
            != patch_components_[static_cast<std::size_t>(connection.second.patch)]) {
            throw std::invalid_argument(
                "NURBS connection joins different components");
        }
    }
}

int NurbsSurfaceModel3D::num_patches() const
{
    return static_cast<int>(patches_.size());
}

int NurbsSurfaceModel3D::num_components() const
{
    return count_components(patch_components_);
}

const NurbsSurfacePatch3D& NurbsSurfaceModel3D::patch(int patch_index) const
{
    require_valid_patch_index(patch_index, num_patches(), "NURBS model patch access");
    return patches_[static_cast<std::size_t>(patch_index)];
}

int NurbsSurfaceModel3D::patch_component(int patch_index) const
{
    require_valid_patch_index(patch_index, num_patches(), "NURBS model patch component access");
    return patch_components_[static_cast<std::size_t>(patch_index)];
}

const std::vector<NurbsSurfacePatch3D>& NurbsSurfaceModel3D::patches() const
{
    return patches_;
}

const std::vector<NurbsPatchEdgeConnection3D>& NurbsSurfaceModel3D::connections() const
{
    return connections_;
}

NurbsSurfaceTopologyDiagnostics3D NurbsSurfaceModel3D::validate_closed(
    double relative_tolerance) const
{
    if (!std::isfinite(relative_tolerance) || relative_tolerance < 0.0)
        throw std::invalid_argument("NURBS topology tolerance must be nonnegative and finite");

    NurbsSurfaceTopologyDiagnostics3D diagnostic;
    diagnostic.patch_count = num_patches();
    diagnostic.component_count = num_components();
    diagnostic.connection_count = static_cast<int>(connections_.size());

    for (int patch_index = 0; patch_index < num_patches(); ++patch_index) {
        const auto& surface_patch = patches_[static_cast<std::size_t>(patch_index)];
        for (const NurbsPatchEdge3D edge : {NurbsPatchEdge3D::UMin,
                                            NurbsPatchEdge3D::UMax,
                                            NurbsPatchEdge3D::VMin,
                                            NurbsPatchEdge3D::VMax}) {
            const auto domain = edge_parameter_domain(surface_patch, edge);
            std::vector<double> endpoints{domain.first, domain.second};
            std::vector<const NurbsPatchEdgeInterval3D*> incidents;
            for (const auto& connection : connections_) {
                for (const auto* interval : {&connection.first, &connection.second}) {
                    if (interval->patch == patch_index && interval->edge == edge) {
                        endpoints.push_back(interval->begin);
                        endpoints.push_back(interval->end);
                        incidents.push_back(interval);
                    }
                }
            }
            std::sort(endpoints.begin(), endpoints.end());
            const double parameter_tolerance = 64.0 * std::numeric_limits<double>::epsilon()
                * std::max({1.0, std::abs(domain.first), std::abs(domain.second)});
            endpoints.erase(std::unique(endpoints.begin(), endpoints.end(),
                                        [parameter_tolerance](double a, double b) {
                                            return std::abs(a - b) <= parameter_tolerance;
                                        }), endpoints.end());
            for (std::size_t i = 1; i < endpoints.size(); ++i) {
                const double begin = endpoints[i - 1];
                const double end = endpoints[i];
                if (end <= begin + parameter_tolerance)
                    continue;
                const double midpoint = 0.5 * (begin + end);
                int coverage = 0;
                for (const auto* interval : incidents) {
                    if (interval->begin <= midpoint + parameter_tolerance
                        && interval->end >= midpoint - parameter_tolerance) {
                        ++coverage;
                    }
                }
                if (coverage == 0)
                    ++diagnostic.uncovered_interval_count;
                else if (coverage != 1)
                    ++diagnostic.multiply_covered_interval_count;
            }
        }
    }

    const double model_diameter = control_bounds().diameter();
    const double physical_tolerance = std::max(
        1.0e-14, relative_tolerance * model_diameter);
    for (const auto& connection : connections_) {
        const auto& first_patch = patches_[static_cast<std::size_t>(connection.first.patch)];
        const auto& second_patch = patches_[static_cast<std::size_t>(connection.second.patch)];
        for (int sample = 0; sample < 9; ++sample) {
            const double fraction = static_cast<double>(sample) / 8.0;
            const double first_parameter = connection.first.begin
                + fraction * (connection.first.end - connection.first.begin);
            const double second_parameter = mapped_parameter(connection, first_parameter);
            const EdgeSample3D first = edge_sample(
                first_patch, connection.first.edge, first_parameter);
            const EdgeSample3D second = edge_sample(
                second_patch, connection.second.edge, second_parameter);
            if ((first.point - second.point).norm() > physical_tolerance)
                ++diagnostic.position_mismatch_count;

            const double first_tangent_norm = first.oriented_boundary_tangent.norm();
            const double second_tangent_norm = second.oriented_boundary_tangent.norm();
            const double tangent_dot = first_tangent_norm > 0.0 && second_tangent_norm > 0.0
                ? first.oriented_boundary_tangent.dot(second.oriented_boundary_tangent)
                    / (first_tangent_norm * second_tangent_norm)
                : 1.0;
            if (!std::isfinite(tangent_dot) || tangent_dot > -1.0 + 1.0e-8)
                ++diagnostic.orientation_mismatch_count;

            if (connection.g1) {
                const Eigen::Vector3d first_normal = patch_normal(
                    first_patch, connection.first.edge, first_parameter);
                const Eigen::Vector3d second_normal = patch_normal(
                    second_patch, connection.second.edge, second_parameter);
                const double normal_dot = first_normal.dot(second_normal);
                if (!std::isfinite(normal_dot) || normal_dot < 1.0 - 1.0e-8)
                    ++diagnostic.g1_normal_mismatch_count;
            }
        }
    }

    if (diagnostic.uncovered_interval_count != 0)
        throw std::runtime_error("uncovered patch-edge interval");
    if (diagnostic.multiply_covered_interval_count != 0)
        throw std::runtime_error("multiply covered patch-edge interval");
    if (diagnostic.position_mismatch_count != 0)
        throw std::runtime_error("connected NURBS edges disagree in position");
    if (diagnostic.orientation_mismatch_count != 0)
        throw std::runtime_error("connected NURBS edges have inconsistent orientation");
    if (diagnostic.g1_normal_mismatch_count != 0)
        throw std::runtime_error("declared G1 NURBS edges disagree in normal");
    return diagnostic;
}

NurbsAabb3D NurbsSurfaceModel3D::control_bounds() const
{
    if (patches_.empty())
        return {};
    NurbsAabb3D bounds;
    bounds.lower = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
    bounds.upper = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
    for (const auto& surface_patch : patches_) {
        for (const auto& row : surface_patch.control_net()) {
            for (const Eigen::Vector3d& point : row) {
                bounds.lower = bounds.lower.cwiseMin(point);
                bounds.upper = bounds.upper.cwiseMax(point);
            }
        }
    }
    return bounds;
}

NurbsPatchEdgeInterval3D full_patch_edge_interval(
    const NurbsSurfacePatch3D& patch,
    int patch_index,
    NurbsPatchEdge3D edge)
{
    const auto domain = edge_parameter_domain(patch, edge);
    return {patch_index, edge, domain.first, domain.second};
}

} // namespace kfbim::geometry3d
