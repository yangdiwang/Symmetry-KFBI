#include <apps/harmonic_cauchy_fit_3d.hpp>

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <utility>

namespace kfbim::app3d {

namespace {

HarmonicCauchyError3D shared_edge_failure(
    int connection_id,
    const std::array<std::vector<int>, 2>& sectors,
    const std::string& message);

struct EdgeGeometry3D {
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d positive_parameter_tangent = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
};

EdgeGeometry3D edge_geometry(
    const geometry3d::NurbsSurfacePatch3D& patch,
    geometry3d::NurbsPatchEdge3D edge,
    double parameter,
    int connection_id,
    const std::array<std::vector<int>, 2>& sectors)
{
    double u = patch.domain_start_u();
    double v = patch.domain_start_v();
    switch (edge) {
    case geometry3d::NurbsPatchEdge3D::UMin:
        v = parameter;
        break;
    case geometry3d::NurbsPatchEdge3D::UMax:
        u = patch.domain_end_u();
        v = parameter;
        break;
    case geometry3d::NurbsPatchEdge3D::VMin:
        u = parameter;
        break;
    case geometry3d::NurbsPatchEdge3D::VMax:
        u = parameter;
        v = patch.domain_end_v();
        break;
    }
    const geometry3d::NurbsSurfaceDerivatives3D derivatives =
        patch.evaluate_with_derivatives(u, v);
    const bool varying_v = edge == geometry3d::NurbsPatchEdge3D::UMin
                        || edge == geometry3d::NurbsPatchEdge3D::UMax;
    const Eigen::Vector3d tangent =
        varying_v ? derivatives.dv : derivatives.du;
    const double tangent_norm = tangent.norm();
    if (!tangent.allFinite() || !std::isfinite(tangent_norm)
        || tangent_norm <= 1.0e-14) {
        throw shared_edge_failure(
            connection_id, sectors,
            "native patch edge has a zero or non-finite tangent");
    }
    Eigen::Vector3d normal = derivatives.du.cross(derivatives.dv);
    const double normal_norm = normal.norm();
    if (!std::isfinite(normal_norm) || normal_norm <= 1.0e-14) {
        throw shared_edge_failure(
            connection_id, sectors,
            "native patch edge has a degenerate normal");
    }
    normal /= normal_norm;
    return {{u, v},
            derivatives.point,
            tangent,
            normal};
}

std::uint64_t append_hash_bytes(std::uint64_t hash,
                                const void* data,
                                std::size_t size)
{
    constexpr std::uint64_t prime = 1099511628211ULL;
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= prime;
    }
    return hash;
}

template <class Value>
std::uint64_t append_hash(std::uint64_t hash, const Value& value)
{
    return append_hash_bytes(hash, &value, sizeof(Value));
}

int minimum_patch_id(const std::vector<int>& sector)
{
    if (sector.empty())
        return std::numeric_limits<int>::max();
    return *std::min_element(sector.begin(), sector.end());
}

HarmonicCauchyError3D shared_edge_failure(
    int connection_id,
    const std::array<std::vector<int>, 2>& sectors,
    const std::string& message)
{
    HarmonicCauchyFailure3D diagnostic;
    diagnostic.stage = "shared_edge_geometry";
    diagnostic.entity_kind = "connection";
    diagnostic.entity_id = connection_id;
    diagnostic.connection_id = connection_id;
    diagnostic.incident_sectors = {sectors[0], sectors[1]};
    diagnostic.message = message;
    return HarmonicCauchyError3D(std::move(diagnostic));
}

} // namespace

HarmonicCauchyError3D::HarmonicCauchyError3D(
    HarmonicCauchyFailure3D diagnostic)
    : std::runtime_error(diagnostic.message)
    , diagnostic_(std::move(diagnostic))
{}

const HarmonicCauchyFailure3D&
HarmonicCauchyError3D::diagnostic() const noexcept
{
    return diagnostic_;
}

SharedEdgePointSet3D make_shared_edge_points_3d(
    const NativeNurbsSurface3D& surface,
    double h,
    int edge_length_parameter_samples)
{
    if (!std::isfinite(h) || h <= 0.0)
        throw std::invalid_argument("shared edge points require positive h");
    if (edge_length_parameter_samples < 2) {
        throw std::invalid_argument(
            "shared edge length requires at least two parameter samples");
    }
    if (surface.patches.empty()
        || surface.smooth_neighbors.size() != surface.patches.size()) {
        throw std::invalid_argument(
            "shared edge points require complete native surface metadata");
    }
    const double model_diameter = surface.geometry_model().control_bounds().diameter();
    if (!std::isfinite(model_diameter) || model_diameter <= 0.0) {
        throw std::invalid_argument(
            "shared edge points require a positive model diameter");
    }
    const double physical_tolerance =
        std::max(1.0e-12 * model_diameter, 1.0e-14);

    SharedEdgePointSet3D result;
    result.point_ids_by_connection.resize(surface.geometric_connections.size());
    constexpr std::uint64_t offset = 1469598103934665603ULL;
    std::uint64_t fingerprint = offset;
    for (int connection_id = 0;
         connection_id < static_cast<int>(surface.geometric_connections.size());
         ++connection_id) {
        const geometry3d::NurbsPatchEdgeConnection3D& connection =
            surface.geometric_connections[static_cast<std::size_t>(connection_id)];
        if (connection.g1)
            continue;
        if (connection.first.patch < 0
            || connection.first.patch >= static_cast<int>(surface.patches.size())
            || connection.second.patch < 0
            || connection.second.patch >= static_cast<int>(surface.patches.size())) {
            throw shared_edge_failure(connection_id, {},
                                      "connection has an invalid incident patch");
        }
        std::array<std::vector<int>, 2> sectors{{
            smooth_patch_component(surface, connection.first.patch),
            smooth_patch_component(surface, connection.second.patch)}};
        if (sectors[0] == sectors[1]) {
            throw shared_edge_failure(
                connection_id,
                sectors,
                "non-G1 connection incident sectors are not distinct");
        }
        const geometry3d::NurbsSurfacePatch3D& first_patch =
            surface.patches[static_cast<std::size_t>(connection.first.patch)];
        const geometry3d::NurbsSurfacePatch3D& second_patch =
            surface.patches[static_cast<std::size_t>(connection.second.patch)];
        const double interval_length =
            geometry3d::estimate_nurbs_patch_edge_interval_length_3d(
                first_patch,
                connection.first.edge,
                connection.first.begin,
                connection.first.end,
                edge_length_parameter_samples);
        if (!std::isfinite(interval_length)) {
            const double midpoint = 0.5
                * (connection.first.begin + connection.first.end);
            (void)edge_geometry(
                first_patch, connection.first.edge, midpoint,
                connection_id, sectors);
            throw shared_edge_failure(
                connection_id, sectors,
                "connection native edge length is non-finite");
        }
        if (interval_length <= 0.0) {
            throw shared_edge_failure(
                connection_id, sectors,
                "connection native edge length is non-positive");
        }
        const int cell_count = std::max(
            1, static_cast<int>(std::ceil(interval_length / h)));
        auto& connection_ids = result.point_ids_by_connection[
            static_cast<std::size_t>(connection_id)];
        connection_ids.reserve(static_cast<std::size_t>(cell_count));
        for (int cell = 0; cell < cell_count; ++cell) {
            const double fraction =
                (static_cast<double>(cell) + 0.5)
                / static_cast<double>(cell_count);
            const double first_parameter = connection.first.begin
                + fraction * (connection.first.end - connection.first.begin);
            const double second_fraction =
                connection.reversed ? 1.0 - fraction : fraction;
            const double second_parameter = connection.second.begin
                + second_fraction
                    * (connection.second.end - connection.second.begin);
            const EdgeGeometry3D first = edge_geometry(
                first_patch, connection.first.edge, first_parameter,
                connection_id, sectors);
            const EdgeGeometry3D second = edge_geometry(
                second_patch, connection.second.edge, second_parameter,
                connection_id, sectors);
            const double mismatch = (first.point - second.point).norm();
            result.max_position_mismatch =
                std::max(result.max_position_mismatch, mismatch);
            if (!std::isfinite(mismatch) || mismatch > physical_tolerance) {
                throw shared_edge_failure(
                    connection_id,
                    sectors,
                    "connection native edge points exceed physical tolerance");
            }

            Eigen::Vector3d first_mapped_tangent =
                (connection.first.end - connection.first.begin)
                * first.positive_parameter_tangent;
            Eigen::Vector3d second_mapped_tangent =
                (connection.reversed ? -1.0 : 1.0)
                * (connection.second.end - connection.second.begin)
                * second.positive_parameter_tangent;
            const double first_speed = first_mapped_tangent.norm();
            const double second_speed = second_mapped_tangent.norm();
            if (!std::isfinite(first_speed) || first_speed <= 1.0e-14
                || !std::isfinite(second_speed) || second_speed <= 1.0e-14) {
                throw shared_edge_failure(
                    connection_id, sectors,
                    "connection has a degenerate mapped tangent");
            }
            first_mapped_tangent /= first_speed;
            second_mapped_tangent /= second_speed;
            const double tangent_dot =
                first_mapped_tangent.dot(second_mapped_tangent);
            result.min_mapped_tangent_dot =
                std::min(result.min_mapped_tangent_dot, tangent_dot);
            if (!std::isfinite(tangent_dot) || tangent_dot < 1.0 - 1.0e-10) {
                throw shared_edge_failure(
                    connection_id,
                    sectors,
                    "connection mapped unit tangents are inconsistent");
            }
            Eigen::Vector3d tangent =
                first_mapped_tangent + second_mapped_tangent;
            if (tangent.norm() <= 1.0e-14)
                tangent = first_mapped_tangent;
            tangent.normalize();

            Eigen::Vector3d bisector = first.normal + second.normal;
            bisector -= bisector.dot(tangent) * tangent;
            if (bisector.norm() <= 1.0e-14) {
                const bool first_is_lower =
                    minimum_patch_id(sectors[0]) < minimum_patch_id(sectors[1]);
                bisector = first_is_lower ? first.normal : second.normal;
                bisector -= bisector.dot(tangent) * tangent;
            }
            if (!bisector.allFinite() || bisector.norm() <= 1.0e-14) {
                throw shared_edge_failure(
                    connection_id,
                    sectors,
                    "connection cannot construct a projected normal bisector");
            }
            bisector.normalize();
            const Eigen::Vector3d transverse = bisector.cross(tangent).normalized();

            SharedEdgePoint3D point;
            point.id = static_cast<int>(result.points.size());
            point.connection_id = connection_id;
            point.cell_id = cell;
            point.cell_count = cell_count;
            point.fraction = fraction;
            point.quadrature_weight = interval_length
                                    / static_cast<double>(cell_count);
            point.native_parameters = {{first_parameter, second_parameter}};
            point.native_uv = {{first.uv, second.uv}};
            point.point = 0.5 * (first.point + second.point);
            point.tangent = tangent;
            point.frame.col(0) = tangent;
            point.frame.col(1) = transverse;
            point.frame.col(2) = bisector;
            point.sector_patch_ids = sectors;
            connection_ids.push_back(point.id);
            fingerprint = append_hash(fingerprint, point.id);
            fingerprint = append_hash(fingerprint, point.connection_id);
            fingerprint = append_hash(fingerprint, point.cell_id);
            fingerprint = append_hash(fingerprint, point.cell_count);
            fingerprint = append_hash(fingerprint, point.fraction);
            fingerprint = append_hash(fingerprint, point.native_parameters[0]);
            fingerprint = append_hash(fingerprint, point.native_parameters[1]);
            result.points.push_back(std::move(point));
        }
    }
    result.fingerprint = fingerprint;
    return result;
}

namespace {

double snapped_edge_distance(double distance,
                             double h,
                             double physical_tolerance)
{
    if (std::abs(distance - h) <= physical_tolerance)
        return h;
    if (std::abs(distance - 2.0 * h) <= physical_tolerance)
        return 2.0 * h;
    return distance;
}

std::array<double, 2> mapped_connection_parameters(
    const geometry3d::NurbsPatchEdgeConnection3D& connection,
    int incident_side,
    double parameter)
{
    if (incident_side == 0) {
        const double fraction =
            (parameter - connection.first.begin)
            / (connection.first.end - connection.first.begin);
        const double second_fraction =
            connection.reversed ? 1.0 - fraction : fraction;
        return {{parameter,
                 connection.second.begin
                     + second_fraction
                         * (connection.second.end - connection.second.begin)}};
    }
    const double second_fraction =
        (parameter - connection.second.begin)
        / (connection.second.end - connection.second.begin);
    const double first_fraction =
        connection.reversed ? 1.0 - second_fraction : second_fraction;
    return {{connection.first.begin
                 + first_fraction
                     * (connection.first.end - connection.first.begin),
             parameter}};
}

HarmonicCauchyError3D route_failure(const std::string& stage,
                                    const std::string& entity_kind,
                                    int entity_id,
                                    int connection_id,
                                    const std::string& message)
{
    HarmonicCauchyFailure3D diagnostic;
    diagnostic.stage = stage;
    diagnostic.entity_kind = entity_kind;
    diagnostic.entity_id = entity_id;
    diagnostic.connection_id = connection_id;
    diagnostic.message = message;
    return HarmonicCauchyError3D(std::move(diagnostic));
}

void validate_surface_cloud(const NativeNurbsSurface3D& surface,
                            const SurfaceDofCloud3D& cloud,
                            double h)
{
    if (!std::isfinite(h) || h <= 0.0)
        throw std::invalid_argument("Cauchy geometry requires positive h");
    if (surface.patches.size() != cloud.patches.size()
        || surface.smooth_neighbors.size() != surface.patches.size()) {
        throw std::invalid_argument(
            "Cauchy geometry requires matching surface and cloud metadata");
    }
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q) {
        const int patch = cloud.dofs[static_cast<std::size_t>(q)].patch_id;
        if (patch < 0 || patch >= static_cast<int>(surface.patches.size())) {
            throw std::invalid_argument(
                "Cauchy cloud contains an invalid patch owner");
        }
    }
}

const SurfaceNonG1EdgeNeighborhood3D& cached_center(
    const SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
    const SurfaceDofCloud3D& cloud,
    int center_dof)
{
    if (center_dof < 0 || center_dof >= static_cast<int>(cloud.dofs.size()))
        throw std::out_of_range("Cauchy center DOF is outside the cloud");
    if (neighborhoods.centers.size() != cloud.dofs.size()) {
        throw std::invalid_argument(
            "Cauchy selector requires a complete cached neighborhood set");
    }
    const auto& center =
        neighborhoods.centers[static_cast<std::size_t>(center_dof)];
    if (center.center_dof != center_dof) {
        throw std::invalid_argument(
            "Cauchy cached neighborhood center IDs are inconsistent");
    }
    return center;
}

bool same_sector(const std::vector<int>& first,
                 const std::vector<int>& second)
{
    return first == second;
}

void append_unique_sector(std::vector<std::vector<int>>& sectors,
                          std::vector<int> sector)
{
    if (std::none_of(sectors.begin(), sectors.end(),
                     [&](const std::vector<int>& existing) {
                         return same_sector(existing, sector);
                     })) {
        sectors.push_back(std::move(sector));
    }
}

int sector_index_for_patch(const std::vector<std::vector<int>>& sectors,
                           int patch)
{
    for (int sector = 0; sector < static_cast<int>(sectors.size()); ++sector) {
        const auto& patches = sectors[static_cast<std::size_t>(sector)];
        if (std::binary_search(patches.begin(), patches.end(), patch))
            return sector;
    }
    return -1;
}

} // namespace

SurfaceNonG1EdgeNeighborhoodSet3D
build_surface_non_g1_edge_neighborhoods_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h)
{
    validate_surface_cloud(surface, cloud, h);
    const double model_diameter = surface.geometry_model().control_bounds().diameter();
    if (!std::isfinite(model_diameter) || model_diameter <= 0.0) {
        throw std::invalid_argument(
            "Cauchy edge neighborhoods require a positive model diameter");
    }
    const double physical_tolerance =
        std::max(1.0e-12 * model_diameter, 1.0e-14);
    std::vector<std::vector<int>> components(surface.patches.size());
    for (int patch = 0; patch < static_cast<int>(surface.patches.size()); ++patch)
        components[static_cast<std::size_t>(patch)] =
            smooth_patch_component(surface, patch);

    SurfaceNonG1EdgeNeighborhoodSet3D result;
    result.centers.reserve(cloud.dofs.size());
    constexpr std::uint64_t offset = 1469598103934665603ULL;
    std::uint64_t fingerprint = offset;
    for (int center_dof = 0;
         center_dof < static_cast<int>(cloud.dofs.size());
         ++center_dof) {
        const SurfaceDof3D& center =
            cloud.dofs[static_cast<std::size_t>(center_dof)];
        const std::vector<int>& center_component =
            components[static_cast<std::size_t>(center.patch_id)];
        SurfaceNonG1EdgeNeighborhood3D neighborhood;
        neighborhood.center_dof = center_dof;
        double nearest_distance = std::numeric_limits<double>::infinity();
        for (int connection_id = 0;
             connection_id
                 < static_cast<int>(surface.geometric_connections.size());
             ++connection_id) {
            const auto& connection = surface.geometric_connections[
                static_cast<std::size_t>(connection_id)];
            if (connection.g1)
                continue;
            int incident_side = -1;
            if (std::binary_search(center_component.begin(),
                                   center_component.end(),
                                   connection.first.patch)) {
                incident_side = 0;
            }
            if (std::binary_search(center_component.begin(),
                                   center_component.end(),
                                   connection.second.patch)) {
                if (incident_side >= 0) {
                    throw route_failure(
                        "edge_neighborhood", "connection", connection_id,
                        connection_id,
                        "non-G1 connection has both sides in one G1 component");
                }
                incident_side = 1;
            }
            if (incident_side < 0)
                continue;
            const auto& interval = incident_side == 0
                ? connection.first : connection.second;
            const auto& patch = surface.patches[
                static_cast<std::size_t>(interval.patch)];
            const geometry3d::NurbsPatchEdgeClosestPoint3D closest =
                geometry3d::closest_point_to_nurbs_patch_edge_interval_3d(
                    patch, interval.edge, interval.begin, interval.end,
                    center.point, model_diameter);
            ++result.geometry_query_count;
            if (!closest.converged) {
                std::ostringstream message;
                message << "native partial-edge closest-point solve did not converge"
                        << " center_patch=" << center.patch_id
                        << " interval_patch=" << interval.patch
                        << " parameter=" << closest.parameter
                        << " knot_spans=" << closest.knot_span_count
                        << " refinement=" << closest.refinement_level
                        << " bracket_image=" << closest.distance_error_bound;
                throw route_failure(
                    "edge_neighborhood", "surface_dof", center_dof,
                    connection_id, message.str());
            }
            const double distance = snapped_edge_distance(
                closest.distance, h, physical_tolerance);
            SurfaceNonG1EdgeDistance3D incident;
            incident.connection_id = connection_id;
            incident.distance = distance;
            incident.native_parameters = mapped_connection_parameters(
                connection, incident_side, closest.parameter);
            incident.closest_point = closest.point;
            incident.distance_error_bound = closest.distance_error_bound;
            neighborhood.incident_distances.push_back(std::move(incident));
            nearest_distance = std::min(nearest_distance, distance);
            if (distance <= 2.0 * h)
                neighborhood.relevant_connection_ids.push_back(connection_id);
        }
        if (std::isfinite(nearest_distance))
            neighborhood.nearest_distance_over_h = nearest_distance / h;
        fingerprint = append_hash(fingerprint, neighborhood.center_dof);
        fingerprint = append_hash(
            fingerprint, neighborhood.nearest_distance_over_h);
        const std::size_t incident_count =
            neighborhood.incident_distances.size();
        fingerprint = append_hash(fingerprint, incident_count);
        for (const auto& incident : neighborhood.incident_distances) {
            fingerprint = append_hash(fingerprint, incident.connection_id);
            fingerprint = append_hash(fingerprint, incident.distance);
            fingerprint = append_hash(fingerprint, incident.native_parameters[0]);
            fingerprint = append_hash(fingerprint, incident.native_parameters[1]);
        }
        const std::size_t relevant_count =
            neighborhood.relevant_connection_ids.size();
        fingerprint = append_hash(fingerprint, relevant_count);
        for (int connection_id : neighborhood.relevant_connection_ids)
            fingerprint = append_hash(fingerprint, connection_id);
        result.centers.push_back(std::move(neighborhood));
    }
    fingerprint = append_hash(fingerprint, result.geometry_query_count);
    result.fingerprint = fingerprint;
    return result;
}

DirectCrossFaceSelection3D
select_direct_cross_face_value_dofs_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
    int center_dof,
    int count,
    double h)
{
    validate_surface_cloud(surface, cloud, h);
    if (count <= 0)
        throw std::invalid_argument("direct Cauchy sample count must be positive");
    const SurfaceNonG1EdgeNeighborhood3D& neighborhood =
        cached_center(neighborhoods, cloud, center_dof);
    DirectCrossFaceSelection3D result;
    result.relevant_connection_ids = neighborhood.relevant_connection_ids;
    result.nearest_edge_distance_over_h = neighborhood.nearest_distance_over_h;
    const int center_patch =
        cloud.dofs[static_cast<std::size_t>(center_dof)].patch_id;
    const std::vector<int> center_sector =
        smooth_patch_component(surface, center_patch);
    if (neighborhood.relevant_connection_ids.empty()
        || neighborhood.nearest_distance_over_h > 2.0) {
        result.dof_ids = nearest_g1_cauchy_dofs(
            surface, cloud, center_dof, count);
        result.sector_patch_ids = {center_sector};
        result.sector_sample_counts = {
            static_cast<int>(result.dof_ids.size())};
        if (static_cast<int>(result.dof_ids.size()) != count) {
            HarmonicCauchyFailure3D diagnostic;
            diagnostic.stage = "direct_selector";
            diagnostic.entity_kind = "surface_dof";
            diagnostic.entity_id = center_dof;
            diagnostic.incident_sectors = result.sector_patch_ids;
            diagnostic.actual_value_counts = result.sector_sample_counts;
            diagnostic.required_value_count = count;
            const Eigen::Vector3d& center_point =
                cloud.dofs[static_cast<std::size_t>(center_dof)].point;
            for (int id : result.dof_ids) {
                diagnostic.value_radius_over_h = std::max(
                    diagnostic.value_radius_over_h,
                    (cloud.dofs[static_cast<std::size_t>(id)].point
                        - center_point).norm() / h);
            }
            diagnostic.message =
                "outside-band G1 sector cannot fill direct Cauchy sample count";
            throw HarmonicCauchyError3D(std::move(diagnostic));
        }
        return result;
    }

    result.sector_patch_ids.push_back(center_sector);
    std::vector<std::vector<int>> other_sectors;
    for (int connection_id : neighborhood.relevant_connection_ids) {
        if (connection_id < 0
            || connection_id
                   >= static_cast<int>(surface.geometric_connections.size())) {
            throw route_failure(
                "direct_selector", "surface_dof", center_dof,
                connection_id, "cached connection ID is invalid");
        }
        const auto& connection = surface.geometric_connections[
            static_cast<std::size_t>(connection_id)];
        const std::vector<int> first_sector =
            smooth_patch_component(surface, connection.first.patch);
        const std::vector<int> second_sector =
            smooth_patch_component(surface, connection.second.patch);
        const bool first_is_center = same_sector(first_sector, center_sector);
        const bool second_is_center = same_sector(second_sector, center_sector);
        if (first_is_center == second_is_center) {
            HarmonicCauchyFailure3D diagnostic;
            diagnostic.stage = "direct_selector_topology";
            diagnostic.entity_kind = "surface_dof";
            diagnostic.entity_id = center_dof;
            diagnostic.connection_id = connection_id;
            diagnostic.incident_sectors = {
                center_sector, first_sector, second_sector};
            diagnostic.message =
                "cached relevant connection is not incident to exactly one center sector";
            throw HarmonicCauchyError3D(std::move(diagnostic));
        }
        for (int patch : {connection.first.patch, connection.second.patch}) {
            std::vector<int> sector = smooth_patch_component(surface, patch);
            if (!same_sector(sector, center_sector))
                append_unique_sector(other_sectors, std::move(sector));
        }
    }
    std::sort(other_sectors.begin(), other_sectors.end(),
              [](const std::vector<int>& first,
                 const std::vector<int>& second) {
                  return minimum_patch_id(first) < minimum_patch_id(second);
              });
    for (auto& sector : other_sectors)
        result.sector_patch_ids.push_back(std::move(sector));
    const int sector_count =
        static_cast<int>(result.sector_patch_ids.size());
    if (sector_count <= 1) {
        throw route_failure(
            "direct_selector", "surface_dof", center_dof, -1,
            "edge-band direct selection has fewer than two G1 sectors");
    }

    using Candidate = std::pair<double, int>;
    std::vector<std::vector<Candidate>> candidates(
        static_cast<std::size_t>(sector_count));
    const Eigen::Vector3d& center_point =
        cloud.dofs[static_cast<std::size_t>(center_dof)].point;
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q) {
        const int patch = cloud.dofs[static_cast<std::size_t>(q)].patch_id;
        const int sector = sector_index_for_patch(result.sector_patch_ids, patch);
        if (sector < 0)
            continue;
        candidates[static_cast<std::size_t>(sector)].push_back({
            (cloud.dofs[static_cast<std::size_t>(q)].point - center_point)
                .squaredNorm(),
            q});
    }
    const auto candidate_less = [](const Candidate& first,
                                   const Candidate& second) {
        return first.first != second.first
            ? first.first < second.first : first.second < second.second;
    };
    for (auto& sector_candidates : candidates)
        std::sort(sector_candidates.begin(), sector_candidates.end(),
                  candidate_less);

    std::vector<int> target(static_cast<std::size_t>(sector_count),
                            count / sector_count);
    int remainder = count % sector_count;
    if (remainder > 0) {
        ++target[0];
        --remainder;
    }
    for (int sector = 1; sector < sector_count && remainder > 0;
         ++sector, --remainder) {
        ++target[static_cast<std::size_t>(sector)];
    }

    std::vector<bool> selected(cloud.dofs.size(), false);
    result.sector_sample_counts.assign(
        static_cast<std::size_t>(sector_count), 0);
    for (int sector = 0; sector < sector_count; ++sector) {
        const auto& sector_candidates =
            candidates[static_cast<std::size_t>(sector)];
        const int take = std::min(
            target[static_cast<std::size_t>(sector)],
            static_cast<int>(sector_candidates.size()));
        for (int q = 0; q < take; ++q) {
            const int id = sector_candidates[static_cast<std::size_t>(q)].second;
            selected[static_cast<std::size_t>(id)] = true;
            result.dof_ids.push_back(id);
            ++result.sector_sample_counts[static_cast<std::size_t>(sector)];
        }
    }
    if (static_cast<int>(result.dof_ids.size()) < count) {
        std::vector<std::pair<Candidate, int>> remaining;
        for (int sector = 0; sector < sector_count; ++sector) {
            for (const Candidate& candidate :
                 candidates[static_cast<std::size_t>(sector)]) {
                if (!selected[static_cast<std::size_t>(candidate.second)])
                    remaining.push_back({candidate, sector});
            }
        }
        std::sort(remaining.begin(), remaining.end(),
                  [&](const std::pair<Candidate, int>& first,
                      const std::pair<Candidate, int>& second) {
                      return candidate_less(first.first, second.first);
                  });
        for (const auto& item : remaining) {
            if (static_cast<int>(result.dof_ids.size()) == count)
                break;
            const int id = item.first.second;
            selected[static_cast<std::size_t>(id)] = true;
            result.dof_ids.push_back(id);
            ++result.sector_sample_counts[static_cast<std::size_t>(item.second)];
        }
    }
    if (static_cast<int>(result.dof_ids.size()) != count) {
        HarmonicCauchyFailure3D diagnostic;
        diagnostic.stage = "direct_selector";
        diagnostic.entity_kind = "surface_dof";
        diagnostic.entity_id = center_dof;
        diagnostic.incident_sectors = result.sector_patch_ids;
        diagnostic.actual_value_counts = result.sector_sample_counts;
        diagnostic.required_value_count = count;
        for (int id : result.dof_ids) {
            diagnostic.value_radius_over_h = std::max(
                diagnostic.value_radius_over_h,
                (cloud.dofs[static_cast<std::size_t>(id)].point
                    - center_point).norm() / h);
        }
        diagnostic.message =
            "admitted G1 sectors cannot fill direct Cauchy sample count";
        throw HarmonicCauchyError3D(std::move(diagnostic));
    }
    if (!selected[static_cast<std::size_t>(center_dof)]) {
        const int center_sector_index = 0;
        int replace_at = -1;
        for (int q = static_cast<int>(result.dof_ids.size()) - 1; q >= 0; --q) {
            const int id = result.dof_ids[static_cast<std::size_t>(q)];
            if (sector_index_for_patch(
                    result.sector_patch_ids,
                    cloud.dofs[static_cast<std::size_t>(id)].patch_id)
                == center_sector_index) {
                replace_at = q;
                break;
            }
        }
        if (replace_at < 0) {
            throw route_failure(
                "direct_selector", "surface_dof", center_dof, -1,
                "direct selection cannot reserve its center DOF");
        }
        selected[static_cast<std::size_t>(
            result.dof_ids[static_cast<std::size_t>(replace_at)])] = false;
        result.dof_ids[static_cast<std::size_t>(replace_at)] = center_dof;
        selected[static_cast<std::size_t>(center_dof)] = true;
    }
    std::sort(result.dof_ids.begin(), result.dof_ids.end(),
              [&](int first, int second) {
                  return candidate_less(
                      {(cloud.dofs[static_cast<std::size_t>(first)].point
                            - center_point).squaredNorm(), first},
                      {(cloud.dofs[static_cast<std::size_t>(second)].point
                            - center_point).squaredNorm(), second});
              });
    if (std::find(result.dof_ids.begin(), result.dof_ids.end(), center_dof)
        == result.dof_ids.end()) {
        throw route_failure(
            "direct_selector", "surface_dof", center_dof, -1,
            "direct 48-ID selection lost its center DOF");
    }
    return result;
}

SurfaceEdgePointSelection3D
select_surface_edge_points_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const SharedEdgePointSet3D& edge_points,
    const SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
    int center_dof,
    double h,
    int max_points_per_edge)
{
    validate_surface_cloud(surface, cloud, h);
    if (max_points_per_edge <= 0) {
        throw std::invalid_argument(
            "edge-point selector requires a positive per-edge limit");
    }
    const SurfaceNonG1EdgeNeighborhood3D& neighborhood =
        cached_center(neighborhoods, cloud, center_dof);
    if (edge_points.point_ids_by_connection.size()
        != surface.geometric_connections.size()) {
        throw std::invalid_argument(
            "edge-point selector requires a complete shared-point set");
    }
    SurfaceEdgePointSelection3D result;
    result.relevant_connection_ids = neighborhood.relevant_connection_ids;
    result.nearest_edge_distance_over_h = neighborhood.nearest_distance_over_h;
    const double model_diameter =
        surface.geometry_model().control_bounds().diameter();
    if (!std::isfinite(model_diameter) || model_diameter <= 0.0) {
        throw std::invalid_argument(
            "edge-point selector requires a positive model diameter");
    }
    const double physical_tolerance =
        std::max(1.0e-12 * model_diameter, 1.0e-14);
    const Eigen::Vector3d& center_point =
        cloud.dofs[static_cast<std::size_t>(center_dof)].point;
    using Candidate = std::pair<double, int>;
    const auto candidate_less = [](const Candidate& first,
                                   const Candidate& second) {
        return first.first != second.first
            ? first.first < second.first : first.second < second.second;
    };
    std::vector<Candidate> selected;
    for (int connection_id : neighborhood.relevant_connection_ids) {
        if (connection_id < 0
            || connection_id
                   >= static_cast<int>(edge_points.point_ids_by_connection.size())) {
            throw route_failure(
                "edge_point_selector", "surface_dof", center_dof,
                connection_id, "cached relevant connection ID is invalid");
        }
        std::vector<Candidate> candidates;
        for (int point_id : edge_points.point_ids_by_connection[
                 static_cast<std::size_t>(connection_id)]) {
            if (point_id < 0
                || point_id >= static_cast<int>(edge_points.points.size())) {
                throw route_failure(
                    "edge_point_selector", "connection", connection_id,
                    connection_id, "shared edge point ID is invalid");
            }
            candidates.push_back({
                (edge_points.points[static_cast<std::size_t>(point_id)].point
                    - center_point).squaredNorm(),
                point_id});
        }
        if (candidates.empty()) {
            throw route_failure(
                "edge_point_selector", "connection", connection_id,
                connection_id, "relevant connection has no shared points");
        }
        std::sort(candidates.begin(), candidates.end(), candidate_less);
        selected.push_back(candidates.front());
        int taken = 1;
        for (std::size_t q = 1;
             q < candidates.size() && taken < max_points_per_edge;
             ++q) {
            const double distance = snapped_edge_distance(
                std::sqrt(candidates[q].first), h, physical_tolerance);
            if (distance <= 2.0 * h) {
                selected.push_back(candidates[q]);
                ++taken;
            }
        }
    }
    std::sort(selected.begin(), selected.end(), candidate_less);
    result.edge_point_ids.reserve(selected.size());
    for (const Candidate& candidate : selected)
        result.edge_point_ids.push_back(candidate.second);
    return result;
}

namespace {

bool sector_contains_patch(const std::vector<int>& sector, int patch)
{
    return std::binary_search(sector.begin(), sector.end(), patch);
}

std::vector<int> nearest_sector_dofs(
    const SurfaceDofCloud3D& cloud,
    const Eigen::Vector3d& center,
    const std::vector<int>& sector,
    int count,
    double h)
{
    using Candidate = std::pair<long long, int>;
    std::vector<Candidate> candidates;
    const double quantum = 1.0e-12 * h * h;
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q) {
        const auto& dof = cloud.dofs[static_cast<std::size_t>(q)];
        if (!sector_contains_patch(sector, dof.patch_id))
            continue;
        const double distance_sq = (dof.point - center).squaredNorm();
        candidates.push_back({
            static_cast<long long>(std::llround(distance_sq / quantum)), q});
    }
    std::sort(candidates.begin(), candidates.end());
    std::vector<int> result;
    const int take = std::min(count, static_cast<int>(candidates.size()));
    result.reserve(static_cast<std::size_t>(take));
    for (int q = 0; q < take; ++q)
        result.push_back(candidates[static_cast<std::size_t>(q)].second);
    return result;
}

double sample_radius_over_h(const SurfaceDofCloud3D& cloud,
                            const Eigen::Vector3d& center,
                            const std::vector<int>& ids,
                            double h)
{
    double radius_sq = 0.0;
    for (int id : ids) {
        radius_sq = std::max(
            radius_sq,
            (cloud.dofs[static_cast<std::size_t>(id)].point - center)
                .squaredNorm());
    }
    return std::sqrt(radius_sq) / h;
}

int patch_imbalance(const SurfaceDofCloud3D& cloud,
                    const std::vector<int>& ids)
{
    std::map<int, int> counts;
    for (int id : ids)
        ++counts[cloud.dofs[static_cast<std::size_t>(id)].patch_id];
    if (counts.empty())
        return 0;
    int minimum = std::numeric_limits<int>::max();
    int maximum = 0;
    for (const auto& item : counts) {
        minimum = std::min(minimum, item.second);
        maximum = std::max(maximum, item.second);
    }
    return maximum - minimum;
}

int incident_patch_count(const SurfaceDofCloud3D& cloud,
                         const std::vector<int>& value_ids,
                         const std::vector<int>& normal_ids)
{
    std::set<int> patches;
    for (int id : value_ids)
        patches.insert(cloud.dofs[static_cast<std::size_t>(id)].patch_id);
    for (int id : normal_ids)
        patches.insert(cloud.dofs[static_cast<std::size_t>(id)].patch_id);
    return static_cast<int>(patches.size());
}

Eigen::VectorXd gather(const Eigen::VectorXd& source,
                       const std::vector<int>& ids)
{
    Eigen::VectorXd result(static_cast<int>(ids.size()));
    for (int q = 0; q < result.size(); ++q)
        result[q] = source[ids[static_cast<std::size_t>(q)]];
    return result;
}

struct DesignInverse3D {
    Eigen::MatrixXd pinv;
    double sigma_max = 0.0;
    double sigma_min = 0.0;
    double condition = 0.0;
};

DesignInverse3D factor_design(
    const Eigen::MatrixXd& weighted,
    int dimension,
    double relative_cutoff,
    HarmonicCauchyFailure3D diagnostic,
    std::size_t& svd_count)
{
    ++svd_count;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        weighted, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular = svd.singularValues();
    if (singular.size() > 0) {
        diagnostic.sigma_max = singular[0];
        diagnostic.sigma_min = singular[singular.size() - 1];
        if (diagnostic.sigma_min > 0.0)
            diagnostic.condition = diagnostic.sigma_max / diagnostic.sigma_min;
        else if (diagnostic.sigma_max > 0.0)
            diagnostic.condition = std::numeric_limits<double>::infinity();
    }
    if (singular.size() != dimension || !singular.allFinite()
        || !(diagnostic.sigma_max > 0.0)
        || !(diagnostic.sigma_min
             > relative_cutoff * diagnostic.sigma_max)) {
        std::ostringstream message;
        message << diagnostic.stage << " rank failure entity="
                << diagnostic.entity_id << " sigma_max="
                << diagnostic.sigma_max << " sigma_min="
                << diagnostic.sigma_min << " condition="
                << diagnostic.condition;
        diagnostic.message = message.str();
        throw HarmonicCauchyError3D(std::move(diagnostic));
    }
    Eigen::VectorXd inverse = singular.cwiseInverse();
    DesignInverse3D result;
    result.pinv = svd.matrixV() * inverse.asDiagonal()
                * svd.matrixU().transpose();
    result.sigma_max = diagnostic.sigma_max;
    result.sigma_min = diagnostic.sigma_min;
    result.condition = diagnostic.condition;
    return result;
}

HarmonicCauchyError3D edge_sample_failure(
    const SharedEdgePoint3D& point,
    const std::array<int, 2>& value_counts,
    const std::array<int, 2>& normal_counts,
    int required_value,
    int required_normal,
    double value_radius,
    double normal_radius)
{
    HarmonicCauchyFailure3D diagnostic;
    diagnostic.stage = "edge_sector_selection";
    diagnostic.entity_kind = "edge_point";
    diagnostic.entity_id = point.id;
    diagnostic.connection_id = point.connection_id;
    diagnostic.incident_sectors = {
        point.sector_patch_ids[0], point.sector_patch_ids[1]};
    diagnostic.actual_value_counts = {
        value_counts[0], value_counts[1]};
    diagnostic.actual_normal_counts = {
        normal_counts[0], normal_counts[1]};
    diagnostic.required_value_count = required_value;
    diagnostic.required_normal_count = required_normal;
    diagnostic.value_radius_over_h = value_radius;
    diagnostic.normal_radius_over_h = normal_radius;
    std::ostringstream message;
    message << "edge point " << point.id << " connection "
            << point.connection_id << " sectors";
    for (const auto& sector : point.sector_patch_ids) {
        message << " [";
        for (int patch : sector)
            message << patch << ',';
        message << ']';
    }
    message << " requires/actual value " << required_value << '/'
            << value_counts[0] << ',' << value_counts[1]
            << " normal " << required_normal << '/'
            << normal_counts[0] << ',' << normal_counts[1]
            << " radii " << value_radius << '/' << normal_radius;
    diagnostic.message = message.str();
    return HarmonicCauchyError3D(std::move(diagnostic));
}

EdgeValueMap3D build_edge_value_map(
    const SharedEdgePoint3D& point,
    const SurfaceDofCloud3D& cloud,
    const HarmonicPolynomialSpace3D& space,
    double h,
    double relative_cutoff,
    std::size_t& svd_count)
{
    constexpr int value_per_sector = 24;
    constexpr int normal_per_sector = 14;
    EdgeValueMap3D result;
    result.point = point;
    result.sector_patch_ids = point.sector_patch_ids;
    for (int sector = 0; sector < 2; ++sector) {
        const std::vector<int> values = nearest_sector_dofs(
            cloud, point.point,
            point.sector_patch_ids[static_cast<std::size_t>(sector)],
            value_per_sector, h);
        const std::vector<int> normals = nearest_sector_dofs(
            cloud, point.point,
            point.sector_patch_ids[static_cast<std::size_t>(sector)],
            normal_per_sector, h);
        result.value_sector_counts[static_cast<std::size_t>(sector)] =
            static_cast<int>(values.size());
        result.normal_sector_counts[static_cast<std::size_t>(sector)] =
            static_cast<int>(normals.size());
        result.value_ids.insert(
            result.value_ids.end(), values.begin(), values.end());
        result.normal_ids.insert(
            result.normal_ids.end(), normals.begin(), normals.end());
    }
    result.value_radius_over_h = sample_radius_over_h(
        cloud, point.point, result.value_ids, h);
    result.normal_radius_over_h = sample_radius_over_h(
        cloud, point.point, result.normal_ids, h);
    if (result.value_sector_counts
            != std::array<int, 2>{{value_per_sector, value_per_sector}}
        || result.normal_sector_counts
            != std::array<int, 2>{{normal_per_sector, normal_per_sector}}) {
        throw edge_sample_failure(
            point, result.value_sector_counts, result.normal_sector_counts,
            value_per_sector, normal_per_sector,
            result.value_radius_over_h, result.normal_radius_over_h);
    }

    const int value_count = static_cast<int>(result.value_ids.size());
    const int normal_count = static_cast<int>(result.normal_ids.size());
    Eigen::MatrixXd design(value_count + normal_count, space.dimension());
    Eigen::VectorXd sqrt_weights(value_count + normal_count);
    for (int k = 0; k < value_count; ++k) {
        const auto& sample = cloud.dofs[static_cast<std::size_t>(
            result.value_ids[static_cast<std::size_t>(k)])];
        const Eigen::Vector3d xi =
            point.frame.transpose() * (sample.point - point.point) / h;
        design.row(k) =
            space.basis(xi.x(), xi.y(), xi.z()).transpose();
        sqrt_weights[k] = 1.0 / (0.35 + xi.norm());
    }
    for (int k = 0; k < normal_count; ++k) {
        const auto& sample = cloud.dofs[static_cast<std::size_t>(
            result.normal_ids[static_cast<std::size_t>(k)])];
        const Eigen::Vector3d xi =
            point.frame.transpose() * (sample.point - point.point) / h;
        const Eigen::Vector3d normal_components =
            point.frame.transpose() * sample.normal;
        design.row(value_count + k) = normal_components.transpose()
            * space.gradient(xi.x(), xi.y(), xi.z());
        sqrt_weights[value_count + k] =
            std::sqrt(0.85) / (0.35 + xi.norm());
    }
    HarmonicCauchyFailure3D diagnostic;
    diagnostic.stage = "edge_map_factorization";
    diagnostic.entity_kind = "edge_point";
    diagnostic.entity_id = point.id;
    diagnostic.connection_id = point.connection_id;
    diagnostic.incident_sectors = {
        point.sector_patch_ids[0], point.sector_patch_ids[1]};
    diagnostic.actual_value_counts = {
        result.value_sector_counts[0], result.value_sector_counts[1]};
    diagnostic.actual_normal_counts = {
        result.normal_sector_counts[0], result.normal_sector_counts[1]};
    diagnostic.required_value_count = value_per_sector;
    diagnostic.required_normal_count = normal_per_sector;
    diagnostic.value_radius_over_h = result.value_radius_over_h;
    diagnostic.normal_radius_over_h = result.normal_radius_over_h;
    const DesignInverse3D inverse = factor_design(
        sqrt_weights.asDiagonal() * design, space.dimension(),
        relative_cutoff, std::move(diagnostic), svd_count);
    result.sigma_max = inverse.sigma_max;
    result.sigma_min = inverse.sigma_min;
    result.condition = inverse.condition;
    result.E_value.resize(value_count);
    result.E_normal.resize(normal_count);
    const Eigen::VectorXd origin = space.basis(0.0, 0.0, 0.0);
    for (int k = 0; k < value_count; ++k) {
        result.E_value[k] =
            origin.dot(inverse.pinv.col(k)) * sqrt_weights[k];
    }
    for (int k = 0; k < normal_count; ++k) {
        result.E_normal[k] = origin.dot(inverse.pinv.col(value_count + k))
                           * sqrt_weights[value_count + k] * h;
    }
    return result;
}

SurfaceCauchyMap3D select_surface_map_inputs(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const SharedEdgePointSet3D* edge_points,
    const SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
    int center,
    double h,
    HarmonicCauchyRoute3D route,
    int value_count,
    int normal_count,
    std::optional<LegacySurfaceCauchyPolicy3D> legacy_policy)
{
    SurfaceCauchyMap3D result;
    result.neighborhood_fingerprint = neighborhoods.fingerprint;
    const auto& neighborhood =
        neighborhoods.centers[static_cast<std::size_t>(center)];
    result.nearest_edge_distance_over_h =
        neighborhood.nearest_distance_over_h;
    result.relevant_connection_ids = neighborhood.relevant_connection_ids;
    const int center_patch =
        cloud.dofs[static_cast<std::size_t>(center)].patch_id;
    const std::vector<int> center_sector =
        smooth_patch_component(surface, center_patch);
    const auto legacy_ids = [&](int count) {
        switch (*legacy_policy) {
        case LegacySurfaceCauchyPolicy3D::G1Nearest:
            return nearest_g1_cauchy_dofs(surface, cloud, center, count);
        case LegacySurfaceCauchyPolicy3D::TopologicalNearest:
            return nearest_topological_cauchy_dofs(
                surface, cloud, center, count);
        case LegacySurfaceCauchyPolicy3D::SamePatch:
            return nearest_same_patch_cauchy_dofs(cloud, center, count);
        case LegacySurfaceCauchyPolicy3D::BalancedPatches:
            return balanced_topological_cauchy_dofs(
                surface, cloud, center, count);
        }
        throw std::logic_error("unknown legacy Cauchy policy");
    };
    if (legacy_policy) {
        result.value_ids = legacy_ids(value_count);
        result.value_sector_patch_ids = {center_sector};
        result.value_sector_counts = {
            static_cast<int>(result.value_ids.size())};
    } else if (route == HarmonicCauchyRoute3D::DirectCrossFaceValue) {
        const DirectCrossFaceSelection3D direct =
            select_direct_cross_face_value_dofs_3d(
                surface, cloud, neighborhoods, center, value_count, h);
        result.value_ids = direct.dof_ids;
        result.value_sector_patch_ids = direct.sector_patch_ids;
        result.value_sector_counts = direct.sector_sample_counts;
    } else {
        result.value_ids = nearest_g1_cauchy_dofs(
            surface, cloud, center, value_count);
        result.value_sector_patch_ids = {center_sector};
        result.value_sector_counts = {
            static_cast<int>(result.value_ids.size())};
    }
    result.normal_ids = legacy_policy
        ? legacy_ids(normal_count)
        : nearest_g1_cauchy_dofs(surface, cloud, center, normal_count);
    result.normal_sector_patch_ids = {center_sector};
    result.normal_sector_counts = {
        static_cast<int>(result.normal_ids.size())};
    const Eigen::Vector3d& center_point =
        cloud.dofs[static_cast<std::size_t>(center)].point;
    result.value_radius_over_h = sample_radius_over_h(
        cloud, center_point, result.value_ids, h);
    result.normal_radius_over_h = sample_radius_over_h(
        cloud, center_point, result.normal_ids, h);
    if (!legacy_policy
        && (static_cast<int>(result.value_ids.size()) != value_count
            || static_cast<int>(result.normal_ids.size()) != normal_count)) {
        HarmonicCauchyFailure3D diagnostic;
        diagnostic.stage = "surface_sector_selection";
        diagnostic.entity_kind = "surface_dof";
        diagnostic.entity_id = center;
        diagnostic.incident_sectors = result.value_sector_patch_ids;
        diagnostic.actual_value_counts = result.value_sector_counts;
        diagnostic.actual_normal_counts = result.normal_sector_counts;
        diagnostic.required_value_count = value_count;
        diagnostic.required_normal_count = normal_count;
        diagnostic.value_radius_over_h = result.value_radius_over_h;
        diagnostic.normal_radius_over_h = result.normal_radius_over_h;
        diagnostic.message =
            "surface Cauchy sectors cannot fill the requested sample counts";
        throw HarmonicCauchyError3D(std::move(diagnostic));
    }
    if (!legacy_policy
        && route == HarmonicCauchyRoute3D::EdgeReconstructedValue) {
        if (edge_points == nullptr)
            throw std::logic_error("edge route requires shared edge points");
        const SurfaceEdgePointSelection3D edge =
            select_surface_edge_points_3d(
                surface, cloud, *edge_points, neighborhoods, center, h);
        result.edge_point_ids = edge.edge_point_ids;
    }
    result.incident_patch_count = incident_patch_count(
        cloud, result.value_ids, result.normal_ids);
    result.value_patch_imbalance = patch_imbalance(cloud, result.value_ids);
    result.normal_patch_imbalance = patch_imbalance(cloud, result.normal_ids);
    return result;
}

SurfaceCauchyMap3D build_surface_map(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const SharedEdgePointSet3D* edge_points,
    const SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
    const HarmonicPolynomialSpace3D& space,
    int center,
    double h,
    HarmonicCauchyRoute3D route,
    int value_count,
    int normal_count,
    double relative_cutoff,
    std::size_t& svd_count,
    std::optional<LegacySurfaceCauchyPolicy3D> legacy_policy = std::nullopt)
{
    SurfaceCauchyMap3D result = select_surface_map_inputs(
        surface, cloud, edge_points, neighborhoods, center, h, route,
        value_count, normal_count, legacy_policy);
    const auto& target = cloud.dofs[static_cast<std::size_t>(center)];
    Eigen::Matrix3d frame;
    frame.col(0) = target.tangent1;
    frame.col(1) = target.tangent2;
    frame.col(2) = target.normal;
    const int selected_value_count =
        static_cast<int>(result.value_ids.size());
    const int selected_normal_count =
        static_cast<int>(result.normal_ids.size());
    const int edge_count = static_cast<int>(result.edge_point_ids.size());
    const int rows = selected_value_count + selected_normal_count + edge_count;
    Eigen::MatrixXd design(rows, space.dimension());
    Eigen::VectorXd sqrt_weights(rows);
    for (int k = 0; k < selected_value_count; ++k) {
        const auto& sample = cloud.dofs[static_cast<std::size_t>(
            result.value_ids[static_cast<std::size_t>(k)])];
        const Eigen::Vector3d xi =
            frame.transpose() * (sample.point - target.point) / h;
        design.row(k) =
            space.basis(xi.x(), xi.y(), xi.z()).transpose();
        sqrt_weights[k] = 1.0 / (0.35 + xi.norm());
    }
    for (int k = 0; k < selected_normal_count; ++k) {
        const auto& sample = cloud.dofs[static_cast<std::size_t>(
            result.normal_ids[static_cast<std::size_t>(k)])];
        const Eigen::Vector3d xi =
            frame.transpose() * (sample.point - target.point) / h;
        const Eigen::Vector3d normal_components =
            frame.transpose() * sample.normal;
        design.row(selected_value_count + k) = normal_components.transpose()
            * space.gradient(xi.x(), xi.y(), xi.z());
        sqrt_weights[selected_value_count + k] =
            std::sqrt(0.85) / (0.35 + xi.norm());
    }
    double edge_radius_sq = 0.0;
    for (int k = 0; k < edge_count; ++k) {
        const int point_id =
            result.edge_point_ids[static_cast<std::size_t>(k)];
        if (edge_points == nullptr || point_id < 0
            || point_id >= static_cast<int>(edge_points->points.size())) {
            HarmonicCauchyFailure3D diagnostic;
            diagnostic.stage = "surface_edge_rows";
            diagnostic.entity_kind = "surface_dof";
            diagnostic.entity_id = center;
            diagnostic.actual_edge_count = edge_count;
            diagnostic.message = "surface map references an invalid edge point";
            throw HarmonicCauchyError3D(std::move(diagnostic));
        }
        const Eigen::Vector3d displacement =
            edge_points->points[static_cast<std::size_t>(point_id)].point
            - target.point;
        const Eigen::Vector3d xi = frame.transpose() * displacement / h;
        design.row(selected_value_count + selected_normal_count + k) =
            space.basis(xi.x(), xi.y(), xi.z()).transpose();
        sqrt_weights[selected_value_count + selected_normal_count + k] =
            1.0 / (0.35 + xi.norm());
        edge_radius_sq = std::max(edge_radius_sq, displacement.squaredNorm());
    }
    result.edge_radius_over_h = std::sqrt(edge_radius_sq) / h;

    HarmonicCauchyFailure3D diagnostic;
    diagnostic.stage = "surface_map_factorization";
    diagnostic.entity_kind = "surface_dof";
    diagnostic.entity_id = center;
    diagnostic.connection_id = result.relevant_connection_ids.empty()
        ? -1 : result.relevant_connection_ids.front();
    diagnostic.incident_sectors = result.value_sector_patch_ids;
    diagnostic.actual_value_counts = result.value_sector_counts;
    diagnostic.actual_normal_counts = result.normal_sector_counts;
    diagnostic.required_value_count = value_count;
    diagnostic.required_normal_count = normal_count;
    diagnostic.actual_edge_count = edge_count;
    diagnostic.value_radius_over_h = result.value_radius_over_h;
    diagnostic.normal_radius_over_h = result.normal_radius_over_h;
    diagnostic.edge_radius_over_h = result.edge_radius_over_h;
    const DesignInverse3D inverse = factor_design(
        sqrt_weights.asDiagonal() * design, space.dimension(),
        relative_cutoff, std::move(diagnostic), svd_count);
    result.sigma_max = inverse.sigma_max;
    result.sigma_min = inverse.sigma_min;
    result.condition = inverse.condition;
    result.M_value.resize(space.dimension(), selected_value_count);
    result.M_normal.resize(space.dimension(), selected_normal_count);
    result.M_edge.resize(space.dimension(), edge_count);
    for (int k = 0; k < selected_value_count; ++k)
        result.M_value.col(k) = inverse.pinv.col(k) * sqrt_weights[k];
    for (int k = 0; k < selected_normal_count; ++k) {
        result.M_normal.col(k) = inverse.pinv.col(selected_value_count + k)
            * sqrt_weights[selected_value_count + k] * h;
    }
    for (int k = 0; k < edge_count; ++k) {
        result.M_edge.col(k) = inverse.pinv.col(
            selected_value_count + selected_normal_count + k)
            * sqrt_weights[selected_value_count + selected_normal_count + k];
    }
    return result;
}

std::uint64_t hash_int_vector(std::uint64_t hash,
                              const std::vector<int>& values)
{
    hash = append_hash(hash, values.size());
    for (int value : values)
        hash = append_hash(hash, value);
    return hash;
}

std::uint64_t hash_sector_vector(
    std::uint64_t hash,
    const std::vector<std::vector<int>>& sectors)
{
    hash = append_hash(hash, sectors.size());
    for (const auto& sector : sectors)
        hash = hash_int_vector(hash, sector);
    return hash;
}

template <class Derived>
std::uint64_t hash_eigen(std::uint64_t hash,
                         const Eigen::MatrixBase<Derived>& matrix)
{
    const Eigen::Index rows = matrix.rows();
    const Eigen::Index cols = matrix.cols();
    hash = append_hash(hash, rows);
    hash = append_hash(hash, cols);
    for (Eigen::Index row = 0; row < rows; ++row) {
        for (Eigen::Index col = 0; col < cols; ++col)
            hash = append_hash(hash, matrix(row, col));
    }
    return hash;
}

std::uint64_t hash_edge_map(std::uint64_t hash,
                            const EdgeValueMap3D& map)
{
    const SharedEdgePoint3D& point = map.point;
    hash = append_hash(hash, point.id);
    hash = append_hash(hash, point.connection_id);
    hash = append_hash(hash, point.cell_id);
    hash = append_hash(hash, point.cell_count);
    hash = append_hash(hash, point.fraction);
    hash = append_hash(hash, point.quadrature_weight);
    hash = append_hash(hash, point.native_parameters[0]);
    hash = append_hash(hash, point.native_parameters[1]);
    hash = hash_eigen(hash, point.native_uv[0]);
    hash = hash_eigen(hash, point.native_uv[1]);
    hash = hash_eigen(hash, point.point);
    hash = hash_eigen(hash, point.tangent);
    hash = hash_eigen(hash, point.frame);
    hash = hash_int_vector(hash, map.sector_patch_ids[0]);
    hash = hash_int_vector(hash, map.sector_patch_ids[1]);
    hash = append_hash(hash, map.value_sector_counts[0]);
    hash = append_hash(hash, map.value_sector_counts[1]);
    hash = append_hash(hash, map.normal_sector_counts[0]);
    hash = append_hash(hash, map.normal_sector_counts[1]);
    hash = hash_int_vector(hash, map.value_ids);
    hash = hash_int_vector(hash, map.normal_ids);
    hash = hash_eigen(hash, map.E_value);
    hash = hash_eigen(hash, map.E_normal);
    hash = append_hash(hash, map.value_radius_over_h);
    hash = append_hash(hash, map.normal_radius_over_h);
    hash = append_hash(hash, map.sigma_max);
    hash = append_hash(hash, map.sigma_min);
    hash = append_hash(hash, map.condition);
    return hash;
}

std::uint64_t hash_surface_map(std::uint64_t hash,
                               const SurfaceCauchyMap3D& map)
{
    hash = append_hash(hash, map.neighborhood_fingerprint);
    hash = hash_int_vector(hash, map.value_ids);
    hash = hash_int_vector(hash, map.normal_ids);
    hash = hash_int_vector(hash, map.edge_point_ids);
    hash = hash_sector_vector(hash, map.value_sector_patch_ids);
    hash = hash_int_vector(hash, map.value_sector_counts);
    hash = hash_sector_vector(hash, map.normal_sector_patch_ids);
    hash = hash_int_vector(hash, map.normal_sector_counts);
    hash = hash_int_vector(hash, map.relevant_connection_ids);
    hash = append_hash(hash, map.incident_patch_count);
    hash = append_hash(hash, map.value_patch_imbalance);
    hash = append_hash(hash, map.normal_patch_imbalance);
    hash = hash_eigen(hash, map.M_value);
    hash = hash_eigen(hash, map.M_normal);
    hash = hash_eigen(hash, map.M_edge);
    hash = append_hash(hash, map.value_radius_over_h);
    hash = append_hash(hash, map.normal_radius_over_h);
    hash = append_hash(hash, map.edge_radius_over_h);
    hash = append_hash(hash, map.nearest_edge_distance_over_h);
    hash = append_hash(hash, map.sigma_max);
    hash = append_hash(hash, map.sigma_min);
    hash = append_hash(hash, map.condition);
    return hash;
}

} // namespace

HarmonicCauchyFit3D::HarmonicCauchyFit3D(int degree)
    : space_(degree)
{}

HarmonicCauchyFit3D HarmonicCauchyFit3D::build(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
    double h,
    HarmonicCauchyRoute3D route,
    int degree,
    int value_count,
    int normal_count,
    double relative_svd_cutoff)
{
    validate_surface_cloud(surface, cloud, h);
    if (degree < 1 || value_count <= 0 || normal_count < 0
        || normal_count > value_count || !std::isfinite(relative_svd_cutoff)
        || relative_svd_cutoff <= 0.0) {
        throw std::invalid_argument("invalid harmonic Cauchy fit parameters");
    }
    if (degree != 3 || value_count != 48 || normal_count != 28) {
        throw std::invalid_argument(
            "native harmonic Cauchy routes require cubic 48/28 maps");
    }
    if (neighborhoods.centers.size() != cloud.dofs.size()) {
        throw std::invalid_argument(
            "harmonic Cauchy fit requires complete cached neighborhoods");
    }
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q) {
        if (neighborhoods.centers[static_cast<std::size_t>(q)].center_dof != q) {
            throw std::invalid_argument(
                "harmonic Cauchy neighborhood center IDs are inconsistent");
        }
    }
    HarmonicCauchyFit3D result(degree);
    result.value_count_ = value_count;
    result.normal_count_ = normal_count;
    result.cloud_size_ = static_cast<int>(cloud.dofs.size());
    result.route_ = route;
    result.audit_.geometry_query_count = neighborhoods.geometry_query_count;
    result.audit_.svd_factorization_count = degree >= 2 ? 1U : 0U;
    std::optional<SharedEdgePointSet3D> points;
    if (route == HarmonicCauchyRoute3D::EdgeReconstructedValue) {
        points = make_shared_edge_points_3d(surface, h);
        for (const auto& point : points->points) {
            result.edge_maps_.push_back(build_edge_value_map(
                point, cloud, result.space_, h, relative_svd_cutoff,
                result.audit_.svd_factorization_count));
        }
        result.audit_.geometry_query_count += 2 * points->points.size();
        result.audit_.geometry_query_count += static_cast<std::size_t>(
            std::count_if(
                surface.geometric_connections.begin(),
                surface.geometric_connections.end(),
                [](const geometry3d::NurbsPatchEdgeConnection3D& connection) {
                    return !connection.g1;
                }));
    }
    result.surface_maps_.reserve(cloud.dofs.size());
    for (int center = 0; center < result.cloud_size_; ++center) {
        result.surface_maps_.push_back(build_surface_map(
            surface, cloud, points ? &*points : nullptr, neighborhoods,
            result.space_, center, h, route, value_count, normal_count,
            relative_svd_cutoff,
            result.audit_.svd_factorization_count));
    }
    constexpr std::uint64_t offset = 1469598103934665603ULL;
    std::uint64_t fingerprint = offset;
    fingerprint = append_hash(fingerprint, route);
    fingerprint = append_hash(fingerprint, degree);
    fingerprint = append_hash(fingerprint, value_count);
    fingerprint = append_hash(fingerprint, normal_count);
    fingerprint = append_hash(
        fingerprint, result.audit_.geometry_query_count);
    fingerprint = append_hash(
        fingerprint, result.audit_.svd_factorization_count);
    for (const auto& map : result.edge_maps_)
        fingerprint = hash_edge_map(fingerprint, map);
    for (const auto& map : result.surface_maps_)
        fingerprint = hash_surface_map(fingerprint, map);
    result.audit_.fingerprint = fingerprint;
    return result;
}

HarmonicCauchyFit3D HarmonicCauchyFit3D::build_legacy(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h,
    LegacySurfaceCauchyPolicy3D policy,
    int degree,
    int value_count,
    int normal_count,
    double relative_svd_cutoff)
{
    validate_surface_cloud(surface, cloud, h);
    if (degree < 1 || value_count <= 0 || normal_count < 0
        || normal_count > value_count || !std::isfinite(relative_svd_cutoff)
        || relative_svd_cutoff <= 0.0) {
        throw std::invalid_argument("invalid legacy harmonic Cauchy parameters");
    }
    SurfaceNonG1EdgeNeighborhoodSet3D neighborhoods;
    neighborhoods.centers.resize(cloud.dofs.size());
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q)
        neighborhoods.centers[static_cast<std::size_t>(q)].center_dof = q;

    HarmonicCauchyFit3D result(degree);
    result.value_count_ = value_count;
    result.normal_count_ = normal_count;
    result.cloud_size_ = static_cast<int>(cloud.dofs.size());
    result.route_ = HarmonicCauchyRoute3D::G1ValueG1Normal;
    result.legacy_policy_ = policy;
    result.audit_.svd_factorization_count = degree >= 2 ? 1U : 0U;
    LegacyCauchySummary3D summary;
    summary.policy = policy;
    summary.degree = degree;
    summary.value_count = value_count;
    summary.normal_count = normal_count;
    summary.value_count_min = std::numeric_limits<int>::max();
    summary.normal_count_min = std::numeric_limits<int>::max();
    summary.incident_patch_count_min = std::numeric_limits<int>::max();
    result.surface_maps_.reserve(cloud.dofs.size());
    for (int center = 0; center < result.cloud_size_; ++center) {
        SurfaceCauchyMap3D map = build_surface_map(
            surface, cloud, nullptr, neighborhoods, result.space_, center, h,
            result.route_, value_count, normal_count, relative_svd_cutoff,
            result.audit_.svd_factorization_count, policy);
        const int actual_values = static_cast<int>(map.value_ids.size());
        const int actual_normals = static_cast<int>(map.normal_ids.size());
        summary.value_count_min =
            std::min(summary.value_count_min, actual_values);
        summary.value_count_max =
            std::max(summary.value_count_max, actual_values);
        summary.normal_count_min =
            std::min(summary.normal_count_min, actual_normals);
        summary.normal_count_max =
            std::max(summary.normal_count_max, actual_normals);
        const double radius =
            std::max(map.value_radius_over_h, map.normal_radius_over_h);
        summary.radius_max_over_h =
            std::max(summary.radius_max_over_h, radius);
        summary.radius_mean_over_h += radius;
        summary.incident_patch_count_min = std::min(
            summary.incident_patch_count_min, map.incident_patch_count);
        summary.incident_patch_count_max = std::max(
            summary.incident_patch_count_max, map.incident_patch_count);
        summary.value_patch_imbalance_max = std::max(
            summary.value_patch_imbalance_max, map.value_patch_imbalance);
        summary.normal_patch_imbalance_max = std::max(
            summary.normal_patch_imbalance_max, map.normal_patch_imbalance);
        result.surface_maps_.push_back(std::move(map));
    }
    if (!result.surface_maps_.empty()) {
        summary.radius_mean_over_h /=
            static_cast<double>(result.surface_maps_.size());
    }
    result.legacy_summary_ = summary;
    constexpr std::uint64_t offset = 1469598103934665603ULL;
    std::uint64_t fingerprint = offset;
    fingerprint = append_hash(fingerprint, policy);
    fingerprint = append_hash(fingerprint, degree);
    fingerprint = append_hash(fingerprint, value_count);
    fingerprint = append_hash(fingerprint, normal_count);
    fingerprint = append_hash(
        fingerprint, result.audit_.geometry_query_count);
    fingerprint = append_hash(
        fingerprint, result.audit_.svd_factorization_count);
    for (const auto& map : result.surface_maps_)
        fingerprint = hash_surface_map(fingerprint, map);
    result.audit_.fingerprint = fingerprint;
    return result;
}

HarmonicCauchyApplyResult3D HarmonicCauchyFit3D::apply(
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump) const
{
    if (value_jump.size() != cloud_size_ || normal_jump.size() != cloud_size_)
        throw std::invalid_argument("jump data size does not match surface DOFs");
    HarmonicCauchyApplyResult3D result;
    result.fingerprint_before = audit_.fingerprint;
    result.edge_values.resize(static_cast<int>(edge_maps_.size()));
    for (int e = 0; e < result.edge_values.size(); ++e) {
        const auto& map = edge_maps_[static_cast<std::size_t>(e)];
        result.edge_values[e] =
            map.E_value.dot(gather(value_jump, map.value_ids))
            + map.E_normal.dot(gather(normal_jump, map.normal_ids));
    }
    result.coefficients = Eigen::MatrixXd::Zero(
        static_cast<int>(surface_maps_.size()), space_.dimension());
    for (int center = 0;
         center < static_cast<int>(surface_maps_.size()); ++center) {
        const auto& map = surface_maps_[static_cast<std::size_t>(center)];
        const Eigen::VectorXd values = gather(value_jump, map.value_ids);
        const Eigen::VectorXd normals = gather(normal_jump, map.normal_ids);
        Eigen::VectorXd coefficients;
        if (map.edge_point_ids.empty()) {
            coefficients = map.M_value * values + map.M_normal * normals;
        } else {
            coefficients = map.M_value * values + map.M_normal * normals
                + map.M_edge * gather(result.edge_values, map.edge_point_ids);
        }
        result.coefficients.row(center) = coefficients.transpose();
    }
    result.fingerprint_after = audit_.fingerprint;
    return result;
}

const HarmonicPolynomialSpace3D& HarmonicCauchyFit3D::space() const noexcept
{
    return space_;
}

int HarmonicCauchyFit3D::degree() const noexcept { return space_.degree(); }
int HarmonicCauchyFit3D::value_count() const noexcept { return value_count_; }
int HarmonicCauchyFit3D::normal_count() const noexcept { return normal_count_; }

std::optional<LegacySurfaceCauchyPolicy3D>
HarmonicCauchyFit3D::legacy_policy() const noexcept
{
    return legacy_policy_;
}

std::optional<LegacyCauchySummary3D>
HarmonicCauchyFit3D::legacy_summary() const
{
    return legacy_summary_;
}

std::vector<double> HarmonicCauchyFit3D::condition_values() const
{
    std::vector<double> result;
    result.reserve(surface_maps_.size());
    for (const auto& map : surface_maps_)
        result.push_back(map.condition);
    return result;
}

const std::vector<EdgeValueMap3D>&
HarmonicCauchyFit3D::edge_maps() const noexcept
{
    return edge_maps_;
}

const std::vector<SurfaceCauchyMap3D>&
HarmonicCauchyFit3D::surface_maps() const noexcept
{
    return surface_maps_;
}

const HarmonicCauchyPreprocessAudit3D&
HarmonicCauchyFit3D::audit() const noexcept
{
    return audit_;
}

} // namespace kfbim::app3d
