#include <apps/harmonic_cauchy_fit_3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

namespace kfbim::app3d {

namespace {

struct EdgeGeometry3D {
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d positive_parameter_tangent = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
};

EdgeGeometry3D edge_geometry(
    const geometry3d::NurbsSurfacePatch3D& patch,
    geometry3d::NurbsPatchEdge3D edge,
    double parameter)
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
    Eigen::Vector3d normal = derivatives.du.cross(derivatives.dv);
    const double normal_norm = normal.norm();
    if (!std::isfinite(normal_norm) || normal_norm <= 1.0e-14) {
        HarmonicCauchyFailure3D diagnostic;
        diagnostic.stage = "shared_edge_geometry";
        diagnostic.entity_kind = "patch_edge";
        diagnostic.message = "native patch edge has a degenerate normal";
        throw HarmonicCauchyError3D(std::move(diagnostic));
    }
    normal /= normal_norm;
    return {{u, v},
            derivatives.point,
            varying_v ? derivatives.dv : derivatives.du,
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
                first_patch, connection.first.edge, first_parameter);
            const EdgeGeometry3D second = edge_geometry(
                second_patch, connection.second.edge, second_parameter);
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

} // namespace kfbim::app3d
