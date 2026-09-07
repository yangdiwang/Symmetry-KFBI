#include "src/support/trace/geometric_feature_trace_fit_3d.hpp"

#include "src/support/cauchy/harmonic_polynomial_space_3d.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace kfbim::app3d {
namespace {

using geometry3d::NurbsPatchEdge3D;
using geometry3d::NurbsPatchEdgeConnection3D;

double integer_power(double value, int power)
{
    double result = 1.0;
    for (int k = 0; k < power; ++k)
        result *= value;
    return result;
}

struct MonomialPower2D {
    int x = 0;
    int y = 0;
};

std::vector<MonomialPower2D> monomial_powers_2d(int degree)
{
    std::vector<MonomialPower2D> result;
    for (int total = 0; total <= degree; ++total) {
        for (int px = 0; px <= total; ++px)
            result.push_back({px, total - px});
    }
    return result;
}

Eigen::VectorXd monomial_basis_2d(
    const std::vector<MonomialPower2D>& powers,
    double x,
    double y)
{
    Eigen::VectorXd result(static_cast<int>(powers.size()));
    for (int k = 0; k < result.size(); ++k) {
        const MonomialPower2D power =
            powers[static_cast<std::size_t>(k)];
        result[k] = integer_power(x, power.x)
                  * integer_power(y, power.y);
    }
    return result;
}

std::vector<Eigen::Vector2d> canonical_triangle_nodes(int degree)
{
    if (degree <= 0)
        return {Eigen::Vector2d::Zero()};
    const Eigen::Vector2d a(-1.0, -1.0);
    const Eigen::Vector2d b(1.0, -1.0);
    const Eigen::Vector2d c(0.0, 1.0);
    std::vector<Eigen::Vector2d> result;
    result.reserve(static_cast<std::size_t>(
        (degree + 1) * (degree + 2) / 2));
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree - i; ++j) {
            const int k = degree - i - j;
            result.push_back(
                (static_cast<double>(i) * a
                 + static_cast<double>(j) * b
                 + static_cast<double>(k) * c)
                / static_cast<double>(degree));
        }
    }
    return result;
}

class DisjointSet {
public:
    explicit DisjointSet(int size)
        : parent_(static_cast<std::size_t>(size))
        , rank_(static_cast<std::size_t>(size), 0)
    {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int find(int value)
    {
        int& parent = parent_[static_cast<std::size_t>(value)];
        if (parent != value)
            parent = find(parent);
        return parent;
    }

    void unite(int first, int second)
    {
        first = find(first);
        second = find(second);
        if (first == second)
            return;
        if (rank_[static_cast<std::size_t>(first)]
            < rank_[static_cast<std::size_t>(second)]) {
            std::swap(first, second);
        }
        parent_[static_cast<std::size_t>(second)] = first;
        if (rank_[static_cast<std::size_t>(first)]
            == rank_[static_cast<std::size_t>(second)]) {
            ++rank_[static_cast<std::size_t>(first)];
        }
    }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

Eigen::Vector3d edge_point(
    const NativeNurbsSurface3D& surface,
    const geometry3d::NurbsPatchEdgeInterval3D& interval,
    double fraction)
{
    if (interval.patch < 0
        || interval.patch >= static_cast<int>(surface.patches.size())) {
        throw std::out_of_range("feature interval has an invalid patch");
    }
    const auto& patch =
        surface.patches[static_cast<std::size_t>(interval.patch)];
    const double parameter =
        interval.begin + fraction * (interval.end - interval.begin);
    double u = patch.domain_start_u();
    double v = patch.domain_start_v();
    switch (interval.edge) {
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
    return patch.evaluate(u, v);
}

struct TangentFrame3D {
    Eigen::Vector3d tangent1 = Eigen::Vector3d::UnitX();
    Eigen::Vector3d tangent2 = Eigen::Vector3d::UnitY();
};

TangentFrame3D tangent_frame(const SurfaceDof3D& dof)
{
    Eigen::Vector3d normal = dof.normal.normalized();
    Eigen::Vector3d tangent1 =
        dof.tangent1 - dof.tangent1.dot(normal) * normal;
    if (!(tangent1.norm() > 1.0e-12))
        tangent1 = normal.unitOrthogonal();
    tangent1.normalize();
    Eigen::Vector3d tangent2 = normal.cross(tangent1);
    if (!(tangent2.norm() > 1.0e-12))
        throw std::runtime_error("surface DOF has a degenerate tangent frame");
    tangent2.normalize();
    return {tangent1, tangent2};
}

struct FeatureSegment3D {
    NurbsPatchEdgeConnection3D connection;
    int first_sheet = -1;
    int second_sheet = -1;
    std::vector<Eigen::Vector3d> polyline;
    double length = 0.0;
    std::array<int, 2> endpoint_vertices{{-1, -1}};
};

struct FeatureVertex3D {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    std::vector<int> incident_segments;
    std::set<int> sheets;
    bool corner = false;
};

struct SegmentClosestPoint3D {
    double distance = std::numeric_limits<double>::infinity();
    double fraction = 0.0;
};

SegmentClosestPoint3D closest_point(
    const FeatureSegment3D& segment,
    const Eigen::Vector3d& point)
{
    SegmentClosestPoint3D result;
    const int pieces = static_cast<int>(segment.polyline.size()) - 1;
    for (int piece = 0; piece < pieces; ++piece) {
        const Eigen::Vector3d& a =
            segment.polyline[static_cast<std::size_t>(piece)];
        const Eigen::Vector3d& b =
            segment.polyline[static_cast<std::size_t>(piece + 1)];
        const Eigen::Vector3d direction = b - a;
        const double denominator = direction.squaredNorm();
        double local = 0.0;
        if (denominator > 0.0) {
            local = std::clamp(
                (point - a).dot(direction) / denominator, 0.0, 1.0);
        }
        const double distance = (point - (a + local * direction)).norm();
        if (distance < result.distance) {
            result.distance = distance;
            result.fraction =
                (static_cast<double>(piece) + local)
                / static_cast<double>(pieces);
        }
    }
    return result;
}

std::vector<int> build_patch_sheets(
    const NativeNurbsSurface3D& surface)
{
    const int patch_count = static_cast<int>(surface.patches.size());
    DisjointSet sets(patch_count);
    for (const NurbsPatchEdgeConnection3D& connection :
         surface.geometric_connections) {
        if (connection.g1)
            sets.unite(connection.first.patch, connection.second.patch);
    }
    std::map<int, int> sheet_by_root;
    std::vector<int> result(static_cast<std::size_t>(patch_count), -1);
    for (int patch = 0; patch < patch_count; ++patch) {
        const int root = sets.find(patch);
        auto inserted = sheet_by_root.emplace(
            root, static_cast<int>(sheet_by_root.size()));
        result[static_cast<std::size_t>(patch)] = inserted.first->second;
    }
    return result;
}

Eigen::MatrixXd build_even_harmonic_lift(
    const HarmonicPolynomialSpace3D& space,
    const std::vector<MonomialPower2D>& powers)
{
    const std::vector<Eigen::Vector2d> nodes =
        canonical_triangle_nodes(space.degree());
    const int count = static_cast<int>(nodes.size());
    Eigen::MatrixXd design(2 * count, space.dimension());
    Eigen::MatrixXd data =
        Eigen::MatrixXd::Zero(2 * count, static_cast<int>(powers.size()));
    for (int row = 0; row < count; ++row) {
        const Eigen::Vector2d point = nodes[static_cast<std::size_t>(row)];
        design.row(row) =
            space.basis(point.x(), point.y(), 0.0).transpose();
        design.row(count + row) =
            space.gradient(point.x(), point.y(), 0.0).row(2);
        data.row(row) =
            monomial_basis_2d(powers, point.x(), point.y()).transpose();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(design);
    const Eigen::VectorXd singular = svd.singularValues();
    if (singular.size() != space.dimension()
        || !(singular[singular.size() - 1] > 1.0e-12 * singular[0])) {
        throw std::runtime_error(
            "even harmonic trace lift is rank deficient");
    }
    const Eigen::MatrixXd lift =
        svd_pseudoinverse_3d(design, 1.0e-13) * data;
    if ((design * lift - data).cwiseAbs().maxCoeff() > 2.0e-10) {
        throw std::runtime_error(
            "even harmonic trace lift failed polynomial reproduction");
    }
    return lift;
}

} // namespace

struct GeometricFeatureTraceFit3D::Impl {
    Impl(const NativeNurbsSurface3D& surface,
         const SurfaceDofCloud3D& cloud,
         double input_h,
         int input_degree,
         GeometricFeatureTraceOptions3D input_options)
        : h(input_h)
        , degree(input_degree)
        , options(input_options)
        , space(input_degree)
        , powers(monomial_powers_2d(input_degree))
        , even_lift(build_even_harmonic_lift(space, powers))
        , value_size(static_cast<int>(cloud.dofs.size()))
    {
        validate_inputs(surface, cloud);
        patch_sheets = build_patch_sheets(surface);
        build_sheet_dofs(cloud);
        build_features(surface);
        maps.resize(cloud.dofs.size());
        for (int center = 0; center < value_size; ++center)
            maps[static_cast<std::size_t>(center)] =
                build_map(surface, cloud, center);
    }

    void validate_inputs(const NativeNurbsSurface3D& surface,
                         const SurfaceDofCloud3D& cloud) const
    {
        if (!(h > 0.0) || !std::isfinite(h))
            throw std::invalid_argument("feature trace fit requires positive h");
        if (degree < 1)
            throw std::invalid_argument(
                "feature trace polynomial degree must be positive");
        if (surface.patches.size() != cloud.patches.size()
            || surface.patches.empty() || cloud.dofs.empty()) {
            throw std::invalid_argument(
                "feature trace fit requires matching surface patches and DOFs");
        }
        if (!(options.activation_radius_over_h > 0.0)
            || !(options.vertex_radius_over_h > 0.0)
            || options.polyline_segments < 2
            || options.minimum_samples_per_sheet < 1) {
            throw std::invalid_argument("invalid feature trace fit options");
        }
    }

    void build_sheet_dofs(const SurfaceDofCloud3D& cloud)
    {
        const int sheet_count =
            1 + *std::max_element(patch_sheets.begin(), patch_sheets.end());
        sheet_dofs.resize(static_cast<std::size_t>(sheet_count));
        for (int dof_id = 0; dof_id < value_size; ++dof_id) {
            const int patch =
                cloud.dofs[static_cast<std::size_t>(dof_id)].patch_id;
            if (patch < 0
                || patch >= static_cast<int>(patch_sheets.size())) {
                throw std::runtime_error(
                    "feature trace DOF has an invalid patch");
            }
            sheet_dofs[static_cast<std::size_t>(
                patch_sheets[static_cast<std::size_t>(patch)])]
                .push_back(dof_id);
        }
    }

    void build_features(const NativeNurbsSurface3D& surface)
    {
        for (const NurbsPatchEdgeConnection3D& connection :
             surface.geometric_connections) {
            if (connection.g1)
                continue;
            const int first_sheet = patch_sheets[static_cast<std::size_t>(
                connection.first.patch)];
            const int second_sheet = patch_sheets[static_cast<std::size_t>(
                connection.second.patch)];
            if (first_sheet == second_sheet)
                continue;
            FeatureSegment3D segment;
            segment.connection = connection;
            segment.first_sheet = first_sheet;
            segment.second_sheet = second_sheet;
            segment.polyline.reserve(
                static_cast<std::size_t>(options.polyline_segments + 1));
            for (int sample = 0;
                 sample <= options.polyline_segments;
                 ++sample) {
                segment.polyline.push_back(edge_point(
                    surface,
                    connection.first,
                    static_cast<double>(sample)
                        / static_cast<double>(options.polyline_segments)));
                if (sample > 0) {
                    segment.length +=
                        (segment.polyline[static_cast<std::size_t>(sample)]
                         - segment.polyline[
                             static_cast<std::size_t>(sample - 1)])
                            .norm();
                }
            }
            if (!(segment.length > 0.0)
                || !std::isfinite(segment.length)) {
                throw std::runtime_error(
                    "feature trace encountered a degenerate edge");
            }
            segments.push_back(std::move(segment));
        }
        summary.feature_segments = static_cast<int>(segments.size());
        build_vertices();
    }

    void build_vertices()
    {
        double scale = 1.0;
        for (const FeatureSegment3D& segment : segments) {
            for (const Eigen::Vector3d& point : segment.polyline)
                scale = std::max(scale, point.lpNorm<Eigen::Infinity>());
        }
        const double tolerance = 2.0e-9 * scale;
        for (int segment_id = 0;
             segment_id < static_cast<int>(segments.size());
             ++segment_id) {
            FeatureSegment3D& segment =
                segments[static_cast<std::size_t>(segment_id)];
            for (int endpoint = 0; endpoint < 2; ++endpoint) {
                const Eigen::Vector3d& point =
                    endpoint == 0 ? segment.polyline.front()
                                  : segment.polyline.back();
                int vertex_id = -1;
                for (int candidate = 0;
                     candidate < static_cast<int>(vertices.size());
                     ++candidate) {
                    if ((vertices[static_cast<std::size_t>(candidate)].point
                         - point)
                            .norm()
                        <= tolerance) {
                        vertex_id = candidate;
                        break;
                    }
                }
                if (vertex_id < 0) {
                    vertex_id = static_cast<int>(vertices.size());
                    FeatureVertex3D vertex;
                    vertex.point = point;
                    vertices.push_back(std::move(vertex));
                }
                segment.endpoint_vertices[static_cast<std::size_t>(endpoint)] =
                    vertex_id;
                vertices[static_cast<std::size_t>(vertex_id)]
                    .incident_segments.push_back(segment_id);
            }
        }

        for (int vertex_id = 0;
             vertex_id < static_cast<int>(vertices.size());
             ++vertex_id) {
            FeatureVertex3D& vertex =
                vertices[static_cast<std::size_t>(vertex_id)];
            std::vector<Eigen::Vector3d> direction_axes;
            for (int segment_id : vertex.incident_segments) {
                const FeatureSegment3D& segment =
                    segments[static_cast<std::size_t>(segment_id)];
                vertex.sheets.insert(segment.first_sheet);
                vertex.sheets.insert(segment.second_sheet);
                Eigen::Vector3d direction;
                if (segment.endpoint_vertices[0]
                    == vertex_id) {
                    direction = segment.polyline[1]
                              - segment.polyline.front();
                } else {
                    direction =
                        segment.polyline[segment.polyline.size() - 2]
                        - segment.polyline.back();
                }
                if (!(direction.norm() > 0.0))
                    continue;
                direction.normalize();
                bool known_axis = false;
                for (const Eigen::Vector3d& axis : direction_axes) {
                    if (std::abs(axis.dot(direction)) > 0.985) {
                        known_axis = true;
                        break;
                    }
                }
                if (!known_axis)
                    direction_axes.push_back(direction);
            }
            vertex.corner =
                vertex.sheets.size() >= 3 || direction_axes.size() >= 2;
            if (vertex.corner)
                ++summary.feature_vertices;
        }
    }

    TangentFrame3D frame_for_sheet(
        const SurfaceDofCloud3D& cloud,
        int sheet,
        int target_sheet,
        int center) const
    {
        if (sheet == target_sheet)
            return tangent_frame(cloud.dofs[static_cast<std::size_t>(center)]);
        const Eigen::Vector3d& target =
            cloud.dofs[static_cast<std::size_t>(center)].point;
        int nearest = -1;
        double nearest_sq = std::numeric_limits<double>::infinity();
        for (int dof : sheet_dofs[static_cast<std::size_t>(sheet)]) {
            const double distance_sq =
                (cloud.dofs[static_cast<std::size_t>(dof)].point - target)
                    .squaredNorm();
            if (distance_sq < nearest_sq) {
                nearest_sq = distance_sq;
                nearest = dof;
            }
        }
        if (nearest < 0)
            throw std::runtime_error("feature sheet contains no surface DOFs");
        return tangent_frame(cloud.dofs[static_cast<std::size_t>(nearest)]);
    }

    Eigen::Vector2d sheet_coordinate(
        const Eigen::Vector3d& origin,
        const TangentFrame3D& frame,
        const Eigen::Vector3d& point) const
    {
        const Eigen::Vector3d displacement = (point - origin) / h;
        return {displacement.dot(frame.tangent1),
                displacement.dot(frame.tangent2)};
    }

    std::vector<double> constraint_fractions(
        const FeatureSegment3D& segment,
        double closest_fraction,
        bool force_first,
        bool force_second) const
    {
        const int count = degree + 1;
        const double window =
            std::min(1.0, 3.0 * h / segment.length);
        double lower = 0.0;
        double upper = 1.0;
        if (force_first && !force_second) {
            upper = window;
        } else if (force_second && !force_first) {
            lower = 1.0 - window;
        } else if (!force_first && !force_second) {
            lower = std::clamp(
                closest_fraction - 0.5 * window, 0.0, 1.0 - window);
            upper = lower + window;
        }
        std::vector<double> result(static_cast<std::size_t>(count), lower);
        for (int k = 0; k < count; ++k) {
            result[static_cast<std::size_t>(k)] =
                lower + (upper - lower) * static_cast<double>(k)
                      / static_cast<double>(count - 1);
        }
        return result;
    }

    GeometricFeatureTraceMap3D build_map(
        const NativeNurbsSurface3D& surface,
        const SurfaceDofCloud3D& cloud,
        int center) const
    {
        const SurfaceDof3D& target =
            cloud.dofs[static_cast<std::size_t>(center)];
        const int target_sheet = patch_sheets[static_cast<std::size_t>(
            target.patch_id)];
        std::set<int> active_segments;
        std::vector<SegmentClosestPoint3D> closest(segments.size());
        for (int segment_id = 0;
             segment_id < static_cast<int>(segments.size());
             ++segment_id) {
            const FeatureSegment3D& segment =
                segments[static_cast<std::size_t>(segment_id)];
            closest[static_cast<std::size_t>(segment_id)] =
                closest_point(segment, target.point);
            if ((segment.first_sheet == target_sheet
                 || segment.second_sheet == target_sheet)
                && closest[static_cast<std::size_t>(segment_id)].distance
                    <= options.activation_radius_over_h * h) {
                active_segments.insert(segment_id);
            }
        }

        std::vector<int> active_vertices;
        for (int vertex_id = 0;
             vertex_id < static_cast<int>(vertices.size());
             ++vertex_id) {
            const FeatureVertex3D& vertex =
                vertices[static_cast<std::size_t>(vertex_id)];
            if (!vertex.corner
                || vertex.sheets.count(target_sheet) == 0
                || (vertex.point - target.point).norm()
                    > options.vertex_radius_over_h * h) {
                continue;
            }
            active_vertices.push_back(vertex_id);
            active_segments.insert(
                vertex.incident_segments.begin(),
                vertex.incident_segments.end());
        }

        GeometricFeatureTraceMap3D result;
        if (active_segments.empty())
            return result;
        result.kind = active_vertices.empty()
            ? GeometricFeatureTraceKind3D::Edge
            : GeometricFeatureTraceKind3D::Vertex;
        result.feature_count = static_cast<int>(active_segments.size());

        std::set<int> active_sheet_set;
        active_sheet_set.insert(target_sheet);
        for (int segment_id : active_segments) {
            const FeatureSegment3D& segment =
                segments[static_cast<std::size_t>(segment_id)];
            active_sheet_set.insert(segment.first_sheet);
            active_sheet_set.insert(segment.second_sheet);
        }
        const std::vector<int> active_sheets(
            active_sheet_set.begin(), active_sheet_set.end());
        result.sheet_count = static_cast<int>(active_sheets.size());
        std::map<int, int> sheet_block;
        std::vector<TangentFrame3D> frames;
        frames.reserve(active_sheets.size());
        for (int block = 0;
             block < static_cast<int>(active_sheets.size());
             ++block) {
            const int sheet =
                active_sheets[static_cast<std::size_t>(block)];
            sheet_block[sheet] = block;
            frames.push_back(frame_for_sheet(
                cloud, sheet, target_sheet, center));
        }

        const int monomial_count = static_cast<int>(powers.size());
        const int coefficient_count =
            monomial_count * static_cast<int>(active_sheets.size());
        const int requested_samples = std::max(
            options.minimum_samples_per_sheet, 2 * monomial_count + 4);
        std::vector<std::vector<int>> selected_by_sheet(
            active_sheets.size());
        int sample_count = 0;
        for (int block = 0;
             block < static_cast<int>(active_sheets.size());
             ++block) {
            const int sheet =
                active_sheets[static_cast<std::size_t>(block)];
            std::vector<std::pair<double, int>> candidates;
            candidates.reserve(
                sheet_dofs[static_cast<std::size_t>(sheet)].size());
            for (int dof : sheet_dofs[static_cast<std::size_t>(sheet)]) {
                candidates.emplace_back(
                    (cloud.dofs[static_cast<std::size_t>(dof)].point
                     - target.point)
                        .squaredNorm(),
                    dof);
            }
            std::sort(candidates.begin(), candidates.end());
            const int selected = std::min(
                requested_samples, static_cast<int>(candidates.size()));
            if (selected < monomial_count) {
                throw std::runtime_error(
                    "feature trace sheet has too few polynomial samples");
            }
            auto& ids =
                selected_by_sheet[static_cast<std::size_t>(block)];
            ids.reserve(static_cast<std::size_t>(selected));
            for (int k = 0; k < selected; ++k)
                ids.push_back(candidates[static_cast<std::size_t>(k)].second);
            sample_count += selected;
        }

        Eigen::MatrixXd design =
            Eigen::MatrixXd::Zero(sample_count, coefficient_count);
        Eigen::VectorXd sqrt_weights(sample_count);
        result.value_ids.reserve(static_cast<std::size_t>(sample_count));
        int row = 0;
        for (int block = 0;
             block < static_cast<int>(active_sheets.size());
             ++block) {
            for (int dof :
                 selected_by_sheet[static_cast<std::size_t>(block)]) {
                const Eigen::Vector3d& point =
                    cloud.dofs[static_cast<std::size_t>(dof)].point;
                const Eigen::Vector2d coordinate = sheet_coordinate(
                    target.point,
                    frames[static_cast<std::size_t>(block)],
                    point);
                design.block(row, block * monomial_count,
                             1, monomial_count) =
                    monomial_basis_2d(
                        powers, coordinate.x(), coordinate.y())
                        .transpose();
                sqrt_weights[row] =
                    1.0 / (0.55 + (point - target.point).norm() / h);
                result.value_ids.push_back(dof);
                ++row;
            }
        }

        const int constraint_count =
            static_cast<int>(active_segments.size()) * (degree + 1);
        Eigen::MatrixXd constraints =
            Eigen::MatrixXd::Zero(constraint_count, coefficient_count);
        row = 0;
        for (int segment_id : active_segments) {
            const FeatureSegment3D& segment =
                segments[static_cast<std::size_t>(segment_id)];
            bool force_first = false;
            bool force_second = false;
            for (int vertex_id : active_vertices) {
                force_first =
                    force_first
                    || segment.endpoint_vertices[0] == vertex_id;
                force_second =
                    force_second
                    || segment.endpoint_vertices[1] == vertex_id;
            }
            const std::vector<double> fractions = constraint_fractions(
                segment,
                closest[static_cast<std::size_t>(segment_id)].fraction,
                force_first,
                force_second);
            const int first_block = sheet_block.at(segment.first_sheet);
            const int second_block = sheet_block.at(segment.second_sheet);
            for (double fraction : fractions) {
                const Eigen::Vector3d point =
                    edge_point(surface, segment.connection.first, fraction);
                const Eigen::Vector2d first_coordinate = sheet_coordinate(
                    target.point,
                    frames[static_cast<std::size_t>(first_block)],
                    point);
                const Eigen::Vector2d second_coordinate = sheet_coordinate(
                    target.point,
                    frames[static_cast<std::size_t>(second_block)],
                    point);
                constraints.block(
                    row, first_block * monomial_count,
                    1, monomial_count) =
                    monomial_basis_2d(
                        powers,
                        first_coordinate.x(),
                        first_coordinate.y())
                        .transpose();
                constraints.block(
                    row, second_block * monomial_count,
                    1, monomial_count) =
                    -monomial_basis_2d(
                        powers,
                        second_coordinate.x(),
                        second_coordinate.y())
                         .transpose();
                ++row;
            }
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> constraint_svd(
            constraints, Eigen::ComputeFullV);
        const Eigen::VectorXd constraint_singular =
            constraint_svd.singularValues();
        int constraint_rank = 0;
        if (constraint_singular.size() > 0
            && constraint_singular[0] > 0.0) {
            const double cutoff = 2.0e-11 * constraint_singular[0];
            for (int k = 0; k < constraint_singular.size(); ++k) {
                if (constraint_singular[k] > cutoff)
                    ++constraint_rank;
            }
        }
        const int nullity = coefficient_count - constraint_rank;
        if (nullity <= 0) {
            throw std::runtime_error(
                "feature trace constraints removed every polynomial mode");
        }
        const Eigen::MatrixXd nullspace =
            constraint_svd.matrixV().rightCols(nullity);
        Eigen::MatrixXd weighted_design = design * nullspace;
        for (int sample = 0; sample < sample_count; ++sample)
            weighted_design.row(sample) *= sqrt_weights[sample];
        Eigen::JacobiSVD<Eigen::MatrixXd> fit_svd(weighted_design);
        const Eigen::VectorXd fit_singular = fit_svd.singularValues();
        if (fit_singular.size() != nullity
            || !fit_singular.allFinite()
            || !(fit_singular[0] > 0.0)
            || !(fit_singular[fit_singular.size() - 1]
                 > 3.0e-12 * fit_singular[0])) {
            throw std::runtime_error(
                "feature trace constrained fit is rank deficient at surface DOF "
                + std::to_string(center));
        }
        result.condition =
            fit_singular[0] / fit_singular[fit_singular.size() - 1];
        Eigen::MatrixXd fitted_coefficients =
            nullspace
            * svd_pseudoinverse_3d(weighted_design, 3.0e-12);
        for (int sample = 0; sample < sample_count; ++sample)
            fitted_coefficients.col(sample) *= sqrt_weights[sample];
        result.constraint_residual =
            (constraints * fitted_coefficients).cwiseAbs().maxCoeff();

        const int target_block = sheet_block.at(target_sheet);
        result.harmonic_map =
            even_lift
            * fitted_coefficients.middleRows(
                target_block * monomial_count, monomial_count);
        if (!result.harmonic_map.allFinite()) {
            throw std::runtime_error(
                "feature trace map contains NaN/Inf at surface DOF "
                + std::to_string(center));
        }
        return result;
    }

    double h = 0.0;
    int degree = 0;
    GeometricFeatureTraceOptions3D options;
    HarmonicPolynomialSpace3D space;
    std::vector<MonomialPower2D> powers;
    Eigen::MatrixXd even_lift;
    int value_size = 0;
    std::vector<int> patch_sheets;
    std::vector<std::vector<int>> sheet_dofs;
    std::vector<FeatureSegment3D> segments;
    std::vector<FeatureVertex3D> vertices;
    std::vector<GeometricFeatureTraceMap3D> maps;
    mutable GeometricFeatureTraceSummary3D summary;
};

GeometricFeatureTraceFit3D::GeometricFeatureTraceFit3D(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h,
    int degree,
    GeometricFeatureTraceOptions3D options)
    : impl_(std::make_shared<Impl>(
          surface, cloud, h, degree, std::move(options)))
{
    for (const GeometricFeatureTraceMap3D& map : impl_->maps) {
        if (!map.active())
            continue;
        ++impl_->summary.active_maps;
        if (map.kind == GeometricFeatureTraceKind3D::Vertex)
            ++impl_->summary.vertex_maps;
        else
            ++impl_->summary.edge_maps;
        impl_->summary.maximum_features_per_map = std::max(
            impl_->summary.maximum_features_per_map, map.feature_count);
        impl_->summary.maximum_sheets_per_map = std::max(
            impl_->summary.maximum_sheets_per_map, map.sheet_count);
        impl_->summary.maximum_constraint_residual = std::max(
            impl_->summary.maximum_constraint_residual,
            map.constraint_residual);
    }
}

int GeometricFeatureTraceFit3D::degree() const noexcept
{
    return impl_->degree;
}

int GeometricFeatureTraceFit3D::harmonic_dimension() const noexcept
{
    return impl_->space.dimension();
}

bool GeometricFeatureTraceFit3D::active(int center_dof) const
{
    return map(center_dof).active();
}

const GeometricFeatureTraceMap3D& GeometricFeatureTraceFit3D::map(
    int center_dof) const
{
    if (center_dof < 0
        || center_dof >= static_cast<int>(impl_->maps.size())) {
        throw std::out_of_range(
            "feature trace center is outside the surface cloud");
    }
    return impl_->maps[static_cast<std::size_t>(center_dof)];
}

Eigen::VectorXd GeometricFeatureTraceFit3D::coefficients(
    int center_dof,
    const Eigen::VectorXd& value_jump) const
{
    if (value_jump.size() != impl_->value_size)
        throw std::invalid_argument(
            "feature trace values do not match the surface cloud");
    const GeometricFeatureTraceMap3D& trace_map = map(center_dof);
    if (!trace_map.active())
        throw std::invalid_argument(
            "regular surface DOF has no geometric feature trace map");
    Eigen::VectorXd local_values(
        static_cast<int>(trace_map.value_ids.size()));
    for (int k = 0; k < local_values.size(); ++k) {
        local_values[k] =
            value_jump[trace_map.value_ids[static_cast<std::size_t>(k)]];
    }
    return trace_map.harmonic_map * local_values;
}

std::vector<double> GeometricFeatureTraceFit3D::condition_values() const
{
    std::vector<double> result;
    result.reserve(static_cast<std::size_t>(impl_->summary.active_maps));
    for (const GeometricFeatureTraceMap3D& map : impl_->maps) {
        if (map.active())
            result.push_back(map.condition);
    }
    return result;
}

const GeometricFeatureTraceSummary3D&
GeometricFeatureTraceFit3D::summary() const noexcept
{
    return impl_->summary;
}

} // namespace kfbim::app3d
