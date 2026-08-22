#include "topology_density_constraints_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace kfbim::app3d {
namespace {

using geometry3d::NurbsPatchEdge3D;
using geometry3d::NurbsPatchEdgeConnection3D;
using geometry3d::NurbsPatchEdgeInterval3D;

struct NormalizedInterval {
    double begin = 0.0;
    double end = 1.0;
};

constexpr std::array<double, 5> kGaussNodes{{
    -0.9061798459386639928,
    -0.5384693101056830910,
     0.0,
     0.5384693101056830910,
     0.9061798459386639928}};

constexpr std::array<double, 5> kGaussWeights{{
    0.2369268850561890875,
    0.4786286704993664680,
    0.5688888888888888889,
    0.4786286704993664680,
    0.2369268850561890875}};

bool is_u_edge(NurbsPatchEdge3D edge)
{
    return edge == NurbsPatchEdge3D::UMin
        || edge == NurbsPatchEdge3D::UMax;
}

std::pair<double, double> edge_domain(
    const NativeNurbsSurface3D& surface,
    const NurbsPatchEdgeInterval3D& interval)
{
    if (interval.patch < 0
        || interval.patch >= static_cast<int>(surface.patches.size())) {
        throw std::out_of_range(
            "Topology density connection has invalid patch index");
    }
    const auto& patch = surface.patches[static_cast<std::size_t>(interval.patch)];
    return is_u_edge(interval.edge)
        ? std::pair<double, double>{patch.domain_start_v(), patch.domain_end_v()}
        : std::pair<double, double>{patch.domain_start_u(), patch.domain_end_u()};
}

NormalizedInterval normalized_interval(
    const NativeNurbsSurface3D& surface,
    const NurbsPatchEdgeInterval3D& interval)
{
    const auto domain = edge_domain(surface, interval);
    const double span = domain.second - domain.first;
    if (!(span > 0.0) || !(interval.end > interval.begin))
        throw std::runtime_error("Topology density edge interval is degenerate");
    NormalizedInterval result{
        (interval.begin - domain.first) / span,
        (interval.end - domain.first) / span};
    constexpr double tolerance = 2.0e-12;
    if (result.begin < -tolerance || result.end > 1.0 + tolerance)
        throw std::out_of_range("Topology density edge interval leaves patch");
    result.begin = std::clamp(result.begin, 0.0, 1.0);
    result.end = std::clamp(result.end, 0.0, 1.0);
    return result;
}

double mapped_parameter(const NormalizedInterval& interval,
                        double fraction,
                        bool reversed)
{
    return reversed
        ? interval.end - fraction * (interval.end - interval.begin)
        : interval.begin + fraction * (interval.end - interval.begin);
}

std::pair<double, double> edge_uv(NurbsPatchEdge3D edge, double parameter)
{
    switch (edge) {
    case NurbsPatchEdge3D::UMin:
        return {0.0, parameter};
    case NurbsPatchEdge3D::UMax:
        return {1.0, parameter};
    case NurbsPatchEdge3D::VMin:
        return {parameter, 0.0};
    case NurbsPatchEdge3D::VMax:
        return {parameter, 1.0};
    }
    throw std::logic_error("Unknown topology density patch edge");
}

std::vector<double> overlay_breaks(
    const NativeNurbsSurface3D& surface,
    const NurbsPatchEdgeConnection3D& connection,
    int elements)
{
    const NormalizedInterval first =
        normalized_interval(surface, connection.first);
    const NormalizedInterval second =
        normalized_interval(surface, connection.second);
    std::vector<double> breaks{0.0, 1.0};
    auto append = [&](const NormalizedInterval& interval, bool reversed) {
        const double length = interval.end - interval.begin;
        for (int knot_index = 1; knot_index < elements; ++knot_index) {
            const double knot = static_cast<double>(knot_index) / elements;
            if (knot <= interval.begin + 1.0e-13
                || knot >= interval.end - 1.0e-13) {
                continue;
            }
            const double fraction = reversed
                ? (interval.end - knot) / length
                : (knot - interval.begin) / length;
            if (fraction > 1.0e-13 && fraction < 1.0 - 1.0e-13)
                breaks.push_back(fraction);
        }
    };
    append(first, false);
    append(second, connection.reversed);
    std::sort(breaks.begin(), breaks.end());
    std::vector<double> unique;
    unique.reserve(breaks.size());
    for (double value : breaks) {
        if (unique.empty() || std::abs(value - unique.back()) > 2.0e-12)
            unique.push_back(value);
    }
    return unique;
}

struct FeatureQuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

FeatureQuadratureRule feature_gauss_legendre(int order)
{
    switch (order) {
    case 4:
        return {{-0.8611363115940525752,
                 -0.3399810435848562648,
                  0.3399810435848562648,
                  0.8611363115940525752},
                {0.3478548451374538574,
                 0.6521451548625461426,
                 0.6521451548625461426,
                 0.3478548451374538574}};
    case 5:
        return {{-0.9061798459386639928,
                 -0.5384693101056830910,
                  0.0,
                  0.5384693101056830910,
                  0.9061798459386639928},
                {0.2369268850561890875,
                 0.4786286704993664680,
                 0.5688888888888888889,
                 0.4786286704993664680,
                 0.2369268850561890875}};
    default:
        throw std::invalid_argument(
            "Feature jump-jet mortar order must lie in [4,5]");
    }
}

bool same_edge_interval(const NurbsPatchEdgeInterval3D& first,
                        const NurbsPatchEdgeInterval3D& second)
{
    constexpr double tolerance = 2.0e-12;
    return first.patch == second.patch && first.edge == second.edge
        && std::abs(first.begin - second.begin) <= tolerance
        && std::abs(first.end - second.end) <= tolerance;
}

void validate_matching_feature_topology(
    const NativeNurbsSurface3D& first,
    const NativeNurbsSurface3D& second)
{
    if (first.patches.size() != second.patches.size()
        || first.patch_names != second.patch_names
        || first.geometric_connections.size()
               != second.geometric_connections.size()) {
        throw std::invalid_argument(
            "Sparse feature jump-jet spaces have different topology");
    }
    for (std::size_t index = 0;
         index < first.geometric_connections.size(); ++index) {
        const NurbsPatchEdgeConnection3D& a =
            first.geometric_connections[index];
        const NurbsPatchEdgeConnection3D& b =
            second.geometric_connections[index];
        if (!same_edge_interval(a.first, b.first)
            || !same_edge_interval(a.second, b.second)
            || a.reversed != b.reversed || a.g1 != b.g1) {
            throw std::invalid_argument(
                "Sparse feature jump-jet spaces have different connections");
        }
    }
}

std::vector<double> feature_common_breaks(
    const NativeNurbsSurface3D& surface,
    const NurbsPatchEdgeConnection3D& connection,
    const std::vector<int>& element_counts)
{
    const NormalizedInterval first =
        normalized_interval(surface, connection.first);
    const NormalizedInterval second =
        normalized_interval(surface, connection.second);
    std::vector<double> breaks{0.0, 1.0};
    auto append = [&](const NormalizedInterval& interval,
                      bool reversed,
                      int elements) {
        if (elements < 1)
            throw std::invalid_argument(
                "Sparse feature jump-jet density has no knot elements");
        const double length = interval.end - interval.begin;
        for (int knot_index = 1; knot_index < elements; ++knot_index) {
            const double knot =
                static_cast<double>(knot_index) / elements;
            if (knot <= interval.begin + 1.0e-13
                || knot >= interval.end - 1.0e-13) {
                continue;
            }
            const double fraction = reversed
                ? (interval.end - knot) / length
                : (knot - interval.begin) / length;
            if (fraction > 1.0e-13 && fraction < 1.0 - 1.0e-13)
                breaks.push_back(fraction);
        }
    };
    for (int elements : element_counts) {
        append(first, false, elements);
        append(second, connection.reversed, elements);
    }
    std::sort(breaks.begin(), breaks.end());
    std::vector<double> unique;
    unique.reserve(breaks.size());
    for (double value : breaks) {
        if (unique.empty() || std::abs(value - unique.back()) > 2.0e-12)
            unique.push_back(value);
    }
    return unique;
}

double legendre_value(int degree, double x)
{
    switch (degree) {
    case 0:
        return 1.0;
    case 1:
        return x;
    case 2:
        return 0.5 * (3.0 * x * x - 1.0);
    case 3:
        return 0.5 * (5.0 * x * x * x - 3.0 * x);
    default:
        throw std::invalid_argument(
            "Topology density moments support Legendre P0 through P3");
    }
}

void accumulate_stencil(
    std::unordered_map<Eigen::Index, double>& row,
    const NativeDensityC0Stencil3D& stencil,
    double factor)
{
    for (int q = 0; q < stencil.count; ++q) {
        row[stencil.indices[static_cast<std::size_t>(q)]] +=
            factor * stencil.weights[static_cast<std::size_t>(q)];
    }
}

std::vector<std::pair<Eigen::Index, double>> sorted_entries(
    const std::unordered_map<Eigen::Index, double>& row,
    double tolerance)
{
    std::vector<std::pair<Eigen::Index, double>> entries;
    entries.reserve(row.size());
    for (const auto& [column, value] : row) {
        if (std::abs(value) > tolerance)
            entries.emplace_back(column, value);
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto& first, const auto& second) {
                  return first.first < second.first;
              });
    return entries;
}

std::array<double, 3> rounded_point(const Eigen::Vector3d& point, int digits)
{
    const double scale = std::pow(10.0, digits);
    return {{std::round(point.x() * scale) / scale,
             std::round(point.y() * scale) / scale,
             std::round(point.z() * scale) / scale}};
}

std::optional<std::array<double, 3>> cell_star(
    const NativeNurbsDensitySpace3D& space,
    const NurbsPatchEdgeConnection3D& connection,
    const NormalizedInterval& first,
    const NormalizedInterval& second,
    int cell,
    int cell_count,
    int mode,
    const TopologyDensityConstraintOptions3D& options)
{
    const bool at_start = cell < options.star_layers;
    const bool at_end = cell_count - 1 - cell < options.star_layers;
    if (!at_start && !at_end)
        return std::nullopt;
    // A one-cell coarse edge touches both vertices.  Split its independent
    // moment rows deterministically so that both physical stars are present.
    const bool use_start = at_start && (!at_end
        || mode <= options.moment_degree / 2);
    const double fraction = use_start ? 0.0 : 1.0;
    const double ta = mapped_parameter(first, fraction, false);
    const double tb = mapped_parameter(second, fraction, connection.reversed);
    const auto [ua, va] = edge_uv(connection.first.edge, ta);
    const auto [ub, vb] = edge_uv(connection.second.edge, tb);
    const Eigen::Vector3d a =
        space.geometry(connection.first.patch, ua, va).point;
    const Eigen::Vector3d b =
        space.geometry(connection.second.patch, ub, vb).point;
    const double mismatch = (a - b).norm();
    if (mismatch > 2.0e-9 * std::max(1.0, std::max(a.norm(), b.norm())))
        throw std::runtime_error(
            "Topology density connection endpoints do not coincide");
    return rounded_point(0.5 * (a + b), options.star_coordinate_digits);
}

ConstraintMeta3D make_meta(
    const NativeNurbsDensitySpace3D& space,
    const NurbsPatchEdgeConnection3D& connection,
    const NormalizedInterval& first,
    const NormalizedInterval& second,
    int connection_index,
    double cell_begin,
    double cell_end,
    int cell,
    int cell_count,
    int mode,
    std::string kind,
    const TopologyDensityConstraintOptions3D& options,
    int sheet_id = -1)
{
    ConstraintMeta3D meta;
    meta.kind = std::move(kind);
    meta.connection = connection_index;
    meta.sheet_id = sheet_id;
    meta.macro = "connection-" + std::to_string(connection_index);
    meta.segment = meta.macro;
    meta.cell = {{cell_begin, cell_end}};
    meta.mode = mode;
    meta.star = cell_star(space, connection, first, second,
                          cell, cell_count, mode, options);
    meta.side = "both";
    return meta;
}

std::string star_key(const std::array<double, 3>& point, int sheet_id)
{
    std::ostringstream out;
    out << std::setprecision(17)
        << "vertex:" << point[0] << ',' << point[1] << ',' << point[2];
    if (sheet_id >= 0)
        out << "|sheet_id=" << sheet_id;
    return out.str();
}

void validate_options(const TopologyDensityConstraintOptions3D& options)
{
    if (options.gauss_order != 5)
        throw std::invalid_argument(
            "Cubic topology moments require the five-point Gauss rule");
    if (options.moment_degree != 3)
        throw std::invalid_argument(
            "Cubic topology constraints require Legendre P0 through P3");
    if (options.star_layers < 0)
        throw std::invalid_argument("Topology star_layers cannot be negative");
    if (options.star_coordinate_digits < 0
        || options.star_coordinate_digits > 14) {
        throw std::invalid_argument(
            "Topology star coordinate digits must lie in [0,14]");
    }
    if (!(options.coefficient_drop_tolerance >= 0.0)
        || !(options.zero_row_tolerance >= 0.0)
        || !(options.rhs_tolerance >= 0.0)) {
        throw std::invalid_argument(
            "Topology constraint tolerances must be nonnegative");
    }
}

} // namespace

std::vector<int> physical_smooth_sheet_ids_3d(
    const NativeNurbsSurface3D& surface)
{
    const int patch_count = static_cast<int>(surface.patches.size());
    if (surface.patch_names.size() != surface.patches.size()) {
        throw std::invalid_argument(
            "Physical sheet construction requires one name per patch");
    }

    std::vector<int> parent(static_cast<std::size_t>(patch_count));
    std::vector<int> minimum_patch(static_cast<std::size_t>(patch_count));
    for (int patch = 0; patch < patch_count; ++patch) {
        parent[static_cast<std::size_t>(patch)] = patch;
        minimum_patch[static_cast<std::size_t>(patch)] = patch;
    }
    auto find_root = [&](int patch) {
        int root = patch;
        while (parent[static_cast<std::size_t>(root)] != root)
            root = parent[static_cast<std::size_t>(root)];
        while (parent[static_cast<std::size_t>(patch)] != patch) {
            const int next = parent[static_cast<std::size_t>(patch)];
            parent[static_cast<std::size_t>(patch)] = root;
            patch = next;
        }
        return root;
    };
    for (const NurbsPatchEdgeConnection3D& connection :
         surface.geometric_connections) {
        if (!connection.g1)
            continue;
        const int first = connection.first.patch;
        const int second = connection.second.patch;
        if (first < 0 || first >= patch_count
            || second < 0 || second >= patch_count) {
            throw std::out_of_range(
                "Physical sheet connection has invalid patch index");
        }
        int root_first = find_root(first);
        int root_second = find_root(second);
        if (root_first == root_second)
            continue;
        // The smaller patch root wins.  This is deterministic even when the
        // geometric connection array is reordered.
        if (minimum_patch[static_cast<std::size_t>(root_second)]
            < minimum_patch[static_cast<std::size_t>(root_first)]) {
            std::swap(root_first, root_second);
        }
        parent[static_cast<std::size_t>(root_second)] = root_first;
        minimum_patch[static_cast<std::size_t>(root_first)] = std::min(
            minimum_patch[static_cast<std::size_t>(root_first)],
            minimum_patch[static_cast<std::size_t>(root_second)]);
    }

    struct Component {
        int root = -1;
        std::vector<std::pair<std::string, int>> signature;
    };
    std::map<int, std::vector<std::pair<std::string, int>>> grouped;
    for (int patch = 0; patch < patch_count; ++patch) {
        grouped[find_root(patch)].emplace_back(
            surface.patch_names[static_cast<std::size_t>(patch)], patch);
    }
    std::vector<Component> components;
    components.reserve(grouped.size());
    for (auto& [root, signature] : grouped) {
        std::sort(signature.begin(), signature.end());
        components.push_back({root, std::move(signature)});
    }
    std::sort(components.begin(), components.end(),
              [](const Component& first, const Component& second) {
                  return first.signature < second.signature;
              });

    std::map<int, int> root_to_sheet;
    for (int sheet = 0; sheet < static_cast<int>(components.size()); ++sheet)
        root_to_sheet[components[static_cast<std::size_t>(sheet)].root] = sheet;
    std::vector<int> result(static_cast<std::size_t>(patch_count), -1);
    for (int patch = 0; patch < patch_count; ++patch)
        result[static_cast<std::size_t>(patch)] = root_to_sheet.at(find_root(patch));
    return result;
}

Eigen::VectorXd TopologyFeatureJumpJetOperators3D::normal_target(
    const Eigen::Ref<const Eigen::VectorXd>& normal_c0) const
{
    if (normal_c0.size() != normal_target_matrix.cols()) {
        throw std::invalid_argument(
            "Sparse feature jump-jet normal coefficient size mismatch");
    }
    return normal_target_matrix * normal_c0;
}

Eigen::VectorXd TopologyFeatureJumpJetOperators3D::residual(
    const Eigen::Ref<const Eigen::VectorXd>& value_c0,
    const Eigen::Ref<const Eigen::VectorXd>& normal_c0) const
{
    if (value_c0.size() != value_matrix.cols()) {
        throw std::invalid_argument(
            "Sparse feature jump-jet value coefficient size mismatch");
    }
    return value_matrix * value_c0 - normal_target(normal_c0);
}

ConstraintSystem3D
TopologyFeatureJumpJetOperators3D::bind_normal_target(
    const Eigen::Ref<const Eigen::VectorXd>& normal_c0) const
{
    ConstraintSystem3D result;
    result.C = value_matrix;
    result.d = normal_target(normal_c0);
    result.meta = meta;
    result.validate();
    return result;
}

TopologyFeatureJumpJetOperators3D
make_topology_feature_jump_jet_operators_3d(
    const NativeNurbsDensitySpace3D& value_base_space,
    const NativeNurbsDensitySpace3D& normal_base_space,
    NativeFeatureEdgeJumpJetOptions3D options,
    TopologyDensityConstraintOptions3D topology_options,
    double coefficient_drop_tolerance)
{
    validate_options(topology_options);
    if (value_base_space.options().field
            != NativeDensityField3D::ValueTrace
        || normal_base_space.options().field
            != NativeDensityField3D::NormalTrace) {
        throw std::invalid_argument(
            "Sparse feature jump jet requires value and normal trace spaces");
    }
    if (options.mortar_gauss_order < 4
        || options.mortar_gauss_order > 5) {
        throw std::invalid_argument(
            "Sparse feature jump-jet mortar order must lie in [4,5]");
    }
    if (!(options.minimum_abs_dihedral_sine > 0.0)
        || !(options.minimum_abs_dihedral_sine < 1.0)
        || !std::isfinite(options.minimum_abs_dihedral_sine)) {
        throw std::invalid_argument(
            "Sparse feature minimum dihedral sine must be finite and in (0,1)");
    }
    if (!(coefficient_drop_tolerance >= 0.0)
        || !std::isfinite(coefficient_drop_tolerance)) {
        throw std::invalid_argument(
            "Sparse feature coefficient drop tolerance must be finite and nonnegative");
    }
    auto require_topology_base = [](const NativeNurbsDensitySpace3D& space,
                                    const char* field) {
        if (space.options().reduction_backend
                == NativeDensityReductionBackend3D::Legacy
            || !space.uses_identity_reduction()
            || space.reduced_coefficient_count()
                   != space.c0_coefficient_count()) {
            throw std::invalid_argument(
                std::string("Sparse feature jump-jet ") + field
                + " space must use the BaseOnly topology backend");
        }
    };
    require_topology_base(value_base_space, "value");
    require_topology_base(normal_base_space, "normal");
    validate_matching_feature_topology(
        value_base_space.surface(), normal_base_space.surface());

    TopologyFeatureJumpJetOperators3D result;
    result.options = std::move(options);
    const Eigen::Index value_columns =
        value_base_space.c0_coefficient_count();
    const Eigen::Index normal_columns =
        normal_base_space.c0_coefficient_count();
    result.value_matrix.resize(0, value_columns);
    result.normal_target_matrix.resize(0, normal_columns);
    result.row_scalings.resize(0);
    if (!result.options.enabled)
        return result;

    const NativeNurbsSurface3D& surface = value_base_space.surface();
    const auto& value_seams = value_base_space.seams();
    const auto& normal_seams = normal_base_space.seams();
    const FeatureQuadratureRule rule =
        feature_gauss_legendre(result.options.mortar_gauss_order);
    const std::vector<int> element_counts{
        value_base_space.elements_per_direction(),
        normal_base_space.elements_per_direction()};

    using SparseRow = std::unordered_map<Eigen::Index, double>;
    std::vector<SparseRow> value_rows;
    std::vector<SparseRow> normal_rows;
    for (int connection_index = 0;
         connection_index
             < static_cast<int>(surface.geometric_connections.size());
         ++connection_index) {
        const NurbsPatchEdgeConnection3D& connection =
            surface.geometric_connections[
                static_cast<std::size_t>(connection_index)];
        if (connection.g1)
            continue;
        const NativeDensitySeamInfo3D& value_seam =
            value_seams[static_cast<std::size_t>(connection_index)];
        const NativeDensitySeamInfo3D& normal_seam =
            normal_seams[static_cast<std::size_t>(connection_index)];
        if (value_seam.connection != connection_index
            || normal_seam.connection != connection_index
            || value_seam.coupling == NativeDensitySeamCoupling3D::Broken
            || normal_seam.coupling != NativeDensitySeamCoupling3D::Broken) {
            throw std::logic_error(
                "Sparse feature jump-jet seam policies are inconsistent");
        }

        const NormalizedInterval first =
            normalized_interval(surface, connection.first);
        const NormalizedInterval second =
            normalized_interval(surface, connection.second);
        const std::vector<double> breaks = feature_common_breaks(
            surface, connection, element_counts);
        const int cell_count = static_cast<int>(breaks.size()) - 1;
        if (cell_count < 1)
            throw std::logic_error("Sparse feature edge has no overlay cell");

        NativeFeatureEdgeJumpJetInfo3D info;
        info.connection = connection_index;
        info.first_row = static_cast<int>(value_rows.size());
        info.test_function_count =
            cell_count * (topology_options.moment_degree + 1);
        info.row_count = 2 * info.test_function_count;
        info.full_edge_pair = value_seam.full_edge_pair;
        info.reversed = connection.reversed;
        info.label = value_seam.label;

        for (int cell = 0; cell < cell_count; ++cell) {
            const double a = breaks[static_cast<std::size_t>(cell)];
            const double b = breaks[static_cast<std::size_t>(cell + 1)];
            std::vector<SparseRow> local_value_rows(
                static_cast<std::size_t>(
                    2 * (topology_options.moment_degree + 1)));
            std::vector<SparseRow> local_normal_rows(
                static_cast<std::size_t>(
                    2 * (topology_options.moment_degree + 1)));
            for (std::size_t q = 0; q < rule.nodes.size(); ++q) {
                const double local_coordinate = rule.nodes[q];
                const double fraction =
                    0.5 * ((b - a) * local_coordinate + b + a);
                const double quadrature_weight =
                    0.5 * (b - a) * rule.weights[q];
                const double ta = mapped_parameter(first, fraction, false);
                const double tb = mapped_parameter(
                    second, fraction, connection.reversed);
                const auto [ua, va] = edge_uv(connection.first.edge, ta);
                const auto [ub, vb] = edge_uv(connection.second.edge, tb);
                const NativeDensitySurfaceEvaluation3D geometry_a =
                    value_base_space.geometry(
                        connection.first.patch, ua, va);
                const NativeDensitySurfaceEvaluation3D geometry_b =
                    value_base_space.geometry(
                        connection.second.patch, ub, vb);
                const double point_mismatch =
                    (geometry_a.point - geometry_b.point).norm();
                const double point_scale = std::max(
                    {1.0, geometry_a.point.norm(), geometry_b.point.norm()});
                if (!std::isfinite(point_mismatch)
                    || point_mismatch > 5.0e-9 * point_scale) {
                    throw std::runtime_error(
                        "Sparse feature jump-jet geometry does not meet");
                }
                info.maximum_point_mismatch = std::max(
                    info.maximum_point_mismatch, point_mismatch);

                Eigen::Vector3d dxi = is_u_edge(connection.first.edge)
                    ? geometry_a.tangents.col(1)
                    : geometry_a.tangents.col(0);
                dxi *= first.end - first.begin;
                const double ds_dxi = dxi.norm();
                if (!(ds_dxi > 0.0) || !std::isfinite(ds_dxi)) {
                    throw std::runtime_error(
                        "Sparse feature jump-jet tangent is degenerate");
                }
                const Eigen::Vector3d tangent = dxi / ds_dxi;
                Eigen::Vector3d conormal_a =
                    geometry_a.normal.cross(tangent);
                Eigen::Vector3d conormal_b =
                    geometry_b.normal.cross(tangent);
                const double conormal_a_norm = conormal_a.norm();
                const double conormal_b_norm = conormal_b.norm();
                if (!(conormal_a_norm > 0.0)
                    || !(conormal_b_norm > 0.0)) {
                    throw std::runtime_error(
                        "Sparse feature jump-jet co-normal is degenerate");
                }
                conormal_a /= conormal_a_norm;
                conormal_b /= conormal_b_norm;
                const double cosine = std::clamp(
                    geometry_a.normal.dot(geometry_b.normal), -1.0, 1.0);
                const double sine = geometry_b.normal.dot(conormal_a);
                const double abs_sine = std::abs(sine);
                if (!std::isfinite(abs_sine)
                    || abs_sine
                           < result.options.minimum_abs_dihedral_sine) {
                    throw std::runtime_error(
                        "Sparse feature jump-jet dihedral angle is singular");
                }
                info.minimum_abs_dihedral_sine = std::min(
                    info.minimum_abs_dihedral_sine, abs_sine);
                info.maximum_abs_dihedral_sine = std::max(
                    info.maximum_abs_dihedral_sine, abs_sine);

                const NativeDensityC0Stencil3D value_a =
                    value_base_space
                        .c0_physical_directional_derivative_stencil(
                            connection.first.patch, ua, va, conormal_a);
                const NativeDensityC0Stencil3D value_b =
                    value_base_space
                        .c0_physical_directional_derivative_stencil(
                            connection.second.patch, ub, vb, conormal_b);
                const NativeDensityC0Stencil3D normal_a =
                    normal_base_space.c0_basis_stencil(
                        connection.first.patch, ua, va);
                const NativeDensityC0Stencil3D normal_b =
                    normal_base_space.c0_basis_stencil(
                        connection.second.patch, ub, vb);

                const double factor = quadrature_weight * ds_dxi;
                for (int mode = 0;
                     mode <= topology_options.moment_degree; ++mode) {
                    const double moment = factor
                        * legendre_value(mode, local_coordinate);
                    SparseRow& value_first = local_value_rows[
                        static_cast<std::size_t>(2 * mode)];
                    SparseRow& normal_first = local_normal_rows[
                        static_cast<std::size_t>(2 * mode)];
                    SparseRow& value_second = local_value_rows[
                        static_cast<std::size_t>(2 * mode + 1)];
                    SparseRow& normal_second = local_normal_rows[
                        static_cast<std::size_t>(2 * mode + 1)];
                    accumulate_stencil(value_first, value_a, moment);
                    accumulate_stencil(value_second, value_b, moment);
                    accumulate_stencil(
                        normal_first, normal_b, moment / sine);
                    accumulate_stencil(
                        normal_first, normal_a,
                        -moment * cosine / sine);
                    accumulate_stencil(
                        normal_second, normal_b,
                        moment * cosine / sine);
                    accumulate_stencil(
                        normal_second, normal_a, -moment / sine);
                }
            }

            for (int mode = 0;
                 mode <= topology_options.moment_degree; ++mode) {
                for (int side = 0; side < 2; ++side) {
                    const int local_row = 2 * mode + side;
                    value_rows.push_back(std::move(local_value_rows[
                        static_cast<std::size_t>(local_row)]));
                    normal_rows.push_back(std::move(local_normal_rows[
                        static_cast<std::size_t>(local_row)]));
                    ConstraintMeta3D meta = make_meta(
                        value_base_space, connection, first, second,
                        connection_index, a, b, cell, cell_count, mode,
                        "feature_jet", topology_options);
                    meta.macro = "feature:" + info.label;
                    meta.segment = meta.macro;
                    meta.side = side == 0 ? "first" : "second";
                    result.meta.push_back(std::move(meta));
                }
            }
        }
        if (static_cast<int>(value_rows.size())
                != info.first_row + info.row_count
            || normal_rows.size() != value_rows.size()) {
            throw std::logic_error(
                "Sparse feature overlay row count is inconsistent");
        }
        result.maximum_point_mismatch = std::max(
            result.maximum_point_mismatch, info.maximum_point_mismatch);
        result.minimum_abs_dihedral_sine = std::min(
            result.minimum_abs_dihedral_sine,
            info.minimum_abs_dihedral_sine);
        result.edges.push_back(std::move(info));
    }

    const Eigen::Index row_count =
        static_cast<Eigen::Index>(value_rows.size());
    if (normal_rows.size() != value_rows.size()
        || result.meta.size() != value_rows.size()) {
        throw std::logic_error(
            "Sparse feature jump-jet row assembly lost alignment");
    }
    result.row_scalings.resize(row_count);
    std::vector<Eigen::Triplet<double>> value_triplets;
    std::vector<Eigen::Triplet<double>> normal_triplets;
    value_triplets.reserve(static_cast<std::size_t>(row_count) * 16U);
    normal_triplets.reserve(static_cast<std::size_t>(row_count) * 8U);
    for (Eigen::Index row = 0; row < row_count; ++row) {
        const SparseRow& value = value_rows[static_cast<std::size_t>(row)];
        const SparseRow& normal = normal_rows[static_cast<std::size_t>(row)];
        double squared_norm = 0.0;
        for (const auto& [column, entry] : value) {
            (void)column;
            squared_norm += entry * entry;
        }
        for (const auto& [column, entry] : normal) {
            (void)column;
            squared_norm += entry * entry;
        }
        const double norm = std::sqrt(squared_norm);
        if (!(norm > std::numeric_limits<double>::epsilon())
            || !std::isfinite(norm)) {
            throw std::runtime_error(
                "Sparse feature jump-jet contains a zero mortar row");
        }
        const double scaling = result.options.normalize_rows
            ? 1.0 / norm : 1.0;
        result.row_scalings[row] = scaling;
        for (const auto& [column, entry] : value) {
            const double scaled = scaling * entry;
            if (std::abs(scaled) > coefficient_drop_tolerance)
                value_triplets.emplace_back(row, column, scaled);
        }
        for (const auto& [column, entry] : normal) {
            const double scaled = scaling * entry;
            if (std::abs(scaled) > coefficient_drop_tolerance)
                normal_triplets.emplace_back(row, column, scaled);
        }
    }
    result.value_matrix.resize(row_count, value_columns);
    result.value_matrix.setFromTriplets(
        value_triplets.begin(), value_triplets.end());
    result.value_matrix.makeCompressed();
    result.normal_target_matrix.resize(row_count, normal_columns);
    result.normal_target_matrix.setFromTriplets(
        normal_triplets.begin(), normal_triplets.end());
    result.normal_target_matrix.makeCompressed();

    if (row_count > 0) {
        result.constant_value_residual = (
            result.value_matrix
            * Eigen::VectorXd::Ones(value_columns))
                .lpNorm<Eigen::Infinity>();
        if (!std::isfinite(result.constant_value_residual)) {
            throw std::runtime_error(
                "Sparse feature jump-jet constant residual is not finite");
        }
    }
    ConstraintSystem3D block_system;
    block_system.C = result.value_matrix;
    block_system.d = Eigen::VectorXd::Zero(row_count);
    block_system.meta = result.meta;
    result.blocks = build_topology_density_constraint_blocks_3d(
        block_system,
        topology_options.merge_blocks_by_support,
        topology_options.coefficient_drop_tolerance);
    return result;
}

TopologyDirichletFeatureConstraintPlan3D
make_topology_dirichlet_feature_constraint_plan_3d(
    const NativeNurbsDensitySpace3D& normal_base_space,
    const KnownDirichletValueGradientHessianCallback3D& known_dirichlet,
    TopologyDirichletFeatureConstraintOptions3D options,
    TopologyDensityConstraintOptions3D topology_options)
{
    validate_options(topology_options);
    if (normal_base_space.options().field
            != NativeDensityField3D::NormalTrace) {
        throw std::invalid_argument(
            "Dirichlet feature constraints require a NormalTrace space");
    }
    if (normal_base_space.options().reduction_backend
            == NativeDensityReductionBackend3D::Legacy
        || !normal_base_space.uses_identity_reduction()
        || normal_base_space.reduced_coefficient_count()
               != normal_base_space.c0_coefficient_count()) {
        throw std::invalid_argument(
            "Dirichlet feature constraints require the BaseOnly topology "
            "backend");
    }
    if (options.mortar_gauss_order < 4
        || options.mortar_gauss_order > 5) {
        throw std::invalid_argument(
            "Dirichlet feature mortar order must lie in [4,5]");
    }
    if (!(options.minimum_abs_dihedral_sine > 0.0)
        || !(options.minimum_abs_dihedral_sine < 1.0)
        || !std::isfinite(options.minimum_abs_dihedral_sine)) {
        throw std::invalid_argument(
            "Dirichlet feature minimum dihedral sine must be finite and in "
            "(0,1)");
    }
    if (!(options.tangential_compatibility_tolerance >= 0.0)
        || !std::isfinite(options.tangential_compatibility_tolerance)) {
        throw std::invalid_argument(
            "Dirichlet feature tangential compatibility tolerance must be "
            "finite and nonnegative");
    }

    TopologyDirichletFeatureConstraintPlan3D result;
    result.options = std::move(options);
    const Eigen::Index columns =
        normal_base_space.c0_coefficient_count();
    SparseConstraintBuilder3D empty_builder(columns);
    result.system = empty_builder.finalize(
        topology_options.zero_row_tolerance,
        topology_options.rhs_tolerance);
    result.row_scalings.resize(0);
    if (!result.options.enabled)
        return result;
    if (!known_dirichlet) {
        throw std::invalid_argument(
            "Enabled Dirichlet feature constraints require an analytic g_D "
            "callback");
    }

    const NativeNurbsSurface3D& surface = normal_base_space.surface();
    const auto& seams = normal_base_space.seams();
    const FeatureQuadratureRule rule =
        feature_gauss_legendre(result.options.mortar_gauss_order);
    const std::vector<int> element_counts{
        normal_base_space.elements_per_direction()};
    using SparseRow = std::unordered_map<Eigen::Index, double>;
    std::vector<SparseRow> rows;
    std::vector<double> right_hand_sides;
    std::vector<ConstraintMeta3D> metadata;

    for (int connection_index = 0;
         connection_index
             < static_cast<int>(surface.geometric_connections.size());
         ++connection_index) {
        const NurbsPatchEdgeConnection3D& connection =
            surface.geometric_connections[
                static_cast<std::size_t>(connection_index)];
        if (connection.g1)
            continue;
        const NativeDensitySeamInfo3D& seam =
            seams[static_cast<std::size_t>(connection_index)];
        if (seam.connection != connection_index
            || seam.coupling != NativeDensitySeamCoupling3D::Broken) {
            throw std::logic_error(
                "Dirichlet feature constraints require broken NormalTrace "
                "feature seams");
        }

        const NormalizedInterval first =
            normalized_interval(surface, connection.first);
        const NormalizedInterval second =
            normalized_interval(surface, connection.second);
        const std::vector<double> breaks = feature_common_breaks(
            surface, connection, element_counts);
        const int cell_count = static_cast<int>(breaks.size()) - 1;
        if (cell_count < 1)
            throw std::logic_error("Dirichlet feature edge has no overlay cell");

        NativeFeatureEdgeJumpJetInfo3D info;
        info.connection = connection_index;
        info.first_row = static_cast<int>(rows.size());
        info.test_function_count =
            cell_count * (topology_options.moment_degree + 1);
        info.row_count = 2 * info.test_function_count;
        info.full_edge_pair = seam.full_edge_pair;
        info.reversed = connection.reversed;
        info.label = seam.label;

        for (int cell = 0; cell < cell_count; ++cell) {
            const double a = breaks[static_cast<std::size_t>(cell)];
            const double b = breaks[static_cast<std::size_t>(cell + 1)];
            std::vector<SparseRow> local_rows(
                static_cast<std::size_t>(
                    2 * (topology_options.moment_degree + 1)));
            std::vector<double> local_rhs(
                static_cast<std::size_t>(
                    2 * (topology_options.moment_degree + 1)),
                0.0);

            for (std::size_t q = 0; q < rule.nodes.size(); ++q) {
                const double local_coordinate = rule.nodes[q];
                const double fraction =
                    0.5 * ((b - a) * local_coordinate + b + a);
                const double quadrature_weight =
                    0.5 * (b - a) * rule.weights[q];
                const double ta = mapped_parameter(first, fraction, false);
                const double tb = mapped_parameter(
                    second, fraction, connection.reversed);
                const auto [ua, va] = edge_uv(connection.first.edge, ta);
                const auto [ub, vb] = edge_uv(connection.second.edge, tb);
                const NativeDensitySurfaceEvaluation3D geometry_a =
                    normal_base_space.geometry(
                        connection.first.patch, ua, va);
                const NativeDensitySurfaceEvaluation3D geometry_b =
                    normal_base_space.geometry(
                        connection.second.patch, ub, vb);

                const double point_mismatch =
                    (geometry_a.point - geometry_b.point).norm();
                const double point_scale = std::max(
                    {1.0, geometry_a.point.norm(), geometry_b.point.norm()});
                if (!std::isfinite(point_mismatch)
                    || point_mismatch > 5.0e-9 * point_scale) {
                    throw std::runtime_error(
                        "Dirichlet feature constraint geometry does not meet");
                }
                info.maximum_point_mismatch = std::max(
                    info.maximum_point_mismatch, point_mismatch);

                Eigen::Vector3d dxi = is_u_edge(connection.first.edge)
                    ? geometry_a.tangents.col(1)
                    : geometry_a.tangents.col(0);
                dxi *= first.end - first.begin;
                const double ds_dxi = dxi.norm();
                if (!(ds_dxi > 0.0) || !std::isfinite(ds_dxi)) {
                    throw std::runtime_error(
                        "Dirichlet feature constraint tangent is degenerate");
                }
                const Eigen::Vector3d tangent = dxi / ds_dxi;
                Eigen::Vector3d conormal_a =
                    geometry_a.normal.cross(tangent);
                Eigen::Vector3d conormal_b =
                    geometry_b.normal.cross(tangent);
                const double conormal_a_norm = conormal_a.norm();
                const double conormal_b_norm = conormal_b.norm();
                if (!(conormal_a_norm > 0.0)
                    || !(conormal_b_norm > 0.0)) {
                    throw std::runtime_error(
                        "Dirichlet feature constraint co-normal is degenerate");
                }
                conormal_a /= conormal_a_norm;
                conormal_b /= conormal_b_norm;
                const double cosine = std::clamp(
                    geometry_a.normal.dot(geometry_b.normal), -1.0, 1.0);
                const double sine = geometry_b.normal.dot(conormal_a);
                const double abs_sine = std::abs(sine);
                if (!std::isfinite(abs_sine)
                    || abs_sine
                           < result.options.minimum_abs_dihedral_sine) {
                    throw std::runtime_error(
                        "Dirichlet feature constraint dihedral angle is "
                        "singular");
                }
                info.minimum_abs_dihedral_sine = std::min(
                    info.minimum_abs_dihedral_sine, abs_sine);
                info.maximum_abs_dihedral_sine = std::max(
                    info.maximum_abs_dihedral_sine, abs_sine);

                const KnownDirichletValueGradientHessian3D known_a =
                    known_dirichlet(
                        connection.first.patch, ua, va,
                        geometry_a.point, geometry_a.normal);
                const KnownDirichletValueGradientHessian3D known_b =
                    known_dirichlet(
                        connection.second.patch, ub, vb,
                        geometry_b.point, geometry_b.normal);
                if (!std::isfinite(known_a.value)
                    || !known_a.ambient_gradient.allFinite()
                    || !known_a.ambient_hessian.allFinite()
                    || !std::isfinite(known_b.value)
                    || !known_b.ambient_gradient.allFinite()
                    || !known_b.ambient_hessian.allFinite()) {
                    throw std::runtime_error(
                        "Dirichlet feature callback returned non-finite data");
                }
                const double tangent_derivative_mismatch = std::abs(
                    known_a.ambient_gradient.dot(tangent)
                    - known_b.ambient_gradient.dot(tangent));
                result.maximum_tangential_derivative_mismatch = std::max(
                    result.maximum_tangential_derivative_mismatch,
                    tangent_derivative_mismatch);
                const double gradient_scale = std::max(
                    {1.0, known_a.ambient_gradient.norm(),
                     known_b.ambient_gradient.norm()});
                if (tangent_derivative_mismatch
                    > result.options.tangential_compatibility_tolerance
                        * gradient_scale) {
                    throw std::runtime_error(
                        "Known Dirichlet data have incompatible edge-tangent "
                        "derivatives");
                }

                const double derivative_a =
                    known_a.ambient_gradient.dot(conormal_a);
                const double derivative_b =
                    known_b.ambient_gradient.dot(conormal_b);
                const NativeDensityC0Stencil3D normal_a =
                    normal_base_space.c0_basis_stencil(
                        connection.first.patch, ua, va);
                const NativeDensityC0Stencil3D normal_b =
                    normal_base_space.c0_basis_stencil(
                        connection.second.patch, ub, vb);
                const double factor = quadrature_weight * ds_dxi;
                for (int mode = 0;
                     mode <= topology_options.moment_degree; ++mode) {
                    const double moment = factor
                        * legendre_value(mode, local_coordinate);
                    SparseRow& first_row = local_rows[
                        static_cast<std::size_t>(2 * mode)];
                    SparseRow& second_row = local_rows[
                        static_cast<std::size_t>(2 * mode + 1)];
                    // These are exactly the normal_target rows of the dual
                    // Neumann feature operator, now used as the unknown side.
                    accumulate_stencil(
                        first_row, normal_b, moment / sine);
                    accumulate_stencil(
                        first_row, normal_a, -moment * cosine / sine);
                    accumulate_stencil(
                        second_row, normal_b, moment * cosine / sine);
                    accumulate_stencil(
                        second_row, normal_a, -moment / sine);
                    local_rhs[static_cast<std::size_t>(2 * mode)] +=
                        moment * derivative_a;
                    local_rhs[static_cast<std::size_t>(2 * mode + 1)] +=
                        moment * derivative_b;
                }
            }

            for (int mode = 0;
                 mode <= topology_options.moment_degree; ++mode) {
                for (int side = 0; side < 2; ++side) {
                    const int local_row = 2 * mode + side;
                    rows.push_back(std::move(local_rows[
                        static_cast<std::size_t>(local_row)]));
                    right_hand_sides.push_back(
                        local_rhs[static_cast<std::size_t>(local_row)]);
                    ConstraintMeta3D meta = make_meta(
                        normal_base_space, connection, first, second,
                        connection_index, a, b, cell, cell_count, mode,
                        "dirichlet_feature_jet", topology_options);
                    meta.macro = "feature:" + info.label;
                    meta.segment = meta.macro;
                    meta.side = side == 0 ? "first" : "second";
                    metadata.push_back(std::move(meta));
                }
            }
        }

        if (static_cast<int>(rows.size())
                != info.first_row + info.row_count
            || right_hand_sides.size() != rows.size()
            || metadata.size() != rows.size()) {
            throw std::logic_error(
                "Dirichlet feature constraint row assembly lost alignment");
        }
        result.maximum_point_mismatch = std::max(
            result.maximum_point_mismatch, info.maximum_point_mismatch);
        result.minimum_abs_dihedral_sine = std::min(
            result.minimum_abs_dihedral_sine,
            info.minimum_abs_dihedral_sine);
        result.edges.push_back(std::move(info));
    }

    SparseConstraintBuilder3D builder(columns);
    result.row_scalings.resize(static_cast<Eigen::Index>(rows.size()));
    for (std::size_t row = 0; row < rows.size(); ++row) {
        std::vector<std::pair<Eigen::Index, double>> entries =
            sorted_entries(rows[row],
                           topology_options.coefficient_drop_tolerance);
        double squared_norm = 0.0;
        for (const auto& [column, entry] : entries) {
            (void)column;
            squared_norm += entry * entry;
        }
        const double norm = std::sqrt(squared_norm);
        if (!(norm > std::numeric_limits<double>::epsilon())
            || !std::isfinite(norm)) {
            throw std::runtime_error(
                "Dirichlet feature constraint contains a zero mortar row");
        }
        const double scaling = result.options.normalize_rows
            ? 1.0 / norm : 1.0;
        result.row_scalings[static_cast<Eigen::Index>(row)] = scaling;
        for (auto& [column, entry] : entries) {
            (void)column;
            entry *= scaling;
        }
        builder.append_row(
            entries, scaling * right_hand_sides[row],
            std::move(metadata[row]));
    }
    result.system = builder.finalize(
        topology_options.zero_row_tolerance,
        topology_options.rhs_tolerance);
    if (result.system.C.rows()
        != static_cast<Eigen::Index>(rows.size())) {
        throw std::logic_error(
            "Dirichlet feature constraint normalization lost a row");
    }
    result.blocks = build_topology_density_constraint_blocks_3d(
        result.system,
        topology_options.merge_blocks_by_support,
        topology_options.coefficient_drop_tolerance);
    return result;
}

ConstraintSystem3D assemble_topology_density_constraints_3d(
    const NativeNurbsDensitySpace3D& base_space,
    TopologyDensityConstraintOptions3D options)
{
    validate_options(options);
    if (base_space.options().reduction_backend
        == NativeDensityReductionBackend3D::Legacy) {
        throw std::invalid_argument(
            "Sparse topology constraints require a BaseOnly density space");
    }
    if (base_space.reduced_coefficient_count()
            != base_space.c0_coefficient_count()
        || !base_space.uses_identity_reduction()) {
        throw std::logic_error(
            "BaseOnly density space unexpectedly contains a reduction");
    }

    SparseConstraintBuilder3D builder(base_space.c0_coefficient_count());
    const auto& surface = base_space.surface();
    const std::vector<int> sheet_ids =
        physical_smooth_sheet_ids_3d(surface);
    const int elements = base_space.elements_per_direction();
    for (const NativeDensitySeamInfo3D& seam : base_space.seams()) {
        const auto& connection = surface.geometric_connections[
            static_cast<std::size_t>(seam.connection)];
        const bool assemble_c0 =
            seam.coupling != NativeDensitySeamCoupling3D::Broken
            && !seam.strong_c0_merged;
        const bool assemble_c1 = seam.coupling
            == NativeDensitySeamCoupling3D::StrongC0WeakC1;
        if (!assemble_c0 && !assemble_c1)
            continue;
        int physical_sheet_id = -1;
        if (assemble_c1) {
            physical_sheet_id = sheet_ids[static_cast<std::size_t>(
                connection.first.patch)];
            if (physical_sheet_id < 0
                || physical_sheet_id
                       != sheet_ids[static_cast<std::size_t>(
                           connection.second.patch)]) {
                throw std::logic_error(
                    "G1 connection crosses physical smooth-sheet ids");
            }
        }
        // Neumann ValueTrace feature jets deliberately join all incident
        // sheets in one physical star.  Dirichlet NormalTrace has no feature
        // jet coupling and must keep a separate star for every G1 sheet.
        const int vertex_sheet_id = base_space.options().field
                == NativeDensityField3D::NormalTrace
            ? physical_sheet_id : -1;

        const NormalizedInterval first =
            normalized_interval(surface, connection.first);
        const NormalizedInterval second =
            normalized_interval(surface, connection.second);
        const std::vector<double> breaks =
            overlay_breaks(surface, connection, elements);
        const int cell_count = static_cast<int>(breaks.size()) - 1;
        for (int cell = 0; cell < cell_count; ++cell) {
            const double a = breaks[static_cast<std::size_t>(cell)];
            const double b = breaks[static_cast<std::size_t>(cell + 1)];
            for (int mode = 0; mode <= options.moment_degree; ++mode) {
                std::unordered_map<Eigen::Index, double> c0_row;
                std::unordered_map<Eigen::Index, double> c1_row;
                for (std::size_t q = 0; q < kGaussNodes.size(); ++q) {
                    const double local = kGaussNodes[q];
                    const double fraction = 0.5 * ((b - a) * local + b + a);
                    const double ta = mapped_parameter(first, fraction, false);
                    const double tb = mapped_parameter(
                        second, fraction, connection.reversed);
                    const auto [ua, va] = edge_uv(connection.first.edge, ta);
                    const auto [ub, vb] = edge_uv(connection.second.edge, tb);
                    const NativeDensitySurfaceEvaluation3D geometry_a =
                        base_space.geometry(connection.first.patch, ua, va);
                    const NativeDensitySurfaceEvaluation3D geometry_b =
                        base_space.geometry(connection.second.patch, ub, vb);
                    const double mismatch =
                        (geometry_a.point - geometry_b.point).norm();
                    if (mismatch > 2.0e-9 * std::max(
                            1.0, std::max(geometry_a.point.norm(),
                                          geometry_b.point.norm()))) {
                        throw std::runtime_error(
                            "Topology density connection points do not coincide");
                    }

                    Eigen::Vector3d dx_ds =
                        is_u_edge(connection.first.edge)
                        ? geometry_a.tangents.col(1)
                        : geometry_a.tangents.col(0);
                    dx_ds *= first.end - first.begin;
                    const double line_jacobian = dx_ds.norm();
                    if (!(line_jacobian > 0.0))
                        throw std::runtime_error(
                            "Topology density connection tangent is zero");
                    const double factor = 0.5 * (b - a) * kGaussWeights[q]
                        * line_jacobian * legendre_value(mode, local);

                    if (assemble_c0) {
                        accumulate_stencil(
                            c0_row,
                            base_space.c0_basis_stencil(
                                connection.first.patch, ua, va),
                            factor);
                        accumulate_stencil(
                            c0_row,
                            base_space.c0_basis_stencil(
                                connection.second.patch, ub, vb),
                            -factor);
                    }
                    if (assemble_c1) {
                        const Eigen::Vector3d tangent = dx_ds / line_jacobian;
                        Eigen::Vector3d conormal =
                            geometry_a.normal.cross(tangent);
                        const double conormal_norm = conormal.norm();
                        if (!(conormal_norm > 0.0))
                            throw std::runtime_error(
                                "Topology density common conormal is zero");
                        conormal /= conormal_norm;
                        accumulate_stencil(
                            c1_row,
                            base_space
                                .c0_physical_directional_derivative_stencil(
                                    connection.first.patch, ua, va, conormal),
                            factor);
                        accumulate_stencil(
                            c1_row,
                            base_space
                                .c0_physical_directional_derivative_stencil(
                                    connection.second.patch, ub, vb, conormal),
                            -factor);
                    }
                }

                if (assemble_c0) {
                    builder.append_row(
                        sorted_entries(c0_row,
                                       options.coefficient_drop_tolerance),
                        0.0,
                        make_meta(base_space, connection, first, second,
                                  seam.connection, a, b, cell, cell_count,
                                  mode, "c0", options,
                                  vertex_sheet_id));
                }
                if (assemble_c1) {
                    builder.append_row(
                        sorted_entries(c1_row,
                                       options.coefficient_drop_tolerance),
                        0.0,
                        make_meta(base_space, connection, first, second,
                                  seam.connection, a, b, cell, cell_count,
                                  mode, "smooth_c1", options,
                                  vertex_sheet_id));
                }
            }
        }
    }
    return builder.finalize(
        options.zero_row_tolerance, options.rhs_tolerance);
}

std::vector<ConstraintBlock3D>
build_topology_density_constraint_blocks_3d(
    const ConstraintSystem3D& system,
    bool merge_by_support,
    double support_tolerance)
{
    system.validate();
    std::map<std::string, std::vector<Eigen::Index>> vertex_rows;
    std::map<std::string, std::vector<Eigen::Index>> edge_rows;
    for (Eigen::Index row = 0; row < system.C.rows(); ++row) {
        const ConstraintMeta3D& meta =
            system.meta[static_cast<std::size_t>(row)];
        if (meta.star) {
            vertex_rows[star_key(*meta.star, meta.sheet_id)].push_back(row);
        } else {
            const std::string key = "edge:"
                + (meta.macro.empty() ? meta.segment : meta.macro);
            edge_rows[key].push_back(row);
        }
    }

    std::vector<ConstraintBlock3D> blocks;
    blocks.reserve(vertex_rows.size() + edge_rows.size());
    auto append = [&](const auto& source, ConstraintBlockKind3D kind) {
        for (const auto& [key, rows] : source) {
            ConstraintBlock3D block;
            block.kind = kind;
            block.key = key;
            block.row_ids = rows;
            block.source_keys = {key};
            blocks.push_back(std::move(block));
        }
    };
    append(vertex_rows, ConstraintBlockKind3D::Vertex);
    append(edge_rows, ConstraintBlockKind3D::Edge);
    blocks = order_constraint_blocks_vertex_before_edge_3d(std::move(blocks));
    if (merge_by_support) {
        blocks = merge_constraint_blocks_by_support_3d(
            system.C, std::move(blocks), support_tolerance);
        blocks = order_constraint_blocks_vertex_before_edge_3d(
            std::move(blocks));
    }
    return blocks;
}

TopologyDensityConstraintPlan3D
make_topology_density_constraint_plan_3d(
    const NativeNurbsDensitySpace3D& base_space,
    TopologyDensityConstraintOptions3D options)
{
    TopologyDensityConstraintPlan3D result;
    result.system =
        assemble_topology_density_constraints_3d(base_space, options);
    result.blocks = build_topology_density_constraint_blocks_3d(
        result.system,
        options.merge_blocks_by_support,
        options.coefficient_drop_tolerance);
    for (const ConstraintMeta3D& meta : result.system.meta) {
        if (meta.kind == "c0")
            ++result.c0_row_count;
        else if (meta.kind == "smooth_c1")
            ++result.smooth_c1_row_count;
    }
    for (const ConstraintBlock3D& block : result.blocks) {
        if (block.kind == ConstraintBlockKind3D::Vertex)
            ++result.vertex_block_count;
        else
            ++result.edge_block_count;
    }
    result.connection_count = static_cast<int>(std::count_if(
        base_space.seams().begin(), base_space.seams().end(),
        [](const NativeDensitySeamInfo3D& seam) {
            return (!seam.strong_c0_merged
                    && seam.coupling != NativeDensitySeamCoupling3D::Broken)
                || seam.coupling
                       == NativeDensitySeamCoupling3D::StrongC0WeakC1;
        }));
    std::map<std::pair<std::string, std::array<double, 2>>, bool> cells;
    for (const ConstraintMeta3D& meta : result.system.meta)
        cells[{meta.segment, meta.cell}] = true;
    result.overlay_cell_count = static_cast<int>(cells.size());
    return result;
}

} // namespace kfbim::app3d
