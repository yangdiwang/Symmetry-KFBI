#include "native_nurbs_density_space_3d.hpp"

#include "src/geometry/nurbs_basis.hpp"

#include <Eigen/Cholesky>
#include <Eigen/QR>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d {
namespace {

using geometry::NurbsBasis1D;
using geometry3d::NurbsPatchEdge3D;
using geometry3d::NurbsPatchEdgeConnection3D;
using geometry3d::NurbsPatchEdgeInterval3D;

struct QuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

QuadratureRule gauss_legendre(int order)
{
    switch (order) {
    case 1:
        return {{0.0}, {2.0}};
    case 2: {
        const double a = 1.0 / std::sqrt(3.0);
        return {{-a, a}, {1.0, 1.0}};
    }
    case 3: {
        const double a = std::sqrt(3.0 / 5.0);
        return {{-a, 0.0, a}, {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0}};
    }
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
        throw std::invalid_argument("Gauss order must lie between 1 and 5");
    }
}

NurbsBasis1D make_cubic_open_uniform_basis(int coefficients)
{
    if (coefficients < 4)
        throw std::invalid_argument(
            "A cubic density direction needs at least four coefficients");
    const int elements = coefficients - 3;
    std::vector<double> knots;
    knots.reserve(static_cast<std::size_t>(coefficients + 4));
    for (int i = 0; i < 4; ++i)
        knots.push_back(0.0);
    for (int i = 1; i < elements; ++i)
        knots.push_back(static_cast<double>(i) / elements);
    for (int i = 0; i < 4; ++i)
        knots.push_back(1.0);
    return NurbsBasis1D(3, std::move(knots));
}

class UnionFind {
public:
    explicit UnionFind(int count)
        : parent_(static_cast<std::size_t>(count)),
          rank_(static_cast<std::size_t>(count), 0)
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

    void unite(int a, int b)
    {
        a = find(a);
        b = find(b);
        if (a == b)
            return;
        int& rank_a = rank_[static_cast<std::size_t>(a)];
        int& rank_b = rank_[static_cast<std::size_t>(b)];
        if (rank_a < rank_b)
            std::swap(a, b);
        parent_[static_cast<std::size_t>(b)] = a;
        if (rank_a == rank_b)
            ++rank_a;
    }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

bool is_u_edge(NurbsPatchEdge3D edge)
{
    return edge == NurbsPatchEdge3D::UMin
        || edge == NurbsPatchEdge3D::UMax;
}

std::string edge_name(NurbsPatchEdge3D edge)
{
    switch (edge) {
    case NurbsPatchEdge3D::UMin:
        return "u0";
    case NurbsPatchEdge3D::UMax:
        return "u1";
    case NurbsPatchEdge3D::VMin:
        return "v0";
    case NurbsPatchEdge3D::VMax:
        return "v1";
    }
    return "?";
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
    return {0.0, parameter};
}

std::pair<double, double> edge_domain(
    const geometry3d::NurbsSurfacePatch3D& patch,
    NurbsPatchEdge3D edge)
{
    if (is_u_edge(edge))
        return {patch.domain_start_v(), patch.domain_end_v()};
    return {patch.domain_start_u(), patch.domain_end_u()};
}

struct NormalizedInterval {
    double begin = 0.0;
    double end = 1.0;
};

NormalizedInterval normalized_interval(
    const NativeNurbsSurface3D& surface,
    const NurbsPatchEdgeInterval3D& interval)
{
    if (interval.patch < 0
        || interval.patch >= static_cast<int>(surface.patches.size())) {
        throw std::out_of_range("Density seam has invalid patch index");
    }
    const auto domain = edge_domain(
        surface.patches[static_cast<std::size_t>(interval.patch)],
        interval.edge);
    const double span = domain.second - domain.first;
    if (!(span > 0.0))
        throw std::runtime_error("Density seam has degenerate edge domain");
    return {(interval.begin - domain.first) / span,
            (interval.end - domain.first) / span};
}

bool is_full_interval(const NormalizedInterval& interval)
{
    constexpr double tolerance = 2.0e-12;
    return std::abs(interval.begin) <= tolerance
        && std::abs(interval.end - 1.0) <= tolerance;
}

double mapped_parameter(
    const NormalizedInterval& interval,
    double fraction,
    bool reversed)
{
    if (reversed)
        return interval.end - fraction * (interval.end - interval.begin);
    return interval.begin + fraction * (interval.end - interval.begin);
}

std::vector<int> edge_local_ids(NurbsPatchEdge3D edge, int coefficients)
{
    std::vector<int> ids(static_cast<std::size_t>(coefficients));
    for (int k = 0; k < coefficients; ++k) {
        switch (edge) {
        case NurbsPatchEdge3D::UMin:
            ids[static_cast<std::size_t>(k)] = k;
            break;
        case NurbsPatchEdge3D::UMax:
            ids[static_cast<std::size_t>(k)] =
                (coefficients - 1) * coefficients + k;
            break;
        case NurbsPatchEdge3D::VMin:
            ids[static_cast<std::size_t>(k)] = k * coefficients;
            break;
        case NurbsPatchEdge3D::VMax:
            ids[static_cast<std::size_t>(k)] =
                k * coefficients + coefficients - 1;
            break;
        }
    }
    return ids;
}

NativeDensitySeamCoupling3D seam_coupling(
    NativeDensityField3D field,
    bool geometry_g1)
{
    if (geometry_g1)
        return NativeDensitySeamCoupling3D::StrongC0WeakC1;
    return field == NativeDensityField3D::ValueTrace
        ? NativeDensitySeamCoupling3D::StrongC0
        : NativeDensitySeamCoupling3D::Broken;
}

struct NullspaceReduction {
    Eigen::MatrixXd matrix;
    int rank = 0;
};

NullspaceReduction nullspace_reduction(
    const Eigen::MatrixXd& constraints,
    double tolerance)
{
    const int columns = static_cast<int>(constraints.cols());
    if (columns <= 0)
        throw std::invalid_argument("Constraint matrix has no columns");
    if (constraints.rows() == 0) {
        return {Eigen::MatrixXd::Identity(columns, columns), 0};
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> row_qr;
    row_qr.setThreshold(tolerance);
    row_qr.compute(constraints.transpose());
    const int rank = static_cast<int>(row_qr.rank());
    if (rank == 0)
        return {Eigen::MatrixXd::Identity(columns, columns), 0};
    if (rank >= columns)
        throw std::runtime_error("Density constraints eliminate every coefficient");

    Eigen::MatrixXd independent(rank, columns);
    const auto rows = row_qr.colsPermutation().indices();
    for (int i = 0; i < rank; ++i)
        independent.row(i) = constraints.row(rows(i));

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> column_qr;
    column_qr.setThreshold(tolerance);
    column_qr.compute(independent);
    if (column_qr.rank() != rank)
        throw std::runtime_error("Independent density constraints lost rank");
    const auto permutation = column_qr.colsPermutation().indices();
    std::vector<int> pivot(static_cast<std::size_t>(rank));
    std::vector<int> free(static_cast<std::size_t>(columns - rank));
    for (int i = 0; i < rank; ++i)
        pivot[static_cast<std::size_t>(i)] = permutation(i);
    for (int i = rank; i < columns; ++i)
        free[static_cast<std::size_t>(i - rank)] = permutation(i);

    Eigen::MatrixXd pivot_matrix(rank, rank);
    Eigen::MatrixXd free_matrix(rank, columns - rank);
    for (int i = 0; i < rank; ++i)
        pivot_matrix.col(i) = independent.col(
            pivot[static_cast<std::size_t>(i)]);
    for (int i = 0; i < columns - rank; ++i)
        free_matrix.col(i) = independent.col(
            free[static_cast<std::size_t>(i)]);

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> pivot_qr;
    pivot_qr.setThreshold(tolerance);
    pivot_qr.compute(pivot_matrix);
    if (pivot_qr.rank() != rank)
        throw std::runtime_error("Density constraint pivot block is singular");
    const Eigen::MatrixXd eliminated = pivot_qr.solve(-free_matrix);

    Eigen::MatrixXd reduction = Eigen::MatrixXd::Zero(
        columns, columns - rank);
    for (int j = 0; j < columns - rank; ++j) {
        reduction(free[static_cast<std::size_t>(j)], j) = 1.0;
        for (int i = 0; i < rank; ++i) {
            reduction(pivot[static_cast<std::size_t>(i)], j) =
                eliminated(i, j);
        }
    }
    if (!reduction.allFinite())
        throw std::runtime_error("Density nullspace basis is not finite");
    return {std::move(reduction), rank};
}

double matrix_residual(
    const Eigen::MatrixXd& constraints,
    const Eigen::MatrixXd& reduction)
{
    if (constraints.rows() == 0 || reduction.cols() == 0)
        return 0.0;
    return (constraints * reduction).cwiseAbs().maxCoeff();
}

bool same_edge_interval(const NurbsPatchEdgeInterval3D& first,
                        const NurbsPatchEdgeInterval3D& second)
{
    constexpr double tolerance = 2.0e-12;
    return first.patch == second.patch && first.edge == second.edge
        && std::abs(first.begin - second.begin) <= tolerance
        && std::abs(first.end - second.end) <= tolerance;
}

void validate_matching_density_topology(
    const NativeNurbsSurface3D& first,
    const NativeNurbsSurface3D& second)
{
    if (first.patches.size() != second.patches.size()
        || first.patch_names != second.patch_names
        || first.geometric_connections.size()
               != second.geometric_connections.size()) {
        throw std::invalid_argument(
            "Feature edge jet spaces have different surface topology");
    }
    for (std::size_t i = 0; i < first.geometric_connections.size(); ++i) {
        const auto& a = first.geometric_connections[i];
        const auto& b = second.geometric_connections[i];
        if (!same_edge_interval(a.first, b.first)
            || !same_edge_interval(a.second, b.second)
            || a.reversed != b.reversed || a.g1 != b.g1) {
            throw std::invalid_argument(
                "Feature edge jet spaces have different connections");
        }
    }
}

std::vector<double> feature_common_breakpoints(
    const NativeNurbsSurface3D& surface,
    const NurbsPatchEdgeConnection3D& connection,
    const std::vector<int>& element_counts)
{
    const NormalizedInterval first =
        normalized_interval(surface, connection.first);
    const NormalizedInterval second =
        normalized_interval(surface, connection.second);
    std::vector<double> breaks{0.0, 1.0};
    auto append_interval = [&](const NormalizedInterval& interval,
                               bool reversed,
                               int elements) {
        if (elements < 1)
            throw std::invalid_argument(
                "Feature edge jet density has no knot elements");
        const double length = interval.end - interval.begin;
        for (int k = 1; k < elements; ++k) {
            const double knot = static_cast<double>(k) / elements;
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
        append_interval(first, false, elements);
        append_interval(second, connection.reversed, elements);
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

} // namespace

double NativeDensityC0Stencil3D::dot(
    const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
{
    double value = 0.0;
    for (int i = 0; i < count; ++i) {
        const int index = indices[static_cast<std::size_t>(i)];
        if (index < 0 || index >= coefficients.size())
            throw std::invalid_argument("Density stencil coefficient size mismatch");
        value += weights[static_cast<std::size_t>(i)] * coefficients[index];
    }
    return value;
}

struct NativeNurbsDensitySpace3D::Impl {
    Impl(NativeNurbsSurface3D supplied_surface,
         NativeNurbsDensityOptions3D supplied_options)
        : surface(std::move(supplied_surface)),
          options(std::move(supplied_options)),
          spline(make_cubic_open_uniform_basis(
              options.coefficients_per_direction))
    {
        validate();
        build_seam_metadata();
        merge_full_c0_seams();
        if (options.reduction_backend
            == NativeDensityReductionBackend3D::Legacy) {
            build_partial_c0_constraints();
            const NullspaceReduction c0_nullspace = nullspace_reduction(
                partial_c0_constraints, options.rank_tolerance);
            c0_reduction = c0_nullspace.matrix;
            partial_c0_rank = c0_nullspace.rank;
            build_weak_c1_constraints();
            continuous_c1_constraints = weak_c1_constraints * c0_reduction;
            const NullspaceReduction c1_nullspace = nullspace_reduction(
                continuous_c1_constraints, options.rank_tolerance);
            c1_reduction = c1_nullspace.matrix;
            weak_c1_rank = c1_nullspace.rank;
            reduction = c0_reduction * c1_reduction;
        } else {
            // The sparse topology backend owns every non-trivial reduction.
            // Retain only the exact full/full C0 merges performed above and
            // expose their coefficient space without constructing a dense
            // global constraint matrix or dense nullspace.
            partial_c0_constraints.resize(0, c0_count);
            weak_c1_constraints.resize(0, c0_count);
            continuous_c1_constraints.resize(0, c0_count);
            c0_reduction.resize(0, 0);
            c1_reduction.resize(0, 0);
            reduction.resize(0, 0);
            partial_c0_rank = 0;
            weak_c1_rank = 0;
        }
        if (options.reduction_backend
                == NativeDensityReductionBackend3D::Legacy
            && !reduction.allFinite())
            throw std::runtime_error("Native density reduction is not finite");
        build_trace_quadrature();
        build_constant();
    }

    void validate()
    {
        if (surface.patches.empty()
            || surface.patch_names.size() != surface.patches.size()) {
            throw std::invalid_argument(
                "Native density surface has inconsistent patch metadata");
        }
        if (options.mortar_gauss_order < 4
            || options.mortar_gauss_order > 5
            || options.trace_gauss_order < 1
            || options.trace_gauss_order > 5) {
            throw std::invalid_argument(
                "Cubic density mortar order must lie in [4,5] and trace "
                "Gauss order in [1,5]");
        }
        if (!(options.rank_tolerance > 0.0)
            || !(options.rank_tolerance < 1.0)
            || !std::isfinite(options.rank_tolerance)) {
            throw std::invalid_argument(
                "Density rank tolerance must be finite and in (0,1)");
        }
        (void)surface.geometry_model().validate_closed();
    }

    NativeDensitySurfaceEvaluation3D geometry(
        int patch_index, double u, double v) const
    {
        if (patch_index < 0
            || patch_index >= static_cast<int>(surface.patches.size())) {
            throw std::out_of_range("Density geometry patch is out of range");
        }
        if (!std::isfinite(u) || !std::isfinite(v)
            || u < -1.0e-12 || u > 1.0 + 1.0e-12
            || v < -1.0e-12 || v > 1.0 + 1.0e-12) {
            throw std::out_of_range(
                "Density geometry parameter lies outside [0,1]^2");
        }
        u = std::clamp(u, 0.0, 1.0);
        v = std::clamp(v, 0.0, 1.0);
        const auto& patch = surface.patches[static_cast<std::size_t>(patch_index)];
        const double scale_u = patch.domain_end_u() - patch.domain_start_u();
        const double scale_v = patch.domain_end_v() - patch.domain_start_v();
        const double native_u = patch.domain_start_u() + scale_u * u;
        const double native_v = patch.domain_start_v() + scale_v * v;
        const auto native = patch.evaluate_with_derivatives(native_u, native_v);
        NativeDensitySurfaceEvaluation3D result;
        result.point = native.point;
        result.tangents.col(0) = scale_u * native.du;
        result.tangents.col(1) = scale_v * native.dv;
        const Eigen::Vector3d cross =
            result.tangents.col(0).cross(result.tangents.col(1));
        result.area_element = cross.norm();
        if (!(result.area_element > 0.0) || !std::isfinite(result.area_element))
            throw std::runtime_error("Native density patch Jacobian is degenerate");
        result.normal = cross / result.area_element;
        return result;
    }

    void build_seam_metadata()
    {
        const auto& connections = surface.geometric_connections;
        seams.reserve(connections.size());
        for (int index = 0; index < static_cast<int>(connections.size()); ++index) {
            const auto& connection = connections[static_cast<std::size_t>(index)];
            const NormalizedInterval first =
                normalized_interval(surface, connection.first);
            const NormalizedInterval second =
                normalized_interval(surface, connection.second);
            NativeDensitySeamInfo3D info;
            info.connection = index;
            info.geometry_g1 = connection.g1;
            info.coupling = seam_coupling(options.field, connection.g1);
            info.full_edge_pair =
                is_full_interval(first) && is_full_interval(second);
            std::ostringstream label;
            label << surface.patch_names[static_cast<std::size_t>(
                         connection.first.patch)]
                  << ':' << edge_name(connection.first.edge)
                  << '[' << connection.first.begin << ',' << connection.first.end
                  << "]="
                  << surface.patch_names[static_cast<std::size_t>(
                         connection.second.patch)]
                  << ':' << edge_name(connection.second.edge)
                  << '[' << connection.second.begin << ',' << connection.second.end
                  << ']' << (connection.reversed ? " reversed" : "")
                  << (connection.g1 ? " G1" : " feature");
            info.label = label.str();
            seams.push_back(std::move(info));
        }
    }

    void merge_full_c0_seams()
    {
        const int n = options.coefficients_per_direction;
        const int per_patch = n * n;
        const int raw_count =
            static_cast<int>(surface.patches.size()) * per_patch;
        UnionFind sets(raw_count);
        for (NativeDensitySeamInfo3D& info : seams) {
            if (info.coupling == NativeDensitySeamCoupling3D::Broken
                || !info.full_edge_pair) {
                continue;
            }
            const auto& connection = surface.geometric_connections[
                static_cast<std::size_t>(info.connection)];
            const std::vector<int> first =
                edge_local_ids(connection.first.edge, n);
            const std::vector<int> second =
                edge_local_ids(connection.second.edge, n);
            for (int k = 0; k < n; ++k) {
                const int other = connection.reversed ? n - 1 - k : k;
                sets.unite(connection.first.patch * per_patch
                               + first[static_cast<std::size_t>(k)],
                           connection.second.patch * per_patch
                               + second[static_cast<std::size_t>(other)]);
            }
            info.strong_c0_merged = true;
        }

        local_to_c0_map.resize(static_cast<std::size_t>(raw_count));
        std::vector<int> root_to_index(static_cast<std::size_t>(raw_count), -1);
        int next = 0;
        for (int local = 0; local < raw_count; ++local) {
            const int root = sets.find(local);
            int& index = root_to_index[static_cast<std::size_t>(root)];
            if (index < 0)
                index = next++;
            local_to_c0_map[static_cast<std::size_t>(local)] = index;
        }
        c0_count = next;
    }

    void active_basis(double parameter,
                      int derivative,
                      std::vector<int>& indices,
                      std::vector<double>& values) const
    {
        indices = spline.active_basis_indices(parameter);
        if (derivative == 0)
            values = spline.evaluate_nonzero(parameter);
        else if (derivative == 1)
            values = spline.evaluate_nonzero_first_derivatives(parameter);
        else if (derivative == 2)
            values = spline.evaluate_nonzero_second_derivatives(parameter);
        else
            throw std::invalid_argument(
                "Only density derivatives of order zero, one and two are supported");
    }

    Eigen::RowVectorXd c0_row(int patch,
                              double u,
                              double v,
                              int derivative_u,
                              int derivative_v) const
    {
        if (derivative_u < 0 || derivative_v < 0
            || derivative_u + derivative_v > 2) {
            throw std::invalid_argument(
                "Density parameter derivatives must have total order at most two");
        }
        if (patch < 0 || patch >= static_cast<int>(surface.patches.size()))
            throw std::out_of_range("Density basis patch is out of range");
        std::vector<int> iu;
        std::vector<int> iv;
        std::vector<double> bu;
        std::vector<double> bv;
        active_basis(u, derivative_u, iu, bu);
        active_basis(v, derivative_v, iv, bv);
        Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(c0_count);
        const int n = options.coefficients_per_direction;
        const int per_patch = n * n;
        for (std::size_t a = 0; a < iu.size(); ++a) {
            for (std::size_t b = 0; b < iv.size(); ++b) {
                const int raw = patch * per_patch + iu[a] * n + iv[b];
                row[local_to_c0_map[static_cast<std::size_t>(raw)]] +=
                    bu[a] * bv[b];
            }
        }
        return row;
    }

    NativeDensityC0Stencil3D c0_stencil(
        int patch,
        double u,
        double v,
        int derivative_u = 0,
        int derivative_v = 0) const
    {
        if (derivative_u < 0 || derivative_v < 0
            || derivative_u + derivative_v > 2) {
            throw std::invalid_argument(
                "Density parameter derivatives must have total order at most two");
        }
        if (patch < 0 || patch >= static_cast<int>(surface.patches.size()))
            throw std::out_of_range("Density stencil patch is out of range");
        std::vector<int> iu;
        std::vector<int> iv;
        std::vector<double> bu;
        std::vector<double> bv;
        active_basis(u, derivative_u, iu, bu);
        active_basis(v, derivative_v, iv, bv);
        NativeDensityC0Stencil3D stencil;
        const int n = options.coefficients_per_direction;
        const int per_patch = n * n;
        for (std::size_t a = 0; a < iu.size(); ++a) {
            for (std::size_t b = 0; b < iv.size(); ++b) {
                const double weight = bu[a] * bv[b];
                if (weight == 0.0)
                    continue;
                const int raw = patch * per_patch + iu[a] * n + iv[b];
                const int index =
                    local_to_c0_map[static_cast<std::size_t>(raw)];
                int entry = 0;
                while (entry < stencil.count
                       && stencil.indices[static_cast<std::size_t>(entry)]
                              != index) {
                    ++entry;
                }
                if (entry == stencil.count) {
                    if (stencil.count
                        >= static_cast<int>(stencil.indices.size())) {
                        throw std::runtime_error(
                            "Native cubic density stencil exceeded 16 entries");
                    }
                    stencil.indices[static_cast<std::size_t>(entry)] = index;
                    ++stencil.count;
                }
                stencil.weights[static_cast<std::size_t>(entry)] += weight;
            }
        }
        return stencil;
    }

    NativeDensityC0Stencil3D physical_directional_stencil(
        int patch,
        double u,
        double v,
        const Eigen::Vector3d& tangent_direction) const
    {
        const NativeDensitySurfaceEvaluation3D sample = geometry(patch, u, v);
        const Eigen::Matrix2d metric =
            sample.tangents.transpose() * sample.tangents;
        Eigen::LDLT<Eigen::Matrix2d> factor(metric);
        if (factor.info() != Eigen::Success)
            throw std::runtime_error("Density surface metric is singular");
        const Eigen::Vector2d coordinates =
            factor.solve(sample.tangents.transpose() * tangent_direction);
        if (!coordinates.allFinite())
            throw std::runtime_error(
                "Density directional derivative coordinates are invalid");

        const NativeDensityC0Stencil3D du = c0_stencil(patch, u, v, 1, 0);
        const NativeDensityC0Stencil3D dv = c0_stencil(patch, u, v, 0, 1);
        NativeDensityC0Stencil3D result;
        auto append = [&](const NativeDensityC0Stencil3D& source,
                          double factor_value) {
            for (int q = 0; q < source.count; ++q) {
                const int index = source.indices[static_cast<std::size_t>(q)];
                int slot = 0;
                while (slot < result.count
                       && result.indices[static_cast<std::size_t>(slot)]
                              != index) {
                    ++slot;
                }
                if (slot == result.count) {
                    if (result.count
                        >= static_cast<int>(result.indices.size())) {
                        throw std::runtime_error(
                            "Native cubic derivative stencil exceeded 16 entries");
                    }
                    result.indices[static_cast<std::size_t>(slot)] = index;
                    ++result.count;
                }
                result.weights[static_cast<std::size_t>(slot)] +=
                    factor_value
                    * source.weights[static_cast<std::size_t>(q)];
            }
        };
        append(du, coordinates.x());
        append(dv, coordinates.y());
        return result;
    }

    std::vector<double> common_breakpoints(
        const NurbsPatchEdgeConnection3D& connection) const
    {
        const NormalizedInterval first =
            normalized_interval(surface, connection.first);
        const NormalizedInterval second =
            normalized_interval(surface, connection.second);
        std::vector<double> breaks{0.0, 1.0};
        const int elements = options.coefficients_per_direction - 3;
        auto append = [&](const NormalizedInterval& interval,
                          bool reversed) {
            const double length = interval.end - interval.begin;
            for (int k = 1; k < elements; ++k) {
                const double knot = static_cast<double>(k) / elements;
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
        for (double value : breaks) {
            if (unique.empty() || std::abs(value - unique.back()) > 2.0e-12)
                unique.push_back(value);
        }
        return unique;
    }

    void build_partial_c0_constraints()
    {
        const QuadratureRule collocation = gauss_legendre(4);
        std::vector<Eigen::RowVectorXd> rows;
        for (NativeDensitySeamInfo3D& info : seams) {
            info.partial_c0_first_row = static_cast<int>(rows.size());
            if (info.coupling == NativeDensitySeamCoupling3D::Broken
                || info.strong_c0_merged) {
                continue;
            }
            const auto& connection = surface.geometric_connections[
                static_cast<std::size_t>(info.connection)];
            const NormalizedInterval first =
                normalized_interval(surface, connection.first);
            const NormalizedInterval second =
                normalized_interval(surface, connection.second);
            const std::vector<double> breaks = common_breakpoints(connection);
            for (std::size_t span = 1; span < breaks.size(); ++span) {
                const double a = breaks[span - 1];
                const double b = breaks[span];
                for (double node : collocation.nodes) {
                    const double fraction =
                        0.5 * ((b - a) * node + b + a);
                    const double ta = mapped_parameter(first, fraction, false);
                    const double tb = mapped_parameter(
                        second, fraction, connection.reversed);
                    const auto [ua, va] = edge_uv(connection.first.edge, ta);
                    const auto [ub, vb] = edge_uv(connection.second.edge, tb);
                    rows.push_back(
                        c0_row(connection.first.patch, ua, va, 0, 0)
                        - c0_row(connection.second.patch, ub, vb, 0, 0));
                }
            }
            info.partial_c0_row_count =
                static_cast<int>(rows.size()) - info.partial_c0_first_row;
        }
        partial_c0_constraints.resize(
            static_cast<Eigen::Index>(rows.size()), c0_count);
        for (Eigen::Index row = 0; row < partial_c0_constraints.rows(); ++row)
            partial_c0_constraints.row(row) =
                rows[static_cast<std::size_t>(row)];
    }

    Eigen::RowVectorXd physical_directional_row(
        int patch,
        double u,
        double v,
        const Eigen::Vector3d& tangent_direction) const
    {
        const NativeDensitySurfaceEvaluation3D sample = geometry(patch, u, v);
        const Eigen::Matrix2d metric =
            sample.tangents.transpose() * sample.tangents;
        Eigen::LDLT<Eigen::Matrix2d> factor(metric);
        if (factor.info() != Eigen::Success)
            throw std::runtime_error("Density surface metric is singular");
        const Eigen::Vector2d coordinates =
            factor.solve(sample.tangents.transpose() * tangent_direction);
        if (!coordinates.allFinite())
            throw std::runtime_error(
                "Density directional derivative coordinates are invalid");
        return coordinates.x() * c0_row(patch, u, v, 1, 0)
             + coordinates.y() * c0_row(patch, u, v, 0, 1);
    }

    Eigen::RowVectorXd physical_conormal_row(
        int patch,
        NurbsPatchEdge3D edge,
        double parameter,
        const Eigen::Vector3d& conormal) const
    {
        const auto [u, v] = edge_uv(edge, parameter);
        return physical_directional_row(patch, u, v, conormal);
    }

    void build_weak_c1_constraints()
    {
        const QuadratureRule rule = gauss_legendre(options.mortar_gauss_order);
        std::vector<Eigen::RowVectorXd> rows;
        const int n = options.coefficients_per_direction;
        for (NativeDensitySeamInfo3D& info : seams) {
            info.weak_c1_first_row = static_cast<int>(rows.size());
            if (info.coupling
                != NativeDensitySeamCoupling3D::StrongC0WeakC1) {
                continue;
            }
            const auto& connection = surface.geometric_connections[
                static_cast<std::size_t>(info.connection)];
            if (!connection.g1)
                throw std::logic_error("Feature seam requested a C1 constraint");
            std::vector<Eigen::RowVectorXd> seam_rows(
                static_cast<std::size_t>(n),
                Eigen::RowVectorXd::Zero(c0_count));
            const NormalizedInterval first =
                normalized_interval(surface, connection.first);
            const NormalizedInterval second =
                normalized_interval(surface, connection.second);
            const std::vector<double> breaks = common_breakpoints(connection);
            for (std::size_t span = 1; span < breaks.size(); ++span) {
                const double a = breaks[span - 1];
                const double b = breaks[span];
                for (std::size_t q = 0; q < rule.nodes.size(); ++q) {
                    const double fraction =
                        0.5 * ((b - a) * rule.nodes[q] + b + a);
                    const double quadrature_weight =
                        0.5 * (b - a) * rule.weights[q];
                    const double ta = mapped_parameter(first, fraction, false);
                    const double tb = mapped_parameter(
                        second, fraction, connection.reversed);
                    const auto [ua, va] = edge_uv(connection.first.edge, ta);
                    const NativeDensitySurfaceEvaluation3D sample_a =
                        geometry(connection.first.patch, ua, va);
                    Eigen::Vector3d dxi = is_u_edge(connection.first.edge)
                        ? sample_a.tangents.col(1)
                        : sample_a.tangents.col(0);
                    dxi *= first.end - first.begin;
                    const double ds_dxi = dxi.norm();
                    if (!(ds_dxi > 0.0))
                        throw std::runtime_error("Density seam tangent is zero");
                    const Eigen::Vector3d tangent = dxi / ds_dxi;
                    Eigen::Vector3d conormal = sample_a.normal.cross(tangent);
                    const double conormal_norm = conormal.norm();
                    if (!(conormal_norm > 0.0))
                        throw std::runtime_error("Density seam conormal is zero");
                    conormal /= conormal_norm;
                    const Eigen::RowVectorXd jump =
                        physical_conormal_row(
                            connection.first.patch,
                            connection.first.edge,
                            ta,
                            conormal)
                        - physical_conormal_row(
                            connection.second.patch,
                            connection.second.edge,
                            tb,
                            conormal);
                    const std::vector<int> test_indices =
                        spline.active_basis_indices(fraction);
                    const std::vector<double> test_values =
                        spline.evaluate_nonzero(fraction);
                    const double factor = quadrature_weight * ds_dxi;
                    for (std::size_t local = 0;
                         local < test_indices.size(); ++local) {
                        seam_rows[static_cast<std::size_t>(
                            test_indices[local])].noalias() +=
                            factor * test_values[local] * jump;
                    }
                }
            }
            for (Eigen::RowVectorXd& row : seam_rows)
                rows.push_back(std::move(row));
            info.weak_c1_row_count = n;
        }
        weak_c1_constraints.resize(
            static_cast<Eigen::Index>(rows.size()), c0_count);
        for (Eigen::Index row = 0; row < weak_c1_constraints.rows(); ++row)
            weak_c1_constraints.row(row) = rows[static_cast<std::size_t>(row)];
    }

    void build_trace_quadrature()
    {
        const QuadratureRule rule = gauss_legendre(options.trace_gauss_order);
        const int elements = options.coefficients_per_direction - 3;
        const std::size_t expected = surface.patches.size()
            * static_cast<std::size_t>(elements * elements)
            * rule.nodes.size() * rule.nodes.size();
        trace.reserve(expected);
        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(expected * 16U);
        std::vector<double> weights;
        weights.reserve(expected);
        mass_c0 = Eigen::RowVectorXd::Zero(c0_count);

        for (int patch = 0; patch < static_cast<int>(surface.patches.size()); ++patch) {
            for (int eu = 0; eu < elements; ++eu) {
                const double ua = static_cast<double>(eu) / elements;
                const double ub = static_cast<double>(eu + 1) / elements;
                for (int ev = 0; ev < elements; ++ev) {
                    const double va = static_cast<double>(ev) / elements;
                    const double vb = static_cast<double>(ev + 1) / elements;
                    for (std::size_t gu = 0; gu < rule.nodes.size(); ++gu) {
                        const double u = 0.5
                            * ((ub - ua) * rule.nodes[gu] + ub + ua);
                        const double wu =
                            0.5 * (ub - ua) * rule.weights[gu];
                        for (std::size_t gv = 0; gv < rule.nodes.size(); ++gv) {
                            const double v = 0.5
                                * ((vb - va) * rule.nodes[gv] + vb + va);
                            const double wv =
                                0.5 * (vb - va) * rule.weights[gv];
                            const NativeDensitySurfaceEvaluation3D sample =
                                geometry(patch, u, v);
                            const double weight =
                                wu * wv * sample.area_element;
                            NativeDensityGaussPoint3D point;
                            point.patch = patch;
                            point.element_u = eu;
                            point.element_v = ev;
                            point.u = u;
                            point.v = v;
                            point.point = sample.point;
                            point.normal = sample.normal;
                            point.surface_weight = weight;
                            const int row = static_cast<int>(trace.size());
                            const NativeDensityC0Stencil3D stencil =
                                c0_stencil(patch, u, v);
                            for (int q = 0; q < stencil.count; ++q) {
                                const int index =
                                    stencil.indices[static_cast<std::size_t>(q)];
                                const double value =
                                    stencil.weights[static_cast<std::size_t>(q)];
                                entries.emplace_back(row, index, value);
                                mass_c0[index] += weight * value;
                            }
                            area += weight;
                            weights.push_back(weight);
                            trace.push_back(std::move(point));
                        }
                    }
                }
            }
        }
        trace_design.resize(static_cast<int>(trace.size()), c0_count);
        trace_design.setFromTriplets(entries.begin(), entries.end());
        trace_design.makeCompressed();
        trace_weight_vector.resize(static_cast<Eigen::Index>(weights.size()));
        for (Eigen::Index i = 0; i < trace_weight_vector.size(); ++i)
            trace_weight_vector[i] = weights[static_cast<std::size_t>(i)];
        if (options.reduction_backend
            == NativeDensityReductionBackend3D::Legacy) {
            mass = mass_c0 * reduction;
        } else {
            mass = mass_c0;
        }
        if (!(area > 0.0) || !std::isfinite(area)
            || !mass.allFinite() || !trace_weight_vector.allFinite()) {
            throw std::runtime_error("Native density quadrature is invalid");
        }
    }

    void build_constant()
    {
        const Eigen::VectorXd one = Eigen::VectorXd::Ones(c0_count);
        if (options.reduction_backend
            != NativeDensityReductionBackend3D::Legacy) {
            constant_coefficients = one;
            constant_error = 0.0;
            return;
        }
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(reduction);
        constant_coefficients = qr.solve(one);
        constant_error =
            (reduction * constant_coefficients - one)
                .lpNorm<Eigen::Infinity>();
        if (!constant_coefficients.allFinite() || constant_error > 2.0e-8)
            throw std::runtime_error("Native density space lost the constant");
    }

    Eigen::VectorXd expand_reduced(
        const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
    {
        const Eigen::Index reduced_size = options.reduction_backend
                == NativeDensityReductionBackend3D::Legacy
            ? reduction.cols()
            : c0_count;
        if (coefficients.size() != reduced_size)
            throw std::invalid_argument("Reduced native density size mismatch");
        if (options.reduction_backend
            == NativeDensityReductionBackend3D::Legacy) {
            return reduction * coefficients;
        }
        return Eigen::VectorXd(coefficients);
    }

    double evaluate_c0(int patch,
                       double u,
                       double v,
                       const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
    {
        if (coefficients.size() != c0_count)
            throw std::invalid_argument("C0 native density size mismatch");
        return c0_stencil(patch, u, v).dot(coefficients);
    }

    double max_c0_jump(
        const Eigen::Ref<const Eigen::VectorXd>& coefficients,
        int samples) const
    {
        if (samples < 2)
            throw std::invalid_argument("At least two seam samples are required");
        const Eigen::VectorXd c0 = expand_reduced(coefficients);
        double maximum = 0.0;
        for (const NativeDensitySeamInfo3D& info : seams) {
            if (info.coupling == NativeDensitySeamCoupling3D::Broken)
                continue;
            const auto& connection = surface.geometric_connections[
                static_cast<std::size_t>(info.connection)];
            const NormalizedInterval first =
                normalized_interval(surface, connection.first);
            const NormalizedInterval second =
                normalized_interval(surface, connection.second);
            for (int k = 0; k < samples; ++k) {
                const double fraction = static_cast<double>(k) / (samples - 1);
                const double ta = mapped_parameter(first, fraction, false);
                const double tb = mapped_parameter(
                    second, fraction, connection.reversed);
                const auto [ua, va] = edge_uv(connection.first.edge, ta);
                const auto [ub, vb] = edge_uv(connection.second.edge, tb);
                maximum = std::max(
                    maximum,
                    std::abs(evaluate_c0(connection.first.patch, ua, va, c0)
                             - evaluate_c0(
                                 connection.second.patch, ub, vb, c0)));
            }
        }
        return maximum;
    }

    double max_conormal_jump(
        const Eigen::Ref<const Eigen::VectorXd>& coefficients,
        int samples) const
    {
        if (samples < 1)
            throw std::invalid_argument("At least one seam sample is required");
        const Eigen::VectorXd c0 = expand_reduced(coefficients);
        double maximum = 0.0;
        for (const NativeDensitySeamInfo3D& info : seams) {
            if (info.coupling
                != NativeDensitySeamCoupling3D::StrongC0WeakC1) {
                continue;
            }
            const auto& connection = surface.geometric_connections[
                static_cast<std::size_t>(info.connection)];
            const NormalizedInterval first =
                normalized_interval(surface, connection.first);
            const NormalizedInterval second =
                normalized_interval(surface, connection.second);
            for (int k = 0; k < samples; ++k) {
                const double fraction =
                    (static_cast<double>(k) + 0.5) / samples;
                const double ta = mapped_parameter(first, fraction, false);
                const double tb = mapped_parameter(
                    second, fraction, connection.reversed);
                const auto [ua, va] = edge_uv(connection.first.edge, ta);
                const NativeDensitySurfaceEvaluation3D sample_a =
                    geometry(connection.first.patch, ua, va);
                Eigen::Vector3d tangent = is_u_edge(connection.first.edge)
                    ? sample_a.tangents.col(1)
                    : sample_a.tangents.col(0);
                tangent.normalize();
                Eigen::Vector3d conormal = sample_a.normal.cross(tangent);
                conormal.normalize();
                const double first_value = physical_conormal_row(
                    connection.first.patch,
                    connection.first.edge,
                    ta,
                    conormal).dot(c0);
                const double second_value = physical_conormal_row(
                    connection.second.patch,
                    connection.second.edge,
                    tb,
                    conormal).dot(c0);
                maximum = std::max(
                    maximum, std::abs(first_value - second_value));
            }
        }
        return maximum;
    }

    int seam_count(NativeDensitySeamCoupling3D coupling) const
    {
        return static_cast<int>(std::count_if(
            seams.begin(), seams.end(),
            [&](const NativeDensitySeamInfo3D& seam) {
                return seam.coupling == coupling;
            }));
    }

    NativeNurbsSurface3D surface;
    NativeNurbsDensityOptions3D options;
    NurbsBasis1D spline;
    std::vector<NativeDensitySeamInfo3D> seams;
    std::vector<int> local_to_c0_map;
    int c0_count = 0;
    Eigen::MatrixXd partial_c0_constraints;
    Eigen::MatrixXd weak_c1_constraints;
    Eigen::MatrixXd continuous_c1_constraints;
    Eigen::MatrixXd c0_reduction;
    Eigen::MatrixXd c1_reduction;
    Eigen::MatrixXd reduction;
    int partial_c0_rank = 0;
    int weak_c1_rank = 0;
    std::vector<NativeDensityGaussPoint3D> trace;
    Eigen::SparseMatrix<double> trace_design;
    Eigen::VectorXd trace_weight_vector;
    Eigen::RowVectorXd mass_c0;
    Eigen::RowVectorXd mass;
    double area = 0.0;
    Eigen::VectorXd constant_coefficients;
    double constant_error = 0.0;
};

NativeNurbsDensitySpace3D::NativeNurbsDensitySpace3D(
    NativeNurbsSurface3D surface,
    NativeNurbsDensityOptions3D options)
    : impl_(std::make_unique<Impl>(std::move(surface), std::move(options)))
{
}

NativeNurbsDensitySpace3D::NativeNurbsDensitySpace3D(
    NativeNurbsSurface3D surface,
    int coefficients_per_direction,
    NativeDensityField3D field)
    : NativeNurbsDensitySpace3D(
          std::move(surface),
          [&] {
              NativeNurbsDensityOptions3D options;
              options.coefficients_per_direction = coefficients_per_direction;
              options.field = field;
              return options;
          }())
{
}

NativeNurbsDensitySpace3D::~NativeNurbsDensitySpace3D() = default;
NativeNurbsDensitySpace3D::NativeNurbsDensitySpace3D(
    NativeNurbsDensitySpace3D&&) noexcept = default;
NativeNurbsDensitySpace3D& NativeNurbsDensitySpace3D::operator=(
    NativeNurbsDensitySpace3D&&) noexcept = default;

const NativeNurbsSurface3D& NativeNurbsDensitySpace3D::surface() const
{
    return impl_->surface;
}

const NativeNurbsDensityOptions3D& NativeNurbsDensitySpace3D::options() const
{
    return impl_->options;
}

int NativeNurbsDensitySpace3D::patch_count() const
{
    return static_cast<int>(impl_->surface.patches.size());
}

int NativeNurbsDensitySpace3D::coefficients_per_direction() const
{
    return impl_->options.coefficients_per_direction;
}

int NativeNurbsDensitySpace3D::elements_per_direction() const
{
    return coefficients_per_direction() - 3;
}

int NativeNurbsDensitySpace3D::raw_coefficient_count() const
{
    const int n = coefficients_per_direction();
    return patch_count() * n * n;
}

int NativeNurbsDensitySpace3D::c0_coefficient_count() const
{
    return impl_->c0_count;
}

int NativeNurbsDensitySpace3D::continuous_coefficient_count() const
{
    return uses_identity_reduction()
        ? impl_->c0_count
        : static_cast<int>(impl_->c0_reduction.cols());
}

int NativeNurbsDensitySpace3D::reduced_coefficient_count() const
{
    return uses_identity_reduction()
        ? impl_->c0_count
        : static_cast<int>(impl_->reduction.cols());
}

bool NativeNurbsDensitySpace3D::uses_identity_reduction() const noexcept
{
    return impl_->options.reduction_backend
        != NativeDensityReductionBackend3D::Legacy;
}

int NativeNurbsDensitySpace3D::partial_c0_constraint_count() const
{
    return static_cast<int>(impl_->partial_c0_constraints.rows());
}

int NativeNurbsDensitySpace3D::partial_c0_constraint_rank() const
{
    return impl_->partial_c0_rank;
}

int NativeNurbsDensitySpace3D::weak_c1_constraint_count() const
{
    return static_cast<int>(impl_->weak_c1_constraints.rows());
}

int NativeNurbsDensitySpace3D::weak_c1_constraint_rank() const
{
    return impl_->weak_c1_rank;
}

int NativeNurbsDensitySpace3D::broken_seam_count() const
{
    return impl_->seam_count(NativeDensitySeamCoupling3D::Broken);
}

int NativeNurbsDensitySpace3D::strong_c0_seam_count() const
{
    return static_cast<int>(std::count_if(
        impl_->seams.begin(), impl_->seams.end(),
        [](const NativeDensitySeamInfo3D& seam) {
            return seam.coupling != NativeDensitySeamCoupling3D::Broken;
        }));
}

int NativeNurbsDensitySpace3D::weak_c1_seam_count() const
{
    return impl_->seam_count(
        NativeDensitySeamCoupling3D::StrongC0WeakC1);
}

int NativeNurbsDensitySpace3D::merged_c0_seam_count() const
{
    return static_cast<int>(std::count_if(
        impl_->seams.begin(), impl_->seams.end(),
        [](const NativeDensitySeamInfo3D& seam) {
            return seam.strong_c0_merged;
        }));
}

int NativeNurbsDensitySpace3D::collocated_c0_seam_count() const
{
    return static_cast<int>(std::count_if(
        impl_->seams.begin(), impl_->seams.end(),
        [](const NativeDensitySeamInfo3D& seam) {
            return seam.partial_c0_row_count > 0;
        }));
}

const std::vector<NativeDensitySeamInfo3D>&
NativeNurbsDensitySpace3D::seams() const
{
    return impl_->seams;
}

NativeDensitySurfaceEvaluation3D NativeNurbsDensitySpace3D::geometry(
    int patch, double u, double v) const
{
    return impl_->geometry(patch, u, v);
}

NativeDensityC0Stencil3D NativeNurbsDensitySpace3D::c0_basis_stencil(
    int patch, double u, double v) const
{
    return impl_->c0_stencil(patch, u, v);
}

Eigen::RowVectorXd NativeNurbsDensitySpace3D::c0_basis_row(
    int patch, double u, double v) const
{
    return impl_->c0_row(patch, u, v, 0, 0);
}

NativeDensityC0Stencil3D
NativeNurbsDensitySpace3D::c0_parameter_derivative_stencil(
    int patch,
    double u,
    double v,
    int derivative_u,
    int derivative_v) const
{
    return impl_->c0_stencil(
        patch, u, v, derivative_u, derivative_v);
}

Eigen::RowVectorXd
NativeNurbsDensitySpace3D::c0_parameter_derivative_row(
    int patch,
    double u,
    double v,
    int derivative_u,
    int derivative_v) const
{
    return impl_->c0_row(
        patch, u, v, derivative_u, derivative_v);
}

Eigen::RowVectorXd NativeNurbsDensitySpace3D::reduced_basis_row(
    int patch, double u, double v) const
{
    return c0_basis_row(patch, u, v) * impl_->reduction;
}

Eigen::RowVectorXd
NativeNurbsDensitySpace3D::reduced_parameter_derivative_row(
    int patch,
    double u,
    double v,
    int derivative_u,
    int derivative_v) const
{
    const Eigen::RowVectorXd c0_row = c0_parameter_derivative_row(
        patch, u, v, derivative_u, derivative_v);
    return uses_identity_reduction() ? c0_row : c0_row * impl_->reduction;
}

Eigen::RowVectorXd
NativeNurbsDensitySpace3D::c0_physical_directional_derivative_row(
    int patch,
    double u,
    double v,
    const Eigen::Vector3d& tangent_direction) const
{
    if (!tangent_direction.allFinite()
        || !(tangent_direction.norm() > 0.0)) {
        throw std::invalid_argument(
            "Density directional derivative requires a finite nonzero vector");
    }
    const NativeDensitySurfaceEvaluation3D sample = geometry(patch, u, v);
    const double normal_component =
        std::abs(tangent_direction.dot(sample.normal));
    if (normal_component
        > 2.0e-10 * std::max(1.0, tangent_direction.norm())) {
        throw std::invalid_argument(
            "Density directional derivative vector is not surface-tangential");
    }
    return impl_->physical_directional_row(
        patch, u, v, tangent_direction);
}

NativeDensityC0Stencil3D
NativeNurbsDensitySpace3D::c0_physical_directional_derivative_stencil(
    int patch,
    double u,
    double v,
    const Eigen::Vector3d& tangent_direction) const
{
    if (!tangent_direction.allFinite()
        || !(tangent_direction.norm() > 0.0)) {
        throw std::invalid_argument(
            "Density directional derivative requires a finite nonzero vector");
    }
    const NativeDensitySurfaceEvaluation3D sample = geometry(patch, u, v);
    const double normal_component =
        std::abs(tangent_direction.dot(sample.normal));
    if (normal_component
        > 2.0e-10 * std::max(1.0, tangent_direction.norm())) {
        throw std::invalid_argument(
            "Density directional derivative vector is not surface-tangential");
    }
    return impl_->physical_directional_stencil(
        patch, u, v, tangent_direction);
}

Eigen::RowVectorXd
NativeNurbsDensitySpace3D::reduced_physical_directional_derivative_row(
    int patch,
    double u,
    double v,
    const Eigen::Vector3d& tangent_direction) const
{
    const Eigen::RowVectorXd c0_row =
        c0_physical_directional_derivative_row(
            patch, u, v, tangent_direction);
    if (uses_identity_reduction())
        return c0_row;
    return c0_row * impl_->reduction;
}

Eigen::VectorXd NativeNurbsDensitySpace3D::expand_reduced(
    const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
{
    return impl_->expand_reduced(coefficients);
}

Eigen::VectorXd NativeNurbsDensitySpace3D::expand_raw(
    const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
{
    const Eigen::VectorXd c0 = impl_->expand_reduced(coefficients);
    Eigen::VectorXd raw(raw_coefficient_count());
    for (Eigen::Index i = 0; i < raw.size(); ++i)
        raw[i] = c0[impl_->local_to_c0_map[static_cast<std::size_t>(i)]];
    return raw;
}

double NativeNurbsDensitySpace3D::evaluate_c0(
    int patch,
    double u,
    double v,
    const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
{
    return impl_->evaluate_c0(patch, u, v, coefficients);
}

double NativeNurbsDensitySpace3D::evaluate(
    int patch,
    double u,
    double v,
    const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
{
    const Eigen::VectorXd c0 = impl_->expand_reduced(coefficients);
    return impl_->evaluate_c0(patch, u, v, c0);
}

const std::vector<int>& NativeNurbsDensitySpace3D::local_to_c0() const
{
    return impl_->local_to_c0_map;
}

const Eigen::MatrixXd& NativeNurbsDensitySpace3D::c0_reduction_matrix() const
{
    if (uses_identity_reduction())
        throw std::logic_error(
            "TopologyBase uses an implicit identity C0 reduction");
    return impl_->c0_reduction;
}

const Eigen::MatrixXd& NativeNurbsDensitySpace3D::c1_reduction_matrix() const
{
    if (uses_identity_reduction())
        throw std::logic_error(
            "TopologyBase uses an implicit identity C1 reduction");
    return impl_->c1_reduction;
}

const Eigen::MatrixXd& NativeNurbsDensitySpace3D::reduction_matrix() const
{
    if (uses_identity_reduction())
        throw std::logic_error(
            "TopologyBase uses an implicit identity reduction");
    return impl_->reduction;
}

const Eigen::MatrixXd&
NativeNurbsDensitySpace3D::partial_c0_constraint_matrix() const
{
    return impl_->partial_c0_constraints;
}

const Eigen::MatrixXd& NativeNurbsDensitySpace3D::weak_c1_constraint_matrix() const
{
    return impl_->weak_c1_constraints;
}

const Eigen::MatrixXd&
NativeNurbsDensitySpace3D::continuous_weak_c1_constraint_matrix() const
{
    return impl_->continuous_c1_constraints;
}

const std::vector<NativeDensityGaussPoint3D>&
NativeNurbsDensitySpace3D::trace_points() const
{
    return impl_->trace;
}

const Eigen::SparseMatrix<double>&
NativeNurbsDensitySpace3D::trace_c0_design() const
{
    return impl_->trace_design;
}

const Eigen::VectorXd& NativeNurbsDensitySpace3D::trace_weights() const
{
    return impl_->trace_weight_vector;
}

const Eigen::RowVectorXd& NativeNurbsDensitySpace3D::c0_mass_row() const
{
    return impl_->mass_c0;
}

const Eigen::RowVectorXd& NativeNurbsDensitySpace3D::mass_row() const
{
    return impl_->mass;
}

double NativeNurbsDensitySpace3D::surface_area() const
{
    return impl_->area;
}

double NativeNurbsDensitySpace3D::partial_c0_constraint_residual() const
{
    return matrix_residual(impl_->partial_c0_constraints, impl_->reduction);
}

double NativeNurbsDensitySpace3D::weak_c1_constraint_residual() const
{
    return matrix_residual(impl_->weak_c1_constraints, impl_->reduction);
}

double NativeNurbsDensitySpace3D::reduction_constraint_residual() const
{
    return std::max(
        partial_c0_constraint_residual(), weak_c1_constraint_residual());
}

double NativeNurbsDensitySpace3D::max_c0_seam_jump(
    const Eigen::Ref<const Eigen::VectorXd>& coefficients,
    int samples_per_segment) const
{
    return impl_->max_c0_jump(coefficients, samples_per_segment);
}

double NativeNurbsDensitySpace3D::max_smooth_conormal_jump(
    const Eigen::Ref<const Eigen::VectorXd>& coefficients,
    int samples_per_segment) const
{
    return impl_->max_conormal_jump(coefficients, samples_per_segment);
}

double NativeNurbsDensitySpace3D::constant_reproduction_error() const
{
    return impl_->constant_error;
}

NativeFeatureEdgeJumpJetConstraints3D::
NativeFeatureEdgeJumpJetConstraints3D(
    const NativeNurbsDensitySpace3D& value_space,
    const NativeNurbsDensitySpace3D& normal_space,
    NativeFeatureEdgeJumpJetOptions3D options)
    : options_(std::move(options))
{
    if (value_space.options().field != NativeDensityField3D::ValueTrace
        || normal_space.options().field != NativeDensityField3D::NormalTrace) {
        throw std::invalid_argument(
            "Feature edge jump jet requires value and normal trace spaces");
    }
    if (options_.mortar_gauss_order < 4
        || options_.mortar_gauss_order > 5) {
        throw std::invalid_argument(
            "Feature edge jump-jet mortar order must lie in [4,5]");
    }
    if (!(options_.minimum_abs_dihedral_sine > 0.0)
        || !(options_.minimum_abs_dihedral_sine < 1.0)
        || !std::isfinite(options_.minimum_abs_dihedral_sine)) {
        throw std::invalid_argument(
            "Feature edge minimum dihedral sine must be finite and in (0,1)");
    }

    validate_matching_density_topology(
        value_space.surface(), normal_space.surface());
    const int value_columns = value_space.reduced_coefficient_count();
    const int normal_columns = normal_space.reduced_coefficient_count();
    value_matrix_.resize(0, value_columns);
    normal_target_matrix_.resize(0, normal_columns);
    row_scalings_.resize(0);
    if (!options_.enabled)
        return;

    const NativeNurbsSurface3D& surface = value_space.surface();
    const auto& value_seams = value_space.seams();
    const auto& normal_seams = normal_space.seams();
    const int test_count = std::max(
        value_space.coefficients_per_direction(),
        normal_space.coefficients_per_direction());
    const NurbsBasis1D test_spline =
        make_cubic_open_uniform_basis(test_count);
    const QuadratureRule rule = gauss_legendre(options_.mortar_gauss_order);
    const std::vector<int> element_counts{
        value_space.elements_per_direction(),
        normal_space.elements_per_direction(),
        test_count - 3};

    std::vector<Eigen::RowVectorXd> value_rows;
    std::vector<Eigen::RowVectorXd> normal_rows;
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
                "Feature edge jump-jet seam policies are inconsistent");
        }

        NativeFeatureEdgeJumpJetInfo3D info;
        info.connection = connection_index;
        info.first_row = static_cast<int>(value_rows.size());
        info.row_count = 2 * test_count;
        info.test_function_count = test_count;
        info.full_edge_pair = value_seam.full_edge_pair;
        info.reversed = connection.reversed;
        info.label = value_seam.label;

        std::vector<Eigen::RowVectorXd> local_value_rows(
            static_cast<std::size_t>(info.row_count),
            Eigen::RowVectorXd::Zero(value_columns));
        std::vector<Eigen::RowVectorXd> local_normal_rows(
            static_cast<std::size_t>(info.row_count),
            Eigen::RowVectorXd::Zero(normal_columns));
        const NormalizedInterval first =
            normalized_interval(surface, connection.first);
        const NormalizedInterval second =
            normalized_interval(surface, connection.second);
        const std::vector<double> breaks = feature_common_breakpoints(
            surface, connection, element_counts);
        for (std::size_t span = 1; span < breaks.size(); ++span) {
            const double a = breaks[span - 1];
            const double b = breaks[span];
            for (std::size_t q = 0; q < rule.nodes.size(); ++q) {
                const double fraction =
                    0.5 * ((b - a) * rule.nodes[q] + b + a);
                const double quadrature_weight =
                    0.5 * (b - a) * rule.weights[q];
                const double ta = mapped_parameter(first, fraction, false);
                const double tb = mapped_parameter(
                    second, fraction, connection.reversed);
                const auto [ua, va] = edge_uv(connection.first.edge, ta);
                const auto [ub, vb] = edge_uv(connection.second.edge, tb);
                const NativeDensitySurfaceEvaluation3D geometry_a =
                    value_space.geometry(connection.first.patch, ua, va);
                const NativeDensitySurfaceEvaluation3D geometry_b =
                    value_space.geometry(connection.second.patch, ub, vb);
                const double point_mismatch =
                    (geometry_a.point - geometry_b.point).norm();
                const double point_scale = std::max(
                    {1.0, geometry_a.point.norm(), geometry_b.point.norm()});
                if (!std::isfinite(point_mismatch)
                    || point_mismatch > 5.0e-9 * point_scale) {
                    throw std::runtime_error(
                        "Feature edge jump-jet geometry does not meet");
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
                        "Feature edge jump-jet tangent is degenerate");
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
                        "Feature edge jump-jet co-normal is degenerate");
                }
                conormal_a /= conormal_a_norm;
                conormal_b /= conormal_b_norm;
                const double cosine = std::clamp(
                    geometry_a.normal.dot(geometry_b.normal), -1.0, 1.0);
                const double sine = geometry_b.normal.dot(conormal_a);
                const double abs_sine = std::abs(sine);
                if (!std::isfinite(abs_sine)
                    || abs_sine < options_.minimum_abs_dihedral_sine) {
                    throw std::runtime_error(
                        "Feature edge jump-jet dihedral angle is singular");
                }
                info.minimum_abs_dihedral_sine = std::min(
                    info.minimum_abs_dihedral_sine, abs_sine);
                info.maximum_abs_dihedral_sine = std::max(
                    info.maximum_abs_dihedral_sine, abs_sine);

                const Eigen::RowVectorXd value_a =
                    value_space.reduced_physical_directional_derivative_row(
                        connection.first.patch, ua, va, conormal_a);
                const Eigen::RowVectorXd value_b =
                    value_space.reduced_physical_directional_derivative_row(
                        connection.second.patch, ub, vb, conormal_b);
                const Eigen::RowVectorXd normal_a =
                    normal_space.reduced_basis_row(
                        connection.first.patch, ua, va);
                const Eigen::RowVectorXd normal_b =
                    normal_space.reduced_basis_row(
                        connection.second.patch, ub, vb);
                const Eigen::RowVectorXd target_a =
                    (normal_b - cosine * normal_a) / sine;
                const Eigen::RowVectorXd target_b =
                    (cosine * normal_b - normal_a) / sine;

                const std::vector<int> test_indices =
                    test_spline.active_basis_indices(fraction);
                const std::vector<double> test_values =
                    test_spline.evaluate_nonzero(fraction);
                const double factor = quadrature_weight * ds_dxi;
                for (std::size_t local = 0;
                     local < test_indices.size(); ++local) {
                    const int test = test_indices[local];
                    const double moment = factor * test_values[local];
                    local_value_rows[static_cast<std::size_t>(2 * test)]
                        .noalias() += moment * value_a;
                    local_normal_rows[static_cast<std::size_t>(2 * test)]
                        .noalias() += moment * target_a;
                    local_value_rows[static_cast<std::size_t>(2 * test + 1)]
                        .noalias() += moment * value_b;
                    local_normal_rows[static_cast<std::size_t>(2 * test + 1)]
                        .noalias() += moment * target_b;
                }
            }
        }
        for (int row = 0; row < info.row_count; ++row) {
            value_rows.push_back(std::move(
                local_value_rows[static_cast<std::size_t>(row)]));
            normal_rows.push_back(std::move(
                local_normal_rows[static_cast<std::size_t>(row)]));
        }
        maximum_point_mismatch_ = std::max(
            maximum_point_mismatch_, info.maximum_point_mismatch);
        minimum_abs_dihedral_sine_ = std::min(
            minimum_abs_dihedral_sine_, info.minimum_abs_dihedral_sine);
        edges_.push_back(std::move(info));
    }

    const Eigen::Index rows = static_cast<Eigen::Index>(value_rows.size());
    value_matrix_.resize(rows, value_columns);
    normal_target_matrix_.resize(rows, normal_columns);
    row_scalings_.resize(rows);
    for (Eigen::Index row = 0; row < rows; ++row) {
        value_matrix_.row(row) = value_rows[static_cast<std::size_t>(row)];
        normal_target_matrix_.row(row) =
            normal_rows[static_cast<std::size_t>(row)];
        const double norm = std::hypot(
            value_matrix_.row(row).norm(),
            normal_target_matrix_.row(row).norm());
        if (!(norm > std::numeric_limits<double>::epsilon())
            || !std::isfinite(norm)) {
            throw std::runtime_error(
                "Feature edge jump-jet contains a zero mortar row");
        }
        const double scaling = options_.normalize_rows ? 1.0 / norm : 1.0;
        row_scalings_[row] = scaling;
        value_matrix_.row(row) *= scaling;
        normal_target_matrix_.row(row) *= scaling;
    }

    if (rows > 0) {
        const Eigen::VectorXd ones =
            Eigen::VectorXd::Ones(value_space.c0_coefficient_count());
        Eigen::VectorXd constant;
        if (value_space.uses_identity_reduction()) {
            constant = ones;
        } else {
            const Eigen::MatrixXd& reduction =
                value_space.reduction_matrix();
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(reduction);
            constant = qr.solve(ones);
        }
        constant_value_residual_ =
            (value_matrix_ * constant).lpNorm<Eigen::Infinity>();
        if (!std::isfinite(constant_value_residual_)) {
            throw std::runtime_error(
                "Feature edge jump-jet constant residual is not finite");
        }
    }
}

const NativeFeatureEdgeJumpJetOptions3D&
NativeFeatureEdgeJumpJetConstraints3D::options() const noexcept
{
    return options_;
}

int NativeFeatureEdgeJumpJetConstraints3D::feature_edge_count() const noexcept
{
    return static_cast<int>(edges_.size());
}

int NativeFeatureEdgeJumpJetConstraints3D::constraint_count() const noexcept
{
    return static_cast<int>(value_matrix_.rows());
}

const std::vector<NativeFeatureEdgeJumpJetInfo3D>&
NativeFeatureEdgeJumpJetConstraints3D::edges() const noexcept
{
    return edges_;
}

const Eigen::MatrixXd&
NativeFeatureEdgeJumpJetConstraints3D::value_matrix() const noexcept
{
    return value_matrix_;
}

const Eigen::MatrixXd&
NativeFeatureEdgeJumpJetConstraints3D::normal_target_matrix() const noexcept
{
    return normal_target_matrix_;
}

const Eigen::VectorXd&
NativeFeatureEdgeJumpJetConstraints3D::row_scalings() const noexcept
{
    return row_scalings_;
}

Eigen::VectorXd NativeFeatureEdgeJumpJetConstraints3D::normal_target(
    const Eigen::Ref<const Eigen::VectorXd>& normal_coefficients) const
{
    if (normal_coefficients.size() != normal_target_matrix_.cols()) {
        throw std::invalid_argument(
            "Feature edge jump-jet normal coefficient size mismatch");
    }
    return normal_target_matrix_ * normal_coefficients;
}

Eigen::VectorXd NativeFeatureEdgeJumpJetConstraints3D::residual(
    const Eigen::Ref<const Eigen::VectorXd>& value_coefficients,
    const Eigen::Ref<const Eigen::VectorXd>& normal_coefficients) const
{
    if (value_coefficients.size() != value_matrix_.cols()) {
        throw std::invalid_argument(
            "Feature edge jump-jet value coefficient size mismatch");
    }
    return value_matrix_ * value_coefficients
         - normal_target(normal_coefficients);
}

double NativeFeatureEdgeJumpJetConstraints3D::maximum_point_mismatch() const
    noexcept
{
    return maximum_point_mismatch_;
}

double NativeFeatureEdgeJumpJetConstraints3D::minimum_abs_dihedral_sine() const
    noexcept
{
    return minimum_abs_dihedral_sine_;
}

double NativeFeatureEdgeJumpJetConstraints3D::constant_value_residual() const
    noexcept
{
    return constant_value_residual_;
}

} // namespace kfbim::app3d
