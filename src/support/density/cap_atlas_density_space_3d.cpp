#include "src/support/density/cap_atlas_density_space_3d.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
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

struct UnionFind {
    explicit UnionFind(int count)
        : parent(static_cast<std::size_t>(count)),
          rank(static_cast<std::size_t>(count), 0)
    {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int value)
    {
        int& p = parent[static_cast<std::size_t>(value)];
        if (p != value)
            p = find(p);
        return p;
    }

    void unite(int a, int b)
    {
        a = find(a);
        b = find(b);
        if (a == b)
            return;
        int& ra = rank[static_cast<std::size_t>(a)];
        int& rb = rank[static_cast<std::size_t>(b)];
        if (ra < rb)
            std::swap(a, b);
        parent[static_cast<std::size_t>(b)] = a;
        if (ra == rb)
            ++ra;
    }

    std::vector<int> parent;
    std::vector<int> rank;
};

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

std::string side_name(CapSide3D side)
{
    switch (side) {
    case CapSide3D::North:
        return "N";
    case CapSide3D::South:
        return "S";
    case CapSide3D::East:
        return "E";
    case CapSide3D::West:
        return "W";
    case CapSide3D::None:
        return "";
    }
    return "";
}

std::string edge_name(CapPatchEdge3D edge)
{
    switch (edge) {
    case CapPatchEdge3D::U0:
        return "u0";
    case CapPatchEdge3D::U1:
        return "u1";
    case CapPatchEdge3D::V0:
        return "v0";
    case CapPatchEdge3D::V1:
        return "v1";
    }
    return "?";
}

std::pair<double, double> edge_uv(CapPatchEdge3D edge, double t)
{
    switch (edge) {
    case CapPatchEdge3D::U0:
        return {0.0, t};
    case CapPatchEdge3D::U1:
        return {1.0, t};
    case CapPatchEdge3D::V0:
        return {t, 0.0};
    case CapPatchEdge3D::V1:
        return {t, 1.0};
    }
    return {0.0, t};
}

bool edge_parameter_is_v(CapPatchEdge3D edge)
{
    return edge == CapPatchEdge3D::U0 || edge == CapPatchEdge3D::U1;
}

class CubicOpenUniformBasis {
public:
    explicit CubicOpenUniformBasis(int coefficients)
        : coefficients_(coefficients), elements_(coefficients - 3)
    {
        if (coefficients_ < 4)
            throw std::invalid_argument(
                "A cubic open-uniform basis needs at least four coefficients");
        knots_.reserve(static_cast<std::size_t>(coefficients_ + 4));
        for (int i = 0; i < 4; ++i)
            knots_.push_back(0.0);
        for (int i = 1; i < elements_; ++i)
            knots_.push_back(static_cast<double>(i) / elements_);
        for (int i = 0; i < 4; ++i)
            knots_.push_back(1.0);
    }

    int coefficients() const { return coefficients_; }
    int elements() const { return elements_; }

    Eigen::VectorXd values(double parameter, int derivative) const
    {
        if (derivative < 0 || derivative > 2)
            throw std::invalid_argument(
                "Only spline derivatives of order zero, one, and two are supported");
        if (!std::isfinite(parameter) || parameter < -1.0e-12
            || parameter > 1.0 + 1.0e-12)
            throw std::out_of_range("Spline parameter lies outside [0,1]");
        const double x = std::clamp(parameter, 0.0, 1.0);
        const int degree = 3;
        const int initial_count = static_cast<int>(knots_.size()) - 1;
        std::array<Eigen::VectorXd, degree + 1> value;
        std::array<Eigen::VectorXd, degree + 1> first;
        std::array<Eigen::VectorXd, degree + 1> second;
        value[0] = Eigen::VectorXd::Zero(initial_count);
        first[0] = Eigen::VectorXd::Zero(initial_count);
        second[0] = Eigen::VectorXd::Zero(initial_count);

        if (x == 1.0) {
            // The final non-empty degree-zero knot interval is n-1.
            value[0](coefficients_ - 1) = 1.0;
        } else {
            for (int i = 0; i < initial_count; ++i) {
                if (knots_[static_cast<std::size_t>(i)] <= x
                    && x < knots_[static_cast<std::size_t>(i + 1)]) {
                    value[0](i) = 1.0;
                    break;
                }
            }
        }

        for (int p = 1; p <= degree; ++p) {
            const int count = initial_count - p;
            value[p] = Eigen::VectorXd::Zero(count);
            first[p] = Eigen::VectorXd::Zero(count);
            second[p] = Eigen::VectorXd::Zero(count);
            for (int i = 0; i < count; ++i) {
                const double left_den =
                    knots_[static_cast<std::size_t>(i + p)]
                    - knots_[static_cast<std::size_t>(i)];
                const double right_den =
                    knots_[static_cast<std::size_t>(i + p + 1)]
                    - knots_[static_cast<std::size_t>(i + 1)];
                if (left_den > 0.0) {
                    value[p](i) += (x - knots_[static_cast<std::size_t>(i)])
                                   / left_den * value[p - 1](i);
                    first[p](i) += static_cast<double>(p) / left_den
                                   * value[p - 1](i);
                    second[p](i) += static_cast<double>(p) / left_den
                                    * first[p - 1](i);
                }
                if (right_den > 0.0) {
                    value[p](i) +=
                        (knots_[static_cast<std::size_t>(i + p + 1)] - x)
                        / right_den * value[p - 1](i + 1);
                    first[p](i) -= static_cast<double>(p) / right_den
                                   * value[p - 1](i + 1);
                    second[p](i) -= static_cast<double>(p) / right_den
                                    * first[p - 1](i + 1);
                }
            }
        }
        if (derivative == 0)
            return value[degree];
        if (derivative == 1)
            return first[degree];
        return second[degree];
    }

private:
    int coefficients_ = 0;
    int elements_ = 0;
    std::vector<double> knots_;
};

} // namespace

struct CapAtlasDensitySpace3D::Impl {
    explicit Impl(CapAtlasDensityOptions3D supplied)
        : options(std::move(supplied)),
          geometry_model(options.geometry_options()),
          spline(options.coefficients_per_direction)
    {
        validate_options();
        build_patches();
        build_seams();
        merge_c0_coefficients();
        build_mortar_constraints();
        build_reduction();
        build_trace_quadrature_and_mass();
    }

    void validate_options() const
    {
        if (options.mortar_gauss_order < 1
            || options.mortar_gauss_order > 5
            || options.trace_gauss_order < 1
            || options.trace_gauss_order > 5)
            throw std::invalid_argument("Gauss orders must lie between 1 and 5");
        if (!(options.square_half_width > 0.0)
            || std::sqrt(2.0) * options.square_half_width >= 1.0)
            throw std::invalid_argument(
                "square_half_width must make the central square lie inside the unit disk");
        if (!(options.polar_radius > std::sqrt(2.0)
                                           * options.square_half_width)
            || !(options.polar_radius < 1.0))
            throw std::invalid_argument(
                "polar_radius must enclose the square corners and be less than one");
        if (!(options.rank_tolerance > 0.0)
            || !(options.rank_tolerance < 1.0))
            throw std::invalid_argument("rank_tolerance must lie in (0,1)");
    }

    void build_patches()
    {
        patches.reserve(14);
        const std::array<CapSide3D, 4> sides = {
            CapSide3D::North,
            CapSide3D::South,
            CapSide3D::East,
            CapSide3D::West};
        auto append = [&](std::string name,
                          CapPatchKind3D kind,
                          CapSide3D side,
                          int hemisphere) {
            CapPatchDescriptor3D patch;
            patch.index = static_cast<int>(patches.size());
            patch.name = std::move(name);
            patch.kind = kind;
            patch.side = side;
            patch.hemisphere = hemisphere;
            patches.push_back(std::move(patch));
        };

        append("NC", CapPatchKind3D::PolarCentral, CapSide3D::None, +1);
        for (CapSide3D side : sides)
            append("N" + side_name(side),
                   CapPatchKind3D::PolarRing,
                   side,
                   +1);
        for (CapSide3D side : sides)
            append("B" + side_name(side),
                   CapPatchKind3D::MeridianBelt,
                   side,
                   0);
        append("SC", CapPatchKind3D::PolarCentral, CapSide3D::None, -1);
        for (CapSide3D side : sides)
            append("S" + side_name(side),
                   CapPatchKind3D::PolarRing,
                   side,
                   -1);
    }

    CapSurfaceEvaluation3D geometry(int patch_index, double u, double v) const
    {
        return geometry_model.evaluate(patch_index, u, v);
    }

    void build_seams()
    {
        struct EdgeRecord {
            int patch = -1;
            CapPatchEdge3D edge = CapPatchEdge3D::U0;
            bool paired = false;
        };
        std::vector<EdgeRecord> edges;
        const std::array<CapPatchEdge3D, 4> edge_types = {
            CapPatchEdge3D::U0,
            CapPatchEdge3D::U1,
            CapPatchEdge3D::V0,
            CapPatchEdge3D::V1};
        for (int p = 0; p < static_cast<int>(patches.size()); ++p)
            for (CapPatchEdge3D edge : edge_types)
                edges.push_back({p, edge, false});

        const std::array<double, 3> probes = {0.0, 0.371, 1.0};
        auto mismatch = [&](const EdgeRecord& a,
                            const EdgeRecord& b,
                            bool reversed) {
            double maximum = 0.0;
            for (double t : probes) {
                const auto [ua, va] = edge_uv(a.edge, t);
                const auto [ub, vb] = edge_uv(b.edge, reversed ? 1.0 - t : t);
                maximum = std::max(maximum,
                                   (geometry(a.patch, ua, va).point
                                    - geometry(b.patch, ub, vb).point)
                                       .norm());
            }
            return maximum;
        };

        const double shape_scale = options.shape == CapShape3D::Ellipsoid
                                       ? options.ellipsoid_axes.maxCoeff()
                                       : 1.0 + std::abs(options.flower_epsilon)
                                             + 2.0 * std::abs(options.flower_eta);
        const double tolerance = 2.0e-11 * shape_scale;
        seams.reserve(edges.size() / 2);
        for (std::size_t i = 0; i < edges.size(); ++i) {
            if (edges[i].paired)
                continue;
            double best = std::numeric_limits<double>::infinity();
            std::size_t best_j = edges.size();
            bool best_reversed = false;
            for (std::size_t j = i + 1; j < edges.size(); ++j) {
                if (edges[j].paired || edges[i].patch == edges[j].patch)
                    continue;
                const double direct = mismatch(edges[i], edges[j], false);
                const double reverse = mismatch(edges[i], edges[j], true);
                if (direct < best) {
                    best = direct;
                    best_j = j;
                    best_reversed = false;
                }
                if (reverse < best) {
                    best = reverse;
                    best_j = j;
                    best_reversed = true;
                }
            }
            if (best_j == edges.size() || best > tolerance) {
                std::ostringstream message;
                message << "Could not pair atlas edge "
                        << patches[static_cast<std::size_t>(edges[i].patch)].name
                        << ':' << edge_name(edges[i].edge)
                        << "; best geometric mismatch=" << best;
                throw std::runtime_error(message.str());
            }
            edges[i].paired = true;
            edges[best_j].paired = true;
            CapSeamDescriptor3D seam;
            seam.patch_a = edges[i].patch;
            seam.edge_a = edges[i].edge;
            seam.patch_b = edges[best_j].patch;
            seam.edge_b = edges[best_j].edge;
            seam.reversed = best_reversed;
            seam.label =
                patches[static_cast<std::size_t>(seam.patch_a)].name + ':'
                + edge_name(seam.edge_a) + "="
                + patches[static_cast<std::size_t>(seam.patch_b)].name + ':'
                + edge_name(seam.edge_b)
                + (seam.reversed ? "(reversed)" : "");
            seams.push_back(std::move(seam));
        }
        if (seams.size() != 28)
            throw std::runtime_error("The 14-patch cap atlas must have 28 seams");
    }

    std::vector<int> edge_local_ids(CapPatchEdge3D edge) const
    {
        const int n = spline.coefficients();
        std::vector<int> ids(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k) {
            switch (edge) {
            case CapPatchEdge3D::U0:
                ids[static_cast<std::size_t>(k)] = k;
                break;
            case CapPatchEdge3D::U1:
                ids[static_cast<std::size_t>(k)] = (n - 1) * n + k;
                break;
            case CapPatchEdge3D::V0:
                ids[static_cast<std::size_t>(k)] = k * n;
                break;
            case CapPatchEdge3D::V1:
                ids[static_cast<std::size_t>(k)] = k * n + n - 1;
                break;
            }
        }
        return ids;
    }

    void merge_c0_coefficients()
    {
        const int n = spline.coefficients();
        const int per_patch = n * n;
        const int local_count = static_cast<int>(patches.size()) * per_patch;
        UnionFind sets(local_count);
        for (const CapSeamDescriptor3D& seam : seams) {
            const std::vector<int> a = edge_local_ids(seam.edge_a);
            const std::vector<int> b = edge_local_ids(seam.edge_b);
            for (int k = 0; k < n; ++k) {
                const int kb = seam.reversed ? n - 1 - k : k;
                sets.unite(seam.patch_a * per_patch
                               + a[static_cast<std::size_t>(k)],
                           seam.patch_b * per_patch
                               + b[static_cast<std::size_t>(kb)]);
            }
        }

        local_to_c0_map.resize(static_cast<std::size_t>(local_count));
        std::vector<int> root_to_c0(static_cast<std::size_t>(local_count), -1);
        int next = 0;
        for (int local = 0; local < local_count; ++local) {
            const int root = sets.find(local);
            int& global = root_to_c0[static_cast<std::size_t>(root)];
            if (global < 0)
                global = next++;
            local_to_c0_map[static_cast<std::size_t>(local)] = global;
        }
        c0_count = next;
    }

    Eigen::RowVectorXd tensor_local_row(double u,
                                        double v,
                                        int derivative_u,
                                        int derivative_v) const
    {
        if (derivative_u < 0 || derivative_v < 0
            || derivative_u + derivative_v > 2)
            throw std::invalid_argument(
                "Tensor spline derivatives must have nonnegative total order at most two");
        const Eigen::VectorXd bu = spline.values(u, derivative_u);
        const Eigen::VectorXd bv = spline.values(v, derivative_v);
        const int n = spline.coefficients();
        Eigen::RowVectorXd row(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                row(i * n + j) = bu(i) * bv(j);
        return row;
    }

    Eigen::RowVectorXd c0_row(int patch_index,
                              double u,
                              double v,
                              int derivative_u,
                              int derivative_v) const
    {
        if (patch_index < 0 || patch_index >= static_cast<int>(patches.size()))
            throw std::out_of_range("Patch index is out of range");
        const Eigen::RowVectorXd local =
            tensor_local_row(u, v, derivative_u, derivative_v);
        Eigen::RowVectorXd result = Eigen::RowVectorXd::Zero(c0_count);
        const int per_patch = spline.coefficients() * spline.coefficients();
        for (int q = 0; q < per_patch; ++q)
            result(local_to_c0_map[static_cast<std::size_t>(
                patch_index * per_patch + q)]) += local(q);
        return result;
    }

    CapC0BasisStencil3D c0_stencil(int patch_index,
                                   double u,
                                   double v) const
    {
        if (patch_index < 0 || patch_index >= static_cast<int>(patches.size()))
            throw std::out_of_range("Patch index is out of range");
        const Eigen::RowVectorXd local = tensor_local_row(u, v, 0, 0);
        const int per_patch = spline.coefficients() * spline.coefficients();
        CapC0BasisStencil3D stencil;
        for (int q = 0; q < per_patch; ++q) {
            const double weight = local(q);
            if (weight == 0.0)
                continue;
            const int index = local_to_c0_map[static_cast<std::size_t>(
                patch_index * per_patch + q)];
            int entry = 0;
            while (entry < stencil.count
                   && stencil.indices[static_cast<std::size_t>(entry)]
                          != index)
                ++entry;
            if (entry == stencil.count) {
                if (stencil.count
                    >= static_cast<int>(stencil.indices.size()))
                    throw std::runtime_error(
                        "Cubic C0 basis stencil exceeded 16 entries");
                stencil.indices[static_cast<std::size_t>(entry)] = index;
                ++stencil.count;
            }
            stencil.weights[static_cast<std::size_t>(entry)] += weight;
        }
        return stencil;
    }

    double evaluate_c0(
        int patch_index,
        double u,
        double v,
        const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
    {
        if (patch_index < 0 || patch_index >= static_cast<int>(patches.size()))
            throw std::out_of_range("Patch index is out of range");
        if (coefficients.size() != c0_count)
            throw std::invalid_argument("C0 coefficient vector has wrong size");
        return c0_stencil(patch_index, u, v).dot(coefficients);
    }

    Eigen::RowVectorXd reduced_row(int patch_index,
                                   double u,
                                   double v,
                                   int derivative_u,
                                   int derivative_v) const
    {
        if (patch_index < 0 || patch_index >= static_cast<int>(patches.size()))
            throw std::out_of_range("Patch index is out of range");
        const Eigen::RowVectorXd local =
            tensor_local_row(u, v, derivative_u, derivative_v);
        Eigen::RowVectorXd result = Eigen::RowVectorXd::Zero(reduction.cols());
        const int per_patch = spline.coefficients() * spline.coefficients();
        // At an interior point a cubic tensor row has at most 16 nonzeros.
        // Accumulating the corresponding rows of R avoids a dense
        // (1 x n_C0) R product at every crossing/trace sample.
        for (int q = 0; q < per_patch; ++q) {
            if (local(q) == 0.0)
                continue;
            result.noalias() +=
                local(q)
                * reduction.row(local_to_c0_map[static_cast<std::size_t>(
                    patch_index * per_patch + q)]);
        }
        return result;
    }

    Eigen::RowVectorXd physical_conormal_row(int patch_index,
                                             CapPatchEdge3D edge,
                                             double t,
                                             const Eigen::Vector3d& conormal) const
    {
        const auto [u, v] = edge_uv(edge, t);
        const CapSurfaceEvaluation3D surface = geometry(patch_index, u, v);
        const Eigen::Matrix2d metric =
            surface.tangents.transpose() * surface.tangents;
        const Eigen::Vector2d coordinates =
            metric.ldlt().solve(surface.tangents.transpose() * conormal);
        return coordinates.x() * c0_row(patch_index, u, v, 1, 0)
               + coordinates.y() * c0_row(patch_index, u, v, 0, 1);
    }

    void build_mortar_constraints()
    {
        const int n = spline.coefficients();
        const int rows = static_cast<int>(seams.size()) * n;
        constraints = Eigen::MatrixXd::Zero(rows, c0_count);
        const QuadratureRule rule =
            gauss_legendre(options.mortar_gauss_order);
        const int elements = spline.elements();

        int seam_index = 0;
        for (const CapSeamDescriptor3D& seam : seams) {
            for (int element = 0; element < elements; ++element) {
                const double ta = static_cast<double>(element) / elements;
                const double tb = static_cast<double>(element + 1) / elements;
                for (std::size_t g = 0; g < rule.nodes.size(); ++g) {
                    const double t =
                        0.5 * ((tb - ta) * rule.nodes[g] + tb + ta);
                    const double weight = 0.5 * (tb - ta) * rule.weights[g];
                    const double t_b = seam.reversed ? 1.0 - t : t;
                    const auto [ua, va] = edge_uv(seam.edge_a, t);
                    const CapSurfaceEvaluation3D surface_a =
                        geometry(seam.patch_a, ua, va);
                    Eigen::Vector3d tangent =
                        edge_parameter_is_v(seam.edge_a)
                            ? surface_a.tangents.col(1)
                            : surface_a.tangents.col(0);
                    const double ds_dt = tangent.norm();
                    tangent /= ds_dt;
                    Eigen::Vector3d conormal = surface_a.normal.cross(tangent);
                    conormal.normalize();

                    const Eigen::RowVectorXd row_a = physical_conormal_row(
                        seam.patch_a, seam.edge_a, t, conormal);
                    const Eigen::RowVectorXd row_b = physical_conormal_row(
                        seam.patch_b, seam.edge_b, t_b, conormal);
                    const Eigen::VectorXd test = spline.values(t, 0);
                    const double factor = weight * ds_dt;
                    for (int k = 0; k < n; ++k) {
                        if (std::abs(test(k)) > 1.0e-15)
                            constraints.row(seam_index * n + k).noalias() +=
                                factor * test(k) * (row_a - row_b);
                    }
                }
            }
            ++seam_index;
        }
    }

    void build_reduction()
    {
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> row_qr;
        row_qr.setThreshold(options.rank_tolerance);
        row_qr.compute(constraints.transpose());
        constraint_rank = static_cast<int>(row_qr.rank());
        independent_constraints.resize(constraint_rank, c0_count);
        const auto row_permutation = row_qr.colsPermutation().indices();
        for (int i = 0; i < constraint_rank; ++i)
            independent_constraints.row(i) =
                constraints.row(row_permutation(i));

        if (constraint_rank == 0) {
            reduction = Eigen::MatrixXd::Identity(c0_count, c0_count);
            free_columns.resize(static_cast<std::size_t>(c0_count));
            std::iota(free_columns.begin(), free_columns.end(), 0);
            return;
        }

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> column_qr;
        column_qr.setThreshold(options.rank_tolerance);
        column_qr.compute(independent_constraints);
        if (column_qr.rank() != constraint_rank)
            throw std::runtime_error(
                "Independent mortar constraints lost rank during column selection");
        const auto permutation = column_qr.colsPermutation().indices();
        std::vector<int> pivot_columns(static_cast<std::size_t>(constraint_rank));
        free_columns.resize(static_cast<std::size_t>(c0_count - constraint_rank));
        for (int i = 0; i < constraint_rank; ++i)
            pivot_columns[static_cast<std::size_t>(i)] = permutation(i);
        for (int i = constraint_rank; i < c0_count; ++i)
            free_columns[static_cast<std::size_t>(i - constraint_rank)] =
                permutation(i);

        Eigen::MatrixXd pivot_matrix(constraint_rank, constraint_rank);
        Eigen::MatrixXd free_matrix(
            constraint_rank, c0_count - constraint_rank);
        for (int i = 0; i < constraint_rank; ++i)
            pivot_matrix.col(i) = independent_constraints.col(
                pivot_columns[static_cast<std::size_t>(i)]);
        for (int i = 0; i < c0_count - constraint_rank; ++i)
            free_matrix.col(i) = independent_constraints.col(
                free_columns[static_cast<std::size_t>(i)]);

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> pivot_qr;
        pivot_qr.setThreshold(options.rank_tolerance);
        pivot_qr.compute(pivot_matrix);
        if (pivot_qr.rank() != constraint_rank)
            throw std::runtime_error("Mortar pivot block is singular");
        const Eigen::MatrixXd eliminated = pivot_qr.solve(-free_matrix);

        reduction = Eigen::MatrixXd::Zero(
            c0_count, c0_count - constraint_rank);
        for (int j = 0; j < c0_count - constraint_rank; ++j) {
            reduction(free_columns[static_cast<std::size_t>(j)], j) = 1.0;
            for (int i = 0; i < constraint_rank; ++i)
                reduction(pivot_columns[static_cast<std::size_t>(i)], j) =
                    eliminated(i, j);
        }
        if (!reduction.allFinite())
            throw std::runtime_error("Non-finite reduced density basis");
    }

    void build_trace_quadrature_and_mass()
    {
        const QuadratureRule rule = gauss_legendre(options.trace_gauss_order);
        const int elements = spline.elements();
        const std::size_t count =
            patches.size() * static_cast<std::size_t>(elements * elements)
            * rule.nodes.size() * rule.nodes.size();
        trace.reserve(count);
        Eigen::RowVectorXd mass_c0 = Eigen::RowVectorXd::Zero(c0_count);

        for (int patch = 0; patch < static_cast<int>(patches.size()); ++patch) {
            for (int eu = 0; eu < elements; ++eu) {
                const double ua = static_cast<double>(eu) / elements;
                const double ub = static_cast<double>(eu + 1) / elements;
                for (int ev = 0; ev < elements; ++ev) {
                    const double va = static_cast<double>(ev) / elements;
                    const double vb = static_cast<double>(ev + 1) / elements;
                    for (std::size_t gu = 0; gu < rule.nodes.size(); ++gu) {
                        const double u =
                            0.5 * ((ub - ua) * rule.nodes[gu] + ub + ua);
                        const double wu = 0.5 * (ub - ua) * rule.weights[gu];
                        for (std::size_t gv = 0; gv < rule.nodes.size(); ++gv) {
                            const double v =
                                0.5 * ((vb - va) * rule.nodes[gv] + vb + va);
                            const double wv =
                                0.5 * (vb - va) * rule.weights[gv];
                            const CapSurfaceEvaluation3D surface =
                                geometry(patch, u, v);
                            CapGaussTracePoint3D point;
                            point.patch = patch;
                            point.element_u = eu;
                            point.element_v = ev;
                            point.u = u;
                            point.v = v;
                            point.point = surface.point;
                            point.normal = surface.normal;
                            point.surface_weight =
                                wu * wv * surface.area_element;
                            area += point.surface_weight;
                            mass_c0.noalias() +=
                                point.surface_weight * c0_row(patch, u, v, 0, 0);
                            trace.push_back(std::move(point));
                        }
                    }
                }
            }
        }
        mass = mass_c0 * reduction;
        if (!(area > 0.0) || !std::isfinite(area) || !mass.allFinite())
            throw std::runtime_error("Invalid cap-atlas mass functional");
    }

    CapPatchLocation3D locate(const Eigen::Vector3d& physical) const
    {
        return geometry_model.locate(physical);
    }

    Eigen::Vector3d edge_tangent(int patch,
                                 CapPatchEdge3D edge,
                                 double t) const
    {
        const auto [u, v] = edge_uv(edge, t);
        const CapSurfaceEvaluation3D surface = geometry(patch, u, v);
        return edge_parameter_is_v(edge) ? surface.tangents.col(1)
                                         : surface.tangents.col(0);
    }

    CapAtlasDensityOptions3D options;
    geometry3d::AnalyticCapGeometry3D geometry_model;
    CubicOpenUniformBasis spline;
    std::vector<CapPatchDescriptor3D> patches;
    std::vector<CapSeamDescriptor3D> seams;
    std::vector<int> local_to_c0_map;
    int c0_count = 0;
    int constraint_rank = 0;
    Eigen::MatrixXd constraints;
    Eigen::MatrixXd independent_constraints;
    Eigen::MatrixXd reduction;
    std::vector<int> free_columns;
    std::vector<CapGaussTracePoint3D> trace;
    Eigen::RowVectorXd mass;
    double area = 0.0;
};

CapAtlasDensitySpace3D::CapAtlasDensitySpace3D(
    CapAtlasDensityOptions3D options)
    : impl_(std::make_unique<Impl>(std::move(options)))
{
}

CapAtlasDensitySpace3D::~CapAtlasDensitySpace3D() = default;
CapAtlasDensitySpace3D::CapAtlasDensitySpace3D(
    CapAtlasDensitySpace3D&&) noexcept = default;
CapAtlasDensitySpace3D& CapAtlasDensitySpace3D::operator=(
    CapAtlasDensitySpace3D&&) noexcept = default;

const CapAtlasDensityOptions3D& CapAtlasDensitySpace3D::options() const
{
    return impl_->options;
}

int CapAtlasDensitySpace3D::patch_count() const
{
    return static_cast<int>(impl_->patches.size());
}

int CapAtlasDensitySpace3D::coefficients_per_direction() const
{
    return impl_->spline.coefficients();
}

int CapAtlasDensitySpace3D::elements_per_direction() const
{
    return impl_->spline.elements();
}

int CapAtlasDensitySpace3D::local_coefficient_count() const
{
    const int n = coefficients_per_direction();
    return patch_count() * n * n;
}

int CapAtlasDensitySpace3D::c0_coefficient_count() const
{
    return impl_->c0_count;
}

int CapAtlasDensitySpace3D::c1_constraint_count() const
{
    return static_cast<int>(impl_->constraints.rows());
}

int CapAtlasDensitySpace3D::c1_constraint_rank() const
{
    return impl_->constraint_rank;
}

int CapAtlasDensitySpace3D::reduced_coefficient_count() const
{
    return static_cast<int>(impl_->reduction.cols());
}

const std::vector<CapPatchDescriptor3D>&
CapAtlasDensitySpace3D::patches() const
{
    return impl_->patches;
}

const std::vector<CapSeamDescriptor3D>& CapAtlasDensitySpace3D::seams() const
{
    return impl_->seams;
}

const std::vector<int>& CapAtlasDensitySpace3D::local_to_c0() const
{
    return impl_->local_to_c0_map;
}

CapSurfaceEvaluation3D CapAtlasDensitySpace3D::geometry(int patch,
                                                       double u,
                                                       double v) const
{
    return impl_->geometry(patch, u, v);
}

double CapAtlasDensitySpace3D::level_set(const Eigen::Vector3d& point) const
{
    return impl_->geometry_model.level_set(point);
}

Eigen::Vector3d CapAtlasDensitySpace3D::level_set_gradient(
    const Eigen::Vector3d& point) const
{
    return impl_->geometry_model.level_set_gradient(point);
}

Eigen::Vector3d CapAtlasDensitySpace3D::outward_normal(
    const Eigen::Vector3d& point) const
{
    return impl_->geometry_model.outward_normal(point);
}

bool CapAtlasDensitySpace3D::inside(const Eigen::Vector3d& point) const
{
    return impl_->geometry_model.inside(point);
}

Eigen::Vector3d CapAtlasDensitySpace3D::surface_point_from_direction(
    const Eigen::Vector3d& direction) const
{
    return impl_->geometry_model.surface_point_from_direction(direction);
}

CapPatchLocation3D CapAtlasDensitySpace3D::locate(
    const Eigen::Vector3d& point) const
{
    return impl_->locate(point);
}

Eigen::RowVectorXd CapAtlasDensitySpace3D::c0_basis_row(int patch,
                                                       double u,
                                                       double v) const
{
    return impl_->c0_row(patch, u, v, 0, 0);
}

CapC0BasisStencil3D CapAtlasDensitySpace3D::c0_basis_stencil(
    int patch,
    double u,
    double v) const
{
    return impl_->c0_stencil(patch, u, v);
}

Eigen::RowVectorXd CapAtlasDensitySpace3D::reduced_basis_row(int patch,
                                                            double u,
                                                            double v) const
{
    return impl_->reduced_row(patch, u, v, 0, 0);
}

Eigen::RowVectorXd CapAtlasDensitySpace3D::c0_basis_derivative_row(
    int patch,
    double u,
    double v,
    int derivative_u,
    int derivative_v) const
{
    return impl_->c0_row(patch, u, v, derivative_u, derivative_v);
}

Eigen::RowVectorXd CapAtlasDensitySpace3D::reduced_basis_derivative_row(
    int patch,
    double u,
    double v,
    int derivative_u,
    int derivative_v) const
{
    return impl_->reduced_row(
        patch, u, v, derivative_u, derivative_v);
}

Eigen::VectorXd CapAtlasDensitySpace3D::expand_reduced(
    const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
{
    if (coefficients.size() != reduced_coefficient_count())
        throw std::invalid_argument("Reduced coefficient vector has wrong size");
    if (!coefficients.allFinite())
        throw std::invalid_argument("Reduced coefficients must be finite");
    return impl_->reduction * coefficients;
}

double CapAtlasDensitySpace3D::evaluate_c0(
    int patch,
    double u,
    double v,
    const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
{
    return impl_->evaluate_c0(patch, u, v, coefficients);
}

double CapAtlasDensitySpace3D::evaluate(
    int patch,
    double u,
    double v,
    const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
{
    if (coefficients.size() != reduced_coefficient_count())
        throw std::invalid_argument("Reduced coefficient vector has wrong size");
    return reduced_basis_row(patch, u, v).dot(coefficients);
}

const Eigen::MatrixXd& CapAtlasDensitySpace3D::reduction_matrix() const
{
    return impl_->reduction;
}

const Eigen::MatrixXd& CapAtlasDensitySpace3D::c1_constraint_matrix() const
{
    return impl_->constraints;
}

const Eigen::MatrixXd&
CapAtlasDensitySpace3D::independent_c1_constraint_matrix() const
{
    return impl_->independent_constraints;
}

const std::vector<CapGaussTracePoint3D>&
CapAtlasDensitySpace3D::trace_points() const
{
    return impl_->trace;
}

const Eigen::RowVectorXd& CapAtlasDensitySpace3D::mass_row() const
{
    return impl_->mass;
}

double CapAtlasDensitySpace3D::surface_area() const
{
    return impl_->area;
}

double CapAtlasDensitySpace3D::reduction_constraint_residual() const
{
    if (impl_->constraints.rows() == 0 || impl_->reduction.cols() == 0)
        return 0.0;
    return (impl_->constraints * impl_->reduction).cwiseAbs().maxCoeff();
}

double CapAtlasDensitySpace3D::max_c0_seam_jump(
    const Eigen::Ref<const Eigen::VectorXd>& coefficients,
    int samples_per_seam) const
{
    if (coefficients.size() != reduced_coefficient_count())
        throw std::invalid_argument("Reduced coefficient vector has wrong size");
    if (samples_per_seam < 2)
        throw std::invalid_argument("At least two seam samples are required");
    double maximum = 0.0;
    for (const CapSeamDescriptor3D& seam : impl_->seams) {
        for (int k = 0; k < samples_per_seam; ++k) {
            const double t = static_cast<double>(k) / (samples_per_seam - 1);
            const double tb = seam.reversed ? 1.0 - t : t;
            const auto [ua, va] = edge_uv(seam.edge_a, t);
            const auto [ub, vb] = edge_uv(seam.edge_b, tb);
            maximum = std::max(maximum,
                               std::abs(evaluate(seam.patch_a,
                                                 ua,
                                                 va,
                                                 coefficients)
                                        - evaluate(seam.patch_b,
                                                   ub,
                                                   vb,
                                                   coefficients)));
        }
    }
    return maximum;
}

double CapAtlasDensitySpace3D::max_physical_conormal_jump(
    const Eigen::Ref<const Eigen::VectorXd>& coefficients,
    int samples_per_seam) const
{
    if (coefficients.size() != reduced_coefficient_count())
        throw std::invalid_argument("Reduced coefficient vector has wrong size");
    if (samples_per_seam < 2)
        throw std::invalid_argument("At least two seam samples are required");
    const Eigen::VectorXd c0 = impl_->reduction * coefficients;
    double maximum = 0.0;
    for (const CapSeamDescriptor3D& seam : impl_->seams) {
        // Avoid exact corners, where more than two patch charts meet.
        for (int k = 0; k < samples_per_seam; ++k) {
            const double t = (static_cast<double>(k) + 0.5)
                             / samples_per_seam;
            const double tb = seam.reversed ? 1.0 - t : t;
            const auto [ua, va] = edge_uv(seam.edge_a, t);
            const CapSurfaceEvaluation3D surface_a =
                impl_->geometry(seam.patch_a, ua, va);
            Eigen::Vector3d tangent =
                edge_parameter_is_v(seam.edge_a)
                    ? surface_a.tangents.col(1)
                    : surface_a.tangents.col(0);
            tangent.normalize();
            Eigen::Vector3d conormal = surface_a.normal.cross(tangent);
            conormal.normalize();
            const double da = impl_->physical_conormal_row(
                                  seam.patch_a, seam.edge_a, t, conormal)
                                  .dot(c0);
            const double db = impl_->physical_conormal_row(
                                  seam.patch_b, seam.edge_b, tb, conormal)
                                  .dot(c0);
            maximum = std::max(maximum, std::abs(da - db));
        }
    }
    return maximum;
}

} // namespace kfbim::app3d
