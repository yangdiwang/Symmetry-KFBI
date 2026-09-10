#include "spread_restrict_resource_3d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace kfbim::app3d {
namespace {
using Clock = std::chrono::steady_clock;
using Key = std::tuple<int, double, double>;
double seconds(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}
Eigen::Vector3d node_point(const CartesianGrid3D& grid, int node) {
    const auto p = grid.coord(node);
    return {p[0], p[1], p[2]};
}
std::pair<double, double> normalized(const NativeNurbsDensitySpace3D& density,
                                   const RestrictResourceAnchor3D& anchor) {
    const auto& patch = density.surface().patches.at(anchor.patch_id);
    const double u = (anchor.u - patch.domain_start_u()) /
        (patch.domain_end_u() - patch.domain_start_u());
    const double v = (anchor.v - patch.domain_start_v()) /
        (patch.domain_end_v() - patch.domain_start_v());
    if (!std::isfinite(u) || !std::isfinite(v) || u < 0 || u > 1 || v < 0 || v > 1)
        throw std::invalid_argument("resource anchor has invalid native parameters");
    return {u, v};
}
ResourceAffineRow3D affine_row(const NativeDensityC0Stencil3D& row, double known) {
    ResourceAffineRow3D result;
    std::map<int, double> sorted;
    for (int i = 0; i < row.count; ++i) sorted[row.indices[i]] += row.weights[i];
    for (const auto& [column, value] : sorted) {
        result.columns.push_back(column);
        result.values.push_back(value);
    }
    result.known = known;
    return result;
}
struct P2Center {
    NativeSurfaceParameterJet3D geometry;
    DirectCoefficientValueJetPlan3D value;
    DirectCoefficientNormalJetPlan3D normal;
    ValueJet3D known_value = ValueJet3D::Zero();
    NormalJet3D known_normal = NormalJet3D::Zero();
};
void set_known(P2Center& center, NativeDensityField3D unknown,
               const ResourceKnownAmbientCallback3D& callback, double mean) {
    const auto data = callback(center.geometry.point);
    const auto& frame = center.value.frame;
    if (unknown == NativeDensityField3D::NormalTrace) {
        KnownDirichletValueGradientHessian3D lower;
        lower.value = data.value;
        lower.ambient_gradient = data.gradient;
        lower.ambient_hessian = data.hessian;
        center.known_value = known_dirichlet_jet_from_ambient_derivatives_3d(
            lower, center.geometry, frame);
    } else {
        auto n = surface_normal_parameter_jet_3d(center.geometry);
        if (n.normal.dot(frame.normal) < 0) {
            n.normal = -n.normal; n.normal_u = -n.normal_u; n.normal_v = -n.normal_v;
        }
        Eigen::Vector2d parameter;
        parameter << (data.hessian * center.geometry.x_u).dot(n.normal)
                         + data.gradient.dot(n.normal_u),
                     (data.hessian * center.geometry.x_v).dot(n.normal)
                         + data.gradient.dot(n.normal_v);
        const auto tangent = parameter_gradient_to_tangent_gradient_3d(
            center.geometry, frame, parameter);
        center.known_normal << data.gradient.dot(n.normal) - mean, tangent.x(), tangent.y();
    }
}
P2Center independent_p2(const NativeNurbsDensitySpace3D& density,
                        const RestrictResourceAnchor3D& anchor,
                        NativeDensityField3D unknown,
                        const ResourceKnownAmbientCallback3D& known, double mean) {
    const auto [u, v] = normalized(density, anchor);
    P2Center result;
    result.geometry = native_surface_parameter_jet_3d(density, anchor.patch_id, u, v);
    const auto frame = make_local_orthonormal_frame_3d(result.geometry.normal, result.geometry.x_u);
    result.value = build_direct_coefficient_value_jet_plan_3d(
        density, anchor.patch_id, u, v, result.geometry, frame);
    result.normal = build_direct_coefficient_normal_jet_plan_3d(
        density, anchor.patch_id, u, v, result.geometry, frame);
    set_known(result, unknown, known, mean);
    return result;
}
P2Center lower_p2(const DirectCoefficientCubicCauchyPlan3D& cubic,
                  const NativeSurfaceParameterJet3D& geometry,
                  const CubicValueJet3D& known_value, const CubicNormalJet3D& known_normal) {
    P2Center result;
    result.geometry = geometry;
    // Named lower plans select surface-derivative labels and preserve frame,
    // curvature and diagnostics. Ambient P3 coefficients are never truncated.
    result.value = cubic.lower_value_plan();
    result.normal = cubic.lower_normal_plan();
    result.known_value = known_value.head<6>();
    result.known_normal = known_normal.head<3>();
    return result;
}
ResourceAffineRow3D evaluate_p2(const P2Center& center, const Eigen::Vector3d& target,
                               NativeDensityField3D unknown) {
    const auto weights = cauchy_polynomial_weights_3d(center.geometry.point,
        center.value.frame, center.value.graph_hessian, target);
    return unknown == NativeDensityField3D::ValueTrace
        ? affine_row(center.value.compose_value_row(weights.w0), weights.w1.dot(center.known_normal))
        : affine_row(center.normal.compose_normal_row(weights.w1), weights.w0.dot(center.known_value));
}
} // namespace

ResourceGeometryPlan3D build_resource_geometry_skeleton_3d(
    const CartesianGrid3D& grid, const GridPair3D& pair,
    const NativeNurbsSurface3D& surface, const LaplaceCorrectionSupport3D& support,
    std::vector<RestrictResourceAnchor3D> trace, TensorProductCoverKind3D kind,
    bool python_mean_centered_cover, double cover_endpoint_snap) {
    const auto begin = Clock::now();
    ResourceGeometryPlan3D result;
    std::vector<int> sheets(surface.patches.size(), -1);
    for (int p = 0; p < static_cast<int>(sheets.size()); ++p) {
        if (sheets[p] >= 0) continue;
        const auto smooth = smooth_patch_component(surface, p);
        for (int member : smooth) sheets.at(member) = p;
    }
    for (auto& anchor : trace) anchor.sheet_id = sheets.at(anchor.patch_id);
    result.trace_anchors = std::move(trace);
    std::map<Key, int> crossing_indices;
    for (const auto& op : support.crossing_ops) {
        const auto owner = laplace_crossing_owner_3d(pair, op);
        if (owner.nurbs_patch_index < 0 || !owner.nurbs_parameter.allFinite())
            throw std::runtime_error("resource spread requires native crossing owners");
        const Key key{owner.nurbs_patch_index, owner.nurbs_parameter.x(), owner.nurbs_parameter.y()};
        auto found = crossing_indices.find(key);
        if (found == crossing_indices.end()) {
            const int id = static_cast<int>(result.spread_anchors.size());
            RestrictResourceAnchor3D anchor;
            anchor.point = owner.crossing_point;
            anchor.patch_id = owner.nurbs_patch_index;
            anchor.u = owner.nurbs_parameter.x(); anchor.v = owner.nurbs_parameter.y();
            anchor.sheet_id = sheets.at(anchor.patch_id); anchor.crossing_id = id;
            result.spread_anchors.push_back(anchor);
            found = crossing_indices.emplace(key, id).first;
        }
        result.crossing_for_op.push_back(found->second);
    }
    for (int center = 0; center < static_cast<int>(result.trace_anchors.size()); ++center) {
        const auto& anchor = result.trace_anchors[center];
        const auto jet = surface.patches.at(anchor.patch_id).evaluate_with_derivatives(anchor.u, anchor.v);
        const Eigen::Vector3d normal = jet.du.cross(jet.dv).normalized();
        result.stencils.push_back(build_shared_side_cover_restrict_stencil_3d(grid, anchor.point, normal, kind,
            python_mean_centered_cover, cover_endpoint_snap));
        for (const auto& side : result.stencils.back().sides) {
            for (int node : side.grid_ids) {
                result.visits.push_back({center, side.desired_inside, node,
                    node_point(grid, node), pair.domain_label(node) > 0});
            }
        }
    }
    result.planning_seconds = seconds(begin);
    return result;
}

ResourceGeometryPlan3D build_resource_geometry_plan_3d(
    const CartesianGrid3D& grid, const GridPair3D& pair,
    const NativeNurbsSurface3D& surface, const LaplaceCorrectionSupport3D& support,
    std::vector<RestrictResourceAnchor3D> trace, TensorProductCoverKind3D kind,
    RestrictResourceMode3D engine) {
    const auto begin = Clock::now();
    auto result = build_resource_geometry_skeleton_3d(grid, pair, surface, support,
        std::move(trace), kind);
    result.restrict_plan = build_restrict_resource_plan_3d(result.trace_anchors,
        result.spread_anchors, result.visits, grid.spacing()[0], engine);
    result.planning_seconds = seconds(begin);
    return result;
}

Eigen::VectorXd resource_constant_trace_3d(const ResourceGeometryPlan3D& geometry,
    const GridPair3D& pair, const Eigen::VectorXd& potential, bool interior, bool normal) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(geometry.stencils.size());
    for (std::size_t q = 0; q < geometry.stencils.size(); ++q) {
        for (const auto& side : geometry.stencils[q].sides) {
            Eigen::VectorXd values(side.grid_ids.size());
            for (std::size_t k = 0; k < side.grid_ids.size(); ++k) {
                const int id = side.grid_ids[k];
                values[k] = potential[id] + static_cast<int>(side.desired_inside)
                    - static_cast<int>(pair.domain_label(id) > 0);
            }
            Eigen::Vector3d samples = side.sampling_weights * values;
            if (!interior && side.desired_inside) samples.array() -= 1.0;
            if (interior && !side.desired_inside) samples.array() += 1.0;
            result[q] += (normal ? side.normal_recovery : side.value_recovery).dot(samples);
        }
    }
    return result;
}

ResourceBvpOperators3D build_resource_bvp_operators_3d(
    const ResourceGeometryPlan3D& geometry, const CartesianGrid3D& grid,
    const GridPair3D& /*pair*/, const LaplaceCorrectionSupport3D& support,
    const NativeNurbsDensitySpace3D& density, NativeDensityField3D unknown,
    const ResourceKnownAmbientCallback3D& known, double mean) {
    if (!known || !std::isfinite(mean)) throw std::invalid_argument("resource known data are required");
    if (geometry.crossing_for_op.size() != support.crossing_ops.size())
        throw std::invalid_argument("resource crossing catalog/op mismatch");
    const bool reuse = geometry.restrict_plan.mode != RestrictResourceMode3D::ReferencePerVisit;
    ResourceBvpOperators3D result;
    auto& stats = result.statistics;
    stats.planning = geometry.restrict_plan.counts;
    stats.planning_seconds = geometry.planning_seconds;
    stats.spread_centers = geometry.spread_anchors.size();
    const std::set<int> selected(geometry.restrict_plan.required_crossing_ids.begin(),
                                geometry.restrict_plan.required_crossing_ids.end());
    stats.retained_crossing_centers = selected.size();
    const auto spread_begin = Clock::now();
    std::vector<std::vector<int>> operations(geometry.spread_anchors.size());
    for (int i = 0; i < static_cast<int>(support.crossing_ops.size()); ++i)
        operations.at(geometry.crossing_for_op[i]).push_back(i);
    std::map<int, P2Center> retained;
    std::map<std::pair<int, int>, ResourceAffineRow3D> endpoint_rows;
    std::set<std::pair<int, int>> selected_targets;
    for (const auto& request : geometry.restrict_plan.requests)
        if (request.anchor_source == RestrictAnchorSource3D::SpreadCrossing)
            selected_targets.emplace(request.anchor_index, request.grid_full_id);
    std::vector<Eigen::Triplet<double>> triplets;
    result.known_spread = Eigen::VectorXd::Zero(grid.num_dofs());
    for (int id = 0; id < static_cast<int>(geometry.spread_anchors.size()); ++id) {
        const auto& anchor = geometry.spread_anchors[id];
        const auto [u, v] = normalized(density, anchor);
        const auto jet = native_surface_cubic_parameter_jet_3d(density, anchor.patch_id, u, v);
        const auto frame = make_local_orthonormal_frame_3d(jet.lower.normal, jet.lower.x_u);
        const auto cubic = build_direct_coefficient_cubic_cauchy_plan_3d(
            density, anchor.patch_id, u, v, jet, frame);
        const auto data = known(cubic.center);
        CubicValueJet3D j0 = CubicValueJet3D::Zero();
        CubicNormalJet3D j1 = CubicNormalJet3D::Zero();
        if (unknown == NativeDensityField3D::ValueTrace) {
            j1 = known_cubic_neumann_jet_3d(data, cubic.graph, frame); j1[0] -= mean;
        } else j0 = known_cubic_dirichlet_jet_3d(data, cubic.graph, frame);
        if (reuse && selected.count(id)) {
            const auto retained_begin = Clock::now();
            retained.emplace(id, lower_p2(cubic, jet.lower, j0, j1));
            stats.retained_p2_seconds += seconds(retained_begin);
            ++stats.p2_from_p3;
        }
        for (int op_index : operations[id]) {
            const auto& op = support.crossing_ops[op_index];
            const auto target = node_point(grid, op.correction_node);
            const Eigen::Vector3d displacement = target - cubic.center;
            const auto row = unknown == NativeDensityField3D::ValueTrace
                ? cubic.compose_value_row(displacement) : cubic.compose_normal_row(displacement);
            const double offset = unknown == NativeDensityField3D::ValueTrace
                ? cubic.normal_weights(displacement).dot(j1) : cubic.value_weights(displacement).dot(j0);
            const double scale = static_cast<double>(op.side_delta) * op.stencil_weight;
            for (int k = 0; k < row.count; ++k)
                triplets.emplace_back(op.rhs_node, row.indices[k], scale * row.weights[k]);
            result.known_spread[op.rhs_node] += scale * offset;
            if (reuse && selected_targets.count({id, op.correction_node})
                && !endpoint_rows.count({id, op.correction_node})) {
                const auto endpoint_begin = Clock::now();
                endpoint_rows.emplace(std::make_pair(id, op.correction_node),
                    evaluate_p2(retained.at(id), target, unknown));
                stats.endpoint_p2_seconds += seconds(endpoint_begin);
                ++stats.p2_row_evaluations;
            }
        }
    }
    result.S.resize(grid.num_dofs(), density.c0_coefficient_count());
    result.S.setFromTriplets(triplets.begin(), triplets.end());
    result.S.makeCompressed();
    std::vector<Eigen::Triplet<double>>().swap(triplets);
    stats.spread_seconds = seconds(spread_begin);
    const auto restrict_begin = Clock::now();
    std::vector<P2Center> trace;
    for (const auto& anchor : geometry.trace_anchors)
        trace.push_back(independent_p2(density, anchor, unknown, known, mean));
    for (int id : selected) if (!retained.count(id))
        retained.emplace(id, independent_p2(density, geometry.spread_anchors.at(id), unknown, known, mean));
    stats.p2_centers = trace.size() + retained.size();
    std::vector<ResourceAffineRow3D> request_rows;
    request_rows.reserve(geometry.restrict_plan.requests.size());
    for (const auto& request : geometry.restrict_plan.requests) {
        if (request.anchor_source == RestrictAnchorSource3D::SpreadCrossing) {
            const auto found = endpoint_rows.find({request.anchor_index, request.grid_full_id});
            if (found != endpoint_rows.end()) {
                request_rows.push_back(found->second); ++stats.endpoint_row_hits; continue;
            }
        }
        const auto& center = request.anchor_source == RestrictAnchorSource3D::Trace
            ? trace.at(request.anchor_index) : retained.at(request.anchor_index);
        request_rows.push_back(evaluate_p2(center, request.grid_point, unknown));
        ++stats.p2_row_evaluations;
    }
    std::vector<ResourceTraceSampleJumpRows3D> sample_rows(trace.size());
    std::vector<Eigen::Triplet<double>> basis_triplets;
    for (std::size_t q = 0; q < trace.size(); ++q) {
        // The trace value basis is already present in the analytic P2 jet.
        const auto& basis = trace[q].value.cauchy_rows[0];
        for (int k = 0; k < basis.count; ++k) basis_triplets.emplace_back(q, basis.indices[k], basis.weights[k]);
        for (int side = 0; side < 2; ++side) for (int k = 0; k < 3; ++k) {
            sample_rows[q][3 * side + k] = evaluate_p2(trace[q], geometry.stencils[q].sides[side].sample_points[k], unknown);
            ++stats.p2_row_evaluations;
        }
    }
    result.trace_basis.resize(trace.size(), density.c0_coefficient_count());
    result.trace_basis.setFromTriplets(basis_triplets.begin(), basis_triplets.end());
    result.trace_basis.makeCompressed();
    result.restrict = assemble_resource_restrict_3d(geometry.restrict_plan,
        geometry.stencils, geometry.visits, request_rows, sample_rows,
        grid.num_dofs(), density.c0_coefficient_count());
    // All centers, third-order data, request rows and target caches are local:
    // no setup-only resource survives into GMRES.
    stats.temporary_centers_after_build = 0;
    stats.restrict_seconds = seconds(restrict_begin);
    return result;
}

double resource_operator_max_difference_3d(const ResourceBvpOperators3D& a,
                                           const ResourceBvpOperators3D& b) {
    double error = resource_sparse_max_difference_3d(a.S, b.S);
    const auto vector_error = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        if (x.size() != y.size()) throw std::invalid_argument("resource vector dimensions differ");
        return x.size() ? (x - y).cwiseAbs().maxCoeff() : 0.0;
    };
    error = std::max(error, vector_error(a.known_spread, b.known_spread));
    error = std::max(error, resource_sparse_max_difference_3d(a.trace_basis, b.trace_basis));
    error = std::max(error, resource_sparse_max_difference_3d(a.restrict.Rg_value, b.restrict.Rg_value));
    error = std::max(error, resource_sparse_max_difference_3d(a.restrict.Rg_normal, b.restrict.Rg_normal));
    for (const bool inside : {false, true}) {
        const auto& x = inside ? a.restrict.interior : a.restrict.exterior;
        const auto& y = inside ? b.restrict.interior : b.restrict.exterior;
        error = std::max(error, resource_sparse_max_difference_3d(x.Rc_value, y.Rc_value));
        error = std::max(error, resource_sparse_max_difference_3d(x.Rc_normal, y.Rc_normal));
        error = std::max(error, vector_error(x.known_value, y.known_value));
        error = std::max(error, vector_error(x.known_normal, y.known_normal));
    }
    return error;
}

void dump_resource_operators_3d(const ResourceBvpOperators3D& op, const std::string& directory) {
    std::filesystem::create_directories(directory);
    const auto matrix = [&](const char* name, const ResourceRestrictSparseMatrix3D& m) {
        std::ofstream out(std::filesystem::path(directory) / (std::string(name) + ".mtx"));
        out << "%%MatrixMarket matrix coordinate real general\n" << std::setprecision(17)
            << m.rows() << ' ' << m.cols() << ' ' << m.nonZeros() << '\n';
        for (int row = 0; row < m.outerSize(); ++row)
            for (ResourceRestrictSparseMatrix3D::InnerIterator it(m, row); it; ++it)
                out << it.row() + 1 << ' ' << it.col() + 1 << ' ' << it.value() << '\n';
        if (!out) throw std::runtime_error("resource matrix dump failed");
    };
    const auto vector = [&](const char* name, const Eigen::VectorXd& v) {
        std::ofstream out(std::filesystem::path(directory) / (std::string(name) + ".txt"));
        out << std::setprecision(17) << v << '\n';
        if (!out) throw std::runtime_error("resource vector dump failed");
    };
    matrix("S_delta", op.S); vector("bs_delta", op.known_spread); matrix("B", op.trace_basis);
    matrix("Rgv", op.restrict.Rg_value); matrix("Rgn", op.restrict.Rg_normal);
    matrix("Rcv", op.restrict.exterior.Rc_value); matrix("Rcn", op.restrict.exterior.Rc_normal);
    vector("bv", op.restrict.exterior.known_value); vector("bn", op.restrict.exterior.known_normal);
}

} // namespace kfbim::app3d
