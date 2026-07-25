#include "neumann_edge_continuity_3d.hpp"

#include "../src/operators/i_kfbi_operator.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace kfbim::app3d;

void require(bool ok, const std::string& message)
{
    if (!ok) throw std::runtime_error(message);
}

Eigen::Vector3d edge_point(const NativeNurbsSurface3D& surface,
                           const kfbim::geometry3d::NurbsPatchEdgeInterval3D& interval,
                           double parameter)
{
    const auto& patch = surface.patches.at(static_cast<std::size_t>(interval.patch));
    switch (interval.edge) {
    case PatchEdge3D::UMin: return patch.evaluate(patch.domain_start_u(), parameter);
    case PatchEdge3D::UMax: return patch.evaluate(patch.domain_end_u(), parameter);
    case PatchEdge3D::VMin: return patch.evaluate(parameter, patch.domain_start_v());
    case PatchEdge3D::VMax: return patch.evaluate(parameter, patch.domain_end_v());
    }
    throw std::runtime_error("unknown edge");
}

double edge_length(const NativeNurbsSurface3D& surface,
                   const kfbim::geometry3d::NurbsPatchEdgeInterval3D& interval)
{
    constexpr std::array<double, 8> x = {{-0.9602898564975363, -0.7966664774136267,
        -0.5255324099163290, -0.1834346424956498, 0.1834346424956498,
         0.5255324099163290, 0.7966664774136267, 0.9602898564975363}};
    constexpr std::array<double, 8> w = {{0.1012285362903763, 0.2223810344533745,
        0.3137066458778873, 0.3626837833783620, 0.3626837833783620,
        0.3137066458778873, 0.2223810344533745, 0.1012285362903763}};
    const auto& patch = surface.patches.at(static_cast<std::size_t>(interval.patch));
    const double half = 0.5 * (interval.end - interval.begin);
    const double middle = 0.5 * (interval.begin + interval.end);
    double result = 0.0;
    for (std::size_t q = 0; q < x.size(); ++q) {
        const double parameter = middle + half * x[q];
        const auto d = interval.edge == PatchEdge3D::UMin
            ? patch.evaluate_with_derivatives(patch.domain_start_u(), parameter)
            : interval.edge == PatchEdge3D::UMax
            ? patch.evaluate_with_derivatives(patch.domain_end_u(), parameter)
            : interval.edge == PatchEdge3D::VMin
            ? patch.evaluate_with_derivatives(parameter, patch.domain_start_v())
            : patch.evaluate_with_derivatives(parameter, patch.domain_end_v());
        result += w[q] * ((interval.edge == PatchEdge3D::UMin || interval.edge == PatchEdge3D::UMax) ? d.dv.norm() : d.du.norm());
    }
    return half * result;
}

double exact_value(const Eigen::Vector3d& x)
{
    return std::exp(0.35 * x.x()) * std::cos(0.21 * x.y())
         * std::cos(std::sqrt(0.35 * 0.35 - 0.21 * 0.21) * x.z());
}

Eigen::VectorXd exact_density(const SurfaceDofCloud3D& cloud)
{
    Eigen::VectorXd values(static_cast<Eigen::Index>(cloud.dofs.size()));
    for (std::size_t i = 0; i < cloud.dofs.size(); ++i)
        values[static_cast<Eigen::Index>(i)] = exact_value(cloud.dofs[i].point);
    return values;
}

class SyntheticAugmentedOperator final : public kfbim::IKFBIOperator {
public:
    explicit SyntheticAugmentedOperator(const Eigen::VectorXd& mass)
        : mass_(mass), matrix_(Eigen::MatrixXd::Zero(mass.size(), mass.size()))
    {
        if (mass_.size() == 0 || !mass_.allFinite() || (mass_.array() <= 0.0).any())
            throw std::invalid_argument("synthetic mass is invalid");
        for (Eigen::Index row = 0; row < matrix_.rows(); ++row)
            for (Eigen::Index col = 0; col < matrix_.cols(); ++col)
                matrix_(row, col) = (row == col ? 1.7 : 0.0)
                    + 0.015 * std::sin(0.13 * (row + 1) * (col + 2));
    }

    int problem_size() const override { return static_cast<int>(mass_.size()) + 1; }
    const Eigen::MatrixXd& matrix() const { return matrix_; }

    void apply(const Eigen::VectorXd& unknown, Eigen::VectorXd& result) const override
    {
        if (unknown.size() != problem_size())
            throw std::invalid_argument("synthetic augmented input has invalid size");
        result.resize(problem_size());
        result.head(mass_.size()) = matrix_ * unknown.head(mass_.size())
            + unknown[mass_.size()] * Eigen::VectorXd::Ones(mass_.size());
        result[mass_.size()] = mass_.dot(unknown.head(mass_.size())) / mass_.sum();
    }

private:
    Eigen::VectorXd mass_;
    Eigen::MatrixXd matrix_;
};

void test_non_g1_topology_and_rows()
{
    const auto surface = make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const double h = 3.0 / 32.0;
    const auto cloud = make_native_surface_dofs_3d(surface, h);
    const auto constraints = build_neumann_edge_constraints_3d(surface, cloud, h);
    require(constraints.matrix.rows() == static_cast<Eigen::Index>(constraints.samples.size())
                && constraints.matrix.rows() == constraints.quadrature_weights.size()
                && constraints.matrix.cols() == static_cast<Eigen::Index>(cloud.dofs.size()),
            "constraint dimensions");
    std::map<int, int> row_counts;
    std::map<int, double> previous_second;
    const double gap_limit = 1.0e-11 * surface.geometry_model().control_bounds().diameter();
    for (const auto& sample : constraints.samples) {
        const auto& connection = surface.geometric_connections.at(sample.connection_index);
        require(!connection.g1, "G1 edge was constrained");
        require(sample.first_parameter >= connection.first.begin && sample.first_parameter <= connection.first.end
                    && sample.second_parameter >= connection.second.begin && sample.second_parameter <= connection.second.end,
                "partial interval not preserved");
        require(std::isfinite(sample.quadrature_weight) && sample.quadrature_weight > 0.0
                    && sample.mapped_point_gap <= gap_limit
                    && (edge_point(surface, connection.first, sample.first_parameter)
                        - edge_point(surface, connection.second, sample.second_parameter)).norm() <= gap_limit,
                "invalid edge sample geometry or quadrature");
        const int before = row_counts[sample.connection_index]++;
        require(sample.sample_index == before && sample.sample_count >= 2,
                "cell-center sample indexing");
        if (before > 0)
            require(connection.reversed ? sample.second_parameter < previous_second[sample.connection_index]
                                        : sample.second_parameter > previous_second[sample.connection_index],
                    "second-edge orientation");
        previous_second[sample.connection_index] = sample.second_parameter;
    }
    int non_g1 = 0;
    for (int c = 0; c < static_cast<int>(surface.geometric_connections.size()); ++c) {
        if (surface.geometric_connections[c].g1) require(row_counts[c] == 0, "G1 row count");
        else { ++non_g1; require(row_counts[c] == std::max(2, static_cast<int>(std::ceil(edge_length(surface, surface.geometric_connections[c].first) / h))), "non-G1 connection row count"); }
    }
    require(constraints.non_g1_connection_count == non_g1, "non-G1 count");
    require(apply_neumann_edge_constraints_3d(constraints,
                Eigen::VectorXd::Ones(constraints.density_size)).lpNorm<Eigen::Infinity>() <= 5.0e-13,
            "constant partition of unity");
}

void test_fallback_and_order()
{
    const auto surface = make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const auto coarse_cloud = make_native_surface_dofs_3d(surface, 10.0);
    const auto fallback = build_neumann_edge_constraints_3d(surface, coarse_cloud, 10.0);
    require(fallback.reduced_order_row_count > 0, "missing 2x2 fallback");
    for (int row = 0; row < fallback.matrix.outerSize(); ++row)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(fallback.matrix, row); it; ++it)
            require(it.col() >= 0 && it.col() < fallback.matrix.cols() && std::isfinite(it.value()), "fallback entry");
    const double h32 = 3.0 / 32.0, h64 = 3.0 / 64.0;
    const auto cloud32 = make_native_surface_dofs_3d(surface, h32);
    const auto cloud64 = make_native_surface_dofs_3d(surface, h64);
    const double m32 = neumann_edge_mismatch_weighted_rms_3d(
        build_neumann_edge_constraints_3d(surface, cloud32, h32), exact_density(cloud32));
    const double m64 = neumann_edge_mismatch_weighted_rms_3d(
        build_neumann_edge_constraints_3d(surface, cloud64, h64), exact_density(cloud64));
    require(std::isfinite(m32) && std::isfinite(m64) && m32 > 0.0 && m64 > 0.0 && m32 / m64 >= 6.0,
            "exact trace mismatch is not at least third-order");
}

void test_surface_mass_projector()
{
    const auto surface = make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const double h = 3.0 / 32.0;
    const auto cloud = make_native_surface_dofs_3d(surface, h);
    const auto constraints = build_neumann_edge_constraints_3d(surface, cloud, h);
    require(constraints.quadrature_weights.size() == constraints.matrix.rows(),
            "public edge quadrature weights have invalid size");
    for (Eigen::Index row = 0; row < constraints.quadrature_weights.size(); ++row) {
        require(std::isfinite(constraints.quadrature_weights[row])
                    && constraints.quadrature_weights[row] > 0.0,
                "public edge quadrature weight is invalid");
        require(std::abs(constraints.quadrature_weights[row]
                         - constraints.samples[static_cast<std::size_t>(row)].quadrature_weight)
                    <= 1.0e-15 * constraints.quadrature_weights[row],
                "public edge quadrature weights do not match samples");
    }
    Eigen::VectorXd mass(static_cast<Eigen::Index>(cloud.dofs.size()));
    for (std::size_t q = 0; q < cloud.dofs.size(); ++q) {
        mass[static_cast<Eigen::Index>(q)] = cloud.dofs[q].weight;
        require(std::isfinite(mass[static_cast<Eigen::Index>(q)])
                    && mass[static_cast<Eigen::Index>(q)] > 0.0,
                "surface mass is invalid");
    }

    NeumannEdgeContinuityProjector3D projector(constraints, cloud, 1.0e-12);
    Eigen::VectorXd x(static_cast<Eigen::Index>(cloud.dofs.size()));
    for (Eigen::Index q = 0; q < x.size(); ++q)
        x[q] = std::sin(0.37 * (q + 1)) + 0.2 * std::cos(0.11 * (q + 1));
    const Eigen::VectorXd px = projector.project(x);
    const double scale = std::max(1.0, x.lpNorm<Eigen::Infinity>());
    require(apply_neumann_edge_constraints_3d(constraints, px).lpNorm<Eigen::Infinity>() / scale <= 1.0e-11,
            "projected density violates edge constraints");
    require((projector.project(px) - px).lpNorm<Eigen::Infinity>() / scale <= 1.0e-11,
            "projector is not idempotent");
    require((projector.project(Eigen::VectorXd::Ones(x.size()))
             - Eigen::VectorXd::Ones(x.size())).lpNorm<Eigen::Infinity>() <= 1.0e-11,
            "projector does not preserve constants");
    require(projector.retained_rank() > 0
                && projector.retained_rank() <= projector.constraint_count()
                && projector.constraint_count() == constraints.matrix.rows(),
            "projector rank diagnostics are invalid");
    const Eigen::VectorXd complement = x - px;
    const double weighted_orthogonality = px.dot(mass.cwiseProduct(complement));
    const double weighted_scale = std::sqrt(px.dot(mass.cwiseProduct(px))
        * complement.dot(mass.cwiseProduct(complement)));
    require((projector.complement(x) - complement).lpNorm<Eigen::Infinity>() <= 1.0e-13 * scale,
            "projector complement is inconsistent");
    require(std::abs(weighted_orthogonality) <= 1.0e-11 * std::max(1.0, weighted_scale),
            "projector is not orthogonal in the surface-mass metric");
    const Eigen::VectorXd direct = constraints.matrix * x;
    const double direct_linf = direct.lpNorm<Eigen::Infinity>();
    const double direct_rms = std::sqrt((constraints.quadrature_weights.array()
                                         * direct.array().square()).sum()
                                        / constraints.quadrature_weights.sum());
    require((projector.constraint_mismatch(x) - direct).lpNorm<Eigen::Infinity>() <= 1.0e-13 * scale
                && std::abs(projector.mismatch_linf(x) - direct_linf) <= 1.0e-13 * scale
                && std::abs(projector.mismatch_weighted_rms(x) - direct_rms) <= 1.0e-13 * scale,
            "projector mismatch helpers disagree with direct constraints");
    bool bad_size_threw = false;
    try { projector.project(Eigen::VectorXd::Zero(x.size() - 1)); }
    catch (const std::invalid_argument&) { bad_size_threw = true; }
    require(bad_size_threw, "projector accepts an invalid density size");

    NeumannEdgeConstraintSet3D empty;
    empty.density_size = static_cast<int>(x.size());
    empty.matrix.resize(0, x.size());
    NeumannEdgeContinuityProjector3D identity(empty, cloud);
    require((identity.project(x) - x).lpNorm<Eigen::Infinity>() <= 1.0e-13 * scale
                && identity.retained_rank() == 0,
            "zero-row projector is not the identity");

    NeumannEdgeConstraintSet3D redundant;
    redundant.density_size = static_cast<int>(x.size());
    redundant.matrix.resize(2, x.size());
    const std::vector<Eigen::Triplet<double>> redundant_entries = {
        {0, 0, 1.0}, {0, 1, -1.0}, {1, 0, 2.0}, {1, 1, -2.0}};
    redundant.matrix.setFromTriplets(redundant_entries.begin(), redundant_entries.end());
    redundant.quadrature_weights = Eigen::Vector2d(0.7, 1.3);
    NeumannEdgeContinuityProjector3D rank_filtered(redundant, cloud);
    const Eigen::VectorXd rank_filtered_x = rank_filtered.project(x);
    require(rank_filtered.retained_rank() == 1
                && (redundant.matrix * rank_filtered_x).lpNorm<Eigen::Infinity>() <= 1.0e-12 * scale,
            "rank-filtered projector does not satisfy dropped physical rows");
}

void test_projected_augmented_operator()
{
    const auto surface = make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const double h = 3.0 / 32.0;
    const auto cloud = make_native_surface_dofs_3d(surface, h);
    const auto constraints = build_neumann_edge_constraints_3d(surface, cloud, h);
    NeumannEdgeContinuityProjector3D projector(constraints, cloud, 1.0e-12);
    Eigen::VectorXd mass(static_cast<Eigen::Index>(cloud.dofs.size()));
    for (std::size_t q = 0; q < cloud.dofs.size(); ++q)
        mass[static_cast<Eigen::Index>(q)] = cloud.dofs[q].weight;
    SyntheticAugmentedOperator base(mass);
    NeumannEdgeProjectedAugmentedOperator3D wrapper(base, projector);
    const Eigen::Index n = mass.size();
    Eigen::VectorXd unknown(n + 1);
    for (Eigen::Index q = 0; q < n; ++q)
        unknown[q] = std::sin(0.19 * (q + 1)) - 0.3 * std::cos(0.07 * (q + 1));
    unknown[n] = -0.45;
    Eigen::VectorXd baseline, after;
    base.apply(unknown, baseline);
    Eigen::VectorXd result;
    wrapper.apply(unknown, result);
    base.apply(unknown, after);
    Eigen::VectorXd projected_unknown = unknown;
    projected_unknown.head(n) = projector.project(unknown.head(n));
    Eigen::VectorXd base_top = base.matrix() * projected_unknown.head(n)
        + projected_unknown[n] * Eigen::VectorXd::Ones(n);
    Eigen::VectorXd expected(n + 1);
    expected.head(n) = projector.project(base_top)
        + unknown.head(n) - projected_unknown.head(n);
    expected[n] = mass.dot(projected_unknown.head(n)) / mass.sum();
    require((result - expected).lpNorm<Eigen::Infinity>() <= 1.0e-11,
            "projected augmented wrapper has incorrect algebra");
    require((baseline - after).lpNorm<Eigen::Infinity>() <= 1.0e-13,
            "projected augmented wrapper mutates the base operator");
    const Eigen::VectorXd excluded = projector.complement(unknown.head(n));
    Eigen::VectorXd excluded_unknown = Eigen::VectorXd::Zero(n + 1);
    excluded_unknown.head(n) = excluded;
    Eigen::VectorXd excluded_result;
    wrapper.apply(excluded_unknown, excluded_result);
    require((excluded_result.head(n) - excluded).lpNorm<Eigen::Infinity>() <= 1.0e-11,
            "wrapper does not retain an excluded density component");
    const Eigen::VectorXd projected_rhs = wrapper.project_right_hand_side(unknown);
    require((projected_rhs.head(n) - projector.project(unknown.head(n))).lpNorm<Eigen::Infinity>() <= 1.0e-11
                && projected_rhs[n] == unknown[n],
            "projected augmented RHS is incorrect");
    bool bad_apply_threw = false, bad_rhs_threw = false, bad_base_threw = false;
    try { wrapper.apply(Eigen::VectorXd::Zero(n), result); }
    catch (const std::invalid_argument&) { bad_apply_threw = true; }
    try { wrapper.project_right_hand_side(Eigen::VectorXd::Zero(n)); }
    catch (const std::invalid_argument&) { bad_rhs_threw = true; }
    try {
        class WrongSizeOperator final : public kfbim::IKFBIOperator {
        public:
            int problem_size() const override { return 1; }
            void apply(const Eigen::VectorXd&, Eigen::VectorXd&) const override {}
        } wrong;
        NeumannEdgeProjectedAugmentedOperator3D invalid_wrapper(wrong, projector);
    } catch (const std::invalid_argument&) { bad_base_threw = true; }
    require(bad_apply_threw && bad_rhs_threw && bad_base_threw,
            "projected augmented wrapper accepts invalid sizes");
}
}

namespace {
NeumannEdgeContinuityMeasurement3D passing_measurement(
    const std::string& case_id, int N, NeumannDensitySpace3D density_space)
{
    const bool projected = density_space == NeumannDensitySpace3D::NonG1EdgeProjected;
    const double coarse = N == 32 ? 1.0 : 0.5;
    NeumannEdgeContinuityMeasurement3D row;
    row.case_id = case_id;
    row.N = N;
    row.h = 1.0 / static_cast<double>(N);
    row.density_space = density_space;
    row.finite_metrics = true;
    row.gmres_converged = true;
    row.gmres_iterations = projected ? (case_id == "ty_m0083" ? 37 : 31) : (case_id == "ty_m0083" ? 48 : 42);
    row.gmres_relative_residual = 1.0e-11;
    const double error_scale = projected ? 0.8 : 1.0;
    row.density_linf = error_scale * coarse;
    row.density_l2 = error_scale * 2.0 * coarse;
    row.interior_linf = error_scale * 3.0 * coarse;
    row.interior_l2 = error_scale * 4.0 * coarse;
    row.edge_mismatch_linf = projected ? 1.0e-8 * coarse : 1.0e-3 * coarse;
    row.edge_mismatch_weighted_rms = projected ? 2.0e-8 * coarse : 2.0e-3 * coarse;
    row.exact_edge_mismatch_linf = coarse * coarse * coarse;
    row.expected_non_g1_connections = 4;
    row.covered_non_g1_connections = 4;
    row.constraint_rows = 12;
    row.constraint_rank = 10;
    row.constant_constraint_defect = 1.0e-12;
    row.projected_constraint_defect = 1.0e-12;
    row.projection_idempotence_defect = 1.0e-12;
    row.constant_projection_defect = 1.0e-12;
    row.geometry_diagnostics_pass = true;
    row.owner_invariants_pass = true;
    row.shared_preprocess_pass = true;
    return row;
}

std::vector<NeumannEdgeContinuityMeasurement3D> passing_measurements(bool include_64 = true)
{
    std::vector<NeumannEdgeContinuityMeasurement3D> rows;
    for (const std::string& case_id : {std::string("baseline"), std::string("ty_m0083"), std::string("rot_axis123_17deg")}) {
        for (int N : {32, 64}) {
            if (!include_64 && N == 64) continue;
            rows.push_back(passing_measurement(case_id, N, NeumannDensitySpace3D::PatchIndependent));
            rows.push_back(passing_measurement(case_id, N, NeumannDensitySpace3D::NonG1EdgeProjected));
        }
    }
    return rows;
}

const NeumannEdgeContinuityDerivedRow3D& find_evaluation_row(
    const NeumannEdgeContinuityEvaluation3D& evaluation, const std::string& case_id,
    int N, NeumannDensitySpace3D density_space)
{
    for (const auto& row : evaluation.rows)
        if (row.measurement.case_id == case_id && row.measurement.N == N && row.measurement.density_space == density_space) return row;
    throw std::runtime_error("missing evaluated measurement");
}

void require_throws(const std::function<void()>& work, const std::string& message)
{
    bool threw = false;
    try { work(); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, message);
}

void test_edge_study_level_prefixes()
{
    require(normalize_neumann_edge_continuity_levels_3d({}) == std::vector<int>({32, 64}), "empty edge-study levels did not select the pilot default");
    require(normalize_neumann_edge_continuity_levels_3d({32}) == std::vector<int>({32}), "N=32 edge-study prefix was rejected");
    require(normalize_neumann_edge_continuity_levels_3d({32, 64}) == std::vector<int>({32, 64}), "N=32,64 edge-study prefix was rejected");
    require(normalize_neumann_edge_continuity_levels_3d({32, 64, 128}) == std::vector<int>({32, 64, 128}), "N=128 prefix was rejected");
    require_throws([] { normalize_neumann_edge_continuity_levels_3d({64}); }, "N=64 did not require N=32");
    require_throws([] { normalize_neumann_edge_continuity_levels_3d({32, 128}); }, "N=128 did not require N=64");
    require_throws([] { normalize_neumann_edge_continuity_levels_3d({16, 32}); }, "invalid edge-study level was accepted");
}

void test_edge_study_acceptance_and_failure_gates()
{
    using Status = RigidStudyCriterionStatus3D;
    const std::vector<std::string> cases = {"baseline", "ty_m0083", "rot_axis123_17deg"};
    const auto passing = passing_measurements();
    const auto evaluation = evaluate_neumann_edge_continuity_study_3d(passing, cases, true);
    require(evaluation.all_pass, "complete paired edge study did not pass");
    for (Status status : {evaluation.acceptance.completeness_pass, evaluation.acceptance.topology_pass, evaluation.acceptance.projector_pass, evaluation.acceptance.exact_trace_order_pass, evaluation.acceptance.gmres_pass, evaluation.acceptance.error_guard_pass, evaluation.acceptance.edge_reduction_pass, evaluation.acceptance.trend_pass, evaluation.acceptance.geometry_owner_pass, evaluation.acceptance.overall_pass})
        require(status == Status::Pass, "passing edge-study gate was not pass");
    const auto& projected64 = find_evaluation_row(evaluation, "baseline", 64, NeumannDensitySpace3D::NonG1EdgeProjected);
    require(std::abs(projected64.density_linf_order - 1.0) <= 1.0e-14 && std::abs(projected64.exact_edge_mismatch_order - 3.0) <= 1.0e-14 && std::abs(projected64.edge_reduction_ratio - 1.0e5) <= 1.0e-5, "paired edge-study derived values are incorrect");

    auto missing = passing;
    missing.pop_back();
    const auto incomplete = evaluate_neumann_edge_continuity_study_3d(missing, cases, true);
    require(incomplete.acceptance.completeness_pass == Status::NotEvaluated && incomplete.acceptance.overall_pass == Status::NotEvaluated && !incomplete.all_pass, "missing case/level/mode row was hidden");
    auto duplicate = passing;
    duplicate.push_back(duplicate.front());
    require_throws([&] { evaluate_neumann_edge_continuity_study_3d(duplicate, cases, true); }, "duplicate case/N/density-space row was accepted");

    auto topology = passing;
    topology.front().covered_non_g1_connections = 3;
    require(evaluate_neumann_edge_continuity_study_3d(topology, cases, true).acceptance.topology_pass == Status::Fail, "missing non-G1 interval was hidden");
    topology = passing;
    topology.front().duplicate_connection_intervals = 1;
    require(evaluate_neumann_edge_continuity_study_3d(topology, cases, true).acceptance.topology_pass == Status::Fail, "duplicate non-G1 interval was hidden");
    topology = passing;
    topology.front().g1_constraint_rows = 1;
    require(evaluate_neumann_edge_continuity_study_3d(topology, cases, true).acceptance.topology_pass == Status::Fail, "G1 constraint row was hidden");
    topology = passing;
    topology.front().unrelated_constraint_rows = 1;
    require(evaluate_neumann_edge_continuity_study_3d(topology, cases, true).acceptance.topology_pass == Status::Fail, "unrelated constraint row was hidden");

    auto projector = passing;
    projector.front().projection_idempotence_defect = 1.1e-11;
    require(evaluate_neumann_edge_continuity_study_3d(projector, cases, true).acceptance.projector_pass == Status::Fail, "normalized projector defect was hidden");
    auto exact = passing;
    for (auto& row : exact) if (row.case_id == "baseline" && row.N == 64) row.exact_edge_mismatch_linf *= 2.0;
    require(evaluate_neumann_edge_continuity_study_3d(exact, cases, true).acceptance.exact_trace_order_pass == Status::Fail, "sub-six exact mismatch ratio was hidden");
    exact = passing;
    for (auto& row : exact) if (row.case_id == "baseline" && row.N == 64) row.exact_edge_mismatch_linf = 0.0;
    require(evaluate_neumann_edge_continuity_study_3d(exact, cases, true).acceptance.exact_trace_order_pass == Status::Fail,
            "zero fine exact mismatch was treated as a passing ratio");

    auto gmres = passing;
    for (auto& row : gmres) if (row.density_space == NeumannDensitySpace3D::NonG1EdgeProjected) row.gmres_iterations = 60;
    require(evaluate_neumann_edge_continuity_study_3d(gmres, cases, true).acceptance.gmres_pass == Status::Fail, "projected worst GMRES increase was hidden");
    gmres = passing;
    for (auto& row : gmres) if (row.case_id == "ty_m0083") row.gmres_iterations = 48;
    require(evaluate_neumann_edge_continuity_study_3d(gmres, cases, true).acceptance.gmres_pass == Status::Fail, "missing ty_m0083 GMRES decrease was hidden");

    auto errors = passing;
    for (auto& row : errors) if (row.case_id == "baseline" && row.N == 32 && row.density_space == NeumannDensitySpace3D::NonG1EdgeProjected) row.interior_l2 = 4.5;
    require(evaluate_neumann_edge_continuity_study_3d(errors, cases, true).acceptance.error_guard_pass == Status::Fail, "projected error guard was hidden");
    auto edge = passing;
    for (auto& row : edge) if (row.case_id == "baseline" && row.N == 32 && row.density_space == NeumannDensitySpace3D::NonG1EdgeProjected) row.edge_mismatch_linf = 1.1e-7;
    require(evaluate_neumann_edge_continuity_study_3d(edge, cases, true).acceptance.edge_reduction_pass == Status::Fail, "insufficient edge mismatch reduction was hidden");
    auto trend = passing;
    for (auto& row : trend) if (row.case_id == "baseline" && row.N == 64 && row.density_space == NeumannDensitySpace3D::NonG1EdgeProjected) row.density_linf = 0.45;
    require(evaluate_neumann_edge_continuity_study_3d(trend, cases, true).acceptance.trend_pass == Status::Fail, "worse projected refinement trend was hidden");
    auto geometry = passing;
    geometry.front().shared_preprocess_pass = false;
    require(evaluate_neumann_edge_continuity_study_3d(geometry, cases, true).acceptance.geometry_owner_pass == Status::Fail, "changed geometry/owner/shared-preprocess invariant was hidden");
}

void test_edge_study_n32_smoke_keeps_two_level_gates_not_evaluated()
{
    using Status = RigidStudyCriterionStatus3D;
    const auto evaluation = evaluate_neumann_edge_continuity_study_3d(passing_measurements(false), {"baseline", "ty_m0083", "rot_axis123_17deg"}, false);
    require(evaluation.all_pass && evaluation.acceptance.overall_pass == Status::Pass, "valid N=32 edge smoke was not execution-clean");
    require(evaluation.acceptance.exact_trace_order_pass == Status::NotEvaluated && evaluation.acceptance.trend_pass == Status::NotEvaluated, "N=32 edge smoke falsely passed two-level numerical criteria");
}
}

int main()
{
    try {
        test_non_g1_topology_and_rows();
        test_fallback_and_order();
        test_surface_mass_projector();
        test_projected_augmented_operator();
        test_edge_study_level_prefixes();
        test_edge_study_acceptance_and_failure_gates();
        test_edge_study_n32_smoke_keeps_two_level_gates_not_evaluated();
        std::cout << "3D Neumann edge-continuity tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "3D Neumann edge-continuity test failure: " << error.what() << '\n';
        return 1;
    }
}
