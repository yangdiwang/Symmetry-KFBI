#include "neumann_edge_cauchy_study_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace kfbim::app3d;
using Status = RigidStudyCriterionStatus3D;
const std::vector<std::string> kCases{
    "baseline", "ty_m0083", "rot_axis123_17deg"};

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void require_throws(const std::function<void()>& work,
                    const std::string& message)
{
    bool threw = false;
    try { work(); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, message);
}

int case_index(const std::string& case_id)
{
    const auto found = std::find(kCases.begin(), kCases.end(), case_id);
    if (found == kCases.end()) throw std::logic_error("unknown fixture case");
    return static_cast<int>(found - kCases.begin());
}

NeumannEdgeCauchyMeasurement3D passing_measurement(
    const std::string& case_id, int N, NeumannEdgeCauchyMode3D mode)
{
    const bool augmented = mode == NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues;
    const double refinement = N == 32 ? 1.0 : N == 64 ? 0.25 : 0.0625;
    const double pose_scale = 1.0 + 0.2 * case_index(case_id);
    const double error_scale = refinement * pose_scale * (augmented ? 0.90 : 1.0);
    NeumannEdgeCauchyMeasurement3D row;
    row.case_id = case_id;
    row.N = N;
    row.h = 1.0 / N;
    row.mode = mode;
    row.finite_metrics = true;
    row.gmres_converged = true;
    const int pose = case_index(case_id);
    row.gmres_iterations = augmented ? 36 + 2 * pose : 40 + 4 * pose;
    row.gmres_relative_residual = 1.0e-11;
    row.density_linf = error_scale;
    row.density_l2 = 2.0 * error_scale;
    row.interior_linf = 3.0 * error_scale;
    row.interior_l2 = 4.0 * error_scale;
    row.incident_edge_discrepancy_linf =
        refinement * (augmented ? 2.0e-5 : 8.0e-5);
    row.expected_non_g1_connections = 4;
    row.covered_non_g1_connections = 4;
    row.edge_sample_count = 16;
    row.affected_center_count = 12;
    row.corner_center_count = 3;
    row.unrelated_sample_or_attachment_count = 0;
    row.rank_deficient_fit_count = 0;
    row.harmonic_cubic_reproduction_defect = 1.0e-12;
    row.edge_condition_max = 2.0e4;
    row.local_condition_max = 3.0e4;
    row.shared_setup_seconds = 1.0;
    row.mode_runtime_seconds = 2.0;
    row.total_seconds = 3.0;
    row.far_centers_bitwise_legacy = true;
    row.geometry_diagnostics_pass = true;
    row.owner_invariants_pass = true;
    row.shared_preprocess_pass = true;
    return row;
}

std::vector<NeumannEdgeCauchyMeasurement3D> passing_measurements(
    bool include_64 = true, bool include_128 = false)
{
    std::vector<NeumannEdgeCauchyMeasurement3D> rows;
    for (const std::string& case_id : kCases) {
        for (int N : {32, 64, 128}) {
            if (N == 64 && !include_64) continue;
            if (N == 128 && !include_128) continue;
            for (NeumannEdgeCauchyMode3D mode : {
                     NeumannEdgeCauchyMode3D::None,
                     NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues}) {
                rows.push_back(passing_measurement(case_id, N, mode));
            }
        }
    }
    return rows;
}

const NeumannEdgeCauchyDerivedRow3D& find_row(
    const NeumannEdgeCauchyEvaluation3D& evaluation,
    const std::string& case_id, int N, NeumannEdgeCauchyMode3D mode)
{
    const auto found = std::find_if(evaluation.rows.begin(), evaluation.rows.end(),
        [&](const NeumannEdgeCauchyDerivedRow3D& row) {
            return row.measurement.case_id == case_id
                && row.measurement.N == N && row.measurement.mode == mode;
        });
    if (found == evaluation.rows.end()) throw std::runtime_error("missing evaluated row");
    return *found;
}

void test_edge_value_row_finiteness()
{
    const std::array<double, 6> clean{{1.0, 2.0, 3.0, 1.0, 2.0, 1.0}};
    require(neumann_edge_cauchy_edge_value_row_finite_3d(clean),
            "finite edge-value row was rejected");
    for (std::size_t q = 0; q < clean.size(); ++q) {
        auto changed = clean;
        changed[q] = std::numeric_limits<double>::quiet_NaN();
        require(!neumann_edge_cauchy_edge_value_row_finite_3d(changed),
                "NaN edge-value component was accepted");
        changed[q] = std::numeric_limits<double>::infinity();
        require(!neumann_edge_cauchy_edge_value_row_finite_3d(changed),
                "infinite edge-value component was accepted");
    }
}

void test_level_prefixes()
{
    require(normalize_neumann_edge_cauchy_levels_3d({}) == std::vector<int>({32, 64}),
            "empty levels did not select 32,64");
    require(normalize_neumann_edge_cauchy_levels_3d({32}) == std::vector<int>({32}),
            "N=32 prefix was rejected");
    require(normalize_neumann_edge_cauchy_levels_3d({64, 32, 32}) == std::vector<int>({32, 64}),
            "N=32,64 prefix was not normalized");
    require(normalize_neumann_edge_cauchy_levels_3d({128, 64, 32}) == std::vector<int>({32, 64, 128}),
            "N=32,64,128 prefix was not normalized");
    require_throws([] { normalize_neumann_edge_cauchy_levels_3d({64}); },
                   "N=64 did not require N=32");
    require_throws([] { normalize_neumann_edge_cauchy_levels_3d({32, 128}); },
                   "N=128 did not require N=64");
    require_throws([] { normalize_neumann_edge_cauchy_levels_3d({16, 32}); },
                   "invalid N was accepted");
}

void test_input_keys_and_prefix_semantics()
{
    const auto passing = passing_measurements();
    auto missing = passing;
    missing.pop_back();
    const auto required = evaluate_neumann_edge_cauchy_study_3d(missing, kCases, true);
    require(required.acceptance.completeness_pass == Status::Fail
                && required.acceptance.overall_pass == Status::Fail && !required.all_pass,
            "required pilot did not fail a missing coarse key");

    const auto prefix = evaluate_neumann_edge_cauchy_study_3d(
        passing_measurements(false), kCases, false);
    require(prefix.acceptance.completeness_pass == Status::NotEvaluated
                && prefix.acceptance.structure_pass == Status::Pass
                && prefix.acceptance.reproduction_pass == Status::Pass
                && prefix.acceptance.gmres_pass == Status::Pass
                && prefix.acceptance.error_guard_pass == Status::Pass
                && prefix.acceptance.order_pass == Status::NotEvaluated
                && prefix.acceptance.rigid_spread_pass == Status::Pass
                && prefix.acceptance.edge_discrepancy_pass == Status::Pass
                && prefix.acceptance.geometry_owner_pass == Status::Pass
                && prefix.acceptance.extended_evidence_pass == Status::NotEvaluated
                && neumann_edge_cauchy_study_exit_pass_3d(prefix, false),
            "valid N=32 prefix did not preserve execution gates");

    auto duplicate = passing;
    duplicate.push_back(duplicate.front());
    require_throws([&] { evaluate_neumann_edge_cauchy_study_3d(duplicate, kCases, true); },
                   "duplicate case/N/mode key was accepted");
    auto unknown = passing;
    unknown.front().case_id = "unknown";
    require_throws([&] { evaluate_neumann_edge_cauchy_study_3d(unknown, kCases, true); },
                   "unknown case ID was accepted");
    auto invalid_level = passing;
    invalid_level.front().N = 16;
    require_throws([&] { evaluate_neumann_edge_cauchy_study_3d(invalid_level, kCases, true); },
                   "invalid measurement level was accepted");
    auto invalid_mode = passing;
    invalid_mode.front().mode = static_cast<NeumannEdgeCauchyMode3D>(99);
    require_throws([&] { evaluate_neumann_edge_cauchy_study_3d(invalid_mode, kCases, true); },
                   "invalid measurement mode was accepted");
}

void test_passing_fixture_and_derived_values()
{
    const auto evaluation = evaluate_neumann_edge_cauchy_study_3d(
        passing_measurements(), kCases, true);
    require(evaluation.all_pass && neumann_edge_cauchy_study_exit_pass_3d(evaluation, true),
            "literal complete coarse fixture did not pass");
    for (Status status : {evaluation.acceptance.completeness_pass,
                          evaluation.acceptance.structure_pass,
                          evaluation.acceptance.reproduction_pass,
                          evaluation.acceptance.gmres_pass,
                          evaluation.acceptance.error_guard_pass,
                          evaluation.acceptance.order_pass,
                          evaluation.acceptance.rigid_spread_pass,
                          evaluation.acceptance.edge_discrepancy_pass,
                          evaluation.acceptance.geometry_owner_pass,
                          evaluation.acceptance.overall_pass})
        require(status == Status::Pass, "passing coarse gate was not Pass");
    require(evaluation.acceptance.extended_evidence_pass == Status::NotEvaluated,
            "missing N=128 evidence was evaluated");
    const auto& row = find_row(evaluation, "baseline", 64,
        NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues);
    require(std::abs(row.density_linf_order - 2.0) <= 1.0e-14
                && std::abs(row.density_linf_ratio_to_legacy - 0.90) <= 1.0e-14
                && std::abs(row.edge_discrepancy_ratio_to_legacy - 0.25) <= 1.0e-14,
            "derived order or ratios are incorrect");
}

void test_coarse_gate_mutations()
{
    const auto passing = passing_measurements();
    const auto gate = [&](const auto& mutate,
        Status NeumannEdgeCauchyAcceptance3D::*member,
        const std::string& message) {
        auto rows = passing;
        mutate(rows);
        const auto evaluation = evaluate_neumann_edge_cauchy_study_3d(rows, kCases, true);
        require(evaluation.acceptance.*member == Status::Fail
                    && evaluation.acceptance.overall_pass == Status::Fail
                    && !evaluation.all_pass
                    && !neumann_edge_cauchy_study_exit_pass_3d(
                        evaluation, true),
                message);
    };
    gate([](auto& r) { r.front().covered_non_g1_connections = 3; },
         &NeumannEdgeCauchyAcceptance3D::structure_pass,
         "uncovered connection was hidden");
    gate([](auto& r) { r.front().unrelated_sample_or_attachment_count = 1; },
         &NeumannEdgeCauchyAcceptance3D::structure_pass,
         "unrelated attachment was hidden");
    gate([](auto& r) { r.front().far_centers_bitwise_legacy = false; },
         &NeumannEdgeCauchyAcceptance3D::structure_pass,
         "far-center mutation was hidden");
    gate([](auto& r) { r.front().harmonic_cubic_reproduction_defect = 1.01e-11; },
         &NeumannEdgeCauchyAcceptance3D::reproduction_pass,
         "reproduction defect was hidden");
    gate([](auto& r) { r.front().gmres_converged = false; },
         &NeumannEdgeCauchyAcceptance3D::gmres_pass,
         "nonconvergence was hidden");
    gate([](auto& r) { r.front().gmres_relative_residual = 2.01e-10; },
         &NeumannEdgeCauchyAcceptance3D::gmres_pass,
         "GMRES residual was hidden");
    gate([](auto& r) { r.front().gmres_iterations = 81; },
         &NeumannEdgeCauchyAcceptance3D::gmres_pass,
         "GMRES iteration cap was hidden");
    gate([](auto& r) { for (auto& x : r) if (x.mode == NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) x.gmres_iterations = 49; },
         &NeumannEdgeCauchyAcceptance3D::gmres_pass,
         "augmented worst GMRES increase was hidden");
    gate([](auto& r) { for (auto& x : r) if (x.case_id == "baseline" && x.N == 32 && x.mode == NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) x.interior_l2 = 4.41; },
         &NeumannEdgeCauchyAcceptance3D::error_guard_pass,
         "error ratio above 1.10 was hidden");
    gate([](auto& r) { for (auto& x : r) if (x.case_id == "baseline" && x.N == 64 && x.mode == NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) x.density_linf = 0.27; },
         &NeumannEdgeCauchyAcceptance3D::order_pass,
         "augmented order below 1.8 was hidden");
    gate([](auto& r) { for (auto& x : r) if (x.case_id == "rot_axis123_17deg" && x.N == 32 && x.mode == NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) x.density_linf = 1.5; },
         &NeumannEdgeCauchyAcceptance3D::rigid_spread_pass,
         "increased error spread was hidden");
    gate([](auto& r) { for (auto& x : r) if (x.case_id == "rot_axis123_17deg" && x.N == 32 && x.mode == NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) x.gmres_iterations = 45; },
         &NeumannEdgeCauchyAcceptance3D::rigid_spread_pass,
         "increased iteration spread was hidden");
    gate([](auto& r) { for (auto& x : r) if (x.case_id == "baseline" && x.N == 32 && x.mode == NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) x.incident_edge_discrepancy_linf = 8.0e-5; },
         &NeumannEdgeCauchyAcceptance3D::edge_discrepancy_pass,
         "non-reduced edge discrepancy was hidden");
    gate([](auto& r) { r.front().geometry_diagnostics_pass = false; },
         &NeumannEdgeCauchyAcceptance3D::geometry_owner_pass,
         "geometry failure was hidden");
    gate([](auto& r) { r.front().owner_invariants_pass = false; },
         &NeumannEdgeCauchyAcceptance3D::geometry_owner_pass,
         "owner failure was hidden");
    gate([](auto& r) { r.front().shared_preprocess_pass = false; },
         &NeumannEdgeCauchyAcceptance3D::geometry_owner_pass,
         "shared-preprocess failure was hidden");
}

void test_exact_thresholds_and_ratio_allowance()
{
    auto rows = passing_measurements();
    rows.front().gmres_iterations = 80;
    rows.front().gmres_relative_residual = 2.0e-10;
    rows.front().harmonic_cubic_reproduction_defect = 1.0e-11;
    for (auto& row : rows) {
        if (row.case_id == "baseline"
            && row.mode
                == NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) {
            if (row.N == 32) row.interior_l2 = 4.4;
            if (row.N == 64)
                row.density_linf = 0.9 / std::pow(2.0, 1.8);
        }
    }
    const auto exact = evaluate_neumann_edge_cauchy_study_3d(
        rows, kCases, true);
    require(exact.acceptance.reproduction_pass == Status::Pass
                && exact.acceptance.gmres_pass == Status::Pass
                && exact.acceptance.error_guard_pass == Status::Pass
                && exact.acceptance.order_pass == Status::Pass
                && exact.acceptance.overall_pass == Status::Pass,
            "an inclusive exact threshold was rejected");

    rows = passing_measurements();
    const double allowance =
        64.0 * std::numeric_limits<double>::epsilon();
    for (auto& row : rows) {
        if (row.case_id == "baseline" && row.N == 32
            && row.mode
                == NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) {
            row.interior_l2 = 4.0 * 1.10 * (1.0 + 0.5 * allowance);
            require(row.interior_l2 / 4.0 > 1.10,
                    "roundoff-allowance fixture did not exceed 1.10");
        }
    }
    const auto allowed = evaluate_neumann_edge_cauchy_study_3d(
        rows, kCases, true);
    require(allowed.acceptance.error_guard_pass == Status::Pass
                && allowed.acceptance.overall_pass == Status::Pass,
            "64-epsilon relative ratio allowance was not honored");
}

void test_failed_n128_evidence_is_isolated()
{
    auto rows = passing_measurements();
    auto legacy = passing_measurement(
        "baseline", 128, NeumannEdgeCauchyMode3D::None);
    auto augmented = passing_measurement(
        "baseline", 128,
        NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues);
    legacy.finite_metrics = false;
    augmented.finite_metrics = false;
    rows.push_back(legacy);
    rows.push_back(augmented);
    const auto evaluation = evaluate_neumann_edge_cauchy_study_3d(
        rows, kCases, true);
    require(evaluation.acceptance.extended_evidence_pass == Status::Fail
                && evaluation.acceptance.overall_pass == Status::Pass
                && evaluation.all_pass
                && neumann_edge_cauchy_study_exit_pass_3d(
                    evaluation, true),
            "explicit failed N=128 evidence changed coarse process success");
}

void test_n128_isolation()
{
    auto rows = passing_measurements(true, true);
    const auto clean = evaluate_neumann_edge_cauchy_study_3d(rows, kCases, true);
    require(clean.acceptance.extended_evidence_pass == Status::Pass
                && clean.acceptance.overall_pass == Status::Pass,
            "clean N=128 evidence did not pass");
    for (auto& row : rows) {
        if (row.N != 128 || row.mode != NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) continue;
        const auto legacy = std::find_if(rows.begin(), rows.end(), [&](const auto& x) {
            return x.case_id == row.case_id && x.N == 128
                && x.mode == NeumannEdgeCauchyMode3D::None;
        });
        require(legacy != rows.end(), "missing N=128 legacy fixture");
        row.gmres_iterations = 80;
        row.density_linf = 2.0 * legacy->density_linf;
        row.density_l2 = 2.0 * legacy->density_l2;
        row.interior_linf = 2.0 * legacy->interior_linf;
        row.interior_l2 = 2.0 * legacy->interior_l2;
        row.incident_edge_discrepancy_linf = legacy->incident_edge_discrepancy_linf;
    }
    const auto bad = evaluate_neumann_edge_cauchy_study_3d(rows, kCases, true);
    require(bad.acceptance.extended_evidence_pass == Status::Fail
                && bad.acceptance.overall_pass == Status::Pass && bad.all_pass
                && neumann_edge_cauchy_study_exit_pass_3d(bad, true),
            "failing N=128 evidence changed coarse acceptance or exit");
}
} // namespace

int main()
{
    try {
        test_edge_value_row_finiteness();
        test_level_prefixes();
        test_input_keys_and_prefix_semantics();
        test_passing_fixture_and_derived_values();
        test_coarse_gate_mutations();
        test_exact_thresholds_and_ratio_allowance();
        test_failed_n128_evidence_is_isolated();
        test_n128_isolation();
        std::cout << "3D Neumann edge-Cauchy study tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "3D Neumann edge-Cauchy study test failure: "
                  << error.what() << '\n';
        return 1;
    }
}