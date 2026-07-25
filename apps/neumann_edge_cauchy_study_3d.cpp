#include "neumann_edge_cauchy_study_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace kfbim::app3d {
namespace {

using Status = RigidStudyCriterionStatus3D;
using Key = std::tuple<std::string, int, NeumannEdgeCauchyMode3D>;
using RowMap = std::map<Key, const NeumannEdgeCauchyMeasurement3D*>;
constexpr double kRelativeAllowance =
    64.0 * std::numeric_limits<double>::epsilon();

Status status(bool pass)
{
    return pass ? Status::Pass : Status::Fail;
}

const NeumannEdgeCauchyMeasurement3D* measurement(
    const RowMap& rows, const std::string& case_id, int N,
    NeumannEdgeCauchyMode3D mode)
{
    const auto found = rows.find({case_id, N, mode});
    return found == rows.end() ? nullptr : found->second;
}

bool finite_nonnegative(double value)
{
    return std::isfinite(value) && value >= 0.0;
}

bool measurement_finite(const NeumannEdgeCauchyMeasurement3D& row)
{
    return row.pair_completed && row.finite_metrics && row.N > 0
        && std::isfinite(row.h)
        && row.h > 0.0 && finite_nonnegative(row.gmres_relative_residual)
        && finite_nonnegative(row.density_linf)
        && finite_nonnegative(row.density_l2)
        && finite_nonnegative(row.interior_linf)
        && finite_nonnegative(row.interior_l2)
        && finite_nonnegative(row.incident_edge_discrepancy_linf)
        && finite_nonnegative(row.harmonic_cubic_reproduction_defect)
        && finite_nonnegative(row.edge_condition_max)
        && finite_nonnegative(row.local_condition_max)
        && finite_nonnegative(row.shared_setup_seconds)
        && finite_nonnegative(row.mode_runtime_seconds)
        && finite_nonnegative(row.total_seconds);
}

bool structure_ok(const NeumannEdgeCauchyMeasurement3D& row)
{
    return row.expected_non_g1_connections > 0
        && row.expected_non_g1_connections
            == row.covered_non_g1_connections
        && row.edge_sample_count > 0
        && row.affected_center_count > 0
        && row.corner_center_count > 0
        && row.unrelated_sample_or_attachment_count == 0
        && row.rank_deficient_fit_count == 0
        && row.far_centers_bitwise_legacy;
}

bool reproduction_ok(const NeumannEdgeCauchyMeasurement3D& row)
{
    return std::isfinite(row.harmonic_cubic_reproduction_defect)
        && row.harmonic_cubic_reproduction_defect <= 1.0e-11;
}

bool gmres_row_ok(const NeumannEdgeCauchyMeasurement3D& row)
{
    return row.gmres_converged && row.gmres_iterations >= 0
        && row.gmres_iterations <= 80
        && std::isfinite(row.gmres_relative_residual)
        && row.gmres_relative_residual <= 2.0e-10;
}

bool geometry_owner_ok(const NeumannEdgeCauchyMeasurement3D& row)
{
    return row.geometry_diagnostics_pass && row.owner_invariants_pass
        && row.shared_preprocess_pass;
}

double observed_order(double coarse_error, double fine_error,
                      double coarse_h, double fine_h)
{
    if (!std::isfinite(coarse_error) || !std::isfinite(fine_error)
        || !std::isfinite(coarse_h) || !std::isfinite(fine_h)
        || coarse_error <= 0.0 || fine_error <= 0.0
        || coarse_h <= fine_h || fine_h <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log(coarse_error / fine_error)
         / std::log(coarse_h / fine_h);
}

double ratio(double numerator, double denominator)
{
    if (!finite_nonnegative(numerator) || !std::isfinite(denominator)
        || denominator <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return numerator / denominator;
}

bool ratio_at_most(double numerator, double denominator, double limit)
{
    const double value = ratio(numerator, denominator);
    return std::isfinite(value)
        && value <= limit * (1.0 + kRelativeAllowance);
}

template <class Rows, class Predicate>
bool all_rows(const Rows& rows, const Predicate& predicate)
{
    return std::all_of(rows.begin(), rows.end(),
        [&](const auto* row) { return predicate(*row); });
}

bool complete_levels(const RowMap& rows,
                     const std::vector<std::string>& case_ids,
                     const std::vector<int>& levels)
{
    for (const std::string& case_id : case_ids) {
        for (int N : levels) {
            if (measurement(rows, case_id, N,
                    NeumannEdgeCauchyMode3D::None) == nullptr
                || measurement(rows, case_id, N,
                    NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues)
                    == nullptr) {
                return false;
            }
        }
    }
    return true;
}

bool level_has_rows(const RowMap& rows,
                    const std::vector<std::string>& case_ids, int N)
{
    for (const std::string& case_id : case_ids) {
        if (measurement(rows, case_id, N,
                NeumannEdgeCauchyMode3D::None) != nullptr
            || measurement(rows, case_id, N,
                NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues)
                != nullptr) {
            return true;
        }
    }
    return false;
}

bool present_pairs_complete(
    const RowMap& rows,
    const std::vector<const NeumannEdgeCauchyMeasurement3D*>& selected)
{
    for (const auto* row : selected) {
        const NeumannEdgeCauchyMode3D other =
            row->mode == NeumannEdgeCauchyMode3D::None
            ? NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues
            : NeumannEdgeCauchyMode3D::None;
        if (measurement(rows, row->case_id, row->N, other) == nullptr)
            return false;
    }
    return true;
}

bool paired_error_guard(
    const RowMap& rows,
    const std::vector<const NeumannEdgeCauchyMeasurement3D*>& selected)
{
    for (const auto* augmented : selected) {
        if (augmented->mode
            != NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) {
            continue;
        }
        const auto* legacy = measurement(rows, augmented->case_id,
            augmented->N, NeumannEdgeCauchyMode3D::None);
        if (legacy == nullptr
            || !ratio_at_most(augmented->density_linf,
                              legacy->density_linf, 1.10)
            || !ratio_at_most(augmented->density_l2,
                              legacy->density_l2, 1.10)
            || !ratio_at_most(augmented->interior_linf,
                              legacy->interior_linf, 1.10)
            || !ratio_at_most(augmented->interior_l2,
                              legacy->interior_l2, 1.10)) {
            return false;
        }
    }
    return true;
}

bool paired_edge_discrepancy(
    const RowMap& rows,
    const std::vector<const NeumannEdgeCauchyMeasurement3D*>& selected)
{
    for (const auto* augmented : selected) {
        if (augmented->mode
            != NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) {
            continue;
        }
        const auto* legacy = measurement(rows, augmented->case_id,
            augmented->N, NeumannEdgeCauchyMode3D::None);
        if (legacy == nullptr
            || !std::isfinite(augmented->incident_edge_discrepancy_linf)
            || !std::isfinite(legacy->incident_edge_discrepancy_linf)
            || !(augmented->incident_edge_discrepancy_linf
                 < legacy->incident_edge_discrepancy_linf)) {
            return false;
        }
    }
    return true;
}

bool gmres_pair_comparison(
    const std::vector<const NeumannEdgeCauchyMeasurement3D*>& selected)
{
    int legacy_max = 0;
    int augmented_max = 0;
    for (const auto* row : selected) {
        int& maximum = row->mode == NeumannEdgeCauchyMode3D::None
            ? legacy_max : augmented_max;
        maximum = std::max(maximum, row->gmres_iterations);
    }
    return augmented_max <= legacy_max;
}

using ErrorMember = double NeumannEdgeCauchyMeasurement3D::*;

bool level_spread_ok(const RowMap& rows,
                     const std::vector<std::string>& case_ids, int N)
{
    constexpr std::array<ErrorMember, 4> errors{{
        &NeumannEdgeCauchyMeasurement3D::density_linf,
        &NeumannEdgeCauchyMeasurement3D::density_l2,
        &NeumannEdgeCauchyMeasurement3D::interior_linf,
        &NeumannEdgeCauchyMeasurement3D::interior_l2}};
    for (ErrorMember member : errors) {
        double legacy_min = std::numeric_limits<double>::infinity();
        double legacy_max = 0.0;
        double augmented_min = std::numeric_limits<double>::infinity();
        double augmented_max = 0.0;
        for (const std::string& case_id : case_ids) {
            const auto* legacy = measurement(rows, case_id, N,
                NeumannEdgeCauchyMode3D::None);
            const auto* augmented = measurement(rows, case_id, N,
                NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues);
            if (legacy == nullptr || augmented == nullptr)
                return false;
            const double legacy_error = legacy->*member;
            const double augmented_error = augmented->*member;
            if (!std::isfinite(legacy_error) || legacy_error <= 0.0
                || !std::isfinite(augmented_error)
                || augmented_error <= 0.0) {
                return false;
            }
            legacy_min = std::min(legacy_min, legacy_error);
            legacy_max = std::max(legacy_max, legacy_error);
            augmented_min = std::min(augmented_min, augmented_error);
            augmented_max = std::max(augmented_max, augmented_error);
        }
        if (augmented_max / augmented_min
            > (legacy_max / legacy_min)
                * (1.0 + kRelativeAllowance)) {
            return false;
        }
    }

    int legacy_min_iteration = std::numeric_limits<int>::max();
    int legacy_max_iteration = std::numeric_limits<int>::min();
    int augmented_min_iteration = std::numeric_limits<int>::max();
    int augmented_max_iteration = std::numeric_limits<int>::min();
    for (const std::string& case_id : case_ids) {
        const auto* legacy = measurement(rows, case_id, N,
            NeumannEdgeCauchyMode3D::None);
        const auto* augmented = measurement(rows, case_id, N,
            NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues);
        if (legacy == nullptr || augmented == nullptr)
            return false;
        legacy_min_iteration = std::min(
            legacy_min_iteration, legacy->gmres_iterations);
        legacy_max_iteration = std::max(
            legacy_max_iteration, legacy->gmres_iterations);
        augmented_min_iteration = std::min(
            augmented_min_iteration, augmented->gmres_iterations);
        augmented_max_iteration = std::max(
            augmented_max_iteration, augmented->gmres_iterations);
    }
    const double legacy_spread = static_cast<double>(
        legacy_max_iteration - legacy_min_iteration);
    const double augmented_spread = static_cast<double>(
        augmented_max_iteration - augmented_min_iteration);
    return augmented_spread
        <= legacy_spread * (1.0 + kRelativeAllowance);
}

Status coarse_spread_status(const RowMap& rows,
                            const std::vector<std::string>& case_ids)
{
    bool any = false;
    bool all_ready = true;
    bool pass = true;
    for (int N : {32, 64}) {
        if (!level_has_rows(rows, case_ids, N)) continue;
        any = true;
        const bool ready = complete_levels(rows, case_ids, {N});
        all_ready = all_ready && ready;
        if (ready) pass = pass && level_spread_ok(rows, case_ids, N);
    }
    if (!any || (pass && !all_ready)) return Status::NotEvaluated;
    return status(pass);
}

bool augmented_order_ok(const RowMap& rows,
                        const std::vector<std::string>& case_ids,
                        int coarse_N, int fine_N)
{
    constexpr std::array<ErrorMember, 4> errors{{
        &NeumannEdgeCauchyMeasurement3D::density_linf,
        &NeumannEdgeCauchyMeasurement3D::density_l2,
        &NeumannEdgeCauchyMeasurement3D::interior_linf,
        &NeumannEdgeCauchyMeasurement3D::interior_l2}};
    for (const std::string& case_id : case_ids) {
        const auto* coarse = measurement(rows, case_id, coarse_N,
            NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues);
        const auto* fine = measurement(rows, case_id, fine_N,
            NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues);
        if (coarse == nullptr || fine == nullptr) return false;
        for (ErrorMember member : errors) {
            const double order = observed_order(
                coarse->*member, fine->*member, coarse->h, fine->h);
            if (!std::isfinite(order) || order < 1.8) return false;
        }
    }
    return true;
}

} // namespace

bool neumann_edge_cauchy_edge_value_row_finite_3d(
    const std::array<double, 6>& values)
{
    return std::all_of(values.begin(), values.end(),
        [](double value) { return std::isfinite(value); });
}
std::vector<int> normalize_neumann_edge_cauchy_levels_3d(
    std::vector<int> levels)
{
    if (levels.empty()) levels = {32, 64};
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    for (int N : levels) {
        if (N != 32 && N != 64 && N != 128) {
            throw std::invalid_argument(
                "Neumann edge-Cauchy study N must be 32, 64, or 128");
        }
    }
    const bool has32 = std::binary_search(levels.begin(), levels.end(), 32);
    const bool has64 = std::binary_search(levels.begin(), levels.end(), 64);
    const bool has128 = std::binary_search(levels.begin(), levels.end(), 128);
    if (has64 && !has32) {
        throw std::invalid_argument(
            "Neumann edge-Cauchy study N=64 requires N=32");
    }
    if (has128 && (!has32 || !has64)) {
        throw std::invalid_argument(
            "Neumann edge-Cauchy study N=128 requires N=32 and N=64");
    }
    return levels;
}

NeumannEdgeCauchyEvaluation3D evaluate_neumann_edge_cauchy_study_3d(
    const std::vector<NeumannEdgeCauchyMeasurement3D>& measurements,
    const std::vector<std::string>& case_ids,
    bool require_complete_pilot)
{
    std::set<std::string> known_cases;
    for (const std::string& case_id : case_ids) {
        if (case_id.empty() || !known_cases.insert(case_id).second) {
            throw std::invalid_argument(
                "Neumann edge-Cauchy case IDs must be nonempty and unique");
        }
    }

    RowMap keyed;
    for (const auto& row : measurements) {
        if (known_cases.count(row.case_id) == 0) {
            throw std::invalid_argument(
                "Neumann edge-Cauchy measurement has an unknown case ID");
        }
        if (row.N != 32 && row.N != 64 && row.N != 128) {
            throw std::invalid_argument(
                "Neumann edge-Cauchy measurement has an invalid level");
        }
        if (row.mode != NeumannEdgeCauchyMode3D::None
            && row.mode
                != NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) {
            throw std::invalid_argument(
                "Neumann edge-Cauchy measurement has an invalid mode");
        }
        if (!keyed.emplace(Key{row.case_id, row.N, row.mode},
                           std::addressof(row)).second) {
            throw std::invalid_argument(
                "Neumann edge-Cauchy measurement key is duplicated");
        }
    }

    std::vector<const NeumannEdgeCauchyMeasurement3D*> coarse_rows;
    std::vector<const NeumannEdgeCauchyMeasurement3D*> extended_rows;
    for (const auto& row : measurements) {
        (row.N == 128 ? extended_rows : coarse_rows)
            .push_back(std::addressof(row));
    }

    NeumannEdgeCauchyEvaluation3D result;
    result.rows.reserve(measurements.size());
    for (const auto& item : measurements) {
        NeumannEdgeCauchyDerivedRow3D row;
        row.measurement = item;
        const NeumannEdgeCauchyMeasurement3D* previous = nullptr;
        for (const auto& candidate : measurements) {
            if (candidate.case_id == item.case_id
                && candidate.mode == item.mode && candidate.N < item.N
                && (previous == nullptr || candidate.N > previous->N)) {
                previous = std::addressof(candidate);
            }
        }
        if (previous != nullptr) {
            row.density_linf_order = observed_order(
                previous->density_linf, item.density_linf,
                previous->h, item.h);
            row.density_l2_order = observed_order(
                previous->density_l2, item.density_l2,
                previous->h, item.h);
            row.interior_linf_order = observed_order(
                previous->interior_linf, item.interior_linf,
                previous->h, item.h);
            row.interior_l2_order = observed_order(
                previous->interior_l2, item.interior_l2,
                previous->h, item.h);
        }
        if (item.mode == NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues) {
            const auto* legacy = measurement(
                keyed, item.case_id, item.N,
                NeumannEdgeCauchyMode3D::None);
            if (legacy != nullptr) {
                row.density_linf_ratio_to_legacy = ratio(
                    item.density_linf, legacy->density_linf);
                row.density_l2_ratio_to_legacy = ratio(
                    item.density_l2, legacy->density_l2);
                row.interior_linf_ratio_to_legacy = ratio(
                    item.interior_linf, legacy->interior_linf);
                row.interior_l2_ratio_to_legacy = ratio(
                    item.interior_l2, legacy->interior_l2);
                row.edge_discrepancy_ratio_to_legacy = ratio(
                    item.incident_edge_discrepancy_linf,
                    legacy->incident_edge_discrepancy_linf);
            }
        }
        row.row_pass = status(measurement_finite(item)
            && structure_ok(item) && reproduction_ok(item)
            && gmres_row_ok(item) && geometry_owner_ok(item));
        result.rows.push_back(std::move(row));
    }

    const bool coarse_complete = complete_levels(keyed, case_ids, {32, 64});
    result.acceptance.completeness_pass = coarse_complete
        ? Status::Pass
        : require_complete_pilot ? Status::Fail : Status::NotEvaluated;
    result.acceptance.structure_pass = coarse_rows.empty()
        ? Status::NotEvaluated
        : status(all_rows(coarse_rows, structure_ok));
    result.acceptance.reproduction_pass = coarse_rows.empty()
        ? Status::NotEvaluated
        : status(all_rows(coarse_rows, reproduction_ok));
    result.acceptance.geometry_owner_pass = coarse_rows.empty()
        ? Status::NotEvaluated
        : status(all_rows(coarse_rows, geometry_owner_ok));

    const bool coarse_pairs = present_pairs_complete(keyed, coarse_rows);
    const bool coarse_execution_ok = all_rows(coarse_rows,
        [](const auto& row) {
            return measurement_finite(row) && gmres_row_ok(row);
        });
    if (coarse_rows.empty()) {
        result.acceptance.gmres_pass = Status::NotEvaluated;
    } else if (!coarse_execution_ok) {
        result.acceptance.gmres_pass = Status::Fail;
    } else if (!coarse_pairs) {
        result.acceptance.gmres_pass = Status::NotEvaluated;
    } else {
        result.acceptance.gmres_pass = status(
            gmres_pair_comparison(coarse_rows));
    }

    if (coarse_rows.empty() || !coarse_pairs) {
        result.acceptance.error_guard_pass = Status::NotEvaluated;
        result.acceptance.edge_discrepancy_pass = Status::NotEvaluated;
    } else {
        result.acceptance.error_guard_pass = status(
            paired_error_guard(keyed, coarse_rows));
        result.acceptance.edge_discrepancy_pass = status(
            paired_edge_discrepancy(keyed, coarse_rows));
    }
    result.acceptance.order_pass = coarse_complete
        ? status(augmented_order_ok(keyed, case_ids, 32, 64))
        : Status::NotEvaluated;
    result.acceptance.rigid_spread_pass =
        coarse_spread_status(keyed, case_ids);

    if (extended_rows.empty()) {
        result.acceptance.extended_evidence_pass = Status::NotEvaluated;
    } else {
        const bool extended_complete =
            complete_levels(keyed, case_ids, {128});
        bool extended_ok = extended_complete
            && all_rows(extended_rows, measurement_finite)
            && all_rows(extended_rows, structure_ok)
            && all_rows(extended_rows, reproduction_ok)
            && all_rows(extended_rows, gmres_row_ok)
            && all_rows(extended_rows, geometry_owner_ok);
        if (extended_complete) {
            extended_ok = extended_ok
                && present_pairs_complete(keyed, extended_rows)
                && gmres_pair_comparison(extended_rows)
                && paired_error_guard(keyed, extended_rows)
                && paired_edge_discrepancy(keyed, extended_rows)
                && level_spread_ok(keyed, case_ids, 128)
                && augmented_order_ok(keyed, case_ids, 64, 128);
        }
        result.acceptance.extended_evidence_pass = status(extended_ok);
    }

    result.acceptance.overall_pass = combine_rigid_study_criteria_3d(
        {result.acceptance.completeness_pass,
         result.acceptance.structure_pass,
         result.acceptance.reproduction_pass,
         result.acceptance.gmres_pass,
         result.acceptance.error_guard_pass,
         result.acceptance.order_pass,
         result.acceptance.rigid_spread_pass,
         result.acceptance.edge_discrepancy_pass,
         result.acceptance.geometry_owner_pass},
        !coarse_rows.empty(), require_complete_pilot);
    result.all_pass = result.acceptance.overall_pass == Status::Pass;
    return result;
}

bool neumann_edge_cauchy_study_exit_pass_3d(
    const NeumannEdgeCauchyEvaluation3D& evaluation,
    bool require_complete_pilot)
{
    if (require_complete_pilot) return evaluation.all_pass;
    return !evaluation.rows.empty()
        && std::all_of(evaluation.rows.begin(), evaluation.rows.end(),
            [](const NeumannEdgeCauchyDerivedRow3D& row) {
                return row.row_pass == Status::Pass;
            })
        && evaluation.acceptance.structure_pass == Status::Pass
        && evaluation.acceptance.reproduction_pass == Status::Pass
        && evaluation.acceptance.gmres_pass != Status::Fail
        && evaluation.acceptance.geometry_owner_pass == Status::Pass;
}

} // namespace kfbim::app3d