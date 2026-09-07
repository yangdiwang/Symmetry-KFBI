#include "src/support/diagnostics/neumann_rigid_transform_study_3d.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d {
namespace {

using Status = RigidStudyCriterionStatus3D;

Status pass_or_fail(bool pass)
{
    return pass ? Status::Pass : Status::Fail;
}

double observed_order(double coarse_error,
                      double fine_error,
                      int coarse_N,
                      int fine_N)
{
    if (!std::isfinite(coarse_error) || !std::isfinite(fine_error)
        || !(coarse_error > 0.0) || !(fine_error > 0.0)
        || coarse_N <= 0 || fine_N <= coarse_N) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log(coarse_error / fine_error)
         / std::log(static_cast<double>(fine_N)
                    / static_cast<double>(coarse_N));
}

template <class Row>
const Row* find_row(const std::vector<Row>& rows,
                    const std::string& case_id,
                    int N)
{
    const auto found = std::find_if(
        rows.begin(), rows.end(), [&](const Row& row) {
            return row.measurement.case_id == case_id
                && row.measurement.N == N;
        });
    return found == rows.end() ? nullptr : std::addressof(*found);
}

Status aggregate_status(
    const std::vector<const NeumannRigidStudyDerivedRow3D*>& rows,
    Status NeumannRigidStudyDerivedRow3D::*member)
{
    if (rows.empty())
        return Status::NotEvaluated;
    bool has_not_evaluated = false;
    for (const NeumannRigidStudyDerivedRow3D* row : rows) {
        const Status status = row->*member;
        if (status == Status::Fail)
            return Status::Fail;
        has_not_evaluated =
            has_not_evaluated || status == Status::NotEvaluated;
    }
    return has_not_evaluated ? Status::NotEvaluated : Status::Pass;
}

} // namespace

std::vector<int> normalize_neumann_rigid_levels_3d(
    std::vector<int> levels)
{
    if (levels.empty())
        levels = {32, 64, 128};
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    for (int N : levels) {
        if (N != 32 && N != 64 && N != 128) {
            throw std::invalid_argument(
                "Neumann rigid-study N must be 32, 64, or 128");
        }
    }
    const bool has32 =
        std::binary_search(levels.begin(), levels.end(), 32);
    const bool has64 =
        std::binary_search(levels.begin(), levels.end(), 64);
    const bool has128 =
        std::binary_search(levels.begin(), levels.end(), 128);
    if (has64 && !has32) {
        throw std::invalid_argument(
            "Neumann rigid-study N=64 requires N=32");
    }
    if (has128 && (!has32 || !has64)) {
        throw std::invalid_argument(
            "Neumann rigid-study N=128 requires N=32 and N=64");
    }
    return levels;
}

NeumannRigidStudyEvaluation3D evaluate_neumann_rigid_study_3d(
    const std::vector<NeumannRigidStudyMeasurement3D>& measurements,
    const std::vector<std::string>& case_ids,
    bool require_complete_acceptance)
{
    std::set<std::string> known_cases;
    for (const std::string& case_id : case_ids) {
        if (case_id.empty() || !known_cases.insert(case_id).second) {
            throw std::invalid_argument(
                "Neumann rigid-study case IDs must be nonempty and unique");
        }
    }

    std::set<std::pair<std::string, int>> keys;
    for (const NeumannRigidStudyMeasurement3D& row : measurements) {
        if (known_cases.count(row.case_id) == 0) {
            throw std::invalid_argument(
                "Neumann rigid-study measurement has an unknown case ID");
        }
        if (!keys.insert({row.case_id, row.N}).second) {
            throw std::invalid_argument(
                "Neumann rigid-study measurement key is duplicated");
        }
    }

    NeumannRigidStudyEvaluation3D result;
    result.rows.reserve(measurements.size());
    for (const NeumannRigidStudyMeasurement3D& measurement : measurements) {
        NeumannRigidStudyDerivedRow3D row;
        row.measurement = measurement;

        const NeumannRigidStudyMeasurement3D* previous = nullptr;
        for (const NeumannRigidStudyMeasurement3D& candidate
             : measurements) {
            if (candidate.case_id == measurement.case_id
                && candidate.N < measurement.N
                && (previous == nullptr || candidate.N > previous->N)) {
                previous = std::addressof(candidate);
            }
        }
        if (previous != nullptr) {
            row.interior_order = observed_order(
                previous->interior_linf,
                measurement.interior_linf,
                previous->N,
                measurement.N);
        }

        const NeumannRigidStudyMeasurement3D* baseline = nullptr;
        for (const NeumannRigidStudyMeasurement3D& candidate
             : measurements) {
            if (candidate.case_id == "baseline"
                && candidate.N == measurement.N) {
                baseline = std::addressof(candidate);
                break;
            }
        }
        if (baseline != nullptr
            && std::isfinite(baseline->interior_linf)
            && baseline->interior_linf > 0.0
            && std::isfinite(measurement.interior_linf)) {
            row.baseline_error_ratio =
                measurement.interior_linf / baseline->interior_linf;
        }
        if (baseline != nullptr && baseline->gmres_iterations > 0) {
            row.baseline_iteration_ratio =
                static_cast<double>(measurement.gmres_iterations)
                / static_cast<double>(baseline->gmres_iterations);
        }

        row.gmres_pass = pass_or_fail(
            measurement.gmres_converged
            && measurement.gmres_iterations <= 80
            && std::isfinite(measurement.gmres_relative_residual)
            && measurement.gmres_relative_residual <= 2.0e-10);
        if (std::isfinite(row.baseline_error_ratio)) {
            row.baseline_ratio_pass =
                pass_or_fail(row.baseline_error_ratio <= 3.0);
        }
        row.geometry_diagnostics_pass =
            pass_or_fail(measurement.geometry_diagnostics_pass);
        row.owner_invariants_pass =
            pass_or_fail(measurement.owner_invariants_pass);
        row.row_pass = pass_or_fail(
            measurement.finite_metrics
            && row.gmres_pass == Status::Pass
            && row.geometry_diagnostics_pass == Status::Pass
            && row.owner_invariants_pass == Status::Pass);
        result.rows.push_back(std::move(row));
    }

    result.cases.reserve(case_ids.size());
    for (const std::string& case_id : case_ids) {
        std::vector<const NeumannRigidStudyDerivedRow3D*> case_rows;
        for (const NeumannRigidStudyDerivedRow3D& row : result.rows) {
            if (row.measurement.case_id == case_id)
                case_rows.push_back(std::addressof(row));
        }

        NeumannRigidStudyAcceptance3D acceptance;
        acceptance.case_id = case_id;
        const NeumannRigidStudyDerivedRow3D* row32 =
            find_row(result.rows, case_id, 32);
        const NeumannRigidStudyDerivedRow3D* row64 =
            find_row(result.rows, case_id, 64);
        const NeumannRigidStudyDerivedRow3D* row128 =
            find_row(result.rows, case_id, 128);
        if (row32 != nullptr && row64 != nullptr && row128 != nullptr)
            acceptance.completeness_pass = Status::Pass;

        acceptance.row_pass = aggregate_status(
            case_rows, &NeumannRigidStudyDerivedRow3D::row_pass);
        acceptance.gmres_pass = aggregate_status(
            case_rows, &NeumannRigidStudyDerivedRow3D::gmres_pass);
        acceptance.baseline_ratio_pass = aggregate_status(
            case_rows,
            &NeumannRigidStudyDerivedRow3D::baseline_ratio_pass);
        acceptance.geometry_diagnostics_pass = aggregate_status(
            case_rows,
            &NeumannRigidStudyDerivedRow3D::geometry_diagnostics_pass);
        acceptance.owner_invariants_pass = aggregate_status(
            case_rows,
            &NeumannRigidStudyDerivedRow3D::owner_invariants_pass);

        if (row32 != nullptr && row64 != nullptr && row128 != nullptr) {
            const double error32 = row32->measurement.interior_linf;
            const double error64 = row64->measurement.interior_linf;
            const double error128 = row128->measurement.interior_linf;
            acceptance.monotone_error_pass = pass_or_fail(
                std::isfinite(error32) && std::isfinite(error64)
                && std::isfinite(error128)
                && error32 > error64 && error64 > error128);
            acceptance.order_64_128_pass = pass_or_fail(
                std::isfinite(row128->interior_order)
                && row128->interior_order >= 1.8);
        }

        acceptance.overall_pass = combine_rigid_study_criteria_3d(
            {acceptance.completeness_pass,
             acceptance.row_pass,
             acceptance.gmres_pass,
             acceptance.monotone_error_pass,
             acceptance.order_64_128_pass,
             acceptance.baseline_ratio_pass,
             acceptance.geometry_diagnostics_pass,
             acceptance.owner_invariants_pass},
            !case_rows.empty(),
            require_complete_acceptance);
        result.cases.push_back(std::move(acceptance));
    }

    result.all_pass = !result.cases.empty()
        && std::all_of(
            result.cases.begin(), result.cases.end(),
            [](const NeumannRigidStudyAcceptance3D& item) {
                return item.overall_pass == Status::Pass;
            });
    return result;
}

bool neumann_rigid_study_exit_pass_3d(
    const NeumannRigidStudyEvaluation3D& evaluation,
    bool require_complete_acceptance)
{
    if (require_complete_acceptance)
        return evaluation.all_pass;
    return !evaluation.rows.empty()
        && std::all_of(
            evaluation.rows.begin(),
            evaluation.rows.end(),
            [](const NeumannRigidStudyDerivedRow3D& row) {
                return row.row_pass
                    == RigidStudyCriterionStatus3D::Pass;
            });
}

} // namespace kfbim::app3d
