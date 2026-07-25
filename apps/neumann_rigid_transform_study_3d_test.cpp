#include "neumann_rigid_transform_study_3d.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using kfbim::app3d::NeumannRigidStudyAcceptance3D;
using kfbim::app3d::NeumannRigidStudyDerivedRow3D;
using kfbim::app3d::NeumannRigidStudyEvaluation3D;
using kfbim::app3d::NeumannRigidStudyMeasurement3D;
using kfbim::app3d::RigidStudyCriterionStatus3D;
using kfbim::app3d::evaluate_neumann_rigid_study_3d;
using kfbim::app3d::normalize_neumann_rigid_levels_3d;

using Status = RigidStudyCriterionStatus3D;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_near(double actual,
                  double expected,
                  double tolerance,
                  const std::string& message)
{
    require(std::abs(actual - expected) <= tolerance, message);
}

void require_throws(const std::function<void()>& operation,
                    const std::string& message)
{
    bool threw = false;
    try {
        operation();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, message);
}

NeumannRigidStudyMeasurement3D passing(
    std::string case_id,
    int N,
    double h,
    double interior_linf,
    int iterations)
{
    NeumannRigidStudyMeasurement3D row;
    row.case_id = std::move(case_id);
    row.N = N;
    row.h = h;
    row.finite_metrics = true;
    row.gmres_converged = true;
    row.gmres_iterations = iterations;
    row.gmres_relative_residual = 1.0e-11;
    row.interior_linf = interior_linf;
    row.geometry_diagnostics_pass = true;
    row.owner_invariants_pass = true;
    return row;
}

std::vector<NeumannRigidStudyMeasurement3D> complete_rows(
    double shifted_ratio)
{
    return {
        passing("baseline", 32, 0.09375, 8.0e-6, 40),
        passing("shifted", 32, 0.09375, shifted_ratio * 8.0e-6, 44),
        passing("baseline", 64, 0.046875, 2.0e-6, 41),
        passing("shifted", 64, 0.046875, shifted_ratio * 2.0e-6, 45),
        passing("baseline", 128, 0.0234375, 5.0e-7, 42),
        passing("shifted", 128, 0.0234375, shifted_ratio * 5.0e-7, 46)};
}

const NeumannRigidStudyDerivedRow3D& find_row(
    const NeumannRigidStudyEvaluation3D& evaluation,
    const std::string& case_id,
    int N)
{
    for (const auto& row : evaluation.rows) {
        if (row.measurement.case_id == case_id && row.measurement.N == N)
            return row;
    }
    throw std::runtime_error("missing derived row");
}

const NeumannRigidStudyAcceptance3D& find_case(
    const NeumannRigidStudyEvaluation3D& evaluation,
    const std::string& case_id)
{
    for (const auto& item : evaluation.cases) {
        if (item.case_id == case_id)
            return item;
    }
    throw std::runtime_error("missing case acceptance");
}

void test_level_prefix_validation()
{
    require(normalize_neumann_rigid_levels_3d({128, 32, 64, 64})
                == std::vector<int>({32, 64, 128}),
            "levels are not sorted and deduplicated");
    require(normalize_neumann_rigid_levels_3d({})
                == std::vector<int>({32, 64, 128}),
            "empty levels do not select the production default");
    require(normalize_neumann_rigid_levels_3d({32})
                == std::vector<int>({32}),
            "N=32 prefix rejected");
    require(normalize_neumann_rigid_levels_3d({32, 64})
                == std::vector<int>({32, 64}),
            "N=32,64 prefix rejected");
    require_throws([] {
        normalize_neumann_rigid_levels_3d({64});
    }, "N=64 did not require N=32");
    require_throws([] {
        normalize_neumann_rigid_levels_3d({32, 128});
    }, "N=128 did not require N=32 and N=64");
    require_throws([] {
        normalize_neumann_rigid_levels_3d({16, 32});
    }, "N=16 was not rejected");
    require_throws([] {
        normalize_neumann_rigid_levels_3d({32, 64, 256});
    }, "N=256 was not rejected");
}

void test_orders_are_grouped_by_case_and_baselines_match_levels()
{
    const NeumannRigidStudyEvaluation3D evaluation =
        evaluate_neumann_rigid_study_3d(
            complete_rows(4.0), {"baseline", "shifted"}, true);

    require_near(find_row(evaluation, "baseline", 64).interior_order,
                 2.0, 2.0e-14, "baseline N=64 order");
    require_near(find_row(evaluation, "baseline", 128).interior_order,
                 2.0, 2.0e-14, "baseline N=128 order");
    require_near(find_row(evaluation, "shifted", 64).interior_order,
                 2.0, 2.0e-14, "shifted N=64 order");
    require_near(find_row(evaluation, "shifted", 128).interior_order,
                 2.0, 2.0e-14, "shifted N=128 order");
    for (int N : {32, 64, 128}) {
        require_near(find_row(evaluation, "shifted", N).baseline_error_ratio,
                     4.0, 2.0e-14, "same-level baseline error ratio");
    }
    require(find_case(evaluation, "baseline").overall_pass == Status::Pass,
            "baseline case did not pass");
    require(find_case(evaluation, "shifted").baseline_ratio_pass
                == Status::Fail,
            "shifted baseline-ratio failure was not reported");
    require(find_case(evaluation, "shifted").overall_pass == Status::Fail,
            "shifted overall failure was not reported");
    require(!evaluation.all_pass,
            "study with a failed pose was marked all-pass");
}

void test_complete_passing_study()
{
    const NeumannRigidStudyEvaluation3D evaluation =
        evaluate_neumann_rigid_study_3d(
            complete_rows(2.0), {"baseline", "shifted"}, true);
    require(evaluation.rows.size() == 6, "derived row count");
    require(evaluation.cases.size() == 2, "acceptance row count");
    require(evaluation.all_pass, "complete passing study did not pass");
    for (const auto& item : evaluation.cases) {
        require(item.completeness_pass == Status::Pass,
                "complete pose is not complete");
        require(item.monotone_error_pass == Status::Pass,
                "monotone errors did not pass");
        require(item.order_64_128_pass == Status::Pass,
                "second-order sequence did not pass");
        require(item.overall_pass == Status::Pass,
                "passing pose did not pass overall");
    }
}

void test_incomplete_and_failed_rows_are_not_hidden()
{
    std::vector<NeumannRigidStudyMeasurement3D> incomplete = {
        passing("baseline", 32, 0.09375, 8.0e-6, 40),
        passing("baseline", 64, 0.046875, 2.0e-6, 41)};
    const NeumannRigidStudyEvaluation3D strict =
        evaluate_neumann_rigid_study_3d(
            incomplete, {"baseline"}, true);
    require(find_case(strict, "baseline").completeness_pass
                == Status::NotEvaluated,
            "incomplete pose was marked complete");
    require(find_case(strict, "baseline").overall_pass
                == Status::NotEvaluated,
            "strict incomplete pose received an overall decision");
    require(!strict.all_pass, "strict incomplete study passed");

    const NeumannRigidStudyEvaluation3D smoke =
        evaluate_neumann_rigid_study_3d(
            incomplete, {"baseline"}, false);
    require(find_case(smoke, "baseline").overall_pass == Status::Pass,
            "valid prefix smoke study did not pass");
    require(smoke.all_pass, "valid prefix smoke study was not all-pass");

    auto gmres_failure = complete_rows(2.0);
    gmres_failure[2].gmres_converged = false;
    require(find_case(evaluate_neumann_rigid_study_3d(
                          gmres_failure, {"baseline", "shifted"}, true),
                      "baseline").gmres_pass == Status::Fail,
            "GMRES failure was hidden");

    auto residual_failure = complete_rows(2.0);
    residual_failure[4].gmres_relative_residual = 2.1e-10;
    require(find_case(evaluate_neumann_rigid_study_3d(
                          residual_failure, {"baseline", "shifted"}, true),
                      "baseline").gmres_pass == Status::Fail,
            "GMRES residual failure was hidden");

    auto finite_failure = complete_rows(2.0);
    finite_failure[0].finite_metrics = false;
    require(find_case(evaluate_neumann_rigid_study_3d(
                          finite_failure, {"baseline", "shifted"}, true),
                      "baseline").row_pass == Status::Fail,
            "non-finite row was hidden");

    auto geometry_failure = complete_rows(2.0);
    geometry_failure[1].geometry_diagnostics_pass = false;
    require(find_case(evaluate_neumann_rigid_study_3d(
                          geometry_failure, {"baseline", "shifted"}, true),
                      "shifted").geometry_diagnostics_pass == Status::Fail,
            "geometry failure was hidden");

    auto owner_failure = complete_rows(2.0);
    owner_failure[3].owner_invariants_pass = false;
    require(find_case(evaluate_neumann_rigid_study_3d(
                          owner_failure, {"baseline", "shifted"}, true),
                      "shifted").owner_invariants_pass == Status::Fail,
            "owner-invariant failure was hidden");
}

} // namespace

int main()
{
    try {
        test_level_prefix_validation();
        test_orders_are_grouped_by_case_and_baselines_match_levels();
        test_complete_passing_study();
        test_incomplete_and_failed_rows_are_not_hidden();
        std::cout << "3D Neumann rigid-transform study tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "3D Neumann rigid-transform study test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
