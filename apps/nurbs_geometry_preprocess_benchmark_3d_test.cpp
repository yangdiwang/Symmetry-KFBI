#include "nurbs_geometry_preprocess_benchmark_3d.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kfbim::app3d::GeometryKind3D;
using namespace kfbim::app3d::benchmark3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class Function>
void require_throws(Function&& function, const std::string& message)
{
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_complete_backend_equivalence()
{
    constexpr int N = 16;
    constexpr std::uint64_t expected_nodes =
        static_cast<std::uint64_t>(N + 1) * (N + 1) * (N + 1);
    constexpr std::uint64_t expected_edges =
        3ULL * N * (N + 1) * (N + 1);

    std::uint64_t first_torus_checksum = 0;
    for (const GeometryKind3D geometry : {
             GeometryKind3D::Torus,
             GeometryKind3D::HollowCylinder,
             GeometryKind3D::LPrism}) {
        const BenchmarkSupportCase3D result =
            run_benchmark_support_case_3d(geometry, N);
        require(result.runs.size() == 3,
                "support case runs all three preprocessing strategies");

        const std::uint64_t topology_checksum =
            result.runs.front().topology_checksum;
        for (const BackendMaterialization3D& run : result.runs) {
            require(run.node_count == expected_nodes,
                    "support case scans every Cartesian node");
            require(run.edge_count == expected_edges,
                    "support case scans every structured Cartesian edge");
            require(run.interface_edge_count > 0,
                    "support case materializes interface edges");
            require(run.correction_owner_count > 0,
                    "support case materializes every correction owner");
            require(run.topology_checksum != 0
                        && run.materialization_checksum != 0,
                    "support case produces nonzero deterministic checksums");
            require(run.topology_checksum == topology_checksum,
                    "all backends have the same discrete topology checksum");
            require(run.analytic_label_mismatch_count == 0,
                    "all node labels match the independent analytic oracle");
            require(run.baseline_label_mismatch_count == 0
                        && run.edge_flag_mismatch_count == 0
                        && run.edge_classification_mismatch_count == 0
                        && run.crossing_count_mismatch_count == 0
                        && run.crossing_field_mismatch_count == 0
                        && run.correction_crossing_mismatch_count == 0
                        && run.correction_owner_mismatch_count == 0
                        && run.unsafe_label_changing_edge_count == 0,
                    "accelerated backend materialization matches baseline");
        }
        require(result.mismatches.empty(),
                "complete comparison has no mismatch diagnostics");
        if (geometry == GeometryKind3D::Torus)
            first_torus_checksum = result.runs.front().materialization_checksum;
    }

    const BenchmarkSupportCase3D repeated =
        run_benchmark_support_case_3d(GeometryKind3D::Torus, N);
    require(repeated.runs.front().materialization_checksum
                == first_torus_checksum,
            "materialization checksum is deterministic across fresh domains");
}

void test_cli_statistics_and_csv_support()
{
    const BenchmarkOptions3D options = parse_benchmark_cli_3d({
        "--backend", "all",
        "--geometry", "all",
        "--N", "16", "32",
        "--warmup", "1",
        "--reps", "2",
        "--out", "benchmark out"});
    require(options.backends.size() == 3
                && options.geometries.size() == 3
                && options.levels == std::vector<int>({16, 32})
                && options.warmup == 1
                && options.repetitions == 2
                && options.output_directory
                    == std::filesystem::path("benchmark out"),
            "CLI parser expands the complete production matrix");

    require_throws(
        [] { (void)parse_benchmark_cli_3d({"--unknown", "x"}); },
        "CLI parser rejects unknown options");
    require_throws(
        [] {
            (void)parse_benchmark_cli_3d(
                {"--backend", "fast", "--geometry", "torus",
                 "--N", "16", "--warmup", "0", "--reps", "1",
                 "--out", "out"});
        },
        "CLI parser rejects unknown backend names");
    require_throws(
        [] {
            (void)parse_benchmark_cli_3d(
                {"--backend", "baseline", "--geometry", "torus",
                 "--N", "0", "--warmup", "0", "--reps", "1",
                 "--out", "out"});
        },
        "CLI parser rejects nonpositive grids");
    require_throws(
        [] {
            (void)parse_benchmark_cli_3d(
                {"--backend", "baseline", "--geometry", "torus",
                 "--N", "16", "--warmup", "0", "--reps", "0",
                 "--out", "out"});
        },
        "CLI parser rejects nonpositive repetitions");
    require_throws(
        [] {
            (void)parse_benchmark_cli_3d(
                {"--backend", "baseline", "--geometry", "torus",
                 "--N", "--warmup", "0", "--reps", "1",
                 "--out", "out"});
        },
        "CLI parser rejects a missing N value");

    const TimingSummary3D summary =
        summarize_timings_3d({4.0, 1.0, 3.0, 2.0}, 6.0);
    require(summary.sample_count == 4
                && summary.minimum == 1.0
                && summary.median == 2.5
                && summary.maximum == 4.0
                && std::abs(summary.coefficient_of_variation
                            - std::sqrt(1.25) / 2.5) < 1.0e-15
                && summary.speedup == 2.4,
            "statistics report population CV and baseline speedup");
    const TimingSummary3D singleton =
        summarize_timings_3d({2.0}, 2.0);
    require(singleton.coefficient_of_variation == 0.0,
            "single-sample CV is zero");

    const std::vector<std::string> raw_header = raw_csv_header_3d();
    const std::vector<std::string> raw_row =
        raw_csv_row_3d(RawBenchmarkRecord3D{});
    const std::vector<std::string> summary_header = summary_csv_header_3d();
    const std::vector<std::string> summary_row =
        summary_csv_row_3d(SummaryBenchmarkRecord3D{});
    require(!raw_header.empty() && raw_header.size() == raw_row.size(),
            "raw CSV header and row widths are mechanically aligned");
    require(!summary_header.empty()
                && summary_header.size() == summary_row.size(),
            "summary CSV header and row widths are mechanically aligned");
    require(encode_csv_row_3d({"plain", "a,b", "c\"d"})
                == "plain,\"a,b\",\"c\"\"d\"",
            "CSV encoder quotes commas and doubles embedded quotes");
}

void test_successful_run_removes_stale_mismatch_artifact()
{
    const std::filesystem::path output =
        ".superpowers/sdd/geometry-preprocess-benchmark-test-output";
    std::filesystem::create_directories(output);
    const std::filesystem::path mismatch =
        output / "nurbs_geometry_preprocess_mismatches.csv";
    {
        std::ofstream stale(mismatch);
        stale << "stale mismatch\n";
    }
    BenchmarkOptions3D options;
    options.backends = {Backend3D::Baseline};
    options.geometries = {GeometryKind3D::Torus};
    options.levels = {16};
    options.warmup = 0;
    options.repetitions = 1;
    options.output_directory = output;
    const BenchmarkRunResult3D result =
        run_nurbs_geometry_preprocess_benchmark_3d(options);
    require(result.mismatches.empty(),
            "focused successful runner has no mismatches");
    require(!std::filesystem::exists(mismatch),
            "successful runner removes a stale mismatch artifact");
}

} // namespace

int main()
{
    try {
        test_complete_backend_equivalence();
        test_cli_statistics_and_csv_support();
        test_successful_run_removes_stale_mismatch_artifact();
        std::cout << "nurbs geometry preprocess benchmark 3d tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "nurbs geometry preprocess benchmark 3d test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
