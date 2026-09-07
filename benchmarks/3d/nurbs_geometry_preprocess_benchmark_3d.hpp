#pragma once

#include "src/support/geometry/native_nurbs_surface_3d.hpp"
#include "src/geometry/nurbs_cartesian_domain_3d.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace kfbim {
struct P2CrossingOwner3D;
}

namespace kfbim::app3d::benchmark3d {

enum class Backend3D {
    Baseline,
    Pure,
    Hybrid
};

struct BenchmarkOptions3D {
    std::vector<Backend3D> backends{
        Backend3D::Baseline, Backend3D::Pure, Backend3D::Hybrid};
    std::vector<GeometryKind3D> geometries{
        GeometryKind3D::Torus,
        GeometryKind3D::HollowCylinder,
        GeometryKind3D::LPrism};
    std::vector<int> levels{32, 64, 128};
    int warmup = 1;
    int repetitions = 3;
    std::filesystem::path output_directory{"."};
};

struct BackendMaterialization3D {
    Backend3D backend = Backend3D::Baseline;
    std::uint64_t node_count = 0;
    std::uint64_t edge_count = 0;
    std::uint64_t barrier_edge_count = 0;
    std::uint64_t interface_edge_count = 0;
    std::uint64_t crossing_count = 0;
    std::uint64_t label_changing_edge_count = 0;
    std::uint64_t correction_owner_count = 0;
    std::uint64_t analytic_label_mismatch_count = 0;
    std::uint64_t baseline_label_mismatch_count = 0;
    std::uint64_t edge_flag_mismatch_count = 0;
    std::uint64_t edge_classification_mismatch_count = 0;
    std::uint64_t crossing_count_mismatch_count = 0;
    std::uint64_t crossing_field_mismatch_count = 0;
    std::uint64_t correction_crossing_mismatch_count = 0;
    std::uint64_t correction_owner_mismatch_count = 0;
    std::uint64_t unsafe_label_changing_edge_count = 0;
    double maximum_point_error = 0.0;
    double maximum_normal_error = 0.0;
    double maximum_uv_point_error = 0.0;
    double maximum_edge_parameter_error = 0.0;
    double maximum_residual_error = 0.0;
    double maximum_transversality_error = 0.0;
    double maximum_reliability_tolerance_error = 0.0;
    double maximum_owner_error = 0.0;
    std::uint64_t topology_checksum = 0;
    std::uint64_t materialization_checksum = 0;
};

struct BenchmarkSupportCase3D {
    std::vector<BackendMaterialization3D> runs;
    std::vector<std::string> mismatches;
};

struct RawBenchmarkRecord3D {
    std::string geometry;
    int N = 0;
    Backend3D backend = Backend3D::Baseline;
    int repetition = 0;
    int execution_order = 0;
    double domain_seconds = 0.0;
    double grid_pair_seconds = 0.0;
    double combined_seconds = 0.0;
    geometry3d::NurbsCartesianDomainDiagnostics3D diagnostics;
    BackendMaterialization3D materialization;
};

struct TimingSummary3D {
    std::size_t sample_count = 0;
    double minimum = 0.0;
    double median = 0.0;
    double maximum = 0.0;
    double coefficient_of_variation = 0.0;
    double speedup = 0.0;
};

struct SummaryBenchmarkRecord3D {
    std::string geometry;
    int N = 0;
    Backend3D backend = Backend3D::Baseline;
    TimingSummary3D domain;
    TimingSummary3D grid_pair;
    TimingSummary3D combined;
};

struct BenchmarkRunResult3D {
    std::vector<RawBenchmarkRecord3D> raw;
    std::vector<SummaryBenchmarkRecord3D> summary;
    std::vector<std::string> mismatches;
};

[[nodiscard]] const char* backend_name_3d(Backend3D backend) noexcept;
[[nodiscard]] const char* geometry_name_3d(GeometryKind3D geometry) noexcept;

[[nodiscard]] BenchmarkOptions3D parse_benchmark_cli_3d(
    const std::vector<std::string>& arguments);
void validate_benchmark_options_3d(const BenchmarkOptions3D& options);

[[nodiscard]] TimingSummary3D summarize_timings_3d(
    const std::vector<double>& samples,
    double matching_baseline_median);

struct ConditionedCrossingPositionTolerance3D {
    double edge = 0.0;
    double point = 0.0;
};

[[nodiscard]] ConditionedCrossingPositionTolerance3D
conditioned_crossing_position_tolerance_3d(
    const geometry3d::NurbsSurfaceCrossing3D& baseline,
    const geometry3d::NurbsSurfaceCrossing3D& candidate,
    double edge_length,
    double fixed_physical_tolerance);

[[nodiscard]] std::vector<std::string> raw_csv_header_3d();
[[nodiscard]] std::vector<std::string> raw_csv_row_3d(
    const RawBenchmarkRecord3D& record);
[[nodiscard]] std::vector<std::string> summary_csv_header_3d();
[[nodiscard]] std::vector<std::string> summary_csv_row_3d(
    const SummaryBenchmarkRecord3D& record);
[[nodiscard]] std::string encode_csv_row_3d(
    const std::vector<std::string>& fields);
[[nodiscard]] std::string serialize_edge_classification_3d(
    const geometry3d::NurbsCartesianEdgeClassification3D& value);
[[nodiscard]] std::string serialize_surface_crossing_3d(
    const geometry3d::NurbsSurfaceCrossing3D& value);
[[nodiscard]] std::string serialize_crossing_owner_3d(
    const P2CrossingOwner3D& value);
[[nodiscard]] std::string compose_mismatch_detail_3d(
    const std::string& baseline,
    const std::string& candidate,
    const std::string& errors,
    const std::string& tolerance);

[[nodiscard]] BenchmarkSupportCase3D run_benchmark_support_case_3d(
    GeometryKind3D geometry,
    int N);

[[nodiscard]] BenchmarkRunResult3D run_nurbs_geometry_preprocess_benchmark_3d(
    const BenchmarkOptions3D& options);

} // namespace kfbim::app3d::benchmark3d
