#include "kfbi_phase_profile_3d.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool value, const std::string& message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <class Function>
void require_invalid_argument(Function&& function, const std::string& message)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int run_tests()
{
    using namespace kfbim::app3d;

    const std::array<const char*, 18> expected_names{{
        "geometry_and_domain",
        "surface_dofs_and_stencils",
        "grid_pair_and_label_validation",
        "pipeline_fixed_initialization",
        "crossing_rows",
        "nurbs_segment_intersections",
        "restrict_owner_geometry_preprocessing",
        "trace_owner_template_assembly",
        "exact_fields_and_other_setup",
        "edge_auxiliary_values",
        "cauchy_coefficients",
        "spread_rhs_assembly",
        "fft_bulk_solve",
        "restrict_continued_samples",
        "restrict_recovery",
        "gmres_and_other_route",
        "diagnostic_output",
        "wall_overhead",
    }};
    require(phase_profile_kind_count_3d() == expected_names.size(),
            "phase count is wrong");
    for (std::size_t q = 0; q < expected_names.size(); ++q) {
        const auto kind = static_cast<PhaseProfileKind3D>(q);
        require(std::string(phase_profile_name_3d(kind)) == expected_names[q],
                "phase name is wrong");
        require(phase_profile_is_algorithm_3d(kind) == (q < 16),
                "algorithm phase classification is wrong");
    }

    PhaseProfile3D profile;
    profile.add(PhaseProfileKind3D::GeometryAndDomain, 2.0, 4);
    profile.add(
        PhaseProfileKind3D::RestrictOwnerGeometryPreprocessing, 0.25, 1);
    profile.add(
        PhaseProfileKind3D::TraceOwnerTemplateAssembly, 0.5, 1);
    profile.add(PhaseProfileKind3D::EdgeAuxiliaryValues, 0.125, 3);
    profile.add(PhaseProfileKind3D::FftBulkSolve, 1.0, 2);
    profile.add(PhaseProfileKind3D::DiagnosticOutput, 0.5, 1);
    profile.note_timer_reads(20);
    profile.set_seconds_per_clock_read(1.0e-7);
    profile.finalize(4.5);

    require(std::abs(profile.algorithm_seconds() - 3.875) < 1.0e-14,
            "algorithm denominator is wrong");
    require(std::abs(profile.measured_seconds_without_wall_overhead() - 4.375)
                < 1.0e-14,
            "measured denominator is wrong");
    require(std::abs(profile.record(
                PhaseProfileKind3D::WallOverhead).seconds - 0.125)
                < 1.0e-14,
            "wall remainder is wrong");
    require(profile.record(
                PhaseProfileKind3D::EdgeAuxiliaryValues).calls == 3,
            "edge-value phase call count is wrong");
    require(profile.record(
                PhaseProfileKind3D::RestrictOwnerGeometryPreprocessing).calls
                == 1
            && profile.record(
                PhaseProfileKind3D::TraceOwnerTemplateAssembly).calls == 1
            && profile.record(
                PhaseProfileKind3D::GmresAndOtherRoute).calls == 0,
            "fixed preprocessing leaves are not disjoint from GMRES");
    require(profile.record(
                PhaseProfileKind3D::GeometryAndDomain).calls == 4,
            "call count is wrong");
    require(std::abs(profile.estimated_timer_overhead_seconds() - 2.0e-6)
                < 1.0e-14,
            "timer overhead estimate is wrong");
    require(std::abs(profile.wall_seconds() - 4.5) < 1.0e-14,
            "wall time is wrong");

    require_invalid_argument(
        [&] {
            profile.add(
                PhaseProfileKind3D::GeometryAndDomain, -1.0);
        },
        "negative duration was accepted");
    require_invalid_argument(
        [&] {
            profile.add(PhaseProfileKind3D::WallOverhead, 1.0);
        },
        "wall overhead was accepted as an additive phase");

    PhaseProfile3D overlap;
    overlap.add(PhaseProfileKind3D::GeometryAndDomain, 2.0);
    require_invalid_argument(
        [&] { overlap.finalize(1.0); },
        "overlapping phases were accepted");

    const double calibrated =
        calibrate_steady_clock_read_seconds_3d(2000);
    require(std::isfinite(calibrated) && calibrated >= 0.0,
            "clock calibration is invalid");

    const std::filesystem::path output =
        std::filesystem::temp_directory_path()
        / "kfbim_phase_profile_3d_test.csv";
    write_phase_profile_csv_3d(output, "rot_axis123_17deg", 128, profile);
    std::ifstream input(output);
    std::string header;
    std::getline(input, header);
    require(header ==
                "case_id,N,phase,seconds,calls,average_seconds,"
                "algorithm_percent,wall_percent,timer_reads,"
                "estimated_timer_overhead_seconds,"
                "estimated_timer_overhead_percent",
            "profile CSV header is wrong");
    std::size_t rows = 0;
    std::string line;
    while (std::getline(input, line))
        ++rows;
    require(rows == phase_profile_kind_count_3d(),
            "profile CSV row count is wrong");
    input.close();
    std::filesystem::remove(output);
    return 0;
}

int main()
{
    try {
        return run_tests();
    } catch (const std::exception& error) {
        std::cerr << "phase profile test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
