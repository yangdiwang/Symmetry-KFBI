#include "kfbi_phase_profile_3d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace kfbim {
namespace app3d {
namespace {

constexpr std::array<const char*, phase_profile_kind_count_3d()>
    kPhaseNames{{
        "geometry_and_domain",
        "surface_dofs_and_stencils",
        "grid_pair_and_label_validation",
        "pipeline_fixed_initialization",
        "crossing_rows",
        "nurbs_segment_intersections",
        "trace_owner_template_assembly",
        "exact_fields_and_other_setup",
        "cauchy_coefficients",
        "spread_rhs_assembly",
        "fft_bulk_solve",
        "restrict_continued_samples",
        "restrict_recovery",
        "gmres_and_other_route",
        "diagnostic_output",
        "wall_overhead",
    }};

std::size_t phase_index(PhaseProfileKind3D kind)
{
    const std::size_t result = static_cast<std::size_t>(kind);
    if (result >= phase_profile_kind_count_3d())
        throw std::invalid_argument("invalid 3D phase-profile kind");
    return result;
}

void validate_duration(double seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0)
        throw std::invalid_argument(
            "3D phase-profile duration must be finite and nonnegative");
}

void validate_direct_phase(PhaseProfileKind3D kind)
{
    phase_index(kind);
    if (kind == PhaseProfileKind3D::WallOverhead)
        throw std::invalid_argument(
            "wall overhead is derived by phase-profile finalization");
}

} // namespace

const char* phase_profile_name_3d(PhaseProfileKind3D kind)
{
    return kPhaseNames[phase_index(kind)];
}

bool phase_profile_is_algorithm_3d(PhaseProfileKind3D kind)
{
    const std::size_t index = phase_index(kind);
    return index
        < static_cast<std::size_t>(PhaseProfileKind3D::DiagnosticOutput);
}

void PhaseProfile3D::add(PhaseProfileKind3D kind,
                         double seconds,
                         std::uint64_t calls)
{
    validate_direct_phase(kind);
    validate_duration(seconds);
    PhaseProfileRecord3D& item = records_[phase_index(kind)];
    item.seconds += seconds;
    item.calls += calls;
}

void PhaseProfile3D::set(PhaseProfileKind3D kind,
                         double seconds,
                         std::uint64_t calls)
{
    validate_direct_phase(kind);
    validate_duration(seconds);
    records_[phase_index(kind)] = {seconds, calls};
}

const PhaseProfileRecord3D&
PhaseProfile3D::record(PhaseProfileKind3D kind) const
{
    return records_[phase_index(kind)];
}

double PhaseProfile3D::algorithm_seconds() const
{
    double result = 0.0;
    for (std::size_t q = 0;
         q < static_cast<std::size_t>(
                 PhaseProfileKind3D::DiagnosticOutput);
         ++q) {
        result += records_[q].seconds;
    }
    return result;
}

double PhaseProfile3D::measured_seconds_without_wall_overhead() const
{
    double result = 0.0;
    for (std::size_t q = 0;
         q < static_cast<std::size_t>(PhaseProfileKind3D::WallOverhead);
         ++q) {
        result += records_[q].seconds;
    }
    return result;
}

void PhaseProfile3D::note_timer_reads(std::uint64_t count)
{
    timer_reads_ += count;
}

void PhaseProfile3D::set_seconds_per_clock_read(double seconds)
{
    validate_duration(seconds);
    seconds_per_clock_read_ = seconds;
}

std::uint64_t PhaseProfile3D::timer_reads() const
{
    return timer_reads_;
}

double PhaseProfile3D::estimated_timer_overhead_seconds() const
{
    return seconds_per_clock_read_
        * static_cast<double>(timer_reads_);
}

void PhaseProfile3D::finalize(double wall_seconds)
{
    validate_duration(wall_seconds);
    const double remainder =
        wall_seconds - measured_seconds_without_wall_overhead();
    const double tolerance = std::max(1.0e-9, 1.0e-8 * wall_seconds);
    if (remainder < -tolerance) {
        throw std::invalid_argument(
            "3D phase-profile leaf durations overlap wall time");
    }
    records_[phase_index(PhaseProfileKind3D::WallOverhead)] = {
        std::max(0.0, remainder), 1};
    wall_seconds_ = wall_seconds;
}

double PhaseProfile3D::wall_seconds() const
{
    return wall_seconds_;
}

double calibrate_steady_clock_read_seconds_3d(std::size_t reads)
{
    if (reads == 0)
        throw std::invalid_argument(
            "clock calibration requires at least one read");
    using Clock = std::chrono::steady_clock;
    volatile Clock::duration::rep sink = 0;
    const Clock::time_point begin = Clock::now();
    for (std::size_t q = 0; q < reads; ++q) {
        sink ^= Clock::now().time_since_epoch().count();
    }
    const double elapsed = std::chrono::duration<double>(
        Clock::now() - begin).count();
    if (sink == std::numeric_limits<Clock::duration::rep>::min())
        throw std::logic_error("unreachable clock calibration state");
    return elapsed / static_cast<double>(reads);
}

void write_phase_profile_csv_3d(
    const std::filesystem::path& path,
    const std::string& case_id,
    int N,
    const PhaseProfile3D& profile)
{
    if (N <= 0)
        throw std::invalid_argument(
            "phase-profile grid level must be positive");
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error(
            "failed to open 3D phase-profile CSV: " + path.string());

    output << std::setprecision(17)
           << "case_id,N,phase,seconds,calls,average_seconds,"
              "algorithm_percent,wall_percent,timer_reads,"
              "estimated_timer_overhead_seconds,"
              "estimated_timer_overhead_percent\n";
    const double algorithm_seconds = profile.algorithm_seconds();
    const double wall_seconds = profile.wall_seconds();
    const double timer_overhead =
        profile.estimated_timer_overhead_seconds();
    const double timer_overhead_percent = wall_seconds > 0.0
        ? 100.0 * timer_overhead / wall_seconds : 0.0;
    for (std::size_t q = 0; q < phase_profile_kind_count_3d(); ++q) {
        const auto kind = static_cast<PhaseProfileKind3D>(q);
        const PhaseProfileRecord3D& item = profile.record(kind);
        const double average = item.calls > 0
            ? item.seconds / static_cast<double>(item.calls) : 0.0;
        const double algorithm_percent =
            phase_profile_is_algorithm_3d(kind)
                && algorithm_seconds > 0.0
            ? 100.0 * item.seconds / algorithm_seconds : 0.0;
        const double wall_percent = wall_seconds > 0.0
            ? 100.0 * item.seconds / wall_seconds : 0.0;
        output << case_id << ',' << N << ','
               << phase_profile_name_3d(kind) << ','
               << item.seconds << ',' << item.calls << ','
               << average << ',' << algorithm_percent << ','
               << wall_percent << ',' << profile.timer_reads() << ','
               << timer_overhead << ',' << timer_overhead_percent << '\n';
    }
}

} // namespace app3d
} // namespace kfbim
