#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace kfbim {
namespace app3d {

enum class PhaseProfileKind3D {
    GeometryAndDomain,
    SurfaceDofsAndStencils,
    GridPairAndLabelValidation,
    PipelineFixedInitialization,
    CrossingRows,
    NurbsSegmentIntersections,
    RestrictOwnerGeometryPreprocessing,
    TraceOwnerTemplateAssembly,
    ExactFieldsAndOtherSetup,
    EdgeAuxiliaryValues,
    CauchyCoefficients,
    SpreadRhsAssembly,
    FftBulkSolve,
    RestrictContinuedSamples,
    RestrictRecovery,
    GmresAndOtherRoute,
    DiagnosticOutput,
    WallOverhead,
    Count
};

struct PhaseProfileRecord3D {
    double seconds = 0.0;
    std::uint64_t calls = 0;
};

constexpr std::size_t phase_profile_kind_count_3d()
{
    return static_cast<std::size_t>(PhaseProfileKind3D::Count);
}

const char* phase_profile_name_3d(PhaseProfileKind3D kind);
bool phase_profile_is_algorithm_3d(PhaseProfileKind3D kind);

class PhaseProfile3D {
public:
    void add(PhaseProfileKind3D kind,
             double seconds,
             std::uint64_t calls = 1);
    void set(PhaseProfileKind3D kind,
             double seconds,
             std::uint64_t calls = 1);
    const PhaseProfileRecord3D& record(PhaseProfileKind3D kind) const;
    double algorithm_seconds() const;
    double measured_seconds_without_wall_overhead() const;
    void note_timer_reads(std::uint64_t count);
    void set_seconds_per_clock_read(double seconds);
    std::uint64_t timer_reads() const;
    double estimated_timer_overhead_seconds() const;
    void finalize(double wall_seconds);
    double wall_seconds() const;

private:
    std::array<PhaseProfileRecord3D, phase_profile_kind_count_3d()> records_{};
    std::uint64_t timer_reads_ = 0;
    double seconds_per_clock_read_ = 0.0;
    double wall_seconds_ = 0.0;
};

double calibrate_steady_clock_read_seconds_3d(
    std::size_t reads = 200000);

void write_phase_profile_csv_3d(
    const std::filesystem::path& path,
    const std::string& case_id,
    int N,
    const PhaseProfile3D& profile);

} // namespace app3d
} // namespace kfbim
