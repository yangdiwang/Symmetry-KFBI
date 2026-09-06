#include "direct_coefficient_cauchy_3d.hpp"
#include "harmonic_polynomial_space_3d.hpp"
#include "native_nurbs_surface_3d.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {

using namespace kfbim::app3d;
constexpr double kSvdCutoff = 3.0e-12;
constexpr int kSampleCount = 32;
using JetRows = std::array<NativeDensityC0Stencil3D, 6>;

struct Sample {
    int patch;
    double u;
    double v;
    NativeSurfaceParameterJet3D geometry;
    LocalOrthonormalFrame3D frame;
};

struct Timing {
    double cpu_seconds;
    double wall_seconds;
    double checksum;
};

double process_cpu_seconds()
{
#ifdef _WIN32
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user))
        throw std::runtime_error("GetProcessTimes failed");
    auto ticks = [](const FILETIME& value) {
        return (static_cast<std::uint64_t>(value.dwHighDateTime) << 32)
            | static_cast<std::uint64_t>(value.dwLowDateTime);
    };
    return static_cast<double>(ticks(kernel) + ticks(user)) / 1.0e7;
#else
    const std::clock_t ticks = std::clock();
    if (ticks == static_cast<std::clock_t>(-1))
        throw std::runtime_error("process CPU clock is unavailable");
    return static_cast<double>(ticks) / CLOCKS_PER_SEC;
#endif
}

template <typename Work>
Timing measure(int iterations, const Work& work)
{
    double warmup = 0.0;
    for (int i = 0; i < 8; ++i)
        warmup += work(i);
    if (!std::isfinite(warmup))
        throw std::runtime_error("non-finite warmup checksum");
    const double cpu_start = process_cpu_seconds();
    const auto wall_start = std::chrono::steady_clock::now();
    double checksum = 0.0;
    for (int i = 0; i < iterations; ++i)
        checksum += work(i);
    const auto wall_end = std::chrono::steady_clock::now();
    const double cpu_end = process_cpu_seconds();
    if (!std::isfinite(checksum))
        throw std::runtime_error("non-finite measured checksum");
    return {cpu_end - cpu_start,
            std::chrono::duration<double>(wall_end - wall_start).count(),
            checksum};
}

void print_row(const char* name, const char* variant, int iterations,
               int samples, const Timing& result, double difference)
{
    std::cout << name << ',' << variant << ',' << iterations << ',' << samples
              << ',' << result.cpu_seconds << ',' << result.wall_seconds
              << ',' << result.checksum << ',' << difference << '\n';
}

double stencil_checksum(const NativeDensityC0Stencil3D& row)
{
    double sum = 0.001 * row.count;
    for (int q = 0; q < row.count; ++q) {
        const auto i = static_cast<std::size_t>(q);
        sum += row.weights[i] * (1.0 + 0.031 * row.indices[i]);
    }
    return sum;
}

template <typename Rows>
double rows_checksum(const Rows& rows)
{
    double sum = 0.0;
    for (std::size_t q = 0; q < rows.size(); ++q)
        sum += static_cast<double>(q + 1) * stencil_checksum(rows[q]);
    return sum;
}

template <typename Plan>
double plan_checksum(const Plan& plan)
{
    return rows_checksum(plan.cauchy_rows)
        + 0.01 * plan.diagnostics.parameter_to_tangent_condition
        + 0.03 * plan.diagnostics.parameter_to_tangent_determinant
        + 0.07 * (plan.graph_hessian.h11 + plan.graph_hessian.h22);
}

template <typename Rows>
double rows_difference(const Rows& first, const Rows& second)
{
    double difference = 0.0;
    for (std::size_t r = 0; r < first.size(); ++r) {
        if (first[r].count != second[r].count)
            throw std::runtime_error("equivalence check: support count differs");
        for (int q = 0; q < first[r].count; ++q) {
            const auto i = static_cast<std::size_t>(q);
            if (first[r].indices[i] != second[r].indices[i])
                throw std::runtime_error("equivalence check: support differs");
            const double a = first[r].weights[i];
            const double b = second[r].weights[i];
            if (!std::isfinite(a) || !std::isfinite(b))
                throw std::runtime_error("equivalence check: non-finite row");
            difference = std::max(difference, std::abs(a - b));
        }
    }
    return difference;
}

JetRows separate_jet_rows(const NativeNurbsDensitySpace3D& density,
                         const Sample& sample)
{
    constexpr std::array<std::array<int, 2>, 6> derivatives{{
        {{0, 0}}, {{1, 0}}, {{0, 1}}, {{2, 0}}, {{1, 1}}, {{0, 2}}}};
    JetRows result;
    for (std::size_t q = 0; q < result.size(); ++q) {
        result[q] = density.c0_parameter_derivative_stencil(
            sample.patch, sample.u, sample.v,
            derivatives[q][0], derivatives[q][1]);
    }
    return result;
}

struct InverseResult {
    Eigen::MatrixXd inverse;
    double condition;
};

InverseResult separate_svd(const Eigen::MatrixXd& matrix)
{
    const Eigen::JacobiSVD<Eigen::MatrixXd> condition_svd(matrix);
    const auto& singular = condition_svd.singularValues();
    return {svd_pseudoinverse_3d(matrix, kSvdCutoff),
            singular[0] / singular[singular.size() - 1]};
}

InverseResult reused_svd(const Eigen::MatrixXd& matrix)
{
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& singular = svd.singularValues();
    return {svd_pseudoinverse_from_decomposition_3d(svd, kSvdCutoff),
            singular[0] / singular[singular.size() - 1]};
}

double inverse_checksum(const InverseResult& result)
{
    double sum = result.condition;
    for (Eigen::Index i = 0; i < result.inverse.rows(); ++i) {
        for (Eigen::Index j = 0; j < result.inverse.cols(); ++j)
            sum += result.inverse(i, j) * (1.0 + 0.017 * i + 0.023 * j);
    }
    return sum;
}

void run(int iterations)
{
    Eigen::setNbThreads(1);
    NativeNurbsDensityOptions3D options;
    options.field = NativeDensityField3D::ValueTrace;
    options.reduction_backend = NativeDensityReductionBackend3D::BaseOnly;
    options.coefficients_per_direction = 7;
    const auto surface = make_native_nurbs_surface_3d(
        GeometryKind3D::HollowCylinder);
    const NativeNurbsDensitySpace3D value_density(surface, options);
    options.field = NativeDensityField3D::NormalTrace;
    const NativeNurbsDensitySpace3D normal_density(surface, options);
    std::vector<Sample> samples;
    samples.reserve(kSampleCount);
    for (int i = 0; i < kSampleCount; ++i) {
        const int patch = i % value_density.patch_count();
        const double u = 0.03 + 0.94 * (i + 1.0) / (kSampleCount + 1.0);
        const double v = 0.03
            + 0.94 * ((11 * i) % kSampleCount + 1.0) / (kSampleCount + 1.0);
        const auto geometry = native_surface_parameter_jet_3d(
            value_density, patch, u, v);
        const auto frame = make_local_orthonormal_frame_3d(
            geometry.normal, geometry.x_u + 0.23 * geometry.x_v);
        samples.push_back({patch, u, v, geometry, frame});
    }
    auto old_value = [&](const Sample& s) {
        return build_direct_coefficient_value_jet_plan_3d(
            value_density, s.patch, s.u, s.v, s.frame);
    };
    auto new_value = [&](const Sample& s) {
        return build_direct_coefficient_value_jet_plan_3d(
            value_density, s.patch, s.u, s.v, s.geometry, s.frame);
    };
    auto old_normal = [&](const Sample& s) {
        return build_direct_coefficient_normal_jet_plan_3d(
            normal_density, s.patch, s.u, s.v, s.frame);
    };
    auto new_normal = [&](const Sample& s) {
        return build_direct_coefficient_normal_jet_plan_3d(
            normal_density, s.patch, s.u, s.v, s.geometry, s.frame);
    };
    double jet_difference = 0.0;
    double value_difference = 0.0;
    double normal_difference = 0.0;
    for (const Sample& s : samples) {
        jet_difference = std::max(jet_difference, rows_difference(
            separate_jet_rows(value_density, s),
            value_density.c0_parameter_jet_stencils(s.patch, s.u, s.v)));
        const auto va = old_value(s), vb = new_value(s);
        const auto na = old_normal(s), nb = new_normal(s);
        value_difference = std::max(
            {value_difference, rows_difference(va.cauchy_rows, vb.cauchy_rows),
             std::abs(plan_checksum(va) - plan_checksum(vb))});
        normal_difference = std::max(
            {normal_difference, rows_difference(na.cauchy_rows, nb.cauchy_rows),
             std::abs(plan_checksum(na) - plan_checksum(nb))});
    }
    if (jet_difference != 0.0 || value_difference != 0.0
        || normal_difference != 0.0) {
        throw std::runtime_error("geometry/density reuse is not exactly equal");
    }
    // A fixed, full-rank rectangular design isolates decomposition reuse.
    Eigen::MatrixXd matrix(48, 16);
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
            matrix(i, j) = (i == j ? 2.0 : 0.0)
                + 0.13 * std::sin(0.37 * (i + 1) * (j + 2))
                + 0.07 * std::cos(0.19 * (i + 3) * (j + 1));
        }
    }
    const auto a = separate_svd(matrix), b = reused_svd(matrix);
    const double svd_difference = std::max(
        (a.inverse - b.inverse).cwiseAbs().maxCoeff(),
        std::abs(a.condition - b.condition));
    const double svd_scale = std::max(
        {1.0, a.condition, a.inverse.cwiseAbs().maxCoeff()});
    if (!std::isfinite(svd_difference)
        || svd_difference > 1.0e-12 * svd_scale) {
        throw std::runtime_error("SVD reuse equivalence check failed");
    }
    auto sample_at = [&](int i) -> const Sample& {
        return samples[static_cast<std::size_t>(i % kSampleCount)];
    };
    std::cout << std::setprecision(17)
              << "case,variant,iterations,samples,cpu_seconds,wall_seconds,"
                 "checksum,equivalence_linf\n";
    print_row("density_jet6", "six_single_rows", iterations, kSampleCount,
        measure(iterations, [&](int i) {
            return rows_checksum(separate_jet_rows(value_density, sample_at(i)));
        }), jet_difference);
    print_row("density_jet6", "batched_rows", iterations, kSampleCount,
        measure(iterations, [&](int i) {
            const auto& s = sample_at(i);
            return rows_checksum(value_density.c0_parameter_jet_stencils(
                s.patch, s.u, s.v));
        }), jet_difference);
    // Geometry/frame preparation is outside BOTH timings: the real caller
    // already has it, while the old entry evaluates that same geometry again.
    print_row("value_plan", "geometry_reevaluated", iterations, kSampleCount,
        measure(iterations, [&](int i) {
            return plan_checksum(old_value(sample_at(i)));
        }), value_difference);
    print_row("value_plan", "geometry_reused", iterations, kSampleCount,
        measure(iterations, [&](int i) {
            return plan_checksum(new_value(sample_at(i)));
        }), value_difference);
    print_row("normal_plan", "geometry_reevaluated", iterations, kSampleCount,
        measure(iterations, [&](int i) {
            return plan_checksum(old_normal(sample_at(i)));
        }), normal_difference);
    print_row("normal_plan", "geometry_reused", iterations, kSampleCount,
        measure(iterations, [&](int i) {
            return plan_checksum(new_normal(sample_at(i)));
        }), normal_difference);
    print_row("svd_48x16", "two_decompositions", iterations, 1,
        measure(iterations, [&](int) {
            return inverse_checksum(separate_svd(matrix));
        }), svd_difference);
    print_row("svd_48x16", "one_decomposition", iterations, 1,
        measure(iterations, [&](int) {
            return inverse_checksum(reused_svd(matrix));
        }), svd_difference);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        int iterations = 2000;
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "usage: data_reuse_3d_benchmark [--iterations N]\n"
                         "Default: 2000 repetitions per variant; no PDE solve.\n"
                         "All equivalence checks precede CSV timing output.\n";
            return 0;
        }
        if (argc != 1) {
            if (argc != 3 || std::string(argv[1]) != "--iterations")
                throw std::invalid_argument("expected --iterations N or --help");
            std::size_t used = 0;
            const std::string text = argv[2];
            iterations = std::stoi(text, &used);
            if (used != text.size() || iterations < 1 || iterations > 1000000)
                throw std::invalid_argument("iterations must be in [1,1000000]");
        }
        run(iterations);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "data_reuse_3d_benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
