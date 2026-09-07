#include "src/support/density/native_nurbs_density_space_3d.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::NativeDensityField3D;
using kfbim::app3d::NativeDensitySeamCoupling3D;
using kfbim::app3d::NativeFeatureEdgeJumpJetConstraints3D;
using kfbim::app3d::NativeFeatureEdgeJumpJetOptions3D;
using kfbim::app3d::NativeNurbsDensityOptions3D;
using kfbim::app3d::NativeNurbsDensitySpace3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::geometry3d::NurbsPatchEdge3D;
using kfbim::geometry3d::NurbsPatchEdgeConnection3D;
using kfbim::geometry3d::NurbsPatchEdgeInterval3D;

constexpr double kAlgebraTolerance = 2.0e-8;
constexpr double kJumpTolerance = 5.0e-8;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double max_abs(const Eigen::MatrixXd& matrix)
{
    return matrix.size() == 0 ? 0.0 : matrix.cwiseAbs().maxCoeff();
}

double relative_error(double actual, double expected)
{
    return std::abs(actual - expected) / std::max(1.0, std::abs(expected));
}

struct ExpectedPolicy {
    int total = 0;
    int broken = 0;
    int strong_c0 = 0;
    int weak_c1 = 0;
    int merged_c0 = 0;
    int collocated_c0 = 0;
};

void check_policy_metadata(const NativeNurbsDensitySpace3D& space,
                           const ExpectedPolicy& expected,
                           const std::string& context)
{
    const auto& seams = space.seams();
    const auto& connections = space.surface().geometric_connections;
    require(static_cast<int>(seams.size()) == expected.total,
            context + " seam count");
    require(seams.size() == connections.size(),
            context + " seam/connection alignment");
    require(space.broken_seam_count() == expected.broken,
            context + " broken seam count");
    require(space.strong_c0_seam_count() == expected.strong_c0,
            context + " strong-C0 seam count");
    require(space.weak_c1_seam_count() == expected.weak_c1,
            context + " weak-C1 seam count");
    require(space.merged_c0_seam_count() == expected.merged_c0,
            context + " merged-C0 seam count");
    require(space.collocated_c0_seam_count() == expected.collocated_c0,
            context + " collocated-C0 seam count");

    int broken = 0;
    int strong_c0 = 0;
    int weak_c1 = 0;
    int merged_c0 = 0;
    int collocated_c0 = 0;
    int partial_rows = 0;
    int weak_rows = 0;
    std::vector<bool> seen(connections.size(), false);
    for (const auto& seam : seams) {
        require(seam.connection >= 0
                    && seam.connection < static_cast<int>(connections.size()),
                context + " valid connection index");
        require(!seen[static_cast<std::size_t>(seam.connection)],
                context + " unique connection index");
        seen[static_cast<std::size_t>(seam.connection)] = true;
        const auto& connection =
            connections[static_cast<std::size_t>(seam.connection)];
        require(seam.geometry_g1 == connection.g1,
                context + " G1 metadata agrees with geometry");
        require(!seam.label.empty(), context + " nonempty seam label");

        const bool coupled =
            seam.coupling != NativeDensitySeamCoupling3D::Broken;
        const bool c1 = seam.coupling
                     == NativeDensitySeamCoupling3D::StrongC0WeakC1;
        broken += !coupled;
        strong_c0 += coupled;
        weak_c1 += c1;
        merged_c0 += seam.strong_c0_merged;
        collocated_c0 += seam.partial_c0_row_count > 0;
        partial_rows += seam.partial_c0_row_count;
        weak_rows += seam.weak_c1_row_count;

        require(c1 == connection.g1,
                context + " C1 policy is exactly the G1 geometry policy");
        if (!connection.g1) {
            require(seam.weak_c1_row_count == 0,
                    context + " feature seam has zero C1 rows");
        } else {
            require(seam.weak_c1_row_count
                        == space.coefficients_per_direction(),
                    context + " G1 seam has one mortar row per trace basis");
        }
        if (!coupled) {
            require(!seam.strong_c0_merged
                        && seam.partial_c0_row_count == 0,
                    context + " broken seam has no C0 constraints");
        } else if (seam.full_edge_pair) {
            require(seam.strong_c0_merged
                        && seam.partial_c0_row_count == 0,
                    context + " full C0 seam is merged exactly");
        } else {
            require(!seam.strong_c0_merged
                        && seam.partial_c0_row_count > 0,
                    context + " partial C0 seam is collocated");
        }
        require(seam.partial_c0_first_row >= 0
                    && seam.partial_c0_first_row
                           + seam.partial_c0_row_count
                       <= space.partial_c0_constraint_count(),
                context + " partial-C0 row range");
        require(seam.weak_c1_first_row >= 0
                    && seam.weak_c1_first_row + seam.weak_c1_row_count
                       <= space.weak_c1_constraint_count(),
                context + " weak-C1 row range");
    }
    require(broken == expected.broken && strong_c0 == expected.strong_c0
                && weak_c1 == expected.weak_c1
                && merged_c0 == expected.merged_c0
                && collocated_c0 == expected.collocated_c0,
            context + " enumerated seam counts");
    require(partial_rows == space.partial_c0_constraint_count(),
            context + " partial-C0 row ownership");
    require(weak_rows == space.weak_c1_constraint_count(),
            context + " weak-C1 row ownership");
}

Eigen::VectorXd constant_coefficients(const NativeNurbsDensitySpace3D& space,
                                      const std::string& context)
{
    const Eigen::MatrixXd& reduction = space.reduction_matrix();
    const Eigen::VectorXd one =
        Eigen::VectorXd::Ones(space.c0_coefficient_count());
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(reduction);
    const Eigen::VectorXd coefficients = qr.solve(one);
    require(coefficients.size() == space.reduced_coefficient_count()
                && coefficients.allFinite(),
            context + " finite reduced constant");
    require((reduction * coefficients - one).lpNorm<Eigen::Infinity>()
                < kAlgebraTolerance,
            context + " constant lies in the reduced space");
    return coefficients;
}

Eigen::VectorXd deterministic_coefficients(int count)
{
    Eigen::VectorXd coefficients(count);
    for (int i = 0; i < count; ++i) {
        const double index = static_cast<double>(i + 1);
        coefficients[i] = std::sin(0.371 * index)
                        + 0.25 * std::cos(0.113 * index);
    }
    return coefficients;
}

Eigen::VectorXd random_coefficients(int count, unsigned int seed)
{
    std::mt19937 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd coefficients(count);
    for (int i = 0; i < count; ++i)
        coefficients[i] = normal(generator);
    return coefficients;
}

void check_dimensions_algebra_and_constants(
    const NativeNurbsDensitySpace3D& space,
    int expected_patches,
    int expected_coefficients,
    bool expect_partial_c0,
    const std::string& context)
{
    const int raw = expected_patches * expected_coefficients
                  * expected_coefficients;
    require(space.patch_count() == expected_patches,
            context + " patch count");
    require(space.coefficients_per_direction() == expected_coefficients,
            context + " coefficient count per direction");
    require(space.elements_per_direction() == expected_coefficients - 3,
            context + " cubic element count");
    require(space.raw_coefficient_count() == raw,
            context + " raw tensor coefficient count");
    require(space.c0_coefficient_count() > 0
                && space.c0_coefficient_count() <= raw,
            context + " merged dimension bounds");
    require(space.continuous_coefficient_count() > 0
                && space.continuous_coefficient_count()
                       <= space.c0_coefficient_count(),
            context + " continuous dimension bounds");
    require(space.reduced_coefficient_count() > 0
                && space.reduced_coefficient_count()
                       <= space.continuous_coefficient_count(),
            context + " reduced dimension bounds");
    require(space.continuous_coefficient_count()
                == space.c0_coefficient_count()
                   - space.partial_c0_constraint_rank(),
            context + " C0 rank/dimension identity");
    require(space.reduced_coefficient_count()
                == space.continuous_coefficient_count()
                   - space.weak_c1_constraint_rank(),
            context + " C1 rank/dimension identity");
    require(space.partial_c0_constraint_rank() >= 0
                && space.partial_c0_constraint_rank()
                       <= space.partial_c0_constraint_count(),
            context + " partial-C0 rank bounds");
    require(space.weak_c1_constraint_rank() > 0
                && space.weak_c1_constraint_rank()
                       <= space.weak_c1_constraint_count(),
            context + " weak-C1 rank bounds");
    require((space.partial_c0_constraint_count() > 0) == expect_partial_c0
                && (space.partial_c0_constraint_rank() > 0)
                       == expect_partial_c0,
            context + " expected partial-C0 reduction");

    const auto& local_to_c0 = space.local_to_c0();
    require(static_cast<int>(local_to_c0.size()) == raw,
            context + " raw-to-C0 map size");
    std::vector<bool> used(
        static_cast<std::size_t>(space.c0_coefficient_count()), false);
    for (int index : local_to_c0) {
        require(index >= 0 && index < space.c0_coefficient_count(),
                context + " raw-to-C0 map range");
        used[static_cast<std::size_t>(index)] = true;
    }
    require(std::all_of(used.begin(), used.end(), [](bool value) {
                return value;
            }),
            context + " raw-to-C0 map is onto");

    const Eigen::MatrixXd& r0 = space.c0_reduction_matrix();
    const Eigen::MatrixXd& r1 = space.c1_reduction_matrix();
    const Eigen::MatrixXd& reduction = space.reduction_matrix();
    const Eigen::MatrixXd& c0 = space.partial_c0_constraint_matrix();
    const Eigen::MatrixXd& c1 = space.weak_c1_constraint_matrix();
    const Eigen::MatrixXd& continuous_c1 =
        space.continuous_weak_c1_constraint_matrix();
    require(r0.rows() == space.c0_coefficient_count()
                && r0.cols() == space.continuous_coefficient_count(),
            context + " R0 shape");
    require(r1.rows() == space.continuous_coefficient_count()
                && r1.cols() == space.reduced_coefficient_count(),
            context + " R1 shape");
    require(reduction.rows() == space.c0_coefficient_count()
                && reduction.cols() == space.reduced_coefficient_count(),
            context + " total reduction shape");
    require(c0.rows() == space.partial_c0_constraint_count()
                && c0.cols() == space.c0_coefficient_count(),
            context + " partial-C0 matrix shape");
    require(c1.rows() == space.weak_c1_constraint_count()
                && c1.cols() == space.c0_coefficient_count(),
            context + " weak-C1 matrix shape");
    require(continuous_c1.rows() == c1.rows()
                && continuous_c1.cols()
                       == space.continuous_coefficient_count(),
            context + " continuous weak-C1 matrix shape");
    require(max_abs(reduction - r0 * r1) < 1.0e-12,
            context + " R=R0*R1");
    require(max_abs(c0 * r0) < kAlgebraTolerance,
            context + " C0*R0 residual");
    require(max_abs(continuous_c1 - c1 * r0) < 1.0e-12,
            context + " continuous C1 composition");
    require(max_abs(continuous_c1 * r1) < kAlgebraTolerance,
            context + " C1*R1 residual");
    require(max_abs(c0 * reduction) < kAlgebraTolerance,
            context + " C0*R residual");
    require(max_abs(c1 * reduction) < kAlgebraTolerance,
            context + " C1*R residual");
    require(space.partial_c0_constraint_residual() < kAlgebraTolerance,
            context + " reported partial-C0 residual");
    require(space.weak_c1_constraint_residual() < kAlgebraTolerance,
            context + " reported weak-C1 residual");
    require(space.reduction_constraint_residual() < kAlgebraTolerance,
            context + " reported total residual");

    const int gauss = space.options().trace_gauss_order;
    const int elements = space.elements_per_direction();
    const int expected_trace =
        expected_patches * elements * elements * gauss * gauss;
    require(static_cast<int>(space.trace_points().size()) == expected_trace,
            context + " trace quadrature count");
    require(space.trace_c0_design().rows() == expected_trace
                && space.trace_c0_design().cols()
                       == space.c0_coefficient_count(),
            context + " sparse trace design shape");
    require(space.trace_weights().size() == expected_trace
                && space.trace_weights().minCoeff() > 0.0,
            context + " positive trace weights");
    require(relative_error(space.trace_weights().sum(), space.surface_area())
                < 2.0e-14,
            context + " quadrature area accumulation");
    const Eigen::VectorXd c0_one =
        Eigen::VectorXd::Ones(space.c0_coefficient_count());
    require((space.trace_c0_design() * c0_one
             - Eigen::VectorXd::Ones(expected_trace))
                .lpNorm<Eigen::Infinity>()
                < 2.0e-13,
            context + " trace partition of unity");
    require(space.c0_mass_row().size() == space.c0_coefficient_count()
                && space.mass_row().size()
                       == space.reduced_coefficient_count(),
            context + " mass row shapes");
    require(relative_error(space.c0_mass_row().dot(c0_one),
                           space.surface_area())
                < 2.0e-14,
            context + " C0 mass reproduces area");

    require(space.constant_reproduction_error() < kAlgebraTolerance,
            context + " reported constant reproduction");
    const Eigen::VectorXd constant = constant_coefficients(space, context);
    const Eigen::VectorXd expanded = space.expand_reduced(constant);
    const Eigen::VectorXd raw_constant = space.expand_raw(constant);
    require((expanded - c0_one).lpNorm<Eigen::Infinity>()
                < kAlgebraTolerance,
            context + " expanded C0 constant");
    require(raw_constant.size() == raw
                && (raw_constant - Eigen::VectorXd::Ones(raw))
                       .lpNorm<Eigen::Infinity>()
                       < kAlgebraTolerance,
            context + " expanded raw constant");
    require(relative_error(space.mass_row().dot(constant),
                           space.surface_area())
                < kAlgebraTolerance,
            context + " reduced mass reproduces area");
    for (int patch = 0; patch < expected_patches; ++patch) {
        for (const auto& uv : std::vector<std::pair<double, double>>{
                 {0.13, 0.21}, {0.51, 0.69}, {0.87, 0.43}}) {
            const auto stencil = space.c0_basis_stencil(
                patch, uv.first, uv.second);
            require(stencil.count > 0 && stencil.count <= 16,
                    context + " compact cubic stencil");
            require(std::abs(stencil.dot(c0_one) - 1.0) < 2.0e-13,
                    context + " stencil partition of unity");
            require(std::abs(space.evaluate(
                                 patch, uv.first, uv.second, constant)
                             - 1.0)
                        < kAlgebraTolerance,
                    context + " pointwise constant reproduction");
            require(std::abs(space.reduced_basis_row(
                                 patch, uv.first, uv.second).dot(constant)
                             - 1.0)
                        < kAlgebraTolerance,
                    context + " reduced-row constant reproduction");
        }
    }

    const Eigen::VectorXd constructed =
        deterministic_coefficients(space.reduced_coefficient_count());
    const Eigen::VectorXd random = random_coefficients(
        space.reduced_coefficient_count(),
        static_cast<unsigned int>(7919 + 17 * expected_patches
                                  + expected_coefficients));
    require(space.max_c0_seam_jump(constant, 23) < kJumpTolerance,
            context + " constant C0 seam jump");
    require(space.max_c0_seam_jump(constructed, 23) < kJumpTolerance,
            context + " constructed C0 seam jump");
    require(space.max_c0_seam_jump(random, 23) < kJumpTolerance,
            context + " random C0 seam jump");
    const double smooth_conormal_jump =
        space.max_smooth_conormal_jump(random, 19);
    require(std::isfinite(smooth_conormal_jump)
                && smooth_conormal_jump < kJumpTolerance,
            context + " sampled smooth co-normal continuity");
}

bool is_u_edge(NurbsPatchEdge3D edge)
{
    return edge == NurbsPatchEdge3D::UMin
        || edge == NurbsPatchEdge3D::UMax;
}

double normalized_parameter(const NativeNurbsSurface3D& surface,
                            const NurbsPatchEdgeInterval3D& interval,
                            double fraction,
                            bool reversed)
{
    const auto& patch =
        surface.patches[static_cast<std::size_t>(interval.patch)];
    const double domain_begin = is_u_edge(interval.edge)
        ? patch.domain_start_v()
        : patch.domain_start_u();
    const double domain_end = is_u_edge(interval.edge)
        ? patch.domain_end_v()
        : patch.domain_end_u();
    const double native = reversed
        ? interval.end - fraction * (interval.end - interval.begin)
        : interval.begin + fraction * (interval.end - interval.begin);
    return (native - domain_begin) / (domain_end - domain_begin);
}

std::pair<double, double> edge_uv(NurbsPatchEdge3D edge, double parameter)
{
    switch (edge) {
    case NurbsPatchEdge3D::UMin:
        return {0.0, parameter};
    case NurbsPatchEdge3D::UMax:
        return {1.0, parameter};
    case NurbsPatchEdge3D::VMin:
        return {parameter, 0.0};
    case NurbsPatchEdge3D::VMax:
        return {parameter, 1.0};
    }
    throw std::logic_error("unknown patch edge");
}

std::pair<std::pair<double, double>, std::pair<double, double>>
connection_uv(const NativeNurbsSurface3D& surface,
              const NurbsPatchEdgeConnection3D& connection,
              double fraction)
{
    const double first = normalized_parameter(
        surface, connection.first, fraction, false);
    const double second = normalized_parameter(
        surface, connection.second, fraction, connection.reversed);
    return {edge_uv(connection.first.edge, first),
            edge_uv(connection.second.edge, second)};
}

void check_l_prism_partial_and_reversed(
    const NativeNurbsDensitySpace3D& value_space)
{
    const auto& surface = value_space.surface();
    const auto& connections = surface.geometric_connections;
    const Eigen::VectorXd coefficients = random_coefficients(
        value_space.reduced_coefficient_count(), 0x5eedU);
    int partial = 0;
    int reversed = 0;
    int partial_reversed = 0;
    int checked = 0;
    for (const auto& seam : value_space.seams()) {
        const auto& connection =
            connections[static_cast<std::size_t>(seam.connection)];
        partial += !seam.full_edge_pair;
        reversed += connection.reversed;
        partial_reversed += !seam.full_edge_pair && connection.reversed;
        if (seam.full_edge_pair && !connection.reversed)
            continue;
        ++checked;
        require(seam.coupling != NativeDensitySeamCoupling3D::Broken,
                "L-prism ValueTrace partial/reversed seam is C0");
        if (!seam.full_edge_pair) {
            require(seam.partial_c0_row_count > 0
                        && !seam.strong_c0_merged,
                    "L-prism partial seam uses collocated C0 rows");
        }
        if (connection.reversed) {
            require(seam.label.find("reversed") != std::string::npos,
                    "L-prism reversed seam is labeled");
        }
        for (double fraction : {0.07, 0.23, 0.51, 0.83, 0.97}) {
            const auto uv = connection_uv(surface, connection, fraction);
            const auto first_geometry = value_space.geometry(
                connection.first.patch, uv.first.first, uv.first.second);
            const auto second_geometry = value_space.geometry(
                connection.second.patch, uv.second.first, uv.second.second);
            require((first_geometry.point - second_geometry.point).norm()
                        < 2.0e-12,
                    "L-prism partial/reversed geometric correspondence");
            const double first_value = value_space.evaluate(
                connection.first.patch,
                uv.first.first,
                uv.first.second,
                coefficients);
            const double second_value = value_space.evaluate(
                connection.second.patch,
                uv.second.first,
                uv.second.second,
                coefficients);
            require(std::abs(first_value - second_value) < kJumpTolerance,
                    "L-prism partial/reversed reduced C0 jump");
        }
    }
    require(partial == 8, "L-prism has eight partial connections");
    require(reversed == 8, "L-prism has eight reversed connections");
    require(partial_reversed == 4,
            "L-prism has four reversed partial connections");
    require(checked == 12,
            "L-prism exercises every partial or reversed connection");
}

void check_broken_feature_mode(const NativeNurbsDensitySpace3D& space,
                               const std::string& context)
{
    const auto& surface = space.surface();
    double largest_norm = 0.0;
    Eigen::RowVectorXd largest;
    for (const auto& seam : space.seams()) {
        if (seam.coupling != NativeDensitySeamCoupling3D::Broken)
            continue;
        const auto& connection = surface.geometric_connections[
            static_cast<std::size_t>(seam.connection)];
        for (double fraction : {0.19, 0.43, 0.71}) {
            const auto uv = connection_uv(surface, connection, fraction);
            const Eigen::RowVectorXd jump =
                space.reduced_basis_row(connection.first.patch,
                                        uv.first.first,
                                        uv.first.second)
                - space.reduced_basis_row(connection.second.patch,
                                          uv.second.first,
                                          uv.second.second);
            if (jump.norm() > largest_norm) {
                largest_norm = jump.norm();
                largest = jump;
            }
        }
    }
    require(largest_norm > 1.0e-4,
            context + " feature traces retain an independent jump mode");
    const Eigen::VectorXd mode = largest.transpose() / largest_norm;
    require(std::abs(largest.dot(mode)) > 1.0e-4,
            context + " constructed broken-feature coefficient mode");
}

Eigen::VectorXd fit_affine_trace(
    const NativeNurbsDensitySpace3D& space,
    const Eigen::Vector3d& gradient,
    double offset,
    const std::string& context)
{
    const int samples_per_direction =
        space.coefficients_per_direction() + 1;
    const int rows = space.patch_count()
                   * samples_per_direction * samples_per_direction;
    Eigen::MatrixXd design(rows, space.reduced_coefficient_count());
    Eigen::VectorXd values(rows);
    int row = 0;
    for (int patch = 0; patch < space.patch_count(); ++patch) {
        for (int i = 0; i < samples_per_direction; ++i) {
            const double u = (static_cast<double>(i) + 0.37)
                           / samples_per_direction;
            for (int j = 0; j < samples_per_direction; ++j) {
                const double v = (static_cast<double>(j) + 0.61)
                               / samples_per_direction;
                const auto point = space.geometry(patch, u, v);
                design.row(row) = space.reduced_basis_row(patch, u, v);
                values[row] = space.options().field
                                  == NativeDensityField3D::ValueTrace
                    ? offset + gradient.dot(point.point)
                    : gradient.dot(point.normal);
                ++row;
            }
        }
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
    const Eigen::VectorXd coefficients = qr.solve(values);
    require(coefficients.size() == space.reduced_coefficient_count()
                && coefficients.allFinite(),
            context + " finite affine coefficients");
    require((design * coefficients - values).lpNorm<Eigen::Infinity>()
                < 5.0e-8,
            context + " affine trace lies in density space");
    return coefficients;
}

void check_feature_edge_jump_jet(
    const NativeNurbsDensitySpace3D& value_space,
    const NativeNurbsDensitySpace3D& normal_space,
    int expected_feature_edges,
    const Eigen::Vector3d& gradient,
    const std::string& context)
{
    const NativeFeatureEdgeJumpJetConstraints3D constraints(
        value_space, normal_space);
    const int test_count = std::max(
        value_space.coefficients_per_direction(),
        normal_space.coefficients_per_direction());
    require(constraints.options().enabled,
            context + " feature jump jet enabled");
    require(constraints.feature_edge_count() == expected_feature_edges,
            context + " feature-edge count");
    require(constraints.constraint_count()
                == 2 * test_count * expected_feature_edges,
            context + " two mortar equations per feature test function");
    require(constraints.value_matrix().rows()
                    == constraints.constraint_count()
                && constraints.value_matrix().cols()
                       == value_space.reduced_coefficient_count(),
            context + " value jump-jet matrix shape");
    require(constraints.normal_target_matrix().rows()
                    == constraints.constraint_count()
                && constraints.normal_target_matrix().cols()
                       == normal_space.reduced_coefficient_count(),
            context + " normal target matrix shape");
    require(constraints.value_matrix().allFinite()
                && constraints.normal_target_matrix().allFinite()
                && constraints.row_scalings().allFinite()
                && constraints.row_scalings().minCoeff() > 0.0,
            context + " finite normalized jump-jet matrices");
    for (int row = 0; row < constraints.constraint_count(); ++row) {
        const double norm = std::hypot(
            constraints.value_matrix().row(row).norm(),
            constraints.normal_target_matrix().row(row).norm());
        require(std::abs(norm - 1.0) < 2.0e-12,
                context + " unit combined mortar row norm");
    }
    require(constraints.maximum_point_mismatch() < 2.0e-11,
            context + " feature patches meet geometrically");
    require(constraints.minimum_abs_dihedral_sine() > 1.0e-3,
            context + " nonsingular feature dihedral angles");
    require(constraints.constant_value_residual() < 5.0e-8,
            context + " constants have zero feature derivative");

    int next_row = 0;
    for (const auto& edge : constraints.edges()) {
        require(edge.connection >= 0
                    && edge.connection
                           < static_cast<int>(
                               value_space.surface()
                                   .geometric_connections.size()),
                context + " valid feature connection");
        require(!value_space.surface()
                     .geometric_connections[
                         static_cast<std::size_t>(edge.connection)]
                     .g1,
                context + " jump jet skips G1 seams");
        require(edge.first_row == next_row
                    && edge.row_count == 2 * test_count
                    && edge.test_function_count == test_count,
                context + " contiguous feature row ownership");
        require(edge.minimum_abs_dihedral_sine
                        >= constraints.options().minimum_abs_dihedral_sine
                    && edge.maximum_abs_dihedral_sine
                           >= edge.minimum_abs_dihedral_sine
                    && !edge.label.empty(),
                context + " feature diagnostics");
        next_row += edge.row_count;
    }
    require(next_row == constraints.constraint_count(),
            context + " every jump-jet row has an owner");

    const Eigen::VectorXd value_coefficients = fit_affine_trace(
        value_space, gradient, 0.37, context + " value");
    const Eigen::VectorXd normal_coefficients = fit_affine_trace(
        normal_space, gradient, 0.0, context + " normal");
    const Eigen::VectorXd target =
        constraints.normal_target(normal_coefficients);
    require((target
             - constraints.normal_target_matrix() * normal_coefficients)
                .lpNorm<Eigen::Infinity>()
                < 2.0e-14,
            context + " normal-target convenience method");
    const Eigen::VectorXd jump_jet_residual =
        constraints.residual(value_coefficients, normal_coefficients);
    const double jump_jet_linf =
        jump_jet_residual.lpNorm<Eigen::Infinity>();
    Eigen::Index worst_jump_jet_row = 0;
    jump_jet_residual.cwiseAbs().maxCoeff(&worst_jump_jet_row);
    std::string worst_edge_label = "unknown";
    int worst_second_patch = -1;
    for (const auto& edge : constraints.edges()) {
        if (worst_jump_jet_row >= edge.first_row
            && worst_jump_jet_row < edge.first_row + edge.row_count) {
            worst_edge_label = edge.label;
            worst_second_patch = value_space.surface()
                .geometric_connections[static_cast<std::size_t>(edge.connection)]
                .second.patch;
            break;
        }
    }
    const Eigen::VectorXd affine_raw =
        value_space.expand_raw(value_coefficients);
    double worst_patch_span = 0.0;
    if (worst_second_patch >= 0) {
        const int patch_size = value_space.coefficients_per_direction()
                             * value_space.coefficients_per_direction();
        const Eigen::VectorXd patch_coefficients = affine_raw.segment(
            worst_second_patch * patch_size, patch_size);
        worst_patch_span = patch_coefficients.maxCoeff()
                         - patch_coefficients.minCoeff();
    }
    require(jump_jet_linf < 5.0e-8,
            context + " affine ambient gradient satisfies jump jet; residual="
                + std::to_string(jump_jet_linf)
                + " row=" + std::to_string(worst_jump_jet_row)
                + " edge=" + worst_edge_label
                + " value=" + std::to_string(
                    (constraints.value_matrix() * value_coefficients)
                        [worst_jump_jet_row])
                + " target=" + std::to_string(target[worst_jump_jet_row])
                + " patch_span=" + std::to_string(worst_patch_span));

    NativeFeatureEdgeJumpJetOptions3D disabled_options;
    disabled_options.enabled = false;
    const NativeFeatureEdgeJumpJetConstraints3D disabled(
        value_space, normal_space, disabled_options);
    require(disabled.feature_edge_count() == 0
                && disabled.constraint_count() == 0
                && disabled.value_matrix().cols()
                       == value_space.reduced_coefficient_count()
                && disabled.normal_target_matrix().cols()
                       == normal_space.reduced_coefficient_count(),
            context + " disabled jump jet is a shaped empty operator");
}

void test_batched_parameter_jet_stencils()
{
    constexpr std::array<std::array<int, 2>, 6> derivatives{{
        {{0, 0}}, {{1, 0}}, {{0, 1}}, {{2, 0}}, {{1, 1}}, {{0, 2}}}};
    for (const GeometryKind3D kind : {GeometryKind3D::HollowCylinder,
                                    GeometryKind3D::LPrism,
                                    GeometryKind3D::UPrism}) {
        for (const int coefficients : {4, 6}) {
            std::vector<double> parameters{0.0, 0.23, 0.5, 0.79, 1.0};
            const int elements = coefficients - 3;
            for (int knot = 1; knot < elements; ++knot) {
                const double parameter = static_cast<double>(knot) / elements;
                parameters.push_back(std::nextafter(parameter, 0.0));
                parameters.push_back(parameter);
                parameters.push_back(std::nextafter(parameter, 1.0));
            }
            for (const NativeDensityField3D field : {
                     NativeDensityField3D::ValueTrace,
                     NativeDensityField3D::NormalTrace}) {
                NativeNurbsDensityOptions3D options;
                options.field = field;
                options.coefficients_per_direction = coefficients;
                options.reduction_backend = kfbim::app3d::
                    NativeDensityReductionBackend3D::BaseOnly;
                NativeNurbsDensitySpace3D space(
                    make_native_nurbs_surface_3d(kind), options);
                const std::string context = "batch parameter jet geometry="
                    + std::to_string(static_cast<int>(kind)) + " field="
                    + std::to_string(static_cast<int>(field)) + " n="
                    + std::to_string(coefficients);
                for (int patch = 0; patch < space.patch_count(); ++patch) {
                    for (const double u : parameters) {
                        for (const double v : parameters) {
                            const auto jet = space.c0_parameter_jet_stencils(
                                patch, u, v);
                            for (std::size_t row = 0; row < jet.size(); ++row) {
                                const auto expected =
                                    space.c0_parameter_derivative_stencil(
                                        patch, u, v,
                                        derivatives[row][0], derivatives[row][1]);
                                require(jet[row].count == expected.count,
                                        context + " active count");
                                // Exact equality also verifies zero-product
                                // pruning, C0 index order, and cancellation
                                // after contributions share a C0 coordinate.
                                require(jet[row].indices == expected.indices,
                                        context + " C0 indices");
                                require(jet[row].weights == expected.weights,
                                        context + " coefficient weights");
                            }
                        }
                    }
                }
                for (const int invalid_patch : {-1, space.patch_count()}) {
                    bool rejected = false;
                    try {
                        (void)space.c0_parameter_jet_stencils(
                            invalid_patch, 0.5, 0.5);
                    } catch (const std::out_of_range&) {
                        rejected = true;
                    }
                    require(rejected, context + " invalid patch rejected");
                }
            }
        }
    }
}

void test_hollow_cylinder()
{
    NativeNurbsDensityOptions3D underintegrated;
    underintegrated.coefficients_per_direction = 4;
    underintegrated.mortar_gauss_order = 3;
    bool rejected_underintegration = false;
    try {
        NativeNurbsDensitySpace3D invalid(
            make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder),
            underintegrated);
    } catch (const std::invalid_argument&) {
        rejected_underintegration = true;
    }
    require(rejected_underintegration,
            "cubic density rejects an underintegrated mortar rule");

    // This is the actual coarse-grid resolution selected at N=32.  A
    // one-span cubic trace has four modes, so the mortar construction must
    // constrain all four rather than merely be finite at sample points.
    NativeNurbsDensitySpace3D coarse_value(
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder),
        4,
        NativeDensityField3D::ValueTrace);
    check_policy_metadata(coarse_value, {32, 0, 32, 16, 32, 0},
                          "coarse cylinder ValueTrace");
    check_dimensions_algebra_and_constants(
        coarse_value, 16, 4, false, "coarse cylinder ValueTrace");

    NativeNurbsDensitySpace3D coarse_normal(
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder),
        4,
        NativeDensityField3D::NormalTrace);
    check_policy_metadata(coarse_normal, {32, 16, 16, 16, 16, 0},
                          "coarse cylinder NormalTrace");
    check_dimensions_algebra_and_constants(
        coarse_normal, 16, 4, false, "coarse cylinder NormalTrace");
    check_feature_edge_jump_jet(
        coarse_value,
        coarse_normal,
        16,
        Eigen::Vector3d(0.0, 0.0, 0.7),
        "coarse cylinder");

    NativeNurbsDensitySpace3D value(
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder),
        5,
        NativeDensityField3D::ValueTrace);
    check_policy_metadata(value, {32, 0, 32, 16, 32, 0},
                          "cylinder ValueTrace");
    check_dimensions_algebra_and_constants(
        value, 16, 5, false, "cylinder ValueTrace");

    NativeNurbsDensitySpace3D normal(
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder),
        5,
        NativeDensityField3D::NormalTrace);
    check_policy_metadata(normal, {32, 16, 16, 16, 16, 0},
                          "cylinder NormalTrace");
    check_dimensions_algebra_and_constants(
        normal, 16, 5, false, "cylinder NormalTrace");
    require(normal.c0_coefficient_count() > value.c0_coefficient_count(),
            "cylinder broken normal features add C0 degrees of freedom");
    require(normal.reduced_coefficient_count()
                > value.reduced_coefficient_count(),
            "cylinder broken normal features add reduced degrees of freedom");
    check_broken_feature_mode(normal, "cylinder NormalTrace");
}

void test_l_prism()
{
    NativeNurbsDensityOptions3D value_options;
    value_options.field = NativeDensityField3D::ValueTrace;
    value_options.coefficients_per_direction = 6;
    NativeNurbsDensitySpace3D value(
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism), value_options);
    check_policy_metadata(value, {26, 0, 26, 4, 18, 8},
                          "L-prism ValueTrace");
    check_dimensions_algebra_and_constants(
        value, 12, 6, true, "L-prism ValueTrace");
    check_l_prism_partial_and_reversed(value);

    NativeNurbsDensityOptions3D normal_options = value_options;
    normal_options.field = NativeDensityField3D::NormalTrace;
    NativeNurbsDensitySpace3D normal(
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism), normal_options);
    check_policy_metadata(normal, {26, 22, 4, 4, 4, 0},
                          "L-prism NormalTrace");
    check_dimensions_algebra_and_constants(
        normal, 12, 6, false, "L-prism NormalTrace");
    require(normal.c0_coefficient_count() > value.c0_coefficient_count(),
            "L-prism broken normal features add C0 degrees of freedom");
    require(normal.reduced_coefficient_count()
                > value.reduced_coefficient_count(),
            "L-prism broken normal features add reduced degrees of freedom");
    check_broken_feature_mode(normal, "L-prism NormalTrace");
    check_feature_edge_jump_jet(
        value,
        normal,
        22,
        Eigen::Vector3d(0.4, -0.7, 0.5),
        "L-prism");
}

} // namespace

int main()
{
    try {
        test_batched_parameter_jet_stencils();
        test_hollow_cylinder();
        test_l_prism();
        std::cout << "native NURBS density-space tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native NURBS density-space test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
