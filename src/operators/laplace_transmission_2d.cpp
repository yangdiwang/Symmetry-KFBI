#include "laplace_transmission_2d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "../gmres/gmres.hpp"
#include "../local_cauchy/jump_data.hpp"
#include "../bulk_solvers/zfft_bc_type.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "detail/laplace_operator_utils.hpp"

namespace kfbim {

namespace {

constexpr const char* kContext = "LaplaceTransmission2D";

bool profile_solve_2d()
{
    return std::getenv("KFBIM_PROFILE_SOLVE_2D") != nullptr;
}

struct StageTimer2D {
    using Clock = std::chrono::steady_clock;

    Clock::time_point start = Clock::now();
    Clock::time_point last = start;

    double lap()
    {
        const Clock::time_point now = Clock::now();
        const std::chrono::duration<double> elapsed = now - last;
        last = now;
        return elapsed.count();
    }

    double total() const
    {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }
};

double profile_total_without_potential(const LaplacePotentialEvalProfile2D& profile,
                                       double total)
{
    return std::max(0.0, total - profile.total_sec);
}

void print_potential_profile_2d(const char* label,
                                const LaplacePotentialEvalProfile2D& profile)
{
    if (profile.calls == 0)
        return;

    std::printf("      %-18s calls=%3d total %.3fs avg %.4fs | rhs %.3fs spread %.3fs bulk %.3fs restrict %.3fs avg_trace %.3fs\n",
                label,
                profile.calls,
                profile.total_sec,
                profile.total_sec / static_cast<double>(profile.calls),
                profile.rhs_copy_sec,
                profile.spread_sec,
                profile.bulk_solve_sec,
                profile.restrict_sec,
                profile.average_sec);
}

void require_vector_size(const char* name, Eigen::Index actual, Eigen::Index expected)
{
    operators_detail::require_vector_size(kContext, name, actual, expected);
}

double checked_lambda_sq(const char* phase, double beta, double kappa_sq)
{
    return operators_detail::checked_lambda_sq(kContext, phase, beta, kappa_sq);
}

void require_common_ratio(double lambda_sq_int, double lambda_sq_ext)
{
    operators_detail::require_common_ratio(kContext, lambda_sq_int, lambda_sq_ext);
}

void validate_rhs_derivs(const char* name,
                         const std::vector<Eigen::VectorXd>& derivs,
                         int n_iface)
{
    operators_detail::validate_rhs_derivs(kContext, name, derivs, n_iface);
}

void validate_rhs_data(const LaplaceTransmissionRhsData2D& rhs_data,
                       int n_iface,
                       int n_dof)
{
    require_vector_size("rhs_data.reduced_rhs_bulk",
                        rhs_data.reduced_rhs_bulk.size(),
                        n_dof);
    validate_rhs_derivs("rhs_data.reduced_rhs_int_derivs",
                        rhs_data.reduced_rhs_int_derivs,
                        n_iface);
    validate_rhs_derivs("rhs_data.reduced_rhs_ext_derivs",
                        rhs_data.reduced_rhs_ext_derivs,
                        n_iface);
}

std::vector<Eigen::VectorXd> combine_rhs_derivs(
    const std::vector<Eigen::VectorXd>& int_derivs,
    const std::vector<Eigen::VectorXd>& ext_derivs)
{
    return operators_detail::combine_rhs_derivs(kContext, int_derivs, ext_derivs);
}

std::vector<Eigen::VectorXd> scaled_rhs_derivs(
    const std::vector<Eigen::VectorXd>& derivs,
    double scale)
{
    return operators_detail::scaled_rhs_derivs(derivs, scale);
}

std::vector<Eigen::VectorXd> zero_rhs_derivs(int n_iface)
{
    return operators_detail::zero_rhs_derivs(n_iface);
}

std::vector<LaplaceJumpData2D> make_jumps(
    int                                n_iface,
    const Eigen::VectorXd&             u_jump,
    const Eigen::VectorXd&             un_jump,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    return operators_detail::make_laplace_jumps_2d(
        kContext, n_iface, u_jump, un_jump, rhs_derivs);
}

void split_phase_rhs(const GridPair2D&    grid_pair,
                     const Eigen::VectorXd& rhs_bulk,
                     Eigen::VectorXd&       rhs_int,
                     Eigen::VectorXd&       rhs_ext)
{
    operators_detail::split_phase_rhs(
        kContext, grid_pair, rhs_bulk, rhs_int, rhs_ext);
}

LaplaceTransmissionOptions2D options_from_bulk_bc(ZfftBcType bulk_bc)
{
    LaplaceTransmissionOptions2D options;
    options.bulk_bc = bulk_bc;
    return options;
}

std::shared_ptr<const Interface2D> make_effective_interface(
    const Interface2D& iface,
    const CornerPatchConfig2D& corner_patch)
{
    return std::make_shared<Interface2D>(
        apply_corner_patch_config_2d(iface, corner_patch));
}

} // namespace

LaplaceTransmission2D::LaplaceTransmission2D(
    const CartesianGrid2D&            grid,
    const Interface2D&                iface,
    LaplaceTransmissionMode2D         mode,
    LaplaceTransmissionCoefficients2D coefficients,
    ZfftBcType                        bulk_bc)
    : LaplaceTransmission2D(grid,
                            iface,
                            mode,
                            coefficients,
                            options_from_bulk_bc(bulk_bc))
{}

LaplaceTransmission2D::LaplaceTransmission2D(
    const CartesianGrid2D&            grid,
    const Interface2D&                iface,
    LaplaceTransmissionMode2D         mode,
    LaplaceTransmissionCoefficients2D coefficients,
    LaplaceTransmissionOptions2D      options)
    : effective_iface_(make_effective_interface(iface, options.corner_patch))
    , grid_pair_(grid, *effective_iface_)
    , mode_(mode)
    , coefficients_(coefficients)
    , bulk_bc_(options.bulk_bc)
    , lambda_sq_int_(checked_lambda_sq("interior",
                                       coefficients.beta_int,
                                       coefficients.kappa_sq_int))
    , lambda_sq_ext_(checked_lambda_sq("exterior",
                                       coefficients.beta_ext,
                                       coefficients.kappa_sq_ext))
    , spread_int_(grid_pair_,
                  lambda_sq_int_,
                  options.correction_method,
                  options.restrict_stencil_radius,
                  options.corner_patch_solver)
    , spread_ext_(grid_pair_,
                  lambda_sq_ext_,
                  options.correction_method,
                  options.restrict_stencil_radius,
                  options.corner_patch_solver)
    , bulk_solver_int_(grid, bulk_bc_, lambda_sq_int_)
    , bulk_solver_ext_(grid, bulk_bc_, lambda_sq_ext_)
    , restrict_op_(grid_pair_, options.restrict_stencil_radius)
    , potentials_int_(spread_int_, bulk_solver_int_, restrict_op_)
    , potentials_ext_(spread_ext_, bulk_solver_ext_, restrict_op_)
{
    if (mode_ == LaplaceTransmissionMode2D::CommonRatio)
        require_common_ratio(lambda_sq_int_, lambda_sq_ext_);
}

void LaplaceTransmission2D::apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const
{
    switch (mode_) {
    case LaplaceTransmissionMode2D::CommonRatio:
        apply_common_ratio(x, y);
        return;
    case LaplaceTransmissionMode2D::DifferentRatios:
        apply_different_ratios(x, y);
        return;
    }

    throw std::invalid_argument(std::string(kContext) + ": unsupported transmission mode");
}

int LaplaceTransmission2D::problem_size() const
{
    const int n_iface = grid_pair_.interface().num_points();
    switch (mode_) {
    case LaplaceTransmissionMode2D::CommonRatio:
        return n_iface;
    case LaplaceTransmissionMode2D::DifferentRatios:
        return 2 * n_iface;
    }

    throw std::invalid_argument(std::string(kContext) + ": unsupported transmission mode");
}

Eigen::VectorXd LaplaceTransmission2D::apply_dirichlet_boundary_elimination(
    const Eigen::VectorXd& reduced_rhs_bulk,
    const Eigen::VectorXd& outer_dirichlet_values) const
{
    const auto& grid = grid_pair_.grid();
    const int n_dof = grid.num_dofs();
    require_vector_size("reduced_rhs_bulk", reduced_rhs_bulk.size(), n_dof);
    if (bulk_bc_ != ZfftBcType::Dirichlet) {
        if (outer_dirichlet_values.size() != 0) {
            throw std::invalid_argument(
                std::string(kContext)
                + ": outer Dirichlet values are only valid with Dirichlet bulk BC");
        }
        return reduced_rhs_bulk;
    }
    if (outer_dirichlet_values.size() == 0)
        return reduced_rhs_bulk;

    return structured_grid::apply_dirichlet_boundary_elimination(
        kContext, grid, reduced_rhs_bulk, outer_dirichlet_values);
}

void LaplaceTransmission2D::restore_dirichlet_boundary(
    Eigen::VectorXd&       u_bulk,
    const Eigen::VectorXd& outer_dirichlet_values) const
{
    if (outer_dirichlet_values.size() == 0)
        return;
    if (bulk_bc_ != ZfftBcType::Dirichlet) {
        throw std::invalid_argument(
            std::string(kContext)
            + ": outer Dirichlet values are only valid with Dirichlet bulk BC");
    }

    structured_grid::restore_dirichlet_boundary(
        kContext, grid_pair_.grid(), u_bulk, outer_dirichlet_values);
}

void LaplaceTransmission2D::apply_common_ratio(
    const Eigen::VectorXd& x,
    Eigen::VectorXd&       y) const
{
    require_common_ratio(lambda_sq_int_, lambda_sq_ext_);

    const int n_iface = grid_pair_.interface().num_points();
    const int n_dof = grid_pair_.grid().num_dofs();
    require_vector_size("common-ratio operator input", x.size(), n_iface);

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n_iface);
    const Eigen::VectorXd zero_rhs = Eigen::VectorXd::Zero(n_dof);
    const auto zero_derivs = zero_rhs_derivs(n_iface);
    auto jumps = make_jumps(n_iface, zeros, x, zero_derivs);
    auto result = potentials_int_.evaluate(jumps, zero_rhs);

    const double beta_sum = coefficients_.beta_int + coefficients_.beta_ext;
    const double gamma =
        2.0 * (coefficients_.beta_int - coefficients_.beta_ext) / beta_sum;
    y = x + gamma * result.un_avg;
}

void LaplaceTransmission2D::apply_different_ratios(
    const Eigen::VectorXd& x,
    Eigen::VectorXd&       y) const
{
    const int n_iface = grid_pair_.interface().num_points();
    require_vector_size("different-ratio operator input", x.size(), 2 * n_iface);

    const Eigen::VectorXd phi = x.head(n_iface);
    const Eigen::VectorXd psi = x.tail(n_iface);

    const double beta_int = coefficients_.beta_int;
    const double beta_ext = coefficients_.beta_ext;
    const double beta_sum = beta_int + beta_ext;
    const double alpha_int = 2.0 * beta_ext / beta_sum;
    const double alpha_ext = 2.0 * beta_int / beta_sum;
    const double flux_row_scale = 2.0 / beta_sum;

    Eigen::VectorXd trace_int;
    Eigen::VectorXd normal_int;
    Eigen::VectorXd trace_ext;
    Eigen::VectorXd normal_ext;
    const Eigen::VectorXd phi_int = alpha_int * phi;
    const Eigen::VectorXd phi_ext = alpha_ext * phi;
    potentials_int_.eval_layer_combination(phi_int, psi, trace_int, normal_int);
    potentials_ext_.eval_layer_combination(phi_ext, psi, trace_ext, normal_ext);

    y.resize(2 * n_iface);
    y.head(n_iface) = phi + trace_int - trace_ext;
    y.tail(n_iface) =
        psi + flux_row_scale * (beta_int * normal_int - beta_ext * normal_ext);
}

LaplaceTransmissionSolveResult2D LaplaceTransmission2D::solve(
    const Eigen::VectorXd&              u_jump,
    const Eigen::VectorXd&              beta_flux_jump,
    const LaplaceTransmissionRhsData2D& rhs_data,
    const Eigen::VectorXd&              outer_dirichlet_values,
    int                                max_iter,
    double                             tol,
    int                                restart) const
{
    const int n_iface = grid_pair_.interface().num_points();
    const int n_dof = grid_pair_.grid().num_dofs();

    require_vector_size("u_jump", u_jump.size(), n_iface);
    require_vector_size("beta_flux_jump", beta_flux_jump.size(), n_iface);
    validate_rhs_data(rhs_data, n_iface, n_dof);
    if (outer_dirichlet_values.size() != 0) {
        if (bulk_bc_ != ZfftBcType::Dirichlet) {
            throw std::invalid_argument(
                std::string(kContext)
                + ": outer Dirichlet values are only valid with Dirichlet bulk BC");
        }
        require_vector_size("outer_dirichlet_values", outer_dirichlet_values.size(), n_dof);
    }

    switch (mode_) {
    case LaplaceTransmissionMode2D::CommonRatio:
        return solve_common_ratio(u_jump,
                                  beta_flux_jump,
                                  rhs_data,
                                  outer_dirichlet_values,
                                  max_iter,
                                  tol,
                                  restart);
    case LaplaceTransmissionMode2D::DifferentRatios:
        return solve_different_ratios(u_jump,
                                      beta_flux_jump,
                                      rhs_data,
                                      outer_dirichlet_values,
                                      max_iter,
                                      tol,
                                      restart);
    }

    throw std::invalid_argument(std::string(kContext) + ": unsupported transmission mode");
}

LaplaceTransmissionSolveResult2D LaplaceTransmission2D::solve_common_ratio(
    const Eigen::VectorXd&              u_jump,
    const Eigen::VectorXd&              beta_flux_jump,
    const LaplaceTransmissionRhsData2D& rhs_data,
    const Eigen::VectorXd&              outer_dirichlet_values,
    int                                max_iter,
    double                             tol,
    int                                restart) const
{
    StageTimer2D timer;
    const bool profile = profile_solve_2d();

    require_common_ratio(lambda_sq_int_, lambda_sq_ext_);
    const double t_validate = timer.lap();

    const int n_iface = grid_pair_.interface().num_points();
    const Eigen::VectorXd rhs = apply_dirichlet_boundary_elimination(
        rhs_data.reduced_rhs_bulk, outer_dirichlet_values);
    const double t_boundary_elim = timer.lap();

    const auto rhs_derivs = combine_rhs_derivs(
        rhs_data.reduced_rhs_int_derivs,
        rhs_data.reduced_rhs_ext_derivs);
    const double t_rhs_derivs = timer.lap();

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n_iface);

    auto rhs_setup_jumps = make_jumps(n_iface, u_jump, zeros, rhs_derivs);
    const double t_rhs_jumps = timer.lap();

    if (profile)
        potentials_int_.reset_profile();
    auto rhs_setup_res = potentials_int_.evaluate(rhs_setup_jumps, rhs);
    const LaplacePotentialEvalProfile2D rhs_profile =
        profile ? potentials_int_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_rhs_potential = timer.lap();

    const double beta_sum = coefficients_.beta_int + coefficients_.beta_ext;
    const double gamma =
        2.0 * (coefficients_.beta_int - coefficients_.beta_ext) / beta_sum;
    Eigen::VectorXd bie_rhs =
        (2.0 / beta_sum) * beta_flux_jump
        - gamma * rhs_setup_res.un_avg;
    const double t_bie_rhs = timer.lap();

    GMRES gmres(max_iter, tol, restart);
    Eigen::VectorXd psi = Eigen::VectorXd::Zero(n_iface);
    if (profile)
        potentials_int_.reset_profile();
    const int iterations = gmres.solve(*this, bie_rhs, psi);
    const LaplacePotentialEvalProfile2D gmres_profile =
        profile ? potentials_int_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_gmres = timer.lap();

    auto jumps = make_jumps(n_iface, u_jump, psi, rhs_derivs);
    const double t_final_jumps = timer.lap();

    if (profile)
        potentials_int_.reset_profile();
    auto full_res = potentials_int_.evaluate(jumps, rhs);
    const LaplacePotentialEvalProfile2D final_profile =
        profile ? potentials_int_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_final_potential = timer.lap();

    restore_dirichlet_boundary(full_res.u_bulk, outer_dirichlet_values);
    restore_dirichlet_boundary(full_res.u_physical, outer_dirichlet_values);
    const double t_restore = timer.lap();

    Eigen::VectorXd phi = u_jump;
    auto residuals = gmres.residuals();
    const double t_result = timer.lap();
    const double t_total = timer.total();

    if (profile) {
        std::printf("    LaplaceTransmission2D::solve_common_ratio profile iface=%d grid=%d iterations=%d\n",
                    n_iface, grid_pair_.grid().num_dofs(), iterations);
        std::printf("      setup validate %.3fs boundary_elim %.3fs rhs_derivs %.3fs rhs_jumps %.3fs rhs_potential %.3fs bie_rhs %.3fs\n",
                    t_validate, t_boundary_elim, t_rhs_derivs, t_rhs_jumps,
                    t_rhs_potential, t_bie_rhs);
        std::printf("      gmres %.3fs final_jumps %.3fs final_potential %.3fs restore %.3fs result %.3fs total %.3fs\n",
                    t_gmres, t_final_jumps, t_final_potential, t_restore,
                    t_result, t_total);
        print_potential_profile_2d("rhs_potential", rhs_profile);
        print_potential_profile_2d("gmres_int", gmres_profile);
        print_potential_profile_2d("final_int", final_profile);
        std::printf("      gmres_nonpotential %.3fs\n",
                    profile_total_without_potential(gmres_profile, t_gmres));
    }

    return {std::move(full_res.u_bulk),
            std::move(full_res.u_physical),
            std::move(phi),
            std::move(psi),
            std::move(full_res.corner_patch_corrections),
            std::move(residuals),
            iterations,
            gmres.converged()};
}

LaplaceTransmissionSolveResult2D LaplaceTransmission2D::solve_different_ratios(
    const Eigen::VectorXd&              u_jump,
    const Eigen::VectorXd&              beta_flux_jump,
    const LaplaceTransmissionRhsData2D& rhs_data,
    const Eigen::VectorXd&              outer_dirichlet_values,
    int                                max_iter,
    double                             tol,
    int                                restart) const
{
    StageTimer2D timer;
    const bool profile = profile_solve_2d();

    const int n_iface = grid_pair_.interface().num_points();
    const int n_dof = grid_pair_.grid().num_dofs();
    const double t_sizes = timer.lap();

    Eigen::VectorXd rhs_int_raw;
    Eigen::VectorXd rhs_ext_raw;
    split_phase_rhs(grid_pair_, rhs_data.reduced_rhs_bulk, rhs_int_raw, rhs_ext_raw);
    const double t_split_rhs = timer.lap();

    const Eigen::VectorXd rhs_int = rhs_int_raw;
    const Eigen::VectorXd rhs_ext =
        apply_dirichlet_boundary_elimination(rhs_ext_raw, outer_dirichlet_values);
    const double t_boundary_elim = timer.lap();

    const auto rhs_int_derivs = rhs_data.reduced_rhs_int_derivs;
    const auto rhs_ext_derivs = scaled_rhs_derivs(
        rhs_data.reduced_rhs_ext_derivs, -1.0);
    const double t_rhs_derivs = timer.lap();

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n_iface);

    auto volume_int_jumps = make_jumps(n_iface, zeros, zeros, rhs_int_derivs);
    const double t_volume_int_jumps = timer.lap();

    auto volume_ext_jumps = make_jumps(n_iface, zeros, zeros, rhs_ext_derivs);
    const double t_volume_ext_jumps = timer.lap();

    if (profile)
        potentials_int_.reset_profile();
    auto volume_int = potentials_int_.evaluate(volume_int_jumps, rhs_int);
    const LaplacePotentialEvalProfile2D volume_int_profile =
        profile ? potentials_int_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_volume_int = timer.lap();

    if (profile)
        potentials_ext_.reset_profile();
    auto volume_ext = potentials_ext_.evaluate(volume_ext_jumps, rhs_ext);
    const LaplacePotentialEvalProfile2D volume_ext_profile =
        profile ? potentials_ext_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_volume_ext = timer.lap();

    const double beta_int = coefficients_.beta_int;
    const double beta_ext = coefficients_.beta_ext;
    const double beta_sum = beta_int + beta_ext;

    Eigen::VectorXd rhs(2 * n_iface);
    rhs.head(n_iface) = u_jump - (volume_int.u_avg - volume_ext.u_avg);
    rhs.tail(n_iface) =
        (2.0 / beta_sum)
        * (beta_flux_jump - (beta_int * volume_int.un_avg
                             - beta_ext * volume_ext.un_avg));
    const double t_bie_rhs = timer.lap();

    GMRES gmres(max_iter, tol, restart);
    Eigen::VectorXd density = Eigen::VectorXd::Zero(2 * n_iface);
    if (profile) {
        potentials_int_.reset_profile();
        potentials_ext_.reset_profile();
    }
    const int iterations = gmres.solve(*this, rhs, density);
    const LaplacePotentialEvalProfile2D gmres_int_profile =
        profile ? potentials_int_.profile() : LaplacePotentialEvalProfile2D{};
    const LaplacePotentialEvalProfile2D gmres_ext_profile =
        profile ? potentials_ext_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_gmres = timer.lap();

    Eigen::VectorXd phi = density.head(n_iface);
    Eigen::VectorXd psi = density.tail(n_iface);
    const double t_density_split = timer.lap();

    const double alpha_int = 2.0 * beta_ext / beta_sum;
    const double alpha_ext = 2.0 * beta_int / beta_sum;

    auto full_int_jumps =
        make_jumps(n_iface, alpha_int * phi, psi, rhs_int_derivs);
    auto full_ext_jumps =
        make_jumps(n_iface, alpha_ext * phi, psi, rhs_ext_derivs);
    const double t_final_jumps = timer.lap();

    if (profile)
        potentials_int_.reset_profile();
    auto full_int = potentials_int_.evaluate(full_int_jumps, rhs_int);
    const LaplacePotentialEvalProfile2D final_int_profile =
        profile ? potentials_int_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_final_int = timer.lap();

    if (profile)
        potentials_ext_.reset_profile();
    auto full_ext = potentials_ext_.evaluate(full_ext_jumps, rhs_ext);
    const LaplacePotentialEvalProfile2D final_ext_profile =
        profile ? potentials_ext_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_final_ext = timer.lap();

    Eigen::VectorXd u_bulk(n_dof);
    Eigen::VectorXd u_physical(n_dof);
    for (int n = 0; n < n_dof; ++n) {
        u_bulk[n] = (grid_pair_.domain_label(n) > 0)
                        ? full_int.u_bulk[n]
                        : full_ext.u_bulk[n];
        u_physical[n] = (grid_pair_.domain_label(n) > 0)
                            ? full_int.u_physical[n]
                            : full_ext.u_physical[n];
    }
    const double t_combine_bulk = timer.lap();

    restore_dirichlet_boundary(u_bulk, outer_dirichlet_values);
    restore_dirichlet_boundary(u_physical, outer_dirichlet_values);
    const double t_restore = timer.lap();

    auto residuals = gmres.residuals();
    const double t_result = timer.lap();
    const double t_total = timer.total();

    if (profile) {
        const LaplacePotentialEvalProfile2D gmres_combined{
            gmres_int_profile.calls + gmres_ext_profile.calls,
            gmres_int_profile.rhs_copy_sec + gmres_ext_profile.rhs_copy_sec,
            gmres_int_profile.spread_sec + gmres_ext_profile.spread_sec,
            gmres_int_profile.bulk_solve_sec + gmres_ext_profile.bulk_solve_sec,
            gmres_int_profile.restrict_sec + gmres_ext_profile.restrict_sec,
            gmres_int_profile.average_sec + gmres_ext_profile.average_sec,
            gmres_int_profile.total_sec + gmres_ext_profile.total_sec};
        std::printf("    LaplaceTransmission2D::solve_different_ratios profile iface=%d grid=%d iterations=%d\n",
                    n_iface, n_dof, iterations);
        std::printf("      setup sizes %.3fs split_rhs %.3fs boundary_elim %.3fs rhs_derivs %.3fs volume_jumps(int/ext) %.3fs/%.3fs\n",
                    t_sizes, t_split_rhs, t_boundary_elim, t_rhs_derivs,
                    t_volume_int_jumps, t_volume_ext_jumps);
        std::printf("      volume_int %.3fs volume_ext %.3fs bie_rhs %.3fs gmres %.3fs density_split %.3fs\n",
                    t_volume_int, t_volume_ext, t_bie_rhs, t_gmres,
                    t_density_split);
        std::printf("      final_jumps %.3fs final_int %.3fs final_ext %.3fs combine_bulk %.3fs restore %.3fs result %.3fs total %.3fs\n",
                    t_final_jumps, t_final_int, t_final_ext, t_combine_bulk,
                    t_restore, t_result, t_total);
        print_potential_profile_2d("volume_int", volume_int_profile);
        print_potential_profile_2d("volume_ext", volume_ext_profile);
        print_potential_profile_2d("gmres_int", gmres_int_profile);
        print_potential_profile_2d("gmres_ext", gmres_ext_profile);
        print_potential_profile_2d("final_int", final_int_profile);
        print_potential_profile_2d("final_ext", final_ext_profile);
        std::printf("      gmres_nonpotential %.3fs\n",
                    profile_total_without_potential(gmres_combined, t_gmres));
    }

    return {std::move(u_bulk),
            std::move(u_physical),
            std::move(phi),
            std::move(psi),
            std::move(full_int.corner_patch_corrections),
            std::move(residuals),
            iterations,
            gmres.converged()};
}

} // namespace kfbim
