#include "laplace_transmission_3d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "../bulk_solvers/zfft_bc_type.hpp"
#include "../gmres/gmres.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "../local_cauchy/jump_data.hpp"
#include "detail/laplace_operator_utils.hpp"

namespace kfbim {

namespace {

constexpr const char* kContext = "LaplaceTransmission3D";

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

void validate_rhs_data(const LaplaceTransmissionRhsData3D& rhs_data,
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

std::vector<LaplaceJumpData3D> make_jumps(
    int                                n_iface,
    const Eigen::VectorXd&             u_jump,
    const Eigen::VectorXd&             un_jump,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    return operators_detail::make_laplace_jumps_3d(
        kContext, n_iface, u_jump, un_jump, rhs_derivs);
}

void split_phase_rhs(const GridPair3D&      grid_pair,
                     const Eigen::VectorXd& rhs_bulk,
                     Eigen::VectorXd&       rhs_int,
                     Eigen::VectorXd&       rhs_ext)
{
    operators_detail::split_phase_rhs(
        kContext, grid_pair, rhs_bulk, rhs_int, rhs_ext);
}

} // namespace

LaplaceTransmission3D::LaplaceTransmission3D(
    const CartesianGrid3D&            grid,
    const Interface3D&                iface,
    LaplaceTransmissionMode3D         mode,
    LaplaceTransmissionCoefficients3D coefficients,
    LaplaceTransmissionOptions3D      options)
    : grid_pair_(grid, iface)
    , mode_(mode)
    , coefficients_(coefficients)
    , lambda_sq_int_(checked_lambda_sq("interior",
                                       coefficients.beta_int,
                                       coefficients.kappa_sq_int))
    , lambda_sq_ext_(checked_lambda_sq("exterior",
                                       coefficients.beta_ext,
                                       coefficients.kappa_sq_ext))
    , spread_int_(grid_pair_,
                  lambda_sq_int_,
                  options.correction_method,
                  options.restrict_stencil_radius)
    , spread_ext_(grid_pair_,
                  lambda_sq_ext_,
                  options.correction_method,
                  options.restrict_stencil_radius)
    , bulk_solver_int_(grid, ZfftBcType::Dirichlet, lambda_sq_int_)
    , bulk_solver_ext_(grid, ZfftBcType::Dirichlet, lambda_sq_ext_)
    , restrict_op_(grid_pair_, options.restrict_stencil_radius)
    , potentials_int_(spread_int_, bulk_solver_int_, restrict_op_)
    , potentials_ext_(spread_ext_, bulk_solver_ext_, restrict_op_)
{
    if (mode_ == LaplaceTransmissionMode3D::CommonRatio)
        require_common_ratio(lambda_sq_int_, lambda_sq_ext_);
}

void LaplaceTransmission3D::apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const
{
    switch (mode_) {
    case LaplaceTransmissionMode3D::CommonRatio:
        apply_common_ratio(x, y);
        return;
    case LaplaceTransmissionMode3D::DifferentRatios:
        apply_different_ratios(x, y);
        return;
    }

    throw std::invalid_argument(std::string(kContext) + ": unsupported transmission mode");
}

int LaplaceTransmission3D::problem_size() const
{
    const int n_iface = grid_pair_.interface().num_points();
    switch (mode_) {
    case LaplaceTransmissionMode3D::CommonRatio:
        return n_iface;
    case LaplaceTransmissionMode3D::DifferentRatios:
        return 2 * n_iface;
    }

    throw std::invalid_argument(std::string(kContext) + ": unsupported transmission mode");
}

Eigen::VectorXd LaplaceTransmission3D::apply_dirichlet_boundary_elimination(
    const Eigen::VectorXd& reduced_rhs_bulk,
    const Eigen::VectorXd& outer_dirichlet_values) const
{
    const auto& grid = grid_pair_.grid();
    const int n_dof = grid.num_dofs();

    require_vector_size("reduced_rhs_bulk", reduced_rhs_bulk.size(), n_dof);
    if (outer_dirichlet_values.size() == 0)
        return reduced_rhs_bulk;

    return structured_grid::apply_dirichlet_boundary_elimination(
        kContext, grid, reduced_rhs_bulk, outer_dirichlet_values);
}

void LaplaceTransmission3D::restore_dirichlet_boundary(
    Eigen::VectorXd&       u_bulk,
    const Eigen::VectorXd& outer_dirichlet_values) const
{
    if (outer_dirichlet_values.size() == 0)
        return;

    const auto& grid = grid_pair_.grid();
    structured_grid::restore_dirichlet_boundary(
        kContext, grid, u_bulk, outer_dirichlet_values);
}

void LaplaceTransmission3D::apply_common_ratio(
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

void LaplaceTransmission3D::apply_different_ratios(
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

LaplaceTransmissionSolveResult3D LaplaceTransmission3D::solve(
    const Eigen::VectorXd&              u_jump,
    const Eigen::VectorXd&              beta_flux_jump,
    const LaplaceTransmissionRhsData3D& rhs_data,
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
    if (outer_dirichlet_values.size() != 0)
        require_vector_size("outer_dirichlet_values", outer_dirichlet_values.size(), n_dof);

    switch (mode_) {
    case LaplaceTransmissionMode3D::CommonRatio:
        return solve_common_ratio(u_jump,
                                  beta_flux_jump,
                                  rhs_data,
                                  outer_dirichlet_values,
                                  max_iter,
                                  tol,
                                  restart);
    case LaplaceTransmissionMode3D::DifferentRatios:
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

LaplaceTransmissionSolveResult3D LaplaceTransmission3D::solve_common_ratio(
    const Eigen::VectorXd&              u_jump,
    const Eigen::VectorXd&              beta_flux_jump,
    const LaplaceTransmissionRhsData3D& rhs_data,
    const Eigen::VectorXd&              outer_dirichlet_values,
    int                                max_iter,
    double                             tol,
    int                                restart) const
{
    require_common_ratio(lambda_sq_int_, lambda_sq_ext_);

    const int n_iface = grid_pair_.interface().num_points();
    const Eigen::VectorXd rhs = apply_dirichlet_boundary_elimination(
        rhs_data.reduced_rhs_bulk, outer_dirichlet_values);
    const auto rhs_derivs = combine_rhs_derivs(
        rhs_data.reduced_rhs_int_derivs,
        rhs_data.reduced_rhs_ext_derivs);

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n_iface);

    auto rhs_setup_jumps = make_jumps(n_iface, u_jump, zeros, rhs_derivs);
    auto rhs_setup_res = potentials_int_.evaluate(rhs_setup_jumps, rhs);

    const double beta_sum = coefficients_.beta_int + coefficients_.beta_ext;
    const double gamma =
        2.0 * (coefficients_.beta_int - coefficients_.beta_ext) / beta_sum;
    Eigen::VectorXd bie_rhs =
        (2.0 / beta_sum) * beta_flux_jump
        - gamma * rhs_setup_res.un_avg;

    GMRES gmres(max_iter, tol, restart);
    Eigen::VectorXd psi = Eigen::VectorXd::Zero(n_iface);
    const int iterations = gmres.solve(*this, bie_rhs, psi);

    auto jumps = make_jumps(n_iface, u_jump, psi, rhs_derivs);
    auto full_res = potentials_int_.evaluate(jumps, rhs);
    restore_dirichlet_boundary(full_res.u_bulk, outer_dirichlet_values);

    Eigen::VectorXd phi = u_jump;
    auto residuals = gmres.residuals();
    return {std::move(full_res.u_bulk),
            std::move(phi),
            std::move(psi),
            std::move(residuals),
            iterations,
            gmres.converged()};
}

LaplaceTransmissionSolveResult3D LaplaceTransmission3D::solve_different_ratios(
    const Eigen::VectorXd&              u_jump,
    const Eigen::VectorXd&              beta_flux_jump,
    const LaplaceTransmissionRhsData3D& rhs_data,
    const Eigen::VectorXd&              outer_dirichlet_values,
    int                                max_iter,
    double                             tol,
    int                                restart) const
{
    const int n_iface = grid_pair_.interface().num_points();
    const int n_dof = grid_pair_.grid().num_dofs();

    Eigen::VectorXd rhs_int_raw;
    Eigen::VectorXd rhs_ext_raw;
    split_phase_rhs(grid_pair_, rhs_data.reduced_rhs_bulk, rhs_int_raw, rhs_ext_raw);
    const Eigen::VectorXd rhs_int = rhs_int_raw;
    const Eigen::VectorXd rhs_ext =
        apply_dirichlet_boundary_elimination(rhs_ext_raw, outer_dirichlet_values);

    const auto rhs_int_derivs = rhs_data.reduced_rhs_int_derivs;
    const auto rhs_ext_derivs = scaled_rhs_derivs(
        rhs_data.reduced_rhs_ext_derivs, -1.0);

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n_iface);

    auto volume_int_jumps = make_jumps(n_iface, zeros, zeros, rhs_int_derivs);
    auto volume_ext_jumps = make_jumps(n_iface, zeros, zeros, rhs_ext_derivs);
    auto volume_int = potentials_int_.evaluate(volume_int_jumps, rhs_int);
    auto volume_ext = potentials_ext_.evaluate(volume_ext_jumps, rhs_ext);

    const double beta_int = coefficients_.beta_int;
    const double beta_ext = coefficients_.beta_ext;
    const double beta_sum = beta_int + beta_ext;

    Eigen::VectorXd rhs(2 * n_iface);
    rhs.head(n_iface) = u_jump - (volume_int.u_avg - volume_ext.u_avg);
    rhs.tail(n_iface) =
        (2.0 / beta_sum)
        * (beta_flux_jump - (beta_int * volume_int.un_avg
                             - beta_ext * volume_ext.un_avg));

    GMRES gmres(max_iter, tol, restart);
    Eigen::VectorXd density = Eigen::VectorXd::Zero(2 * n_iface);
    const int iterations = gmres.solve(*this, rhs, density);

    Eigen::VectorXd phi = density.head(n_iface);
    Eigen::VectorXd psi = density.tail(n_iface);

    const double alpha_int = 2.0 * beta_ext / beta_sum;
    const double alpha_ext = 2.0 * beta_int / beta_sum;

    auto full_int_jumps =
        make_jumps(n_iface, alpha_int * phi, psi, rhs_int_derivs);
    auto full_ext_jumps =
        make_jumps(n_iface, alpha_ext * phi, psi, rhs_ext_derivs);

    auto full_int = potentials_int_.evaluate(full_int_jumps, rhs_int);
    auto full_ext = potentials_ext_.evaluate(full_ext_jumps, rhs_ext);

    Eigen::VectorXd u_bulk(n_dof);
    for (int n = 0; n < n_dof; ++n) {
        u_bulk[n] = (grid_pair_.domain_label(n) > 0)
                        ? full_int.u_bulk[n]
                        : full_ext.u_bulk[n];
    }
    restore_dirichlet_boundary(u_bulk, outer_dirichlet_values);

    auto residuals = gmres.residuals();
    return {std::move(u_bulk),
            std::move(phi),
            std::move(psi),
            std::move(residuals),
            iterations,
            gmres.converged()};
}

} // namespace kfbim
