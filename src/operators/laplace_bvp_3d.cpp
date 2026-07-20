#include "laplace_bvp_3d.hpp"

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

using operators_detail::MeanProjectedOperator;
using operators_detail::project_mean_zero;
using operators_detail::require_vector_size;
using operators_detail::signed_rhs_derivs;

namespace {

bool is_neumann_type(LaplaceBvpType3D type)
{
    return type == LaplaceBvpType3D::InteriorNeumann
        || type == LaplaceBvpType3D::ExteriorNeumann;
}

bool is_exterior_type(LaplaceBvpType3D type)
{
    return type == LaplaceBvpType3D::ExteriorDirichlet
        || type == LaplaceBvpType3D::ExteriorNeumann;
}

double side_jump_sign(LaplaceBvpType3D type)
{
    return is_exterior_type(type) ? -1.0 : 1.0;
}

bool uses_neumann_nullspace_projection(LaplaceBvpType3D type, double eta)
{
    return operators_detail::uses_interior_neumann_nullspace_projection(
        type == LaplaceBvpType3D::InteriorNeumann, eta);
}

double rhs_deriv_sign_for_type(LaplaceBvpType3D type)
{
    switch (type) {
    case LaplaceBvpType3D::InteriorDirichlet:
    case LaplaceBvpType3D::InteriorNeumann:
        return 1.0;
    case LaplaceBvpType3D::ExteriorDirichlet:
    case LaplaceBvpType3D::ExteriorNeumann:
        return -1.0;
    }
    throw std::invalid_argument("unsupported Laplace BVP type");
}

std::vector<LaplaceJumpData3D> make_jumps(
    const Interface3D&                 iface,
    const Eigen::VectorXd&             density,
    LaplaceBvpType3D                   type,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    const int n_iface = iface.num_points();
    std::vector<LaplaceJumpData3D> jumps(n_iface);
    const bool neumann = is_neumann_type(type);
    for (int q = 0; q < n_iface; ++q) {
        jumps[q].u_jump = neumann ? 0.0 : density[q];
        jumps[q].un_jump = neumann ? density[q] : 0.0;
        jumps[q].rhs_derivs = rhs_derivs[q];
    }
    return jumps;
}

} // namespace

LaplaceBvp3D::LaplaceBvp3D(
    const CartesianGrid3D& grid,
    const Interface3D&     iface,
    LaplaceBvpType3D       type,
    LaplaceBvpOptions3D    options)
    : LaplaceBvp3D(grid, iface, iface, type, std::move(options))
{
}

LaplaceBvp3D::LaplaceBvp3D(
    const CartesianGrid3D& grid,
    const Interface3D&     iface,
    const Interface3D&     crossing_geometry,
    LaplaceBvpType3D       type,
    LaplaceBvpOptions3D    options)
    : grid_pair_(grid, iface, crossing_geometry)
    , spread_(grid_pair_,
              options.eta,
              options.correction_method,
              options.restrict_stencil_radius)
    , bulk_solver_(grid, ZfftBcType::Dirichlet, options.eta)
    , restrict_op_(grid_pair_, options.restrict_stencil_radius)
    , potentials_(spread_, bulk_solver_, restrict_op_)
    , type_(type)
    , rhs_deriv_sign_(rhs_deriv_sign_for_type(type))
    , eta_(options.eta)
    , outer_dirichlet_values_(std::move(options.outer_dirichlet_values))
{
    if (outer_dirichlet_values_.size() != 0) {
        require_vector_size("LaplaceBvp3D",
                            "outer_dirichlet_values",
                            outer_dirichlet_values_.size(),
                            grid.num_dofs());
    }
}

void LaplaceBvp3D::apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const
{
    const int n_iface = grid_pair_.interface().num_points();
    require_vector_size("LaplaceBvp3D", "operator input", x.size(), n_iface);

    const Eigen::VectorXd zero_rhs =
        Eigen::VectorXd::Zero(grid_pair_.grid().num_dofs());
    const std::vector<Eigen::VectorXd> zero_derivs(
        n_iface, Eigen::VectorXd::Zero(1));
    apply_with_rhs(x, zero_rhs, zero_derivs, y);
}

int LaplaceBvp3D::problem_size() const
{
    return grid_pair_.interface().num_points();
}

Eigen::VectorXd LaplaceBvp3D::apply_dirichlet_boundary_elimination(
    const Eigen::VectorXd& rhs) const
{
    return structured_grid::apply_dirichlet_boundary_elimination(
        "LaplaceBvp3D", grid_pair_.grid(), rhs, outer_dirichlet_values_);
}

void LaplaceBvp3D::restore_dirichlet_boundary(Eigen::VectorXd& u_bulk) const
{
    structured_grid::restore_dirichlet_boundary(
        "LaplaceBvp3D", grid_pair_.grid(), u_bulk, outer_dirichlet_values_);
}

void LaplaceBvp3D::apply_with_rhs(
    const Eigen::VectorXd&              density,
    const Eigen::VectorXd&              rhs,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    Eigen::VectorXd&                    y) const
{
    const int n_iface = grid_pair_.interface().num_points();
    require_vector_size("LaplaceBvp3D", "operator input", density.size(), n_iface);
    require_vector_size("LaplaceBvp3D", "rhs", rhs.size(), grid_pair_.grid().num_dofs());

    auto jumps = make_jumps(grid_pair_.interface(), density, type_, rhs_derivs);
    auto res = potentials_.evaluate(jumps, rhs);

    y.resize(n_iface);
    const double sign = side_jump_sign(type_);
    if (!is_neumann_type(type_)) {
        for (int q = 0; q < n_iface; ++q)
            y[q] = res.u_avg[q] + sign * 0.5 * jumps[q].u_jump;
    } else {
        for (int q = 0; q < n_iface; ++q)
            y[q] = res.un_avg[q] + sign * 0.5 * jumps[q].un_jump;
    }
}

LaplaceBvpSolveResult3D LaplaceBvp3D::solve(
    const Eigen::VectorXd&              boundary_data,
    const Eigen::VectorXd&              f_bulk,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    int max_iter,
    double tol,
    int restart) const
{
    const int n_iface = grid_pair_.interface().num_points();
    require_vector_size("LaplaceBvp3D", "boundary_data", boundary_data.size(), n_iface);
    require_vector_size("LaplaceBvp3D", "f_bulk", f_bulk.size(), grid_pair_.grid().num_dofs());

    const auto signed_derivs = signed_rhs_derivs(
        "LaplaceBvp3D", rhs_derivs, n_iface, rhs_deriv_sign_);
    const Eigen::VectorXd rhs = apply_dirichlet_boundary_elimination(f_bulk);

    Eigen::VectorXd volume_gamma;
    apply_with_rhs(Eigen::VectorXd::Zero(n_iface), rhs, signed_derivs, volume_gamma);

    Eigen::VectorXd b = boundary_data - volume_gamma;

    GMRES gmres(max_iter, tol, restart);
    Eigen::VectorXd density = Eigen::VectorXd::Zero(n_iface);
    int iterations = 0;
    if (uses_neumann_nullspace_projection(type_, eta_)) {
        project_mean_zero(b);

        MeanProjectedOperator projected_op(*this);
        iterations = gmres.solve(projected_op, b, density);
        project_mean_zero(density);
    } else {
        iterations = gmres.solve(*this, b, density);
    }

    auto jumps = make_jumps(grid_pair_.interface(), density, type_, signed_derivs);
    auto full_res = potentials_.evaluate(jumps, rhs);
    restore_dirichlet_boundary(full_res.u_bulk);

    auto residuals = gmres.residuals();
    return {std::move(full_res.u_bulk),
            std::move(density),
            std::move(residuals),
            iterations,
            gmres.converged()};
}

Eigen::VectorXd LaplaceBvp3D::boundary_residual(
    const Eigen::VectorXd&              density,
    const Eigen::VectorXd&              boundary_data,
    const Eigen::VectorXd&              f_bulk,
    const std::vector<Eigen::VectorXd>& rhs_derivs) const
{
    const int n_iface = grid_pair_.interface().num_points();
    require_vector_size("LaplaceBvp3D", "density", density.size(), n_iface);
    require_vector_size("LaplaceBvp3D",
                        "boundary_data",
                        boundary_data.size(),
                        n_iface);
    require_vector_size("LaplaceBvp3D",
                        "f_bulk",
                        f_bulk.size(),
                        grid_pair_.grid().num_dofs());

    const auto signed_derivs = signed_rhs_derivs(
        "LaplaceBvp3D", rhs_derivs, n_iface, rhs_deriv_sign_);
    const Eigen::VectorXd rhs = apply_dirichlet_boundary_elimination(f_bulk);

    Eigen::VectorXd volume_gamma;
    apply_with_rhs(Eigen::VectorXd::Zero(n_iface),
                   rhs,
                   signed_derivs,
                   volume_gamma);

    Eigen::VectorXd operator_density;
    apply(density, operator_density);
    return boundary_data - volume_gamma - operator_density;
}

} // namespace kfbim
