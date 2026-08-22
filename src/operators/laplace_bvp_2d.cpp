#include "laplace_bvp_2d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "../gmres/gmres.hpp"
#include "../local_cauchy/jump_data.hpp"
#include "../bulk_solvers/zfft_bc_type.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "detail/laplace_operator_utils.hpp"

namespace kfbim {

using operators_detail::MeanProjectedOperator;
using operators_detail::project_mean_zero;
using operators_detail::require_vector_size;
using operators_detail::signed_rhs_derivs;

LaplaceCrossingTraceStencil2D laplace_crossing_trace_stencil_for_bvp_2d(
    LaplaceBvpType2D type)
{
    switch (type) {
    case LaplaceBvpType2D::InteriorDirichlet:
    case LaplaceBvpType2D::ExteriorDirichlet:
        return LaplaceCrossingTraceStencil2D::PhiP3;
    case LaplaceBvpType2D::InteriorNeumann:
    case LaplaceBvpType2D::ExteriorNeumann:
        return LaplaceCrossingTraceStencil2D::PsiP2;
    }
    throw std::invalid_argument("unsupported Laplace BVP type");
}

namespace {

bool is_neumann_type(LaplaceBvpType2D type)
{
    return type == LaplaceBvpType2D::InteriorNeumann
        || type == LaplaceBvpType2D::ExteriorNeumann;
}

bool is_exterior_type(LaplaceBvpType2D type)
{
    return type == LaplaceBvpType2D::ExteriorDirichlet
        || type == LaplaceBvpType2D::ExteriorNeumann;
}

double side_jump_sign(LaplaceBvpType2D type)
{
    return is_exterior_type(type) ? -1.0 : 1.0;
}

bool uses_neumann_nullspace_projection(LaplaceBvpType2D type, double eta)
{
    return operators_detail::uses_interior_neumann_nullspace_projection(
        type == LaplaceBvpType2D::InteriorNeumann, eta);
}

void maybe_write_bvp_rhs_diagnostic_csv(
    const char* path,
    const GridPair2D& grid_pair,
    const std::vector<int>& active_points,
    const Eigen::VectorXd& boundary_data,
    const Eigen::VectorXd& volume_gamma,
    const Eigen::VectorXd& b_before_projection,
    const Eigen::VectorXd& b_after_projection)
{
    if (path == nullptr)
        return;
    if (std::FILE* file = std::fopen(path, "w")) {
        std::fprintf(file,
                     "a,q,x,y,nx,ny,boundary_data,volume_gamma,b_before_projection,b_after_projection\n");
        const Interface2D& iface = grid_pair.interface();
        for (int a = 0; a < static_cast<int>(active_points.size()); ++a) {
            const int q = active_points[static_cast<std::size_t>(a)];
            const Eigen::Vector2d point = iface.points().row(q).transpose();
            const Eigen::Vector2d normal = iface.normals().row(q).transpose();
            std::fprintf(file,
                         "%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                         a,
                         q,
                         point[0],
                         point[1],
                         normal[0],
                         normal[1],
                         boundary_data[a],
                         volume_gamma[a],
                         b_before_projection[a],
                         b_after_projection[a]);
        }
        std::fclose(file);
    }
}

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

    std::printf("      %-15s calls=%3d total %.3fs avg %.4fs | rhs %.3fs spread %.3fs bulk %.3fs restrict %.3fs avg_trace %.3fs\n",
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

std::unique_ptr<ILaplaceSpread2D> make_laplace_spread_2d(
    LaplaceBvpPanelMethod2D method,
    const GridPair2D&       grid_pair,
    double                  eta,
    LaplaceCorrectionMethod2D correction_method,
    int                     restrict_stencil_radius,
    CornerPatchSolverOptions2D corner_patch_solver,
    SdCornerLiftingOptions2D sd_corner_lifting,
    LaplaceCrossingTraceStencil2D crossing_trace_stencil,
    LaplaceCrossingJetScheme2D crossing_jet_scheme)
{
    switch (method) {
    case LaplaceBvpPanelMethod2D::QuadraticPanelCenter:
        return std::make_unique<LaplaceQuadraticPanelCenterSpread2D>(
            grid_pair,
            eta,
            correction_method,
            restrict_stencil_radius,
            std::move(corner_patch_solver),
            std::move(sd_corner_lifting),
            LaplaceP2PanelCenterSpreadMode2D::QuadraticCauchy,
            LaplaceP2CubicHarmonicSpreadOptions2D{},
            crossing_trace_stencil,
            crossing_jet_scheme);
    }
    throw std::invalid_argument("unsupported Laplace BVP panel method");
}

CornerPatchSolverOptions2D make_bvp_corner_patch_solver_options(
    LaplaceBvpType2D type,
    CornerPatchSolverOptions2D options)
{
    if (!is_neumann_type(type) && options.singular_fit.enabled) {
        if (options.singular_fit.family_policy
            == CornerPatchSingularFamilyPolicy2D::Sine) {
            options.singular_fit.family_policy =
                CornerPatchSingularFamilyPolicy2D::ExteriorCosineInteriorSine;
        }
        options.singular_fit.max_sine_modes =
            std::max(options.singular_fit.max_sine_modes, 2);
    }
    if (is_neumann_type(type) && options.singular_fit.enabled
        && options.singular_fit.family_policy
               == CornerPatchSingularFamilyPolicy2D::
                      InteriorNeumannTransmissionPair) {
        options.singular_fit.max_sine_modes =
            std::max(options.singular_fit.max_sine_modes, 2);
    }
    return options;
}

CornerPatchConfig2D make_bvp_corner_patch_config(
    LaplaceBvpType2D type,
    CornerPatchConfig2D config,
    const CornerPatchSolverOptions2D& solver_options)
{
    const bool dirichlet_pair =
        !is_neumann_type(type) && solver_options.singular_fit.enabled
        && solver_options.singular_fit.family_policy
               == CornerPatchSingularFamilyPolicy2D::
                      InteriorDirichletTransmissionPair;
    const bool neumann_pair =
        is_neumann_type(type) && solver_options.singular_fit.enabled
        && solver_options.singular_fit.family_policy
               == CornerPatchSingularFamilyPolicy2D::
                      InteriorNeumannTransmissionPair;
    if (dirichlet_pair || neumann_pair) {
        config.enabled = true;
        config.build_mode =
            CornerPatchBuildMode2D::ReentrantCircleDominantOpening;
        config.radius = std::max(config.radius,
                                 solver_options.singular_fit.basis_radius);
        config.radius = std::max(
            config.radius,
            solver_options.singular_fit.edge_p2_fit_radius);
    }
    return config;
}

std::unique_ptr<ILaplaceRestrict2D> make_laplace_restrict_2d(
    LaplaceBvpPanelMethod2D method,
    const GridPair2D&       grid_pair,
    int                     restrict_stencil_radius,
    LaplaceBvpRestrictMethod2D restrict_method,
    LaplaceP2JointPolynomialRestrictOptions2D joint_options)
{
    switch (method) {
    case LaplaceBvpPanelMethod2D::QuadraticPanelCenter:
        switch (restrict_method) {
        case LaplaceBvpRestrictMethod2D::
                 SixPointQuadraticCrossingOwner:
            return std::make_unique<
                LaplaceQuadraticPanelCenterRestrict2D>(
                    grid_pair, restrict_stencil_radius);
        case LaplaceBvpRestrictMethod2D::
                 JointPolynomialCrossingOwner:
            return std::make_unique<
                LaplaceP2CrossingOwnerJointPolynomialRestrict2D>(
                    grid_pair, std::move(joint_options));
        case LaplaceBvpRestrictMethod2D::
                 UnifiedSpatialNormalP2CrossingOwner:
            return std::make_unique<
                LaplaceP2CrossingOwnerJointPolynomialRestrict2D>(
                    grid_pair,
                    make_laplace_p2_unified_spatial_normal_restrict_options_2d());
        case LaplaceBvpRestrictMethod2D::
                 UnifiedSpatialNormalP2DofCauchyExterior:
            return std::make_unique<
                LaplaceP2CrossingOwnerJointPolynomialRestrict2D>(
                    grid_pair,
                    make_laplace_p2_dof_cauchy_exterior_restrict_options_2d());
        }
        break;
    }
    throw std::invalid_argument(
        "unsupported Laplace BVP panel/restrict method combination");
}

double rhs_deriv_sign_for_type(LaplaceBvpType2D type)
{
    switch (type) {
    case LaplaceBvpType2D::InteriorDirichlet:
    case LaplaceBvpType2D::InteriorNeumann:
        return 1.0;
    case LaplaceBvpType2D::ExteriorDirichlet:
    case LaplaceBvpType2D::ExteriorNeumann:
        return -1.0;
    }
    throw std::invalid_argument("unsupported Laplace BVP type");
}

std::vector<LaplaceJumpData2D> make_jumps(
    const Interface2D&                 iface,
    const Eigen::VectorXd&             density,
    LaplaceBvpType2D                   type,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    const int n_iface = iface.num_points();
    std::vector<LaplaceJumpData2D> jumps(n_iface);
    const bool neumann = is_neumann_type(type);
    for (int q = 0; q < n_iface; ++q) {
        jumps[q].u_jump = neumann ? 0.0 : density[q];
        jumps[q].un_jump = neumann ? density[q] : 0.0;
        jumps[q].rhs_derivs = rhs_derivs[q];
    }
    return jumps;
}

std::vector<LaplaceJumpData2D> make_cauchy_jumps(
    const Interface2D&                  iface,
    const Eigen::VectorXd&              value_jump,
    const Eigen::VectorXd&              normal_jump,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    const int n_iface = iface.num_points();
    if (value_jump.size() != n_iface || normal_jump.size() != n_iface
        || static_cast<int>(rhs_derivs.size()) != n_iface) {
        throw std::invalid_argument(
            "LaplaceBvp2D Cauchy jump data must match the interface size");
    }
    std::vector<LaplaceJumpData2D> jumps(
        static_cast<std::size_t>(n_iface));
    for (int q = 0; q < n_iface; ++q) {
        jumps[static_cast<std::size_t>(q)].u_jump = value_jump[q];
        jumps[static_cast<std::size_t>(q)].un_jump = normal_jump[q];
        jumps[static_cast<std::size_t>(q)].rhs_derivs =
            rhs_derivs[static_cast<std::size_t>(q)];
    }
    return jumps;
}

std::shared_ptr<const Interface2D> make_effective_interface(
    const Interface2D& iface,
    const CornerPatchConfig2D& corner_patch)
{
    return std::make_shared<Interface2D>(
        apply_corner_patch_config_2d(iface, corner_patch));
}

SdCornerLiftingOptions2D make_bvp_sd_corner_lifting_options(
    const CornerPatchConfig2D& corner_patch,
    SdCornerLiftingOptions2D sd_options)
{
    if (sd_options.enabled && corner_patch.enabled
        && corner_patch.build_mode == CornerPatchBuildMode2D::Circle) {
        sd_options.regularized_jump_mode =
            SdCornerRegularizedJumpMode2D::InterpolateResidualAtP2Nodes;
    }
    return sd_options;
}

} // namespace

LaplaceBvp2D::LaplaceBvp2D(
    const CartesianGrid2D& grid,
    const Interface2D&     iface,
    LaplaceBvpType2D       type,
    LaplaceBvpOptions2D    options)
    : effective_iface_(make_effective_interface(
          iface,
          make_bvp_corner_patch_config(type,
                                       options.corner_patch,
                                       options.corner_patch_solver)))
    , grid_pair_(grid, *effective_iface_)
    , spread_(make_laplace_spread_2d(options.panel_method,
                                     grid_pair_,
                                     options.eta,
                                     options.correction_method,
                                     options.restrict_stencil_radius,
                                     make_bvp_corner_patch_solver_options(
                                         type,
                                         options.corner_patch_solver),
                                     make_bvp_sd_corner_lifting_options(
                                         make_bvp_corner_patch_config(
                                             type,
                                             options.corner_patch,
                                             options.corner_patch_solver),
                                         options.sd_corner_lifting),
                                     options.formulation
                                             == LaplaceBvpFormulation2D::
                                                    ExteriorTraceCauchy
                                         ? LaplaceCrossingTraceStencil2D::
                                               PhiP3PsiP2
                                         : laplace_crossing_trace_stencil_for_bvp_2d(
                                               type),
                                     options.crossing_jet_scheme))
    , bulk_solver_(grid, ZfftBcType::Dirichlet, options.eta)
    , restrict_op_(make_laplace_restrict_2d(options.panel_method,
                                            grid_pair_,
                                            options.restrict_stencil_radius,
                                            options.restrict_method,
                                            options.joint_restrict))
    , potentials_(*spread_, bulk_solver_, *restrict_op_)
    , type_(type)
    , formulation_(options.formulation)
    , rhs_deriv_sign_(rhs_deriv_sign_for_type(type))
    , eta_(options.eta)
    , outer_dirichlet_values_(std::move(options.outer_dirichlet_values))
    , active_interface_points_(std::move(options.active_interface_points))
{
    if (options.restrict_method
            != LaplaceBvpRestrictMethod2D::
                   SixPointQuadraticCrossingOwner
        && options.correction_method
               != LaplaceCorrectionMethod2D::CrossingOwner) {
        throw std::invalid_argument(
            "LaplaceBvp2D joint crossing-owner restrict requires crossing-owner spread correction");
    }
    if (options.restrict_method
            != LaplaceBvpRestrictMethod2D::
                   SixPointQuadraticCrossingOwner
        && (!effective_iface_->corners().empty()
            || !effective_iface_->corner_patches().empty()
            || options.sd_corner_lifting.enabled)) {
        throw std::invalid_argument(
            "LaplaceBvp2D joint crossing-owner restrict currently requires a smooth interface without corner lifting");
    }
    if (formulation_ == LaplaceBvpFormulation2D::ExteriorTraceCauchy) {
        if (outer_dirichlet_values_.size() != 0) {
            throw std::invalid_argument(
                "LaplaceBvp2D exterior-trace formulation currently requires homogeneous outer-box Dirichlet data");
        }
        if (!effective_iface_->corners().empty()
            || !effective_iface_->corner_patches().empty()
            || options.sd_corner_lifting.enabled) {
            throw std::invalid_argument(
                "LaplaceBvp2D exterior-trace formulation currently supports smooth interfaces without corner lifting");
        }
    }
    if (outer_dirichlet_values_.size() != 0) {
        require_vector_size("LaplaceBvp2D",
                            "outer_dirichlet_values",
                            outer_dirichlet_values_.size(),
                            grid.num_dofs());
    }
    const int n_iface = grid_pair_.interface().num_points();
    if (active_interface_points_.empty()) {
        active_interface_points_.resize(static_cast<std::size_t>(n_iface));
        std::iota(active_interface_points_.begin(),
                  active_interface_points_.end(),
                  0);
    }
    iface_to_active_.assign(static_cast<std::size_t>(n_iface), -1);
    for (int a = 0; a < static_cast<int>(active_interface_points_.size());
         ++a) {
        const int q = active_interface_points_[static_cast<std::size_t>(a)];
        if (q < 0 || q >= n_iface)
            throw std::invalid_argument(
                "LaplaceBvp2D active_interface_points contains an out-of-range index");
        if (iface_to_active_[static_cast<std::size_t>(q)] >= 0)
            throw std::invalid_argument(
                "LaplaceBvp2D active_interface_points contains a duplicate index");
        iface_to_active_[static_cast<std::size_t>(q)] = a;
    }
    if (active_interface_points_.empty())
        throw std::invalid_argument(
            "LaplaceBvp2D active_interface_points must not be empty");
}

void LaplaceBvp2D::apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const
{
    const int n_iface = grid_pair_.interface().num_points();
    const int n_active = problem_size();
    require_vector_size("LaplaceBvp2D", "operator input", x.size(), n_active);

    if (formulation_ == LaplaceBvpFormulation2D::ExteriorTraceCauchy) {
        apply_exterior_trace_unknown(x, y);
        return;
    }

    const Eigen::VectorXd zero_rhs =
        Eigen::VectorXd::Zero(grid_pair_.grid().num_dofs());
    const std::vector<Eigen::VectorXd> zero_derivs(
        n_iface, Eigen::VectorXd::Zero(1));
    apply_with_rhs(x, zero_rhs, zero_derivs, y);
}

void LaplaceBvp2D::apply_exterior_trace_unknown(
    const Eigen::VectorXd& density,
    Eigen::VectorXd&       y) const
{
    const Interface2D& iface = grid_pair_.interface();
    const int n_iface = iface.num_points();
    const int n_active = problem_size();
    require_vector_size(
        "LaplaceBvp2D", "exterior-trace density", density.size(), n_active);

    const Eigen::VectorXd full_density = expand_active_density(density);
    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n_iface);
    const std::vector<Eigen::VectorXd> zero_derivs(
        static_cast<std::size_t>(n_iface), Eigen::VectorXd::Zero(1));
    const Eigen::VectorXd zero_rhs =
        Eigen::VectorXd::Zero(grid_pair_.grid().num_dofs());

    const bool neumann = is_neumann_type(type_);
    const Eigen::VectorXd& value_jump = neumann ? full_density : zeros;
    const Eigen::VectorXd& normal_jump = neumann ? zeros : full_density;
    const LaplaceCrossingTraceStencil2D trace_stencil = neumann
        ? LaplaceCrossingTraceStencil2D::PhiP3
        : LaplaceCrossingTraceStencil2D::PsiP2;
    const auto result = potentials_.evaluate(
        make_cauchy_jumps(
            iface, value_jump, normal_jump, zero_derivs),
        zero_rhs,
        trace_stencil);

    // s=+1 means the physical domain is the '+' (interior) side, so the
    // ghost trace is the '-' trace.  For an exterior physical BVP s=-1 and
    // the ghost trace is the '+' trace.  In both cases R_ghost = avg-s*jump/2.
    const double physical_side = side_jump_sign(type_);
    y.resize(n_active);
    for (int a = 0; a < n_active; ++a) {
        const int q = active_interface_points_[static_cast<std::size_t>(a)];
        y[a] = neumann
            ? result.u_avg[q] - 0.5 * physical_side * full_density[q]
            : result.un_avg[q] - 0.5 * physical_side * full_density[q];
    }
}

int LaplaceBvp2D::problem_size() const
{
    return static_cast<int>(active_interface_points_.size());
}

Eigen::VectorXd LaplaceBvp2D::apply_dirichlet_boundary_elimination(
    const Eigen::VectorXd& rhs) const
{
    return structured_grid::apply_dirichlet_boundary_elimination(
        "LaplaceBvp2D", grid_pair_.grid(), rhs, outer_dirichlet_values_);
}

void LaplaceBvp2D::restore_dirichlet_boundary(Eigen::VectorXd& u_bulk) const
{
    structured_grid::restore_dirichlet_boundary(
        "LaplaceBvp2D", grid_pair_.grid(), u_bulk, outer_dirichlet_values_);
}

void LaplaceBvp2D::apply_with_rhs(
    const Eigen::VectorXd&              density,
    const Eigen::VectorXd&              rhs,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    Eigen::VectorXd&                    y) const
{
    const int n_iface = grid_pair_.interface().num_points();
    const int n_active = problem_size();
    require_vector_size("LaplaceBvp2D", "operator input", density.size(), n_active);
    require_vector_size("LaplaceBvp2D", "rhs", rhs.size(), grid_pair_.grid().num_dofs());

    const Eigen::VectorXd full_density = expand_active_density(density);
    const std::vector<Eigen::VectorXd> full_rhs_derivs =
        expand_active_rhs_derivs(rhs_derivs);
    auto jumps = make_jumps(grid_pair_.interface(),
                            full_density,
                            type_,
                            full_rhs_derivs);
    auto res = potentials_.evaluate(jumps, rhs);

    y.resize(n_active);
    const double sign = side_jump_sign(type_);
    if (!is_neumann_type(type_)) {
        for (int a = 0; a < n_active; ++a) {
            const int q = active_interface_points_[static_cast<std::size_t>(a)];
            y[a] = res.u_avg[q] + sign * 0.5 * jumps[q].u_jump;
        }
    } else {
        for (int a = 0; a < n_active; ++a) {
            const int q = active_interface_points_[static_cast<std::size_t>(a)];
            y[a] = res.un_avg[q] + sign * 0.5 * jumps[q].un_jump;
        }
    }
}

Eigen::VectorXd LaplaceBvp2D::expand_active_density(
    const Eigen::VectorXd& active_density) const
{
    const int n_active = problem_size();
    require_vector_size("LaplaceBvp2D",
                        "operator input",
                        active_density.size(),
                        n_active);
    const int n_iface = grid_pair_.interface().num_points();
    Eigen::VectorXd full = Eigen::VectorXd::Zero(n_iface);
    for (int a = 0; a < n_active; ++a) {
        const int q = active_interface_points_[static_cast<std::size_t>(a)];
        full[q] = active_density[a];
    }
    return full;
}

std::vector<Eigen::VectorXd> LaplaceBvp2D::expand_active_rhs_derivs(
    const std::vector<Eigen::VectorXd>& rhs_derivs) const
{
    const int n_iface = grid_pair_.interface().num_points();
    const int n_active = problem_size();
    if (static_cast<int>(rhs_derivs.size()) == n_iface)
        return rhs_derivs;
    if (static_cast<int>(rhs_derivs.size()) != n_active) {
        throw std::invalid_argument(
            "LaplaceBvp2D: rhs_derivs length must match active or full interface size");
    }
    std::vector<Eigen::VectorXd> full(n_iface, Eigen::VectorXd::Zero(1));
    for (int a = 0; a < n_active; ++a) {
        const int q = active_interface_points_[static_cast<std::size_t>(a)];
        full[static_cast<std::size_t>(q)] =
            rhs_derivs[static_cast<std::size_t>(a)];
    }
    return full;
}

LaplaceBvpSolveResult2D LaplaceBvp2D::solve(
    const Eigen::VectorXd&              boundary_data,
    const Eigen::VectorXd&              f_bulk,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    int max_iter,
    double tol,
    int restart) const
{
    if (formulation_ == LaplaceBvpFormulation2D::ExteriorTraceCauchy) {
        return solve_exterior_trace(
            boundary_data, f_bulk, rhs_derivs, max_iter, tol, restart);
    }

    StageTimer2D timer;
    const bool profile = profile_solve_2d();

    const int n_iface = grid_pair_.interface().num_points();
    const int n_active = problem_size();
    require_vector_size("LaplaceBvp2D", "boundary_data", boundary_data.size(), n_active);
    require_vector_size("LaplaceBvp2D", "f_bulk", f_bulk.size(), grid_pair_.grid().num_dofs());
    const double t_validate = timer.lap();

    const std::vector<Eigen::VectorXd> full_rhs_derivs =
        expand_active_rhs_derivs(rhs_derivs);
    const auto signed_derivs = signed_rhs_derivs(
        "LaplaceBvp2D", full_rhs_derivs, n_iface, rhs_deriv_sign_);
    const double t_rhs_derivs = timer.lap();

    const Eigen::VectorXd rhs = apply_dirichlet_boundary_elimination(f_bulk);
    const double t_boundary_elim = timer.lap();

    Eigen::VectorXd volume_gamma;
    if (profile)
        potentials_.reset_profile();
    apply_with_rhs(Eigen::VectorXd::Zero(n_active), rhs, signed_derivs, volume_gamma);
    const LaplacePotentialEvalProfile2D volume_profile =
        profile ? potentials_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_volume_apply = timer.lap();

    Eigen::VectorXd b = boundary_data - volume_gamma;
    const Eigen::VectorXd b_before_projection = b;
    const double t_bie_rhs = timer.lap();

    GMRES gmres(max_iter, tol, restart);
    Eigen::VectorXd density = Eigen::VectorXd::Zero(n_active);
    int iterations = 0;
    if (profile)
        potentials_.reset_profile();
    if (uses_neumann_nullspace_projection(type_, eta_)) {
        project_mean_zero(b);
        maybe_write_bvp_rhs_diagnostic_csv(
            std::getenv("KFBIM_DIAG_BVP_RHS_CSV"),
            grid_pair_,
            active_interface_points_,
            boundary_data,
            volume_gamma,
            b_before_projection,
            b);

        MeanProjectedOperator projected_op(*this);
        iterations = gmres.solve(projected_op, b, density);
        project_mean_zero(density);
    } else {
        maybe_write_bvp_rhs_diagnostic_csv(
            std::getenv("KFBIM_DIAG_BVP_RHS_CSV"),
            grid_pair_,
            active_interface_points_,
            boundary_data,
            volume_gamma,
            b_before_projection,
            b);
        iterations = gmres.solve(*this, b, density);
    }
    const LaplacePotentialEvalProfile2D gmres_profile =
        profile ? potentials_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_gmres = timer.lap();

    const Eigen::VectorXd full_density = expand_active_density(density);
    auto jumps = make_jumps(grid_pair_.interface(),
                            full_density,
                            type_,
                            signed_derivs);
    const double t_final_jumps = timer.lap();

    if (profile)
        potentials_.reset_profile();
    auto full_res = potentials_.evaluate(jumps, rhs);
    const LaplacePotentialEvalProfile2D final_profile =
        profile ? potentials_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_final_potential = timer.lap();

    restore_dirichlet_boundary(full_res.u_bulk);
    restore_dirichlet_boundary(full_res.u_physical);
    const double t_restore = timer.lap();

    auto residuals = gmres.residuals();
    const double t_result = timer.lap();
    const double t_total = timer.total();

    if (profile) {
        std::printf("    LaplaceBvp2D::solve profile iface=%d grid=%d iterations=%d\n",
                    n_active, grid_pair_.grid().num_dofs(), iterations);
        std::printf("      setup validate %.3fs rhs_derivs %.3fs boundary_elim %.3fs volume_apply %.3fs bie_rhs %.3fs\n",
                    t_validate, t_rhs_derivs, t_boundary_elim,
                    t_volume_apply, t_bie_rhs);
        std::printf("      gmres %.3fs final_jumps %.3fs final_potential %.3fs restore %.3fs result %.3fs total %.3fs\n",
                    t_gmres, t_final_jumps, t_final_potential, t_restore,
                    t_result, t_total);
        print_potential_profile_2d("volume", volume_profile);
        print_potential_profile_2d("gmres", gmres_profile);
        print_potential_profile_2d("final", final_profile);
        std::printf("      gmres_nonpotential %.3fs\n",
                    profile_total_without_potential(gmres_profile, t_gmres));
    }

    return {std::move(full_res.u_bulk),
            std::move(full_res.u_physical),
            std::move(density),
            std::move(full_res.corner_patch_corrections),
            std::move(residuals),
            iterations,
            gmres.converged()};
}

LaplaceBvpSolveResult2D LaplaceBvp2D::solve_exterior_trace(
    const Eigen::VectorXd&              boundary_data,
    const Eigen::VectorXd&              f_bulk,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    int max_iter,
    double tol,
    int restart) const
{
    const Interface2D& iface = grid_pair_.interface();
    const int n_iface = iface.num_points();
    const int n_active = problem_size();
    require_vector_size(
        "LaplaceBvp2D", "boundary_data", boundary_data.size(), n_active);
    require_vector_size("LaplaceBvp2D",
                        "f_bulk",
                        f_bulk.size(),
                        grid_pair_.grid().num_dofs());
    if (!boundary_data.allFinite() || !f_bulk.allFinite()) {
        throw std::invalid_argument(
            "LaplaceBvp2D exterior-trace inputs must be finite");
    }

    const std::vector<Eigen::VectorXd> full_rhs_derivs =
        expand_active_rhs_derivs(rhs_derivs);
    const std::vector<Eigen::VectorXd> signed_derivs = signed_rhs_derivs(
        "LaplaceBvp2D", full_rhs_derivs, n_iface, rhs_deriv_sign_);
    const Eigen::VectorXd rhs =
        apply_dirichlet_boundary_elimination(f_bulk);

    const bool neumann = is_neumann_type(type_);
    const double physical_side = side_jump_sign(type_);
    // ExteriorTraceCauchy receives the prescribed Cauchy block in the
    // repository's fixed jump convention already.  It is density data, not a
    // physical-side trace that still needs orientation by physical_side.
    const Eigen::VectorXd full_boundary_jump =
        expand_active_density(boundary_data);
    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n_iface);
    const Eigen::VectorXd& fixed_value_jump =
        neumann ? zeros : full_boundary_jump;
    const Eigen::VectorXd& fixed_normal_jump =
        neumann ? full_boundary_jump : zeros;
    const LaplaceCrossingTraceStencil2D fixed_stencil = neumann
        ? LaplaceCrossingTraceStencil2D::PsiP2
        : LaplaceCrossingTraceStencil2D::PhiP3;

    // The prescribed jump block, volume forcing, and RHS jumps form one fixed
    // field. Keeping it outside apply() makes the Krylov operator strictly
    // linear.
    const auto fixed_result = potentials_.evaluate(
        make_cauchy_jumps(iface,
                          fixed_value_jump,
                          fixed_normal_jump,
                          signed_derivs),
        rhs,
        fixed_stencil);
    Eigen::VectorXd b(n_active);
    for (int a = 0; a < n_active; ++a) {
        const int q = active_interface_points_[static_cast<std::size_t>(a)];
        const double fixed_ghost_trace = neumann
            ? fixed_result.u_avg[q]
                - 0.5 * physical_side * fixed_value_jump[q]
            : fixed_result.un_avg[q]
                - 0.5 * physical_side * fixed_normal_jump[q];
        b[a] = -fixed_ghost_trace;
    }

    GMRES gmres(max_iter, tol, restart);
    Eigen::VectorXd density = Eigen::VectorXd::Zero(n_active);
    int iterations = 0;
    if (uses_neumann_nullspace_projection(type_, eta_)) {
        project_mean_zero(b);
        MeanProjectedOperator projected_op(*this);
        iterations = gmres.solve(projected_op, b, density);
        project_mean_zero(density);
    } else {
        iterations = gmres.solve(*this, b, density);
    }

    const Eigen::VectorXd full_density = expand_active_density(density);
    const Eigen::VectorXd& final_value_jump =
        neumann ? full_density : full_boundary_jump;
    const Eigen::VectorXd& final_normal_jump =
        neumann ? full_boundary_jump : full_density;
    auto full_result = potentials_.evaluate(
        make_cauchy_jumps(iface,
                          final_value_jump,
                          final_normal_jump,
                          signed_derivs),
        rhs,
        LaplaceCrossingTraceStencil2D::PhiP3PsiP2);
    restore_dirichlet_boundary(full_result.u_bulk);
    restore_dirichlet_boundary(full_result.u_physical);

    Eigen::VectorXd trace(n_active);
    Eigen::VectorXd normal_trace(n_active);
    Eigen::VectorXd ghost_trace(n_active);
    Eigen::VectorXd ghost_normal_trace(n_active);
    for (int a = 0; a < n_active; ++a) {
        const int q = active_interface_points_[static_cast<std::size_t>(a)];
        trace[a] = full_result.u_avg[q]
                 + 0.5 * physical_side * final_value_jump[q];
        normal_trace[a] = full_result.un_avg[q]
                        + 0.5 * physical_side * final_normal_jump[q];
        ghost_trace[a] = full_result.u_avg[q]
                       - 0.5 * physical_side * final_value_jump[q];
        ghost_normal_trace[a] = full_result.un_avg[q]
                              - 0.5 * physical_side
                                    * final_normal_jump[q];
    }

    return {std::move(full_result.u_bulk),
            std::move(full_result.u_physical),
            std::move(density),
            std::move(full_result.corner_patch_corrections),
            gmres.residuals(),
            iterations,
            gmres.converged(),
            std::move(trace),
            std::move(normal_trace),
            std::move(ghost_trace),
            std::move(ghost_normal_trace)};
}

} // namespace kfbim
