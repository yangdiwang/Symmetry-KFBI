#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "../../geometry/grid_pair_2d.hpp"
#include "../../geometry/grid_pair_3d.hpp"
#include "../../local_cauchy/jump_data.hpp"
#include "../i_kfbi_operator.hpp"

namespace kfbim::operators_detail {

inline void require_vector_size(const char* context,
                                const char* name,
                                Eigen::Index actual,
                                Eigen::Index expected)
{
    if (actual != expected) {
        throw std::invalid_argument(
            std::string(context) + ": " + name
            + " has size " + std::to_string(actual)
            + ", expected " + std::to_string(expected));
    }
}

inline void project_mean_zero(Eigen::VectorXd& v)
{
    if (v.size() == 0)
        throw std::invalid_argument("project_mean_zero: vector must be nonempty");
    v.array() -= v.mean();
}

class MeanProjectedOperator final : public IKFBIOperator {
public:
    explicit MeanProjectedOperator(const IKFBIOperator& op)
        : op_(op)
    {
        if (op_.problem_size() <= 0) {
            throw std::invalid_argument(
                "MeanProjectedOperator: problem size must be positive");
        }
    }

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override
    {
        Eigen::VectorXd x_projected = x;
        project_mean_zero(x_projected);

        op_.apply(x_projected, y);
        project_mean_zero(y);
    }

    int problem_size() const override { return op_.problem_size(); }

private:
    const IKFBIOperator& op_;
};

inline double side_jump_sign(bool is_exterior)
{
    return is_exterior ? -1.0 : 1.0;
}

inline bool uses_interior_neumann_nullspace_projection(bool is_interior_neumann,
                                                       double eta)
{
    return is_interior_neumann && std::abs(eta) < 1.0e-14;
}

inline double checked_lambda_sq(const char* context,
                                const char* phase,
                                double beta,
                                double kappa_sq)
{
    if (beta <= 0.0) {
        throw std::invalid_argument(
            std::string(context) + ": " + phase + " beta must be positive");
    }
    if (kappa_sq < 0.0) {
        throw std::invalid_argument(
            std::string(context) + ": " + phase
            + " kappa_sq must be nonnegative");
    }
    return kappa_sq / beta;
}

inline void require_common_ratio(const char* context,
                                 double lambda_sq_int,
                                 double lambda_sq_ext)
{
    const double scale =
        std::max({1.0, std::abs(lambda_sq_int), std::abs(lambda_sq_ext)});
    if (std::abs(lambda_sq_int - lambda_sq_ext) > 1.0e-12 * scale) {
        throw std::invalid_argument(
            std::string(context)
            + ": CommonRatio mode requires kappa_sq_int/beta_int == "
              "kappa_sq_ext/beta_ext");
    }
}

inline void validate_rhs_derivs(const char* context,
                                const char* name,
                                const std::vector<Eigen::VectorXd>& derivs,
                                int n_iface)
{
    if (static_cast<int>(derivs.size()) != n_iface) {
        throw std::invalid_argument(
            std::string(context) + ": " + name
            + " length does not match interface size");
    }
    for (int q = 0; q < n_iface; ++q) {
        if (derivs[q].size() == 0) {
            throw std::invalid_argument(
                std::string(context) + ": " + name
                + " entries must contain at least the value derivative");
        }
    }
}

inline std::vector<Eigen::VectorXd> signed_rhs_derivs(
    const char* context,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    int n_iface,
    double sign)
{
    if (static_cast<int>(rhs_derivs.size()) != n_iface) {
        throw std::invalid_argument(
            std::string(context)
            + ": rhs_derivs length does not match interface size");
    }

    std::vector<Eigen::VectorXd> signed_derivs(n_iface);
    for (int q = 0; q < n_iface; ++q)
        signed_derivs[q] = sign * rhs_derivs[q];
    return signed_derivs;
}

inline std::vector<Eigen::VectorXd> combine_rhs_derivs(
    const char* context,
    const std::vector<Eigen::VectorXd>& int_derivs,
    const std::vector<Eigen::VectorXd>& ext_derivs)
{
    const int n_iface = static_cast<int>(int_derivs.size());
    std::vector<Eigen::VectorXd> combined(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        if (int_derivs[q].size() != ext_derivs[q].size()) {
            throw std::invalid_argument(
                std::string(context)
                + ": interior and exterior rhs derivative entries must have matching sizes");
        }
        combined[q] = int_derivs[q] - ext_derivs[q];
    }
    return combined;
}

inline std::vector<Eigen::VectorXd> scaled_rhs_derivs(
    const std::vector<Eigen::VectorXd>& derivs,
    double scale)
{
    const int n_iface = static_cast<int>(derivs.size());
    std::vector<Eigen::VectorXd> scaled(n_iface);
    for (int q = 0; q < n_iface; ++q)
        scaled[q] = scale * derivs[q];
    return scaled;
}

inline std::vector<Eigen::VectorXd> zero_rhs_derivs(int n_iface)
{
    return std::vector<Eigen::VectorXd>(n_iface, Eigen::VectorXd::Zero(1));
}

inline std::vector<LaplaceJumpData2D> make_laplace_jumps_2d(
    const char* context,
    int n_iface,
    const Eigen::VectorXd& u_jump,
    const Eigen::VectorXd& un_jump,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    require_vector_size(context, "u_jump", u_jump.size(), n_iface);
    require_vector_size(context, "un_jump", un_jump.size(), n_iface);
    validate_rhs_derivs(context, "rhs_derivs", rhs_derivs, n_iface);

    std::vector<LaplaceJumpData2D> jumps(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        jumps[q].u_jump = u_jump[q];
        jumps[q].un_jump = un_jump[q];
        jumps[q].rhs_derivs = rhs_derivs[q];
    }
    return jumps;
}

inline std::vector<LaplaceJumpData3D> make_laplace_jumps_3d(
    const char* context,
    int n_iface,
    const Eigen::VectorXd& u_jump,
    const Eigen::VectorXd& un_jump,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    require_vector_size(context, "u_jump", u_jump.size(), n_iface);
    require_vector_size(context, "un_jump", un_jump.size(), n_iface);
    validate_rhs_derivs(context, "rhs_derivs", rhs_derivs, n_iface);

    std::vector<LaplaceJumpData3D> jumps(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        jumps[q].u_jump = u_jump[q];
        jumps[q].un_jump = un_jump[q];
        jumps[q].rhs_derivs = rhs_derivs[q];
    }
    return jumps;
}

inline void split_phase_rhs(const char* context,
                            const GridPair2D& grid_pair,
                            const Eigen::VectorXd& rhs_bulk,
                            Eigen::VectorXd& rhs_int,
                            Eigen::VectorXd& rhs_ext)
{
    const int n_dof = grid_pair.grid().num_dofs();
    require_vector_size(context,
                        "rhs_data.reduced_rhs_bulk",
                        rhs_bulk.size(),
                        n_dof);

    rhs_int = Eigen::VectorXd::Zero(n_dof);
    rhs_ext = Eigen::VectorXd::Zero(n_dof);
    for (int n = 0; n < n_dof; ++n) {
        if (grid_pair.domain_label(n) > 0)
            rhs_int[n] = rhs_bulk[n];
        else
            rhs_ext[n] = rhs_bulk[n];
    }
}

inline void split_phase_rhs(const char* context,
                            const GridPair3D& grid_pair,
                            const Eigen::VectorXd& rhs_bulk,
                            Eigen::VectorXd& rhs_int,
                            Eigen::VectorXd& rhs_ext)
{
    const int n_dof = grid_pair.grid().num_dofs();
    require_vector_size(context,
                        "rhs_data.reduced_rhs_bulk",
                        rhs_bulk.size(),
                        n_dof);

    rhs_int = Eigen::VectorXd::Zero(n_dof);
    rhs_ext = Eigen::VectorXd::Zero(n_dof);
    for (int n = 0; n < n_dof; ++n) {
        if (grid_pair.domain_label(n) > 0)
            rhs_int[n] = rhs_bulk[n];
        else
            rhs_ext[n] = rhs_bulk[n];
    }
}

} // namespace kfbim::operators_detail
