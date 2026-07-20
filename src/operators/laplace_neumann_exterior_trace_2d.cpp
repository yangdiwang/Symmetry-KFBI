#include "laplace_neumann_exterior_trace_2d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "../gmres/gmres.hpp"
#include "detail/laplace_operator_utils.hpp"

namespace kfbim {

namespace {

constexpr const char* kContext = "LaplaceNeumannExteriorTrace2D";

double inf_norm(const Eigen::VectorXd& values)
{
    return values.size() == 0
        ? 0.0
        : values.cwiseAbs().maxCoeff();
}

double relative_norm(const Eigen::VectorXd& residual,
                     const Eigen::VectorXd& rhs)
{
    const double denominator = rhs.norm();
    return denominator > std::numeric_limits<double>::min()
        ? residual.norm() / denominator
        : residual.norm();
}

class BorderedExteriorTraceOperator2D final : public IKFBIOperator {
public:
    BorderedExteriorTraceOperator2D(
        const LaplaceNeumannExteriorTrace2D& physical_operator,
        const Eigen::VectorXd& border_column,
        const Eigen::VectorXd& normalized_weights)
        : physical_operator_(physical_operator)
        , border_column_(border_column)
        , normalized_weights_(normalized_weights)
    {}

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override
    {
        const int n = physical_operator_.problem_size();
        operators_detail::require_vector_size(
            kContext, "bordered operator input", x.size(), n + 1);

        Eigen::VectorXd af;
        physical_operator_.apply(x.head(n), af);

        y.resize(n + 1);
        y.head(n) = af + x[n] * border_column_;
        y[n] = normalized_weights_.dot(x.head(n));
    }

    int problem_size() const override
    {
        return physical_operator_.problem_size() + 1;
    }

private:
    const LaplaceNeumannExteriorTrace2D& physical_operator_;
    const Eigen::VectorXd& border_column_;
    const Eigen::VectorXd& normalized_weights_;
};

} // namespace

LaplaceNeumannExteriorTrace2D::LaplaceNeumannExteriorTrace2D(
    const CartesianGrid2D& grid,
    const Interface2D& iface,
    LaplaceNeumannExteriorTraceOptions2D options)
    : grid_pair_(grid, iface)
    , spread_(grid_pair_,
              0.0,
              options.correction_method,
              options.restrict_stencil_radius,
              {},
              {})
    , bulk_solver_(grid, ZfftBcType::Dirichlet, 0.0)
    , restrict_op_(grid_pair_, options.restrict_stencil_radius)
    , active_interface_points_(
          std::move(options.active_interface_points))
    , border_column_(std::move(options.border_column))
    , zero_bulk_rhs_(Eigen::VectorXd::Zero(grid.num_dofs()))
    , zero_rhs_derivs_(
          static_cast<std::size_t>(iface.num_points()),
          Eigen::VectorXd::Zero(1))
{
    if (iface.num_components() != 1) {
        throw std::invalid_argument(
            "LaplaceNeumannExteriorTrace2D currently requires one interface component");
    }
    if (iface.points_per_panel() != 3
        || iface.panel_node_layout()
               != PanelNodeLayout2D::QuadraticLagrange) {
        throw std::invalid_argument(
            "LaplaceNeumannExteriorTrace2D requires P2 quadratic 3-point panels");
    }
    if (!iface.corner_patches().empty()) {
        throw std::invalid_argument(
            "LaplaceNeumannExteriorTrace2D does not accept corner patches or singular corrections");
    }

    const int n_iface = iface.num_points();
    if (active_interface_points_.empty()) {
        active_interface_points_.resize(static_cast<std::size_t>(n_iface));
        std::iota(active_interface_points_.begin(),
                  active_interface_points_.end(),
                  0);
    }
    if (active_interface_points_.empty()) {
        throw std::invalid_argument(
            "LaplaceNeumannExteriorTrace2D requires at least one active interface point");
    }

    iface_to_active_.assign(static_cast<std::size_t>(n_iface), -1);
    for (int a = 0;
         a < static_cast<int>(active_interface_points_.size());
         ++a) {
        const int q =
            active_interface_points_[static_cast<std::size_t>(a)];
        if (q < 0 || q >= n_iface) {
            throw std::invalid_argument(
                "LaplaceNeumannExteriorTrace2D active point index is out of range");
        }
        if (iface_to_active_[static_cast<std::size_t>(q)] >= 0) {
            throw std::invalid_argument(
                "LaplaceNeumannExteriorTrace2D active point list contains a duplicate");
        }
        iface_to_active_[static_cast<std::size_t>(q)] = a;
    }

    std::vector<char> referenced(static_cast<std::size_t>(n_iface), 0);
    for (int panel = 0; panel < iface.num_panels(); ++panel) {
        for (int local = 0; local < iface.points_per_panel(); ++local) {
            const int q = iface.point_index(panel, local);
            referenced[static_cast<std::size_t>(q)] = 1;
            if (iface_to_active_[static_cast<std::size_t>(q)] < 0) {
                throw std::invalid_argument(
                    "LaplaceNeumannExteriorTrace2D every panel point must be active");
            }
        }
    }

    const double max_weight = iface.weights().size() == 0
        ? 0.0
        : iface.weights().cwiseAbs().maxCoeff();
    const double zero_weight_tolerance =
        64.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, max_weight);
    for (int q = 0; q < n_iface; ++q) {
        const bool active =
            iface_to_active_[static_cast<std::size_t>(q)] >= 0;
        if (!active
            && (referenced[static_cast<std::size_t>(q)] != 0
                || std::abs(iface.weights()[q]) > zero_weight_tolerance)) {
            throw std::invalid_argument(
                "LaplaceNeumannExteriorTrace2D inactive points must be unreferenced zero-weight metadata");
        }
    }

    const int n_active = problem_size();
    active_weights_.resize(n_active);
    for (int a = 0; a < n_active; ++a) {
        const int q =
            active_interface_points_[static_cast<std::size_t>(a)];
        const double weight = iface.weights()[q];
        if (!(weight > 0.0) || !std::isfinite(weight)) {
            throw std::invalid_argument(
                "LaplaceNeumannExteriorTrace2D active quadrature weights must be finite and positive");
        }
        active_weights_[a] = weight;
    }
    const double total_weight = active_weights_.sum();
    if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
        throw std::invalid_argument(
            "LaplaceNeumannExteriorTrace2D active quadrature weight sum must be positive");
    }
    normalized_weights_ = active_weights_ / total_weight;

    if (border_column_.size() == 0)
        border_column_ = Eigen::VectorXd::Ones(n_active);
    operators_detail::require_vector_size(
        kContext, "border_column", border_column_.size(), n_active);
    if (!border_column_.allFinite() || border_column_.norm() == 0.0) {
        throw std::invalid_argument(
            "LaplaceNeumannExteriorTrace2D border column must be finite and nonzero");
    }
}

int LaplaceNeumannExteriorTrace2D::problem_size() const
{
    return static_cast<int>(active_interface_points_.size());
}

Eigen::VectorXd LaplaceNeumannExteriorTrace2D::expand_active(
    const Eigen::VectorXd& values) const
{
    const int n_active = problem_size();
    operators_detail::require_vector_size(
        kContext, "active interface vector", values.size(), n_active);

    Eigen::VectorXd full =
        Eigen::VectorXd::Zero(grid_pair_.interface().num_points());
    for (int a = 0; a < n_active; ++a) {
        const int q =
            active_interface_points_[static_cast<std::size_t>(a)];
        full[q] = values[a];
    }
    return full;
}

LaplaceNeumannExteriorTrace2D::JumpFieldResult2D
LaplaceNeumannExteriorTrace2D::evaluate_jump(
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump,
    bool recover_exterior_virtual) const
{
    const int n_active = problem_size();
    operators_detail::require_vector_size(
        kContext, "value jump", value_jump.size(), n_active);
    operators_detail::require_vector_size(
        kContext, "normal jump", normal_jump.size(), n_active);

    const Eigen::VectorXd full_value_jump = expand_active(value_jump);
    const Eigen::VectorXd full_normal_jump = expand_active(normal_jump);
    const auto jumps = operators_detail::make_laplace_jumps_2d(
        kContext,
        grid_pair_.interface().num_points(),
        full_value_jump,
        full_normal_jump,
        zero_rhs_derivs_);

    Eigen::VectorXd rhs = zero_bulk_rhs_;
    const LaplaceSpreadResult2D spread_result = spread_.apply(jumps, rhs);

    Eigen::VectorXd u_bulk;
    bulk_solver_.solve(-rhs, u_bulk);
    const std::vector<LocalPoly2D> interior_polys =
        restrict_op_.apply_interior_virtual(u_bulk, spread_result);
    std::vector<LocalPoly2D> exterior_polys;
    if (recover_exterior_virtual) {
        exterior_polys =
            restrict_op_.apply_exterior_virtual(u_bulk, spread_result);
    }

    JumpFieldResult2D result;
    result.u_bulk = std::move(u_bulk);
    result.trace_interior.resize(n_active);
    result.normal_trace_interior.resize(n_active);
    if (recover_exterior_virtual) {
        result.trace_exterior_virtual.resize(n_active);
        result.normal_trace_exterior_virtual.resize(n_active);
    }
    const Interface2D& iface = grid_pair_.interface();
    for (int a = 0; a < n_active; ++a) {
        const int q =
            active_interface_points_[static_cast<std::size_t>(a)];
        const LocalPoly2D& poly =
            interior_polys[static_cast<std::size_t>(q)];
        if (poly.coeffs.size() < 3) {
            throw std::runtime_error(
                "LaplaceNeumannExteriorTrace2D restrict polynomial is incomplete");
        }
        result.trace_interior[a] = poly.coeffs[0];
        result.normal_trace_interior[a] =
            poly.coeffs[1] * iface.normals()(q, 0)
          + poly.coeffs[2] * iface.normals()(q, 1);
        if (recover_exterior_virtual) {
            const LocalPoly2D& exterior_poly =
                exterior_polys[static_cast<std::size_t>(q)];
            if (exterior_poly.coeffs.size() < 3) {
                throw std::runtime_error(
                    "LaplaceNeumannExteriorTrace2D exterior restrict polynomial is incomplete");
            }
            result.trace_exterior_virtual[a] = exterior_poly.coeffs[0];
            result.normal_trace_exterior_virtual[a] =
                exterior_poly.coeffs[1] * iface.normals()(q, 0)
              + exterior_poly.coeffs[2] * iface.normals()(q, 1);
        }
    }
    return result;
}

void LaplaceNeumannExteriorTrace2D::apply(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& y) const
{
    const int n_active = problem_size();
    operators_detail::require_vector_size(
        kContext, "operator input", x.size(), n_active);

    const JumpFieldResult2D value_field = evaluate_jump(
        x, Eigen::VectorXd::Zero(n_active), true);

    // Match the Python harmonic-jet iteration literally:
    //
    //     A f = R_ext H_D(f),
    //
    // where the exterior trace is recovered directly from exterior virtual
    // samples.  The fixed Neumann contribution is kept out of this matvec.
    y = value_field.trace_exterior_virtual;
}

void LaplaceNeumannExteriorTrace2D::apply_direct_exterior(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& y) const
{
    const int n_active = problem_size();
    operators_detail::require_vector_size(
        kContext, "direct exterior operator input", x.size(), n_active);
    apply(x, y);
}

LaplaceNeumannExteriorTraceRhs2D
LaplaceNeumannExteriorTrace2D::build_rhs(
    const Eigen::VectorXd& neumann_data) const
{
    const int n_active = problem_size();
    operators_detail::require_vector_size(
        kContext, "neumann_data", neumann_data.size(), n_active);
    if (!neumann_data.allFinite()) {
        throw std::invalid_argument(
            "LaplaceNeumannExteriorTrace2D Neumann data must be finite");
    }

    const double compatibility_mean =
        normalized_weights_.dot(neumann_data);
    const Eigen::VectorXd compatible_neumann =
        neumann_data
        - compatibility_mean * Eigen::VectorXd::Ones(n_active);
    const JumpFieldResult2D fixed_field = evaluate_jump(
        Eigen::VectorXd::Zero(n_active), compatible_neumann, true);

    LaplaceNeumannExteriorTraceRhs2D result;
    result.compatible_neumann_data = compatible_neumann;
    // Python equation:
    //
    //     R_ext H_D(f) = -R_ext H_N(g).
    //
    // Store the production RHS in trace_rhs.  Keep the explicitly named
    // field equal to it so callers cannot accidentally use the old sign.
    result.trace_rhs = -fixed_field.trace_exterior_virtual;
    result.trace_rhs_exterior_virtual = result.trace_rhs;
    result.fixed_u_bulk = fixed_field.u_bulk;
    result.fixed_normal_trace_interior =
        fixed_field.normal_trace_interior;
    result.compatibility_mean_removed = compatibility_mean;
    result.compatibility_residual =
        normalized_weights_.dot(compatible_neumann);
    result.trace_route_mismatch_inf = inf_norm(
        fixed_field.trace_interior
        - fixed_field.trace_exterior_virtual);
    return result;
}

LaplaceExteriorTraceField2D
LaplaceNeumannExteriorTrace2D::evaluate_cauchy(
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump) const
{
    const JumpFieldResult2D field = evaluate_jump(
        value_jump, normal_jump, true);

    LaplaceExteriorTraceField2D result;
    result.u_bulk = field.u_bulk;
    result.trace_interior = field.trace_interior;
    result.trace_exterior_from_jump =
        field.trace_interior - value_jump;
    result.trace_exterior_virtual = field.trace_exterior_virtual;
    result.normal_trace_interior = field.normal_trace_interior;
    result.normal_trace_exterior_from_jump =
        field.normal_trace_interior - normal_jump;
    result.normal_trace_exterior_virtual =
        field.normal_trace_exterior_virtual;
    result.trace_route_mismatch_inf = inf_norm(
        result.trace_exterior_from_jump
        - result.trace_exterior_virtual);
    return result;
}

LaplaceNeumannExteriorTraceSolveResult2D
LaplaceNeumannExteriorTrace2D::solve(
    const Eigen::VectorXd& neumann_data,
    int max_iter,
    double tol,
    int restart) const
{
    return solve(build_rhs(neumann_data), max_iter, tol, restart);
}

LaplaceNeumannExteriorTraceSolveResult2D
LaplaceNeumannExteriorTrace2D::solve(
    const LaplaceNeumannExteriorTraceRhs2D& rhs_data,
    int max_iter,
    double tol,
    int restart) const
{
    const int n_active = problem_size();
    operators_detail::require_vector_size(
        kContext,
        "compatible_neumann_data",
        rhs_data.compatible_neumann_data.size(),
        n_active);
    operators_detail::require_vector_size(
        kContext, "trace_rhs", rhs_data.trace_rhs.size(), n_active);
    operators_detail::require_vector_size(
        kContext,
        "trace_rhs_exterior_virtual",
        rhs_data.trace_rhs_exterior_virtual.size(),
        n_active);
    operators_detail::require_vector_size(
        kContext,
        "fixed_u_bulk",
        rhs_data.fixed_u_bulk.size(),
        grid_pair_.grid().num_dofs());
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(n_active);
    const Eigen::VectorXd& compatible_neumann =
        rhs_data.compatible_neumann_data;
    const Eigen::VectorXd& b = rhs_data.trace_rhs;

    Eigen::VectorXd constant_mode_image;
    apply(ones, constant_mode_image);

    BorderedExteriorTraceOperator2D bordered(
        *this, border_column_, normalized_weights_);
    Eigen::VectorXd bordered_rhs = Eigen::VectorXd::Zero(n_active + 1);
    bordered_rhs.head(n_active) = b;
    Eigen::VectorXd bordered_solution =
        Eigen::VectorXd::Zero(n_active + 1);

    GMRES gmres(max_iter, tol, restart);
    const int iterations =
        gmres.solve(bordered, bordered_rhs, bordered_solution);

    const Eigen::VectorXd f = bordered_solution.head(n_active);
    const double lambda = bordered_solution[n_active];
    const JumpFieldResult2D value_field = evaluate_jump(
        f, Eigen::VectorXd::Zero(n_active), true);
    const JumpFieldResult2D combined_field = evaluate_jump(
        f, compatible_neumann, true);

    const Eigen::VectorXd& af = value_field.trace_exterior_virtual;
    const Eigen::VectorXd physical_residual = b - af;
    const Eigen::VectorXd& trace_interior =
        combined_field.trace_interior;
    const Eigen::VectorXd trace_exterior =
        combined_field.trace_interior - f;
    const Eigen::VectorXd& normal_trace_interior =
        combined_field.normal_trace_interior;
    const Eigen::VectorXd normal_trace_exterior =
        normal_trace_interior - compatible_neumann;

    Eigen::VectorXd bordered_image;
    bordered.apply(bordered_solution, bordered_image);
    const Eigen::VectorXd augmented_residual =
        bordered_rhs - bordered_image;

    LaplaceNeumannExteriorTraceSolveResult2D result;
    result.u_bulk = combined_field.u_bulk;
    result.dirichlet_trace = f;
    result.compatible_neumann_data = compatible_neumann;
    result.fixed_trace_rhs = b;
    result.trace_interior = trace_interior;
    result.trace_exterior = trace_exterior;
    result.trace_exterior_virtual =
        combined_field.trace_exterior_virtual;
    result.normal_trace_interior = normal_trace_interior;
    result.normal_trace_exterior = normal_trace_exterior;
    result.physical_trace_residual = physical_residual;
    result.augmented_residual = augmented_residual;
    result.augmented_relative_residuals = gmres.residuals();
    result.compatibility_mean_removed =
        rhs_data.compatibility_mean_removed;
    result.bordered_multiplier = lambda;
    result.weighted_trace_mean = normalized_weights_.dot(f);
    result.constant_mode_residual_inf = inf_norm(constant_mode_image);
    // With the Python sign convention b-Af is the negative total exterior
    // trace.  Check the production direct route first, then audit the
    // independent jump-derived exterior trace.
    result.trace_residual_closure_inf = inf_norm(
        combined_field.trace_exterior_virtual + physical_residual);
    result.independent_exterior_closure_inf = inf_norm(
        trace_exterior + physical_residual);
    result.rhs_trace_route_mismatch_inf =
        rhs_data.trace_route_mismatch_inf;
    result.superposition_bulk_inf = inf_norm(
        combined_field.u_bulk
        - value_field.u_bulk
        - rhs_data.fixed_u_bulk);
    result.physical_relative_residual =
        relative_norm(physical_residual, b);
    result.exterior_trace_relative = relative_norm(
        combined_field.trace_exterior_virtual, b);
    result.augmented_relative_residual =
        relative_norm(augmented_residual, bordered_rhs);
    result.iterations = iterations;
    result.augmented_converged = gmres.converged();
    const double gauge_scale = std::max(1.0, inf_norm(f));
    const double trace_scale = std::max(1.0, inf_norm(b));
    const double bulk_scale = std::max(1.0, inf_norm(combined_field.u_bulk));
    result.physical_converged =
        result.physical_relative_residual <= 10.0 * tol
        && result.exterior_trace_relative <= 10.0 * tol
        && std::abs(result.weighted_trace_mean)
               <= 10.0 * tol * gauge_scale
        && result.trace_residual_closure_inf <= 10.0 * tol * trace_scale
        && result.superposition_bulk_inf <= 10.0 * tol * bulk_scale;
    return result;
}

} // namespace kfbim
