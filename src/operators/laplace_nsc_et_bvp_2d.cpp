#include "laplace_nsc_et_bvp_2d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "../gmres/gmres.hpp"

namespace kfbim {
namespace {

const geometry2d::NurbsBoundaryPanelGeometry2D& require_nurbs_geometry(
    const Interface2D& iface)
{
    if (!iface.has_panel_geometry()) {
        throw std::invalid_argument(
            "NSC-ET-KFBI requires an exact NURBS panel-geometry provider");
    }
    const auto* geometry = dynamic_cast<const
        geometry2d::NurbsBoundaryPanelGeometry2D*>(&iface.panel_geometry());
    if (geometry == nullptr
        || geometry->num_parameterized_points() != iface.num_points()) {
        throw std::invalid_argument(
            "NSC-ET-KFBI requires parameterized NURBS data for every compatibility point");
    }
    return *geometry;
}

std::vector<LaplaceJumpData2D> make_harmonic_jumps(
    const Eigen::VectorXd& phi,
    const Eigen::VectorXd& psi)
{
    if (phi.size() != psi.size()) {
        throw std::invalid_argument(
            "NSC-ET-KFBI phi/psi interface sizes differ");
    }
    std::vector<LaplaceJumpData2D> jumps(
        static_cast<std::size_t>(phi.size()));
    for (int q = 0; q < phi.size(); ++q) {
        jumps[static_cast<std::size_t>(q)].u_jump = phi[q];
        jumps[static_cast<std::size_t>(q)].un_jump = psi[q];
        jumps[static_cast<std::size_t>(q)].rhs_derivs =
            Eigen::VectorXd::Zero(1);
    }
    return jumps;
}

double relative_norm(const Eigen::VectorXd& residual,
                     const Eigen::VectorXd& reference)
{
    const double denominator = reference.norm();
    return residual.norm()
         / std::max(denominator, std::numeric_limits<double>::min());
}

double inf_norm(const Eigen::VectorXd& values)
{
    return values.size() == 0 ? 0.0 : values.cwiseAbs().maxCoeff();
}

struct GaussLegendreRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

GaussLegendreRule gauss_legendre_rule(int order)
{
    switch (order) {
    case 2:
        return {{-0.57735026918962576451,
                  0.57735026918962576451},
                {1.0, 1.0}};
    case 3:
        return {{-0.77459666924148337704,
                  0.0,
                  0.77459666924148337704},
                {0.55555555555555555556,
                 0.88888888888888888889,
                 0.55555555555555555556}};
    case 4:
        return {{-0.86113631159405257522,
                 -0.33998104358485626480,
                  0.33998104358485626480,
                  0.86113631159405257522},
                {0.34785484513745385737,
                 0.65214515486254614263,
                 0.65214515486254614263,
                 0.34785484513745385737}};
    case 8:
        return {{-0.96028985649753623168,
                 -0.79666647741362673959,
                 -0.52553240991632898582,
                 -0.18343464249564980494,
                  0.18343464249564980494,
                  0.52553240991632898582,
                  0.79666647741362673959,
                  0.96028985649753623168},
                {0.10122853629037625915,
                 0.22238103445337447054,
                 0.31370664587788728734,
                 0.36268378337836198297,
                 0.36268378337836198297,
                 0.31370664587788728734,
                 0.22238103445337447054,
                 0.10122853629037625915}};
    default:
        throw std::invalid_argument(
            "NSC-ET-KFBI supports trace Gauss orders 2, 3, 4, and 8");
    }
}

} // namespace

LaplaceNscEtBvp2D::LaplaceNscEtBvp2D(
    const CartesianGrid2D& grid,
    const Interface2D& iface,
    LaplaceNscEtBvpType2D type,
    LaplaceNscEtOptions2D options)
    : type_(type)
    , grid_pair_(grid, iface)
    , spread_(grid_pair_,
              0.0,
              options.correction_method,
              options.restrict_stencil_radius,
              {},
              {},
              LaplaceP2PanelCenterSpreadMode2D::QuadraticCauchy,
              {},
              LaplaceCrossingTraceStencil2D::PhiP3PsiP2,
              // Every production application below supplies the immutable
              // direct coefficient state.  Do not construct the legacy
              // nodal-fit NSP-CJ plan merely as an unused fallback: its
              // component-level completeness rules are stricter than the
              // exact provider required by the coefficient spaces and can
              // reject otherwise valid refined/cornered geometries.
              LaplaceCrossingJetScheme2D::LocalArclengthLagrange)
    , bulk_solver_(grid, ZfftBcType::Dirichlet, 0.0)
    , restrict_(grid_pair_, std::move(options.restrict_options))
    , use_full_neumann_bordered_system_(
          options.use_full_neumann_bordered_system)
    , trace_gauss_order_(options.trace_gauss_order)
    , density_derivative_scheme_(options.density_derivative_scheme)
    , density_difference_step_over_span_(
          options.density_derivative_scheme
                  == NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference
              ? options.covariant_parameter_difference_step_over_span
              : options.density_difference_step_over_span)
{
    if (iface.points_per_panel() != 3
        || iface.panel_node_layout()
               != PanelNodeLayout2D::QuadraticLagrange) {
        throw std::invalid_argument(
            "NSC-ET-KFBI requires quadratic-Lagrange compatibility panels");
    }
    if (options.correction_method != LaplaceCorrectionMethod2D::CrossingOwner) {
        throw std::invalid_argument(
            "NSC-ET-KFBI requires true crossing-owner spread correction");
    }
    if (!(options.density_spacing_over_h > 0.0)
        || !std::isfinite(options.density_spacing_over_h)) {
        throw std::invalid_argument(
            "NSC-ET-KFBI density spacing ratio must be finite and positive");
    }
    if (!(density_difference_step_over_span_ > 0.0)
        || !std::isfinite(density_difference_step_over_span_)) {
        throw std::invalid_argument(
            "NSC-ET-KFBI density difference spacing must be finite and positive");
    }
    if (density_derivative_scheme_
            == NurbsDensityDerivativeScheme2D::SampledFiniteDifference
        && options.density_coordinate
               != NurbsDensityCoordinate2D::PhysicalArclength) {
        throw std::invalid_argument(
            "NSC-ET-KFBI sampled density differences require physical arclength");
    }
    if (density_derivative_scheme_
            == NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference
        && options.density_coordinate
               != NurbsDensityCoordinate2D::LegacyNurbsParameter) {
        throw std::invalid_argument(
            "NSC-ET-KFBI covariant parameter differences require NURBS-parameter coordinates");
    }
    const auto& geometry = require_nurbs_geometry(iface);
    const int branches = geometry.num_spans();
    if (static_cast<int>(options.phi_connections.size()) != branches
        || static_cast<int>(options.psi_connections.size()) != branches) {
        throw std::invalid_argument(
            "NSC-ET-KFBI needs one phi/psi continuity entry per NURBS branch connection");
    }
    if (std::any_of(options.psi_connections.begin(),
                    options.psi_connections.end(),
                    [](NurbsDensityContinuity2D continuity) {
                        return continuity == NurbsDensityContinuity2D::C2;
                    })) {
        throw std::invalid_argument(
            "NSC-ET-KFBI P2 psi density supports at most C1 continuity");
    }
    const double h = std::max(grid.spacing()[0], grid.spacing()[1]);
    const double target_spacing = options.density_spacing_over_h * h;
    phi_space_ = std::make_shared<NurbsDensitySpace2D>(
        geometry,
        3,
        target_spacing,
        std::move(options.phi_connections),
        options.density_coordinate);
    psi_space_ = std::make_shared<NurbsDensitySpace2D>(
        geometry,
        2,
        target_spacing,
        std::move(options.psi_connections),
        options.density_coordinate);
    zero_phi_ = Eigen::VectorXd::Zero(
        phi_space_->reduced_coefficient_count());
    zero_psi_ = Eigen::VectorXd::Zero(
        psi_space_->reduced_coefficient_count());

    const NurbsDensitySpace2D& trial = unknown_space();
    const int density_count = trial.reduced_coefficient_count();
    build_gauss_trace_references(trace_gauss_order_);
    if (type_ == LaplaceNscEtBvpType2D::InteriorNeumann) {
        const Eigen::VectorXd mass = make_mass_row(trial);
        if (use_full_neumann_bordered_system_) {
            const Eigen::VectorXd constant = trial.fit_function(
                [](int, double, const Eigen::Vector2d&) { return 1.0; });
            const double constant_mass = mass.dot(constant);
            if (!(std::abs(constant_mass) > 1.0e-14 * mass.norm())
                || !std::isfinite(constant_mass)) {
                throw std::runtime_error(
                    "NSC-ET-KFBI Neumann rank-one gauge has an invalid constant mode");
            }
            gauge_reduction_ = Eigen::MatrixXd::Identity(
                density_count, density_count);
            neumann_mass_functional_ = mass / constant_mass;
            neumann_border_mass_functional_ = mass / mass.norm();
            neumann_stabilization_direction_ =
                constant / constant.norm();
        } else if (options.use_orthogonal_neumann_gauge) {
            Eigen::VectorXd direction = mass / mass.norm();
            const double sign = direction[0] >= 0.0 ? 1.0 : -1.0;
            Eigen::VectorXd reflector = direction;
            reflector[0] += sign;
            const double reflector_norm = reflector.norm();
            if (!(reflector_norm > 1.0e-14)) {
                throw std::runtime_error(
                    "NSC-ET-KFBI cannot construct the orthogonal Neumann gauge");
            }
            reflector /= reflector_norm;
            gauge_reduction_ = Eigen::MatrixXd::Zero(
                density_count, density_count - 1);
            gauge_reduction_.bottomRows(density_count - 1).setIdentity();
            gauge_reduction_.noalias() -= 2.0 * reflector
                * reflector.tail(density_count - 1).transpose();
            const double null_residual =
                (mass.transpose() * gauge_reduction_).norm()
                / mass.norm();
            const double orthogonal_residual =
                (gauge_reduction_.transpose() * gauge_reduction_
                 - Eigen::MatrixXd::Identity(
                       density_count - 1, density_count - 1))
                    .norm()
                / static_cast<double>(density_count - 1);
            if (!gauge_reduction_.allFinite()
                || null_residual > 5.0e-12
                || orthogonal_residual > 5.0e-12) {
                throw std::runtime_error(
                    "NSC-ET-KFBI orthogonal Neumann gauge failed certification");
            }
        } else {
            int pivot = 0;
            mass.cwiseAbs().maxCoeff(&pivot);
            if (!(std::abs(mass[pivot]) > 1.0e-14 * mass.norm())) {
                throw std::runtime_error(
                    "NSC-ET-KFBI cannot construct the Neumann mean-zero gauge");
            }
            gauge_reduction_ = Eigen::MatrixXd::Zero(
                density_count, density_count - 1);
            int column = 0;
            for (int raw = 0; raw < density_count; ++raw) {
                if (raw == pivot)
                    continue;
                gauge_reduction_(raw, column) = 1.0;
                gauge_reduction_(pivot, column) =
                    -mass[raw] / mass[pivot];
                ++column;
            }
        }
    } else {
        gauge_reduction_ = Eigen::MatrixXd::Identity(
            density_count, density_count);
    }
    if (gauge_reduction_.cols() < 1) {
        throw std::invalid_argument(
            "NSC-ET-KFBI density space is too small after gauge reduction");
    }
    restrict_.configure_trace_references(trace_reference_points_);
    build_trace_projection(options.collocation_rank_tolerance);
}

const NurbsDensitySpace2D& LaplaceNscEtBvp2D::unknown_space() const
{
    return type_ == LaplaceNscEtBvpType2D::InteriorNeumann
        ? *phi_space_ : *psi_space_;
}

int LaplaceNscEtBvp2D::density_unknown_count() const
{
    return unknown_space().reduced_coefficient_count();
}

int LaplaceNscEtBvp2D::raw_density_unknown_count() const
{
    return unknown_space().raw_coefficient_count();
}

int LaplaceNscEtBvp2D::problem_size() const
{
    return coefficient_solver_size()
        + (type_ == LaplaceNscEtBvpType2D::InteriorNeumann
               && use_full_neumann_bordered_system_
           ? 1 : 0);
}

int LaplaceNscEtBvp2D::coefficient_solver_size() const
{
    return static_cast<int>(gauge_reduction_.cols());
}

Eigen::VectorXd LaplaceNscEtBvp2D::expand_solver_coordinates(
    const Eigen::VectorXd& solver_coordinates) const
{
    if (solver_coordinates.size() != problem_size()
        || !solver_coordinates.allFinite()) {
        throw std::invalid_argument(
            "NSC-ET-KFBI solver-coordinate vector has an invalid size or value");
    }
    return gauge_reduction_
        * solver_coordinates.head(coefficient_solver_size());
}

Eigen::VectorXd LaplaceNscEtBvp2D::density_values_at_interface(
    const NurbsDensitySpace2D& space,
    const Eigen::VectorXd& coefficients) const
{
    const auto& geometry = require_nurbs_geometry(grid_pair_.interface());
    Eigen::VectorXd values(grid_pair_.interface().num_points());
    for (int q = 0; q < values.size(); ++q) {
        values[q] = space.evaluate(
            geometry.point_span(q),
            geometry.point_parameter(q),
            coefficients).value;
    }
    return values;
}

Eigen::VectorXd LaplaceNscEtBvp2D::density_values_at_trace(
    const NurbsDensitySpace2D& space,
    const Eigen::VectorXd& coefficients) const
{
    Eigen::VectorXd values(trace_reference_points_.size());
    for (int row = 0; row < values.size(); ++row) {
        const LaplaceNurbsTraceReferencePoint2D& reference =
            trace_reference_points_[static_cast<std::size_t>(row)];
        values[row] = space.evaluate(
            reference.branch,
            reference.parameter,
            coefficients).value;
    }
    return values;
}

LaplaceNscEtBvp2D::FieldResult LaplaceNscEtBvp2D::evaluate_field(
    const Eigen::VectorXd& phi_coefficients,
    const Eigen::VectorXd& psi_coefficients,
    LaplaceCrossingTraceStencil2D trace_stencil,
    bool recover_physical_trace) const
{
    const Eigen::VectorXd phi_values = density_values_at_interface(
        *phi_space_, phi_coefficients);
    const Eigen::VectorXd psi_values = density_values_at_interface(
        *psi_space_, psi_coefficients);
    const auto jumps = make_harmonic_jumps(phi_values, psi_values);
    auto state = std::make_shared<LaplaceNurbsDensityTraceState2D>(
        crossing_trace_uses_phi_p3(trace_stencil) ? phi_space_ : nullptr,
        crossing_trace_uses_phi_p3(trace_stencil)
            ? phi_coefficients : Eigen::VectorXd(),
        crossing_trace_uses_psi_p2(trace_stencil) ? psi_space_ : nullptr,
        crossing_trace_uses_psi_p2(trace_stencil)
            ? psi_coefficients : Eigen::VectorXd(),
        trace_stencil,
        density_derivative_scheme_,
        density_difference_step_over_span_);

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(
        grid_pair_.grid().num_dofs());
    const LaplaceSpreadResult2D spread_result =
        spread_.apply_with_direct_nurbs_density_state(
            jumps, rhs, std::move(state));
    Eigen::VectorXd u_bulk;
    bulk_solver_.solve(-rhs, u_bulk);
    const LaplaceRestrictedReferenceTrace2D exterior =
        restrict_.apply_exterior_virtual_at_references(
            u_bulk, spread_result);

    FieldResult result;
    result.u_bulk = std::move(u_bulk);
    result.exterior_value = exterior.value;
    result.exterior_normal = exterior.normal_derivative;
    if (recover_physical_trace) {
        result.interior_value = exterior.value
            + density_values_at_trace(*phi_space_, phi_coefficients);
        result.interior_normal = exterior.normal_derivative
            + density_values_at_trace(*psi_space_, psi_coefficients);
    }
    return result;
}

Eigen::VectorXd LaplaceNscEtBvp2D::exterior_equation_trace(
    const FieldResult& field) const
{
    return type_ == LaplaceNscEtBvpType2D::InteriorNeumann
        ? field.exterior_value : field.exterior_normal;
}

Eigen::VectorXd LaplaceNscEtBvp2D::project_exterior_trace(
    const Eigen::VectorXd& trace) const
{
    if (trace.size() != trace_projection_matrix_.cols()
        || !trace.allFinite()) {
        throw std::invalid_argument(
            "NSC-ET-KFBI Gauss exterior trace has an invalid size or value");
    }
    return trace_projection_matrix_ * trace;
}

void LaplaceNscEtBvp2D::apply(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& y) const
{
    const Eigen::VectorXd coefficients = expand_solver_coordinates(x);
    const FieldResult field =
        type_ == LaplaceNscEtBvpType2D::InteriorNeumann
        ? evaluate_field(coefficients,
                         zero_psi_,
                         LaplaceCrossingTraceStencil2D::PhiP3)
        : evaluate_field(zero_phi_,
                         coefficients,
                         LaplaceCrossingTraceStencil2D::PsiP2);
    const Eigen::VectorXd coefficient_trace =
        project_exterior_trace(exterior_equation_trace(field));
    if (type_ == LaplaceNscEtBvpType2D::InteriorNeumann
        && use_full_neumann_bordered_system_) {
        y.resize(problem_size());
        y.head(coefficient_solver_size()) = coefficient_trace
            + neumann_stabilization_direction_
                  * x[coefficient_solver_size()];
        y[coefficient_solver_size()] =
            neumann_border_mass_functional_.dot(coefficients);
    } else {
        y = coefficient_trace;
    }
}

Eigen::VectorXd LaplaceNscEtBvp2D::make_mass_row(
    const NurbsDensitySpace2D& space) const
{
    Eigen::VectorXd mass = Eigen::VectorXd::Zero(
        space.reduced_coefficient_count());
    for (const LaplaceNurbsTraceReferencePoint2D& reference
         : trace_reference_points_) {
        mass += reference.physical_weight
            * space.evaluation_rows(
                  reference.branch, reference.parameter)
                  .value.transpose();
    }
    if (!mass.allFinite() || !(mass.norm() > 0.0)) {
        throw std::runtime_error(
            "NSC-ET-KFBI density mass functional is invalid");
    }
    return mass;
}

void LaplaceNscEtBvp2D::build_gauss_trace_references(int gauss_order)
{
    const GaussLegendreRule rule = gauss_legendre_rule(gauss_order);
    const NurbsDensitySpace2D& space = unknown_space();
    const auto& geometry = require_nurbs_geometry(grid_pair_.interface());
    trace_reference_points_.clear();
    int global_span = 0;
    for (int branch = 0; branch < space.branch_count(); ++branch) {
        const NurbsDensityBranchInfo2D& info = space.branch_info(branch);
        if (static_cast<int>(info.parameter_breaks.size())
                != info.spline_span_count + 1
            || info.spline_span_count < 1) {
            throw std::runtime_error(
                "NSC-ET-KFBI density span metadata is inconsistent");
        }
        for (int span = 0; span < info.spline_span_count;
             ++span, ++global_span) {
            const double left =
                info.parameter_breaks[static_cast<std::size_t>(span)];
            const double right =
                info.parameter_breaks[static_cast<std::size_t>(span + 1)];
            if (!(right > left) || !std::isfinite(right - left)) {
                throw std::runtime_error(
                    "NSC-ET-KFBI density span has invalid parameter length");
            }
            // A density span may cross ordinary NURBS knots.  Split there so
            // the metric J(xi) is smooth on every Gauss integration cell.
            std::vector<double> integration_breaks{left, right};
            for (double knot : geometry.curve().basis().knots()) {
                if (knot > left && knot < right)
                    integration_breaks.push_back(knot);
            }
            std::sort(integration_breaks.begin(), integration_breaks.end());
            integration_breaks.erase(
                std::unique(
                    integration_breaks.begin(), integration_breaks.end()),
                integration_breaks.end());
            int density_span_node = 0;
            for (std::size_t cell = 1;
                 cell < integration_breaks.size(); ++cell) {
                const double cell_left = integration_breaks[cell - 1];
                const double cell_right = integration_breaks[cell];
                const double half = 0.5 * (cell_right - cell_left);
                const double midpoint = 0.5 * (cell_right + cell_left);
                for (int node = 0; node < gauss_order;
                     ++node, ++density_span_node) {
                    const double parameter = midpoint
                        + half
                              * rule.nodes[static_cast<std::size_t>(node)];
                    const geometry2d::NurbsBoundaryGeometry2D exact =
                        geometry.evaluate_on_span(branch, parameter);
                    LaplaceNurbsTraceReferencePoint2D reference;
                    reference.branch = branch;
                    reference.density_span = global_span;
                    reference.gauss_node = density_span_node;
                    reference.parameter = parameter;
                    reference.point = exact.point;
                    reference.normal = exact.normal;
                    reference.physical_weight = half
                        * rule.weights[static_cast<std::size_t>(node)]
                        * exact.speed;
                    if (!(reference.physical_weight > 0.0)
                        || !std::isfinite(reference.physical_weight)) {
                        throw std::runtime_error(
                            "NSC-ET-KFBI Gauss trace has invalid physical weight");
                    }
                    trace_reference_points_.push_back(
                        std::move(reference));
                }
            }
        }
    }
    if (trace_reference_points_.empty()) {
        throw std::runtime_error(
            "NSC-ET-KFBI constructed no Gauss trace references");
    }
    trace_quadrature_weights_.resize(
        static_cast<int>(trace_reference_points_.size()));
    for (int row = 0; row < trace_quadrature_weights_.size(); ++row) {
        trace_quadrature_weights_[row] =
            trace_reference_points_[static_cast<std::size_t>(row)]
                .physical_weight;
    }
    trace_points_.clear();
}

void LaplaceNscEtBvp2D::build_trace_projection(double rank_tolerance)
{
    if (!(rank_tolerance > 0.0) || !std::isfinite(rank_tolerance)) {
        throw std::invalid_argument(
            "NSC-ET-KFBI projection rank tolerance is invalid");
    }
    if (trace_projection_build_count_ != 0) {
        throw std::logic_error(
            "NSC-ET-KFBI trace projection may only be built once");
    }
    const int rows = static_cast<int>(trace_reference_points_.size());
    const int columns = coefficient_solver_size();
    if (rows < columns) {
        throw std::runtime_error(
            "NSC-ET-KFBI has fewer Gauss traces than density unknowns");
    }
    density_collocation_matrix_.resize(rows, columns);
    for (int row = 0; row < rows; ++row) {
        const LaplaceNurbsTraceReferencePoint2D& reference =
            trace_reference_points_[static_cast<std::size_t>(row)];
        density_collocation_matrix_.row(row) =
            unknown_space().evaluation_rows(
                reference.branch, reference.parameter).value
            * gauge_reduction_;
    }
    Eigen::MatrixXd weighted_design = density_collocation_matrix_;
    for (int row = 0; row < rows; ++row) {
        weighted_design.row(row) *=
            std::sqrt(trace_quadrature_weights_[row]);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        weighted_design, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(rank_tolerance);
    if (svd.rank() != columns) {
        throw std::runtime_error(
            "NSC-ET-KFBI weighted Gauss trace design is rank deficient");
    }
    const Eigen::VectorXd singular = svd.singularValues();
    trace_projection_condition_ = singular[0] / singular[columns - 1];
    Eigen::VectorXd inverse_singular = singular.head(columns);
    for (int index = 0; index < columns; ++index)
        inverse_singular[index] = 1.0 / inverse_singular[index];
    trace_projection_matrix_ =
        svd.matrixV().leftCols(columns)
        * inverse_singular.asDiagonal()
        * svd.matrixU().leftCols(columns).transpose();
    for (int row = 0; row < rows; ++row) {
        trace_projection_matrix_.col(row) *=
            std::sqrt(trace_quadrature_weights_[row]);
    }
    const Eigen::MatrixXd identity_error =
        trace_projection_matrix_ * density_collocation_matrix_
        - Eigen::MatrixXd::Identity(columns, columns);
    trace_projection_identity_residual_ =
        identity_error.norm() / std::sqrt(static_cast<double>(columns));
    const double max_identity_error = identity_error.cwiseAbs().maxCoeff();
    if (!trace_projection_matrix_.allFinite()
        || !std::isfinite(trace_projection_condition_)
        || trace_projection_identity_residual_ > 1.0e-9
        || max_identity_error > 5.0e-9) {
        throw std::runtime_error(
            "NSC-ET-KFBI cached Gauss-to-coefficient projection failed certification");
    }
    density_collocation_right_preconditioner_ =
        Eigen::MatrixXd::Identity(columns, columns);
    ++trace_projection_build_count_;
}

void LaplaceNscEtBvp2D::build_trace_collocation(
    const std::vector<int>& requested_candidates,
    double rank_tolerance)
{
    if (!(rank_tolerance > 0.0) || !std::isfinite(rank_tolerance)) {
        throw std::invalid_argument(
            "NSC-ET-KFBI collocation rank tolerance is invalid");
    }
    const Interface2D& iface = grid_pair_.interface();
    const auto& geometry = require_nurbs_geometry(iface);
    const bool use_default_candidates = requested_candidates.empty();
    std::vector<int> candidates = requested_candidates;
    if (candidates.empty()) {
        for (int q = 0; q < iface.num_points(); ++q) {
            if (!iface.is_corner_point(q))
                candidates.push_back(q);
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    for (int q : candidates) {
        if (q < 0 || q >= iface.num_points() || iface.is_corner_point(q)) {
            throw std::invalid_argument(
                "NSC-ET-KFBI trace candidates must be valid non-feature points");
        }
    }
    const int unknowns = problem_size();
    if (static_cast<int>(candidates.size()) < unknowns) {
        throw std::invalid_argument(
            "NSC-ET-KFBI has fewer exterior-trace candidates than trial unknowns");
    }

    // On one smooth closed NURBS branch, compatibility endpoints and
    // midpoints form an equal-arclength half-panel grid.  Uniformly subsample
    // that grid to the required square row count.  In particular, the
    // Neumann mean-zero reduction needs one fewer row than density spans;
    // sampling the denser half-panel grid distributes that deficit around the
    // whole curve instead of deleting one midpoint and creating a double gap.
    // A pure rank-pivoted choice is algebraically valid but can cluster rows
    // and leave a large unsampled arc, permitting high-frequency exterior
    // trace modes on refined flower geometries.
    bool selected_uniform_trace_points = false;
    if (use_default_candidates && geometry.num_spans() == 1
        && unknown_space().coordinate()
               == NurbsDensityCoordinate2D::PhysicalArclength) {
        std::vector<int> uniform_candidates;
        uniform_candidates.reserve(candidates.size());
        for (int q : candidates) {
            if (geometry.point_span(q) == 0)
                uniform_candidates.push_back(q);
        }
        std::sort(uniform_candidates.begin(), uniform_candidates.end(),
                  [&](int lhs, int rhs) {
                      return geometry.point_parameter(lhs)
                           < geometry.point_parameter(rhs);
                  });
        uniform_candidates.erase(
            std::unique(uniform_candidates.begin(), uniform_candidates.end()),
            uniform_candidates.end());
        if (static_cast<int>(uniform_candidates.size()) >= unknowns) {
            std::vector<int> preferred(static_cast<std::size_t>(unknowns));
            const double ratio =
                static_cast<double>(uniform_candidates.size())
                               / static_cast<double>(unknowns);
            for (int row = 0; row < unknowns; ++row) {
                const int index = std::min(
                    static_cast<int>(uniform_candidates.size()) - 1,
                    static_cast<int>(std::floor(
                        (static_cast<double>(row) + 0.5) * ratio)));
                preferred[static_cast<std::size_t>(row)] =
                    uniform_candidates[static_cast<std::size_t>(index)];
            }
            Eigen::MatrixXd preferred_design(unknowns, unknowns);
            for (int row = 0; row < unknowns; ++row) {
                const int q = preferred[static_cast<std::size_t>(row)];
                preferred_design.row(row) =
                    unknown_space().evaluation_rows(
                        geometry.point_span(q),
                        geometry.point_parameter(q)).value
                    * gauge_reduction_;
            }
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> preferred_qr(
                preferred_design);
            preferred_qr.setThreshold(rank_tolerance);
            if (preferred_qr.rank() == unknowns) {
                trace_points_ = std::move(preferred);
                selected_uniform_trace_points = true;
            }
        }
    }

    if (!selected_uniform_trace_points) {
        Eigen::MatrixXd design(candidates.size(), unknowns);
        for (int row = 0; row < static_cast<int>(candidates.size()); ++row) {
            const int q = candidates[static_cast<std::size_t>(row)];
            design.row(row) = unknown_space().evaluation_rows(
                geometry.point_span(q), geometry.point_parameter(q)).value
                * gauge_reduction_;
        }
        // Select rows from an orthonormal basis of the physical trial
        // subspace.  Pivoting design.transpose() directly depends on the
        // coefficient coordinates (and, for Neumann, on the gauge basis),
        // so two bases spanning the same functions can otherwise select
        // different exterior equations.
        Eigen::JacobiSVD<Eigen::MatrixXd> subspace_svd(
            design, Eigen::ComputeThinU | Eigen::ComputeThinV);
        subspace_svd.setThreshold(rank_tolerance);
        if (subspace_svd.rank() < unknowns) {
            throw std::runtime_error(
                "NSC-ET-KFBI exterior-trace candidates do not resolve the reduced density space");
        }
        const Eigen::MatrixXd physical_subspace =
            subspace_svd.matrixU().leftCols(unknowns);
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> row_qr(
            physical_subspace.transpose());
        row_qr.setThreshold(rank_tolerance);
        if (row_qr.rank() < unknowns) {
            throw std::runtime_error(
                "NSC-ET-KFBI invariant trace selection is rank deficient");
        }
        trace_points_.resize(static_cast<std::size_t>(unknowns));
        const auto& permutation = row_qr.colsPermutation().indices();
        for (int row = 0; row < unknowns; ++row) {
            trace_points_[static_cast<std::size_t>(row)] =
                candidates[static_cast<std::size_t>(permutation[row])];
        }
    }
    std::sort(trace_points_.begin(), trace_points_.end(),
              [&](int lhs, int rhs) {
                  const int left_branch = geometry.point_span(lhs);
                  const int right_branch = geometry.point_span(rhs);
                  if (left_branch != right_branch)
                      return left_branch < right_branch;
                  return geometry.point_parameter(lhs)
                       < geometry.point_parameter(rhs);
              });

    density_collocation_matrix_.resize(unknowns, unknowns);
    for (int row = 0; row < unknowns; ++row) {
        const int q = trace_points_[static_cast<std::size_t>(row)];
        density_collocation_matrix_.row(row) =
            unknown_space().evaluation_rows(
                geometry.point_span(q), geometry.point_parameter(q)).value
            * gauge_reduction_;
    }
    Eigen::FullPivLU<Eigen::MatrixXd> collocation_lu(
        density_collocation_matrix_);
    collocation_lu.setThreshold(rank_tolerance);
    if (collocation_lu.rank() < unknowns) {
        throw std::runtime_error(
            "NSC-ET-KFBI selected density collocation matrix is rank deficient");
    }
    density_collocation_right_preconditioner_ = collocation_lu.solve(
        Eigen::MatrixXd::Identity(unknowns, unknowns));
    const double inverse_residual =
        (density_collocation_matrix_
             * density_collocation_right_preconditioner_
         - Eigen::MatrixXd::Identity(unknowns, unknowns))
            .norm()
        / static_cast<double>(unknowns);
    if (!density_collocation_right_preconditioner_.allFinite()
        || !(inverse_residual <= 1.0e-8)) {
        throw std::runtime_error(
            "NSC-ET-KFBI density collocation preconditioner is inaccurate");
    }
}

LaplaceNscEtSolveResult2D LaplaceNscEtBvp2D::solve_harmonic(
    const NurbsDensitySpace2D::Function& boundary_function,
    int max_iter,
    double tolerance,
    int restart) const
{
    if (!boundary_function)
        throw std::invalid_argument("NSC-ET-KFBI needs boundary data");
    Eigen::VectorXd fixed_phi = zero_phi_;
    Eigen::VectorXd fixed_psi = zero_psi_;
    if (type_ == LaplaceNscEtBvpType2D::InteriorDirichlet) {
        fixed_phi = phi_space_->fit_function(boundary_function);
    } else {
        fixed_psi = psi_space_->fit_function(boundary_function);
        const Eigen::VectorXd mass = make_mass_row(*psi_space_);
        const Eigen::VectorXd constant = psi_space_->fit_function(
            [](int, double, const Eigen::Vector2d&) { return 1.0; });
        const double denominator = mass.dot(constant);
        if (!(std::abs(denominator) > 0.0)) {
            throw std::runtime_error(
                "NSC-ET-KFBI fixed Neumann density has no constant mode");
        }
        fixed_psi -= (mass.dot(fixed_psi) / denominator) * constant;
    }

    const FieldResult fixed =
        type_ == LaplaceNscEtBvpType2D::InteriorDirichlet
        ? evaluate_field(fixed_phi,
                         zero_psi_,
                         LaplaceCrossingTraceStencil2D::PhiP3)
        : evaluate_field(zero_phi_,
                         fixed_psi,
                         LaplaceCrossingTraceStencil2D::PsiP2);
    const Eigen::VectorXd fixed_exterior = exterior_equation_trace(fixed);
    const Eigen::VectorXd coefficient_rhs =
        -project_exterior_trace(fixed_exterior);
    Eigen::VectorXd b;
    if (type_ == LaplaceNscEtBvpType2D::InteriorNeumann
        && use_full_neumann_bordered_system_) {
        b = Eigen::VectorXd::Zero(problem_size());
        b.head(coefficient_solver_size()) = coefficient_rhs;
    } else {
        b = coefficient_rhs;
    }

    GMRES gmres(max_iter, tolerance, restart);
    Eigen::VectorXd solver_coordinates =
        Eigen::VectorXd::Zero(problem_size());
    const int iterations = gmres.solve(*this, b, solver_coordinates);
    const Eigen::VectorXd unknown =
        expand_solver_coordinates(solver_coordinates);

    Eigen::VectorXd final_phi = fixed_phi;
    Eigen::VectorXd final_psi = fixed_psi;
    if (type_ == LaplaceNscEtBvpType2D::InteriorNeumann)
        final_phi = unknown;
    else
        final_psi = unknown;
    FieldResult final = evaluate_field(
        final_phi,
        final_psi,
        LaplaceCrossingTraceStencil2D::PhiP3PsiP2,
        true);
    const Eigen::VectorXd exterior = exterior_equation_trace(final);
    Eigen::VectorXd projected_exterior = project_exterior_trace(exterior);
    Eigen::VectorXd projected_equation_residual = projected_exterior;
    if (type_ == LaplaceNscEtBvpType2D::InteriorNeumann
        && use_full_neumann_bordered_system_) {
        projected_equation_residual.noalias() +=
            neumann_stabilization_direction_
            * solver_coordinates[coefficient_solver_size()];
    }

    const auto weighted_norm = [&](const Eigen::VectorXd& values) {
        return std::sqrt(
            (trace_quadrature_weights_.array()
             * values.array().square()).sum());
    };

    LaplaceNscEtSolveResult2D result;
    result.u_bulk = std::move(final.u_bulk);
    result.unknown_reduced_coefficients = unknown;
    result.unknown_raw_coefficients =
        unknown_space().expand_reduced_coefficients(unknown);
    result.unknown_at_trace_points =
        density_values_at_trace(unknown_space(), unknown);
    result.exterior_trace = exterior;
    result.projected_exterior_trace = projected_exterior;
    result.physical_trace = std::move(final.interior_value);
    result.physical_normal_trace = std::move(final.interior_normal);
    result.residuals = gmres.residuals();
    result.iterations = iterations;
    result.gmres_converged = gmres.converged();
    result.exterior_trace_relative = weighted_norm(exterior)
        / std::max(
              weighted_norm(fixed_exterior),
              std::numeric_limits<double>::min());
    result.exterior_trace_inf = inf_norm(exterior);
    result.projected_exterior_trace_relative =
        relative_norm(projected_exterior, coefficient_rhs);
    result.exterior_trace_weighted_mean =
        trace_quadrature_weights_.dot(exterior)
        / trace_quadrature_weights_.sum();
    result.density_mean =
        type_ == LaplaceNscEtBvpType2D::InteriorNeumann
            && use_full_neumann_bordered_system_
        ? neumann_mass_functional_.dot(unknown)
        : 0.0;
    if (type_ == LaplaceNscEtBvpType2D::InteriorNeumann
        && use_full_neumann_bordered_system_) {
        const double border_constraint_residual =
            neumann_border_mass_functional_.dot(unknown);
        result.bordered_residual_relative = std::sqrt(
            projected_equation_residual.squaredNorm()
            + border_constraint_residual * border_constraint_residual)
            / std::max(b.norm(), std::numeric_limits<double>::min());
    } else {
        result.bordered_residual_relative =
            result.projected_exterior_trace_relative;
    }
    result.neumann_gauge_multiplier =
        type_ == LaplaceNscEtBvpType2D::InteriorNeumann
            && use_full_neumann_bordered_system_
        ? solver_coordinates[coefficient_solver_size()]
        : 0.0;
    return result;
}

} // namespace kfbim
