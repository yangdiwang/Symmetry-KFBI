#pragma once

#include <cstddef>

namespace kfbim {

// Formal reconstruction policy used to turn interface density DOFs into the
// trace jet frozen at a geometric crossing.
//
// ArcLengthBSplineCrossingJet is the ALS-CJ scheme: every smooth closed
// component uses one global physical-arclength coordinate, a periodic cubic
// C2 B-spline for phi=[u], and a periodic quadratic C1 B-spline for
// psi=[u_n].  LocalArclengthLagrange retains the earlier nearest-DOF fit as a
// compatibility fallback for open, cornered, or non-smooth components.
enum class LaplaceCrossingJetScheme2D {
    LocalArclengthLagrange,
    ArcLengthBSplineCrossingJet,
    // NURBS Same-Parameter Cauchy Jet (NSP-CJ): exact NURBS geometry and
    // density B-splines share the curve parameter. Phi uses degree three,
    // psi degree two, and physical arclength derivatives are recovered by
    // the exact NURBS Jacobian at the crossing.
    NurbsSameParameterCrossingJet
};

inline constexpr const char* laplace_crossing_jet_scheme_name_2d(
    LaplaceCrossingJetScheme2D scheme)
{
    return scheme
            == LaplaceCrossingJetScheme2D::NurbsSameParameterCrossingJet
        ? "nurbs_same_parameter_crossing_jet"
        : scheme == LaplaceCrossingJetScheme2D::ArcLengthBSplineCrossingJet
            ? "arc_length_bspline_crossing_jet"
            : "local_arclength_lagrange";
}

// Active interface-trace blocks used to reconstruct a crossing-local
// quadratic spatial polynomial.  The choice is made by the BVP/layer
// formulation; it must not be inferred from a sampled jump happening to be
// numerically zero.
enum class LaplaceCrossingTraceStencil2D {
    RhsOnly,
    PhiP3,
    PsiP2,
    PhiP3PsiP2
};

inline constexpr bool is_valid_crossing_trace_stencil(
    LaplaceCrossingTraceStencil2D stencil)
{
    return stencil == LaplaceCrossingTraceStencil2D::RhsOnly
        || stencil == LaplaceCrossingTraceStencil2D::PhiP3
        || stencil == LaplaceCrossingTraceStencil2D::PsiP2
        || stencil == LaplaceCrossingTraceStencil2D::PhiP3PsiP2;
}

inline constexpr bool crossing_trace_uses_phi_p3(
    LaplaceCrossingTraceStencil2D stencil)
{
    return stencil == LaplaceCrossingTraceStencil2D::PhiP3
        || stencil == LaplaceCrossingTraceStencil2D::PhiP3PsiP2;
}

inline constexpr bool crossing_trace_uses_psi_p2(
    LaplaceCrossingTraceStencil2D stencil)
{
    return stencil == LaplaceCrossingTraceStencil2D::PsiP2
        || stencil == LaplaceCrossingTraceStencil2D::PhiP3PsiP2;
}

inline constexpr std::size_t crossing_trace_required_dofs(
    LaplaceCrossingTraceStencil2D stencil)
{
    return crossing_trace_uses_phi_p3(stencil)
        ? std::size_t{4}
        : crossing_trace_uses_psi_p2(stencil) ? std::size_t{3}
                                              : std::size_t{0};
}

} // namespace kfbim
