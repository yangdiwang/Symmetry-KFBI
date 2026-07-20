#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>
#include <Eigen/Dense>
#include "../interface/interface_2d.hpp"
#include "../geometry/p2_curve_2d.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Panel-based collocation Cauchy solver for the 2D Laplace equation.
//
// Solves the local Cauchy problem at each interface Gauss point q:
//   −ΔC = f_int − f_ext   (in a small neighbourhood of the interface)
//   C    = a          (Dirichlet jump [u] = u_int − u_ext on the interface)
//   ∂_n C = b         (Neumann jump [∂_n u] = ∂_n u_int − ∂_n u_ext on Γ)
//
// For each Gauss point (center), a 6×6 collocation system is assembled
// using a local quadratic interpolation X(s) of the containing panel:
//   Rows 0–2  Dirichlet : X(t−δ), X(t), X(t+δ)       → value [u]
//   Rows 3–4  Neumann   : X(t−δ), X(t+δ)             → normal-deriv [∂_n u]·h
//   Row  5    PDE       : X(t)                       → (−Δ+κ)C · h²
//
// Here t is the panel-local coordinate of the target Gauss point, δ=0.05, and
// all jump data at off-node collocation locations is interpolated from the
// panel's 3 Gauss nodes.
//
// Monomial basis (Taylor-scaled):
//   φ₀=1,  φ₁=dx,  φ₂=dy,  φ₃=½dx²,  φ₄=½dy²,  φ₅=dx·dy
// where dx=(x−cx)/h, dy=(y−cy)/h, and h is the local collocation radius.
//
// After QR solve and rescaling (c[1:2]/=h, c[3:5]/=h²), the output is:
//   c[0]=C,  c[1]=Cx,  c[2]=Cy,  c[3]=Cxx,  c[4]=Cyy,  c[5]=Cxy
// all evaluated at the center point.
//
// Reference: solve2DCauchyProblem2 in third_party/old-codes/CauchyProblemSolvers.h
// ---------------------------------------------------------------------------

namespace detail {

inline constexpr double kPanelGLS[3] = {
    -0.7745966692414834,
     0.0,
     0.7745966692414834
};

inline constexpr double kPanelP2S[3] = {
    -1.0,
     0.0,
     1.0
};

inline constexpr int kQuadraticPanelExpansionCount = 4;
inline constexpr double kQuadraticPanelExpansionS[kQuadraticPanelExpansionCount] = {
    -0.75,
    -0.25,
     0.25,
     0.75
};

inline constexpr double kPanelLobattoS[3] = {-1.0, 0.0, 1.0};
inline constexpr int kLobattoExpansionCount = kQuadraticPanelExpansionCount;
inline constexpr double kLobattoExpansionS[kLobattoExpansionCount] = {
    -0.75,
    -0.25,
     0.25,
     0.75
};

inline constexpr double kCollocationDelta = 0.05;

inline bool profile_cauchy_detail_2d()
{
    return std::getenv("KFBIM_PROFILE_CAUCHY_DETAIL_2D") != nullptr;
}

inline double cauchy_elapsed_seconds_2d(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

inline void lagrange_basis_3_at_nodes(const double nodes[3], double s, double L[3]) {
    const double s0 = nodes[0];
    const double s1 = nodes[1];
    const double s2 = nodes[2];
    L[0] = ((s - s1) * (s - s2)) / ((s0 - s1) * (s0 - s2));
    L[1] = ((s - s0) * (s - s2)) / ((s1 - s0) * (s1 - s2));
    L[2] = ((s - s0) * (s - s1)) / ((s2 - s0) * (s2 - s1));
}

inline void lagrange_basis_deriv_3_at_nodes(const double nodes[3], double s, double dL[3]) {
    const double s0 = nodes[0];
    const double s1 = nodes[1];
    const double s2 = nodes[2];
    dL[0] = (2.0 * s - s1 - s2) / ((s0 - s1) * (s0 - s2));
    dL[1] = (2.0 * s - s0 - s2) / ((s1 - s0) * (s1 - s2));
    dL[2] = (2.0 * s - s0 - s1) / ((s2 - s0) * (s2 - s1));
}

inline void lagrange_basis_3(double s, double L[3]) {
    lagrange_basis_3_at_nodes(kPanelGLS, s, L);
}

inline void lagrange_basis_deriv_3(double s, double dL[3]) {
    lagrange_basis_deriv_3_at_nodes(kPanelGLS, s, dL);
}

inline double interpolate_scalar_3_at_nodes(const double nodes[3],
                                            const double values[3],
                                            double       s)
{
    double L[3];
    lagrange_basis_3_at_nodes(nodes, s, L);
    return L[0] * values[0] + L[1] * values[1] + L[2] * values[2];
}

inline void interpolate_vector_3_at_nodes(const double nodes[3],
                                          const double values[3][2],
                                          double       s,
                                          double       out[2])
{
    double L[3];
    lagrange_basis_3_at_nodes(nodes, s, L);
    out[0] = L[0] * values[0][0] + L[1] * values[1][0] + L[2] * values[2][0];
    out[1] = L[0] * values[0][1] + L[1] * values[1][1] + L[2] * values[2][1];
}

inline void interpolate_vector_deriv_3_at_nodes(const double nodes[3],
                                                const double values[3][2],
                                                double       s,
                                                double       out[2])
{
    double dL[3];
    lagrange_basis_deriv_3_at_nodes(nodes, s, dL);
    out[0] = dL[0] * values[0][0] + dL[1] * values[1][0] + dL[2] * values[2][0];
    out[1] = dL[0] * values[0][1] + dL[1] * values[1][1] + dL[2] * values[2][1];
}

inline double interpolate_scalar_3(const double values[3], double s) {
    return interpolate_scalar_3_at_nodes(kPanelGLS, values, s);
}

inline void interpolate_vector_3(const double values[3][2], double s, double out[2]) {
    interpolate_vector_3_at_nodes(kPanelGLS, values, s, out);
}

inline void interpolate_vector_deriv_3(const double values[3][2], double s, double out[2]) {
    interpolate_vector_deriv_3_at_nodes(kPanelGLS, values, s, out);
}

inline void panel_normal_2d_at_nodes(const double nodes[3],
                                     const double pts[3][2],
                                     const double stored_normals[3][2],
                                     double       s,
                                     double       out[2])
{
    double tangent[2];
    interpolate_vector_deriv_3_at_nodes(nodes, pts, s, tangent);
    const double tlen = std::hypot(tangent[0], tangent[1]);

    double ref[2];
    interpolate_vector_3_at_nodes(nodes, stored_normals, s, ref);
    const double ref_len = std::hypot(ref[0], ref[1]);

    if (tlen <= 1e-14) {
        if (ref_len > 1e-14) {
            out[0] = ref[0] / ref_len;
            out[1] = ref[1] / ref_len;
        } else {
            out[0] = 1.0;
            out[1] = 0.0;
        }
        return;
    }

    out[0] = tangent[1] / tlen;
    out[1] = -tangent[0] / tlen;

    if (ref_len > 1e-14 && out[0] * ref[0] + out[1] * ref[1] < 0.0) {
        out[0] = -out[0];
        out[1] = -out[1];
    }
}

inline void panel_normal_2d(const double pts[3][2],
                            const double stored_normals[3][2],
                            double       s,
                            double       out[2])
{
    panel_normal_2d_at_nodes(kPanelGLS, pts, stored_normals, s, out);
}

inline double panel_chord_length_2d(const double pts[3][2])
{
    return std::hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
         + std::hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1]);
}

} // namespace detail

// ---------------------------------------------------------------------------
// Per-node 6×6 solve  (internal helper, mirrors solve2DCauchyProblem2)
//
// bdry1[3][2]     : coords of 3 Dirichlet collocation pts
// a[3]            : [u] at bdry1 pts
// bdry2[2][2]     : coords of 2 Neumann collocation pts
// bdry2_nml[2][2] : outward normals at bdry2 pts
// b[2]            : [∂_n u] at bdry2 pts
// bulk[2]         : coord of 1 interior PDE collocation pt
// Lu              : (f_int − f_ext) at bulk pt
// center[2]       : expansion origin (the target interface point)
// kappa           : coefficient of u in PDE (0 for pure Laplace)
// h               : scaling parameter (local collocation radius)
// c[6]            : output Taylor coefficients at center (see header doc)
// ---------------------------------------------------------------------------
inline void solve_local_6x6_2d(
    const double bdry1[3][2], const double a[3],
    const double bdry2[2][2], const double bdry2_nml[2][2], const double b[2],
    const double bulk[2], double Lu,
    const double center[2], double kappa, double h,
    double c[6])
{
    const double h2      = h * h;
    const double kap_h2  = kappa * h2;

    Eigen::Matrix<double, 6, 6> A;
    Eigen::Matrix<double, 6, 1> rhs;
    A.setZero();

    int m = 0;

    // ── Rows 0–2: Dirichlet  [u] = a ─────────────────────────────────────
    for (int l = 0; l < 3; ++l) {
        const double dx = (bdry1[l][0] - center[0]) / h;
        const double dy = (bdry1[l][1] - center[1]) / h;
        A(m,0) = 1.0;
        A(m,1) = dx;          A(m,2) = dy;
        A(m,3) = 0.5*dx*dx;   A(m,4) = 0.5*dy*dy;  A(m,5) = dx*dy;
        rhs[m] = a[l];
        ++m;
    }

    // ── Rows 3–4: Neumann  [∂_n u] = b  (scaled by h) ───────────────────
    for (int l = 0; l < 2; ++l) {
        const double dx = (bdry2[l][0] - center[0]) / h;
        const double dy = (bdry2[l][1] - center[1]) / h;
        const double nx = bdry2_nml[l][0];
        const double ny = bdry2_nml[l][1];
        A(m,0) = 0.0;
        A(m,1) = nx;               A(m,2) = ny;
        A(m,3) = dx*nx;            A(m,4) = dy*ny;
        A(m,5) = nx*dy + dx*ny;
        rhs[m] = b[l] * h;
        ++m;
    }

    // ── Row 5: PDE  (−Δ + κ)C = Lu  (scaled by h²) ──────────────────────
    {
        const double dx = (bulk[0] - center[0]) / h;
        const double dy = (bulk[1] - center[1]) / h;
        A(m,0) = kap_h2;
        A(m,1) = kap_h2*dx;           A(m,2) = kap_h2*dy;
        A(m,3) = kap_h2*0.5*dx*dx - 1.0;
        A(m,4) = kap_h2*0.5*dy*dy - 1.0;
        A(m,5) = kap_h2*dx*dy;
        rhs[m] = Lu * h2;
        ++m;
    }

    // ── Column-norm preconditioning (improves QR conditioning) ───────────
    Eigen::Matrix<double,6,1> col_scale;
    for (int k = 0; k < 6; ++k) {
        col_scale[k] = A.col(k).norm();
        if (col_scale[k] > 1e-14) A.col(k) /= col_scale[k];
        else                       col_scale[k] = 1.0;
    }

    // ── QR solve ─────────────────────────────────────────────────────────
    Eigen::Matrix<double,6,1> c_raw =
        A.colPivHouseholderQr().solve(rhs);

    for (int k = 0; k < 6; ++k)
        c_raw[k] /= col_scale[k];

    // ── Rescale: raw poly coefficients → physical derivatives at center ───
    // c[k] from QR: coefficient of φₖ(dx,dy)
    // After rescaling: c[k] = k-th partial derivative of C at center
    //   c[0]  = C
    //   c[1] /= h   → Cx
    //   c[2] /= h   → Cy
    //   c[3] /= h²  → Cxx   (basis was ½dx², d²/dx²= 1/h² factor)
    //   c[4] /= h²  → Cyy
    //   c[5] /= h²  → Cxy
    c[0] = c_raw[0];
    c[1] = c_raw[1] / h;
    c[2] = c_raw[2] / h;
    c[3] = c_raw[3] / h2;
    c[4] = c_raw[4] / h2;
    c[5] = c_raw[5] / h2;
}

// ---------------------------------------------------------------------------
// Output type: per-quadrature-point derivatives of C
// ---------------------------------------------------------------------------
struct PanelCauchyResult2D {
    Eigen::VectorXd C;    // C     at each Gauss pt
    Eigen::VectorXd Cx;   // ∂C/∂x
    Eigen::VectorXd Cy;   // ∂C/∂y
    Eigen::VectorXd Cxx;  // ∂²C/∂x²
    Eigen::VectorXd Cyy;  // ∂²C/∂y²
    Eigen::VectorXd Cxy;  // ∂²C/∂x∂y
};

// ---------------------------------------------------------------------------
// Output type for quadratic Lagrange P2 geometry panels.
//
// One panel has 3 quadratic geometry/jump nodes at s={-1,0,1}; the
// local Cauchy solve is performed at four generated expansion centers per
// panel, s={-0.75,-0.25,0.25,0.75}.
// ---------------------------------------------------------------------------
struct PanelCenterCauchyResult2D {
    Eigen::MatrixX2d centers;       // [center] physical expansion center
    Eigen::MatrixX2d normals;       // [center] panel normal at center
    Eigen::VectorXi  panel_index;   // [center] source panel index
    Eigen::VectorXd  local_s;       // [center] panel-local expansion parameter
    Eigen::VectorXd  C;
    Eigen::VectorXd  Cx;
    Eigen::VectorXd  Cy;
    Eigen::VectorXd  Cxx;
    Eigen::VectorXd  Cyy;
    Eigen::VectorXd  Cxy;
};

using PanelCenterJumpEvaluator2D = std::function<bool(
    int panel,
    double local_s,
    Eigen::Vector2d point,
    Eigen::Vector2d normal,
    double& value_jump,
    double& normal_jump,
    double& rhs_jump)>;

struct PanelCenterCauchyGeometryPoint2D {
    int panel = -1;
    double local_s = 0.0;
    std::array<Eigen::Vector2d, 3> collocation_points;
    std::array<Eigen::Vector2d, 3> collocation_normals;
    std::array<std::array<double, 3>, 3> interpolation_weights;
    Eigen::Matrix<double, 6, 6> rhs_to_coeff =
        Eigen::Matrix<double, 6, 6>::Zero();
};

struct PanelCenterCauchyGeometryCache2D {
    int num_panels = 0;
    int num_centers = 0;
    Eigen::MatrixX2d centers;
    Eigen::MatrixX2d normals;
    Eigen::VectorXi panel_index;
    Eigen::VectorXd local_s;
    std::vector<PanelCenterCauchyGeometryPoint2D> points;
};

inline Eigen::Matrix<double, 6, 6> local_6x6_rhs_to_coeff_map_2d(
    const double bdry1[3][2],
    const double bdry2[2][2],
    const double bdry2_nml[2][2],
    const double bulk[2],
    const double center[2],
    double kappa,
    double h)
{
    const double h2 = h * h;
    const double kap_h2 = kappa * h2;

    Eigen::Matrix<double, 6, 6> A;
    A.setZero();
    int m = 0;
    for (int l = 0; l < 3; ++l) {
        const double dx = (bdry1[l][0] - center[0]) / h;
        const double dy = (bdry1[l][1] - center[1]) / h;
        A(m, 0) = 1.0;
        A(m, 1) = dx;
        A(m, 2) = dy;
        A(m, 3) = 0.5 * dx * dx;
        A(m, 4) = 0.5 * dy * dy;
        A(m, 5) = dx * dy;
        ++m;
    }
    for (int l = 0; l < 2; ++l) {
        const double dx = (bdry2[l][0] - center[0]) / h;
        const double dy = (bdry2[l][1] - center[1]) / h;
        const double nx = bdry2_nml[l][0];
        const double ny = bdry2_nml[l][1];
        A(m, 0) = 0.0;
        A(m, 1) = nx;
        A(m, 2) = ny;
        A(m, 3) = dx * nx;
        A(m, 4) = dy * ny;
        A(m, 5) = nx * dy + dx * ny;
        ++m;
    }
    {
        const double dx = (bulk[0] - center[0]) / h;
        const double dy = (bulk[1] - center[1]) / h;
        A(m, 0) = kap_h2;
        A(m, 1) = kap_h2 * dx;
        A(m, 2) = kap_h2 * dy;
        A(m, 3) = kap_h2 * 0.5 * dx * dx - 1.0;
        A(m, 4) = kap_h2 * 0.5 * dy * dy - 1.0;
        A(m, 5) = kap_h2 * dx * dy;
    }

    Eigen::Matrix<double, 6, 1> col_scale;
    for (int k = 0; k < 6; ++k) {
        col_scale[k] = A.col(k).norm();
        if (col_scale[k] > 1e-14)
            A.col(k) /= col_scale[k];
        else
            col_scale[k] = 1.0;
    }

    const Eigen::Matrix<double, 6, 6> inv_scaled =
        A.colPivHouseholderQr().solve(
            Eigen::Matrix<double, 6, 6>::Identity());

    Eigen::Matrix<double, 6, 6> map = inv_scaled;
    for (int k = 0; k < 6; ++k)
        map.row(k) /= col_scale[k];
    map.row(1) /= h;
    map.row(2) /= h;
    map.row(3) /= h2;
    map.row(4) /= h2;
    map.row(5) /= h2;
    map.col(3) *= h;
    map.col(4) *= h;
    map.col(5) *= h2;
    return map;
}

inline PanelCenterCauchyGeometryCache2D
build_panel_center_cauchy_geometry_cache_2d(const Interface2D& iface,
                                            double kappa = 0.0)
{
    assert(iface.points_per_panel() == 3);
    const int Np = iface.num_panels();
    const int Nc = detail::kQuadraticPanelExpansionCount * Np;

    PanelCenterCauchyGeometryCache2D cache;
    cache.num_panels = Np;
    cache.num_centers = Nc;
    cache.centers = Eigen::MatrixX2d::Zero(Nc, 2);
    cache.normals = Eigen::MatrixX2d::Zero(Nc, 2);
    cache.panel_index = Eigen::VectorXi::Zero(Nc);
    cache.local_s = Eigen::VectorXd::Zero(Nc);
    cache.points.resize(static_cast<std::size_t>(Nc));

    for (int p = 0; p < Np; ++p) {
        const int g0 = iface.point_index(p, 0);
        const int g1 = iface.point_index(p, 1);
        const int g2 = iface.point_index(p, 2);
        const double panel_pts[3][2] = {
            {iface.points()(g0, 0), iface.points()(g0, 1)},
            {iface.points()(g1, 0), iface.points()(g1, 1)},
            {iface.points()(g2, 0), iface.points()(g2, 1)}
        };
        const double panel_nml[3][2] = {
            {iface.panel_normal(p, 0)[0], iface.panel_normal(p, 0)[1]},
            {iface.panel_normal(p, 1)[0], iface.panel_normal(p, 1)[1]},
            {iface.panel_normal(p, 2)[0], iface.panel_normal(p, 2)[1]}
        };
        const double panel_length = detail::panel_chord_length_2d(panel_pts);

        for (int i = 0; i < detail::kQuadraticPanelExpansionCount; ++i) {
            const int ci = detail::kQuadraticPanelExpansionCount * p + i;
            const double t = detail::kQuadraticPanelExpansionS[i];
            const double s_values[3] = {
                t - detail::kCollocationDelta,
                t,
                t + detail::kCollocationDelta
            };

            PanelCenterCauchyGeometryPoint2D& point =
                cache.points[static_cast<std::size_t>(ci)];
            point.panel = p;
            point.local_s = t;
            cache.panel_index[ci] = p;
            cache.local_s[ci] = t;

            double center[2];
            detail::interpolate_vector_3_at_nodes(
                detail::kPanelP2S, panel_pts, t, center);
            double center_normal[2];
            detail::panel_normal_2d_at_nodes(
                detail::kPanelP2S, panel_pts, panel_nml, t, center_normal);
            cache.centers(ci, 0) = center[0];
            cache.centers(ci, 1) = center[1];
            cache.normals(ci, 0) = center_normal[0];
            cache.normals(ci, 1) = center_normal[1];

            double bdry1[3][2];
            for (int j = 0; j < 3; ++j) {
                double weights[3];
                detail::lagrange_basis_3_at_nodes(
                    detail::kPanelP2S, s_values[j], weights);
                for (int k = 0; k < 3; ++k)
                    point.interpolation_weights[j][k] = weights[k];
                detail::interpolate_vector_3_at_nodes(
                    detail::kPanelP2S,
                    panel_pts,
                    s_values[j],
                    bdry1[j]);
                point.collocation_points[j] =
                    Eigen::Vector2d(bdry1[j][0], bdry1[j][1]);
                double nml[2];
                detail::panel_normal_2d_at_nodes(
                    detail::kPanelP2S,
                    panel_pts,
                    panel_nml,
                    s_values[j],
                    nml);
                point.collocation_normals[j] =
                    Eigen::Vector2d(nml[0], nml[1]);
            }

            double bdry2[2][2] = {
                {bdry1[0][0], bdry1[0][1]},
                {bdry1[2][0], bdry1[2][1]}
            };
            double bdry2_nml[2][2] = {
                {point.collocation_normals[0][0],
                 point.collocation_normals[0][1]},
                {point.collocation_normals[2][0],
                 point.collocation_normals[2][1]}
            };
            double bulk[2] = {center[0], center[1]};
            double h = 0.0;
            for (int l = 0; l < 3; ++l) {
                h = std::max(h, std::hypot(bdry1[l][0] - center[0],
                                           bdry1[l][1] - center[1]));
            }
            if (h <= 1e-14)
                h = panel_length;
            point.rhs_to_coeff =
                local_6x6_rhs_to_coeff_map_2d(bdry1,
                                               bdry2,
                                               bdry2_nml,
                                               bulk,
                                               center,
                                               kappa,
                                               h);
        }
    }

    return cache;
}

// ---------------------------------------------------------------------------
// laplace_panel_cauchy_2d
//
// Solves the local Cauchy problem at every Gauss point of the interface Γ.
// Requires iface.points_per_panel() == 3.
//
// Inputs:
//   iface      — Interface2D with 3 Gauss points per panel
//   a          — [u] = u_int − u_ext   at each Gauss point (length Nq)
//   b          — [∂_n u] = ∂_n u_int − ∂_n u_ext   at each Gauss point
//   Lu_iface   — f_int − f_ext  at each Gauss point (length Nq);
//                interpolated to each target's PDE collocation point
//   kappa      — coefficient of u in the PDE (0 for pure Laplace)
//
// Conventions: u_int is the interior smooth branch (label 1), u_ext is
//              the exterior smooth branch (label 0).
//
// Returns PanelCauchyResult2D with C, Cx, Cy, Cxx, Cyy, Cxy at every node.
// ---------------------------------------------------------------------------
inline PanelCauchyResult2D laplace_panel_cauchy_2d(
    const Interface2D&     iface,
    const Eigen::VectorXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& Lu_iface,
    double                 kappa = 0.0)
{
    assert(iface.points_per_panel() == 3);
    const int Nq = iface.num_points();
    const int Np = iface.num_panels();

    PanelCauchyResult2D res;
    res.C   = Eigen::VectorXd::Zero(Nq);
    res.Cx  = Eigen::VectorXd::Zero(Nq);
    res.Cy  = Eigen::VectorXd::Zero(Nq);
    res.Cxx = Eigen::VectorXd::Zero(Nq);
    res.Cyy = Eigen::VectorXd::Zero(Nq);
    res.Cxy = Eigen::VectorXd::Zero(Nq);

    for (int p = 0; p < Np; ++p) {

        // Global indices of the 3 Gauss points on this panel
        const int g0 = iface.point_index(p, 0);
        const int g1 = iface.point_index(p, 1);
        const int g2 = iface.point_index(p, 2);

        const double panel_pts[3][2] = {
            { iface.points()(g0, 0), iface.points()(g0, 1) },
            { iface.points()(g1, 0), iface.points()(g1, 1) },
            { iface.points()(g2, 0), iface.points()(g2, 1) }
        };
        const double panel_nml[3][2] = {
            { iface.panel_normal(p, 0)[0], iface.panel_normal(p, 0)[1] },
            { iface.panel_normal(p, 1)[0], iface.panel_normal(p, 1)[1] },
            { iface.panel_normal(p, 2)[0], iface.panel_normal(p, 2)[1] }
        };
        const double panel_a[3]  = { a[g0], a[g1], a[g2] };
        const double panel_b[3]  = { b[g0], b[g1], b[g2] };
        const double panel_Lu[3] = { Lu_iface[g0], Lu_iface[g1], Lu_iface[g2] };
        const double panel_length = detail::panel_chord_length_2d(panel_pts);

        // Solve for each of the 3 Gauss points as the local center
        const int gidx[3] = { g0, g1, g2 };
        for (int i = 0; i < 3; ++i) {
            const int gi = gidx[i];
            const double t = detail::kPanelGLS[i];
            const double s_minus = t - detail::kCollocationDelta;
            const double s_plus  = t + detail::kCollocationDelta;

            double center[2];
            detail::interpolate_vector_3(panel_pts, t, center);

            double bdry1[3][2];
            detail::interpolate_vector_3(panel_pts, s_minus, bdry1[0]);
            detail::interpolate_vector_3(panel_pts, t,       bdry1[1]);
            detail::interpolate_vector_3(panel_pts, s_plus,  bdry1[2]);
            double av[3] = {
                detail::interpolate_scalar_3(panel_a, s_minus),
                detail::interpolate_scalar_3(panel_a, t),
                detail::interpolate_scalar_3(panel_a, s_plus)
            };

            double bdry2[2][2];
            detail::interpolate_vector_3(panel_pts, s_minus, bdry2[0]);
            detail::interpolate_vector_3(panel_pts, s_plus,  bdry2[1]);
            double bdry2_nml[2][2];
            detail::panel_normal_2d(panel_pts, panel_nml, s_minus, bdry2_nml[0]);
            detail::panel_normal_2d(panel_pts, panel_nml, s_plus,  bdry2_nml[1]);
            double bv[2] = {
                detail::interpolate_scalar_3(panel_b, s_minus),
                detail::interpolate_scalar_3(panel_b, s_plus)
            };

            double bulk[2];
            detail::interpolate_vector_3(panel_pts, t, bulk);
            const double Lu = detail::interpolate_scalar_3(panel_Lu, t);

            double h = 0.0;
            for (int l = 0; l < 3; ++l) {
                h = std::max(h, std::hypot(bdry1[l][0] - center[0],
                                           bdry1[l][1] - center[1]));
            }
            if (h <= 1e-14)
                h = panel_length;

            double c[6];
            solve_local_6x6_2d(bdry1, av,
                                bdry2, bdry2_nml, bv,
                                bulk, Lu,
                                center, kappa, h,
                                c);

            res.C  [gi] = c[0];
            res.Cx [gi] = c[1];
            res.Cy [gi] = c[2];
            res.Cxx[gi] = c[3];
            res.Cyy[gi] = c[4];
            res.Cxy[gi] = c[5];
        }
    }

    return res;
}

// ---------------------------------------------------------------------------
// Quadratic Lagrange P2 panel-center solve.
//
// Requires iface.points_per_panel() == 3, interpreted as panel geometry and
// jump DOFs at P2 nodes s={-1,0,1}. Jump data and geometry are
// interpolated to generated expansion centers and shifted collocation points.
// The spread/restrict pipeline stores correction polynomials at these generated
// centers and evaluates grid-node corrections from the nearest center.
// ---------------------------------------------------------------------------
inline PanelCenterCauchyResult2D laplace_panel_quadratic_center_cauchy_2d(
    const Interface2D&     iface,
    const Eigen::VectorXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& Lu_iface,
    double                 kappa = 0.0,
    PanelCenterJumpEvaluator2D jump_evaluator = {},
    const std::vector<int>* selected_center_indices = nullptr,
    const PanelCenterCauchyGeometryCache2D* geometry_cache = nullptr)
{
    assert(iface.points_per_panel() == 3);
    const bool profile_detail = detail::profile_cauchy_detail_2d();
    const auto profile_total_start = std::chrono::steady_clock::now();
    double selected_build_sec = 0.0;
    double result_alloc_sec = 0.0;
    double panel_setup_sec = 0.0;
    double center_geom_sec = 0.0;
    double jump_eval_sec = 0.0;
    double solve6x6_sec = 0.0;
    double writeback_sec = 0.0;
    int considered_centers = 0;
    int skipped_centers = 0;
    int solved_centers = 0;

    const int Np = iface.num_panels();
    const int Nc = detail::kQuadraticPanelExpansionCount * Np;
    std::vector<char> selected;
    const auto selected_start = std::chrono::steady_clock::now();
    if (selected_center_indices != nullptr) {
        selected.assign(static_cast<std::size_t>(Nc), 0);
        for (int center_idx : *selected_center_indices) {
            if (center_idx >= 0 && center_idx < Nc)
                selected[static_cast<std::size_t>(center_idx)] = 1;
        }
    }
    if (profile_detail) {
        selected_build_sec = detail::cauchy_elapsed_seconds_2d(
            selected_start, std::chrono::steady_clock::now());
    }

    const auto result_alloc_start = std::chrono::steady_clock::now();
    PanelCenterCauchyResult2D res;
    res.centers     = Eigen::MatrixX2d::Zero(Nc, 2);
    res.normals     = Eigen::MatrixX2d::Zero(Nc, 2);
    res.panel_index = Eigen::VectorXi::Zero(Nc);
    res.local_s     = Eigen::VectorXd::Zero(Nc);
    res.C           = Eigen::VectorXd::Zero(Nc);
    res.Cx          = Eigen::VectorXd::Zero(Nc);
    res.Cy          = Eigen::VectorXd::Zero(Nc);
    res.Cxx         = Eigen::VectorXd::Zero(Nc);
    res.Cyy         = Eigen::VectorXd::Zero(Nc);
    res.Cxy         = Eigen::VectorXd::Zero(Nc);
    if (profile_detail) {
        result_alloc_sec = detail::cauchy_elapsed_seconds_2d(
            result_alloc_start, std::chrono::steady_clock::now());
    }

    if (geometry_cache != nullptr
        && geometry_cache->num_panels == Np
        && geometry_cache->num_centers == Nc
        && static_cast<int>(geometry_cache->points.size()) == Nc) {
        res.centers = geometry_cache->centers;
        res.normals = geometry_cache->normals;
        res.panel_index = geometry_cache->panel_index;
        res.local_s = geometry_cache->local_s;

        for (int p = 0; p < Np; ++p) {
            const auto panel_setup_start = std::chrono::steady_clock::now();
            const int g0 = iface.point_index(p, 0);
            const int g1 = iface.point_index(p, 1);
            const int g2 = iface.point_index(p, 2);
            const double panel_a[3]  = { a[g0], a[g1], a[g2] };
            const double panel_b[3]  = { b[g0], b[g1], b[g2] };
            const double panel_Lu[3] = {
                Lu_iface[g0], Lu_iface[g1], Lu_iface[g2]
            };
            if (profile_detail) {
                panel_setup_sec += detail::cauchy_elapsed_seconds_2d(
                    panel_setup_start, std::chrono::steady_clock::now());
            }

            for (int i = 0; i < detail::kQuadraticPanelExpansionCount; ++i) {
                const int ci = detail::kQuadraticPanelExpansionCount * p + i;
                ++considered_centers;
                if (!selected.empty()
                    && !selected[static_cast<std::size_t>(ci)]) {
                    ++skipped_centers;
                    continue;
                }
                ++solved_centers;
                const PanelCenterCauchyGeometryPoint2D& geom =
                    geometry_cache->points[static_cast<std::size_t>(ci)];

                auto fallback = [&](const std::array<double, 3>& weights,
                                    const double values[3]) {
                    return weights[0] * values[0]
                         + weights[1] * values[1]
                         + weights[2] * values[2];
                };
                auto evaluate_jump = [&](int local,
                                         double fallback_a,
                                         double fallback_b,
                                         double fallback_rhs,
                                         double& out_a,
                                         double& out_b,
                                         double& out_rhs) {
                    if (jump_evaluator) {
                        double value = 0.0;
                        double normal_value = 0.0;
                        double rhs_value = 0.0;
                        if (jump_evaluator(
                                p,
                                geom.local_s
                                    + (local == 0
                                           ? -detail::kCollocationDelta
                                           : (local == 2
                                                  ? detail::kCollocationDelta
                                                  : 0.0)),
                                geom.collocation_points[
                                    static_cast<std::size_t>(local)],
                                geom.collocation_normals[
                                    static_cast<std::size_t>(local)],
                                value,
                                normal_value,
                                rhs_value)) {
                            out_a = value;
                            out_b = normal_value;
                            out_rhs = rhs_value;
                            return;
                        }
                    }
                    out_a = fallback_a;
                    out_b = fallback_b;
                    out_rhs = fallback_rhs;
                };

                double av[3];
                double bv[2];
                double rhs_at_center = 0.0;
                double unused_b = 0.0;
                const auto jump_eval_start =
                    std::chrono::steady_clock::now();
                evaluate_jump(
                    0,
                    fallback(geom.interpolation_weights[0], panel_a),
                    fallback(geom.interpolation_weights[0], panel_b),
                    fallback(geom.interpolation_weights[0], panel_Lu),
                    av[0],
                    bv[0],
                    rhs_at_center);
                evaluate_jump(
                    1,
                    fallback(geom.interpolation_weights[1], panel_a),
                    fallback(geom.interpolation_weights[1], panel_b),
                    fallback(geom.interpolation_weights[1], panel_Lu),
                    av[1],
                    unused_b,
                    rhs_at_center);
                evaluate_jump(
                    2,
                    fallback(geom.interpolation_weights[2], panel_a),
                    fallback(geom.interpolation_weights[2], panel_b),
                    fallback(geom.interpolation_weights[2], panel_Lu),
                    av[2],
                    bv[1],
                    unused_b);
                if (profile_detail) {
                    jump_eval_sec += detail::cauchy_elapsed_seconds_2d(
                        jump_eval_start, std::chrono::steady_clock::now());
                }

                const auto solve_start = std::chrono::steady_clock::now();
                Eigen::Matrix<double, 6, 1> rhs;
                rhs << av[0], av[1], av[2], bv[0], bv[1], rhs_at_center;
                const Eigen::Matrix<double, 6, 1> c =
                    geom.rhs_to_coeff * rhs;
                if (profile_detail) {
                    solve6x6_sec += detail::cauchy_elapsed_seconds_2d(
                        solve_start, std::chrono::steady_clock::now());
                }

                const auto writeback_start =
                    std::chrono::steady_clock::now();
                res.C[ci] = c[0];
                res.Cx[ci] = c[1];
                res.Cy[ci] = c[2];
                res.Cxx[ci] = c[3];
                res.Cyy[ci] = c[4];
                res.Cxy[ci] = c[5];
                if (profile_detail) {
                    writeback_sec += detail::cauchy_elapsed_seconds_2d(
                        writeback_start, std::chrono::steady_clock::now());
                }
            }
        }

        if (profile_detail) {
            static int cauchy_detail_call = 0;
            ++cauchy_detail_call;
            const double total_sec = detail::cauchy_elapsed_seconds_2d(
                profile_total_start, std::chrono::steady_clock::now());
            std::printf("KFBIM_PROFILE_CAUCHY_DETAIL"
                        " call=%d panels=%d total_centers=%d"
                        " selected_mode=%d requested=%zu"
                        " considered=%d solved=%d skipped=%d evaluator=%d"
                        " cached=1"
                        " selected_build=%.6f result_alloc=%.6f"
                        " panel_setup=%.6f center_geom=%.6f"
                        " jump_eval=%.6f solve6x6=%.6f"
                        " writeback=%.6f total=%.6f\n",
                        cauchy_detail_call,
                        Np,
                        Nc,
                        selected_center_indices != nullptr ? 1 : 0,
                        selected_center_indices != nullptr
                            ? selected_center_indices->size()
                            : static_cast<std::size_t>(0),
                        considered_centers,
                        solved_centers,
                        skipped_centers,
                        jump_evaluator ? 1 : 0,
                        selected_build_sec,
                        result_alloc_sec,
                        panel_setup_sec,
                        center_geom_sec,
                        jump_eval_sec,
                        solve6x6_sec,
                        writeback_sec,
                        total_sec);
        }

        return res;
    }

    for (int p = 0; p < Np; ++p) {
        const auto panel_setup_start = std::chrono::steady_clock::now();
        const int g0 = iface.point_index(p, 0);
        const int g1 = iface.point_index(p, 1);
        const int g2 = iface.point_index(p, 2);

        const double panel_pts[3][2] = {
            { iface.points()(g0, 0), iface.points()(g0, 1) },
            { iface.points()(g1, 0), iface.points()(g1, 1) },
            { iface.points()(g2, 0), iface.points()(g2, 1) }
        };
        const double panel_nml[3][2] = {
            { iface.panel_normal(p, 0)[0], iface.panel_normal(p, 0)[1] },
            { iface.panel_normal(p, 1)[0], iface.panel_normal(p, 1)[1] },
            { iface.panel_normal(p, 2)[0], iface.panel_normal(p, 2)[1] }
        };
        const double panel_a[3]  = { a[g0], a[g1], a[g2] };
        const double panel_b[3]  = { b[g0], b[g1], b[g2] };
        const double panel_Lu[3] = { Lu_iface[g0], Lu_iface[g1], Lu_iface[g2] };
        const double panel_length = detail::panel_chord_length_2d(panel_pts);
        if (profile_detail) {
            panel_setup_sec += detail::cauchy_elapsed_seconds_2d(
                panel_setup_start, std::chrono::steady_clock::now());
        }

        for (int i = 0; i < detail::kQuadraticPanelExpansionCount; ++i) {
            const int ci = detail::kQuadraticPanelExpansionCount * p + i;
            ++considered_centers;
            if (!selected.empty()
                && !selected[static_cast<std::size_t>(ci)]) {
                ++skipped_centers;
                continue;
            }
            ++solved_centers;
            const auto center_geom_start = std::chrono::steady_clock::now();
            const double t = detail::kQuadraticPanelExpansionS[i];
            const double s_minus = t - detail::kCollocationDelta;
            const double s_plus  = t + detail::kCollocationDelta;

            double center[2];
            detail::interpolate_vector_3_at_nodes(detail::kPanelP2S, panel_pts, t, center);

            double center_normal[2];
            detail::panel_normal_2d_at_nodes(detail::kPanelP2S,
                                             panel_pts, panel_nml, t, center_normal);

            double bdry1[3][2];
            detail::interpolate_vector_3_at_nodes(detail::kPanelP2S, panel_pts, s_minus, bdry1[0]);
            detail::interpolate_vector_3_at_nodes(detail::kPanelP2S, panel_pts, t,       bdry1[1]);
            detail::interpolate_vector_3_at_nodes(detail::kPanelP2S, panel_pts, s_plus,  bdry1[2]);

            double bdry2[2][2];
            detail::interpolate_vector_3_at_nodes(detail::kPanelP2S, panel_pts, s_minus, bdry2[0]);
            detail::interpolate_vector_3_at_nodes(detail::kPanelP2S, panel_pts, s_plus,  bdry2[1]);
            double bdry2_nml[2][2];
            detail::panel_normal_2d_at_nodes(detail::kPanelP2S,
                                             panel_pts, panel_nml, s_minus, bdry2_nml[0]);
            detail::panel_normal_2d_at_nodes(detail::kPanelP2S,
                                             panel_pts, panel_nml, s_plus,  bdry2_nml[1]);

            double bulk[2];
            detail::interpolate_vector_3_at_nodes(detail::kPanelP2S, panel_pts, t, bulk);
            if (profile_detail) {
                center_geom_sec += detail::cauchy_elapsed_seconds_2d(
                    center_geom_start, std::chrono::steady_clock::now());
            }
            auto evaluate_jump = [&](double s,
                                     const double pt[2],
                                     const double nml[2],
                                     double fallback_a,
                                     double fallback_b,
                                     double fallback_rhs,
                                     double& out_a,
                                     double& out_b,
                                     double& out_rhs) {
                if (jump_evaluator) {
                    double value = 0.0;
                    double normal_value = 0.0;
                    double rhs_value = 0.0;
                    if (jump_evaluator(p,
                                       s,
                                       Eigen::Vector2d(pt[0], pt[1]),
                                       Eigen::Vector2d(nml[0], nml[1]),
                                       value,
                                       normal_value,
                                       rhs_value)) {
                        out_a = value;
                        out_b = normal_value;
                        out_rhs = rhs_value;
                        return;
                    }
                }
                out_a = fallback_a;
                out_b = fallback_b;
                out_rhs = fallback_rhs;
            };

            double av[3];
            double bv[2];
            double rhs_at_center = 0.0;
            double unused_b = 0.0;
            const auto jump_eval_start = std::chrono::steady_clock::now();
            evaluate_jump(
                s_minus,
                bdry1[0],
                bdry2_nml[0],
                detail::interpolate_scalar_3_at_nodes(detail::kPanelP2S, panel_a, s_minus),
                detail::interpolate_scalar_3_at_nodes(detail::kPanelP2S, panel_b, s_minus),
                detail::interpolate_scalar_3_at_nodes(detail::kPanelP2S, panel_Lu, s_minus),
                av[0],
                bv[0],
                rhs_at_center);
            evaluate_jump(
                t,
                bdry1[1],
                center_normal,
                detail::interpolate_scalar_3_at_nodes(detail::kPanelP2S, panel_a, t),
                detail::interpolate_scalar_3_at_nodes(detail::kPanelP2S, panel_b, t),
                detail::interpolate_scalar_3_at_nodes(detail::kPanelP2S, panel_Lu, t),
                av[1],
                unused_b,
                rhs_at_center);
            evaluate_jump(
                s_plus,
                bdry1[2],
                bdry2_nml[1],
                detail::interpolate_scalar_3_at_nodes(detail::kPanelP2S, panel_a, s_plus),
                detail::interpolate_scalar_3_at_nodes(detail::kPanelP2S, panel_b, s_plus),
                detail::interpolate_scalar_3_at_nodes(detail::kPanelP2S, panel_Lu, s_plus),
                av[2],
                bv[1],
                unused_b);
            const double Lu = rhs_at_center;
            if (profile_detail) {
                jump_eval_sec += detail::cauchy_elapsed_seconds_2d(
                    jump_eval_start, std::chrono::steady_clock::now());
            }

            const auto solve_start = std::chrono::steady_clock::now();
            double h = 0.0;
            for (int l = 0; l < 3; ++l) {
                h = std::max(h, std::hypot(bdry1[l][0] - center[0],
                                           bdry1[l][1] - center[1]));
            }
            if (h <= 1e-14)
                h = panel_length;

            double c[6];
            solve_local_6x6_2d(bdry1, av,
                                bdry2, bdry2_nml, bv,
                                bulk, Lu,
                                center, kappa, h,
                                c);
            if (profile_detail) {
                solve6x6_sec += detail::cauchy_elapsed_seconds_2d(
                    solve_start, std::chrono::steady_clock::now());
            }

            const auto writeback_start = std::chrono::steady_clock::now();
            res.centers(ci, 0) = center[0];
            res.centers(ci, 1) = center[1];
            res.normals(ci, 0) = center_normal[0];
            res.normals(ci, 1) = center_normal[1];
            res.panel_index[ci] = p;
            res.local_s[ci] = t;
            res.C  [ci] = c[0];
            res.Cx [ci] = c[1];
            res.Cy [ci] = c[2];
            res.Cxx[ci] = c[3];
            res.Cyy[ci] = c[4];
            res.Cxy[ci] = c[5];
            if (profile_detail) {
                writeback_sec += detail::cauchy_elapsed_seconds_2d(
                    writeback_start, std::chrono::steady_clock::now());
            }
        }
    }

    if (profile_detail) {
        static int cauchy_detail_call = 0;
        ++cauchy_detail_call;
        const double total_sec = detail::cauchy_elapsed_seconds_2d(
            profile_total_start, std::chrono::steady_clock::now());
        std::printf("KFBIM_PROFILE_CAUCHY_DETAIL"
                    " call=%d panels=%d total_centers=%d"
                    " selected_mode=%d requested=%zu"
                    " considered=%d solved=%d skipped=%d evaluator=%d"
                    " selected_build=%.6f result_alloc=%.6f"
                    " panel_setup=%.6f center_geom=%.6f"
                    " jump_eval=%.6f solve6x6=%.6f"
                    " writeback=%.6f total=%.6f\n",
                    cauchy_detail_call,
                    Np,
                    Nc,
                    selected_center_indices != nullptr ? 1 : 0,
                    selected_center_indices != nullptr
                        ? selected_center_indices->size()
                        : static_cast<std::size_t>(0),
                    considered_centers,
                    solved_centers,
                    skipped_centers,
                    jump_evaluator ? 1 : 0,
                    selected_build_sec,
                    result_alloc_sec,
                    panel_setup_sec,
                    center_geom_sec,
                    jump_eval_sec,
                    solve6x6_sec,
                    writeback_sec,
                    total_sec);
    }

    return res;
}

inline PanelCenterCauchyResult2D laplace_panel_lobatto_center_cauchy_2d(
    const Interface2D&     iface,
    const Eigen::VectorXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& Lu_iface,
    double                 kappa = 0.0)
{
    return laplace_panel_quadratic_center_cauchy_2d(
        iface, a, b, Lu_iface, kappa);
}

} // namespace kfbim
