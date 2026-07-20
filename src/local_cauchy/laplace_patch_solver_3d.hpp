#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "../interface/interface_3d.hpp"

namespace kfbim {

namespace detail3d {

struct RefTriangle {
    std::array<Eigen::Vector3d, 3> bary;
};

inline std::vector<RefTriangle> subdivided_reference_triangles()
{
    constexpr int n = 4;
    auto bary = [n](int i, int j) {
        const double l1 = static_cast<double>(i) / static_cast<double>(n);
        const double l2 = static_cast<double>(j) / static_cast<double>(n);
        const double l0 = 1.0 - l1 - l2;
        return Eigen::Vector3d(l0, l1, l2);
    };

    std::vector<RefTriangle> tris;
    tris.reserve(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - i; ++j) {
            tris.push_back({{bary(i, j), bary(i + 1, j), bary(i, j + 1)}});
            if (i + j < n - 1)
                tris.push_back({{bary(i + 1, j),
                                 bary(i + 1, j + 1),
                                 bary(i, j + 1)}});
        }
    }
    return tris;
}

inline void p2_shape(Eigen::Vector3d bary, double N[6])
{
    const double l0 = bary[0];
    const double l1 = bary[1];
    const double l2 = bary[2];
    N[0] = l0 * (2.0 * l0 - 1.0);
    N[1] = l1 * (2.0 * l1 - 1.0);
    N[2] = l2 * (2.0 * l2 - 1.0);
    N[3] = 4.0 * l0 * l1;
    N[4] = 4.0 * l1 * l2;
    N[5] = 4.0 * l2 * l0;
}

inline Eigen::Vector3d panel_point(const Interface3D& iface,
                                   int                panel,
                                   Eigen::Vector3d    bary)
{
    double N[6];
    p2_shape(bary, N);
    Eigen::Vector3d pt = Eigen::Vector3d::Zero();
    for (int q = 0; q < 6; ++q)
        pt += N[q] * iface.points().row(iface.point_index(panel, q)).transpose();
    return pt;
}

inline Eigen::Vector3d panel_normal(const Interface3D& iface,
                                    int                panel,
                                    Eigen::Vector3d    bary)
{
    double N[6];
    p2_shape(bary, N);
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    for (int q = 0; q < 6; ++q)
        normal += N[q] * iface.normals().row(iface.point_index(panel, q)).transpose();

    const double len = normal.norm();
    if (len <= 1.0e-14)
        return iface.normals().row(iface.point_index(panel, 0)).transpose();
    return normal / len;
}

inline double panel_scalar(const Interface3D&     iface,
                           int                    panel,
                           const Eigen::VectorXd& values,
                           Eigen::Vector3d        bary)
{
    double N[6];
    p2_shape(bary, N);
    double value = 0.0;
    for (int q = 0; q < 6; ++q)
        value += N[q] * values[iface.point_index(panel, q)];
    return value;
}

} // namespace detail3d

// Output type for the active 3D P2-patch local Cauchy solve.
//
// One parent triangle is subdivided twice into 16 small triangles. The local
// Cauchy solve is performed at each small triangle barycenter. Coefficients use
// repo LocalPoly3D order:
//   C, Cx, Cy, Cz, Cxx, Cxy, Cxz, Cyy, Cyz, Czz.
struct PatchCenterCauchyResult3D {
    Eigen::MatrixX3d centers;
    Eigen::MatrixX3d normals;
    Eigen::VectorXi  panel_index;
    Eigen::MatrixX3d barycentric;
    Eigen::MatrixXd  coeffs;
};

// Optional pointwise modification of the physical-interface jump data used
// by one sub-triangle-center Cauchy solve.  The incoming value/normal/RHS
// entries are the ordinary P2 interpolants.  A singular-patch caller can
// subtract its trace, normal trace, and (-Delta+kappa) trace in place before
// the 10x10 polynomial coefficients are formed.
using PatchCenterJumpModifier3D = std::function<void(
    int center_index,
    int panel,
    Eigen::Vector3d barycentric,
    Eigen::Vector3d point,
    Eigen::Vector3d normal,
    double& value_jump,
    double& normal_jump,
    double& rhs_jump)>;

inline void solve_local_10x10_3d(const double bdry1[6][3],
                                 const double a[6],
                                 const double bdry2[3][3],
                                 const double bdry2_nml[3][3],
                                 const double b[3],
                                 const double bulk[3],
                                 double       Lu,
                                 const double center[3],
                                 double       kappa,
                                 double       h,
                                 double       c[10])
{
    const double h2 = h * h;
    const double kap_h2 = kappa * h2;

    Eigen::Matrix<double, 10, 10> A;
    Eigen::Matrix<double, 10, 1> rhs;
    A.setZero();
    rhs.setZero();

    int m = 0;

    auto fill_value_row = [&](int row, const double pt[3]) {
        const double dx = (pt[0] - center[0]) / h;
        const double dy = (pt[1] - center[1]) / h;
        const double dz = (pt[2] - center[2]) / h;
        A(row, 0) = 1.0;
        A(row, 1) = dx;
        A(row, 2) = dy;
        A(row, 3) = dz;
        A(row, 4) = 0.5 * dx * dx;
        A(row, 5) = dx * dy;
        A(row, 6) = dx * dz;
        A(row, 7) = 0.5 * dy * dy;
        A(row, 8) = dy * dz;
        A(row, 9) = 0.5 * dz * dz;
    };

    for (int l = 0; l < 6; ++l) {
        fill_value_row(m, bdry1[l]);
        rhs[m] = a[l];
        ++m;
    }

    for (int l = 0; l < 3; ++l) {
        const double dx = (bdry2[l][0] - center[0]) / h;
        const double dy = (bdry2[l][1] - center[1]) / h;
        const double dz = (bdry2[l][2] - center[2]) / h;
        const double nx = bdry2_nml[l][0];
        const double ny = bdry2_nml[l][1];
        const double nz = bdry2_nml[l][2];

        A(m, 0) = 0.0;
        A(m, 1) = nx;
        A(m, 2) = ny;
        A(m, 3) = nz;
        A(m, 4) = dx * nx;
        A(m, 5) = nx * dy + dx * ny;
        A(m, 6) = nx * dz + dx * nz;
        A(m, 7) = dy * ny;
        A(m, 8) = ny * dz + dy * nz;
        A(m, 9) = dz * nz;
        rhs[m] = b[l] * h;
        ++m;
    }

    {
        fill_value_row(m, bulk);
        A.row(m) *= kap_h2;
        A(m, 4) -= 1.0;
        A(m, 7) -= 1.0;
        A(m, 9) -= 1.0;
        rhs[m] = Lu * h2;
        ++m;
    }

    Eigen::Matrix<double, 10, 1> col_scale;
    for (int k = 0; k < 10; ++k) {
        col_scale[k] = A.col(k).norm();
        if (col_scale[k] > 1.0e-14)
            A.col(k) /= col_scale[k];
        else
            col_scale[k] = 1.0;
    }

    Eigen::Matrix<double, 10, 1> c_raw =
        A.colPivHouseholderQr().solve(rhs);
    for (int k = 0; k < 10; ++k)
        c_raw[k] /= col_scale[k];

    c[0] = c_raw[0];
    for (int k = 1; k < 4; ++k)
        c[k] = c_raw[k] / h;
    for (int k = 4; k < 10; ++k)
        c[k] = c_raw[k] / h2;
}

inline PatchCenterCauchyResult3D laplace_p2_patch_center_cauchy_3d(
    const Interface3D&     iface,
    const Eigen::VectorXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& Lu_iface,
    double                 kappa = 0.0,
    PatchCenterJumpModifier3D jump_modifier = {},
    const std::vector<int>* selected_centers = nullptr)
{
    if (iface.points_per_panel() != 6
        || iface.panel_node_layout() != PanelNodeLayout3D::QuadraticLagrange) {
        throw std::invalid_argument(
            "laplace_p2_patch_center_cauchy_3d requires P2 QuadraticLagrange panels");
    }
    if (a.size() != iface.num_points()
        || b.size() != iface.num_points()
        || Lu_iface.size() != iface.num_points()) {
        throw std::invalid_argument(
            "laplace_p2_patch_center_cauchy_3d input vectors must match interface size");
    }

    const int centers_per_panel = 16;
    const int n_centers = centers_per_panel * iface.num_panels();
    const std::vector<detail3d::RefTriangle> ref_tris =
        detail3d::subdivided_reference_triangles();

    PatchCenterCauchyResult3D res;
    res.centers = Eigen::MatrixX3d::Zero(n_centers, 3);
    res.normals = Eigen::MatrixX3d::Zero(n_centers, 3);
    res.panel_index = Eigen::VectorXi::Zero(n_centers);
    res.barycentric = Eigen::MatrixX3d::Zero(n_centers, 3);
    res.coeffs = Eigen::MatrixXd::Zero(n_centers, 10);

    std::vector<char> selected;
    if (selected_centers != nullptr) {
        selected.assign(static_cast<std::size_t>(n_centers), 0);
        for (int center_index : *selected_centers) {
            if (center_index < 0 || center_index >= n_centers) {
                throw std::out_of_range(
                    "laplace_p2_patch_center_cauchy_3d selected center is out of range");
            }
            selected[static_cast<std::size_t>(center_index)] = 1;
        }
    }

    for (int p = 0; p < iface.num_panels(); ++p) {
        for (int t = 0; t < centers_per_panel; ++t) {
            const int ci = centers_per_panel * p + t;
            if (!selected.empty()
                && !selected[static_cast<std::size_t>(ci)]) {
                continue;
            }
            const auto& tri = ref_tris[t];

            const Eigen::Vector3d edge01 = 0.5 * (tri.bary[0] + tri.bary[1]);
            const Eigen::Vector3d edge12 = 0.5 * (tri.bary[1] + tri.bary[2]);
            const Eigen::Vector3d edge20 = 0.5 * (tri.bary[2] + tri.bary[0]);
            const Eigen::Vector3d center_bary =
                (tri.bary[0] + tri.bary[1] + tri.bary[2]) / 3.0;

            const std::array<Eigen::Vector3d, 6> dir_bary = {
                tri.bary[0], tri.bary[1], tri.bary[2], edge01, edge12, edge20};
            const std::array<Eigen::Vector3d, 3> neu_bary = {
                edge01, edge12, edge20};

            const Eigen::Vector3d center =
                detail3d::panel_point(iface, p, center_bary);

            double bdry1[6][3];
            double av[6];
            double max_dist = 0.0;
            for (int q = 0; q < 6; ++q) {
                const Eigen::Vector3d pt =
                    detail3d::panel_point(iface, p, dir_bary[q]);
                const Eigen::Vector3d normal =
                    detail3d::panel_normal(iface, p, dir_bary[q]);
                bdry1[q][0] = pt[0];
                bdry1[q][1] = pt[1];
                bdry1[q][2] = pt[2];
                double value_jump =
                    detail3d::panel_scalar(iface, p, a, dir_bary[q]);
                double normal_jump =
                    detail3d::panel_scalar(iface, p, b, dir_bary[q]);
                double rhs_jump =
                    detail3d::panel_scalar(iface, p, Lu_iface, dir_bary[q]);
                if (jump_modifier) {
                    jump_modifier(ci,
                                  p,
                                  dir_bary[q],
                                  pt,
                                  normal,
                                  value_jump,
                                  normal_jump,
                                  rhs_jump);
                }
                av[q] = value_jump;
                max_dist = std::max(max_dist, (pt - center).norm());
            }

            double bdry2[3][3];
            double bdry2_nml[3][3];
            double bv[3];
            for (int q = 0; q < 3; ++q) {
                const Eigen::Vector3d pt =
                    detail3d::panel_point(iface, p, neu_bary[q]);
                const Eigen::Vector3d normal =
                    detail3d::panel_normal(iface, p, neu_bary[q]);
                bdry2[q][0] = pt[0];
                bdry2[q][1] = pt[1];
                bdry2[q][2] = pt[2];
                bdry2_nml[q][0] = normal[0];
                bdry2_nml[q][1] = normal[1];
                bdry2_nml[q][2] = normal[2];
                double value_jump =
                    detail3d::panel_scalar(iface, p, a, neu_bary[q]);
                double normal_jump =
                    detail3d::panel_scalar(iface, p, b, neu_bary[q]);
                double rhs_jump =
                    detail3d::panel_scalar(iface, p, Lu_iface, neu_bary[q]);
                if (jump_modifier) {
                    jump_modifier(ci,
                                  p,
                                  neu_bary[q],
                                  pt,
                                  normal,
                                  value_jump,
                                  normal_jump,
                                  rhs_jump);
                }
                bv[q] = normal_jump;
            }

            const double bulk[3] = {center[0], center[1], center[2]};
            const Eigen::Vector3d center_normal =
                detail3d::panel_normal(iface, p, center_bary);
            double center_value =
                detail3d::panel_scalar(iface, p, a, center_bary);
            double center_normal_jump =
                detail3d::panel_scalar(iface, p, b, center_bary);
            double Lu =
                detail3d::panel_scalar(iface, p, Lu_iface, center_bary);
            if (jump_modifier) {
                jump_modifier(ci,
                              p,
                              center_bary,
                              center,
                              center_normal,
                              center_value,
                              center_normal_jump,
                              Lu);
            }
            const double h = std::max(max_dist, 1.0e-14);

            double c[10];
            solve_local_10x10_3d(bdry1, av,
                                  bdry2, bdry2_nml, bv,
                                  bulk, Lu,
                                  bulk, kappa, h,
                                  c);

            res.centers.row(ci) = center.transpose();
            res.normals.row(ci) =
                detail3d::panel_normal(iface, p, center_bary).transpose();
            res.panel_index[ci] = p;
            res.barycentric.row(ci) = center_bary.transpose();
            for (int k = 0; k < 10; ++k)
                res.coeffs(ci, k) = c[k];
        }
    }

    return res;
}

} // namespace kfbim
