#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "../interface/interface_3d.hpp"

namespace kfbim::geometry3d {

using P2Shape = Eigen::Matrix<double, 6, 1>;

struct RefTriangle {
    std::array<Eigen::Vector3d, 3> bary;
};

inline Eigen::Vector3d barycentric_from_uv(Eigen::Vector2d uv)
{
    return {1.0 - uv[0] - uv[1], uv[0], uv[1]};
}

inline Eigen::Vector2d uv_from_barycentric(Eigen::Vector3d bary)
{
    return {bary[1], bary[2]};
}

inline P2Shape p2_shape(Eigen::Vector3d bary)
{
    const double l0 = bary[0];
    const double l1 = bary[1];
    const double l2 = bary[2];

    P2Shape N;
    N[0] = l0 * (2.0 * l0 - 1.0);
    N[1] = l1 * (2.0 * l1 - 1.0);
    N[2] = l2 * (2.0 * l2 - 1.0);
    N[3] = 4.0 * l0 * l1;
    N[4] = 4.0 * l1 * l2;
    N[5] = 4.0 * l2 * l0;
    return N;
}

inline P2Shape p2_shape_du(Eigen::Vector3d bary)
{
    const double l0 = bary[0];
    const double l1 = bary[1];
    const double l2 = bary[2];

    P2Shape N;
    N[0] = 1.0 - 4.0 * l0;
    N[1] = 4.0 * l1 - 1.0;
    N[2] = 0.0;
    N[3] = 4.0 * (l0 - l1);
    N[4] = 4.0 * l2;
    N[5] = -4.0 * l2;
    return N;
}

inline P2Shape p2_shape_dv(Eigen::Vector3d bary)
{
    const double l0 = bary[0];
    const double l1 = bary[1];
    const double l2 = bary[2];

    P2Shape N;
    N[0] = 1.0 - 4.0 * l0;
    N[1] = 0.0;
    N[2] = 4.0 * l2 - 1.0;
    N[3] = -4.0 * l1;
    N[4] = 4.0 * l1;
    N[5] = 4.0 * (l0 - l2);
    return N;
}

inline P2Shape p2_shape_duu()
{
    P2Shape N;
    N << 4.0, 4.0, 0.0, -8.0, 0.0, 0.0;
    return N;
}

inline P2Shape p2_shape_duv()
{
    P2Shape N;
    N << 4.0, 0.0, 0.0, -4.0, 4.0, -4.0;
    return N;
}

inline P2Shape p2_shape_dvv()
{
    P2Shape N;
    N << 4.0, 0.0, 4.0, 0.0, 0.0, -8.0;
    return N;
}

inline Eigen::Vector3d panel_combination(const Interface3D& iface,
                                         int                panel,
                                         const P2Shape&     coeffs)
{
    Eigen::Vector3d value = Eigen::Vector3d::Zero();
    for (int q = 0; q < 6; ++q)
        value += coeffs[q] * iface.points().row(iface.point_index(panel, q)).transpose();
    return value;
}

inline Eigen::Vector3d panel_point(const Interface3D& iface,
                                   int                panel,
                                   Eigen::Vector3d    bary)
{
    return panel_combination(iface, panel, p2_shape(bary));
}

inline Eigen::Vector3d panel_tangent_u(const Interface3D& iface,
                                       int                panel,
                                       Eigen::Vector3d    bary)
{
    return panel_combination(iface, panel, p2_shape_du(bary));
}

inline Eigen::Vector3d panel_tangent_v(const Interface3D& iface,
                                       int                panel,
                                       Eigen::Vector3d    bary)
{
    return panel_combination(iface, panel, p2_shape_dv(bary));
}

inline Eigen::Vector3d panel_second_uu(const Interface3D& iface, int panel)
{
    return panel_combination(iface, panel, p2_shape_duu());
}

inline Eigen::Vector3d panel_second_uv(const Interface3D& iface, int panel)
{
    return panel_combination(iface, panel, p2_shape_duv());
}

inline Eigen::Vector3d panel_second_vv(const Interface3D& iface, int panel)
{
    return panel_combination(iface, panel, p2_shape_dvv());
}

inline Eigen::Vector3d panel_interpolated_normal(const Interface3D& iface,
                                                 int                panel,
                                                 Eigen::Vector3d    bary)
{
    const P2Shape N = p2_shape(bary);
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    for (int q = 0; q < 6; ++q)
        normal += N[q] * iface.normals().row(iface.point_index(panel, q)).transpose();

    double len = normal.norm();
    if (len > 1.0e-14)
        return normal / len;

    normal = iface.normals().row(iface.point_index(panel, 0)).transpose();
    len = normal.norm();
    if (len > 1.0e-14)
        return normal / len;

    return Eigen::Vector3d::UnitX();
}

inline Eigen::Vector3d panel_oriented_normal(const Interface3D& iface,
                                             int                panel,
                                             Eigen::Vector3d    bary)
{
    Eigen::Vector3d normal =
        panel_tangent_u(iface, panel, bary).cross(panel_tangent_v(iface, panel, bary));
    const double len = normal.norm();
    if (len <= 1.0e-14)
        return panel_interpolated_normal(iface, panel, bary);

    normal /= len;
    const Eigen::Vector3d ref = panel_interpolated_normal(iface, panel, bary);
    if (normal.dot(ref) < 0.0)
        normal = -normal;
    return normal;
}

inline double panel_scalar(const Interface3D&     iface,
                           int                    panel,
                           const Eigen::VectorXd& values,
                           Eigen::Vector3d        bary)
{
    const P2Shape N = p2_shape(bary);
    double value = 0.0;
    for (int q = 0; q < 6; ++q)
        value += N[q] * values[iface.point_index(panel, q)];
    return value;
}

inline double panel_scalar_combination(const Interface3D&     iface,
                                       int                    panel,
                                       const Eigen::VectorXd& values,
                                       const P2Shape&         coeffs)
{
    double value = 0.0;
    for (int q = 0; q < 6; ++q)
        value += coeffs[q] * values[iface.point_index(panel, q)];
    return value;
}

inline double panel_scalar_du(const Interface3D&     iface,
                              int                    panel,
                              const Eigen::VectorXd& values,
                              Eigen::Vector3d        bary)
{
    return panel_scalar_combination(iface, panel, values, p2_shape_du(bary));
}

inline double panel_scalar_dv(const Interface3D&     iface,
                              int                    panel,
                              const Eigen::VectorXd& values,
                              Eigen::Vector3d        bary)
{
    return panel_scalar_combination(iface, panel, values, p2_shape_dv(bary));
}

inline double panel_scalar_duu(const Interface3D&     iface,
                               int                    panel,
                               const Eigen::VectorXd& values)
{
    return panel_scalar_combination(iface, panel, values, p2_shape_duu());
}

inline double panel_scalar_duv(const Interface3D&     iface,
                               int                    panel,
                               const Eigen::VectorXd& values)
{
    return panel_scalar_combination(iface, panel, values, p2_shape_duv());
}

inline double panel_scalar_dvv(const Interface3D&     iface,
                               int                    panel,
                               const Eigen::VectorXd& values)
{
    return panel_scalar_combination(iface, panel, values, p2_shape_dvv());
}

inline Eigen::Matrix2d panel_metric(const Interface3D& iface,
                                    int                panel,
                                    Eigen::Vector3d    bary)
{
    const Eigen::Vector3d Xu = panel_tangent_u(iface, panel, bary);
    const Eigen::Vector3d Xv = panel_tangent_v(iface, panel, bary);

    Eigen::Matrix2d metric;
    metric << Xu.dot(Xu), Xu.dot(Xv),
              Xv.dot(Xu), Xv.dot(Xv);
    return metric;
}

inline double panel_mean_curvature(const Interface3D& iface,
                                   int                panel,
                                   Eigen::Vector3d    bary)
{
    const Eigen::Vector3d Xu = panel_tangent_u(iface, panel, bary);
    const Eigen::Vector3d Xv = panel_tangent_v(iface, panel, bary);
    const Eigen::Vector3d Xuu = panel_second_uu(iface, panel);
    const Eigen::Vector3d Xuv = panel_second_uv(iface, panel);
    const Eigen::Vector3d Xvv = panel_second_vv(iface, panel);
    const Eigen::Vector3d normal = panel_oriented_normal(iface, panel, bary);

    const double E = Xu.dot(Xu);
    const double F = Xu.dot(Xv);
    const double G = Xv.dot(Xv);
    const double det = E * G - F * F;
    if (std::abs(det) <= 1.0e-28)
        throw std::invalid_argument("degenerate P2 panel metric");

    const double e = normal.dot(Xuu);
    const double f = normal.dot(Xuv);
    const double g = normal.dot(Xvv);

    // The repo convention is H = div(n). With outward normals this is
    // positive for a sphere, hence the negative trace of the shape operator.
    return -(e * G - 2.0 * f * F + g * E) / det;
}

inline double panel_laplace_beltrami_scalar(const Interface3D&     iface,
                                            int                    panel,
                                            const Eigen::VectorXd& values,
                                            Eigen::Vector3d        bary)
{
    const Eigen::Vector3d Xu = panel_tangent_u(iface, panel, bary);
    const Eigen::Vector3d Xv = panel_tangent_v(iface, panel, bary);
    const Eigen::Vector3d Xuu = panel_second_uu(iface, panel);
    const Eigen::Vector3d Xuv = panel_second_uv(iface, panel);
    const Eigen::Vector3d Xvv = panel_second_vv(iface, panel);

    const Eigen::Matrix2d metric = panel_metric(iface, panel, bary);
    const double det = metric.determinant();
    if (std::abs(det) <= 1.0e-28)
        throw std::invalid_argument("degenerate P2 panel metric");
    const Eigen::Matrix2d inv_metric = metric.inverse();

    const double dE_du = 2.0 * Xu.dot(Xuu);
    const double dE_dv = 2.0 * Xu.dot(Xuv);
    const double dF_du = Xuu.dot(Xv) + Xu.dot(Xuv);
    const double dF_dv = Xuv.dot(Xv) + Xu.dot(Xvv);
    const double dG_du = 2.0 * Xv.dot(Xuv);
    const double dG_dv = 2.0 * Xv.dot(Xvv);

    std::array<Eigen::Matrix2d, 2> dg;
    dg[0] << dE_du, dF_du,
             dF_du, dG_du;
    dg[1] << dE_dv, dF_dv,
             dF_dv, dG_dv;

    double gamma[2][2][2] = {};
    for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int l = 0; l < 2; ++l) {
                    gamma[k][i][j] += 0.5 * inv_metric(k, l)
                        * (dg[i](j, l) + dg[j](i, l) - dg[l](i, j));
                }
            }
        }
    }

    Eigen::Vector2d grad;
    grad << panel_scalar_du(iface, panel, values, bary),
            panel_scalar_dv(iface, panel, values, bary);

    Eigen::Matrix2d hess;
    hess << panel_scalar_duu(iface, panel, values),
            panel_scalar_duv(iface, panel, values),
            panel_scalar_duv(iface, panel, values),
            panel_scalar_dvv(iface, panel, values);

    double laplace_beltrami = 0.0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            laplace_beltrami += inv_metric(i, j)
                * (hess(i, j)
                   - gamma[0][i][j] * grad[0]
                   - gamma[1][i][j] * grad[1]);
        }
    }

    return laplace_beltrami;
}

inline std::vector<RefTriangle> subdivided_reference_triangles(int n = 4)
{
    if (n <= 0)
        throw std::invalid_argument("subdivision count must be positive");

    auto bary = [n](int i, int j) {
        const double l1 = static_cast<double>(i) / static_cast<double>(n);
        const double l2 = static_cast<double>(j) / static_cast<double>(n);
        const double l0 = 1.0 - l1 - l2;
        return Eigen::Vector3d(l0, l1, l2);
    };

    std::vector<RefTriangle> tris;
    tris.reserve(static_cast<std::size_t>(n * n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - i; ++j) {
            tris.push_back({{bary(i, j), bary(i + 1, j), bary(i, j + 1)}});
            if (i + j < n - 1) {
                tris.push_back({{bary(i + 1, j),
                                 bary(i + 1, j + 1),
                                 bary(i, j + 1)}});
            }
        }
    }
    return tris;
}

inline std::vector<Eigen::Vector3d> expansion_center_barycentrics(int n = 4)
{
    const std::vector<RefTriangle> tris = subdivided_reference_triangles(n);
    std::vector<Eigen::Vector3d> centers;
    centers.reserve(tris.size());
    for (const auto& tri : tris)
        centers.push_back((tri.bary[0] + tri.bary[1] + tri.bary[2]) / 3.0);
    return centers;
}

} // namespace kfbim::geometry3d
