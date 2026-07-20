#pragma once

#include <stdexcept>
#include <Eigen/Dense>
#include "jump_data.hpp"  // num_monomials, poly_max_degree

namespace kfbim {

// ---------------------------------------------------------------------------
// Local correction polynomial — value type produced by ILocalCauchySolver::fit()
// and consumed by ILocalCauchySolver::evaluate().
//
// Taylor-derivative basis ordering (same convention as rhs_derivs in jump_data.hpp):
// grouped by total degree, then lexicographic within each group.
//
// 2D: 1 | x,y | x²,xy,y² | x³,x²y,xy²,y³ | ...
// 3D: 1 | x,y,z | x²,xy,xz,y²,yz,z² | ...
//
// The stored coefficient for x^a y^b is the Cartesian derivative
// d^(a+b)u / dx^a dy^b at center. Evaluation therefore uses Taylor factorial
// factors 1/(a! b!). For degree <= 2 in 2D:
//   coeffs = [u, u_x, u_y, u_xx, u_xy, u_yy].
//
// coeffs length = num_monomials(poly_max_degree(method_order), dim)
// ---------------------------------------------------------------------------

struct LocalPoly2D {
    Eigen::Vector2d center;  // expansion origin = interface quadrature point
    Eigen::VectorXd coeffs;  // length = num_monomials(poly_max_degree(order), 2)
};

struct LocalPoly3D {
    Eigen::Vector3d center;
    Eigen::VectorXd coeffs;  // length = num_monomials(poly_max_degree(order), 3)
};

// ---------------------------------------------------------------------------
// Stokes: velocity and pressure have separate polynomial degrees.
// velocity: poly_max_degree(order)
// pressure: poly_max_degree(order) - 1   (one degree lower for inf-sup stability)
// ---------------------------------------------------------------------------

struct StokesLocalPoly2D {
    Eigen::Vector2d  center;
    Eigen::MatrixX2d vel_coeffs;   // num_monomials(poly_max_degree(order),   2) x 2
    Eigen::VectorXd  pres_coeffs;  // num_monomials(poly_max_degree(order)-1, 2)
};

struct StokesLocalPoly3D {
    Eigen::Vector3d  center;
    Eigen::MatrixX3d vel_coeffs;   // num_monomials(poly_max_degree(order),   3) x 3
    Eigen::VectorXd  pres_coeffs;  // num_monomials(poly_max_degree(order)-1, 3)
};

inline double local_poly_factorial(int n) {
    double result = 1.0;
    for (int i = 2; i <= n; ++i)
        result *= static_cast<double>(i);
    return result;
}

inline double local_poly_pow(double x, int n) {
    double result = 1.0;
    for (int i = 0; i < n; ++i)
        result *= x;
    return result;
}

inline int taylor_degree_from_num_coeffs_2d(int n_coeffs) {
    int total = 0;
    for (int degree = 0; total < n_coeffs; ++degree) {
        total += degree + 1;
        if (total == n_coeffs)
            return degree;
    }
    throw std::invalid_argument("invalid 2D Taylor polynomial coefficient count");
}

inline int taylor_degree_from_num_coeffs_3d(int n_coeffs) {
    int total = 0;
    for (int degree = 0; total < n_coeffs; ++degree) {
        total += (degree + 1) * (degree + 2) / 2;
        if (total == n_coeffs)
            return degree;
    }
    throw std::invalid_argument("invalid 3D Taylor polynomial coefficient count");
}

// Evaluate a 2D LocalPoly in the Taylor-derivative basis.
inline double evaluate_taylor_poly_2d(const LocalPoly2D& poly,
                                      Eigen::Vector2d    pt)
{
    const int max_degree = taylor_degree_from_num_coeffs_2d(
        static_cast<int>(poly.coeffs.size()));
    const double dx = pt[0] - poly.center[0];
    const double dy = pt[1] - poly.center[1];

    double value = 0.0;
    int idx = 0;
    for (int degree = 0; degree <= max_degree; ++degree) {
        for (int ay = 0; ay <= degree; ++ay) {
            const int ax = degree - ay;
            value += poly.coeffs[idx] * local_poly_pow(dx, ax) * local_poly_pow(dy, ay)
                     / (local_poly_factorial(ax) * local_poly_factorial(ay));
            ++idx;
        }
    }

    return value;
}

inline Eigen::Vector2d evaluate_taylor_poly_gradient_2d(
    const LocalPoly2D& poly,
    Eigen::Vector2d pt)
{
    const int max_degree = taylor_degree_from_num_coeffs_2d(
        static_cast<int>(poly.coeffs.size()));
    const double dx = pt[0] - poly.center[0];
    const double dy = pt[1] - poly.center[1];

    Eigen::Vector2d grad = Eigen::Vector2d::Zero();
    int idx = 0;
    for (int degree = 0; degree <= max_degree; ++degree) {
        for (int ay = 0; ay <= degree; ++ay) {
            const int ax = degree - ay;
            const double coeff = poly.coeffs[idx]
                / (local_poly_factorial(ax) * local_poly_factorial(ay));
            if (ax > 0) {
                grad[0] += coeff * static_cast<double>(ax)
                         * local_poly_pow(dx, ax - 1)
                         * local_poly_pow(dy, ay);
            }
            if (ay > 0) {
                grad[1] += coeff * local_poly_pow(dx, ax)
                         * static_cast<double>(ay)
                         * local_poly_pow(dy, ay - 1);
            }
            ++idx;
        }
    }
    return grad;
}

// Evaluate a 3D LocalPoly in the Taylor-derivative basis.
inline double evaluate_taylor_poly_3d(const LocalPoly3D& poly,
                                      Eigen::Vector3d    pt)
{
    const int max_degree = taylor_degree_from_num_coeffs_3d(
        static_cast<int>(poly.coeffs.size()));
    const double dx = pt[0] - poly.center[0];
    const double dy = pt[1] - poly.center[1];
    const double dz = pt[2] - poly.center[2];

    double value = 0.0;
    int idx = 0;
    for (int degree = 0; degree <= max_degree; ++degree) {
        for (int ax = degree; ax >= 0; --ax) {
            for (int ay = degree - ax; ay >= 0; --ay) {
                const int az = degree - ax - ay;
                value += poly.coeffs[idx]
                         * local_poly_pow(dx, ax)
                         * local_poly_pow(dy, ay)
                         * local_poly_pow(dz, az)
                         / (local_poly_factorial(ax)
                            * local_poly_factorial(ay)
                            * local_poly_factorial(az));
                ++idx;
            }
        }
    }

    return value;
}

} // namespace kfbim
