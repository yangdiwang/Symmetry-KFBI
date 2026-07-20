#pragma once

#include <vector>
#include <Eigen/Dense>

namespace kfbim {

// Number of monomials of total degree <= max_degree in dim variables.
// Equals C(max_degree + dim, dim).
// 2D: degree<=2 → 6,  degree<=4 → 15
// 3D: degree<=2 → 10, degree<=4 → 35
inline int num_monomials(int max_degree, int dim) {
    int result = 1;
    for (int i = 0; i < dim; ++i)
        result = result * (max_degree + 1 + i) / (i + 1);
    return result;
}

// Max degree of the correction polynomial for a given method order.
// The local Cauchy polynomial uses all monomials up to this degree.
// 2nd order → 2:  {1,x,y,x²,xy,y²}              (6 terms in 2D)
// 4th order → 4:  {1,...,x⁴,x³y,...,y⁴}          (15 terms in 2D)
inline int poly_max_degree(int method_order) { return method_order; }

// Max degree of the RHS derivative jumps required for a given method order.
// 2nd order → 0:  [f] only
// 4th order → 2:  [f],[f_x],[f_y],[f_xx],[f_xy],[f_yy]  (6 terms in 2D)
inline int rhs_max_degree(int method_order) { return method_order - 2; }

// ---------------------------------------------------------------------------
// RHS derivative ordering (shared convention across all JumpData types)
//
// Entries are grouped by total derivative degree, then ordered
// lexicographically within each group.
//
// 2D, degree <= 2:
//   idx 0: [f]
//   idx 1: [f_x]   idx 2: [f_y]
//   idx 3: [f_xx]  idx 4: [f_xy]  idx 5: [f_yy]
//
// 3D, degree <= 2:
//   idx 0: [f]
//   idx 1: [f_x]   idx 2: [f_y]   idx 3: [f_z]
//   idx 4: [f_xx]  idx 5: [f_xy]  idx 6: [f_xz]
//   idx 7: [f_yy]  idx 8: [f_yz]  idx 9: [f_zz]
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Laplace/Poisson jump data
// ---------------------------------------------------------------------------

struct LaplaceJumpData2D {
    double          u_jump;     // [u]
    double          un_jump;    // [∂u/∂n]
    // rhs_derivs[i] = jump of the i-th Cartesian partial derivative of f
    // length = num_monomials(rhs_max_degree(method_order), 2)
    Eigen::VectorXd rhs_derivs;
};

struct LaplaceJumpData3D {
    double          u_jump;
    double          un_jump;
    // length = num_monomials(rhs_max_degree(method_order), 3)
    Eigen::VectorXd rhs_derivs;
};

// ---------------------------------------------------------------------------
// Stokes jump data
// ---------------------------------------------------------------------------

struct StokesJumpData2D {
    Eigen::Vector2d u_jump;        // [u] velocity jump
    Eigen::Vector2d traction_jump; // [σ(u,p)·n] full traction jump (viscous + pressure)
    // rhs_derivs.row(i) = jump of the i-th Cartesian partial derivative of body force f
    // rows = num_monomials(rhs_max_degree(method_order), 2)
    Eigen::MatrixX2d rhs_derivs;
};

struct StokesJumpData3D {
    Eigen::Vector3d  u_jump;
    Eigen::Vector3d  traction_jump;
    // rows = num_monomials(rhs_max_degree(method_order), 3)
    Eigen::MatrixX3d rhs_derivs;
};

} // namespace kfbim
