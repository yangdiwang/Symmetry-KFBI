#pragma once

#include <Eigen/Dense>
#include "jump_data.hpp"
#include "local_poly.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Abstract local Cauchy solver
//
// Two-phase interface:
//   1. fit()      — solve the local Cauchy problem at one interface point,
//                   return the correction function as a LocalPoly (stored coefficients)
//   2. evaluate() — evaluate the correction polynomial at an arbitrary nearby point
//
// fit() is called once per interface quadrature point per GMRES iteration.
// evaluate() is called many times per poly: once per nearby bulk node (Corrector)
// and once at the interface point itself (Interpolator).
//
// method_order() determines expected rhs_derivs length in JumpData (validated in fit()).
// ---------------------------------------------------------------------------

class ILaplaceLocalSolver2D {
public:
    virtual ~ILaplaceLocalSolver2D() = default;

    virtual LocalPoly2D fit(const LaplaceJumpData2D& jumps,
                            Eigen::Vector2d          iface_pt,
                            Eigen::Vector2d          normal) const = 0;

    virtual double evaluate(const LocalPoly2D& poly,
                            Eigen::Vector2d    pt) const = 0;

    virtual int method_order() const = 0;
};

class ILaplaceLocalSolver3D {
public:
    virtual ~ILaplaceLocalSolver3D() = default;

    virtual LocalPoly3D fit(const LaplaceJumpData3D& jumps,
                            Eigen::Vector3d          iface_pt,
                            Eigen::Vector3d          normal) const = 0;

    virtual double evaluate(const LocalPoly3D& poly,
                            Eigen::Vector3d    pt) const = 0;

    virtual int method_order() const = 0;
};

// Stokes: fit() returns velocity + pressure polynomials together.
// evaluate() returns the velocity correction (2- or 3-vector) at a point.
// Pressure correction is only needed at the interface itself (for Interpolator).
class IStokesLocalSolver2D {
public:
    virtual ~IStokesLocalSolver2D() = default;

    virtual StokesLocalPoly2D fit(const StokesJumpData2D& jumps,
                                  Eigen::Vector2d         iface_pt,
                                  Eigen::Vector2d         normal) const = 0;

    virtual Eigen::Vector2d eval_velocity(const StokesLocalPoly2D& poly,
                                          Eigen::Vector2d          pt) const = 0;

    virtual double eval_pressure(const StokesLocalPoly2D& poly,
                                 Eigen::Vector2d          pt) const = 0;

    virtual int method_order() const = 0;
};

class IStokesLocalSolver3D {
public:
    virtual ~IStokesLocalSolver3D() = default;

    virtual StokesLocalPoly3D fit(const StokesJumpData3D& jumps,
                                  Eigen::Vector3d         iface_pt,
                                  Eigen::Vector3d         normal) const = 0;

    virtual Eigen::Vector3d eval_velocity(const StokesLocalPoly3D& poly,
                                          Eigen::Vector3d          pt) const = 0;

    virtual double eval_pressure(const StokesLocalPoly3D& poly,
                                 Eigen::Vector3d          pt) const = 0;

    virtual int method_order() const = 0;
};

} // namespace kfbim
