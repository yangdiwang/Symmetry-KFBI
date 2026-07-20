#pragma once

#include <Eigen/Dense>

namespace kfbim {

// ---------------------------------------------------------------------------
// Abstract KFBIM interface operator (Layer 3)
//
// One call to apply() performs a full interface-to-interface map:
//   Spread  →  correct bulk RHS  →  BulkSolve  →  Restrict
//
// The flat-vector API keeps Layer 4 (GMRES) PDE- and dimension-agnostic.
// Concrete subclasses handle packing/unpacking of PDE-specific jump unknowns.
//
// Implementations include problem wrappers such as LaplaceBvp2D and
// LaplaceTransmission2D, plus PDE-specific operator scaffolding.
// ---------------------------------------------------------------------------

class IKFBIOperator {
public:
    virtual ~IKFBIOperator() = default;

    // Matrix-free apply: y = A * x.
    // Both x and y are length problem_size().
    virtual void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const = 0;

    // Length of the flat unknown / result vector.
    virtual int problem_size() const = 0;
};

} // namespace kfbim
