#pragma once

#include <vector>
#include <Eigen/Dense>
#include "../operators/i_kfbi_operator.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Abstract outer solver (Layer 4)
//
// Solves the linear system  A * x = rhs  matrix-free, where A is an
// IKFBIOperator.  Concrete implementations: GMRES, BiCGSTAB.
//
// solve() may use x as an initial guess on entry (warm start); the solver
// must not rely on x being zeroed.  Returns the number of iterations taken.
// residuals() is valid after solve() returns. Implementations should document
// whether this history is recorded per step or per restart cycle.
// ---------------------------------------------------------------------------

class IOuterSolver {
public:
    virtual ~IOuterSolver() = default;

    virtual int solve(const IKFBIOperator& op,
                      const Eigen::VectorXd& rhs,
                      Eigen::VectorXd&       x) = 0;

    virtual const std::vector<double>& residuals() const = 0;

    virtual bool converged() const = 0;
};

} // namespace kfbim
