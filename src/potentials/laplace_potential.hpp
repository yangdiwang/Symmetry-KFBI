#pragma once

#include <vector>
#include <Eigen/Dense>
#include "../local_cauchy/jump_data.hpp"
#include "../local_cauchy/laplace_corner_patch_solver_2d.hpp"

namespace kfbim {

// Forward declarations
class ILaplaceSpread2D;
class ILaplaceBulkSolver2D;
class ILaplaceRestrict2D;
class ILaplaceSpread3D;
class ILaplaceBulkSolver3D;
class ILaplaceRestrict3D;

struct LaplacePotentialEvalResult2D {
    Eigen::VectorXd u_bulk;    // singular-removed solution at grid nodes
    Eigen::VectorXd u_physical; // u_bulk with corner singular branches restored
    Eigen::VectorXd u_avg;     // physical averaged trace (u_int + u_ext)/2
    Eigen::VectorXd un_avg;    // physical averaged normal derivative
    std::vector<CornerPatchCorrectionData2D> corner_patch_corrections;
};

struct LaplacePotentialEvalProfile2D {
    int    calls = 0;
    double rhs_copy_sec = 0.0;
    double spread_sec = 0.0;
    double bulk_solve_sec = 0.0;
    double restrict_sec = 0.0;
    double average_sec = 0.0;
    double total_sec = 0.0;
};

struct LaplacePotentialEvalResult3D {
    Eigen::VectorXd u_bulk;
    Eigen::VectorXd u_avg;
    Eigen::VectorXd un_avg;
};

// ============================================================================
// LaplacePotentialEval2D — general 2D KFBI pipeline and potential operators.
//
// evaluate() runs one Spread -> BulkSolve -> Restrict pass for arbitrary jumps
// and bulk RHS. Restrict returns averaged trace/flux values directly.
//
// The specialized potential helpers are thin wrappers around evaluate():
//   D[φ]: [u]=φ, [∂ₙu]=0, f=0
//   S[ψ]: [u]=0, [∂ₙu]=ψ, f=0
//   N[q]: [u]=0, [∂ₙu]=0, [f]=q
// ============================================================================

class LaplacePotentialEval2D {
public:
    LaplacePotentialEval2D(const ILaplaceSpread2D&     spread,
                           const ILaplaceBulkSolver2D& bulk_solver,
                           const ILaplaceRestrict2D&   restrict_op);

    int problem_size() const;
    double arc_h_ratio() const;

    void reset_profile() const;
    LaplacePotentialEvalProfile2D profile() const;

    LaplacePotentialEvalResult2D evaluate(
        const std::vector<LaplaceJumpData2D>& jumps,
        const Eigen::VectorXd&                f_bulk) const;

    // ── Double-layer potential D[φ] ──────────────────────────────────────
    // [u]=phi, [∂ₙu]=0, f=0
    void eval_double_layer(const Eigen::VectorXd& phi,
                           Eigen::VectorXd&       K_phi,
                           Eigen::VectorXd&       H_phi) const;

    // ── Single-layer potential S[ψ] ──────────────────────────────────────
    // [u]=0, [∂ₙu]=psi, f=0
    void eval_single_layer(const Eigen::VectorXd& psi,
                           Eigen::VectorXd&       S_psi,
                           Eigen::VectorXd&       Kt_psi) const;

    // ── Combined D[phi] + S[psi] evaluation ─────────────────────────────
    // Runs one potential solve with [u]=phi and [∂ₙu]=psi.
    void eval_layer_combination(const Eigen::VectorXd& phi,
                                const Eigen::VectorXd& psi,
                                Eigen::VectorXd&       trace_avg,
                                Eigen::VectorXd&       normal_avg) const;

    // ── Newton potential N[q] ────────────────────────────────────────────
    // [u]=0, [∂ₙu]=0, [f]=q
    void eval_newton(const Eigen::VectorXd& q,
                     Eigen::VectorXd&       N_q,
                     Eigen::VectorXd&       Nn_q) const;

private:
    const ILaplaceSpread2D&     spread_;
    const ILaplaceBulkSolver2D& bulk_solver_;
    const ILaplaceRestrict2D&   restrict_op_;
    double                      arc_h_ratio_;
    mutable LaplacePotentialEvalProfile2D profile_;
};

// ============================================================================
// LaplacePotentialEval3D — same interface for 3D
// ============================================================================

class LaplacePotentialEval3D {
public:
    LaplacePotentialEval3D(const ILaplaceSpread3D&     spread,
                           const ILaplaceBulkSolver3D& bulk_solver,
                           const ILaplaceRestrict3D&   restrict_op);

    int problem_size() const;

    LaplacePotentialEvalResult3D evaluate(
        const std::vector<LaplaceJumpData3D>& jumps,
        const Eigen::VectorXd&                f_bulk) const;

    void eval_double_layer(const Eigen::VectorXd& phi,
                           Eigen::VectorXd&       K_phi,
                           Eigen::VectorXd&       H_phi) const;

    void eval_single_layer(const Eigen::VectorXd& psi,
                           Eigen::VectorXd&       S_psi,
                           Eigen::VectorXd&       Kt_psi) const;

    void eval_layer_combination(const Eigen::VectorXd& phi,
                                const Eigen::VectorXd& psi,
                                Eigen::VectorXd&       trace_avg,
                                Eigen::VectorXd&       normal_avg) const;

    void eval_newton(const Eigen::VectorXd& q,
                     Eigen::VectorXd&       N_q,
                     Eigen::VectorXd&       Nn_q) const;

private:
    const ILaplaceSpread3D&     spread_;
    const ILaplaceBulkSolver3D& bulk_solver_;
    const ILaplaceRestrict3D&   restrict_op_;
};

} // namespace kfbim
