#pragma once
#include "src/support/trace/kfbi_correction_backend_3d.hpp"
#include "src/support/trace/affine_trace_coordinates_3d.hpp"
#include "src/bulk_solvers/i_bulk_solver.hpp"
#include <vector>
namespace kfbim::app3d {
struct AffineKfbiTimings3D {
    double backend_seconds=0,poisson_seconds=0,observe_seconds=0;
    double affine_rhs_seconds=0,gmres_seconds=0,matvec_seconds=0,final_seconds=0;
    int poisson_calls=0,matvec_calls=0;
};
struct AffineKfbiEvaluation3D {
    std::uint64_t evaluation_id=0;
    Eigen::VectorXd grid_solution;
    TracePair3D raw_trace,equation_trace;
    CorrectionEvaluation3D correction;
    AffineKfbiTimings3D timings;
};
struct AffineKfbiSolveOptions3D {
    bool neumann=false;
    ExteriorTarget3D target=ExteriorTarget3D::RawTrace;
    double relative_tolerance=2e-10;
    int restart=100,max_iterations=800;
};
struct AffineKfbiSolveResult3D {
    Eigen::VectorXd raw_coefficients,reduced_coordinates,projected_rhs,projected_residual;
    AffineKfbiEvaluation3D evaluation;
    CorrectionDiagnosticSnapshot3D final_snapshot;
    double projected_relative_residual=0;
    int iterations=0;
    bool gmres_converged=false;
    std::vector<double> recursive_residuals;
    AffineKfbiTimings3D timings;
};
AffineKfbiEvaluation3D evaluate_affine_kfbi_3d(ICorrectionBackend3D&,const ILaplaceBulkSolver3D&,
    const Eigen::VectorXd&,ApplyPart3D,ExteriorTarget3D);
AffineKfbiSolveResult3D solve_affine_kfbi_3d(ICorrectionBackend3D&,const IAffineTraceCoordinates3D&,
    const ILaplaceBulkSolver3D&,const AffineKfbiSolveOptions3D&);
}
