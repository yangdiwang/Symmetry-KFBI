#pragma once
#include "src/support/correction/shared_field_space_3d.hpp"
#include <Eigen/SparseCholesky>
namespace kfbim::app3d {
struct SharedFieldFitMatrices3D {
 SharedFieldSparseMatrix3D trace,normal,laplacian;
 Eigen::VectorXd surface_weights,volume_weights;
 Eigen::MatrixXd surface_polynomial,lift_lattice;
 double h=0.;
};
struct SharedFieldFitOptions3D {
 double value_weight=1.,normal_weight=1.,pde_weight=1.,ridge=1e-12,lift_rcond=1e-13;
 int refinement_count=2;
};
struct SharedFieldFitDiagnostics3D {
 double value_linf=0.,normal_linf=0.,pde_linf=0.,value_rms=0.,normal_rms=0.,pde_rms=0.,normalized_optimality=0.;
};
struct SharedFieldFitSetupDiagnostics3D {
 Eigen::Index trace_nnz=0,normal_nnz=0,laplacian_nnz=0,matrix_nnz=0,gram_nnz=0,factor_nnz=0;
 std::size_t estimated_bytes=0;
 int factorization_count=0;
 double setup_seconds=0.;
};
SharedFieldFitMatrices3D make_shared_field_fit_matrices_3d(const SharedFieldSpace3D& space);
class SharedFieldExtension3D {
public:
 SharedFieldExtension3D(SharedFieldFitMatrices3D matrices,SharedFieldFitOptions3D options={});
 Eigen::VectorXd solve(const Eigen::VectorXd& value_jump,const Eigen::VectorXd& normal_jump)const;
 SharedFieldFitDiagnostics3D diagnostics(const Eigen::VectorXd& coefficients,const Eigen::VectorXd& value_jump,const Eigen::VectorXd& normal_jump)const;
 const SharedFieldFitSetupDiagnostics3D& setup_diagnostics()const{return setup_;}
 const SharedFieldFitOptions3D& options()const{return options_;}
 const SharedFieldFitMatrices3D& matrices()const{return matrices_;}
 const SharedFieldSparseMatrix3D& scaled_matrix()const{return A_;}
 const Eigen::VectorXd& column_scale()const{return scale_;}
 double last_solve_backward_residual()const;
private:
 Eigen::VectorXd target(const Eigen::VectorXd&a,const Eigen::VectorXd&b)const;
 SharedFieldFitMatrices3D matrices_;
 SharedFieldFitOptions3D options_;
 SharedFieldFitSetupDiagnostics3D setup_;
 SharedFieldSparseMatrix3D A_;
 Eigen::VectorXd scale_,wa_,wb_,wp_;
 Eigen::MatrixXd lift_map_;
 double gram_infinity_norm_=0.;
 mutable double last_backward_residual_=-1.;
 Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>,Eigen::Lower,Eigen::AMDOrdering<int>> factor_;
};
}
