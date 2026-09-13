#include "src/support/correction/shared_field_extension_3d.hpp"
#include <Eigen/SVD>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
namespace kfbim::app3d {
namespace {
void valid_sparse(const SharedFieldSparseMatrix3D&m){for(int i=0;i<m.nonZeros();++i)if(!std::isfinite(m.valuePtr()[i]))throw std::invalid_argument("shared field fit matrix contains nonfinite entries");}
double inf(const Eigen::VectorXd&v){return v.size()?v.lpNorm<Eigen::Infinity>():0.;}
}
SharedFieldFitMatrices3D make_shared_field_fit_matrices_3d(const SharedFieldSpace3D&s){
 SharedFieldFitMatrices3D m;m.h=s.options().h;const auto&sf=s.surface_samples();const auto&vf=s.volume_samples();m.surface_weights=sf.weights;m.volume_weights=vf.weights;
 m.trace=s.evaluation_matrix(sf.reference_points);m.normal=SharedFieldSparseMatrix3D(m.trace.rows(),m.trace.cols());
 const SharedFieldDerivative3D ders[]={SharedFieldDerivative3D::Dx,SharedFieldDerivative3D::Dy,SharedFieldDerivative3D::Dz};
 for(int d=0;d<3;++d){auto D=s.evaluation_matrix(sf.reference_points,ders[d]);for(int row=0;row<D.outerSize();++row)for(SharedFieldSparseMatrix3D::InnerIterator e(D,row);e;++e)e.valueRef()*=sf.reference_normals[row][d];m.normal+=D;}
 m.normal.makeCompressed();m.laplacian=s.evaluation_matrix(vf.reference_points,SharedFieldDerivative3D::Laplacian);m.surface_polynomial=shared_field_polynomial_3d(sf.reference_points);
 std::vector<Eigen::Vector3d> lattice;lattice.reserve(s.coefficient_count());for(const auto&j:s.coefficient_lattice())lattice.push_back(j.cast<double>()*s.spacing());
 m.lift_lattice=shared_field_polynomial_3d(lattice);for(int d=0;d<3;++d){Eigen::Vector3i der=Eigen::Vector3i::Zero();der[d]=2;m.lift_lattice-=s.spacing()*s.spacing()/6*shared_field_polynomial_3d(lattice,der);}return m;
}
SharedFieldExtension3D::SharedFieldExtension3D(SharedFieldFitMatrices3D m,SharedFieldFitOptions3D o):matrices_(std::move(m)),options_(o){
 const auto start=std::chrono::steady_clock::now();const auto& f=matrices_;const int ns=static_cast<int>(f.trace.rows()),nv=static_cast<int>(f.laplacian.rows()),nc=static_cast<int>(f.trace.cols());
 for(double v:{o.value_weight,o.normal_weight,o.pde_weight,o.ridge,o.lift_rcond,f.h})if(!std::isfinite(v)||v<=0)throw std::invalid_argument("shared field fit requires positive finite weights, ridge, cutoff and h");
 if(o.lift_rcond>=1||o.refinement_count!=2)throw std::invalid_argument("shared field fit requires cutoff below one and exactly two refinements");
 if(ns<=0||nv<=0||nc<=0||f.normal.rows()!=ns||f.normal.cols()!=nc||f.laplacian.cols()!=nc||f.surface_weights.size()!=ns||f.volume_weights.size()!=nv||f.surface_polynomial.rows()!=ns||f.surface_polynomial.cols()!=20||f.lift_lattice.rows()!=nc||f.lift_lattice.cols()!=20)throw std::invalid_argument("shared field fit matrix dimensions mismatch");
 if(!f.surface_weights.allFinite()||!f.volume_weights.allFinite()||(f.surface_weights.array()<=0).any()||(f.volume_weights.array()<=0).any()||!f.surface_polynomial.allFinite()||!f.lift_lattice.allFinite())throw std::invalid_argument("shared field fit requires finite matrices and positive quadrature weights");
 valid_sparse(f.trace);valid_sparse(f.normal);valid_sparse(f.laplacian);
 const Eigen::VectorXd sv=(f.surface_weights/f.surface_weights.sum()).array().sqrt(),vv=(f.volume_weights/f.volume_weights.sum()).array().sqrt();wa_=std::sqrt(o.value_weight)*sv;wb_=std::sqrt(o.normal_weight)*f.h*sv;wp_=std::sqrt(o.pde_weight)*f.h*f.h*vv;
 std::vector<Eigen::Triplet<double>> entries;entries.reserve(f.trace.nonZeros()+f.normal.nonZeros()+f.laplacian.nonZeros());
 auto append=[&](const SharedFieldSparseMatrix3D&B,const Eigen::VectorXd&w,int offset){for(int row=0;row<B.outerSize();++row)for(SharedFieldSparseMatrix3D::InnerIterator e(B,row);e;++e)entries.emplace_back(offset+row,e.col(),w[row]*e.value());};
 append(f.trace,wa_,0);append(f.normal,wb_,ns);append(f.laplacian,wp_,2*ns);A_.resize(2*ns+nv,nc);A_.setFromTriplets(entries.begin(),entries.end());entries.clear();entries.shrink_to_fit();
 scale_=Eigen::VectorXd::Zero(nc);for(int row=0;row<A_.outerSize();++row)for(SharedFieldSparseMatrix3D::InnerIterator e(A_,row);e;++e)scale_[e.col()]+=e.value()*e.value();scale_=scale_.array().sqrt().max(1e-25);
 for(int row=0;row<A_.outerSize();++row)for(SharedFieldSparseMatrix3D::InnerIterator e(A_,row);e;++e)e.valueRef()/=scale_[e.col()];
 Eigen::SparseMatrix<double> K=A_.transpose()*A_;for(int j=0;j<nc;++j)K.coeffRef(j,j)+=o.ridge;K.makeCompressed();
 Eigen::VectorXd gram_rows=Eigen::VectorXd::Zero(nc);for(int col=0;col<K.outerSize();++col)for(Eigen::SparseMatrix<double>::InnerIterator e(K,col);e;++e)gram_rows[e.row()]+=std::abs(e.value());gram_infinity_norm_=gram_rows.maxCoeff();
 factor_.compute(K);++setup_.factorization_count;
 if(factor_.info()!=Eigen::Success||!factor_.vectorD().allFinite()||(factor_.vectorD().array()<=0).any())throw std::runtime_error("shared field SimplicialLDLT failed or has nonpositive/nonfinite pivots");
 Eigen::MatrixXd weighted=f.surface_polynomial;for(int i=0;i<ns;++i)weighted.row(i)*=std::sqrt(f.surface_weights[i]);
 Eigen::JacobiSVD<Eigen::MatrixXd> svd(weighted,Eigen::ComputeThinU|Eigen::ComputeThinV);if(!svd.singularValues().allFinite())throw std::runtime_error("shared field polynomial lift SVD failed");Eigen::VectorXd inv=svd.singularValues();const double cutoff=o.lift_rcond*inv[0];for(int i=0;i<inv.size();++i)inv[i]=inv[i]>cutoff?1/inv[i]:0.;lift_map_=svd.matrixV()*inv.asDiagonal()*svd.matrixU().transpose();for(int i=0;i<ns;++i)lift_map_.col(i)*=std::sqrt(f.surface_weights[i]);
 setup_.trace_nnz=f.trace.nonZeros();setup_.normal_nnz=f.normal.nonZeros();setup_.laplacian_nnz=f.laplacian.nonZeros();setup_.matrix_nnz=A_.nonZeros();setup_.gram_nnz=K.nonZeros();setup_.factor_nnz=factor_.matrixL().nestedExpression().nonZeros()+nc;
 setup_.estimated_bytes=static_cast<std::size_t>(setup_.trace_nnz+setup_.normal_nnz+setup_.laplacian_nnz+setup_.matrix_nnz+setup_.gram_nnz+setup_.factor_nnz)*(sizeof(double)+sizeof(int))+static_cast<std::size_t>(f.surface_polynomial.size()+f.lift_lattice.size()+lift_map_.size()+scale_.size()+wa_.size()+wb_.size()+wp_.size())*sizeof(double);
 setup_.setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
Eigen::VectorXd SharedFieldExtension3D::target(const Eigen::VectorXd&a,const Eigen::VectorXd&b)const{
 if(a.size()!=wa_.size()||b.size()!=wb_.size()||!a.allFinite()||!b.allFinite())throw std::invalid_argument("shared field Cauchy input dimension or finite-value error");Eigen::VectorXd q=Eigen::VectorXd::Zero(A_.rows());q.head(a.size())=wa_.array()*a.array();q.segment(a.size(),b.size())=wb_.array()*b.array();return q;
}
Eigen::VectorXd SharedFieldExtension3D::solve(const Eigen::VectorXd&a,const Eigen::VectorXd&b)const{
 Eigen::VectorXd q=target(a,b);Eigen::VectorXd lift=matrices_.lift_lattice*(lift_map_*a);Eigen::VectorXd residual=q-A_*(scale_.array()*lift.array()).matrix();Eigen::VectorXd z=factor_.solve(A_.transpose()*residual);
 if(factor_.info()!=Eigen::Success||!z.allFinite())throw std::runtime_error("shared field fixed fit solve failed");
 for(int iteration=0;iteration<options_.refinement_count;++iteration){Eigen::VectorXd r=A_.transpose()*(residual-A_*z)-options_.ridge*z;Eigen::VectorXd correction=factor_.solve(r);if(factor_.info()!=Eigen::Success||!correction.allFinite())throw std::runtime_error("shared field fixed fit refinement failed");z+=correction;}
 const Eigen::VectorXd optimality=A_.transpose()*(A_*z-residual)+options_.ridge*z;
 const double denominator=gram_infinity_norm_*inf(z)+inf(A_.transpose()*residual);
 last_backward_residual_=denominator==0.?0.:inf(optimality)/denominator;
 Eigen::VectorXd result=lift+(z.array()/scale_.array()).matrix();if(!result.allFinite())throw std::runtime_error("shared field nonfinite fitted coefficients");return result;
}
SharedFieldFitDiagnostics3D SharedFieldExtension3D::diagnostics(const Eigen::VectorXd&c,const Eigen::VectorXd&a,const Eigen::VectorXd&b)const{
 Eigen::VectorXd q=target(a,b);if(c.size()!=scale_.size()||!c.allFinite())throw std::invalid_argument("shared field diagnostic coefficient error");const auto&m=matrices_;Eigen::VectorXd ra=m.trace*c-a,rb=m.normal*c-b,rp=m.laplacian*c;SharedFieldFitDiagnostics3D d;d.value_linf=inf(ra);d.normal_linf=inf(rb);d.pde_linf=inf(rp);d.value_rms=std::sqrt((m.surface_weights.array()*ra.array().square()).sum()/m.surface_weights.sum());d.normal_rms=std::sqrt((m.surface_weights.array()*rb.array().square()).sum()/m.surface_weights.sum());d.pde_rms=std::sqrt((m.volume_weights.array()*rp.array().square()).sum()/m.volume_weights.sum());
 Eigen::VectorXd lift=m.lift_lattice*(lift_map_*a);Eigen::VectorXd z=scale_.array()*(c-lift).array();Eigen::VectorXd t=q-A_*(scale_.array()*lift.array()).matrix();Eigen::VectorXd Az=A_*z;Eigen::VectorXd optimality=A_.transpose()*(Az-t)+options_.ridge*z;
 const double denominator=gram_infinity_norm_*inf(z)+inf(A_.transpose()*t);
 d.normalized_optimality=denominator==0.?0.:inf(optimality)/denominator;return d;
}
double SharedFieldExtension3D::last_solve_backward_residual()const{
 if(last_backward_residual_<0)throw std::logic_error("shared field backward residual requested before solve");return last_backward_residual_;
}
}
