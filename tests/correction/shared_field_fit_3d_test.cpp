#include "src/support/correction/shared_field_extension_3d.hpp"
#include "src/support/geometry/trace93_case_3d.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
using namespace kfbim::app3d;
namespace {void require(bool v,const char*m){if(!v)throw std::runtime_error(m);}}
int main(){try{
 SharedFieldSpaceOptions3D so;so.h=2./32;
 SharedFieldSpace3D s(make_trace93_shared_field_geometry_3d(make_trace93_case_3d("u","rotate")),so);
 auto matrices=make_shared_field_fit_matrices_3d(s);
 SharedFieldExtension3D fit(matrices);
 const auto& sx=s.surface_samples();
 Eigen::VectorXd zero=Eigen::VectorXd::Zero(sx.weights.size());require(fit.solve(zero,zero).norm()==0,"zero input exact linear origin");require(fit.last_solve_backward_residual()==0,"zero system backward residual convention");
 Eigen::VectorXd aa,bb,cc;
 for(int degree=0;degree<=3;++degree){
  Eigen::VectorXd a(sx.weights.size()),b(a.size());
  auto value=[degree](const Eigen::Vector3d&x){return degree==0?1.:degree==1?x.x()+2*x.y()-x.z():degree==2?x.x()*x.x()-x.y()*x.y():x.x()*x.x()*x.x()-3*x.x()*x.y()*x.y();};
  auto grad=[degree](const Eigen::Vector3d&x)->Eigen::Vector3d{return degree==0?Eigen::Vector3d(0,0,0):degree==1?Eigen::Vector3d(1,2,-1):degree==2?Eigen::Vector3d(2*x.x(),-2*x.y(),0):Eigen::Vector3d(3*x.x()*x.x()-3*x.y()*x.y(),-6*x.x()*x.y(),0);};
  for(int i=0;i<a.size();++i){a[i]=value(sx.reference_points[i]);b[i]=grad(sx.reference_points[i]).dot(sx.reference_normals[i]);}
  Eigen::VectorXd c=fit.solve(a,b);auto d=fit.diagnostics(c,a,b);
  require(d.value_linf<1e-10&&so.h*d.normal_linf<1e-10&&so.h*so.h*d.pde_linf<1e-10,"actual harmonic P0-P3 fit");
  require(fit.last_solve_backward_residual()<1e-11,"regularized backward residual before coefficient rounding");
  Eigen::VectorXd volume=s.evaluation_matrix(s.volume_samples().reference_points)*c;
  double e=0;for(int i=0;i<volume.size();++i)e=std::max(e,std::abs(volume[i]-value(s.volume_samples().reference_points[i])));require(e<1e-10,"volume polynomial reproduction after solve");
  std::cout<<"P"<<degree<<" value="<<d.value_linf<<" h*normal="<<so.h*d.normal_linf<<" h2*pde="<<so.h*so.h*d.pde_linf<<" volume="<<e<<" backward="<<fit.last_solve_backward_residual()<<'\n';
  aa=a;bb=b;cc=c;
 }
 Eigen::VectorXd a2(aa.size()),b2(bb.size());for(int i=0;i<aa.size();++i){a2[i]=std::sin(.113*i);b2[i]=std::cos(.097*i);}
 Eigen::VectorXd c2=fit.solve(a2,b2);auto d=fit.diagnostics(c2,a2,b2);
 require(d.value_rms>1e-3,"incompatible Cauchy fit retains nonzero residual");
 Eigen::VectorXd mixed=fit.solve(.7*aa-.3*a2,.7*bb-.3*b2);
 const Eigen::VectorXd first=.7*cc,second=-.3*c2;
 const auto linf=[](const Eigen::VectorXd& v){return v.lpNorm<Eigen::Infinity>();};
 const double linearity_error=linf(mixed-first-second)/std::max({1.,linf(mixed),linf(first),linf(second)});
 require(linearity_error<=1e-10,"fixed solver normalized Linf linearity");
 std::cout<<"fit_linearity_normalized_linf="<<linearity_error<<'\n';
 require(fit.setup_diagnostics().factorization_count==1,"one fixed factorization");
 // A visible ridge still preserves a harmonic lift exactly. Penalizing alpha
 // itself instead of alpha-lift would fail this test by a large margin.
 SharedFieldFitOptions3D visible_ridge;visible_ridge.ridge=.1;
 SharedFieldExtension3D centered_ridge(matrices,visible_ridge);
 Eigen::VectorXd centered=centered_ridge.solve(aa,bb);
 require((matrices.trace*centered-aa).lpNorm<Eigen::Infinity>()<1e-11,"ridge centered on polynomial lift");
 SharedFieldFitOptions3D bad;bad.ridge=-1;bool threw=false;try{SharedFieldExtension3D f(matrices,bad);}catch(const std::exception&){threw=true;}require(threw,"invalid ridge rejected");
 bad=SharedFieldFitOptions3D{};bad.normal_weight=0;threw=false;try{SharedFieldExtension3D f(matrices,bad);}catch(const std::exception&){threw=true;}require(threw,"invalid weight rejected");
 matrices.surface_weights[0]=-1;threw=false;try{SharedFieldExtension3D f(matrices);}catch(const std::exception&){threw=true;}require(threw,"invalid quadrature rejected");
 const auto& setup=fit.setup_diagnostics();
 std::cout<<"shared field actual fit checks passed; optimality="<<d.normalized_optimality<<" matrix_nnz="<<setup.matrix_nnz<<" gram_nnz="<<setup.gram_nnz<<" factor_nnz="<<setup.factor_nnz<<" estimated_bytes="<<setup.estimated_bytes<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
