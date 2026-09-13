#include "src/support/correction/shared_field_geometry_3d.hpp"
#include "src/support/geometry/trace93_case_3d.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace kfbim::app3d {
double SharedFieldGeometry3D::distance(const Eigen::Vector3d& x)const {
 double best=std::numeric_limits<double>::infinity();
 for(const auto&r:rectangles){double d=std::pow(x[r.axis]-r.value,2);for(int j=0;j<2;++j){const double t=x[r.axes[j]];const double e=t-std::max(r.limits[j][0],std::min(r.limits[j][1],t));d+=e*e;}best=std::min(best,d);}return std::sqrt(best);
}
SharedFieldGeometry3D make_trace93_shared_field_geometry_3d(const Trace93Case3D&p){
 if(p.geometry_name!="u")throw std::invalid_argument("shared field supports only Trace93 U prism");
 SharedFieldGeometry3D g;g.rotation=p.transform.rotation();g.translation=p.transform.forward_point(Eigen::Vector3d::Zero());
 for(std::size_t pid=0;pid<p.analysis_patches.size();++pid){
  const auto&m=p.analysis_patches[pid];if(m.kind!=SurfaceAnalysisChartKind3D::Affine||!m.planar)throw std::invalid_argument("shared field requires affine rectangular patches");
  SharedFieldRectangle3D r;r.patch=static_cast<int>(pid);r.analysis_origin=m.local_origin;r.analysis_du=m.du;r.analysis_dv=m.dv;
  r.normal=m.du.cross(m.dv).normalized();Eigen::Index ax;r.normal.cwiseAbs().maxCoeff(&ax);r.axis=static_cast<int>(ax);
  if(std::abs(std::abs(r.normal[r.axis])-1)>1e-13||std::abs(m.du.dot(m.dv))>1e-13)throw std::invalid_argument("shared field reference rectangles must be axis aligned and orthogonal");
  int j=0;for(int k=0;k<3;++k)if(k!=r.axis)r.axes[j++]=k;
  r.value=m.local_origin[r.axis];Eigen::Vector3d lo=m.local_origin,hi=lo;
  for(int u=0;u<2;++u)for(int v=0;v<2;++v){Eigen::Vector3d x=m.local_origin+u*m.du+v*m.dv;lo=lo.cwiseMin(x);hi=hi.cwiseMax(x);g.reference_half_extent=g.reference_half_extent.cwiseMax(x.cwiseAbs());}
  for(int k=0;k<2;++k)r.limits[k]=Eigen::Vector2d(lo[r.axes[k]],hi[r.axes[k]]);
  g.rectangles.push_back(r);
 }
 if(g.rectangles.empty())throw std::invalid_argument("shared field empty surface");return g;
}
SharedFieldSurfaceSamples3D sample_shared_field_surface_3d(const SharedFieldGeometry3D&g,double H){
 if(!std::isfinite(H)||H<=0)throw std::invalid_argument("shared field spacing must be positive");
 const double gx[]={-.8611363115940525752,-.3399810435848562648,.3399810435848562648,.8611363115940525752};
 const double gw[]={.3478548451374538574,.6521451548625461426,.6521451548625461426,.3478548451374538574};
 SharedFieldSurfaceSamples3D s;std::vector<double> weights;
 for(const auto&r:g.rectangles){
  std::array<std::vector<double>,2> pts,ws;
  for(int d=0;d<2;++d){const double lo=r.limits[d][0],hi=r.limits[d][1];std::vector<double> breaks{lo};for(int k=static_cast<int>(std::floor(lo/H))+1;k<static_cast<int>(std::ceil(hi/H));++k)if(k*H>lo+1e-13&&k*H<hi-1e-13)breaks.push_back(k*H);breaks.push_back(hi);
   for(std::size_t k=1;k<breaks.size();++k){const double half=(breaks[k]-breaks[k-1])/2,mid=(breaks[k]+breaks[k-1])/2;for(int q=0;q<4;++q){pts[d].push_back(mid+half*gx[q]);ws[d].push_back(half*gw[q]);}}}
  for(std::size_t i=0;i<pts[0].size();++i)for(std::size_t j=0;j<pts[1].size();++j){Eigen::Vector3d x;x[r.axis]=r.value;x[r.axes[0]]=pts[0][i];x[r.axes[1]]=pts[1][j];s.reference_points.push_back(x);s.world_points.push_back(g.to_world(x));s.reference_normals.push_back(r.normal);s.world_normals.push_back(g.normal_to_world(r.normal));s.patches.push_back(r.patch);Eigen::Vector3d dx=x-r.analysis_origin;s.analysis_uv.emplace_back(dx.dot(r.analysis_du)/r.analysis_du.squaredNorm(),dx.dot(r.analysis_dv)/r.analysis_dv.squaredNorm());weights.push_back(ws[0][i]*ws[1][j]);}
 }
 s.weights=Eigen::Map<Eigen::VectorXd>(weights.data(),static_cast<Eigen::Index>(weights.size()));return s;
}
}
