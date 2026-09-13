#include "src/support/correction/shared_field_space_3d.hpp"
#include "src/support/geometry/trace93_case_3d.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
using namespace kfbim::app3d;
namespace { void require(bool ok,const char* m){if(!ok)throw std::runtime_error(m);} }
int main(){try{
 for(int N:{32,64,128}) {
  auto p=make_trace93_case_3d("u","rotate");
  SharedFieldSpaceOptions3D opt;opt.h=2./N;
  SharedFieldSpace3D s(make_trace93_shared_field_geometry_3d(p),opt);
  require(s.coefficient_count()==(N==32?1243:N==64?3365:13687),"reference coefficient count");
  const auto& samples=s.surface_samples();
  require(samples.weights.minCoeff()>0,"positive surface quadrature");
  require(s.volume_samples().weights.minCoeff()>0,"positive volume quadrature");
  require(std::abs(samples.weights.sum()-p.surface.expected_area)<2e-12,"physical surface quadrature area");
  require(std::abs(s.volume_samples().weights.sum()-s.active_cells().size()*std::pow(s.spacing(),3))<2e-12,"physical active volume quadrature");
  require(s.evaluation_matrix({}).rows()==0,"empty transfer query is supported");
  for(std::size_t i=0;i<samples.reference_points.size();i+=101) {
   require((s.geometry().to_reference(samples.world_points[i])-samples.reference_points[i]).norm()<1e-14,"rigid point inverse");
   const auto jet=p.analysis_at(samples.patches[i],samples.analysis_uv[i].x(),samples.analysis_uv[i].y(),1);
   require((jet.d[0][0]-samples.world_points[i]).norm()<1e-13,"analysis sampling chart");
   require(jet.d[1][0].cross(jet.d[0][1]).normalized().dot(samples.world_normals[i])>1-1e-13,"surface outward normal");
  }
  std::vector<Eigen::Vector3d> x{{-.5,-.5,-.5},{-.5,-.5,0},{-.5,0,-.5},{-.5,0,0},{-.5-1e-12,0,0},{-.5+1e-12,0,0}};
  auto one=Eigen::VectorXd::Ones(s.coefficient_count());
  for(auto d:{SharedFieldDerivative3D::Value,SharedFieldDerivative3D::Dx,SharedFieldDerivative3D::Dy,SharedFieldDerivative3D::Dz,SharedFieldDerivative3D::Dxx,SharedFieldDerivative3D::Dyy,SharedFieldDerivative3D::Dzz,SharedFieldDerivative3D::Dxy,SharedFieldDerivative3D::Dxz,SharedFieldDerivative3D::Dyz,SharedFieldDerivative3D::Laplacian}) {
   auto E=s.evaluation_matrix(x,d);Eigen::VectorXd v=E*one;
   const int order=d==SharedFieldDerivative3D::Value?0:(d==SharedFieldDerivative3D::Dx||d==SharedFieldDerivative3D::Dy||d==SharedFieldDerivative3D::Dz?1:2);
   if(!order)v.array()-=1.;
   require(v.lpNorm<Eigen::Infinity>()*std::pow(s.spacing(),order)<1e-13,"cardinal partition/derivative sum");
   Eigen::VectorXd c(s.coefficient_count());for(int i=0;i<c.size();++i)c[i]=std::sin(.071*i);
   Eigen::VectorXd forward=E*c;auto reversed=x;std::reverse(reversed.begin(),reversed.end());Eigen::VectorXd backward=s.evaluation_matrix(reversed,d)*c;
   require((forward-backward.reverse()).norm()==0,"evaluation query-order independence");
   require(std::abs(forward[4]-forward[5])*std::pow(s.spacing(),order)<1e-8,"C2 knot continuity");
  }
  require(!s.contains(Eigen::Vector3d(10,10,10)),"outside coverage");
  bool threw=false;try{s.evaluation_matrix({Eigen::Vector3d(10,10,10)});}catch(const std::exception&){threw=true;}require(threw,"missing support rejected");
 }
 bool unsupported=false;try{make_trace93_shared_field_geometry_3d(make_trace93_case_3d("box","identity"));}catch(const std::invalid_argument&){unsupported=true;}require(unsupported,"unsupported geometry rejected");
 std::cout<<"shared field basis checks passed\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
