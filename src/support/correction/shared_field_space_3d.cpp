#include "src/support/correction/shared_field_space_3d.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace kfbim::app3d {
namespace {
std::array<int,3> cell(const Eigen::Vector3d&x,double H){
 std::array<int,3> c;for(int j=0;j<3;++j){double v=std::floor(x[j]/H);if(!std::isfinite(v)||std::abs(v)>std::numeric_limits<int>::max()-4)throw std::invalid_argument("nonfinite or excessive shared field coordinate");c[j]=static_cast<int>(v);}return c;
}
std::array<std::array<double,4>,3> cardinal(double t,double H){
 const double u=1-t,t2=t*t,t3=t2*t;
 return {{{{u*u*u/6,(3*t3-6*t2+4)/6,(-3*t3+3*t2+3*t+1)/6,t3/6}},
          {{-u*u/(2*H),(9*t2-12*t)/(6*H),(-9*t2+6*t+3)/(6*H),t2/(2*H)}},
          {{u/(H*H),(3*t-2)/(H*H),(1-3*t)/(H*H),t/(H*H)}}}};
}
}
SharedFieldSpace3D::SharedFieldSpace3D(SharedFieldGeometry3D geometry,SharedFieldSpaceOptions3D options):geometry_(std::move(geometry)),options_(options){
 if(!std::isfinite(options.h)||options.h<=0||!std::isfinite(options.ratio)||options.ratio<=0||!std::isfinite(options.width)||options.width<0||options.volume_quadrature_order!=3)throw std::invalid_argument("shared field requires positive h/ratio, nonnegative width and volume quadrature order 3");
 if(geometry_.rectangles.empty()||!geometry_.reference_half_extent.allFinite()||(geometry_.reference_half_extent.array()<=0).any()||!geometry_.rotation.allFinite()||!geometry_.translation.allFinite()||(geometry_.rotation.transpose()*geometry_.rotation-Eigen::Matrix3d::Identity()).norm()>1e-12||geometry_.rotation.determinant()<0)throw std::invalid_argument("shared field requires finite rigid reference geometry");
 const double H=spacing(),W=options.width*options.h;Eigen::Vector3d lim=geometry_.reference_half_extent.array()+W+H;Eigen::Vector3i lo,hi;for(int d=0;d<3;++d){lo[d]=static_cast<int>(std::floor(-lim[d]/H));hi[d]=static_cast<int>(std::ceil(lim[d]/H));}
 std::set<std::array<int,3>> lattice;
 for(int i=lo.x();i<hi.x();++i)for(int j=lo.y();j<hi.y();++j)for(int k=lo.z();k<hi.z();++k){Eigen::Vector3i c(i,j,k);if(geometry_.distance((c.cast<double>().array()+.5).matrix()*H)>W+std::sqrt(3.)*H/2)continue;cells_.push_back(c);cell_set_.insert({{i,j,k}});for(int a=-1;a<=2;++a)for(int b=-1;b<=2;++b)for(int d=-1;d<=2;++d)lattice.insert({{i+a,j+b,k+d}});}
 if(cells_.empty())throw std::invalid_argument("shared field has no active cells");
 for(const auto&c:lattice){indices_[c]=static_cast<int>(lattice_.size());lattice_.emplace_back(c[0],c[1],c[2]);}
 surface_=sample_shared_field_surface_3d(geometry_,H);
 const double g=std::sqrt(3./5);const double gx[]={(1-g)/2,.5,(1+g)/2};const double gw[]={5./18,4./9,5./18};std::vector<double> weights;weights.reserve(cells_.size()*27);volume_.reference_points.reserve(cells_.size()*27);
 for(const auto&c:cells_)for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k){volume_.reference_points.push_back((c.cast<double>()+Eigen::Vector3d(gx[i],gx[j],gx[k]))*H);weights.push_back(H*H*H*gw[i]*gw[j]*gw[k]);}
 volume_.weights=Eigen::Map<Eigen::VectorXd>(weights.data(),static_cast<Eigen::Index>(weights.size()));
}
bool SharedFieldSpace3D::contains(const Eigen::Vector3d&x)const{return cell_set_.count(cell(x,spacing()))!=0;}
std::vector<bool> SharedFieldSpace3D::contains(const std::vector<Eigen::Vector3d>&x)const{std::vector<bool> r;r.reserve(x.size());for(const auto&p:x)r.push_back(contains(p));return r;}
SharedFieldSparseMatrix3D SharedFieldSpace3D::evaluation_matrix(const std::vector<Eigen::Vector3d>&x,SharedFieldDerivative3D derivative)const{
 const double H=spacing();int d[3]={0,0,0};switch(derivative){case SharedFieldDerivative3D::Value:case SharedFieldDerivative3D::Laplacian:break;case SharedFieldDerivative3D::Dx:d[0]=1;break;case SharedFieldDerivative3D::Dy:d[1]=1;break;case SharedFieldDerivative3D::Dz:d[2]=1;break;case SharedFieldDerivative3D::Dxx:d[0]=2;break;case SharedFieldDerivative3D::Dyy:d[1]=2;break;case SharedFieldDerivative3D::Dzz:d[2]=2;break;case SharedFieldDerivative3D::Dxy:d[0]=d[1]=1;break;case SharedFieldDerivative3D::Dxz:d[0]=d[2]=1;break;case SharedFieldDerivative3D::Dyz:d[1]=d[2]=1;break;default:throw std::invalid_argument("invalid shared field derivative");}
 SharedFieldSparseMatrix3D out(static_cast<int>(x.size()),coefficient_count());out.reserve(static_cast<Eigen::Index>(64*x.size()));
 for(int row=0;row<static_cast<int>(x.size());++row){const auto c=cell(x[row],H);if(!cell_set_.count(c))throw std::out_of_range("shared field query outside active cells");std::array<std::array<std::array<double,4>,3>,3> b;for(int axis=0;axis<3;++axis)b[axis]=cardinal(x[row][axis]/H-c[axis],H);
  out.startVec(row);for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int k=0;k<4;++k){auto it=indices_.find({{c[0]+i-1,c[1]+j-1,c[2]+k-1}});if(it==indices_.end())throw std::out_of_range("shared field missing coefficient support");double v=b[0][d[0]][i]*b[1][d[1]][j]*b[2][d[2]][k];if(derivative==SharedFieldDerivative3D::Laplacian)v=b[0][2][i]*b[1][0][j]*b[2][0][k]+b[0][0][i]*b[1][2][j]*b[2][0][k]+b[0][0][i]*b[1][0][j]*b[2][2][k];out.insertBack(row,it->second)=v;}}
 out.finalize();out.makeCompressed();return out;
}
Eigen::MatrixXd shared_field_polynomial_3d(const std::vector<Eigen::Vector3d>&points,const Eigen::Vector3i&d){
 if((d.array()<0).any())throw std::invalid_argument("negative polynomial derivative");Eigen::MatrixXd result(points.size(),20);int col=0;
 for(int total=0;total<=3;++total)for(int i=0;i<=total;++i)for(int j=0;j<=total-i;++j){Eigen::Vector3i p(i,j,total-i-j);for(int row=0;row<static_cast<int>(points.size());++row){double v=1.;for(int k=0;k<3;++k){if(d[k]>p[k]){v=0;break;}for(int q=0;q<d[k];++q)v*=p[k]-q;v*=std::pow(points[row][k],p[k]-d[k]);}result(row,col)=v;}++col;}return result;
}
}
