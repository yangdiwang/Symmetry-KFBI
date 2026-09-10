#include "src/support/cauchy/extended_tubular_cauchy_3d.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfbim::app3d {
namespace {
using Polynomial = Eigen::Matrix<double,20,1>;
constexpr int factorial(int n) { return n < 2 ? 1 : n == 2 ? 2 : 6; }
constexpr int choose(int n,int k) { return factorial(n)/(factorial(k)*factorial(n-k)); }
int index(int a,int b,int c)
{
    const auto& p=cubic_cauchy_powers_3d();
    for(int i=0;i<20;++i) if(p[i].s==a && p[i].t==b && p[i].r==c) return i;
    throw std::invalid_argument("Invalid tubular polynomial exponent");
}
Polynomial product(const Polynomial& a,const Polynomial& b)
{
    Polynomial c=Polynomial::Zero();const auto& p=cubic_cauchy_powers_3d();
    for(int i=0;i<20;++i) if(a[i]!=0.0) for(int j=0;j<20;++j)
        if(b[j]!=0.0 && p[i].degree()+p[j].degree()<=3)
            c[index(p[i].s+p[j].s,p[i].t+p[j].t,p[i].r+p[j].r)]+=a[i]*b[j];
    return c;
}
// Analytic Cox-de Boor derivatives on a fixed exterior knot span. Choosing
// the span at the clamped parameter does NOT clamp the evaluation parameter.
std::array<std::vector<double>,4> extended_basis(
    const geometry::NurbsBasis1D& basis,double x,int order)
{
    const auto& knots=basis.knots();const int degree=basis.degree();
    const int span=basis.find_span(std::clamp(x,basis.domain_start(),basis.domain_end()));
    std::array<std::vector<double>,4> out;
    auto eval=[&](auto&& self,int i,int p,int derivative)->double {
        if(derivative>p || i<0 || i+p+1>=static_cast<int>(knots.size())) return 0.0;
        if(p==0) return i==span?1.0:0.0;
        const double dl=knots[i+p]-knots[i],dr=knots[i+p+1]-knots[i+1];
        if(derivative>0)
            return (dl==0.0?0.0:p*self(self,i,p-1,derivative-1)/dl)
                 -(dr==0.0?0.0:p*self(self,i+1,p-1,derivative-1)/dr);
        return (dl==0.0?0.0:(x-knots[i])*self(self,i,p-1,0)/dl)
             +(dr==0.0?0.0:(knots[i+p+1]-x)*self(self,i+1,p-1,0)/dr);
    };
    for(int d=0;d<=order;++d) {
        out[d].resize(degree+1);
        for(int j=0;j<=degree;++j) out[d][j]=eval(eval,span-degree+j,degree,d);
    }
    return out;
}
std::array<Eigen::Vector3d,6> normal_derivatives(const NativeSurfaceCubicParameterJet3D& g)
{
    const auto& d=g.lower;const Eigen::Vector3d W=d.x_u.cross(d.x_v);
    const double length=W.norm();if(!(length>0)) throw std::runtime_error("Degenerate tubular normal");
    std::array<Eigen::Vector3d,6> n;n[0]=W/length;
    const std::array<Eigen::Vector3d,2> wi{{d.x_uu.cross(d.x_v)+d.x_u.cross(d.x_uv),
        d.x_uv.cross(d.x_v)+d.x_u.cross(d.x_vv)}};
    double li[2];for(int i=0;i<2;++i) {li[i]=n[0].dot(wi[i]);n[i+1]=(wi[i]-n[0]*li[i])/length;}
    const std::array<Eigen::Vector3d,3> wij{{g.x_uuu.cross(d.x_v)+2*d.x_uu.cross(d.x_uv)+d.x_u.cross(g.x_uuv),
        g.x_uuv.cross(d.x_v)+d.x_uu.cross(d.x_vv)+d.x_u.cross(g.x_uvv),
        g.x_uvv.cross(d.x_v)+2*d.x_uv.cross(d.x_vv)+d.x_u.cross(g.x_vvv)}};
    for(int k=0;k<3;++k) {
        const int i=k==2?1:0,j=k==0?0:1;
        const double lij=(wi[i].dot(wi[j])+W.dot(wij[k])-li[i]*li[j])/length;
        n[k+3]=(wij[k]-n[i+1]*li[j]-n[j+1]*li[i]-n[0]*lij)/length;
    }
    return n;
}
} // namespace

static NativeSurfaceCubicParameterJet3D extended_parameter_jet(
    const geometry3d::NurbsSurfacePatch3D& patch,double u,double v,int order)
{
    if(!std::isfinite(u)||!std::isfinite(v)) throw std::invalid_argument("Nonfinite extended parameters");
    const double su=patch.domain_end_u()-patch.domain_start_u(),sv=patch.domain_end_v()-patch.domain_start_v();
    const double a=patch.domain_start_u()+su*u,b=patch.domain_start_v()+sv*v;
    const auto& ub=patch.basis_u();const auto& vb=patch.basis_v();
    const auto iu=ub.active_basis_indices(std::clamp(a,ub.domain_start(),ub.domain_end()));
    const auto iv=vb.active_basis_indices(std::clamp(b,vb.domain_start(),vb.domain_end()));
    const auto bu=extended_basis(ub,a,order),bv=extended_basis(vb,b,order);
    Eigen::Vector3d numerator[4][4],derivative[4][4];double denominator[4][4]{};
    for(int i=0;i<=order;++i) for(int j=0;j<=order-i;++j) {
        numerator[i][j]=Eigen::Vector3d::Zero();
        for(std::size_t k=0;k<iu.size();++k) for(std::size_t l=0;l<iv.size();++l) {
            const double w=bu[i][k]*bv[j][l]*patch.weights()[iu[k]][iv[l]];
            numerator[i][j]+=w*patch.control_net()[iu[k]][iv[l]];denominator[i][j]+=w;
        }
    }
    if(!std::isfinite(denominator[0][0])||std::abs(denominator[0][0])<64*std::numeric_limits<double>::epsilon())
        throw std::runtime_error("Singular extended NURBS weight");
    for(int total=0;total<=order;++total) for(int i=0;i<=total;++i) {
        const int j=total-i;Eigen::Vector3d r=numerator[i][j];
        for(int k=0;k<=i;++k) for(int l=0;l<=j;++l) if(k+l>0)
            r-=choose(i,k)*choose(j,l)*denominator[k][l]*derivative[i-k][j-l];
        derivative[i][j]=r/denominator[0][0];
        if(!derivative[i][j].allFinite()) throw std::runtime_error("Nonfinite extended NURBS jet");
    }
    NativeSurfaceCubicParameterJet3D g;auto& d=g.lower;
    d.point=derivative[0][0];d.x_u=su*derivative[1][0];d.x_v=sv*derivative[0][1];
    if(order>=2) {d.x_uu=su*su*derivative[2][0];d.x_uv=su*sv*derivative[1][1];d.x_vv=sv*sv*derivative[0][2];}
    const Eigen::Vector3d normal=d.x_u.cross(d.x_v);const double area=normal.norm();
    if(!(area>0.0)||!std::isfinite(area)) throw std::runtime_error("Degenerate extended NURBS tangent");
    d.normal=normal/area;
    if(order>=3) {g.x_uuu=su*su*su*derivative[3][0];g.x_uuv=su*su*sv*derivative[2][1];
        g.x_uvv=su*sv*sv*derivative[1][2];g.x_vvv=sv*sv*sv*derivative[0][3];}
    return g;
}

NativeSurfaceCubicParameterJet3D extended_nurbs_parameter_jet_3d(
    const geometry3d::NurbsSurfacePatch3D& patch,double u,double v)
{ return extended_parameter_jet(patch,u,v,3); }

ExtendedTubularProjection3D project_extended_tubular_3d(
    const geometry3d::NurbsSurfacePatch3D& patch,const Eigen::Vector3d& target,
    double u,double v,int maximum_iterations,double step_tolerance)
{
    if(!target.allFinite()||!std::isfinite(u)||!std::isfinite(v)||maximum_iterations<1||!(step_tolerance>0))
        throw std::invalid_argument("Invalid extended tubular projection input");
    ExtendedTubularProjection3D out;
    for(int iteration=0;iteration<maximum_iterations;++iteration) {
        const auto d=extended_parameter_jet(patch,u,v,1).lower;
        const Eigen::Vector3d e=d.point-target;
        const double aa=d.x_u.squaredNorm(),ab=d.x_u.dot(d.x_v),bb=d.x_v.squaredNorm();
        const double determinant=aa*bb-ab*ab;
        if(!(determinant>1e-28)||!std::isfinite(determinant)) throw std::runtime_error("Singular tubular projection metric");
        const double f=e.dot(d.x_u),g=e.dot(d.x_v);
        Eigen::Vector2d step((-bb*f+ab*g)/determinant,(ab*f-aa*g)/determinant);
        const double norm=step.norm();if(norm>0.25) step*=0.25/norm;
        u+=step[0];v+=step[1];out.iterations=iteration+1;
        if(norm<step_tolerance) {out.converged=true;break;}
    }
    const auto d=extended_parameter_jet(patch,u,v,1).lower;
    const Eigen::Vector3d e=target-d.point;out.u=u;out.v=v;out.r=e.dot(d.normal);
    out.tangential_residual=(e-out.r*d.normal).norm();
    if(!std::isfinite(out.tangential_residual)) throw std::runtime_error("Nonfinite tubular projection residual");
    return out;
}

ExtendedTubularCauchyPlan3D build_extended_tubular_cauchy_plan_3d(
    const geometry3d::NurbsSurfacePatch3D& patch,
    const NativeSurfaceCubicParameterJet3D& geometry,
    const DirectCoefficientCubicCauchyPlan3D& physical)
{
    validate_local_orthonormal_frame_3d(physical.frame);
    if((geometry.lower.point-physical.center).norm()>1e-10)
        throw std::invalid_argument("Tubular and physical centers differ");
    const auto normals=normal_derivatives(geometry);const auto& d=geometry.lower;
    Eigen::Matrix3d frame;frame.col(0)=physical.frame.tangent1;frame.col(1)=physical.frame.tangent2;frame.col(2)=physical.frame.normal;
    const double normal_sign=normals[0].dot(physical.frame.normal)>=0?1.0:-1.0;
    std::array<Polynomial,3> F{{Polynomial::Zero(),Polynomial::Zero(),Polynomial::Zero()}};
    auto add=[&](int u,int v,int r,const Eigen::Vector3d& derivative) {
        const Eigen::Vector3d local=frame.transpose()*derivative/factorial(u)/factorial(v)/factorial(r);
        for(int k=0;k<3;++k) F[k][index(u,v,r)]=local[k];
    };
    add(1,0,0,d.x_u);add(0,1,0,d.x_v);add(2,0,0,d.x_uu);add(1,1,0,d.x_uv);add(0,2,0,d.x_vv);
    add(3,0,0,geometry.x_uuu);add(2,1,0,geometry.x_uuv);add(1,2,0,geometry.x_uvv);add(0,3,0,geometry.x_vvv);
    add(0,0,1,normal_sign*normals[0]);add(1,0,1,normal_sign*normals[1]);add(0,1,1,normal_sign*normals[2]);
    add(2,0,1,normal_sign*normals[3]);add(1,1,1,normal_sign*normals[4]);add(0,2,1,normal_sign*normals[5]);
    Eigen::Matrix<double,20,20> compose;const auto& powers=cubic_cauchy_powers_3d();
    for(int q=0;q<20;++q) {
        Polynomial p=Polynomial::Zero();p[index(0,0,0)]=1;
        for(int i=0;i<powers[q].s;++i) p=product(p,F[0]);
        for(int i=0;i<powers[q].t;++i) p=product(p,F[1]);
        for(int i=0;i<powers[q].r;++i) p=product(p,F[2]);compose.col(q)=p;
    }
    ExtendedTubularCauchyPlan3D result;result.surface=&patch;result.u=physical.u;result.v=physical.v;
    result.normal_orientation=normal_sign;
    result.value_map=compose*physical.closure.value_map;result.normal_map=compose*physical.closure.normal_map;
    if(!result.value_map.allFinite()||!result.normal_map.allFinite()) throw std::runtime_error("Nonfinite tubular Cauchy map");
    return result;
}

TubularCauchyWeights3D ExtendedTubularCauchyPlan3D::weights(const Eigen::Vector3d& target,int degree) const
{
    if(surface==nullptr||(degree!=2&&degree!=3)) throw std::invalid_argument("Invalid tubular Cauchy plan/degree");
    const auto projection=project_extended_tubular_3d(*surface,target,u,v);
    Eigen::Matrix<double,1,20> monomial;const auto& p=cubic_cauchy_powers_3d();
    for(int i=0;i<20;++i) monomial[i]=p[i].degree()<=degree?
        std::pow(projection.u-u,p[i].s)*std::pow(projection.v-v,p[i].t)*std::pow(normal_orientation*projection.r,p[i].r):0;
    TubularCauchyWeights3D out;out.value=monomial*value_map;out.normal=monomial*normal_map;
    out.projection_residual=projection.tangential_residual;out.projected_u=projection.u;out.projected_v=projection.v;
    out.projection_iterations=projection.iterations;out.projection_converged=projection.converged;return out;
}
} // namespace kfbim::app3d
