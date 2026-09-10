#include "src/support/cauchy/extended_tubular_cauchy_3d.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace kfbim::app3d;
void require(bool valid,const char* message) { if(!valid) throw std::runtime_error(message); }
void near(double a,double b,double tol,const char* message)
{ require(std::isfinite(a)&&std::abs(a-b)<tol,message); }

kfbim::geometry3d::NurbsSurfacePatch3D patch(double rational,double curvature,
    const Eigen::Matrix3d& rotation=Eigen::Matrix3d::Identity(),
    const Eigen::Vector3d& translation=Eigen::Vector3d::Zero())
{
    kfbim::geometry::NurbsBasis1D ub(3,{2.,5.},{4,4}),vb(3,{-4.,1.},{4,4});
    std::vector<std::vector<Eigen::Vector3d>> control(4,std::vector<Eigen::Vector3d>(4));
    std::vector<std::vector<double>> weight(4,std::vector<double>(4));
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) {
        weight[i][j]=1+rational*i/3.;
        const Eigen::Vector3d p(i/3./weight[i][j],j/3.,curvature*i*(i-1)/6.);
        control[i][j]=rotation*p+translation;
    }
    return {ub,vb,control,weight};
}
DirectCoefficientCubicCauchyPlan3D physical_plan(
    const NativeSurfaceCubicParameterJet3D& g,double u,double v,bool flip=false)
{
    DirectCoefficientCubicCauchyPlan3D p;p.u=u;p.v=v;p.center=g.lower.point;
    p.frame=make_local_orthonormal_frame_3d((flip?-1.:1.)*g.lower.normal,g.lower.x_u);
    p.graph=tangent_graph_third_jet_3d(g,p.frame);p.closure=build_cubic_cauchy_closure_3d(p.graph);return p;
}
double value(const ExtendedTubularCauchyPlan3D& t,const DirectCoefficientCubicCauchyPlan3D& p,
    const KnownAmbientThird3D& known,const Eigen::Vector3d& target,int degree)
{
    const auto w=t.weights(target,degree);
    require(w.projection_residual<2e-11,"Tubular projection residual");
    return (w.value*known_cubic_dirichlet_jet_3d(known,p.graph,p.frame))[0]
        +(w.normal*known_cubic_neumann_jet_3d(known,p.graph,p.frame))[0];
}
void rational_extended_chart()
{
    const auto surface=patch(.4,0);const double u=.94,v=.4,ut=1.08,vt=.26;
    const auto g=extended_nurbs_parameter_jet_3d(surface,u,v);
    const auto physical=physical_plan(g,u,v);
    const auto tube=build_extended_tubular_cauchy_plan_3d(surface,g,physical);
    const Eigen::Vector3d target(ut/(1+.4*ut),vt,.07);
    const auto projection=project_extended_tubular_3d(surface,target,u,v);
    near(projection.u,ut,2e-12,"Extended chart must not clamp normalized u");
    near(projection.v,vt,2e-12,"Nonunit domain normalized v");
    near(projection.r,.07,2e-12,"Normal distance");
    require(projection.converged,"Planar rational projection converges");
    const auto external=extended_nurbs_parameter_jet_3d(surface,ut,vt);
    const double denominator=1+.4*ut;
    near(external.lower.point.x(),ut/denominator,2e-13,"Extended rational point");
    near(external.x_uuu.x(),.96/std::pow(denominator,4),2e-11,"Extended rational third derivative");
    KnownAmbientThird3D data;data.value=g.lower.point.x();data.gradient=Eigen::Vector3d::UnitX();
    const double delta=ut-u;
    const double expected2=g.lower.point.x()+g.lower.x_u.x()*delta+.5*g.lower.x_uu.x()*delta*delta;
    const double expected3=expected2+g.x_uuu.x()*delta*delta*delta/6;
    near(value(tube,physical,data,target,2),expected2,2e-12,"P2 is tubular Taylor truncation");
    near(value(tube,physical,data,target,3),expected3,2e-12,"P3 is tubular Taylor truncation");
    require(std::abs(expected3-target.x())>1e-7,"Test distinguishes tube from physical polynomial");
    const auto w=tube.weights(target,2);
    require(w.value.tail<4>().norm()<1e-14&&w.normal.tail<3>().norm()<1e-14,"P2 never uses cubic input jets");
    bool invalid=false;try { (void)tube.weights(target,1); } catch(const std::invalid_argument&) { invalid=true; }
    require(invalid,"Invalid polynomial order must fail");
}
void curved_chart_and_rigid_transform()
{
    constexpr double curvature=.3,u=.4,v=.45,ut=.43,vt=.42,r=.04;
    const double delta=ut-u,k=2*curvature,z=1+k*k*u*u;
    const double nz=1/std::sqrt(z),nz1=-k*k*u/std::pow(z,1.5);
    const double nz2=-k*k/std::pow(z,1.5)+3*std::pow(k,4)*u*u/std::pow(z,2.5);
    const double expected2=curvature*ut*ut+r*(nz+nz1*delta);
    const double expected3=expected2+.5*r*nz2*delta*delta;
    for(bool transformed:{false,true}) for(bool flip:{false,true}) {
        const Eigen::Matrix3d rotation=transformed?
            Eigen::AngleAxisd(.61,Eigen::Vector3d(1,2,-1).normalized()).toRotationMatrix():Eigen::Matrix3d::Identity();
        const Eigen::Vector3d translation=transformed?Eigen::Vector3d(.2,-.3,.1):Eigen::Vector3d::Zero();
        const auto surface=patch(0,curvature,rotation,translation);
        const auto g=extended_nurbs_parameter_jet_3d(surface,u,v),end=extended_nurbs_parameter_jet_3d(surface,ut,vt);
        const Eigen::Vector3d target=end.lower.point+r*end.lower.normal;
        const auto physical=physical_plan(g,u,v,flip);
        const auto tube=build_extended_tubular_cauchy_plan_3d(surface,g,physical);
        KnownAmbientThird3D data;data.value=curvature*u*u;data.gradient=rotation.col(2);
        near(value(tube,physical,data,target,2),expected2,3e-11,"Curved P2 with rigid/normal orientation");
        near(value(tube,physical,data,target,3),expected3,3e-11,"Curved P3 with normal derivatives and rigid invariance");
    }
}
} // namespace
int main()
{
    try {rational_extended_chart();curved_chart_and_rigid_transform();std::cout<<"extended tubular Cauchy tests passed\n";return 0;}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
