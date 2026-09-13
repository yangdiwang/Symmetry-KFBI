#include "src/support/geometry/trace93_case_3d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d {
namespace {
using Vector = Eigen::Vector3d;
using bicubic_nurbs_model_parameters::pi;
using bicubic_nurbs_model_parameters::radius;
using bicubic_nurbs_model_parameters::z0;
using bicubic_nurbs_model_parameters::z1;
using bicubic_nurbs_model_parameters::cap_corners;

RigidTransform3D transform_for(const std::string& name)
{
    Vector angles = Vector::Zero(), translation = Vector::Zero();
    if(name=="rotate" || name=="rotate_same_translate") angles={17.,-11.,13.};
    else if(name=="rotate_translate") angles={31.,19.,-23.};
    else if(name!="identity" && name!="translate")
        throw std::invalid_argument("unknown trace93 rigid transform");
    if(name=="translate" || name=="rotate_same_translate" || name=="rotate_translate")
        translation={.037,-.029,.041};
    const Eigen::Matrix3d rotation=(
        Eigen::AngleAxisd(angles.z()*pi/180,Vector::UnitZ())
        *Eigen::AngleAxisd(angles.y()*pi/180,Vector::UnitY())
        *Eigen::AngleAxisd(angles.x()*pi/180,Vector::UnitX())).toRotationMatrix();
    return {rotation,Vector::Zero(),translation};
}

double raw_local(const Vector& x)
{
    return std::exp(std::sqrt(.72*.72+.43*.43)*x.x())*std::cos(.72*x.y())*std::cos(.43*x.z())
        +.18*(x.x()*x.x()-x.y()*x.y())+.11*x.x()*x.y()*x.z()+.07*x.x()-.05*x.z();
}
} // namespace

Trace93GeometryJet3D::Trace93GeometryJet3D()
{for(auto& row:d) for(auto& value:row) value.setZero();}

Trace93GeometryJet3D Trace93Case3D::analysis_at(int pid,double u,double v,int order) const
{
    if(pid<0 || static_cast<std::size_t>(pid)>=analysis_patches.size()
        || order<0 || order>4 || !std::isfinite(u) || !std::isfinite(v))
        throw std::invalid_argument("invalid trace93 analysis chart query");
    const auto& m=analysis_patches[pid];Trace93GeometryJet3D out;
    if(m.kind==SurfaceAnalysisChartKind3D::Affine) {
        out.d[0][0]=m.local_origin+u*m.du+v*m.dv;
        if(order>=1) {out.d[1][0]=m.du;out.d[0][1]=m.dv;}
    } else if(m.kind==SurfaceAnalysisChartKind3D::CylinderSide) {
        const double theta=-pi/4+m.quarter*pi/2+(pi/2)*u;
        for(int i=0;i<=order;++i)
            out.d[i][0]={radius*std::pow(pi/2,i)*std::cos(theta+i*pi/2),
                radius*std::pow(pi/2,i)*std::sin(theta+i*pi/2),i==0?z0+(z1-z0)*v:0};
        if(order>=1) out.d[0][1]={0,0,z1-z0};
    } else {
        const auto corners=cap_corners(m.quarter,m.top);
        const Eigen::Vector2d inner=corners[0]+v*(corners[1]-corners[0]);
        const double dt=m.top?pi/2:-pi/2;
        const double theta=-pi/4+m.quarter*pi/2+(m.top?0:pi/2)+dt*v;
        for(int j=0;j<=order;++j) {
            const Eigen::Vector2d outside=radius*std::pow(dt,j)*Eigen::Vector2d(std::cos(theta+j*pi/2),std::sin(theta+j*pi/2));
            Eigen::Vector2d inside=Eigen::Vector2d::Zero();
            if(j==0) inside=inner;else if(j==1) inside=corners[1]-corners[0];
            out.d[0][j].head<2>()=(1-u)*inside+u*outside;
            if(j+1<=order) out.d[1][j].head<2>()=outside-inside;
        }
        out.d[0][0].z()=m.top?z1:z0;
    }
    for(int i=0;i<=order;++i) for(int j=0;j<=order-i;++j)
        out.d[i][j]=i+j==0?transform.forward_point(out.d[i][j]):transform.forward_vector(out.d[i][j]);
    return out;
}

Eigen::Vector2d Trace93Case3D::world_to_analysis_uv(
    int pid,const Vector& world,const Eigen::Vector2d& initial) const
{
    if(pid<0 || static_cast<std::size_t>(pid)>=analysis_patches.size()
        || !world.allFinite() || !initial.allFinite()) throw std::invalid_argument("invalid trace93 inverse chart query");
    const auto& m=analysis_patches[pid];const Vector x=transform.inverse_point(world);
    if(m.kind==SurfaceAnalysisChartKind3D::Affine) {
        Eigen::Matrix<double,3,2> A;A.col(0)=m.du;A.col(1)=m.dv;
        return (A.transpose()*A).ldlt().solve(A.transpose()*(x-m.local_origin));
    }
    const double start=-pi/4+m.quarter*pi/2+(
        m.kind==SurfaceAnalysisChartKind3D::CylinderCapRing && !m.top?pi/2:0);
    const double dt=m.kind==SurfaceAnalysisChartKind3D::CylinderCapRing && !m.top?-pi/2:pi/2;
    double theta=std::atan2(x.y(),x.x());
    theta+=2*pi*std::round((start+.5*dt-theta)/(2*pi));
    if(m.kind==SurfaceAnalysisChartKind3D::CylinderSide) return {(theta-start)/dt,(x.z()-z0)/(z1-z0)};
    Eigen::Vector2d uv(std::clamp(initial.x(),0.,1.),std::clamp((theta-start)/dt,0.,1.));
    // Newton is an inverse parameter-coordinate calculation, never a density
    // fit or an intersection certificate. Accept only verified position residual.
    for(int iteration=0;iteration<40;++iteration) {
        const auto jet=analysis_at(pid,uv.x(),uv.y(),1);
        const Vector residual=world-jet.d[0][0];
        const Vector tangential=residual-residual.dot(transform.forward_vector(Vector::UnitZ()))
            *transform.forward_vector(Vector::UnitZ());
        if(tangential.norm()<2e-13) return uv;
        Eigen::Matrix<double,3,2> A;A.col(0)=jet.d[1][0];A.col(1)=jet.d[0][1];
        const Eigen::Vector2d step=(A.transpose()*A).ldlt().solve(A.transpose()*residual);
        if(!step.allFinite()) break;
        bool improved=false;
        for(double fraction=1.;fraction>=1./1024;fraction*=.5) {
            Eigen::Vector2d candidate=uv+fraction*step;
            candidate.x()=std::clamp(candidate.x(),0.,1.);candidate.y()=std::clamp(candidate.y(),0.,1.);
            if((world-analysis_at(pid,candidate.x(),candidate.y(),0).d[0][0]).squaredNorm()<residual.squaredNorm()) {
                uv=candidate;improved=true;break;
            }
        }
        if(!improved) break;
    }
    if((world-analysis_at(pid,uv.x(),uv.y(),0).d[0][0]).norm()>2e-11)
        throw std::runtime_error("trace93 cap inverse chart failed residual verification");
    return uv;
}

Eigen::Vector2d Trace93Case3D::world_to_native_uv(
    int pid,const Vector& world,const Eigen::Vector2d& initial) const
{
    if(pid<0 || static_cast<std::size_t>(pid)>=analysis_patches.size()
        || !world.allFinite() || !initial.allFinite())
        throw std::invalid_argument("invalid trace93 native inverse chart query");
    const auto& m=analysis_patches[pid];
    if(m.kind==SurfaceAnalysisChartKind3D::Affine) return world_to_analysis_uv(pid,world,initial);
    if(m.kind==SurfaceAnalysisChartKind3D::CylinderSide) {
        const auto analysis=world_to_analysis_uv(pid,world,initial);
        const double half_angle=analysis.x()*pi/4;
        const double numerator=std::sin(half_angle);
        const double parameter=numerator/(numerator+std::sin(pi/4-half_angle));
        return {parameter,analysis.y()};
    }
    const auto& patch=surface.patches[pid];
    Eigen::Vector2d uv=initial.cwiseMax(0.).cwiseMin(1.);
    for(int iteration=0;iteration<40;++iteration) {
        const auto d=patch.evaluate_with_derivatives(uv.x(),uv.y());
        const Vector residual=world-d.point;
        if(residual.norm()<2e-13) return uv;
        Eigen::Matrix<double,3,2> A;A.col(0)=d.du;A.col(1)=d.dv;
        const Eigen::Vector2d step=(A.transpose()*A).ldlt().solve(A.transpose()*residual);
        if(!step.allFinite()) break;
        bool improved=false;
        for(double fraction=1.;fraction>=1./1024;fraction*=.5) {
            const Eigen::Vector2d candidate=(uv+fraction*step).cwiseMax(0.).cwiseMin(1.);
            if((world-patch.evaluate(candidate.x(),candidate.y())).squaredNorm()<residual.squaredNorm()) {
                uv=candidate;improved=true;break;
            }
        }
        if(!improved) break;
    }
    if((world-patch.evaluate(uv.x(),uv.y())).norm()>8e-13)
        throw std::runtime_error("trace93 native inverse chart failed residual verification");
    return uv;
}

Eigen::Vector2d Trace93Case3D::analysis_to_native_uv(int pid,double u,double v) const
{
    return world_to_native_uv(pid,analysis_at(pid,u,v,0).d[0][0],{u,v});
}

Trace93Case3D make_trace93_case_3d(const std::string& geometry,const std::string& name)
{
    Trace93Case3D c;c.geometry_name=geometry;c.id="python_trace93_"+geometry+"_20260909";
    c.transform_name=name;c.transform=transform_for(name);
    // Translate the existing study's CLI selectors into explicit model shapes.
    const auto shape=[&] {
        if(geometry=="box") return BicubicNurbsShape3D::Box;
        if(geometry=="cylinder") return BicubicNurbsShape3D::SolidCylinder;
        if(geometry=="l") return BicubicNurbsShape3D::LPrism;
        if(geometry=="u") return BicubicNurbsShape3D::UPrism;
        throw std::invalid_argument("unknown bicubic NURBS study geometry: "+geometry);
    }();
    auto definition=make_bicubic_nurbs_model_3d(shape);
    c.surface=std::move(definition.surface);
    c.analysis_patches=std::move(definition.analysis_patches);
    c.native_parameters_match_analysis=definition.native_parameters_match_analysis;
    c.surface=transform_native_nurbs_surface_3d(c.surface,c.transform);
    constexpr std::array<double,8> nodes{{-.9602898564975363,-.7966664774136267,-.5255324099163290,-.1834346424956498,
        .1834346424956498,.5255324099163290,.7966664774136267,.9602898564975363}};
    constexpr std::array<double,8> weights{{.1012285362903763,.2223810344533745,.3137066458778873,.3626837833783620,
        .3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763}};
    double integral=0,area=0;
    for(std::size_t p=0;p<c.analysis_patches.size();++p) for(int i=0;i<8;++i) for(int j=0;j<8;++j) {
        const auto d=c.analysis_at(static_cast<int>(p),(nodes[i]+1)/2,(nodes[j]+1)/2,1);
        const double w=weights[i]*weights[j]/4*d.d[1][0].cross(d.d[0][1]).norm();
        integral+=w*raw_local(c.transform.inverse_point(d.d[0][0]));area+=w;
    }
    c.boundary_mean_shift=integral/area;
    return c;
}

} // namespace kfbim::app3d
