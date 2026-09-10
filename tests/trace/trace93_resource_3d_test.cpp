#include "src/support/trace/trace93_resource_3d.hpp"
#include "src/support/density/trace93_density_layout_3d.hpp"
#include "src/support/geometry/trace93_case_3d.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace kfbim::app3d;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
double linear(const ResourceAffineRow3D& row,const Eigen::VectorXd& c){
    double value=0;for(std::size_t k=0;k<row.columns.size();++k)value+=row.values[k]*c[row.columns[k]];return value;
}
struct Fixture {
    Trace93DensityLayout3D density;
    std::vector<RestrictResourceAnchor3D> traces,events;
    Eigen::VectorXd ones,u_coefficients;
};
Fixture fixture(const Trace93Case3D& problem,int pid,bool neumann){
    Fixture f;auto& s=f.density;s.problem=&problem;s.neumann=neumann;s.h=0.1;
    const int count=static_cast<int>(problem.surface.patches.size());
    // One bicubic Bernstein density element per analysis patch is sufficient
    // for exact constants and linear analysis coordinates. No solver, fit,
    // collocation constraints or manufactured coefficient recovery is used.
    for(int p=0;p<count;++p){s.patch_spans.push_back({1,1});s.patch_offsets.push_back(16*p);}
    s.reference_raw_dofs=16*count;f.ones=Eigen::VectorXd::Ones(s.reference_raw_dofs);
    f.u_coefficients.resize(s.reference_raw_dofs);
    for(int p=0;p<count;++p)for(int j=0;j<4;++j)for(int i=0;i<4;++i)f.u_coefficients[16*p+4*j+i]=i/3.0;
    NativeDensityGaussPoint3D q;q.patch=pid;q.u=0.31;q.v=0.43;
    const auto g=s.geometry_jet(pid,q.u,q.v);q.point=g.lower.point;q.normal=g.lower.normal;s.traces.push_back(q);
    RestrictResourceAnchor3D a;a.patch_id=pid;a.point=q.point;
    const auto native=problem.analysis_to_native_uv(pid,q.u,q.v);a.u=native.x();a.v=native.y();
    f.traces.push_back(a);a.crossing_id=0;f.events.push_back(a);return f;
}
void test_angular_chart(){
    const auto problem=make_trace93_case_3d("cylinder","rotate");
    for(bool neumann:{true,false}){
        auto f=fixture(problem,0,neumann);
        require(std::abs(f.traces[0].u-f.density.traces[0].u)>1e-4,
            "test must distinguish angle coordinates from rational NURBS coordinates");
        Trace93PolynomialCatalog3D reuse(problem,f.density,f.traces,f.events,neumann,0,true);
        Trace93PolynomialCatalog3D reference(problem,f.density,f.traces,f.events,neumann,0,false);
        const double u=0.41,v=0.38,r=0.037;
        const auto g=f.density.geometry_jet(0,u,v);
        const Eigen::Vector3d target=g.lower.point+r*g.lower.normal;
        const double radius=g.lower.x_u.norm()/(0.5*std::acos(-1.0));
        for(int degree:{2,3}){
            const auto a=reuse.evaluate(degree,{RestrictAnchorSource3D::Trace,0},target,71);
            const auto b=reference.evaluate(degree,{RestrictAnchorSource3D::Trace,0},target,71);
            const auto& coefficients=neumann?f.u_coefficients:f.ones;
            const double expected=neumann?u:r-r*r/(2*radius)+(degree==3?r*r*r/(3*radius*radius):0.0);
            require(std::abs(linear(a,coefficients)-expected)<3e-11,
                "93 cylinder polynomial must reproduce linear angle or radial logarithm Taylor jet");
            require(std::abs(linear(a,coefficients)-linear(b,coefficients))<1e-14 && std::abs(a.known-b.known)<1e-14,
                "row reuse must not change 93 affine coefficients");
            const auto event=reuse.evaluate(degree,{RestrictAnchorSource3D::SpreadCrossing,0},target,71);
            require(std::abs(linear(event,coefficients)-expected)<3e-10,
                "native event point must recover analysis coordinates before density evaluation");
            (void)reuse.evaluate(degree,{RestrictAnchorSource3D::Trace,0},target,71);
        }
        require(reuse.statistics().row_cache_hits==2,"93 degree/anchor/grid cache keys must remain distinct");
        require(reuse.statistics().maximum_projection_residual<1e-12,"angular projection must be analytic to roundoff");
    }
}
void test_planar_chart(){
    const auto problem=make_trace93_case_3d("cylinder","identity");
    auto f=fixture(problem,4,true); // Affine center of the top cap.
    Trace93PolynomialCatalog3D catalog(problem,f.density,f.traces,f.events,true);
    const auto g=f.density.geometry_jet(4,0.51,0.38);
    const Eigen::Vector3d target=g.lower.point+0.019*g.lower.normal;
    for(int degree:{2,3}){
        const auto row=catalog.evaluate(degree,{RestrictAnchorSource3D::Trace,0},target);
        require(std::abs(linear(row,f.u_coefficients)-0.51)<2e-12,
            "planar 93 chart must reproduce its linear tangent density exactly");
    }
}
} // namespace
int main(){try{test_angular_chart();test_planar_chart();std::cout<<"trace93 analysis Cauchy/catalog tests passed\n";return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
