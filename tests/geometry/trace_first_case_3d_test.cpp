#include "src/support/geometry/trace_first_case_3d.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace kfbim::app3d;
namespace {
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
}
int main()
{
    try {
        double shift=-1;
        for(const std::string name:{"translate","rotate","rotate_same_translate","rotate_translate"}) {
            const auto c=make_trace_first_python_torus_case_3d(name);
            require(c.surface.patches.size()==16,"patch count");
            require(c.density_spans(32)==std::array<int,2>{{4,2}},"base spans");
            require(c.density_spans(128)==std::array<int,2>{{16,8}},"nested spans");
            bool rejected=false; try{(void)c.density_spans(96);}catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"non-dyadic N accepted");
            if(shift<0)shift=c.boundary_mean_shift;
            require(std::abs(shift-c.boundary_mean_shift)<2e-14,"rigid mean mismatch");
            for(const auto& patch:c.surface.patches) {
                require(patch.basis_u().degree()==2 && patch.basis_v().degree()==2,"geometry degree");
                for(double u:{0.,.23,.71,1.})for(double v:{0.,.17,.63,1.}) {
                    const auto x=c.transform.inverse_point(patch.evaluate(u,v));
                    const double rho=std::hypot(x.x(),x.y())-.46;
                    require(std::abs(rho*rho+x.z()*x.z()-.18*.18)<2e-15,"NURBS torus geometry mismatch");
                }
            }
            const Eigen::Vector3d local(.17,-.21,.13), x=c.transform.forward_point(local);
            const auto jet=c.evaluate(x);
            const double a=std::sqrt(.72*.72+.43*.43);
            const double expected=std::exp(a*local.x())*std::cos(.72*local.y())*std::cos(.43*local.z())
                +.18*(local.x()*local.x()-local.y()*local.y())+.11*local.x()*local.y()*local.z()+.07*local.x()-.05*local.z()-shift;
            require(std::abs(jet.value-expected)<2e-14,"manufactured value mismatch");
            require(std::abs(jet.hessian.trace())<2e-13,"nonharmonic Hessian");
            for(int k=0;k<3;++k) {
                require(std::abs(jet.third[k].trace())<3e-13,"nonharmonic third jet");
                const Eigen::Vector3d delta=1e-5*Eigen::Vector3d::Unit(k);
                const auto plus=c.evaluate(x+delta), minus=c.evaluate(x-delta);
                require(std::abs((plus.value-minus.value)/2e-5-jet.gradient[k])<1e-9,"gradient check");
                require(((plus.gradient-minus.gradient)/2e-5-jet.hessian.col(k)).norm()<1e-9,"Hessian check");
                require(((plus.hessian-minus.hessian)/2e-5-jet.third[k]).norm()<1e-9,"third jet check");
            }
        }
        const auto legacy=make_native_nurbs_surface_3d(GeometryKind3D::Torus);
        require((legacy.patches[0].evaluate(0,0)-Eigen::Vector3d(.82,-.04,.03)).norm()<1e-14,"legacy torus changed");
        std::cout<<"trace-first Python case: PASS\n";
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
