#include "src/support/geometry/trace_first_case_3d.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfbim::app3d {
namespace {
constexpr double pi = 3.141592653589793238462643383279502884;
RigidTransform3D python_transform(const std::string& name)
{
    Eigen::Vector3d angles = Eigen::Vector3d::Zero();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    if (name == "rotate" || name == "rotate_same_translate") angles = {17.,-11.,13.};
    else if (name == "rotate_translate") angles = {31.,19.,-23.};
    else if (name != "translate") throw std::invalid_argument("unknown Python trace-first transform");
    if (name != "rotate") translation = {.037,-.029,.041};
    const Eigen::Matrix3d rotation =
        (Eigen::AngleAxisd(angles.z()*pi/180, Eigen::Vector3d::UnitZ())
         * Eigen::AngleAxisd(angles.y()*pi/180, Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(angles.x()*pi/180, Eigen::Vector3d::UnitX())).toRotationMatrix();
    return {rotation, Eigen::Vector3d::Zero(), translation};
}
double local_derivative(const Eigen::Vector3d& x, int i, int j, int k)
{
    const double b=.72, c=.43, a=std::sqrt(b*b+c*c);
    double value=std::pow(a,i)*std::pow(b,j)*std::pow(c,k)*std::exp(a*x.x())
        *std::cos(b*x.y()+j*pi/2)*std::cos(c*x.z()+k*pi/2);
    const auto polynomial=[&](int px,int py,int pz,double coefficient) {
        if(i>px || j>py || k>pz) return 0.0;
        double term=coefficient;
        const std::array<int,3> powers{{px,py,pz}},orders{{i,j,k}};
        for(int axis=0;axis<3;++axis) {
            for(int d=0;d<orders[axis];++d) term*=powers[axis]-d;
            term*=std::pow(x[axis],powers[axis]-orders[axis]);
        }
        return term;
    };
    return value+polynomial(2,0,0,.18)+polynomial(0,2,0,-.18)
        +polynomial(1,1,1,.11)+polynomial(1,0,0,.07)+polynomial(0,0,1,-.05);
}
} // namespace

TraceFirstHarmonicJet3D TraceFirstCase3D::evaluate(const Eigen::Vector3d& world) const
{
    if(!world.allFinite()) throw std::invalid_argument("nonfinite case point");
    const Eigen::Vector3d x=transform.inverse_point(world);
    const Eigen::Matrix3d& R=transform.rotation();
    TraceFirstHarmonicJet3D result;
    result.value=local_derivative(x,0,0,0)-boundary_mean_shift;
    Eigen::Vector3d gradient; Eigen::Matrix3d hessian;
    for(int a=0;a<3;++a) {
        std::array<int,3> order{{0,0,0}}; ++order[a];
        gradient[a]=local_derivative(x,order[0],order[1],order[2]);
        for(int b=0;b<3;++b) {
            ++order[b]; hessian(a,b)=local_derivative(x,order[0],order[1],order[2]);
            for(int c=0;c<3;++c) {
                ++order[c]; const double d=local_derivative(x,order[0],order[1],order[2]);
                for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k)
                    result.third[k](i,j)+=R(i,a)*R(j,b)*R(k,c)*d;
                --order[c];
            }
            --order[b];
        }
    }
    result.gradient=R*gradient; result.hessian=R*hessian*R.transpose();
    return result;
}
std::array<int,2> TraceFirstCase3D::density_spans(int N) const
{
    if(N<density_base_N || N%density_base_N)
        throw std::invalid_argument("trace-first density requires N = 32 * 2^ell");
    const int factor=N/density_base_N;
    if((factor&(factor-1))!=0 || factor>std::numeric_limits<int>::max()/density_base_spans[0])
        throw std::invalid_argument("invalid dyadic density refinement");
    return {{density_base_spans[0]*factor,density_base_spans[1]*factor}};
}
TraceFirstCase3D make_trace_first_python_torus_case_3d(const std::string& name)
{
    TraceFirstCase3D result; result.transform_name=name; result.transform=python_transform(name);
    result.surface=transform_native_nurbs_surface_3d(make_native_nurbs_torus_3d(.46,.18),result.transform);
    result.surface.name="python_torus";
    // Exactly one 8x8 Gauss rule per geometry patch, as Python boundary_mean.
    constexpr std::array<double,8> nodes{{-.9602898564975363,-.7966664774136267,-.5255324099163290,-.1834346424956498,
        .1834346424956498,.5255324099163290,.7966664774136267,.9602898564975363}};
    constexpr std::array<double,8> weights{{.1012285362903763,.2223810344533745,.3137066458778873,.3626837833783620,
        .3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763}};
    double integral=0,area=0;
    for(const auto& patch:result.surface.patches) for(int i=0;i<8;++i) for(int j=0;j<8;++j) {
        const auto d=patch.evaluate_with_derivatives((nodes[i]+1)/2,(nodes[j]+1)/2);
        const double w=weights[i]*weights[j]/4*d.du.cross(d.dv).norm();
        integral+=w*local_derivative(result.transform.inverse_point(d.point),0,0,0); area+=w;
    }
    result.boundary_mean_shift=integral/area;
    return result;
}
} // namespace kfbim::app3d
