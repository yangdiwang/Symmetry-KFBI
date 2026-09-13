#include "construction.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d::bicubic_nurbs_detail {
std::array<Eigen::Vector4d,3> circle_arc(int q,bool forward,double z)
{
    const double start=-pi/4+q*pi/2+(forward?0:pi/2);
    const double step=forward?pi/2:-pi/2;
    const std::array<double,3> weights{{1.,std::sqrt(.5),1.}};
    std::array<Eigen::Vector4d,3> arc;
    for(int j=0;j<3;++j) {
        const double angle=start+j*step/2;
        // Homogeneous middle xy is radius*(cos(mid),sin(mid)).
        arc[j]={radius*std::cos(angle),radius*std::sin(angle),z*weights[j],weights[j]};
    }
    return arc;
}

void solid_cylinder(BicubicNurbsModel3D& c)
{
    c.native_parameters_match_analysis=false;
    for(int q=0;q<4;++q) {
        HNet net(3,std::vector<Eigen::Vector4d>(2));
        const auto lo=circle_arc(q,true,z0),hi=circle_arc(q,true,z1);
        for(int i=0;i<3;++i) {net[i][0]=lo[i];net[i][1]=hi[i];}
        SurfaceAnalysisChart3D m;m.kind=SurfaceAnalysisChartKind3D::CylinderSide;
        m.planar=false;m.quarter=q;m.Lu=radius*pi/2;m.Lv=z1-z0;
        m.density_length=std::max(m.Lu,m.Lv);
        append(c,"side_"+std::to_string(q),bicubic(std::move(net)),m);
    }
    for(bool top:{true,false}) {
        const double z=top?z1:z0;
        const std::string prefix=top?"top_":"bottom_";
        affine(c,prefix+"center",{-half_square,top?-half_square:half_square,z},
               {2*half_square,0,0},{0,top?2*half_square:-2*half_square,0});
        for(int q=0;q<4;++q) {
            // Native cap chart: (1-u)*I(v)+u*O_rational(v).
            // I is linear at the square edge. Multiplying I by the common
            // quadratic arc denominator yields an exact cubic numerator,
            // so square and side seams retain affine edge parameter maps.
            const auto arc=circle_arc(q,top,z);
            const auto inner=cap_corners(q,top);
            HNet net(2,std::vector<Eigen::Vector4d>(4));
            constexpr std::array<double,3> choose2{{1,2,1}};
            constexpr std::array<double,4> choose3{{1,3,3,1}};
            for(int k=0;k<4;++k) {
                Eigen::Vector2d numerator=Eigen::Vector2d::Zero();
                for(int i=0;i<2;++i) {
                    const int j=k-i;
                    if(j>=0 && j<3) numerator+=choose2[j]/choose3[k]*arc[j].w()*inner[i];
                }
                Eigen::Vector4d outer=Eigen::Vector4d::Zero();
                if(k<3) outer+=(1-k/3.)*arc[k];
                if(k>0) outer+=(k/3.)*arc[k-1];
                net[0][k]={numerator.x(),numerator.y(),z*outer.w(),outer.w()};
                net[1][k]=outer;
            }
            SurfaceAnalysisChart3D m;m.kind=SurfaceAnalysisChartKind3D::CylinderCapRing;
            m.quarter=q;m.top=top;m.Lu=radius-half_square;m.Lv=radius*pi/2;
            m.density_length=std::max(m.Lu,m.Lv);
            append(c,prefix+"ring_"+std::to_string(q),bicubic(std::move(net)),m);
        }
    }
    c.surface.expected_area=2*pi*radius*(z1-z0)+2*pi*radius*radius;
    c.surface.exact_inside=[](const Vector& x) {
        return x.x()*x.x()+x.y()*x.y()<radius*radius
            && x.z()>z0+5e-13 && x.z()<z1-5e-13;
    };
}


} // namespace kfbim::app3d::bicubic_nurbs_detail
