#include "src/support/geometry/trace93_case_3d.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace kfbim::app3d {
namespace {
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double radius = 0.54, half_square = 0.22, z0 = -0.5, z1 = 0.5;
using Vector = Eigen::Vector3d;
using HNet = std::vector<std::vector<Eigen::Vector4d>>;
using Patch = geometry3d::NurbsSurfacePatch3D;
using Edge = geometry3d::NurbsPatchEdge3D;

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

Patch bicubic(HNet net)
{
    // Exact homogeneous Bezier degree elevation, separately in u and v.
    while(net.size()<4) {
        const std::size_t n=net.size();
        HNet next(n+1,std::vector<Eigen::Vector4d>(net[0].size()));
        next.front()=net.front(); next.back()=net.back();
        for(std::size_t i=1;i<n;++i) for(std::size_t j=0;j<net[0].size();++j) {
            const double a=static_cast<double>(i)/n;
            next[i][j]=a*net[i-1][j]+(1-a)*net[i][j];
        }
        net=std::move(next);
    }
    while(net[0].size()<4) {
        const std::size_t n=net[0].size();
        HNet next(net.size(),std::vector<Eigen::Vector4d>(n+1));
        for(std::size_t i=0;i<net.size();++i) {
            next[i].front()=net[i].front(); next[i].back()=net[i].back();
            for(std::size_t j=1;j<n;++j) {
                const double a=static_cast<double>(j)/n;
                next[i][j]=a*net[i][j-1]+(1-a)*net[i][j];
            }
        }
        net=std::move(next);
    }
    std::vector<std::vector<Vector>> points(4,std::vector<Vector>(4));
    std::vector<std::vector<double>> weights(4,std::vector<double>(4));
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) {
        weights[i][j]=net[i][j].w();
        points[i][j]=net[i][j].head<3>()/weights[i][j];
    }
    const geometry::NurbsBasis1D basis(3,{0,0,0,0,1,1,1,1});
    return {basis,basis,std::move(points),std::move(weights)};
}

void append(Trace93Case3D& c,const std::string& name,Patch patch,
            Trace93PatchMetadata3D metadata)
{
    c.surface.patch_names.push_back(name);
    c.surface.patches.push_back(std::move(patch));
    c.surface.patch_components.push_back(0);
    c.surface.smooth_neighbors.emplace_back();
    c.surface.topological_patch_neighbors.emplace_back();
    c.analysis_patches.push_back(std::move(metadata));
}

void affine(Trace93Case3D& c,const std::string& name,
            const Vector& origin,const Vector& du,const Vector& dv)
{
    HNet net(2,std::vector<Eigen::Vector4d>(2));
    for(int i=0;i<2;++i) for(int j=0;j<2;++j) {
        net[i][j].head<3>()=origin+i*du+j*dv; net[i][j].w()=1;
    }
    Trace93PatchMetadata3D m;
    m.local_origin=origin; m.du=du; m.dv=dv;
    m.Lu=du.norm();m.Lv=dv.norm();m.density_length=std::max(m.Lu,m.Lv);
    append(c,name,bicubic(std::move(net)),m);
}

std::array<Eigen::Vector2d,2> cap_corners(int q,bool top)
{
    const std::array<Eigen::Vector2d,4> corners{{
        {half_square,-half_square},{half_square,half_square},
        {-half_square,half_square},{-half_square,-half_square}}};
    std::array<Eigen::Vector2d,2> result{{corners[q],corners[(q+1)%4]}};
    if(!top) std::swap(result[0],result[1]);
    return result;
}

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

void cylinder(Trace93Case3D& c)
{
    c.native_parameters_match_analysis=false;
    for(int q=0;q<4;++q) {
        HNet net(3,std::vector<Eigen::Vector4d>(2));
        const auto lo=circle_arc(q,true,z0),hi=circle_arc(q,true,z1);
        for(int i=0;i<3;++i) {net[i][0]=lo[i];net[i][1]=hi[i];}
        Trace93PatchMetadata3D m;m.kind=Trace93PatchKind3D::CylinderSide;
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
            Trace93PatchMetadata3D m;m.kind=Trace93PatchKind3D::CylinderCapRing;
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

void prism(Trace93Case3D& c,const std::string& kind)
{
    using P=Eigen::Vector2d;
    struct Cell {std::string name;double x0,x1,y0,y1;};
    std::vector<P> boundary;std::vector<Cell> cells;
    if(kind=="box") {
        boundary={{-.5,-.5},{.5,-.5},{.5,.5},{-.5,.5}};
        cells={{"whole",-.5,.5,-.5,.5}};
    } else if(kind=="l") {
        boundary={{-.5,-.5},{-.1,-.5},{.5,-.5},{.5,-.1},
            {-.1,-.1},{-.1,.5},{-.5,.5},{-.5,-.1}};
        cells={{"lower_left",-.5,-.1,-.5,-.1},
            {"lower_right",-.1,.5,-.5,-.1},{"upper_left",-.5,-.1,-.1,.5}};
    } else {
        boundary={{-.55,-.55},{-.22,-.55},{.22,-.55},{.55,-.55},
            {.55,-.1},{.55,.55},{.22,.55},{.22,-.1},{-.22,-.1},
            {-.22,.55},{-.55,.55},{-.55,-.1}};
        cells={{"bottom_left",-.55,-.22,-.55,-.1},
            {"bottom_middle",-.22,.22,-.55,-.1},{"bottom_right",.22,.55,-.55,-.1},
            {"upper_left",-.55,-.22,-.1,.55},{"upper_right",.22,.55,-.1,.55}};
    }
    double area=0,perimeter=0;
    for(std::size_t k=0;k<boundary.size();++k) {
        const P a=boundary[k],b=boundary[(k+1)%boundary.size()];
        affine(c,"side_"+std::to_string(k),{a.x(),a.y(),z0},
               {b.x()-a.x(),b.y()-a.y(),0},{0,0,z1-z0});
        perimeter+=(b-a).norm();
    }
    for(const auto& cell:cells) {
        affine(c,"top_"+cell.name,{cell.x0,cell.y0,z1},
               {cell.x1-cell.x0,0,0},{0,cell.y1-cell.y0,0});
        affine(c,"bottom_"+cell.name,{cell.x0,cell.y1,z0},
               {cell.x1-cell.x0,0,0},{0,cell.y0-cell.y1,0});
        area+=(cell.x1-cell.x0)*(cell.y1-cell.y0);
    }
    c.surface.expected_area=2*area+perimeter*(z1-z0);
    c.surface.exact_inside=[kind](const Vector& x) {
        constexpr double eps=5e-13;
        const double a=kind=="u"?.55:.5;
        if(!(x.z()>z0+eps && x.z()<z1-eps
            && x.x()>-a+eps && x.x()<a-eps && x.y()>-a+eps && x.y()<a-eps)) return false;
        if(kind=="l") return x.x()<-.1-eps || x.y()<-.1-eps;
        if(kind=="u") return x.y()<-.1-eps || x.x()<-.22-eps || x.x()>.22+eps;
        return true;
    };
}

Eigen::Vector2d edge_uv(Edge edge,double s)
{
    if(edge==Edge::UMin) return {0,s};if(edge==Edge::UMax) return {1,s};
    if(edge==Edge::VMin) return {s,0};return {s,1};
}

void connect_complete_edges(Trace93Case3D& c)
{
    const int count=static_cast<int>(c.surface.patches.size());
    std::vector<std::array<bool,4>> matched(count);
    std::vector<int> sheet(count);std::iota(sheet.begin(),sheet.end(),0);
    const auto root=[&](int a) {while(sheet[a]!=a) a=sheet[a];return a;};
    const std::array<Edge,4> edges{{Edge::UMin,Edge::UMax,Edge::VMin,Edge::VMax}};
    const auto point=[&](int p,Edge e,double s) {
        const auto uv=edge_uv(e,s);return c.surface.patches[p].evaluate(uv.x(),uv.y());
    };
    for(int a=0;a<count;++a) for(int ea=0;ea<4;++ea) {
        if(matched[a][ea]) continue;
        int found=0;
        for(int b=a+1;b<count;++b) for(int eb=0;eb<4;++eb) {
            bool reversed=false;
            if((point(a,edges[ea],0)-point(b,edges[eb],0)).norm()<2e-12
                && (point(a,edges[ea],1)-point(b,edges[eb],1)).norm()<2e-12) {}
            else if((point(a,edges[ea],0)-point(b,edges[eb],1)).norm()<2e-12
                && (point(a,edges[ea],1)-point(b,edges[eb],0)).norm()<2e-12) reversed=true;
            else continue;
            for(double s:{.23,.5,.79})
                if((point(a,edges[ea],s)-point(b,edges[eb],reversed?1-s:s)).norm()>2e-12)
                    throw std::runtime_error("trace93 connection has a nonmatching native parameter map");
            if(matched[b][eb] || ++found!=1) throw std::runtime_error("trace93 non-manifold edge");
            const auto ua=edge_uv(edges[ea],.5),ub=edge_uv(edges[eb],.5);
            const auto da=c.surface.patches[a].evaluate_with_derivatives(ua.x(),ua.y());
            const auto db=c.surface.patches[b].evaluate_with_derivatives(ub.x(),ub.y());
            const bool smooth=da.du.cross(da.dv).normalized().dot(db.du.cross(db.dv).normalized())>1-1e-12;
            c.surface.geometric_connections.push_back({
                geometry3d::full_patch_edge_interval(c.surface.patches[a],a,edges[ea]),
                geometry3d::full_patch_edge_interval(c.surface.patches[b],b,edges[eb]),reversed,smooth});
            c.surface.topological_patch_neighbors[a].push_back(b);
            c.surface.topological_patch_neighbors[b].push_back(a);
            if(smooth) {
                c.surface.smooth_neighbors[a][static_cast<int>(edges[ea])]=SmoothPatchNeighbor3D{b,edges[eb],reversed};
                c.surface.smooth_neighbors[b][static_cast<int>(edges[eb])]=SmoothPatchNeighbor3D{a,edges[ea],reversed};
                const int ra=root(a),rb=root(b);sheet[std::max(ra,rb)]=std::min(ra,rb);
            }
            matched[a][ea]=matched[b][eb]=true;
        }
        if(found!=1) throw std::runtime_error("trace93 boundary edge has no partner");
    }
    for(int p=0;p<count;++p) c.analysis_patches[p].sheet=root(p);
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
    if(m.kind==Trace93PatchKind3D::Affine) {
        out.d[0][0]=m.local_origin+u*m.du+v*m.dv;
        if(order>=1) {out.d[1][0]=m.du;out.d[0][1]=m.dv;}
    } else if(m.kind==Trace93PatchKind3D::CylinderSide) {
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
    if(m.kind==Trace93PatchKind3D::Affine) {
        Eigen::Matrix<double,3,2> A;A.col(0)=m.du;A.col(1)=m.dv;
        return (A.transpose()*A).ldlt().solve(A.transpose()*(x-m.local_origin));
    }
    const double start=-pi/4+m.quarter*pi/2+(
        m.kind==Trace93PatchKind3D::CylinderCapRing && !m.top?pi/2:0);
    const double dt=m.kind==Trace93PatchKind3D::CylinderCapRing && !m.top?-pi/2:pi/2;
    double theta=std::atan2(x.y(),x.x());
    theta+=2*pi*std::round((start+.5*dt-theta)/(2*pi));
    if(m.kind==Trace93PatchKind3D::CylinderSide) return {(theta-start)/dt,(x.z()-z0)/(z1-z0)};
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
    if(m.kind==Trace93PatchKind3D::Affine) return world_to_analysis_uv(pid,world,initial);
    if(m.kind==Trace93PatchKind3D::CylinderSide) {
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
    if(geometry=="cylinder") cylinder(c);
    else if(geometry=="box" || geometry=="l" || geometry=="u") prism(c,geometry);
    else throw std::invalid_argument("unknown trace93 geometry");
    c.surface.name="trace93_"+geometry;
    c.surface.description="Python 93 physical geometry, exact bicubic NURBS; separate analysis chart";
    connect_complete_edges(c);
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
