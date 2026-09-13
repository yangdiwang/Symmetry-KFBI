#include "construction.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d::bicubic_nurbs_detail {
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

void append(BicubicNurbsModel3D& c,const std::string& name,Patch patch,
            SurfaceAnalysisChart3D metadata)
{
    c.surface.patch_names.push_back(name);
    c.surface.patches.push_back(std::move(patch));
    c.surface.patch_components.push_back(0);
    c.surface.smooth_neighbors.emplace_back();
    c.surface.topological_patch_neighbors.emplace_back();
    c.analysis_patches.push_back(std::move(metadata));
}

void affine(BicubicNurbsModel3D& c,const std::string& name,
            const Vector& origin,const Vector& du,const Vector& dv)
{
    HNet net(2,std::vector<Eigen::Vector4d>(2));
    for(int i=0;i<2;++i) for(int j=0;j<2;++j) {
        net[i][j].head<3>()=origin+i*du+j*dv; net[i][j].w()=1;
    }
    SurfaceAnalysisChart3D m;
    m.local_origin=origin; m.du=du; m.dv=dv;
    m.Lu=du.norm();m.Lv=dv.norm();m.density_length=std::max(m.Lu,m.Lv);
    append(c,name,bicubic(std::move(net)),m);
}

void extrude_prism(BicubicNurbsModel3D& c,
    const std::vector<Eigen::Vector2d>& boundary, const std::vector<PrismCell>& cells)
{
    using P = Eigen::Vector2d;
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
}

Eigen::Vector2d edge_uv(Edge edge,double s)
{
    if(edge==Edge::UMin) return {0,s};if(edge==Edge::UMax) return {1,s};
    if(edge==Edge::VMin) return {s,0};return {s,1};
}

void connect_complete_edges(BicubicNurbsModel3D& c)
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
                    throw std::runtime_error("bicubic NURBS connection has a nonmatching native parameter map");
            if(matched[b][eb] || ++found!=1) throw std::runtime_error("bicubic NURBS non-manifold edge");
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
        if(found!=1) throw std::runtime_error("bicubic NURBS boundary edge has no partner");
    }
    for(int p=0;p<count;++p) c.analysis_patches[p].sheet=root(p);
}


} // namespace kfbim::app3d::bicubic_nurbs_detail
