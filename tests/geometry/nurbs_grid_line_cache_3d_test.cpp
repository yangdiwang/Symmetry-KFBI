#include "src/geometry/nurbs_grid_line_cache_3d.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace kfbim;
using namespace kfbim::geometry3d;
namespace {void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}}
int main()
{
    try {
        CartesianGrid3D grid({0.,0.,0.},{.25,.25,.25},{4,4,4},DofLayout3D::Node);
        NurbsCartesianEdgeIntersections3D line;
        for(double x:{.32,.37,.63,.92}) {
            NurbsSurfaceCrossing3D root;root.point={x,.25,.25};root.normal={1,0,0};
            root.component=0;root.patch_index=3;root.u=.2;root.v=.3;root.edge_parameter=x;root.transversality=1;
            root.reliable_transversality_tolerance=1e-8;
            NurbsSurfaceRootOwner3D owner;owner.patch_index=3;owner.u=.2;owner.v=.3;
            owner.point=root.point;owner.normal=root.normal;owner.transversality=1;root.owners.push_back(owner);
            root.owners.back().reliable_transversality_tolerance=1e-8;
            line.crossings.push_back(root);
        }
        line.confirmed_transverse_count=4;
        require(certified_grid_line_can_partition_3d(grid,0,line,1e-12),"safe line rejected");
        std::size_t total=0;
        for(int i=0;i<4;++i) {
            const NurbsCartesianEdgeQuery3D q{0,i,1,1,{i*.25,.25,.25},{(i+1)*.25,.25,.25}};
            const auto edge=extract_grid_line_edge_3d(q,line.crossings);total+=edge.crossings.size();
            if(i==1) {
                require(edge.crossings.size()==2,"same-edge pair lost");
                require(!edge.changes_inside_outside,"even root parity");
                require(edge.crossings[0].owners[0].patch_index==3,"owner lost");
                require(std::abs(edge.crossings[0].edge_parameter-.28)<1e-14,"local parameter");
            }
        }
        require(total==4,"all-event count changed");
        auto unsafe=line;unsafe.crossings[0].point.x()=.25;
        require(!certified_grid_line_can_partition_3d(grid,0,unsafe,1e-12),"node root silently rounded");
        unsafe=line;unsafe.root_count_known=false;
        require(!certified_grid_line_can_partition_3d(grid,0,unsafe,1e-12),"incomplete root set accepted");
        unsafe=line;unsafe.crossings[0].feature_edge_contact=true;
        require(!certified_grid_line_can_partition_3d(grid,0,unsafe,1e-12),"feature root fast accepted");
        unsafe=line;unsafe.has_near_tangent_candidate=true;
        require(!certified_grid_line_can_partition_3d(grid,0,unsafe,1e-12),"tangent fast accepted");
        // A root can pass the length-one full-line threshold but fail the
        // length-.25 short-edge threshold. Caching must not weaken it.
        unsafe=line;
        constexpr double trans=4e-6;
        const double line_tau=std::sqrt(8e-12), edge_tau=std::sqrt(8e-12/.25);
        require(line_tau<trans && trans<edge_tau,"invalid near-tangent fixture");
        auto& root=unsafe.crossings[0];
        root.normal={trans,std::sqrt(1-trans*trans),0};
        root.transversality=trans;root.reliable_transversality_tolerance=line_tau;
        root.owners[0].normal=root.normal;root.owners[0].transversality=trans;
        root.owners[0].reliable_transversality_tolerance=line_tau;
        require(!certified_grid_line_can_partition_3d(grid,0,unsafe,1e-12),
            "full-line threshold improperly relaxed short-edge certification");
        std::cout<<"whole-grid-line all-event partition: PASS\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
