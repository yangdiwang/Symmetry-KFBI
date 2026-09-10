#include "src/geometry/nurbs_cartesian_domain_3d.hpp"
#include "src/support/geometry/trace_first_case_3d.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace kfbim::geometry3d;
using namespace kfbim::app3d;

int main()
{
    try {
        // Four N=64 edges for which the original full-BVH depth-10 retry
        // reported zero found roots but an incomplete root set.  Absence
        // of found roots (or matching endpoint labels) is NOT a miss proof.
        const auto problem=make_trace_first_python_torus_case_3d("rotate");
        constexpr double h=2.0/64.0;
        const std::array<NurbsCartesianEdgeQuery3D,4> edges{{
            {2,24,29,28,{-0.25,-0.09375,-0.125},{-0.25,-0.09375,-0.09375}},
            {0,46,36,29,{0.4375,0.125,-0.09375},{0.46875,0.125,-0.09375}},
            {0,17,28,35,{-0.46875,-0.125,0.09375},{-0.4375,-0.125,0.09375}},
            {2,40,35,35,{0.25,0.09375,0.09375},{0.25,0.09375,0.125}}
        }};
        bool default_complete_configuration=false;
        std::cout<<std::setprecision(17)<<std::boolalpha;
        for(double extent_factor:{2.0,1.0}) {
            for(int separation_depth:{6,8,12,16}) {
                // Match the production OptimizedIntersection final retry;
                // only the certificate work budget / exact acceleration
                // subdivision changes, never geometry or tolerances.
                NurbsSurfaceIntersectorOptions3D options;
                options.use_early_unique_root_certificate=true;
                options.maximum_element_extent=extent_factor*h;
                options.local_max_subdivision_depth=10;
                options.terminal_separation_subdivision_depth=separation_depth;
                const auto build_begin=std::chrono::steady_clock::now();
                NurbsSurfaceIntersector3D intersector(
                    problem.surface.geometry_model(),options);
                const double build_seconds=std::chrono::duration<double>(
                    std::chrono::steady_clock::now()-build_begin).count();
                std::cout<<"CONFIG extent_factor="<<extent_factor
                         <<" local_depth=10 separation_depth="<<separation_depth
                         <<" query_elements="<<intersector.query_element_count()
                         <<" geometry_tolerance="<<intersector.geometry_tolerance()
                         <<" build_seconds="<<build_seconds<<std::endl;
                bool all_complete=true;
                for(std::size_t e=0;e<edges.size();++e) {
                    const auto query_begin=std::chrono::steady_clock::now();
                    try {
                        // No candidate subset: each attempt queries the full BVH.
                        const auto result=intersector.intersect_cartesian_edge(edges[e]);
                        const auto& d=result.diagnostics;
                        const bool complete=
                            !grid_edge_root_set_requires_targeted_retry_3d(result);
                        const bool certified_empty=complete && result.crossings.empty();
                        all_complete=all_complete && certified_empty;
                        std::cout<<" EDGE="<<e<<" certified_empty="<<certified_empty
                            <<" roots="<<result.crossings.size()
                            <<" root_count_known="<<result.root_count_known
                            <<" parity_known="<<result.parity_known_from_roots
                            <<" near_tangent="<<result.has_near_tangent_candidate
                            <<" ambiguous_clusters="<<result.ambiguous_clusters.size()
                            <<" candidate_elements="<<d.candidate_elements
                            <<" subdivision_boxes="<<d.subdivision_boxes
                            <<" max_depth="<<d.maximum_subdivision_depth_reached
                            <<" terminal_boxes="<<d.terminal_certificate_boxes
                            <<" terminal_depth="<<d.maximum_terminal_certificate_depth_reached
                            <<" unresolved="<<d.unresolved_candidates
                            <<" closest_attempts="<<d.closest_point_attempts
                            <<" closest_failures="<<d.closest_point_failures
                            <<" terminal_misses="<<d.terminal_misses_by_closest_point
                            <<" newton_attempts="<<d.newton_attempts
                            <<" newton_iterations="<<d.newton_iterations
                            <<" seconds="<<std::chrono::duration<double>(
                                std::chrono::steady_clock::now()-query_begin).count()
                            <<std::endl;
                    } catch(const std::runtime_error& error) {
                        all_complete=false;
                        std::cout<<" EDGE="<<e<<" exception="<<error.what()<<std::endl;
                    }
                }
                if(extent_factor==2.0 && separation_depth==6)
                    default_complete_configuration=all_complete;
                std::cout<<" CONFIG_COMPLETE="<<all_complete<<std::endl;
                if(extent_factor==2.0 && separation_depth==6) {
                    // Verified by a full-BVH replay: this neighboring line
                    // has two transverse roots, unlike the four empty
                    // edges above.  Empty-box recovery must preserve both.
                    const NurbsCartesianEdgeQuery3D probe{
                        2,-1,-1,-1,{-0.2505,-0.09375,-0.125},
                        {-0.2505,-0.09375,-0.09375}};
                    const Eigen::Vector3d direction=(probe.end-probe.start).normalized();
                    const auto require_two_roots=[&](
                        const NurbsCartesianEdgeIntersections3D& roots) {
                        if(grid_edge_root_set_requires_targeted_retry_3d(roots)
                            || roots.crossings.size()!=2
                            || roots.confirmed_transverse_count!=2)
                            throw std::runtime_error(
                                "neighboring two-root control is not a certified complete pair");
                        double previous=0.0;
                        for(const auto& root:roots.crossings) {
                            const double tau=root.reliable_transversality_tolerance;
                            if(!(root.edge_parameter>previous && root.edge_parameter<1.0)
                                || !root.point.allFinite() || !root.normal.allFinite()
                                || !(std::isfinite(tau) && tau>0.0)
                                || !(root.transversality>tau)
                                || !(std::abs(root.normal.dot(direction))>tau))
                                throw std::runtime_error(
                                    "neighboring roots are not ordered, interior and reliable transverse");
                            previous=root.edge_parameter;
                        }
                    };
                    const auto result=intersector.intersect_cartesian_edge(probe);
                    require_two_roots(result);
                    std::cout<<" NEIGHBOR_PROBE complete="
                        <<!grid_edge_root_set_requires_targeted_retry_3d(result)
                        <<" roots="<<result.crossings.size()
                        <<" transverse="<<result.confirmed_transverse_count
                        <<" unresolved="<<result.diagnostics.unresolved_candidates
                        <<std::endl;
                    for(const auto& root:result.crossings)
                        std::cout<<"  PROBE_ROOT t="<<root.edge_parameter
                            <<" trans="<<root.transversality
                            <<" tau="<<root.reliable_transversality_tolerance
                            <<" residual="<<root.residual<<std::endl;
                    auto reversed=probe;
                    reversed.start=probe.end;reversed.end=probe.start;
                    const auto backward=intersector.intersect_cartesian_edge(reversed);
                    require_two_roots(backward);
                    const double point_tolerance=128.0*intersector.geometry_tolerance();
                    const double parameter_tolerance=point_tolerance/(probe.end-probe.start).norm();
                    for(std::size_t r=0;r<2;++r) {
                        const auto& forward_root=result.crossings[r];
                        const auto& reverse_root=backward.crossings[1-r];
                        if(forward_root.component!=reverse_root.component
                            || (forward_root.point-reverse_root.point).norm()>point_tolerance
                            || std::abs(forward_root.edge_parameter
                                +reverse_root.edge_parameter-1.0)>parameter_tolerance
                            || !(forward_root.normal.dot(direction)
                                *reverse_root.normal.dot(-direction)<0.0))
                            throw std::runtime_error(
                                "reversing neighboring two-root query changed events or orientation");
                    }
                    std::cout<<" NEIGHBOR_REVERSE_MATCH=true"<<std::endl;
                }
            }
        }
        // A finer acceleration mesh or larger certificate budget must not
        // hide a regression in the unchanged production default.
        if(!default_complete_configuration)
            throw std::runtime_error(
                "default 2h/separation-6 retry did not certify all four empty root sets");
        std::cout<<"torus N64 empty-edge certified replay: PASS\n";
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
