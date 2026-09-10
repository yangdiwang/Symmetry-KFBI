#include "src/geometry/nurbs_grid_line_cache_3d.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace kfbim::geometry3d {
bool certified_grid_line_can_partition_3d(
    const CartesianGrid3D& grid, int axis,
    const NurbsCartesianEdgeIntersections3D& line, double tolerance)
{
    if(axis<0 || axis>2 || !std::isfinite(tolerance) || tolerance<0)
        throw std::invalid_argument("invalid grid-line partition arguments");
    if(grid_edge_root_set_requires_targeted_retry_3d(line)
        || line.confirmed_transverse_count!=static_cast<int>(line.crossings.size())) return false;
    const double h=grid.spacing()[axis], origin=grid.origin()[axis];
    const int cells=grid.num_cells()[axis];
    // The native tolerance depends on query length. A long-line threshold
    // can be smaller than a short-edge threshold. Since the short-edge
    // geometry scale is max(element_extent,h,tolerance) >= h, this is a
    // conservative upper bound for every short-edge tolerance. A stricter
    // cache gate costs only a fallback; it must never relax certification.
    const double short_edge_tau=std::max(
        32*std::sqrt(std::numeric_limits<double>::epsilon()),
        std::sqrt(8*tolerance/h));
    const double end=origin+cells*h;
    const double guard=std::max(8*tolerance,
        128*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(origin),std::abs(end)}));
    for(const auto& root:line.crossings) {
        if(!root.point.allFinite() || !root.normal.allFinite() || root.component<0
            || root.feature_edge_contact || !std::isfinite(root.transversality)
            || !std::isfinite(root.residual) || root.residual<0
            || !std::isfinite(root.reliable_transversality_tolerance)
            || root.reliable_transversality_tolerance<=0
            || root.transversality<=std::max(short_edge_tau,root.reliable_transversality_tolerance)
            || std::abs(root.normal[axis])<=std::max(short_edge_tau,root.reliable_transversality_tolerance)) return false;
        const double s=(root.point[axis]-origin)/h;
        // A small transverse angle magnifies position sensitivity. Route
        // these node-near roots through the original endpoint-aware queries.
        const double root_guard=std::max(guard,
            8*std::max(tolerance,root.residual)/root.transversality);
        if(!(s>0 && s<cells) || std::abs(s-std::round(s))*h<=root_guard) return false;
        for(const auto& owner:root.owners)
            if(owner.feature_edge_contact || !owner.normal.allFinite()
                || !std::isfinite(owner.transversality)
                || !std::isfinite(owner.reliable_transversality_tolerance)
                || owner.reliable_transversality_tolerance<=0
                || owner.transversality<=std::max(short_edge_tau,owner.reliable_transversality_tolerance)
                || std::abs(owner.normal[axis])<=std::max(short_edge_tau,owner.reliable_transversality_tolerance)) return false;
    }
    return true;
}

NurbsCartesianEdgeIntersections3D extract_grid_line_edge_3d(
    const NurbsCartesianEdgeQuery3D& edge,
    const std::vector<NurbsSurfaceCrossing3D>& roots)
{
    if(edge.axis<0 || edge.axis>2 || !edge.start.allFinite() || !edge.end.allFinite()
        || !(edge.end[edge.axis]>edge.start[edge.axis]))
        throw std::invalid_argument("partition requires a positive-axis edge");
    for(int a=0;a<3;++a)
        if(a!=edge.axis && edge.start[a]!=edge.end[a])
            throw std::invalid_argument("partition edge is not Cartesian");
    NurbsCartesianEdgeIntersections3D result;
    std::set<int> toggled;
    for(const auto& root:roots) {
        const double coordinate=root.point[edge.axis];
        if(coordinate<=edge.start[edge.axis] || coordinate>=edge.end[edge.axis]) continue;
        auto copy=root;
        copy.edge_parameter=(coordinate-edge.start[edge.axis])/(edge.end[edge.axis]-edge.start[edge.axis]);
        result.crossings.push_back(std::move(copy));
        if(!toggled.insert(root.component).second) toggled.erase(root.component);
    }
    std::stable_sort(result.crossings.begin(),result.crossings.end(),
        [](const auto& a,const auto& b){return a.edge_parameter<b.edge_parameter;});
    result.confirmed_transverse_count=static_cast<int>(result.crossings.size());
    result.toggled_components.assign(toggled.begin(),toggled.end());
    result.changes_component_membership=!toggled.empty();
    result.changes_inside_outside=toggled.size()%2!=0;
    return result;
}
} // namespace kfbim::geometry3d
