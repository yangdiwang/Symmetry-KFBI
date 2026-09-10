#include "trace_first_resource_3d.hpp"

#include "src/geometry/native_endpoint_path_3d.hpp"
#include "src/support/cauchy/extended_tubular_cauchy_3d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>

namespace kfbim::app3d {
namespace {
using Clock = std::chrono::steady_clock;
using AnchorKey = std::pair<int, int>;
AnchorKey anchor_key(TraceFirstAnchorRef3D a) { return {static_cast<int>(a.source), a.index}; }
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
Eigen::Vector3d node_point(const CartesianGrid3D& grid, int id) {
    const auto p = grid.coord(id); return {p[0], p[1], p[2]};
}
std::pair<double, double> normalized(const NativeNurbsDensitySpace3D& density,
    const RestrictResourceAnchor3D& a) {
    const auto& p = density.surface().patches.at(a.patch_id);
    const double u = (a.u-p.domain_start_u())/(p.domain_end_u()-p.domain_start_u());
    const double v = (a.v-p.domain_start_v())/(p.domain_end_v()-p.domain_start_v());
    if (!std::isfinite(u) || !std::isfinite(v) || u<0 || u>1 || v<0 || v>1)
        throw std::invalid_argument("trace-first anchor is outside its native patch");
    return {u,v};
}
ResourceAffineRow3D make_row(const NativeDensityC0Stencil3D& a, double known) {
    std::map<int,double> entries;
    for (int j=0;j<a.count;++j) entries[a.indices[j]] += a.weights[j];
    ResourceAffineRow3D row;
    for (const auto& [i,w]:entries) {row.columns.push_back(i);row.values.push_back(w);}
    row.known=known; return row;
}
template<std::size_t N, class Weights>
ResourceAffineRow3D combine(const std::array<NativeDensityC0Stencil3D,N>& rows,
    const Weights& weights, double known) {
    std::map<int,double> entries;
    for (std::size_t a=0;a<N;++a) for (int k=0;k<rows[a].count;++k)
        entries[rows[a].indices[k]] += weights[static_cast<int>(a)]*rows[a].weights[k];
    ResourceAffineRow3D result;
    for (const auto& [i,v]:entries) {result.columns.push_back(i);result.values.push_back(v);}
    result.known=known; return result;
}

// One immutable KD tree per patch. Selection is AFTER q-dependent priority,
// unlike the old mixed planner's sheet-node premerge.
class PatchTree {
    struct Node {int id,axis,left=-1,right=-1;};
    const std::vector<RestrictResourceAnchor3D>& points_;
    std::vector<Node> nodes_;
    int root_=-1;
    int build(std::vector<int>& ids,int begin,int end,int depth) {
        if (begin==end) return -1;
        const int mid=(begin+end)/2,axis=depth%3;
        std::nth_element(ids.begin()+begin,ids.begin()+mid,ids.begin()+end,[&](int a,int b){
            const double x=points_[a].point[axis],y=points_[b].point[axis];
            return x<y || (x==y && a<b);
        });
        const int at=static_cast<int>(nodes_.size()); nodes_.push_back({ids[mid],axis});
        const int left=build(ids,begin,mid,depth+1),right=build(ids,mid+1,end,depth+1);
        nodes_[at].left=left;nodes_[at].right=right;return at;
    }
    void search(int at,const Eigen::Vector3d& x,std::pair<double,int>& best) const {
        if(at<0)return;
        const auto& n=nodes_[at]; const auto& p=points_[n.id].point;
        const double d=(x-p).squaredNorm();
        if(d<best.first || (d==best.first && n.id<best.second)) best={d,n.id};
        const double delta=x[n.axis]-p[n.axis];
        search(delta<0?n.left:n.right,x,best);
        if(delta*delta<=best.first) search(delta<0?n.right:n.left,x,best);
    }
public:
    PatchTree(const std::vector<RestrictResourceAnchor3D>& points,std::vector<int> ids):points_(points){
        root_=build(ids,0,static_cast<int>(ids.size()),0);
    }
    std::pair<double,int> nearest(const Eigen::Vector3d& x) const {
        std::pair<double,int> best{std::numeric_limits<double>::infinity(),-1};
        search(root_,x,best); return best;
    }
};
class LocalTraceLookup {
    std::vector<std::vector<int>> neighbours_;
    std::vector<std::unique_ptr<PatchTree>> trees_;
public:
    LocalTraceLookup(const NativeNurbsSurface3D& s,const std::vector<RestrictResourceAnchor3D>& a) {
        const int count=static_cast<int>(s.patches.size()); neighbours_.resize(count);trees_.resize(count);
        std::vector<std::vector<int>> ids(count);
        for(int i=0;i<static_cast<int>(a.size());++i)ids.at(a[i].patch_id).push_back(i);
        for(int p=0;p<count;++p){
            std::set<int> nb{p};
            if(p<static_cast<int>(s.topological_patch_neighbors.size()))
                nb.insert(s.topological_patch_neighbors[p].begin(),s.topological_patch_neighbors[p].end());
            else {const auto smooth=smooth_patch_neighbors_3d(s,p);nb.insert(smooth.begin(),smooth.end());}
            neighbours_[p].assign(nb.begin(),nb.end());
            trees_[p]=std::make_unique<PatchTree>(a,std::move(ids[p]));
        }
    }
    std::pair<double,int> nearest(int patch,const Eigen::Vector3d& x) const {
        std::pair<double,int> best{std::numeric_limits<double>::infinity(),-1};
        for(int p:neighbours_.at(patch)){
            const auto candidate=trees_.at(p)->nearest(x);
            if(candidate.second>=0 && (candidate.first<best.first ||
                (candidate.first==best.first && candidate.second<best.second)))best=candidate;
        }
        if(best.second<0)throw std::runtime_error("trace-first branch has no trace anchors");
        return best;
    }
    std::pair<double,int> nearest_on_owner(int patch,const Eigen::Vector3d& x) const {
        return trees_.at(patch)->nearest(x);
    }
};
bool trace_spread(TraceFirstCenterPolicy3D p) {
    return p==TraceFirstCenterPolicy3D::TracePolynomialFirst || p==TraceFirstCenterPolicy3D::TraceSpreadEventRestrict;
}
bool trace_restrict(TraceFirstCenterPolicy3D p) {
    return p==TraceFirstCenterPolicy3D::TracePolynomialFirst || p==TraceFirstCenterPolicy3D::EventSpreadTraceRestrict;
}
} // namespace

bool owner_validated_trace_candidate_3d(const RestrictResourceAnchor3D& trace,
    const RestrictResourceAnchor3D& event,const Eigen::Vector3d& target,double h,
    const OwnerValidatedTraceOptions3D& options,const Eigen::Vector3d* other){
    if(!(h>0.0) || !std::isfinite(h) || !(options.event_radius_h>0.0) ||
        !(options.target_radius_h>0.0) || !std::isfinite(options.event_radius_h) ||
        !std::isfinite(options.target_radius_h))throw std::invalid_argument("invalid owner-validated trace radii");
    if(!trace.point.allFinite() || !event.point.allFinite() || !target.allFinite() ||
        (other && !other->allFinite()))throw std::invalid_argument("nonfinite owner-validated trace location");
    return trace.patch_id==event.patch_id &&
        (trace.point-event.point).norm()<=options.event_radius_h*h &&
        (trace.point-target).norm()<=options.target_radius_h*h &&
        (!other || (trace.point-*other).norm()<=options.target_radius_h*h);
}

struct TracePolynomialCatalog3D::Impl {
    struct Center {
        Eigen::Vector3d point=Eigen::Vector3d::Zero();
        std::optional<NativeSurfaceCubicParameterJet3D> geometry3;
        std::optional<DirectCoefficientCubicCauchyPlan3D> cubic;
        std::optional<ExtendedTubularCauchyPlan3D> tube;
        std::optional<DirectCoefficientValueJetPlan3D> p2_value;
        std::optional<DirectCoefficientNormalJetPlan3D> p2_normal;
        CubicValueJet3D known_value=CubicValueJet3D::Zero();
        CubicNormalJet3D known_normal=CubicNormalJet3D::Zero();
    };
    const NativeNurbsDensitySpace3D& density;
    const std::vector<RestrictResourceAnchor3D>& traces;
    const std::vector<RestrictResourceAnchor3D>& events;
    NativeDensityField3D unknown;
    ResourceKnownAmbientCallback3D known;
    double mean;
    TraceFirstChart3D chart;
    bool reuse;
    TracePolynomialCatalogStatistics3D stats;
    std::map<AnchorKey,Center> centers;
    using RowKey=std::tuple<int,int,int,int>;
    std::map<RowKey,std::pair<Eigen::Vector3d,ResourceAffineRow3D>> rows;
    const RestrictResourceAnchor3D& anchor(TraceFirstAnchorRef3D a) const {
        return (a.source==RestrictAnchorSource3D::Trace?traces:events).at(a.index);
    }
    void cubic(Center& c,TraceFirstAnchorRef3D id) {
        if(c.cubic)return;
        const auto& a=anchor(id);const auto [u,v]=normalized(density,a);
        c.geometry3=native_surface_cubic_parameter_jet_3d(density,a.patch_id,u,v);
        const auto frame=make_local_orthonormal_frame_3d(c.geometry3->lower.normal,c.geometry3->lower.x_u);
        c.cubic=build_direct_coefficient_cubic_cauchy_plan_3d(density,a.patch_id,u,v,*c.geometry3,frame);
        c.point=c.cubic->center;
        const auto data=known(c.cubic->center);
        if(unknown==NativeDensityField3D::ValueTrace){
            c.known_normal=known_cubic_neumann_jet_3d(data,c.cubic->graph,frame);c.known_normal[0]-=mean;
        }else c.known_value=known_cubic_dirichlet_jet_3d(data,c.cubic->graph,frame);
        if(chart==TraceFirstChart3D::ExtendedTube)
            c.tube=build_extended_tubular_cauchy_plan_3d(density.surface().patches.at(a.patch_id),*c.geometry3,*c.cubic);
        ++stats.p3_centers;
    }
    void quadratic(Center& c,TraceFirstAnchorRef3D id) {
        if(c.p2_value)return;
        // The tubular adapter currently obtains its lower maps from a cubic
        // geometry transfer. This work is counted honestly as a P3 build.
        if(chart==TraceFirstChart3D::ExtendedTube)cubic(c,id);
        if(c.cubic){
            c.p2_value=c.cubic->lower_value_plan();c.p2_normal=c.cubic->lower_normal_plan();
            ++stats.p2_from_p3;
        }else{
            const auto& a=anchor(id);const auto [u,v]=normalized(density,a);
            const auto g=native_surface_parameter_jet_3d(density,a.patch_id,u,v);
            c.point=g.point;
            const auto frame=make_local_orthonormal_frame_3d(g.normal,g.x_u);
            c.p2_value=build_direct_coefficient_value_jet_plan_3d(density,a.patch_id,u,v,g,frame);
            c.p2_normal=build_direct_coefficient_normal_jet_plan_3d(density,a.patch_id,u,v,g,frame);
            const auto data=known(g.point);
            if(unknown==NativeDensityField3D::NormalTrace){
                KnownDirichletValueGradientHessian3D j;
                j.value=data.value;j.ambient_gradient=data.gradient;j.ambient_hessian=data.hessian;
                c.known_value.head<6>()=known_dirichlet_jet_from_ambient_derivatives_3d(j,g,frame);
            }else{
                auto n=surface_normal_parameter_jet_3d(g);
                if(n.normal.dot(frame.normal)<0){n.normal=-n.normal;n.normal_u=-n.normal_u;n.normal_v=-n.normal_v;}
                Eigen::Vector2d parameter;
                parameter << (data.hessian*g.x_u).dot(n.normal)+data.gradient.dot(n.normal_u),
                    (data.hessian*g.x_v).dot(n.normal)+data.gradient.dot(n.normal_v);
                const auto tangent=parameter_gradient_to_tangent_gradient_3d(g,frame,parameter);
                c.known_normal.head<3>() << data.gradient.dot(n.normal)-mean,tangent.x(),tangent.y();
            }
        }
        ++stats.p2_centers;
    }
};

TracePolynomialCatalog3D::TracePolynomialCatalog3D(const NativeNurbsDensitySpace3D& density,
    const std::vector<RestrictResourceAnchor3D>& trace,const std::vector<RestrictResourceAnchor3D>& events,
    NativeDensityField3D unknown,ResourceKnownAmbientCallback3D known,double mean,TraceFirstChart3D chart,bool reuse)
    :impl_(std::make_unique<Impl>(Impl{density,trace,events,unknown,std::move(known),mean,chart,reuse})) {
    if(!impl_->known || !std::isfinite(mean))throw std::invalid_argument("trace catalog requires fixed finite known data");
}
TracePolynomialCatalog3D::~TracePolynomialCatalog3D()=default;
TracePolynomialCatalog3D::TracePolynomialCatalog3D(TracePolynomialCatalog3D&&) noexcept=default;
TracePolynomialCatalog3D& TracePolynomialCatalog3D::operator=(TracePolynomialCatalog3D&&) noexcept=default;
void TracePolynomialCatalog3D::prepare(int degree,TraceFirstAnchorRef3D a){
    if(degree!=2 && degree!=3)throw std::invalid_argument("trace polynomial degree must be two or three");
    auto& c=impl_->centers[anchor_key(a)];
    if(degree==3)impl_->cubic(c,a);else impl_->quadratic(c,a);
}
ResourceAffineRow3D TracePolynomialCatalog3D::evaluate(int degree,TraceFirstAnchorRef3D a,
    const Eigen::Vector3d& target,int node){
    if(!target.allFinite())throw std::invalid_argument("non-finite trace polynomial target");
    const Impl::RowKey key{degree,static_cast<int>(a.source),a.index,node};
    if(node>=0 && impl_->reuse){
        const auto found=impl_->rows.find(key);
        if(found!=impl_->rows.end()){
            if((found->second.first.array()!=target.array()).any())
                throw std::invalid_argument("trace row-cache grid ID reused at a different point");
            ++impl_->stats.row_cache_hits;return found->second.second;
        }
    }
    prepare(degree,a);auto& c=impl_->centers.at(anchor_key(a));ResourceAffineRow3D result;
    if(impl_->chart==TraceFirstChart3D::ExtendedTube){
        const auto weights=c.tube->weights(target,degree);
        impl_->stats.maximum_projection_residual=std::max(impl_->stats.maximum_projection_residual,weights.projection_residual);
        if(!weights.projection_converged)++impl_->stats.projection_not_converged;
        result=impl_->unknown==NativeDensityField3D::ValueTrace
            ?combine(c.cubic->value_rows,weights.value,(weights.normal*c.known_normal).value())
            :combine(c.cubic->normal_rows,weights.normal,(weights.value*c.known_value).value());
    }else if(degree==3){
        const Eigen::Vector3d d=target-c.cubic->center;
        result=impl_->unknown==NativeDensityField3D::ValueTrace
            ?make_row(c.cubic->compose_value_row(d),(c.cubic->normal_weights(d)*c.known_normal).value())
            :make_row(c.cubic->compose_normal_row(d),(c.cubic->value_weights(d)*c.known_value).value());
    }else{
        const auto weights=cauchy_polynomial_weights_3d(c.point,c.p2_value->frame,c.p2_value->graph_hessian,target);
        result=impl_->unknown==NativeDensityField3D::ValueTrace
            ?make_row(c.p2_value->compose_value_row(weights.w0),weights.w1.dot(c.known_normal.head<3>()))
            :make_row(c.p2_normal->compose_normal_row(weights.w1),weights.w0.dot(c.known_value.head<6>()));
    }
    if(degree==2)++impl_->stats.p2_row_evaluations;else ++impl_->stats.p3_row_evaluations;
    if(node>=0 && impl_->reuse)impl_->rows.emplace(key,std::make_pair(target,result));
    return result;
}
NativeDensityC0Stencil3D TracePolynomialCatalog3D::trace_value_basis(int id) const {
    const auto& a=impl_->traces.at(id);const auto [u,v]=normalized(impl_->density,a);
    return impl_->density.c0_basis_stencil(a.patch_id,u,v);
}
const TracePolynomialCatalogStatistics3D& TracePolynomialCatalog3D::statistics() const{return impl_->stats;}

TraceFirstGeometryPlan3D build_trace_first_geometry_plan_3d(
    const CartesianGrid3D& grid,const GridPair3D& pair,const NativeNurbsSurface3D& surface,
    const LaplaceCorrectionSupport3D& support,std::vector<RestrictResourceAnchor3D> traces,
    TensorProductCoverKind3D kind,RestrictResourceMode3D engine,TraceFirstCenterPolicy3D policy,
    TraceFirstEventMode3D event_mode,OwnerValidatedTraceOptions3D owner_options){
    if(engine==RestrictResourceMode3D::GuardedReuseBySheetNode)
        throw std::invalid_argument("trace-first does not implement a guarded DDA fallback");
    const auto begin=Clock::now();TraceFirstGeometryPlan3D result;
    result.center_policy=policy;result.event_mode=event_mode;
    result.resource=build_resource_geometry_skeleton_3d(grid,pair,surface,support,std::move(traces),kind,true,
        owner_options.cover_endpoint_snap);
    auto& geometry=result.resource;LocalTraceLookup lookup(surface,geometry.trace_anchors);
    const double h=grid.spacing()[0];
    const bool owner_validated=policy==TraceFirstCenterPolicy3D::OwnerValidatedTraceFirst;
    if(event_mode==TraceFirstEventMode3D::Python93LastEvent &&
        policy!=TraceFirstCenterPolicy3D::EventCentered && !owner_validated)
        throw std::invalid_argument("Python93LastEvent requires event-centered or owner-validated centers");
    if(owner_validated || event_mode==TraceFirstEventMode3D::Python93LastEvent)
        for(const auto& op:support.crossing_ops)if(!has_grid_edge_event_3d(op))
            throw std::invalid_argument("93 mapping requires the complete certified Cartesian event catalog");
    // A selection changes only the polynomial center. The event catalog,
    // event IDs, native owners and transition signs remain immutable.
    auto select_owner_trace=[&](int event,const Eigen::Vector3d& target,int preferred,
                                const Eigen::Vector3d* other)->TraceFirstAnchorRef3D {
        const auto& e=geometry.spread_anchors.at(event);
        int candidate=preferred;
        if(candidate<0 || !owner_validated_trace_candidate_3d(geometry.trace_anchors.at(candidate),e,target,h,owner_options,other))
            candidate=lookup.nearest_on_owner(e.patch_id,e.point).second;
        if(candidate>=0 && owner_validated_trace_candidate_3d(geometry.trace_anchors[candidate],e,target,h,owner_options,other)){
            const auto& a=geometry.trace_anchors[candidate];
            result.maximum_trace_event_distance_h=std::max(result.maximum_trace_event_distance_h,(a.point-e.point).norm()/h);
            double dt=(a.point-target).norm();if(other)dt=std::max(dt,(a.point-*other).norm());
            result.maximum_trace_target_distance_h=std::max(result.maximum_trace_target_distance_h,dt/h);
            return {RestrictAnchorSource3D::Trace,candidate};
        }
        return {RestrictAnchorSource3D::SpreadCrossing,event};
    };
    result.spread_active.assign(support.crossing_ops.size(),true);
    result.spread_centers.resize(support.crossing_ops.size());
    if(event_mode==TraceFirstEventMode3D::PythonEndpointEvent){
        std::fill(result.spread_active.begin(),result.spread_active.end(),false);
        std::map<std::pair<int,int>,std::pair<double,int>> selected;
        for(int k=0;k<static_cast<int>(support.crossing_ops.size());++k){
            const auto& op=support.crossing_ops[k];
            if((pair.domain_label(op.rhs_node)>0)==(pair.domain_label(op.correction_node)>0))continue;
            const auto& a=geometry.spread_anchors.at(geometry.crossing_for_op[k]);
            const double d=(node_point(grid,op.correction_node)-a.point).squaredNorm();
            const auto key=std::make_pair(op.rhs_node,op.correction_node);
            auto found=selected.find(key);
            if(found==selected.end() || d<found->second.first)selected[key]={d,k};
        }
        for(const auto& [key,best]:selected){(void)key;result.spread_active[best.second]=true;}
    }
    if(event_mode==TraceFirstEventMode3D::Python93LastEvent){
        std::fill(result.spread_active.begin(),result.spread_active.end(),false);
        // Python queries the positive-axis undirected edge once and uses its
        // last event for BOTH directions, not each direction's own last hit.
        std::map<std::pair<int,int>,std::pair<double,int>> selected;
        for(int k=0;k<static_cast<int>(support.crossing_ops.size());++k){
            const auto& op=support.crossing_ops[k];
            if((pair.domain_label(op.rhs_node)>0)==(pair.domain_label(op.correction_node)>0))continue;
            const int positive=std::max(op.rhs_node,op.correction_node);
            const auto& a=geometry.spread_anchors.at(geometry.crossing_for_op[k]);
            const double d=(node_point(grid,positive)-a.point).squaredNorm();
            const auto key=std::make_pair(op.rhs_node,op.correction_node);
            const auto at=selected.find(key);
            if(at==selected.end() || d<at->second.first)selected[key]={d,k};
        }
        for(const auto& [key,best]:selected){(void)key;result.spread_active[best.second]=true;}
        result.python93_discarded_spread_operations=support.crossing_ops.size()-selected.size();
    }
    for(int k=0;k<static_cast<int>(support.crossing_ops.size());++k){
        const int event=geometry.crossing_for_op[k];
        result.spread_centers[k]={RestrictAnchorSource3D::SpreadCrossing,event};
        if(!result.spread_active[k])continue;
        if(owner_validated){
            const auto& op=support.crossing_ops[k];const auto a=node_point(grid,op.rhs_node),b=node_point(grid,op.correction_node);
            result.spread_centers[k]=select_owner_trace(event,a,-1,&b);
            if(result.spread_centers[k].source==RestrictAnchorSource3D::Trace)++result.spread_trace_uses;
            else ++result.spread_event_fallbacks;
            continue;
        }
        if(!trace_spread(policy))continue;
        const auto best=lookup.nearest(geometry.spread_anchors[event].patch_id,node_point(grid,support.crossing_ops[k].correction_node));
        if(best.first<=2.75*2.75*h*h){result.spread_centers[k]={RestrictAnchorSource3D::Trace,best.second};++result.spread_trace_uses;}
        else ++result.spread_event_fallbacks;
    }
    auto& plan=geometry.restrict_plan;plan.mode=engine;plan.visit_to_request.assign(geometry.visits.size(),-1);
    const bool reuse=engine!=RestrictResourceMode3D::ReferencePerVisit;
    std::map<std::pair<int,int>,int> merged;
    using RecipeKey=std::tuple<int,bool,std::vector<std::tuple<int,int,double>>>;
    std::map<RecipeKey,int> owner_merged;
    std::set<int> used_trace,used_event;
    std::map<std::tuple<int,double,double>,int> event_ids;
    for(int i=0;i<static_cast<int>(geometry.spread_anchors.size());++i){const auto& a=geometry.spread_anchors[i];event_ids[{a.patch_id,a.u,a.v}]=i;}
    std::unique_ptr<geometry3d::NurbsSurfaceIntersector3D> intersector;
    struct PathRecipe {std::vector<TraceFirstSignedTerm3D> terms;int incoming=-1;};
    std::map<std::pair<int,int>,PathRecipe> path_cache;
    if(!trace_restrict(policy))intersector=std::make_unique<geometry3d::NurbsSurfaceIntersector3D>(surface.geometry_model());
    for(int i=0;i<static_cast<int>(geometry.visits.size());++i){
        const auto& visit=geometry.visits[i];const auto& q=geometry.trace_anchors.at(visit.trace_center_id);
        ++plan.counts.support_visits;
        const bool wrong=visit.desired_inside!=visit.actual_inside;
        if(wrong)++plan.counts.wrong_side_visits;
        if(!wrong && (trace_restrict(policy) || event_mode!=TraceFirstEventMode3D::LocalAllEvent))continue;
        RestrictResourceRequest3D request;request.sheet_id=q.sheet_id;request.grid_full_id=visit.grid_full_id;request.grid_point=visit.grid_point;
        std::vector<TraceFirstSignedTerm3D> recipe;
        if(trace_restrict(policy)){
            const double dq=(visit.grid_point-q.point).squaredNorm();int id=visit.trace_center_id;double d=dq;
            if(dq<=2.25*2.25*h*h)++result.current_trace_requests;
            else {const auto best=lookup.nearest(q.patch_id,visit.grid_point);d=best.first;id=best.second;++result.nearest_trace_requests;}
            request.anchor_index=id;request.distance=std::sqrt(d);request.nearest_trace_distance=request.distance;
            request.too_far=request.distance>2.75*h;
            if(request.too_far)++result.far_trace_requests;
            if(reuse){const auto found=merged.find({id,visit.grid_full_id});if(found!=merged.end()){plan.visit_to_request[i]=found->second;continue;}}
            recipe.push_back({{RestrictAnchorSource3D::Trace,id},1.0});
            merged[{id,visit.grid_full_id}]=static_cast<int>(plan.requests.size());
        }else{
            const auto key=std::make_pair(visit.trace_center_id,visit.grid_full_id);
            auto found=path_cache.find(key);
            if(found==path_cache.end()){
                geometry3d::NativeEndpointQueryOptions3D options;options.expected_start_inside=visit.actual_inside;
                const auto path=intersector->intersect_segment_to_native_endpoint(visit.grid_point,
                    {q.patch_id,q.u,q.v},options,geometry3d::PathCertificationBudget3D{});
                if(path.status!=geometry3d::PathStatus3D::Certified || !path.coverage.complete)
                    throw std::runtime_error("trace-first event reference: native support path was not certified: "+path.message);
                ++result.certified_support_paths;
                PathRecipe p;p.incoming=static_cast<int>(visit.actual_inside);
                for(const auto& root:path.open_roots){
                    if(!root.transition.certified)throw std::runtime_error("trace-first event path has uncertified transition");
                    if(root.transition.sign==0)continue;
                    if(std::abs(root.transition.sign)!=1)throw std::runtime_error("invalid certified path transition sign");
                    const auto& r=root.representative;const auto owner=std::make_tuple(r.patch_index,r.u,r.v);
                    auto at=event_ids.find(owner);
                    if(at==event_ids.end()){
                        const int id=static_cast<int>(geometry.spread_anchors.size());
                        RestrictResourceAnchor3D a;a.point=r.point;a.patch_id=r.patch_index;a.u=r.u;a.v=r.v;a.crossing_id=id;
                        const auto component=smooth_patch_component(surface,a.patch_id);
                        a.sheet_id=*std::min_element(component.begin(),component.end());
                        geometry.spread_anchors.push_back(a);at=event_ids.emplace(owner,id).first;
                    }
                    p.terms.push_back({{RestrictAnchorSource3D::SpreadCrossing,at->second},static_cast<double>(root.transition.sign)});
                    p.incoming+=root.transition.sign;
                }
                if(p.incoming!=0 && p.incoming!=1)throw std::runtime_error("native path transitions disagree with starting grid label");
                if(p.terms.size()>1)++result.multi_event_support_paths;
                found=path_cache.emplace(key,std::move(p)).first;
            }else ++result.support_path_cache_hits;
            request.anchor_index=visit.trace_center_id;
            if(event_mode==TraceFirstEventMode3D::LocalAllEvent){
                request.signed_extension=true;recipe=found->second.terms;
                const int qsign=static_cast<int>(visit.desired_inside)-found->second.incoming;
                if(qsign)recipe.push_back({{RestrictAnchorSource3D::Trace,visit.trace_center_id},static_cast<double>(qsign)});
            }else{
                const auto selected=found->second.terms.empty()?TraceFirstAnchorRef3D{RestrictAnchorSource3D::Trace,visit.trace_center_id}:found->second.terms.front().anchor;
                request.anchor_source=selected.source;request.anchor_index=selected.index;recipe.push_back({selected,1.0});
            }
            if(owner_validated){
                for(auto& term:recipe){
                    if(term.anchor.source!=RestrictAnchorSource3D::SpreadCrossing)continue;
                    term.anchor=select_owner_trace(term.anchor.index,visit.grid_point,visit.trace_center_id,nullptr);
                    if(term.anchor.source==RestrictAnchorSource3D::Trace){
                        ++result.owner_validated_requests;
                        if(term.anchor.index==visit.trace_center_id)++result.current_trace_requests;
                        else ++result.nearest_trace_requests;
                    }else ++result.restrict_event_fallbacks;
                }
                if(recipe.size()==1){request.anchor_source=recipe[0].anchor.source;request.anchor_index=recipe[0].anchor.index;}
            }
        }
        // The request cache is populated only AFTER a real event selected an
        // owner and the owner/radius guard selected each polynomial center.
        // Compound all-event recipes keep their signs and identities in the
        // key; they must not collapse to just (sheet,grid) or (q,grid).
        if(owner_validated && reuse){
            std::vector<std::tuple<int,int,double>> terms;
            for(const auto& term:recipe)terms.emplace_back(static_cast<int>(term.anchor.source),term.anchor.index,term.sign);
            RecipeKey key{visit.grid_full_id,request.signed_extension,std::move(terms)};
            const auto at=owner_merged.find(key);
            if(at!=owner_merged.end()){plan.visit_to_request[i]=at->second;continue;}
            owner_merged.emplace(std::move(key),static_cast<int>(plan.requests.size()));
        }
        for(const auto& term:recipe)if(term.anchor.source==RestrictAnchorSource3D::Trace)used_trace.insert(term.anchor.index);else used_event.insert(term.anchor.index);
        plan.visit_to_request[i]=static_cast<int>(plan.requests.size());
        if(request.anchor_source==RestrictAnchorSource3D::Trace)++plan.counts.trace_requests;else ++plan.counts.spread_requests;
        if(request.too_far)++plan.counts.too_far_requests;
        plan.requests.push_back(request);result.restrict_recipes.push_back(std::move(recipe));
    }
    plan.counts.requests=plan.requests.size();plan.counts.unique_trace_centers=used_trace.size();plan.counts.unique_spread_centers=used_event.size();
    plan.required_crossing_ids.assign(used_event.begin(),used_event.end());
    result.support_paths_certified=!trace_restrict(policy);
    geometry.planning_seconds=elapsed(begin);return result;
}

ResourceBvpOperators3D build_trace_first_bvp_operators_3d(
    const TraceFirstGeometryPlan3D& geometry,const CartesianGrid3D& grid,const GridPair3D&,
    const LaplaceCorrectionSupport3D& support,const NativeNurbsDensitySpace3D& density,
    NativeDensityField3D unknown,const ResourceKnownAmbientCallback3D& known,double mean,
    TraceFirstChart3D chart,TraceFirstBuildStatistics3D* extra){
    const auto& g=geometry.resource;
    if(geometry.spread_centers.size()!=support.crossing_ops.size() || geometry.spread_active.size()!=support.crossing_ops.size()
        || geometry.restrict_recipes.size()!=g.restrict_plan.requests.size())throw std::invalid_argument("trace-first plan/context size mismatch");
    ResourceBvpOperators3D result;auto& stats=result.statistics;
    stats.planning=g.restrict_plan.counts;stats.planning_seconds=g.planning_seconds;
    TracePolynomialCatalog3D catalog(density,g.trace_anchors,g.spread_anchors,unknown,known,mean,chart,
        g.restrict_plan.mode!=RestrictResourceMode3D::ReferencePerVisit);
    const auto spread_start=Clock::now();std::vector<Eigen::Triplet<double>> triplets;
    result.known_spread=Eigen::VectorXd::Zero(grid.num_dofs());std::size_t active=0;
    for(int i=0;i<static_cast<int>(support.crossing_ops.size());++i){
        if(!geometry.spread_active[i])continue;++active;
        const auto& op=support.crossing_ops[i];
        const auto row=catalog.evaluate(3,geometry.spread_centers[i],node_point(grid,op.correction_node),op.correction_node);
        const double scale=static_cast<double>(op.side_delta)*op.stencil_weight;
        for(std::size_t k=0;k<row.columns.size();++k)triplets.emplace_back(op.rhs_node,row.columns[k],scale*row.values[k]);
        result.known_spread[op.rhs_node]+=scale*row.known;
    }
    result.S.resize(grid.num_dofs(),density.c0_coefficient_count());result.S.setFromTriplets(triplets.begin(),triplets.end());result.S.makeCompressed();
    std::vector<Eigen::Triplet<double>>().swap(triplets);stats.spread_seconds=elapsed(spread_start);
    const auto restrict_start=Clock::now();std::vector<ResourceAffineRow3D> rows;rows.reserve(g.restrict_plan.requests.size());
    for(std::size_t i=0;i<g.restrict_plan.requests.size();++i){
        const auto& request=g.restrict_plan.requests[i];std::map<int,double> combined;ResourceAffineRow3D row;
        for(const auto& term:geometry.restrict_recipes[i]){
            const auto part=catalog.evaluate(2,term.anchor,request.grid_point,request.grid_full_id);
            for(std::size_t k=0;k<part.columns.size();++k)combined[part.columns[k]]+=term.sign*part.values[k];
            row.known+=term.sign*part.known;
        }
        for(const auto& [col,v]:combined){row.columns.push_back(col);row.values.push_back(v);}rows.push_back(std::move(row));
    }
    std::vector<ResourceTraceSampleJumpRows3D> samples(g.trace_anchors.size());
    for(int q=0;q<static_cast<int>(g.trace_anchors.size());++q){
        const auto basis=catalog.trace_value_basis(q);
        for(int k=0;k<basis.count;++k)triplets.emplace_back(q,basis.indices[k],basis.weights[k]);
        for(int side=0;side<2;++side)for(int k=0;k<3;++k)
            samples[q][3*side+k]=catalog.evaluate(2,{RestrictAnchorSource3D::Trace,q},g.stencils[q].sides[side].sample_points[k]);
    }
    result.trace_basis.resize(g.trace_anchors.size(),density.c0_coefficient_count());
    result.trace_basis.setFromTriplets(triplets.begin(),triplets.end());result.trace_basis.makeCompressed();
    result.restrict=assemble_resource_restrict_3d(g.restrict_plan,g.stencils,g.visits,rows,samples,grid.num_dofs(),density.c0_coefficient_count());
    const auto& c=catalog.statistics();stats.spread_centers=c.p3_centers;stats.p2_centers=c.p2_centers;stats.p2_from_p3=c.p2_from_p3;
    stats.endpoint_row_hits=c.row_cache_hits;stats.p2_row_evaluations=c.p2_row_evaluations;stats.retained_crossing_centers=g.restrict_plan.required_crossing_ids.size();
    stats.restrict_seconds=elapsed(restrict_start);stats.temporary_centers_after_build=0;
    if(extra){extra->catalog=c;extra->active_spread_operations=active;}
    return result;
}

} // namespace kfbim::app3d
