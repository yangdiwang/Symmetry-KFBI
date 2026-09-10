#include "trace93_resource_3d.hpp"

#include "src/support/cauchy/extended_tubular_cauchy_3d.hpp"
#include "src/support/density/trace93_density_layout_3d.hpp"
#include "src/support/geometry/trace93_case_3d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace kfbim::app3d {
namespace {
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
Eigen::Vector3d point(const CartesianGrid3D& grid,int id){const auto p=grid.coord(id);return {p[0],p[1],p[2]};}
template<std::size_t N,class Weights>
ResourceAffineRow3D combine_rows(const std::array<NativeDensityC0Stencil3D,N>& rows,
    const Weights& weights,double known){
    std::map<int,double> entries;
    for(std::size_t j=0;j<N;++j)for(int k=0;k<rows[j].count;++k)
        entries[rows[j].indices[k]]+=weights[static_cast<int>(j)]*rows[j].weights[k];
    ResourceAffineRow3D row;row.known=known;
    for(const auto& [i,w]:entries){row.columns.push_back(i);row.values.push_back(w);}
    return row;
}
KnownAmbientThird3D known_data(const Trace93Case3D& problem,const Eigen::Vector3d& x){
    const auto jet=problem.evaluate(x);KnownAmbientThird3D result;
    result.value=jet.value;result.gradient=jet.gradient;result.hessian=jet.hessian;result.third=jet.third;
    return result;
}
} // namespace

struct Trace93PolynomialCatalog3D::Impl {
    struct Center {
        int patch=-1;
        Eigen::Vector2d analysis_uv=Eigen::Vector2d::Zero();
        DirectCoefficientCubicCauchyPlan3D physical;
        std::optional<ExtendedTubularCauchyPlan3D> angular_tube;
        CubicValueJet3D known_value=CubicValueJet3D::Zero();
        CubicNormalJet3D known_normal=CubicNormalJet3D::Zero();
        bool used_p2=false;
    };
    const Trace93Case3D& problem;
    const Trace93DensityLayout3D& density;
    const std::vector<RestrictResourceAnchor3D>& traces;
    const std::vector<RestrictResourceAnchor3D>& events;
    bool neumann;
    double mean;
    bool reuse;
    TracePolynomialCatalogStatistics3D stats;
    using AnchorKey=std::pair<int,int>;
    using RowKey=std::tuple<int,int,int,int>;
    std::map<AnchorKey,Center> centers;
    std::map<RowKey,std::pair<Eigen::Vector3d,ResourceAffineRow3D>> rows;

    Center& get(TraceFirstAnchorRef3D id){
        const AnchorKey key{static_cast<int>(id.source),id.index};
        const auto old=centers.find(key);if(old!=centers.end())return old->second;
        const auto& a=(id.source==RestrictAnchorSource3D::Trace?traces:events).at(id.index);
        Center c;c.patch=a.patch_id;
        if(id.source==RestrictAnchorSource3D::Trace){
            const auto& q=density.traces.at(id.index);
            if(q.patch!=a.patch_id || (q.point-a.point).norm()>1e-9)
                throw std::invalid_argument("93 trace catalog/layout anchor ordering mismatch");
            c.analysis_uv={q.u,q.v};
        }else c.analysis_uv=problem.world_to_analysis_uv(a.patch_id,a.point);
        // Never feed the native event's rational u/v into the angle spline.
        const auto g=density.geometry_jet(a.patch_id,c.analysis_uv.x(),c.analysis_uv.y());
        c.physical=density.cauchy_plan(a.patch_id,c.analysis_uv.x(),c.analysis_uv.y());
        const double mismatch=(g.lower.point-a.point).norm();
        if(!std::isfinite(mismatch) || mismatch>2e-8*(1+a.point.norm()))
            throw std::runtime_error("93 event-to-analysis chart does not reproduce the certified point");
        const auto known=known_data(problem,c.physical.center);
        if(neumann){c.known_normal=known_cubic_neumann_jet_3d(known,c.physical.graph,c.physical.frame);c.known_normal[0]-=mean;}
        else c.known_value=known_cubic_dirichlet_jet_3d(known,c.physical.graph,c.physical.frame);
        const auto& metadata=problem.analysis_patches.at(a.patch_id);
        if(!metadata.planar){
            if(metadata.kind!=Trace93PatchKind3D::CylinderSide)
                throw std::invalid_argument("93 correction chart is neither planar nor an angular cylinder side");
            // The constructor performs a purely analytic jet composition.
            // Its patch argument is ONLY stored, never evaluated here. DO NOT
            // call angular_tube.weights(): that method projects onto NURBS.
            // Below, evaluation uses the Python angle/axial analysis chart.
            c.angular_tube=build_extended_tubular_cauchy_plan_3d(
                problem.surface.patches.at(a.patch_id),g,c.physical);
        }
        // P2 obtains the degree <=2 part of this analytic transfer. Count the
        // cubic construction even when the center is only requested by P2.
        ++stats.p3_centers;
        return centers.emplace(key,std::move(c)).first->second;
    }
    std::pair<Eigen::Matrix<double,1,10>,Eigen::Matrix<double,1,6>> weights(
        const Center& c,const Eigen::Vector3d& target,int degree){
        if(!c.angular_tube)return {c.physical.value_weights(target-c.physical.center,degree),
                                  c.physical.normal_weights(target-c.physical.center,degree)};
        const auto uv=problem.world_to_analysis_uv(c.patch,target,c.analysis_uv);
        const auto projected=problem.analysis_at(c.patch,uv.x(),uv.y(),1);
        const Eigen::Vector3d n=projected.d[1][0].cross(projected.d[0][1]).normalized();
        const Eigen::Vector3d d=target-projected.d[0][0];const double r=d.dot(n);
        const double residual=(d-r*n).norm();
        stats.maximum_projection_residual=std::max(stats.maximum_projection_residual,residual);
        if(!uv.allFinite() || !std::isfinite(residual) || residual>1e-9*(1+target.norm()))
            throw std::runtime_error("93 analytic cylinder projection failed its tangential residual check");
        const auto& powers=cubic_cauchy_powers_3d();Eigen::Matrix<double,1,20> monomials;
        for(int j=0;j<20;++j)monomials[j]=powers[j].degree()<=degree?
            std::pow(uv.x()-c.analysis_uv.x(),powers[j].s)*std::pow(uv.y()-c.analysis_uv.y(),powers[j].t)*
            std::pow(c.angular_tube->normal_orientation*r,powers[j].r):0.0;
        return {monomials*c.angular_tube->value_map,monomials*c.angular_tube->normal_map};
    }
};

Trace93PolynomialCatalog3D::Trace93PolynomialCatalog3D(const Trace93Case3D& problem,
    const Trace93DensityLayout3D& density,const std::vector<RestrictResourceAnchor3D>& traces,
    const std::vector<RestrictResourceAnchor3D>& events,bool neumann,double mean,bool reuse)
    :impl_(std::make_unique<Impl>(Impl{problem,density,traces,events,neumann,mean,reuse})){
    if(density.problem!=&problem || density.neumann!=neumann || !std::isfinite(mean) ||
        density.traces.size()!=traces.size())throw std::invalid_argument("93 catalog requires its own analysis layout and fixed BVP");
}
Trace93PolynomialCatalog3D::~Trace93PolynomialCatalog3D()=default;
Trace93PolynomialCatalog3D::Trace93PolynomialCatalog3D(Trace93PolynomialCatalog3D&&) noexcept=default;
Trace93PolynomialCatalog3D& Trace93PolynomialCatalog3D::operator=(Trace93PolynomialCatalog3D&&) noexcept=default;
void Trace93PolynomialCatalog3D::prepare(int degree,TraceFirstAnchorRef3D id){
    if(degree!=2 && degree!=3)throw std::invalid_argument("93 polynomial degree must be two or three");
    auto& c=impl_->get(id);
    if(degree==2 && !c.used_p2){c.used_p2=true;++impl_->stats.p2_centers;++impl_->stats.p2_from_p3;}
}
ResourceAffineRow3D Trace93PolynomialCatalog3D::evaluate(int degree,TraceFirstAnchorRef3D id,
    const Eigen::Vector3d& target,int node){
    if(!target.allFinite())throw std::invalid_argument("nonfinite 93 polynomial target");
    const Impl::RowKey key{degree,static_cast<int>(id.source),id.index,node};
    if(node>=0 && impl_->reuse){const auto old=impl_->rows.find(key);if(old!=impl_->rows.end()){
        if((old->second.first.array()!=target.array()).any())throw std::invalid_argument("93 row-cache node reused at a different location");
        ++impl_->stats.row_cache_hits;return old->second.second;
    }}
    prepare(degree,id);const auto& c=impl_->get(id);const auto w=impl_->weights(c,target,degree);
    const auto row=impl_->neumann?combine_rows(c.physical.value_rows,w.first,(w.second*c.known_normal).value()):
        combine_rows(c.physical.normal_rows,w.second,(w.first*c.known_value).value());
    if(degree==2)++impl_->stats.p2_row_evaluations;else ++impl_->stats.p3_row_evaluations;
    if(node>=0 && impl_->reuse)impl_->rows.emplace(key,std::make_pair(target,row));
    return row;
}
const TracePolynomialCatalogStatistics3D& Trace93PolynomialCatalog3D::statistics() const{return impl_->stats;}

ResourceBvpOperators3D build_trace93_bvp_operators_3d(
    const TraceFirstGeometryPlan3D& geometry,const CartesianGrid3D& grid,const GridPair3D&,
    const LaplaceCorrectionSupport3D& support,const Trace93Case3D& problem,
    const Trace93DensityLayout3D& density,bool neumann,double mean,TraceFirstBuildStatistics3D* extra){
    const auto& g=geometry.resource;
    if(geometry.spread_centers.size()!=support.crossing_ops.size() || geometry.spread_active.size()!=support.crossing_ops.size()
        || geometry.restrict_recipes.size()!=g.restrict_plan.requests.size())throw std::invalid_argument("93 operator plan/context mismatch");
    if(geometry.center_policy!=TraceFirstCenterPolicy3D::OwnerValidatedTraceFirst &&
        geometry.center_policy!=TraceFirstCenterPolicy3D::EventCentered)
        throw std::invalid_argument("93 operator rejects the old Torus unvalidated-neighbor center policy");
    if(geometry.event_mode==TraceFirstEventMode3D::PythonEndpointEvent)
        throw std::invalid_argument("93 operator rejects the older direction-dependent endpoint event mode");
    ResourceBvpOperators3D result;auto& stats=result.statistics;
    stats.planning=g.restrict_plan.counts;stats.planning_seconds=g.planning_seconds;
    Trace93PolynomialCatalog3D catalog(problem,density,g.trace_anchors,g.spread_anchors,neumann,mean,
        g.restrict_plan.mode!=RestrictResourceMode3D::ReferencePerVisit);
    const auto spread_start=Clock::now();std::vector<Eigen::Triplet<double>> triplets;
    result.known_spread=Eigen::VectorXd::Zero(grid.num_dofs());std::size_t active=0;
    for(int i=0;i<static_cast<int>(support.crossing_ops.size());++i){
        if(!geometry.spread_active[i])continue;++active;const auto& op=support.crossing_ops[i];
        const auto row=catalog.evaluate(3,geometry.spread_centers[i],point(grid,op.correction_node),op.correction_node);
        const double scale=static_cast<double>(op.side_delta)*op.stencil_weight;
        for(std::size_t k=0;k<row.columns.size();++k)triplets.emplace_back(op.rhs_node,row.columns[k],scale*row.values[k]);
        result.known_spread[op.rhs_node]+=scale*row.known;
    }
    result.S.resize(grid.num_dofs(),density.coefficient_count());result.S.setFromTriplets(triplets.begin(),triplets.end());result.S.makeCompressed();
    std::vector<Eigen::Triplet<double>>().swap(triplets);stats.spread_seconds=seconds(spread_start);
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
    for(int q=0;q<static_cast<int>(g.trace_anchors.size());++q)
        for(int side=0;side<2;++side)for(int k=0;k<3;++k)
            samples[q][3*side+k]=catalog.evaluate(2,{RestrictAnchorSource3D::Trace,q},g.stencils[q].sides[side].sample_points[k]);
    result.trace_basis=density.trace_basis;
    result.restrict=assemble_resource_restrict_3d(g.restrict_plan,g.stencils,g.visits,rows,samples,grid.num_dofs(),density.coefficient_count());
    const auto& c=catalog.statistics();stats.spread_centers=c.p3_centers;stats.p2_centers=c.p2_centers;stats.p2_from_p3=c.p2_from_p3;
    stats.endpoint_row_hits=c.row_cache_hits;stats.p2_row_evaluations=c.p2_row_evaluations;stats.retained_crossing_centers=g.restrict_plan.required_crossing_ids.size();
    stats.restrict_seconds=seconds(restrict_start);stats.temporary_centers_after_build=0;
    if(extra){extra->catalog=c;extra->active_spread_operations=active;}
    return result;
}
} // namespace kfbim::app3d
