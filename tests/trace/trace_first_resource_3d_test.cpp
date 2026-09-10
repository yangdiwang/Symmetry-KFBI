#include "src/support/trace/trace_first_resource_3d.hpp"

#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
using namespace kfbim::app3d;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> void throws(F f,const char* message){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,message);}
double difference(const ResourceAffineRow3D& a,const ResourceAffineRow3D& b){
    std::map<int,double> d;
    for(std::size_t i=0;i<a.columns.size();++i)d[a.columns[i]]+=a.values[i];
    for(std::size_t i=0;i<b.columns.size();++i)d[b.columns[i]]-=b.values[i];
    double e=std::abs(a.known-b.known);for(const auto& [id,v]:d){(void)id;e=std::max(e,std::abs(v));}return e;
}
KnownAmbientThird3D harmonic_data(const Eigen::Vector3d& x){
    KnownAmbientThird3D d;d.value=x.x()*x.y()*x.z();
    d.gradient={x.y()*x.z(),x.x()*x.z(),x.x()*x.y()};
    d.hessian(0,1)=d.hessian(1,0)=x.z();d.hessian(0,2)=d.hessian(2,0)=x.y();d.hessian(1,2)=d.hessian(2,1)=x.x();
    for(int k=0;k<3;++k)for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        if(i!=j && i!=k && j!=k)d.third[k](i,j)=1.0;
    return d;
}
void test_catalog(){
    NativeNurbsDensityOptions3D options;options.coefficients_per_direction=4;
    options.reduction_backend=NativeDensityReductionBackend3D::BaseOnly;
    NativeNurbsDensitySpace3D density(make_native_nurbs_surface_3d(GeometryKind3D::Torus),options);
    std::vector<RestrictResourceAnchor3D> trace;
    for(double u:{0.37,0.39}){
        const auto& p=density.surface().patches[0];RestrictResourceAnchor3D a;a.patch_id=0;a.sheet_id=0;
        a.u=p.domain_start_u()+u*(p.domain_end_u()-p.domain_start_u());
        a.v=p.domain_start_v()+0.41*(p.domain_end_v()-p.domain_start_v());
        a.point=density.geometry(0,u,0.41).point;trace.push_back(a);
    }
    const std::vector<RestrictResourceAnchor3D> events;
    const Eigen::Vector3d target=trace[0].point+Eigen::Vector3d(0.013,-0.009,0.011);
    for(auto field:{NativeDensityField3D::ValueTrace,NativeDensityField3D::NormalTrace}){
        TracePolynomialCatalog3D reuse(density,trace,events,field,harmonic_data,0,TraceFirstChart3D::PhysicalGraph,true);
        TracePolynomialCatalog3D reference(density,trace,events,field,harmonic_data,0,TraceFirstChart3D::PhysicalGraph,false);
        const TraceFirstAnchorRef3D q0{RestrictAnchorSource3D::Trace,0};
        reuse.prepare(3,q0);
        const auto a=reuse.evaluate(2,q0,target,123);
        const auto b=reference.evaluate(2,q0,target,123);
        require(difference(a,b)<2e-12,"P3-derived P2 must equal an independently built physical P2");
        require(difference(a,reuse.evaluate(2,q0,target,123))==0.0,"same anchor/grid request must reuse its exact row");
        require(reuse.statistics().row_cache_hits==1,"one repeated request should hit cache once");
        (void)reuse.evaluate(3,q0,target,123);
        require(reuse.statistics().p3_row_evaluations==1,"P2 and P3 cache keys must be distinct");
        (void)reuse.evaluate(2,{RestrictAnchorSource3D::Trace,1},target,123);
        require(reuse.statistics().p2_row_evaluations==2,"different anchors at one grid point must not merge");
        require(reuse.statistics().p2_from_p3==1,"P3 center must supply its lower P2 exactly once");
        throws([&]{(void)reuse.evaluate(2,q0,target+Eigen::Vector3d(0.1,0,0),123);},"mismatched geometry must not enter an existing row-cache key");
        (void)reference.evaluate(2,q0,target,123);
        require(reference.statistics().p2_row_evaluations==2,"reference engine must retain per-visit evaluation");
    }
}
void test_mean_cover(){
    const kfbim::CartesianGrid3D grid({0.0,0.0,0.0},{1.0,1.0,1.0},{8,8,8},kfbim::DofLayout3D::Node);
    const Eigen::Vector3d q(3.5,4,4),n(1,0,0);
    const auto old=build_shared_side_cover_restrict_stencil_3d(grid,q,n,TensorProductCoverKind3D::Q27Cover3);
    const auto py=build_shared_side_cover_restrict_stencil_3d(grid,q,n,TensorProductCoverKind3D::Q27Cover3,true);
    require(old.sides[1].grid_ids[0]==grid.index(4,3,3),"legacy range-midpoint cover must remain unchanged");
    require(py.sides[1].grid_ids[0]==grid.index(3,3,3),"Python cover must use mean of the three nonuniform normal samples");
    for(const auto& side:py.sides)for(int i=0;i<3;++i)
        require(std::abs(side.sampling_weights.row(i).sum()-1)<1e-13,"mean-centered cover must preserve constants");
}
void test_signed_same_side_extension(){
    SharedSideCoverRestrictStencil3D s;s.h=1;s.cubic_pseudoinverse.setZero();
    for(int side=0;side<2;++side){auto& b=s.sides[side];b.desired_inside=side==0;b.grid_ids={side};b.sampling_weights=Eigen::Matrix<double,3,1>::Ones();}
    s.sides[0].value_recovery << 1,0,0;
    std::vector<RestrictSupportVisit3D> visits{{0,true,0,Eigen::Vector3d::Zero(),true},{0,false,1,Eigen::Vector3d::UnitX(),false}};
    RestrictResourcePlan3D plan;plan.visit_to_request={0,-1};RestrictResourceRequest3D request;request.grid_full_id=0;request.signed_extension=true;plan.requests={request};
    ResourceAffineRow3D row;row.columns={0};row.values={2};row.known=3;
    std::vector<ResourceTraceSampleJumpRows3D> samples(1);
    const auto assembled=assemble_resource_restrict_3d(plan,{s},visits,{row},samples,2,1);
    require(assembled.exterior.Rc_value.coeff(0,0)==2 && assembled.exterior.known_value[0]==3,
        "same-label multi-event continuation must survive zero endpoint label delta");
    plan.requests[0].signed_extension=false;
    throws([&]{(void)assemble_resource_restrict_3d(plan,{s},visits,{row},samples,2,1);},
        "an ordinary unsigned jump is not a same-side continuation recipe");
}
void test_owner_validated_guard(){
    RestrictResourceAnchor3D event,trace;event.patch_id=4;event.sheet_id=1;event.point=Eigen::Vector3d::Zero();
    trace.patch_id=4;trace.sheet_id=1;trace.point=Eigen::Vector3d(1.75,0,0);
    const Eigen::Vector3d target(5,0,0);
    require(owner_validated_trace_candidate_3d(trace,event,target,1),"93 radii include exact 1.75h/3.25h boundary");
    trace.patch_id=5;
    require(!owner_validated_trace_candidate_3d(trace,event,target,1),"a same-sheet neighbor may not replace the true event owner");
    trace.patch_id=4;trace.point.x()=1.750001;
    require(!owner_validated_trace_candidate_3d(trace,event,target,1),"event radius is distinct from the target radius");
    trace.point.x()=1.75;
    const Eigen::Vector3d other(-1.500001,0,0);
    require(!owner_validated_trace_candidate_3d(trace,event,target,1,{},&other),"Spread must guard BOTH Cartesian endpoints");
    const Eigen::Vector3d shifted(0.1,-0.4,0.2);event.point+=shifted;trace.point+=shifted;
    require(owner_validated_trace_candidate_3d(trace,event,target+shifted,1),"owner/radius guard is translation invariant");
    throws([&]{(void)owner_validated_trace_candidate_3d(trace,event,target,0);},"owner mapping rejects invalid h");
}
void test_python93_cover_snap(){
    const kfbim::CartesianGrid3D grid({0.0,0.0,0.0},{1.0,1.0,1.0},{8,8,8},kfbim::DofLayout3D::Node);
    const Eigen::Vector3d q(4.5-5e-14,4,4),n(1,0,0);
    const auto torus=build_shared_side_cover_restrict_stencil_3d(grid,q,n,TensorProductCoverKind3D::Q27Cover3,true);
    const auto py93=build_shared_side_cover_restrict_stencil_3d(grid,q,n,TensorProductCoverKind3D::Q27Cover3,true,1e-13);
    require(torus.sides[0].grid_ids[0]==grid.index(2,3,3),"zero snap must retain the older Torus cover");
    require(py93.sides[0].grid_ids[0]==grid.index(3,3,3),"93 cover must snap its near-integer lower sample before floor");
    for(const auto& side:py93.sides)for(int k=0;k<3;++k)
        require(std::abs(side.sampling_weights.row(k).sum()-1)<1e-13,"93 cover snap must preserve constants");
    throws([&]{(void)build_shared_side_cover_restrict_stencil_3d(grid,q,n,TensorProductCoverKind3D::Q27Cover3,true,-1e-13);},
        "cover rejects a negative endpoint snap");
}
}
int main(){try{test_catalog();test_mean_cover();test_signed_same_side_extension();test_owner_validated_guard();test_python93_cover_snap();std::cout<<"trace-first catalog/cover/signed recipe/owner guard tests passed\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
