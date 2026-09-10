#include "src/support/geometry/trace_first_case_3d.hpp"
#include "src/support/density/python_torus_density_layout_3d.hpp"
#include "src/support/trace/trace_first_resource_3d.hpp"
#include "src/geometry/nurbs_cartesian_domain_3d.hpp"
#include "src/geometry/grid_pair_3d.hpp"
#include "src/bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include "src/gmres/gmres.hpp"

#include <Eigen/Cholesky>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using namespace kfbim;
using namespace kfbim::app3d;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
struct Options {
    int N=32, max_iterations=1000;
    std::string bvp="both", transform="rotate", chart="extended", policy="trace", events="all";
    bool grid_lines=true, compare_cache=false, dump=false;
    std::filesystem::path output;
};
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--compare-cache") {o.compare_cache=true;continue;}
        if(key=="--dump-matrices") {o.dump=true;continue;}
        if(i+1>=argc) throw std::invalid_argument("missing option value: "+key);
        const std::string v=argv[++i];
        if(key=="--N") o.N=std::stoi(v);
        else if(key=="--max-iterations") o.max_iterations=std::stoi(v);
        else if(key=="--bvp") o.bvp=v;
        else if(key=="--transform") o.transform=v;
        else if(key=="--chart") o.chart=v;
        else if(key=="--policy") o.policy=v;
        else if(key=="--event-mode") o.events=v;
        else if(key=="--grid-lines") {if(v!="on"&&v!="off") throw std::invalid_argument("grid-lines on/off");o.grid_lines=v=="on";}
        else if(key=="--output") o.output=v;
        else throw std::invalid_argument("unknown option: "+key);
    }
    if(o.output.empty()) throw std::invalid_argument("--output must name a new result directory");
    if(o.bvp!="both"&&o.bvp!="neumann"&&o.bvp!="dirichlet") throw std::invalid_argument("invalid BVP");
    if(o.chart!="extended"&&o.chart!="physical") throw std::invalid_argument("invalid chart");
    if(o.events!="all"&&o.events!="python") throw std::invalid_argument("invalid event mode");
    if(o.max_iterations<1||o.max_iterations>5000) throw std::invalid_argument("invalid GMRES limit");
    return o;
}
TraceFirstCenterPolicy3D policy(const std::string& s) {
    if(s=="trace") return TraceFirstCenterPolicy3D::TracePolynomialFirst;
    if(s=="event") return TraceFirstCenterPolicy3D::EventCentered;
    if(s=="trace-spread") return TraceFirstCenterPolicy3D::TraceSpreadEventRestrict;
    if(s=="event-spread") return TraceFirstCenterPolicy3D::EventSpreadTraceRestrict;
    throw std::invalid_argument("policy must be trace, event, trace-spread or event-spread");
}
class ApplyOperator : public IKFBIOperator {
public:
    int size=0;
    std::function<void(const Eigen::VectorXd&,Eigen::VectorXd&)> fn;
    int problem_size() const override {return size;}
    void apply(const Eigen::VectorXd& x,Eigen::VectorXd& y) const override {fn(x,y);}
};

bool solve_case(const Options& options,const TraceFirstCase3D& problem,
    const CartesianGrid3D& grid,const GridPair3D& pair,
    const LaplaceCorrectionSupport3D& support,bool neumann,double shared_seconds)
{
    const auto start=Clock::now();
    const std::string name=neumann?"neumann":"dirichlet";
    const auto output=options.output/name;
    std::filesystem::create_directories(output);
    const auto spans=problem.density_spans(options.N);
    NativeNurbsDensityOptions3D density_options;
    density_options.coefficients_per_direction=spans[0]+3;
    density_options.field=neumann?NativeDensityField3D::ValueTrace:NativeDensityField3D::NormalTrace;
    density_options.reduction_backend=NativeDensityReductionBackend3D::TopologyBase;
    density_options.trace_gauss_order=1; // Carrier quadrature is not the trace discretization.
    NativeNurbsDensitySpace3D density(problem.surface,density_options);
    auto layout=build_python_torus_density_layout_3d(density,spans[0],spans[1],neumann);
    const double density_seconds=seconds(start);
    std::cout<<name<<" density full="<<layout.reference_raw_dofs<<" carrier_c0="<<layout.Z.rows()
             <<" reduced="<<layout.Z.cols()<<" traces="<<layout.traces.size()<<std::endl;
    ResourceKnownAmbientCallback3D known=[&](const Eigen::Vector3d& x) {
        const auto d=problem.evaluate(x); KnownAmbientThird3D k;
        k.value=d.value;k.gradient=d.gradient;k.hessian=d.hessian;k.third=d.third;return k;
    };
    std::vector<RestrictResourceAnchor3D> anchors;
    for(const auto& q:layout.traces) {
        const auto& patch=problem.surface.patches.at(q.patch);
        RestrictResourceAnchor3D a;a.point=q.point;a.patch_id=q.patch;
        a.u=patch.domain_start_u()+q.u*(patch.domain_end_u()-patch.domain_start_u());
        a.v=patch.domain_start_v()+q.v*(patch.domain_end_v()-patch.domain_start_v());
        anchors.push_back(a);
    }
    const auto cover=neumann?TensorProductCoverKind3D::Q27Cover3:TensorProductCoverKind3D::Q64Cover4;
    const auto events=options.events=="all"?TraceFirstEventMode3D::LocalAllEvent:TraceFirstEventMode3D::PythonEndpointEvent;
    const auto chart=options.chart=="extended"?TraceFirstChart3D::ExtendedTube:TraceFirstChart3D::PhysicalGraph;
    TraceFirstBuildStatistics3D ts;
    ResourceBvpOperators3D op;
    double ab_error=0,ab_seconds=0,plan_seconds=0;
    std::size_t current_requests=0,nearest_requests=0,far_requests=0,spread_fallbacks=0;
    bool paths_certified=false;
    {
        auto plan=build_trace_first_geometry_plan_3d(grid,pair,problem.surface,support,anchors,cover,
            RestrictResourceMode3D::ReuseBySheetNode,policy(options.policy),events);
        current_requests=plan.current_trace_requests;nearest_requests=plan.nearest_trace_requests;
        far_requests=plan.far_trace_requests;spread_fallbacks=plan.spread_event_fallbacks;
        paths_certified=plan.support_paths_certified;
        plan_seconds=plan.resource.planning_seconds;
        op=build_trace_first_bvp_operators_3d(plan,grid,pair,support,density,density_options.field,known,0.0,chart,&ts);
        if(options.compare_cache) {
            const auto ab_start=Clock::now();
            const auto reference_plan=build_trace_first_geometry_plan_3d(grid,pair,problem.surface,support,anchors,cover,
                RestrictResourceMode3D::ReferencePerVisit,policy(options.policy),events);
            const auto reference=build_trace_first_bvp_operators_3d(reference_plan,grid,pair,support,density,
                density_options.field,known,0.0,chart);
            ab_error=resource_operator_max_difference_3d(op,reference);ab_seconds=seconds(ab_start);
            if(ab_error>5e-10) throw std::runtime_error("cached/reference operator mismatch");
        }
    } // Geometry plans, centres, and row caches are not retained through GMRES.
    if(options.dump) dump_resource_operators_3d(op,(output/"matrices").string());
    const auto projection_start=Clock::now();
    const auto& B=op.trace_basis;
    auto weighted_B=B;
    for(int row=0;row<weighted_B.outerSize();++row)
        for(ResourceRestrictSparseMatrix3D::InnerIterator it(weighted_B,row);it;++it)
            it.valueRef()*=layout.weights[it.row()];
    Eigen::SparseMatrix<double> mass=B.transpose()*weighted_B;
    Eigen::MatrixXd M=layout.Z.transpose()*(mass*layout.Z);
    M=(0.5*(M+M.transpose())).eval();
    Eigen::LLT<Eigen::MatrixXd> chol(M);
    if(chol.info()!=Eigen::Success) throw std::runtime_error("trace Gram matrix not positive definite: refine trace sampling");
    const Eigen::VectorXd pivots=chol.matrixL().toDenseMatrix().diagonal();
    const double pivot_ratio=pivots.minCoeff()/pivots.maxCoeff();
    if(!std::isfinite(pivot_ratio)||pivot_ratio<1e-10)
        throw std::runtime_error("trace space is insufficiently observable");
    const auto project=[&](const Eigen::VectorXd& t)->Eigen::VectorXd {
        const Eigen::VectorXd r=layout.Z.transpose()*(B.transpose()*(layout.weights.array()*t.array()).matrix());
        return chol.solve(r);
    };
    const double projection_seconds=seconds(projection_start);
    const auto& Rg=neumann?op.restrict.Rg_value:op.restrict.Rg_normal;
    const auto& Rc=neumann?op.restrict.exterior.Rc_value:op.restrict.exterior.Rc_normal;
    const auto& br=neumann?op.restrict.exterior.known_value:op.restrict.exterior.known_normal;
    LaplaceFftBulkSolverZfft3D poisson(grid,ZfftBcType::Dirichlet,0.0,2);
    double poisson_seconds=0,matvec_seconds=0;
    int poisson_calls=0;
    const auto field=[&](const Eigen::VectorXd& c,bool affine)->Eigen::VectorXd {
        Eigen::VectorXd rhs=op.S*c,U;
        if(affine) rhs+=op.known_spread;
        const auto t=Clock::now();poisson.solve(-rhs,U);poisson_seconds+=seconds(t);++poisson_calls;
        return U;
    };
    double constant_bulk_error=0,constant_trace_error=0;
    if(neumann) {
        const Eigen::VectorXd one=Eigen::VectorXd::Ones(layout.Z.rows());
        const Eigen::VectorXd probe=field(one,false);
        constant_trace_error=(Rg*probe+Rc*one).cwiseAbs().maxCoeff();
        for(int i=0;i<grid.num_dofs();++i)
            constant_bulk_error=std::max(constant_bulk_error,std::abs(probe[i]-(pair.domain_label(i)>0?1.0:0.0)));
        if(std::max(constant_bulk_error,constant_trace_error)>1e-8)
            throw std::runtime_error("constant-jump identity failed before GMRES");
    }
    const auto affine_start=Clock::now();
    const Eigen::VectorXd U0=field(layout.particular,true);
    const Eigen::VectorXd rhs=-project((Rg*U0+Rc*layout.particular+br).eval());
    ApplyOperator A;A.size=int(layout.Z.cols());
    A.fn=[&](const Eigen::VectorXd& y,Eigen::VectorXd& result) {
        const auto t=Clock::now();const Eigen::VectorXd c=layout.Z*y;
        const Eigen::VectorXd U=field(c,false);
        result=project((Rg*U+Rc*c).eval());matvec_seconds+=seconds(t);
    };
    GMRES gmres(options.max_iterations,2e-10,std::min(100,A.size));
    Eigen::VectorXd y=Eigen::VectorXd::Zero(A.size);
    const double affine_rhs_seconds=seconds(affine_start);
    const auto gmres_start=Clock::now();
    const int iterations=gmres.solve(A,rhs,y);
    const double gmres_seconds=seconds(gmres_start);
    const Eigen::VectorXd c=layout.particular+layout.Z*y;
    const Eigen::VectorXd U=field(c,true);
    const Eigen::VectorXd tv=op.restrict.Rg_value*U+op.restrict.exterior.Rc_value*c+op.restrict.exterior.known_value;
    const Eigen::VectorXd tn=op.restrict.Rg_normal*U+op.restrict.exterior.Rc_normal*c+op.restrict.exterior.known_normal;
    const Eigen::VectorXd trace=neumann?tv:tn;
    const Eigen::VectorXd residual=project(trace);
    const double projected_relative=residual.norm()/std::max(rhs.norm(),1e-300);
    const Eigen::VectorXd representable=B*(layout.Z*residual);
    const double full_norm=std::sqrt((layout.weights.array()*trace.array().square()).sum());
    const double leakage=std::sqrt((layout.weights.array()*(trace-representable).array().square()).sum())/std::max(full_norm,1e-300);
    double interior_linf=0,interior_sq=0,density_linf=0,density_sq=0;
    int inside_count=0,max_node=-1;
    Eigen::Vector3d max_point=Eigen::Vector3d::Zero();
    for(int i=0;i<grid.num_dofs();++i) if(pair.domain_label(i)>0) {
        const auto p=grid.coord(i);const Eigen::Vector3d x(p[0],p[1],p[2]);
        const double err=std::abs(U[i]-problem.evaluate(x).value);
        if(err>interior_linf) {interior_linf=err;max_node=i;max_point=x;}
        interior_sq+=err*err;++inside_count;
    }
    const Eigen::VectorXd density_values=B*c;
    for(int i=0;i<density_values.size();++i) {
        const auto& q=layout.traces[i];const auto ex=problem.evaluate(q.point);
        const double target=neumann?ex.value:ex.gradient.dot(q.normal);
        const double e=density_values[i]-target;
        density_linf=std::max(density_linf,std::abs(e));density_sq+=layout.weights[i]*e*e;
    }
    if(!U.allFinite()||!std::isfinite(interior_linf)||inside_count==0)
        throw std::runtime_error("invalid computed field");
    const double total_seconds=seconds(start);
    std::ofstream history(output/"gmres_history.json");history<<std::setprecision(17)<<"[";
    for(std::size_t i=0;i<gmres.residuals().size();++i) {if(i)history<<",";history<<gmres.residuals()[i];}history<<"]\n";
    std::ofstream j(output/"result.json");j<<std::setprecision(17)<<"{\n";
    const auto s=[&](const char* k,const std::string& v){j<<"  \""<<k<<"\": \""<<v<<"\",\n";};
    const auto n=[&](const char* k,double v){j<<"  \""<<k<<"\": "<<v<<",\n";};
    s("case",problem.id);s("transform",options.transform);s("bvp",name);s("chart",options.chart);
    s("center_policy",options.policy);s("event_mode",options.events);s("density","python_anisotropic_exact_knot_embedding");
    n("N",options.N);n("h",2.0/options.N);n("spans_u",spans[0]);n("spans_v",spans[1]);
    n("reference_raw_dofs",layout.reference_raw_dofs);n("carrier_raw_dofs",density.raw_coefficient_count());
    n("carrier_c0_dofs",density.c0_coefficient_count());n("reduced_dofs",A.size);n("trace_points",layout.traces.size());
    n("c0_embedding_error",layout.c0_embedding_error);n("mean_nullspace_error",layout.mean_nullspace_error);
    n("gram_cholesky_pivot_ratio",pivot_ratio);n("gmres_iterations",iterations);n("gmres_tolerance",2e-10);
    n("projected_relative_residual",projected_relative);n("full_exterior_value_linf",tv.cwiseAbs().maxCoeff());
    n("full_exterior_normal_linf",tn.cwiseAbs().maxCoeff());n("active_trace_weighted_l2",full_norm);n("projection_leakage",leakage);
    n("interior_linf",interior_linf);n("interior_l2",std::sqrt(interior_sq/inside_count));
    n("density_linf",density_linf);n("density_l2",std::sqrt(density_sq/layout.weights.sum()));
    n("max_error_grid_id",max_node);j<<"  \"max_error_point\": ["<<max_point.x()<<","<<max_point.y()<<","<<max_point.z()<<"],\n";
    n("shared_geometry_seconds",shared_seconds);n("density_seconds",density_seconds);n("planning_seconds",plan_seconds);
    n("spread_seconds",op.statistics.spread_seconds);n("restrict_seconds",op.statistics.restrict_seconds);
    n("projection_setup_seconds",projection_seconds);n("gmres_seconds",gmres_seconds);n("poisson_seconds",poisson_seconds);
    n("affine_rhs_seconds",affine_rhs_seconds);
    n("poisson_calls",poisson_calls);n("matvec_seconds",matvec_seconds);n("bvp_total_seconds",total_seconds);
    n("cache_comparison_seconds",ab_seconds);n("cached_reference_max_error",ab_error);
    n("current_trace_requests",current_requests);n("nearest_trace_requests",nearest_requests);n("far_trace_requests",far_requests);
    n("spread_event_fallbacks",spread_fallbacks);n("p3_centers",ts.catalog.p3_centers);n("p2_centers",ts.catalog.p2_centers);
    n("p2_from_p3",ts.catalog.p2_from_p3);n("row_cache_hits",ts.catalog.row_cache_hits);
    n("p2_row_evaluations",ts.catalog.p2_row_evaluations);n("p3_row_evaluations",ts.catalog.p3_row_evaluations);
    n("maximum_projection_residual",ts.catalog.maximum_projection_residual);
    n("projection_not_converged",ts.catalog.projection_not_converged);
    n("constant_jump_bulk_error",constant_bulk_error);n("constant_jump_trace_error",constant_trace_error);
    j<<"  \"constant_jump_checked\": "<<(neumann?"true":"false")<<",\n";
    j<<"  \"cache_comparison_checked\": "<<(options.compare_cache?"true":"false")<<",\n"
     <<"  \"support_paths_certified\": "<<(paths_certified?"true":"false")<<",\n"
     <<"  \"gmres_converged\": "<<(gmres.converged()&&projected_relative<3e-10?"true":"false")<<"\n}\n";
    if(!j||!history) throw std::runtime_error("result write failed");
    std::cout<<std::setprecision(10)<<name<<" N="<<options.N<<" error="<<interior_linf<<" iter="<<iterations
             <<" residual="<<projected_relative<<" seconds="<<total_seconds<<std::endl;
    return gmres.converged()&&projected_relative<3e-10;
}
} // namespace

int main(int argc,char** argv) {
    try {
        const auto options=parse(argc,argv);
        if(std::filesystem::exists(options.output)) throw std::invalid_argument("refusing to overwrite existing results");
        const auto problem=make_trace_first_python_torus_case_3d(options.transform);
        (void)problem.density_spans(options.N);(void)policy(options.policy);
        std::filesystem::create_directories(options.output);
        const auto t=Clock::now();const double h=2.0/options.N;
        CartesianGrid3D grid({-1.,-1.,-1.},{h,h,h},{options.N,options.N,options.N},DofLayout3D::Node);
        geometry3d::NurbsPatchTriangulatorOptions3D mesh_options;
        // Smooth seams are already unbuffered by the triangulator. Retain
        // its positive feature-buffer contract for the compatibility mesh.
        auto mesh=geometry3d::triangulate_nurbs_surface_patches_3d(problem.surface.patches,h,mesh_options);
        geometry3d::NurbsCartesianDomainOptions3D domain_options;
        domain_options.strategy=geometry3d::NurbsCartesianPreprocessStrategy3D::OptimizedIntersection;
        domain_options.use_whole_grid_lines=options.grid_lines;
        std::cout<<"building native grid crossing catalog N="<<options.N<<" whole_lines="<<options.grid_lines<<std::endl;
        auto domain=std::make_shared<geometry3d::NurbsCartesianDomain3D>(grid,problem.surface.geometry_model(),domain_options);
        std::vector<int> patch_map;for(const auto& tri:mesh.triangles) patch_map.push_back(tri.patch_index);
        GridPair3D pair(grid,mesh.interface,mesh.geometry_interface,domain,patch_map);
        int mismatches=0;
        for(int i=0;i<grid.num_dofs();++i) {
            const auto p=grid.coord(i);const Eigen::Vector3d x(p[0],p[1],p[2]);
            if((pair.domain_label(i)>0)!=problem.surface.exact_inside(x)) ++mismatches;
        }
        if(mismatches) throw std::runtime_error("native grid labels disagree with exact case oracle");
        const auto& d=domain->diagnostics();
        const auto write_geometry=[&](double shared_seconds,bool correction_ready) {
        std::ofstream g(options.output/"geometry.json");g<<std::setprecision(17)
            <<"{\n\"N\":"<<options.N<<",\n\"label_mismatches\":"<<mismatches
            <<",\n\"correction_support_ready\":"<<(correction_ready?"true":"false")
            <<",\n\"events\":"<<d.canonical_grid_edge_event_count<<",\n\"uncertified_edges\":"<<d.uncertified_grid_edge_count
            <<",\n\"intersection_seconds\":"<<d.edge_intersection_seconds
            <<",\n\"intersector_build_seconds\":"<<d.intersector_build_seconds
            <<",\n\"candidate_seconds\":"<<d.candidate_enumeration_seconds
            <<",\n\"candidate_grid_edge_count\":"<<d.candidate_grid_edge_count
            <<",\n\"whole_line_query_count\":"<<d.whole_line_query_count
            <<",\n\"whole_line_accepted_count\":"<<d.whole_line_accepted_count
            <<",\n\"whole_line_fallback_count\":"<<d.whole_line_fallback_count
            <<",\n\"whole_line_reused_edge_count\":"<<d.whole_line_reused_edge_count
            <<",\n\"direct_edge_query_count\":"<<d.direct_edge_query_count
            <<",\n\"whole_line_intersection_seconds\":"<<d.whole_line_intersection_seconds
            <<",\n\"certified_transverse_event_count\":"<<d.certified_transverse_event_count
            <<",\n\"multi_crossing_edge_count\":"<<d.multi_crossing_edge_count
            <<",\n\"even_parity_interface_edge_count\":"<<d.even_parity_interface_edge_count
            <<",\n\"ambiguous_parity_edge_count\":"<<d.ambiguous_parity_edge_count
            <<",\n\"targeted_retry_count\":"<<d.targeted_retry_count
            <<",\n\"targeted_retry_resolved_count\":"<<d.targeted_retry_resolved_count
            <<",\n\"targeted_retry_unsafe_count\":"<<d.targeted_retry_unsafe_count
            <<",\n\"endpoint_parity_fallback_count\":"<<d.endpoint_parity_fallback_count
            <<",\n\"edge_materialization_seconds\":"<<d.edge_materialization_seconds
            <<",\n\"flood_labeling_seconds\":"<<d.flood_labeling_seconds
            <<",\n\"representative_classification_seconds\":"<<d.representative_classification_seconds
            <<",\n\"invariant_verification_seconds\":"<<d.invariant_verification_seconds
            <<",\n\"candidate_elements\":"<<d.intersections.candidate_elements
            <<",\n\"subdivision_boxes\":"<<d.intersections.subdivision_boxes
            <<",\n\"newton_attempts\":"<<d.intersections.newton_attempts
            <<",\n\"newton_iterations\":"<<d.intersections.newton_iterations
            <<",\n\"terminal_certificate_boxes\":"<<d.intersections.terminal_certificate_boxes
            <<",\n\"maximum_subdivision_depth_reached\":"<<d.intersections.maximum_subdivision_depth_reached
            <<",\n\"maximum_terminal_certificate_depth_reached\":"<<d.intersections.maximum_terminal_certificate_depth_reached
            <<",\n\"unresolved_candidates\":"<<d.intersections.unresolved_candidates
            <<",\n\"retry_candidate_elements\":"<<d.targeted_retry_intersections.candidate_elements
            <<",\n\"retry_subdivision_boxes\":"<<d.targeted_retry_intersections.subdivision_boxes
            <<",\n\"retry_newton_iterations\":"<<d.targeted_retry_intersections.newton_iterations
            <<",\n\"retry_terminal_certificate_boxes\":"<<d.targeted_retry_intersections.terminal_certificate_boxes
            <<",\n\"retry_unresolved_candidates\":"<<d.targeted_retry_intersections.unresolved_candidates
            <<",\n\"domain_total_seconds\":"<<d.total_construction_seconds
            <<",\n\"shared_geometry_seconds\":"<<shared_seconds<<"\n}\n";
        g.flush();g.close();
        if(!g) throw std::runtime_error("geometry diagnostics write failed");
        };
        const auto write_uncertified_edges=[&] {
            std::ofstream out(options.output/"uncertified_edges.txt");
            out<<std::setprecision(17)<<std::boolalpha;
            out<<"N="<<options.N<<" grid_lines="<<options.grid_lines
               <<" uncertified_grid_edge_count="<<d.uncertified_grid_edge_count<<'\n';
            const auto cells=grid.num_cells();const auto dims=grid.dof_dims();
            const auto& sheets=domain->patch_g1_components();
            const auto sheet_of=[&](int patch) {
                return patch>=0 && static_cast<std::size_t>(patch)<sheets.size()?sheets[patch]:-1;
            };
            const auto certification_name=[](geometry3d::GridEdgeEventCertification3D c) {
                using C=geometry3d::GridEdgeEventCertification3D;
                switch(c) {
                case C::CertifiedTransverse:return "CertifiedTransverse";
                case C::IncompleteRootSet:return "IncompleteRootSet";
                case C::UnknownParity:return "UnknownParity";
                case C::NearTangentEdge:return "NearTangentEdge";
                case C::FeatureContact:return "FeatureContact";
                case C::UnreliableTransversality:return "UnreliableTransversality";
                }
                return "UnknownCertification";
            };
            std::size_t dumped=0;
            for(int k=0;k<dims[2];++k) for(int j=0;j<dims[1];++j) for(int i=0;i<dims[0];++i)
                for(int axis=0;axis<3;++axis) {
                    std::array<int,3> ijk{{i,j,k}};
                    if(ijk[axis]>=cells[axis]) continue;
                    const int a=grid.index(i,j,k);++ijk[axis];
                    const int b=grid.index(ijk[0],ijk[1],ijk[2]);
                    const auto info=domain->edge_classification_between(a,b);
                    if(!info.queried || info.physical_events_certified) continue;
                    ++dumped;const auto pa=grid.coord(a),pb=grid.coord(b);
                    out<<"\nEDGE axis="<<axis<<" ijk="<<i<<','<<j<<','<<k<<" nodes="<<a<<','<<b
                       <<" start="<<pa[0]<<','<<pa[1]<<','<<pa[2]
                       <<" end="<<pb[0]<<','<<pb[1]<<','<<pb[2]
                       <<" labels="<<domain->label(a)<<','<<domain->label(b)<<'\n';
                    out<<"CLASS queried="<<info.queried<<" root_count_known="<<info.root_count_known
                       <<" parity_known="<<info.parity_known_from_roots
                       <<" near_tangent="<<info.has_near_tangent_candidate
                       <<" targeted_retry="<<info.used_targeted_retry
                       <<" confirmed_count="<<info.confirmed_crossing_count
                       <<" confirmed_transverse="<<info.confirmed_transverse_count
                       <<" ambiguous_clusters="<<info.ambiguous_cluster_count
                       <<" physical_events_certified="<<info.physical_events_certified
                       <<" correction_safe="<<info.correction_safe<<'\n';
                    const auto roots=domain->crossings_between(a,b);
                    for(std::size_t r=0;r<roots.size();++r) {
                        const auto& root=roots[r];
                        out<<"ROOT ordinal="<<r<<" t="<<root.edge_parameter<<" component="<<root.component
                           <<" patch="<<root.patch_index<<" sheet="<<sheet_of(root.patch_index)
                           <<" uv="<<root.u<<','<<root.v<<" point="<<root.point.transpose()
                           <<" normal="<<root.normal.transpose()<<" residual="<<root.residual
                           <<" trans="<<root.transversality<<" tau="<<root.reliable_transversality_tolerance
                           <<" feature="<<root.feature_edge_contact<<" owners="<<root.owners.size()<<'\n';
                        for(const auto& owner:root.owners)
                            out<<" OWNER patch="<<owner.patch_index<<" sheet="<<sheet_of(owner.patch_index)
                               <<" uv="<<owner.u<<','<<owner.v<<" point="<<owner.point.transpose()
                               <<" normal="<<owner.normal.transpose()<<" residual="<<owner.residual
                               <<" trans="<<owner.transversality<<" tau="<<owner.reliable_transversality_tolerance
                               <<" feature="<<owner.feature_edge_contact<<'\n';
                    }
                    for(const auto& view:domain->grid_edge_events_between(a,b)) {
                        const auto& event=view.canonical_event();
                        out<<"EVENT id="<<event.id.first_node<<','<<event.id.second_node<<','<<event.id.ordinal
                           <<" t="<<event.canonical_parameter<<" sign="<<event.canonical_sign
                           <<" component="<<event.component<<" certification="<<certification_name(event.certification)
                           <<" point="<<event.point.transpose()<<" normal="<<event.normal.transpose()
                           <<" residual="<<event.residual<<" trans="<<event.transversality
                           <<" feature="<<event.feature_edge_contact<<" owners="<<event.owners.size()<<'\n';
                        for(const auto& owner:event.owners)
                            out<<" EVENT_OWNER patch="<<owner.patch_index<<" sheet="<<sheet_of(owner.patch_index)
                               <<" uv="<<owner.u<<','<<owner.v<<" trans="<<owner.transversality
                               <<" tau="<<owner.reliable_transversality_tolerance<<'\n';
                    }
                }
            out<<"\nDUMPED_EDGES="<<dumped<<'\n';out.flush();out.close();
            if(!out) throw std::runtime_error("uncertified-edge diagnostics write failed");
        };
        const auto support=[&] {
            try {return build_laplace_correction_support_3d(pair,"trace-first study");}
            catch(...) {
                const auto original=std::current_exception();
                try {write_geometry(seconds(t),false);write_uncertified_edges();}
                catch(const std::exception& e) {std::cerr<<"diagnostic write failed: "<<e.what()<<std::endl;}
                std::rethrow_exception(original);
            }
        }();
        // Same successful timing boundary as before: after support setup,
        // before diagnostic file I/O or any Neumann/Dirichlet solve.
        const double shared_seconds=seconds(t);
        write_geometry(shared_seconds,true);
        bool passed=true;
        if(options.bvp!="dirichlet") passed=solve_case(options,problem,grid,pair,support,true,shared_seconds)&&passed;
        if(options.bvp!="neumann") passed=solve_case(options,problem,grid,pair,support,false,shared_seconds)&&passed;
        return passed?0:2;
    } catch(const std::exception& e) {std::cerr<<"trace-first study: "<<e.what()<<std::endl;return 1;}
}
