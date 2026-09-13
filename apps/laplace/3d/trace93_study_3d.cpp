#include "src/support/geometry/trace93_case_3d.hpp"
#include "src/support/density/trace93_density_layout_3d.hpp"
#include "src/support/trace/trace93_resource_3d.hpp"
#include "src/geometry/nurbs_cartesian_domain_3d.hpp"
#include "src/geometry/grid_pair_3d.hpp"
#include "src/bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include "src/support/solver/affine_kfbi_solve_3d.hpp"
#ifdef KFBIM_HAS_SHARED_FIELD_3D
#include "src/support/correction/shared_field_backend_3d.hpp"
#endif

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace {
using namespace kfbim;
using namespace kfbim::app3d;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
void write_vector(const std::filesystem::path& path,const Eigen::VectorXd& v) {
    std::ofstream out(path);out<<v.size()<<" 1\n"<<std::setprecision(17);
    for(int i=0;i<v.size();++i) out<<v[i]<<'\n';
    if(!out) throw std::runtime_error("vector output failed: "+path.string());
}
void write_matrix(const std::filesystem::path& path,const Eigen::MatrixXd& m) {
    std::ofstream out(path);out<<m.rows()<<' '<<m.cols()<<'\n'<<std::setprecision(17);
    for(int i=0;i<m.rows();++i) {for(int j=0;j<m.cols();++j) out<<m(i,j)<<' ';out<<'\n';}
    if(!out) throw std::runtime_error("matrix output failed: "+path.string());
}
Eigen::VectorXd read_vector(const std::filesystem::path& path,int expected) {
    std::ifstream in(path);int rows=0,cols=0;in>>rows>>cols;
    if(!in||rows!=expected||cols!=1) throw std::runtime_error("replay vector shape mismatch: "+path.string());
    Eigen::VectorXd v(rows);for(int i=0;i<rows;++i) in>>v[i];
    if(!in||!v.allFinite()) throw std::runtime_error("replay vector read failed: "+path.string());
    return v;
}
double relative_difference(const Eigen::VectorXd& a,const Eigen::VectorXd& b) {
    return (a-b).lpNorm<Eigen::Infinity>()/std::max({1.,a.lpNorm<Eigen::Infinity>(),b.lpNorm<Eigen::Infinity>()});
}
struct Options {
    int N=32, max_iterations=0;
    std::string geometry="cylinder", bvp="both", transform="rotate", policy="trace", events="python93";
    bool grid_lines=true, compare_cache=false, dump=false, geometry_only=false;
    bool spectrum=false,dump_grid=false,verify_replay=false,explicit_policy=false,explicit_events=false,explicit_field=false;
    std::string backend="direct_cauchy",target,field_restrict="staged";
    double field_ratio=4.,field_width=4.,field_ridge=1e-12,value_weight=1.,normal_weight=1.,pde_weight=1.;
    std::filesystem::path output;
};
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--geometry-only") {o.geometry_only=true;continue;}
        if(key=="--compare-cache") {o.compare_cache=true;continue;}
        if(key=="--dump-matrices") {o.dump=true;continue;}
        if(key=="--operator-spectrum") {o.spectrum=true;continue;}
        if(key=="--dump-grid") {o.dump_grid=true;continue;}
        if(key=="--verify-replay") {o.verify_replay=true;continue;}
        if(i+1>=argc) throw std::invalid_argument("missing option value: "+key);
        const std::string v=argv[++i];
        if(key=="--N") o.N=std::stoi(v);
        else if(key=="--max-iterations") o.max_iterations=std::stoi(v);
        else if(key=="--bvp") o.bvp=v;
        else if(key=="--transform") o.transform=v;
        else if(key=="--geometry") o.geometry=v;
        else if(key=="--policy") {o.policy=v;o.explicit_policy=true;}
        else if(key=="--event-mode") {o.events=v;o.explicit_events=true;}
        else if(key=="--correction-backend") o.backend=v;
        else if(key=="--exterior-target") o.target=v;
        else if(key=="--shared-field-restrict") {o.field_restrict=v;o.explicit_field=true;}
        else if(key=="--shared-field-ratio") {o.field_ratio=std::stod(v);o.explicit_field=true;}
        else if(key=="--shared-field-width") {o.field_width=std::stod(v);o.explicit_field=true;}
        else if(key=="--shared-field-ridge") {o.field_ridge=std::stod(v);o.explicit_field=true;}
        else if(key=="--shared-field-value-weight") {o.value_weight=std::stod(v);o.explicit_field=true;}
        else if(key=="--shared-field-normal-weight") {o.normal_weight=std::stod(v);o.explicit_field=true;}
        else if(key=="--shared-field-pde-weight") {o.pde_weight=std::stod(v);o.explicit_field=true;}
        else if(key=="--grid-lines") {if(v!="on"&&v!="off") throw std::invalid_argument("grid-lines on/off");o.grid_lines=v=="on";}
        else if(key=="--output") o.output=v;
        else throw std::invalid_argument("unknown option: "+key);
    }
    if(o.output.empty()) throw std::invalid_argument("--output must name a new result directory");
    if(o.bvp!="both"&&o.bvp!="neumann"&&o.bvp!="dirichlet") throw std::invalid_argument("invalid BVP");
    if(o.geometry!="cylinder"&&o.geometry!="l"&&o.geometry!="u") throw std::invalid_argument("geometry must be cylinder/l/u");
    if(o.N!=32&&o.N!=64&&o.N!=128) throw std::invalid_argument("N must be 32/64/128");
    if(o.events!="all"&&o.events!="python93") throw std::invalid_argument("invalid event mode");
    if(o.max_iterations<0||o.max_iterations>5000) throw std::invalid_argument("invalid GMRES limit");
    if(o.backend!="direct_cauchy"&&o.backend!="shared_field") throw std::invalid_argument("correction-backend must be direct_cauchy/shared_field");
    const bool shared=o.backend=="shared_field";
    if(o.target.empty()) o.target=shared?"input_jump_half":"raw";
    if(o.target!="raw"&&o.target!="input_jump_half") throw std::invalid_argument("exterior-target must be raw/input_jump_half");
    if(o.field_restrict!="staged"&&o.field_restrict!="direct") throw std::invalid_argument("shared-field-restrict must be staged/direct");
    for(double value:{o.field_ratio,o.field_width,o.field_ridge,o.value_weight,o.normal_weight,o.pde_weight})
        if(!std::isfinite(value)||value<=0.) throw std::invalid_argument("shared field ratios and weights must be positive and finite");
    if(shared) {
        if(o.geometry!="u") throw std::invalid_argument("shared_field currently supports only --geometry u");
        if(o.explicit_policy||o.explicit_events||o.compare_cache) throw std::invalid_argument("shared_field conflicts with legacy policy/event/cache options");
#ifndef KFBIM_HAS_SHARED_FIELD_3D
        throw std::invalid_argument("shared_field was not compiled; enable KFBIM_BUILD_EXPERIMENTAL_3D");
#endif
    } else if(o.target!="raw"||o.explicit_field) throw std::invalid_argument("direct_cauchy supports raw target and no shared-field options");
    if(o.spectrum&&o.N!=32) throw std::invalid_argument("full operator spectrum is limited to N32");
    return o;
}
TraceFirstCenterPolicy3D policy(const std::string& s) {
    if(s=="trace") return TraceFirstCenterPolicy3D::OwnerValidatedTraceFirst;
    if(s=="event") return TraceFirstCenterPolicy3D::EventCentered;
    throw std::invalid_argument("policy must be trace or event");
}
bool solve_case(const Options& options,const Trace93Case3D& problem,
    const CartesianGrid3D& grid,const GridPair3D& pair,
    const LaplaceCorrectionSupport3D* support,bool neumann,double shared_seconds)
{
    const auto start=Clock::now();
    const std::string name=neumann?"neumann":"dirichlet";
    const auto output=options.output/name;
    std::filesystem::create_directories(output);
    std::cout<<name<<" building Python93 analysis density and affine constraints"<<std::endl;
    auto layout=build_trace93_density_layout_3d(problem,2.0/options.N,neumann);
    if(layout.Z.rows()!=layout.reference_raw_dofs || layout.Z.cols()<1 ||
       layout.traces.size()<=static_cast<std::size_t>(layout.Z.cols()))
        throw std::runtime_error("invalid density layout: trace count must exceed reduced DOF");
    const double density_seconds=seconds(start);
    std::cout<<name<<" density raw="<<layout.reference_raw_dofs
             <<" reduced="<<layout.Z.cols()<<" constraints="<<layout.C.rows()
             <<" traces="<<layout.traces.size()<<std::endl;
    std::vector<RestrictResourceAnchor3D> anchors;
    for(const auto& q:layout.traces) {
        const auto uv=problem.analysis_to_native_uv(q.patch,q.u,q.v);
        RestrictResourceAnchor3D a;a.point=q.point;a.patch_id=q.patch;
        a.u=uv.x();a.v=uv.y();
        anchors.push_back(a);
    }
    const auto cover=neumann?TensorProductCoverKind3D::Q27Cover3:TensorProductCoverKind3D::Q64Cover4;
    const auto events=options.events=="all"?TraceFirstEventMode3D::LocalAllEvent:TraceFirstEventMode3D::Python93LastEvent;
    TraceFirstBuildStatistics3D ts;
    ResourceBvpOperators3D op;
    double ab_error=0,ab_seconds=0,plan_seconds=0;
    std::size_t current_requests=0,nearest_requests=0,far_requests=0,spread_fallbacks=0;
    std::size_t validated_requests=0,restrict_fallbacks=0,path_queries=0,path_hits=0,multi_paths=0,discarded_spread=0;
    double maximum_event_radius=0,maximum_target_radius=0;
    bool paths_certified=false;
    std::unique_ptr<ICorrectionBackend3D> backend;
    double spread_seconds=0.,restrict_seconds=0.;
    if(options.backend=="direct_cauchy") {
    if(!support) throw std::runtime_error("legacy correction support missing");
    const auto& legacy_support=*support;
    {
        auto plan=build_trace_first_geometry_plan_3d(grid,pair,problem.surface,legacy_support,anchors,cover,
            RestrictResourceMode3D::ReuseBySheetNode,policy(options.policy),events,{1.75,3.25,1e-13});
        current_requests=plan.current_trace_requests;nearest_requests=plan.nearest_trace_requests;
        far_requests=plan.far_trace_requests;spread_fallbacks=plan.spread_event_fallbacks;
        paths_certified=plan.support_paths_certified;
        validated_requests=plan.owner_validated_requests;restrict_fallbacks=plan.restrict_event_fallbacks;
        path_queries=plan.certified_support_paths;path_hits=plan.support_path_cache_hits;
        multi_paths=plan.multi_event_support_paths;discarded_spread=plan.python93_discarded_spread_operations;
        maximum_event_radius=plan.maximum_trace_event_distance_h;maximum_target_radius=plan.maximum_trace_target_distance_h;
        plan_seconds=plan.resource.planning_seconds;
        std::cout<<name<<" geometry plan ready in "<<plan_seconds<<" s; assembling fixed P3/P2 operators"<<std::endl;
        op=build_trace93_bvp_operators_3d(plan,grid,pair,legacy_support,problem,layout,neumann,0.0,&ts);
        if(options.compare_cache) {
            const auto ab_start=Clock::now();
            const auto reference_plan=build_trace_first_geometry_plan_3d(grid,pair,problem.surface,legacy_support,anchors,cover,
                RestrictResourceMode3D::ReferencePerVisit,policy(options.policy),events,{1.75,3.25,1e-13});
            const auto reference=build_trace93_bvp_operators_3d(reference_plan,grid,pair,legacy_support,problem,layout,neumann);
            ab_error=resource_operator_max_difference_3d(op,reference);ab_seconds=seconds(ab_start);
            if(ab_error>5e-10) throw std::runtime_error("cached/reference operator mismatch");
        }
    } // Geometry plans, centres, and row caches are not retained through GMRES.
    if(options.dump) dump_resource_operators_3d(op,(output/"matrices").string());
    const auto& B=op.trace_basis;
    if(B.rows()!=layout.trace_basis.rows() || B.cols()!=layout.trace_basis.cols())
        throw std::runtime_error("trace basis shape mismatch");
    const Eigen::SparseMatrix<double> basis_difference=B-layout.trace_basis;
    if(basis_difference.norm()>1e-10) throw std::runtime_error("operator and analysis trace bases disagree");
    spread_seconds=op.statistics.spread_seconds;restrict_seconds=op.statistics.restrict_seconds;
    backend=std::make_unique<ResourceCorrectionBackend3D>(std::move(op),trace93_density_layout_id_3d(layout),trace93_trace_layout_id_3d(layout));
    } else {
#ifdef KFBIM_HAS_SHARED_FIELD_3D
        SharedFieldBackendOptions3D field_options;
        field_options.ratio=options.field_ratio;field_options.width=options.field_width;
        field_options.fit.ridge=options.field_ridge;field_options.fit.value_weight=options.value_weight;
        field_options.fit.normal_weight=options.normal_weight;field_options.fit.pde_weight=options.pde_weight;
        field_options.restrict_mode=options.field_restrict=="staged"?SharedFieldRestrictMode3D::Staged:SharedFieldRestrictMode3D::Direct;
        Eigen::VectorXi labels(grid.num_dofs());for(int i=0;i<labels.size();++i) labels[i]=pair.domain_label(i)>0?1:0;
        std::cout<<name<<" building shared C2 field and fixed Cauchy fit"<<std::endl;
        backend=make_trace93_shared_field_backend_3d(problem,layout,grid,labels,field_options);
#else
        throw std::runtime_error("shared field unavailable in this build");
#endif
    }
    Trace93AffineTraceCoordinates3D coordinates(layout);
    const auto project=[&](const Eigen::VectorXd& t){return coordinates.project(t);};
    const auto& B=layout.trace_basis;
    const double gram_regularization=coordinates.gram_regularization(),pivot_ratio=coordinates.pivot_ratio();
    const double projection_seconds=coordinates.setup_seconds();
    const auto target=options.target=="raw"?ExteriorTarget3D::RawTrace:ExteriorTarget3D::InputJumpHalf;
    LaplaceFftBulkSolverZfft3D poisson(grid,ZfftBcType::Dirichlet,0.0,2);
    double constant_bulk_error=0,constant_trace_error=0;
    AffineKfbiTimings3D probe_timings;
    if(neumann) {
        const auto probe=evaluate_affine_kfbi_3d(*backend,poisson,Eigen::VectorXd::Ones(layout.Z.rows()),ApplyPart3D::Homogeneous,target);
        probe_timings=probe.timings;
        constant_trace_error=probe.raw_trace.value.cwiseAbs().maxCoeff();
        for(int i=0;i<grid.num_dofs();++i)
            constant_bulk_error=std::max(constant_bulk_error,std::abs(probe.grid_solution[i]-(pair.domain_label(i)>0?1.0:0.0)));
        if(std::max(constant_bulk_error,constant_trace_error)>1e-8)
            throw std::runtime_error("constant-jump identity failed before GMRES");
    }
    const bool planar_dirichlet=!neumann&&problem.geometry_name!="cylinder";
    const int restart=std::min(planar_dirichlet?100:120,coordinates.reduced_size());
    const int max_iterations=options.max_iterations?options.max_iterations:restart*(planar_dirichlet?8:12);
    AffineKfbiSolveOptions3D solve_options;solve_options.neumann=neumann;solve_options.target=target;
    solve_options.restart=restart;solve_options.max_iterations=max_iterations;solve_options.relative_tolerance=2e-10;
    const auto solved=solve_affine_kfbi_3d(*backend,coordinates,poisson,solve_options);
    const auto& c=solved.raw_coefficients;const auto& y=solved.reduced_coordinates;
    const auto& U=solved.evaluation.grid_solution;
    const auto& tv=solved.evaluation.raw_trace.value;const auto& tn=solved.evaluation.raw_trace.normal;
    const auto& ev=solved.evaluation.equation_trace.value;const auto& en=solved.evaluation.equation_trace.normal;
    const auto& trace=neumann?ev:en;const auto& raw_trace=neumann?tv:tn;
    const auto& residual=solved.projected_residual;
    const int iterations=solved.iterations;
    const double projected_relative=solved.projected_relative_residual;
    const double raw_projected_relative=project(raw_trace).norm()/std::max(solved.projected_rhs.norm(),1e-300);
    const auto leakage_for=[&](const Eigen::VectorXd& values,const Eigen::VectorXd& projected) {
        const Eigen::VectorXd representable=B*coordinates.lift_homogeneous(projected);
        const double total=std::sqrt((layout.weights.array()*values.array().square()).sum());
        return std::sqrt((layout.weights.array()*(values-representable).array().square()).sum())/std::max(total,1e-300);
    };
    const double full_norm=std::sqrt((layout.weights.array()*trace.array().square()).sum());
    const double leakage=leakage_for(trace,residual),raw_leakage=leakage_for(raw_trace,project(raw_trace));
    const double poisson_seconds=solved.timings.poisson_seconds+probe_timings.poisson_seconds;
    const int poisson_calls=solved.timings.poisson_calls+probe_timings.poisson_calls;
    const double gmres_seconds=solved.timings.gmres_seconds,matvec_seconds=solved.timings.matvec_seconds;
    const double affine_rhs_seconds=solved.timings.affine_rhs_seconds;
    double interior_linf=0,interior_sq=0,density_linf=0,density_sq=0,core_linf=0,near_boundary_linf=0;
    int inside_count=0,max_node=-1;
    Eigen::Vector3d max_point=Eigen::Vector3d::Zero();
    for(int i=0;i<grid.num_dofs();++i) if(pair.domain_label(i)>0) {
        const auto p=grid.coord(i);const Eigen::Vector3d x(p[0],p[1],p[2]);
        const double err=std::abs(U[i]-problem.evaluate(x).value);
        if(err>interior_linf) {interior_linf=err;max_node=i;max_point=x;}
        interior_sq+=err*err;++inside_count;
        bool core=true;
        for(int axis=0;axis<3;++axis) {
            Eigen::Vector3d offset=Eigen::Vector3d::Zero();offset[axis]=1.5*2.0/options.N;
            core=core&&problem.surface.exact_inside(x+offset)&&problem.surface.exact_inside(x-offset);
        }
        if(core) core_linf=std::max(core_linf,err);
        else near_boundary_linf=std::max(near_boundary_linf,err);
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
    write_vector(output/"density_coefficients.txt",c);
    write_vector(output/"reduced_coordinates.txt",y);
    write_vector(output/"raw_value_trace.txt",tv);
    write_vector(output/"raw_normal_trace.txt",tn);
    write_vector(output/"equation_value_trace.txt",ev);
    write_vector(output/"equation_normal_trace.txt",en);
    write_vector(output/"particular_coefficients.txt",layout.particular);
    if(options.dump_grid) write_vector(output/"grid_solution.txt",U);
    if(solved.final_snapshot.field_coefficients) write_vector(output/"field_coefficients.txt",*solved.final_snapshot.field_coefficients);
    if(solved.evaluation.correction.requested_jump) {
        const auto& requested=*solved.evaluation.correction.requested_jump;
        const auto& fitted=*solved.evaluation.correction.fitted_jump;
        write_vector(output/"requested_value_jump.txt",requested.value);write_vector(output/"requested_normal_jump.txt",requested.normal);
        write_vector(output/"fitted_value_jump.txt",fitted.value);write_vector(output/"fitted_normal_jump.txt",fitted.normal);
    }
    double replay_error=0.,replay_seconds=0.;
    if(options.verify_replay) {
        const auto replay_start=Clock::now();
        const auto saved_c=read_vector(output/"density_coefficients.txt",static_cast<int>(c.size()));
        const auto saved_y=read_vector(output/"reduced_coordinates.txt",static_cast<int>(y.size()));
        replay_error=relative_difference(saved_c,coordinates.particular()+coordinates.lift_homogeneous(saved_y));
        const auto replay=evaluate_affine_kfbi_3d(*backend,poisson,saved_c,ApplyPart3D::WithPrescribedData,target);
        replay_error=std::max({replay_error,relative_difference(replay.grid_solution,U),
            relative_difference(replay.raw_trace.value,read_vector(output/"raw_value_trace.txt",static_cast<int>(tv.size()))),
            relative_difference(replay.raw_trace.normal,read_vector(output/"raw_normal_trace.txt",static_cast<int>(tn.size()))),
            relative_difference(replay.equation_trace.value,read_vector(output/"equation_value_trace.txt",static_cast<int>(ev.size()))),
            relative_difference(replay.equation_trace.normal,read_vector(output/"equation_normal_trace.txt",static_cast<int>(en.size())))});
        if(solved.final_snapshot.field_coefficients) {
            const auto replay_snapshot=backend->snapshot_last_evaluation(replay.evaluation_id);
            if(!replay_snapshot.field_coefficients) throw std::runtime_error("replay lost the shared field snapshot");
            replay_error=std::max(replay_error,relative_difference(*replay_snapshot.field_coefficients,
                read_vector(output/"field_coefficients.txt",static_cast<int>(solved.final_snapshot.field_coefficients->size()))));
        }
        if(!std::isfinite(replay_error)||replay_error>1e-10) throw std::runtime_error("saved coefficient replay failed");
        replay_seconds=seconds(replay_start);
    }
    double spectrum_seconds=0.,sigma_min=0.,sigma_max=0.;
    if(options.spectrum) {
        const auto spectrum_start=Clock::now();
        const int nr=coordinates.reduced_size();Eigen::MatrixXd matrix(nr,nr);
        for(int col=0;col<nr;++col) {
            const Eigen::VectorXd direction=coordinates.lift_homogeneous(Eigen::VectorXd::Unit(nr,col));
            const auto probe=evaluate_affine_kfbi_3d(*backend,poisson,direction,ApplyPart3D::Homogeneous,target);
            matrix.col(col)=project(neumann?probe.equation_trace.value:probe.equation_trace.normal);
        }
        if(!matrix.allFinite()) throw std::runtime_error("nonfinite reduced operator");
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
        sigma_max=svd.singularValues()[0];sigma_min=svd.singularValues().tail(1)[0];
        write_matrix(output/"reduced_operator.txt",matrix);write_matrix(output/"density_nullspace.txt",layout.Z);
        write_vector(output/"operator_singular_values.txt",svd.singularValues());
        spectrum_seconds=seconds(spectrum_start);
    }
    std::ofstream history(output/"gmres_history.json");history<<std::setprecision(17)<<"[";
    for(std::size_t i=0;i<solved.recursive_residuals.size();++i) {if(i)history<<",";history<<solved.recursive_residuals[i];}history<<"]\n";
    std::ofstream j(output/"result.json");j<<std::setprecision(17)<<"{\n";
    const auto s=[&](const char* k,const std::string& v){j<<"  \""<<k<<"\": \""<<v<<"\",\n";};
    const auto n=[&](const char* k,double v){if(!std::isfinite(v))throw std::runtime_error(std::string("nonfinite result: ")+k);j<<"  \""<<k<<"\": "<<v<<",\n";};
    s("case",problem.id);s("transform",options.transform);s("bvp",name);s("chart","python93_analysis");s("geometry",problem.geometry_name);
    s("correction_backend",options.backend);s("exterior_target",options.target);
    s("density_layout_id",backend->descriptor().density_layout_id);s("trace_layout_id",backend->descriptor().trace_layout_id);
#ifdef KFBIM_BUILD_REVISION
    s("local_base_commit",KFBIM_BUILD_REVISION);
#endif
    if(options.backend=="shared_field") {
        s("source_package_sha256","23cfa03489e1d1004a57f39703e1f4237d83afb6da096fed7cc7033ecb884ce5");
        s("field_linear_algebra","Eigen SimplicialLDLT AMD; two fixed refinements");
        s("shared_field_restrict",options.field_restrict);
        n("shared_field_ratio",options.field_ratio);n("shared_field_width",options.field_width);n("shared_field_ridge",options.field_ridge);
        n("shared_field_value_weight",options.value_weight);n("shared_field_normal_weight",options.normal_weight);n("shared_field_pde_weight",options.pde_weight);
        n("density_factor",8.);n("field_lift_rcond",1e-13);n("field_refinement_count",2);
    }
    s("center_policy",options.policy);s("event_mode",options.events);s("density","python93_independent_analysis_coefficients");
    n("N",options.N);n("h",2.0/options.N);n("reference_raw_dofs",layout.reference_raw_dofs);
    n("reduced_dofs",coordinates.reduced_size());n("trace_points",layout.traces.size());
    n("constraint_rows",layout.C.rows());n("constraint_rank",layout.rank);
    n("constraint_residual",layout.constraint_residual);n("nullspace_residual",layout.nullspace_residual);
    n("final_constraint_residual",(layout.C*c-layout.d).cwiseAbs().maxCoeff());
    n("polar_essential_rank",layout.polar.essential_rank);n("polar_rank",layout.polar.polar_rank);
    n("polar_kept_modes",layout.polar.kept);n("polar_defect_modes",layout.polar.defect);
    n("polar_gap_factor",layout.polar.gap_factor);n("polar_left_null_rhs",layout.polar.left_null_rhs);
    n("edge_projector_rows",layout.edge_weights.size());n("vertex_projector_rows",layout.vertex_weights.size());
    n("gram_regularization",gram_regularization);n("gmres_restart",restart);n("gmres_max_iterations",max_iterations);
    j<<"  \"patch_spans\": [";
    for(std::size_t i=0;i<layout.patch_spans.size();++i) {if(i)j<<",";j<<"["<<layout.patch_spans[i][0]<<","<<layout.patch_spans[i][1]<<"]";}
    j<<"],\n";
    n("gram_cholesky_pivot_ratio",pivot_ratio);n("gmres_iterations",iterations);n("gmres_tolerance",2e-10);
    n("projected_relative_residual",projected_relative);n("full_exterior_value_linf",tv.cwiseAbs().maxCoeff());
    n("raw_value_linf",tv.cwiseAbs().maxCoeff());n("raw_normal_linf",tn.cwiseAbs().maxCoeff());
    n("raw_projected_relative_residual",raw_projected_relative);n("raw_projection_leakage",raw_leakage);
    n("equation_value_linf",ev.cwiseAbs().maxCoeff());n("equation_normal_linf",en.cwiseAbs().maxCoeff());
    const auto rms=[&](const Eigen::VectorXd& values){return std::sqrt((layout.weights.array()*values.array().square()).sum()/layout.weights.sum());};
    n("raw_value_rms",rms(tv));n("raw_normal_rms",rms(tn));n("equation_value_rms",rms(ev));n("equation_normal_rms",rms(en));
    n("half_jump_value_change_linf",(ev-tv).cwiseAbs().maxCoeff());n("half_jump_normal_change_linf",(en-tn).cwiseAbs().maxCoeff());
    n("final_residual_certification_limit",3e-10);n("final_evaluation_id",double(solved.final_snapshot.evaluation_id));
    j<<"  \"replay_verified\": "<<(options.verify_replay?"true":"false")<<",\n";
    if(options.verify_replay) {n("replay_normalized_error",replay_error);n("replay_seconds",replay_seconds);}
    for(const auto& item:solved.final_snapshot.scalar_diagnostics) n(item.first.c_str(),item.second);
    if(options.spectrum) {
        n("operator_sigma_min",sigma_min);n("operator_sigma_max",sigma_max);n("operator_spectrum_seconds",spectrum_seconds);
        if(sigma_min>0.) n("operator_condition",sigma_max/sigma_min);else j<<"  \"operator_condition\": null,\n";
    }
    n("full_exterior_normal_linf",tn.cwiseAbs().maxCoeff());n("active_trace_weighted_l2",full_norm);n("projection_leakage",leakage);
    n("interior_linf",interior_linf);n("cartesian_core_linf",core_linf);n("interior_l2",std::sqrt(interior_sq/inside_count));
    n("near_boundary_linf",near_boundary_linf);
    n("density_linf",density_linf);n("density_l2",std::sqrt(density_sq/layout.weights.sum()));
    n("max_error_grid_id",max_node);j<<"  \"max_error_point\": ["<<max_point.x()<<","<<max_point.y()<<","<<max_point.z()<<"],\n";
    n("shared_geometry_seconds",shared_seconds);n("density_seconds",density_seconds);n("planning_seconds",plan_seconds);
    n("spread_seconds",spread_seconds);n("restrict_seconds",restrict_seconds);
    n("projection_setup_seconds",projection_seconds);n("gmres_seconds",gmres_seconds);n("poisson_seconds",poisson_seconds);
    n("affine_rhs_seconds",affine_rhs_seconds);
    n("poisson_calls",poisson_calls);n("matvec_seconds",matvec_seconds);n("bvp_total_seconds",total_seconds);
    n("cache_comparison_seconds",ab_seconds);n("cached_reference_max_error",ab_error);
    n("current_trace_requests",current_requests);n("nearest_trace_requests",nearest_requests);n("far_trace_requests",far_requests);
    n("owner_validated_requests",validated_requests);n("restrict_event_fallbacks",restrict_fallbacks);
    n("certified_support_paths",path_queries);n("support_path_cache_hits",path_hits);
    n("multi_event_support_paths",multi_paths);n("python93_discarded_spread_operations",discarded_spread);
    n("maximum_trace_event_distance_h",maximum_event_radius);n("maximum_trace_target_distance_h",maximum_target_radius);
    n("active_directed_spread_operations",ts.active_spread_operations);
    n("spread_event_fallbacks",spread_fallbacks);n("p3_centers",ts.catalog.p3_centers);n("p2_centers",ts.catalog.p2_centers);
    n("p2_from_p3",ts.catalog.p2_from_p3);n("row_cache_hits",ts.catalog.row_cache_hits);
    n("p2_row_evaluations",ts.catalog.p2_row_evaluations);n("p3_row_evaluations",ts.catalog.p3_row_evaluations);
    n("maximum_projection_residual",ts.catalog.maximum_projection_residual);
    n("projection_not_converged",ts.catalog.projection_not_converged);
    n("constant_jump_bulk_error",constant_bulk_error);n("constant_jump_trace_error",constant_trace_error);
    j<<"  \"constant_jump_checked\": "<<(neumann?"true":"false")<<",\n";
    j<<"  \"cache_comparison_checked\": "<<(options.compare_cache?"true":"false")<<",\n"
     <<"  \"support_paths_certified\": "<<(paths_certified?"true":"false")<<",\n"
     <<"  \"gmres_converged\": "<<(solved.gmres_converged&&projected_relative<3e-10?"true":"false")<<"\n}\n";
    if(!j||!history) throw std::runtime_error("result write failed");
    std::cout<<std::setprecision(10)<<name<<" N="<<options.N<<" error="<<interior_linf<<" iter="<<iterations
             <<" residual="<<projected_relative<<" seconds="<<total_seconds<<std::endl;
    return solved.gmres_converged&&projected_relative<3e-10;
}
} // namespace

int main(int argc,char** argv) {
    try {
        const auto options=parse(argc,argv);
        if(std::filesystem::exists(options.output)) throw std::invalid_argument("refusing to overwrite existing results");
        const auto problem=make_trace93_case_3d(options.geometry,options.transform);
        (void)policy(options.policy);
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
        if(mismatches) {
            write_geometry(seconds(t),false);
            throw std::runtime_error("native grid labels disagree with exact case oracle");
        }
        std::optional<LaplaceCorrectionSupport3D> support;
        {
            try {
                if(options.backend=="shared_field") {
                    if(d.uncertified_grid_edge_count!=0)
                        throw std::runtime_error("shared field requires certified native candidate grid edges");
                    // Retain the existing event certification checks, without
                    // retaining event polynomials or constructing owner paths.
                    for(const auto& event:domain->grid_edge_event_catalog())
                        (void)native_grid_edge_event_correction_ops_3d(grid,event);
                } else support.emplace(build_laplace_correction_support_3d(pair,"trace93 study"));
            }
            catch(...) {
                const auto original=std::current_exception();
                try {write_geometry(seconds(t),false);write_uncertified_edges();}
                catch(const std::exception& e) {std::cerr<<"diagnostic write failed: "<<e.what()<<std::endl;}
                std::rethrow_exception(original);
            }
        }
        // Same successful timing boundary as before: after support setup,
        // before diagnostic file I/O or any Neumann/Dirichlet solve.
        const double shared_seconds=seconds(t);
        write_geometry(shared_seconds,true);
        if(options.geometry_only) return 0;
        bool passed=true;
        if(options.bvp!="dirichlet") passed=solve_case(options,problem,grid,pair,support?&*support:nullptr,true,shared_seconds)&&passed;
        if(options.bvp!="neumann") passed=solve_case(options,problem,grid,pair,support?&*support:nullptr,false,shared_seconds)&&passed;
        return passed?0:2;
    } catch(const std::exception& e) {std::cerr<<"trace93 study: "<<e.what()<<std::endl;return 1;}
}
