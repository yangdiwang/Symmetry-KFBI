#include "src/support/solver/affine_kfbi_solve_3d.hpp"
#include "src/support/trace/affine_trace_coordinates_3d.hpp"
#include "src/support/density/trace93_density_layout_3d.hpp"
#include "src/grid/cartesian_grid_3d.hpp"
#include <iostream>
#include <stdexcept>
using namespace kfbim;using namespace kfbim::app3d;
namespace {
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
struct Bulk:ILaplaceBulkSolver3D {
    CartesianGrid3D g{{0,0,0},{1,1,1},{1,1,1},DofLayout3D::Node};
    void solve(const Eigen::VectorXd& rhs,Eigen::VectorXd& u)const override{u=-0.25*rhs;}
    const ICartesianGrid3D& grid()const override{return g;}
};
ResourceBvpOperators3D operators(){
    ResourceBvpOperators3D o;Eigen::MatrixXd s=Eigen::MatrixXd::Zero(8,2);s(0,0)=4;s(1,1)=8;o.S=s.sparseView();
    o.known_spread=Eigen::VectorXd::Zero(8);o.known_spread[0]=4;o.known_spread[1]=-8;
    Eigen::MatrixXd rg=Eigen::MatrixXd::Zero(2,8);rg(0,0)=1;rg(1,1)=1;
    o.restrict.Rg_value=rg.sparseView();o.restrict.Rg_normal=(2*rg).sparseView();
    o.restrict.exterior.Rc_value=Eigen::MatrixXd::Identity(2,2).sparseView();
    o.restrict.exterior.Rc_normal=(3*Eigen::MatrixXd::Identity(2,2)).sparseView();
    o.restrict.exterior.known_value=Eigen::VectorXd::Constant(2,1);
    o.restrict.exterior.known_normal=Eigen::VectorXd::Constant(2,2);
    o.trace_basis=Eigen::MatrixXd::Identity(2,2).sparseView();return o;
}
// A fully specified linear fitted-jump fixture, separate from the closure
// implementation, makes the equation matrix differ from its raw trace.
struct JumpBackend:ICorrectionBackend3D {
    ResourceCorrectionBackend3D resource{operators(),"density-A","trace-A"};
    CorrectionBackendDescriptor3D d=resource.descriptor();
    CorrectionDiagnosticSnapshot3D last;
    JumpBackend(){d.supports_input_jump_half=true;}
    const CorrectionBackendDescriptor3D& descriptor()const override{return d;}
    CorrectionEvaluation3D evaluate(const Eigen::VectorXd& c,ApplyPart3D p)override{
        auto out=resource.evaluate(c,p);const double known=p==ApplyPart3D::WithPrescribedData?1:0;
        out.requested_jump=TracePair3D{(2*c.array()+3*known).matrix(),(c.array()+3*known).matrix()};
        out.fitted_jump=TracePair3D{(6*c.array()+9*known).matrix(),(3*c.array()+9*known).matrix()};
        last.evaluation_id=out.evaluation_id;last.field_coefficients=c;return out;
    }
    TracePair3D observe_grid(const Eigen::VectorXd& u)const override{return resource.observe_grid(u);}
    CorrectionDiagnosticSnapshot3D snapshot_last_evaluation(std::uint64_t id)const override{
        if(id==0||id!=last.evaluation_id)throw std::invalid_argument("stale fixture snapshot");return last;
    }
};
}
int main(){try {
    Bulk bulk;ResourceCorrectionBackend3D backend(operators(),"density-A","trace-A");
    Eigen::VectorXd cp(2),inc(2);cp<<2,3;inc<<-1,4;
    auto base=evaluate_affine_kfbi_3d(backend,bulk,cp,ApplyPart3D::WithPrescribedData,ExteriorTarget3D::RawTrace);
    auto step=evaluate_affine_kfbi_3d(backend,bulk,inc,ApplyPart3D::Homogeneous,ExteriorTarget3D::RawTrace);
    auto full=evaluate_affine_kfbi_3d(backend,bulk,cp+inc,ApplyPart3D::WithPrescribedData,ExteriorTarget3D::RawTrace);
    require((full.grid_solution-base.grid_solution-step.grid_solution).norm()<1e-13,"affine grid");
    require((full.raw_trace.value-base.raw_trace.value-step.raw_trace.value).norm()<1e-13,"known trace once");
    require(full.grid_solution[0]==2 && full.grid_solution[1]==12,"Delta RHS sign");
    const auto observed=backend.observe_grid(full.grid_solution);
    const auto snapshot=backend.snapshot_last_evaluation(full.evaluation_id);
    auto zero=backend.evaluate(Eigen::VectorXd::Zero(2),ApplyPart3D::Homogeneous);
    require(zero.rhs_for_delta.norm()==0&&zero.raw_correction.value.norm()==0,"zero homogeneous");
    require((backend.observe_grid(full.grid_solution).value-observed.value).norm()==0,"observation state dependent");
    require(!snapshot.field_coefficients&&snapshot.evaluation_id==full.evaluation_id,"legacy snapshot");
    bool threw=false;try{backend.snapshot_last_evaluation(full.evaluation_id);}catch(const std::invalid_argument&){threw=true;}
    require(threw,"stale snapshot accepted");
    Trace93DensityLayout3D l;l.reference_raw_dofs=2;l.Z=Eigen::MatrixXd::Identity(2,2);l.particular=cp;
    l.weights=Eigen::VectorXd::Ones(2);l.trace_basis=Eigen::MatrixXd::Identity(2,2).sparseView();
    Trace93AffineTraceCoordinates3D coordinates(l,"density-A","trace-A");
    AffineKfbiSolveOptions3D options;options.neumann=true;options.restart=2;options.max_iterations=10;
    auto result=solve_affine_kfbi_3d(backend,coordinates,bulk,options);
    Eigen::VectorXd answer(2);answer<<-1,1.0/3;
    require((result.raw_coefficients-answer).norm()<1e-12,"affine solution");
    require(result.gmres_converged&&result.projected_relative_residual<1e-12,"true residual");
    require(result.final_snapshot.evaluation_id==result.evaluation.evaluation_id,"final snapshot ID");
    Trace93AffineTraceCoordinates3D wrong(l,"density-B","trace-A");
    threw=false;try{solve_affine_kfbi_3d(backend,wrong,bulk,options);}catch(const std::invalid_argument&){threw=true;}
    require(threw,"layout mismatch accepted");
    options.target=ExteriorTarget3D::InputJumpHalf;threw=false;
    try{solve_affine_kfbi_3d(backend,coordinates,bulk,options);}catch(const std::invalid_argument&){threw=true;}
    require(threw,"unsupported closure accepted");
    auto bad=operators();bad.restrict.Rg_normal.resize(3,8);threw=false;
    try{ResourceCorrectionBackend3D rejected(std::move(bad),"density-A","trace-A");}catch(const std::invalid_argument&){threw=true;}
    require(threw,"backend shape mismatch accepted");
    JumpBackend jumps;
    const auto raw=evaluate_affine_kfbi_3d(jumps,bulk,cp,ApplyPart3D::WithPrescribedData,ExteriorTarget3D::RawTrace);
    const auto equation=evaluate_affine_kfbi_3d(jumps,bulk,cp,ApplyPart3D::WithPrescribedData,ExteriorTarget3D::InputJumpHalf);
    require((raw.grid_solution-equation.grid_solution).norm()==0,"closure changed grid solve");
    require((raw.raw_trace.value-equation.raw_trace.value).norm()==0,"closure changed raw trace");
    Eigen::VectorXd change(2);change<<7,9;
    require((equation.equation_trace.value-equation.raw_trace.value-change).norm()==0,"closure not in forward");
    options.neumann=true;const auto jump_result=solve_affine_kfbi_3d(jumps,coordinates,bulk,options);
    answer<<-1.25,-0.4;
    require((jump_result.raw_coefficients-answer).norm()<1e-12,"half-jump missing from Neumann matvec or known added twice");
    require(jump_result.evaluation.raw_trace.value.norm()>0.1&&jump_result.projected_relative_residual<1e-12,"raw/equation result conflated");
    jumps.evaluate(cp,ApplyPart3D::Homogeneous);
    require((*jump_result.final_snapshot.field_coefficients-answer).norm()<1e-12,"final field snapshot is mutable");
    options.neumann=false;const auto dirichlet=solve_affine_kfbi_3d(jumps,coordinates,bulk,options);
    answer<<-7.0/6,-1.0/8;
    require((dirichlet.raw_coefficients-answer).norm()<1e-12,"Dirichlet must select closed normal trace");
    std::cout<<"affine_kfbi_solve_3d_test passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
