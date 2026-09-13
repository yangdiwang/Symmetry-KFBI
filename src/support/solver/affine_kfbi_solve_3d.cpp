#include "affine_kfbi_solve_3d.hpp"
#include "src/support/trace/jump_identity_closure_3d.hpp"
#include "src/gmres/gmres.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <stdexcept>
namespace kfbim::app3d {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
void validate(const CorrectionBackendDescriptor3D& d,const ILaplaceBulkSolver3D& bulk,ExteriorTarget3D target){
    if(d.raw_coefficient_count<1||d.trace_count<1||d.full_grid_size!=bulk.grid().num_dofs()||d.density_layout_id.empty()||d.trace_layout_id.empty())
        throw std::invalid_argument("affine KFBI backend/bulk dimensions or layout ID");
    if(target!=ExteriorTarget3D::RawTrace&&target!=ExteriorTarget3D::InputJumpHalf)throw std::invalid_argument("unknown exterior target");
    if(target==ExteriorTarget3D::InputJumpHalf&&!d.supports_input_jump_half)throw std::invalid_argument("backend does not support input jump half at its trace points");
}
void validate_trace(const TracePair3D& t,int size){if(t.value.size()!=size||t.normal.size()!=size||!t.value.allFinite()||!t.normal.allFinite())throw std::invalid_argument("backend trace dimensions/values");}
const Eigen::VectorXd& active(const TracePair3D& p,bool neumann){return neumann?p.value:p.normal;}
void add(AffineKfbiTimings3D& a,const AffineKfbiTimings3D& b){a.backend_seconds+=b.backend_seconds;a.poisson_seconds+=b.poisson_seconds;a.observe_seconds+=b.observe_seconds;a.poisson_calls+=b.poisson_calls;}
class Apply final:public IKFBIOperator{
public:int size=0;std::function<void(const Eigen::VectorXd&,Eigen::VectorXd&)> fn;
    int problem_size()const override{return size;}
    void apply(const Eigen::VectorXd& x,Eigen::VectorXd& y)const override{fn(x,y);}
};
}
AffineKfbiEvaluation3D evaluate_affine_kfbi_3d(ICorrectionBackend3D& backend,const ILaplaceBulkSolver3D& bulk,
    const Eigen::VectorXd& c,ApplyPart3D part,ExteriorTarget3D target){
    const auto& d=backend.descriptor();validate(d,bulk,target);
    if(c.size()!=d.raw_coefficient_count||!c.allFinite())throw std::invalid_argument("affine KFBI coefficient dimensions/values");
    if(part!=ApplyPart3D::Homogeneous&&part!=ApplyPart3D::WithPrescribedData)throw std::invalid_argument("unknown apply part");
    AffineKfbiEvaluation3D result;auto t=Clock::now();result.correction=backend.evaluate(c,part);result.timings.backend_seconds=elapsed(t);
    const auto& correction=result.correction;result.evaluation_id=correction.evaluation_id;
    if(!result.evaluation_id||correction.rhs_for_delta.size()!=d.full_grid_size||!correction.rhs_for_delta.allFinite())throw std::invalid_argument("backend correction RHS dimensions/values/ID");
    validate_trace(correction.raw_correction,d.trace_count);
    t=Clock::now();bulk.solve(correction.rhs_for_delta,result.grid_solution);result.timings.poisson_seconds=elapsed(t);result.timings.poisson_calls=1;
    if(result.grid_solution.size()!=d.full_grid_size||!result.grid_solution.allFinite())throw std::runtime_error("bulk solution dimensions/values");
    t=Clock::now();result.raw_trace=backend.observe_grid(result.grid_solution);validate_trace(result.raw_trace,d.trace_count);
    result.raw_trace.value+=correction.raw_correction.value;result.raw_trace.normal+=correction.raw_correction.normal;
    result.equation_trace=close_exterior_target_3d(result.raw_trace,correction,target);result.timings.observe_seconds=elapsed(t);
    return result;
}
AffineKfbiSolveResult3D solve_affine_kfbi_3d(ICorrectionBackend3D& backend,const IAffineTraceCoordinates3D& coordinates,
    const ILaplaceBulkSolver3D& bulk,const AffineKfbiSolveOptions3D& options){
    const auto& d=backend.descriptor();validate(d,bulk,options.target);
    if(coordinates.raw_coefficient_count()!=d.raw_coefficient_count||coordinates.trace_count()!=d.trace_count||coordinates.reduced_size()<1||
       coordinates.particular().size()!=d.raw_coefficient_count||coordinates.density_layout_id()!=d.density_layout_id||coordinates.trace_layout_id()!=d.trace_layout_id)
        throw std::invalid_argument("affine KFBI coordinate dimensions or layout IDs disagree");
    if(!std::isfinite(options.relative_tolerance)||options.relative_tolerance<=0||options.restart<1||options.max_iterations<1)throw std::invalid_argument("affine KFBI GMRES options");
    AffineKfbiSolveResult3D result;auto t=Clock::now();
    const auto base=evaluate_affine_kfbi_3d(backend,bulk,coordinates.particular(),ApplyPart3D::WithPrescribedData,options.target);
    add(result.timings,base.timings);result.projected_rhs=-coordinates.project(active(base.equation_trace,options.neumann));
    result.timings.affine_rhs_seconds=elapsed(t);
    Apply op;op.size=coordinates.reduced_size();op.fn=[&](const Eigen::VectorXd& y,Eigen::VectorXd& output){
        const auto begin=Clock::now();const Eigen::VectorXd c=coordinates.lift_homogeneous(y);
        const auto evaluation=evaluate_affine_kfbi_3d(backend,bulk,c,ApplyPart3D::Homogeneous,options.target);
        add(result.timings,evaluation.timings);output=coordinates.project(active(evaluation.equation_trace,options.neumann));
        result.timings.matvec_seconds+=elapsed(begin);++result.timings.matvec_calls;};
    GMRES gmres(options.max_iterations,options.relative_tolerance,std::min(options.restart,op.size));
    result.reduced_coordinates=Eigen::VectorXd::Zero(op.size);t=Clock::now();
    result.iterations=gmres.solve(op,result.projected_rhs,result.reduced_coordinates);result.timings.gmres_seconds=elapsed(t);
    result.gmres_converged=gmres.converged();result.recursive_residuals=gmres.residuals();
    t=Clock::now();result.raw_coefficients=coordinates.particular()+coordinates.lift_homogeneous(result.reduced_coordinates);
    result.evaluation=evaluate_affine_kfbi_3d(backend,bulk,result.raw_coefficients,ApplyPart3D::WithPrescribedData,options.target);
    // Snapshot before any possible subsequent backend apply. It owns its data.
    result.final_snapshot=backend.snapshot_last_evaluation(result.evaluation.evaluation_id);
    if(result.final_snapshot.evaluation_id!=result.evaluation.evaluation_id)throw std::runtime_error("backend snapshot ID mismatch");
    add(result.timings,result.evaluation.timings);
    result.projected_residual=coordinates.project(active(result.evaluation.equation_trace,options.neumann));
    result.projected_relative_residual=result.projected_residual.norm()/std::max(result.projected_rhs.norm(),1e-300);
    result.timings.final_seconds=elapsed(t);return result;
}
}
