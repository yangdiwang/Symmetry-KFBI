#include "src/support/correction/shared_field_backend_3d.hpp"
#include "src/support/trace/affine_trace_coordinates_3d.hpp"
#include "src/support/solver/affine_kfbi_solve_3d.hpp"
#include "src/bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <utility>

using namespace kfbim;
using namespace kfbim::app3d;
namespace {
void require(bool b,const char* m) {if(!b) throw std::runtime_error(m);}
double norm(const Eigen::VectorXd& v) {return v.size()?v.lpNorm<Eigen::Infinity>():0.;}
double affine_error(const Eigen::VectorXd& full,const Eigen::VectorXd& base,
    const Eigen::VectorXd& increment,double scale=1.) {
    return scale*norm(full-base-increment)/std::max({1.,scale*norm(full),scale*norm(base),scale*norm(increment)});
}
void run(bool neumann) {
    const int N=32;const double h=2./N;
    auto problem=make_trace93_case_3d("u","rotate");
    CartesianGrid3D grid({-1.,-1.,-1.},{h,h,h},{N,N,N},DofLayout3D::Node);
    Eigen::VectorXi labels(grid.num_dofs());
    for(int i=0;i<labels.size();++i) {const auto p=grid.coord(i);labels[i]=problem.surface.exact_inside({p[0],p[1],p[2]})?1:0;}
    auto prepared=[&]() {
        auto setup_problem=make_trace93_case_3d("u","rotate");
        auto setup_grid=grid;
        auto setup_labels=labels;
        auto layout=build_trace93_density_layout_3d(setup_problem,h,neumann);
        auto backend=make_trace93_shared_field_backend_3d(setup_problem,layout,setup_grid,setup_labels);
        Trace93AffineTraceCoordinates3D coordinates(layout,backend->descriptor().density_layout_id,backend->descriptor().trace_layout_id);
        return std::make_pair(std::move(backend),std::move(coordinates));
    }();
    // All setup case/layout/grid/label objects above have been destroyed.
    // Every apply and the full solve below must use owned, fixed numerical data.
    auto& backend=prepared.first;
    auto& coordinates=prepared.second;
    LaplaceFftBulkSolverZfft3D bulk(grid,ZfftBcType::Dirichlet,0.,2);
    Eigen::VectorXd y=Eigen::VectorXd::LinSpaced(coordinates.reduced_size(),-.1,.2);
    const auto c=coordinates.lift_homogeneous(y);
    const auto full=evaluate_affine_kfbi_3d(*backend,bulk,coordinates.particular()+c,ApplyPart3D::WithPrescribedData,ExteriorTarget3D::InputJumpHalf);
    require(norm(full.equation_trace.value-full.raw_trace.value-.5*(full.correction.fitted_jump->value-full.correction.requested_jump->value))<1e-12,"value half-jump closure identity");
    require(norm(full.equation_trace.normal-full.raw_trace.normal-.5*(full.correction.fitted_jump->normal-full.correction.requested_jump->normal))<1e-12,"normal half-jump closure identity");
    const auto base=evaluate_affine_kfbi_3d(*backend,bulk,coordinates.particular(),ApplyPart3D::WithPrescribedData,ExteriorTarget3D::InputJumpHalf);
    const auto inc=evaluate_affine_kfbi_3d(*backend,bulk,c,ApplyPart3D::Homogeneous,ExteriorTarget3D::InputJumpHalf);
    const double err=affine_error(full.grid_solution,base.grid_solution,inc.grid_solution);
    const double value_err=affine_error(full.equation_trace.value,base.equation_trace.value,inc.equation_trace.value);
    const double normal_err=affine_error(full.equation_trace.normal,base.equation_trace.normal,inc.equation_trace.normal,h);
    require(err<=1e-10,"shared affine normalized Linf grid identity failed");
    require(value_err<=1e-10,"shared affine normalized Linf value identity failed");
    require(normal_err<=1e-10,"shared affine h-scaled normalized Linf normal identity failed");
    std::cout<<(neumann?"N":"D")<<" affine_grid_normalized_linf="<<err
        <<" affine_value_normalized_linf="<<value_err<<" affine_h_normal_normalized_linf="<<normal_err<<'\n';
    const auto zero=backend->evaluate(Eigen::VectorXd::Zero(backend->descriptor().raw_coefficient_count),ApplyPart3D::Homogeneous);
    require(zero.rhs_for_delta.norm()==0.&&zero.raw_correction.value.norm()==0.,"shared homogeneous zero");
    const auto snapshot=backend->snapshot_last_evaluation(zero.evaluation_id);
    require(snapshot.field_coefficients.has_value(),"shared snapshot missing field coefficients");
    backend->evaluate(c,ApplyPart3D::Homogeneous);
    require(snapshot.field_coefficients->norm()==0.,"snapshot changed after next evaluation");
    bool failed=false;try {backend->snapshot_last_evaluation(zero.evaluation_id);}catch(const std::exception&) {failed=true;}
    require(failed,"stale snapshot ID must fail");
    AffineKfbiSolveOptions3D options;options.neumann=neumann;options.target=ExteriorTarget3D::InputJumpHalf;
    options.restart=std::min(neumann?120:100,coordinates.reduced_size());options.max_iterations=options.restart*(neumann?12:8);
    options.relative_tolerance=2e-10;
    const auto solved=solve_affine_kfbi_3d(*backend,coordinates,bulk,options);
    require(solved.final_snapshot.evaluation_id==solved.evaluation.evaluation_id,"final snapshot must match final forward");
    require(solved.gmres_converged&&solved.projected_relative_residual<3e-10,"shared candidate GMRES convergence");
    double error=0.;for(int i=0;i<labels.size();++i) if(labels[i]) {
        const auto p=grid.coord(i);error=std::max(error,std::abs(solved.evaluation.grid_solution[i]-problem.evaluate({p[0],p[1],p[2]}).value));
    }
    require(error<.005,"shared candidate physical grid accuracy regression");
    std::cout<<(neumann?"N":"D")<<" iterations="<<solved.iterations<<" residual="<<solved.projected_relative_residual<<" interior="<<error<<" affine="<<err<<"\n";
}
}
int main(){try{run(true);run(false);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
