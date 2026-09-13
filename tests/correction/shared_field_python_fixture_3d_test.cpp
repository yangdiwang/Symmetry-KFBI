#include "src/support/correction/shared_field_backend_3d.hpp"
#include "src/support/trace/jump_identity_closure_3d.hpp"
#include "src/support/trace/restrict_resource_plan_3d.hpp"
#include "src/geometry/nurbs_cartesian_domain_3d.hpp"
#include "src/bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

using namespace kfbim;
using namespace kfbim::app3d;
namespace {
using Matrix=Eigen::MatrixXd;
void require(bool ok,const std::string& message) {if(!ok) throw std::runtime_error(message);}
Matrix read(const std::filesystem::path& dir,const std::string& name) {
    std::ifstream file(dir/(name+".txt"));
    int rows=0,cols=0;file>>rows>>cols;
    require(bool(file)&&rows>0&&cols>0,"invalid fixture header: "+name);
    Matrix out(rows,cols);
    for(int i=0;i<rows;++i) for(int j=0;j<cols;++j) file>>out(i,j);
    require(bool(file)&&out.allFinite(),"invalid fixture data: "+name);
    std::string extra;require(!(file>>extra),"extra fixture values: "+name);
    return out;
}
void compare(const std::string& name,const Matrix& actual,const Matrix& expected,double tolerance=1e-8) {
    require(actual.rows()==expected.rows()&&actual.cols()==expected.cols(),name+" shape mismatch");
    require(actual.allFinite()&&expected.allFinite(),name+" nonfinite");
    // Induced infinity norm (max absolute row sum), hence max absolute
    // coefficient for vectors; localized trace errors cannot average away.
    const auto inf=[](const Matrix& m){return m.cwiseAbs().rowwise().sum().maxCoeff();};
    const double error=inf(actual-expected)/std::max({1.,inf(actual),inf(expected)});
    std::cout<<name<<" normalized_error="<<error<<" linf="<<(actual-expected).cwiseAbs().maxCoeff()<<std::endl;
    require(error<=tolerance,name+" Python/C++ difference exceeds tolerance");
}
std::vector<Eigen::Vector3d> points(const Matrix& m,int first=0) {
    std::vector<Eigen::Vector3d> result;
    for(int i=0;i<m.rows();++i) result.emplace_back(m.row(i).segment(first,3).transpose());
    return result;
}
void run(const std::filesystem::path& root,const Trace93Case3D& problem,const CartesianGrid3D& grid,
         const Eigen::VectorXi& labels,bool neumann) {
    const auto dir=root/(neumann?"u_neumann_rotate_N32":"u_dirichlet_rotate_N32");
    const auto layout=build_trace93_density_layout_3d(problem,2./32,neumann);
    const Matrix raw=read(dir,"raw_direction"),surface=read(dir,"surface"),jumps=read(dir,"requested_surface");
    require(layout.coefficient_count()==raw.rows(),"raw density count differs");
    compare("legal raw direction",layout.C*raw,Matrix::Zero(layout.C.rows(),1),2e-12);
    SharedFieldSpaceOptions3D options;options.h=2./32;
    SharedFieldSpace3D space(make_trace93_shared_field_geometry_3d(problem),options);
    const Matrix indices=read(dir,"field_indices"),cells=read(dir,"active_cells");
    require(indices.rows()==space.coefficient_count(),"field dimension differs");
    require(cells.rows()==static_cast<int>(space.active_cells().size()),"active cell count differs");
    for(int i=0;i<indices.rows();++i) require((space.coefficient_lattice()[i].cast<double>()-indices.row(i).transpose()).norm()==0.,"field lattice indexing differs");
    for(int i=0;i<cells.rows();++i) require((space.active_cells()[i].cast<double>()-cells.row(i).transpose()).norm()==0.,"active cell indexing differs");
    const auto& samples=space.surface_samples();
    require(surface.rows()==static_cast<int>(samples.reference_points.size()),"surface sample count differs");
    Matrix actual_surface(surface.rows(),8),density_values(surface.rows(),1);
    for(int i=0;i<surface.rows();++i) {
        actual_surface.row(i)<<samples.reference_points[i].transpose(),samples.reference_normals[i].transpose(),samples.weights[i],samples.patches[i];
        const auto uv=samples.analysis_uv[i];density_values(i,0)=(layout.basis_row(samples.patches[i],uv.x(),uv.y())*raw)(0,0);
    }
    compare("surface physical sampling",actual_surface,surface,2e-14);
    compare("density sampling",density_values,jumps.col(neumann?0:1),2e-14);
    const auto volume=read(dir,"volume_samples");
    Matrix actual_volume=volume;
    for(int i=0;i<volume.rows();++i) {
        const int row=int(volume(i,0));actual_volume.row(i).segment(1,3)=space.volume_samples().reference_points.at(row).transpose();
        actual_volume(i,4)=space.volume_samples().weights[row];
    }
    compare("volume sampling",actual_volume,volume,2e-14);
    const Matrix probes=read(dir,"probe_points"),support=read(dir,"probe_support");
    const auto probe_points=points(probes);
    const std::pair<const char*,SharedFieldDerivative3D> derivatives[]={
        {"value",SharedFieldDerivative3D::Value},{"dx",SharedFieldDerivative3D::Dx},
        {"dy",SharedFieldDerivative3D::Dy},{"dz",SharedFieldDerivative3D::Dz},
        {"dxx",SharedFieldDerivative3D::Dxx},{"dxy",SharedFieldDerivative3D::Dxy},{"dzz",SharedFieldDerivative3D::Dzz}};
    for(const auto& item:derivatives) {
        const auto matrix=space.evaluation_matrix(probe_points,item.second);
        Matrix values(support.rows(),support.cols());
        for(int i=0;i<values.rows();++i) for(int j=0;j<values.cols();++j) values(i,j)=matrix.coeff(i,int(support(i,j)));
        const std::string derivative_name=item.first;
        const double derivative_scale=derivative_name=="value"?1.:(derivative_name.size()==2?options.h:options.h*options.h);
        compare(std::string("basis ")+item.first,derivative_scale*values,derivative_scale*read(dir,std::string("basis_")+item.first),2e-13);
    }
    const auto expected_fit=read(dir,"fit_coefficients");
    const auto expected_probe=read(dir,"probe_fit");
    const auto E=space.evaluation_matrix(probe_points);
    {
        // Fit independently from exported interface values, including harmonic P2.
        SharedFieldExtension3D extension(make_shared_field_fit_matrices_3d(space));
        compare("column scaling",extension.column_scale(),expected_fit.col(2),2e-13);
        const auto random_fit=extension.solve(jumps.col(0),jumps.col(1));
        require(extension.last_solve_backward_residual()<2e-13,"random fit backward residual");
        const auto p2_fit=extension.solve(jumps.col(2),jumps.col(3));
        compare("actual random fit response",E*random_fit,expected_probe.col(0));
        compare("actual P2 fit response",E*p2_fit,expected_probe.col(1),2e-12);
        require(extension.last_solve_backward_residual()<2e-13,"P2 fit backward residual");
        // Coefficients are recorded as a diagnostic; response/optimality are the gate.
        std::cout<<"coefficient_relative_difference="<<(random_fit-expected_fit.col(0)).norm()/std::max(1.,expected_fit.col(0).norm())<<std::endl;
    }
    SharedFieldCorrectionBackend3D backend(problem,layout,grid,labels);
    const auto correction=backend.evaluate(raw.col(0),ApplyPart3D::Homogeneous);
    const auto snapshot=backend.snapshot_last_evaluation(correction.evaluation_id);
    require(snapshot.field_coefficients.has_value(),"missing fitted coefficients");
    compare("backend fit response",E*(*snapshot.field_coefficients),expected_probe.col(0));
    const auto spread=read(dir,"spread_rhs");
    Matrix expected_rhs=Matrix::Zero(grid.num_dofs(),1);
    for(int i=0;i<spread.rows();++i) expected_rhs(int(spread(i,0)),0)=spread(i,1);
    compare("Delta spread RHS",correction.rhs_for_delta,expected_rhs);
    LaplaceFftBulkSolverZfft3D poisson(grid,ZfftBcType::Dirichlet,0.,2);
    Eigen::VectorXd U;poisson.solve(correction.rhs_for_delta,U);
    TracePair3D raw_trace=backend.observe_grid(U);
    raw_trace.value+=correction.raw_correction.value;raw_trace.normal+=correction.raw_correction.normal;
    const auto equation=close_exterior_target_3d(raw_trace,correction,ExteriorTarget3D::InputJumpHalf);
    const auto trace_points=read(dir,"trace_points"),expected=read(dir,"trace_outputs");
    Matrix actual(expected.rows(),10),physical(trace_points.rows(),6);
    for(int i=0;i<trace_points.rows();++i) {
        const int row=int(trace_points(i,0));const auto& trace=layout.traces.at(row);
        physical.row(i)<<trace.point.transpose(),trace.normal.transpose();
        actual.row(i)<<correction.raw_correction.value[row],correction.raw_correction.normal[row],raw_trace.value[row],raw_trace.normal[row],
            correction.fitted_jump->value[row],correction.fitted_jump->normal[row],correction.requested_jump->value[row],
            correction.requested_jump->normal[row],equation.value[row],equation.normal[row];
    }
    compare("trace physical alignment",physical,trace_points.middleCols(1,6),2e-14);
    for(int col=0;col<10;++col) compare("trace output column "+std::to_string(col),actual.col(col),expected.col(col));
    const auto union_samples=read(dir,"union_samples");
    std::vector<Eigen::Vector3d> union_local;
    for(int i=0;i<union_samples.rows();++i) {
        const int id=int(union_samples(i,0));const auto xyz=grid.coord(id);
        const Eigen::Vector3d x(xyz[0],xyz[1],xyz[2]);
        require((x-union_samples.row(i).segment(1,3).transpose()).norm()<2e-14,"union grid xyz numbering");
        require(labels[id]==int(union_samples(i,4)),"union native label mismatch");
        union_local.push_back(space.geometry().to_reference(x));
    }
    compare("shared union field",space.evaluation_matrix(union_local)*(*snapshot.field_coefficients),union_samples.col(5));
    require(snapshot.scalar_diagnostics.at("field_fit_calls")==1.,"backend must fit once");
    // Rg merges the two side covers and omits boundary unknowns, so its
    // sparsity cannot certify complete covers. Check all cover nodes separately.
    const auto cover_starts=read(dir,"cover_starts");
    const int kcover=neumann?3:4;
    const auto cover_kind=neumann?TensorProductCoverKind3D::Q27Cover3:TensorProductCoverKind3D::Q64Cover4;
    require(cover_starts.rows()==2*static_cast<int>(layout.traces.size()),"complete cover inventory count");
    for(int row=0;row<cover_starts.rows();++row) {
        const int trace_id=int(cover_starts(row,0)),side_id=int(cover_starts(row,1));
        const auto& trace=layout.traces.at(trace_id);
        const auto stencil=build_shared_side_cover_restrict_stencil_3d(grid,trace.point,trace.normal,cover_kind,true,1e-13);
        const auto& side=stencil.sides[side_id<0?0:1];
        require(side.desired_inside==(side_id<0),"cover side convention");
        require(side.grid_ids.size()==static_cast<std::size_t>(kcover*kcover*kcover),"cover tensor size");
        int slot=0;
        for(int k=0;k<kcover;++k) for(int j=0;j<kcover;++j) for(int i=0;i<kcover;++i,++slot) {
            const int expected_id=grid.index(int(cover_starts(row,2))+i,int(cover_starts(row,3))+j,int(cover_starts(row,4))+k);
            require(side.grid_ids[slot]==expected_id,"Python/C++ complete cover grid index differs");
        }
    }
    std::vector<Eigen::Vector3d> all_points,all_normals;
    for(const auto& trace:layout.traces) {all_points.push_back(trace.point);all_normals.push_back(trace.normal);}
    SharedFieldTransfer3D transfer(grid,labels,all_points,all_normals,space,neumann,SharedFieldRestrictMode3D::Staged);
    const auto observer=read(dir,"observer_rows");
    std::map<int,std::map<int,std::pair<double,double>>> expected_observers;
    for(int row=0;row<observer.rows();++row) {
        const int trace_id=int(observer(row,0)),grid_id=int(observer(row,1));
        require(expected_observers[trace_id].emplace(grid_id,std::make_pair(observer(row,2),observer(row,3))).second,"duplicate observer sparse index");
    }
    require(expected_observers.size()==static_cast<std::size_t>(trace_points.rows()),"observer selected row count");
    const auto& rv=transfer.grid_value_observer();const auto& rn=transfer.grid_normal_observer();
    double value_max=0.,normal_max=0.;
    const auto accumulate_error=[](double actual,double expected,double& maximum) {
        require(std::isfinite(actual)&&std::isfinite(expected),"nonfinite observer coefficient");
        maximum=std::max(maximum,std::abs(actual-expected));
    };
    for(const auto& row:expected_observers) {
        for(const auto& entry:row.second) {
            accumulate_error(rv.coeff(row.first,entry.first),entry.second.first,value_max);
            accumulate_error(rn.coeff(row.first,entry.first),entry.second.second,normal_max);
        }
        // Inspect extra C++ entries too: an unexpected off-support coefficient
        // must not be hidden by checking only exported nonzero entries.
        for(SharedFieldTransfer3D::Sparse::InnerIterator it(rv,row.first);it;++it)
            if(!row.second.count(int(it.col()))) accumulate_error(it.value(),0.,value_max);
        for(SharedFieldTransfer3D::Sparse::InnerIterator it(rn,row.first);it;++it)
            if(!row.second.count(int(it.col()))) accumulate_error(it.value(),0.,normal_max);
    }
    std::cout<<"complete cover nodes exact; Rg absolute max value="<<value_max<<" normal="<<normal_max<<std::endl;
    require(std::isfinite(value_max)&&value_max<=2e-14,"Rg value Python/C++ absolute coefficient threshold");
    require(std::isfinite(normal_max)&&normal_max<=5e-13,"Rg normal Python/C++ absolute coefficient threshold");
}
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"expected fixture directory argument");
        const std::filesystem::path root=argv[1];
        const auto problem=make_trace93_case_3d("u","rotate");
        CartesianGrid3D grid({-1.,-1.,-1.},{2./32,2./32,2./32},{32,32,32},DofLayout3D::Node);
        geometry3d::NurbsCartesianDomainOptions3D options;
        options.strategy=geometry3d::NurbsCartesianPreprocessStrategy3D::OptimizedIntersection;
        options.use_whole_grid_lines=true;
        geometry3d::NurbsCartesianDomain3D native(grid,problem.surface.geometry_model(),options);
        Eigen::VectorXi labels(grid.num_dofs()),expected=Eigen::VectorXi::Zero(grid.num_dofs());
        const auto inside=read(root/"u_dirichlet_rotate_N32","inside_full_grid_ids");
        for(int i=0;i<inside.rows();++i) expected[int(inside(i,0))]=1;
        for(int i=0;i<labels.size();++i) labels[i]=native.label(i)>0?1:0;
        require((labels-expected).cwiseAbs().maxCoeff()==0,"native/Python full-grid labels differ");
        run(root,problem,grid,labels,false);run(root,problem,grid,labels,true);
        std::cout<<"Python shared-field fixtures passed"<<std::endl;return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<std::endl;return 1;}
}
