#include "src/support/correction/shared_field_backend_3d.hpp"
#include "src/support/trace/affine_trace_coordinates_3d.hpp"
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace kfbim::app3d {
namespace {
using Sparse=SharedFieldSparseMatrix3D;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
std::vector<Eigen::Vector3d> trace_points(const Trace93DensityLayout3D& layout,bool normals=false) {
    std::vector<Eigen::Vector3d> out;out.reserve(layout.traces.size());
    for(const auto& q:layout.traces) out.push_back(normals?q.normal:q.point);return out;
}
SharedFieldSpaceOptions3D space_options(double h,const SharedFieldBackendOptions3D& options) {
    SharedFieldSpaceOptions3D out;out.h=h;out.ratio=options.ratio;out.width=options.width;return out;
}
Sparse normal_matrix(const SharedFieldSpace3D& space,const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals) {
    Sparse out(points.size(),space.coefficient_count());
    const SharedFieldDerivative3D derivatives[]={SharedFieldDerivative3D::Dx,SharedFieldDerivative3D::Dy,SharedFieldDerivative3D::Dz};
    for(int axis=0;axis<3;++axis) {
        auto m=space.evaluation_matrix(points,derivatives[axis]);
        for(int row=0;row<m.outerSize();++row) for(Sparse::InnerIterator it(m,row);it;++it) it.valueRef()*=normals[it.row()][axis];
        out+=m;
    }
    return out;
}
}

struct SharedFieldCorrectionBackend3D::Impl {
    Clock::time_point setup_start=Clock::now();
    SharedFieldSpace3D space;
    SharedFieldTransfer3D transfer;
    std::unique_ptr<SharedFieldExtension3D> extension;
    CorrectionBackendDescriptor3D descriptor;
    Sparse density_fit,density_trace,fitted_value,fitted_normal;
    Eigen::VectorXd known_a,known_b,known_trace_a,known_trace_b,last_a,last_b,last_coefficients;
    bool neumann;
    std::uint64_t evaluation_id=0;
    double fit_seconds=0.,transfer_seconds=0.,setup_seconds=0.;

    Impl(const Trace93Case3D& problem,const Trace93DensityLayout3D& layout,
        const CartesianGrid3D& grid,const Eigen::VectorXi& labels,const SharedFieldBackendOptions3D& options)
      :space(make_trace93_shared_field_geometry_3d(problem),space_options(grid.spacing()[0],options)),
       transfer(grid,labels,trace_points(layout),trace_points(layout,true),space,layout.neumann,options.restrict_mode),
       neumann(layout.neumann)
    {
        if(layout.problem!=&problem||layout.reference_raw_dofs<1||std::abs(layout.h-grid.spacing()[0])>1e-14*grid.spacing()[0]
           ||layout.trace_basis.rows()!=static_cast<int>(layout.traces.size())||layout.trace_basis.cols()!=layout.reference_raw_dofs)
            throw std::invalid_argument("shared backend requires the matching Trace93 case, layout and grid");
        const auto& samples=space.surface_samples();
        std::vector<Eigen::Triplet<double>> entries;entries.reserve(samples.reference_points.size()*16);
        for(int q=0;q<static_cast<int>(samples.reference_points.size());++q) {
            const auto uv=samples.analysis_uv[q];
            const auto row=layout.basis_stencil(samples.patches[q],uv.x(),uv.y());
            for(int i=0;i<row.count;++i) entries.emplace_back(q,row.indices[i],row.weights[i]);
        }
        density_fit.resize(samples.reference_points.size(),layout.reference_raw_dofs);
        density_fit.setFromTriplets(entries.begin(),entries.end());density_trace=layout.trace_basis;
        std::vector<Eigen::Vector3d> points,normals;points.reserve(layout.traces.size());normals.reserve(layout.traces.size());
        for(const auto& q:layout.traces) {
            // The half-jump coefficient is valid at face-interior traces only.
            if(!(q.u>0.&&q.u<1.&&q.v>0.&&q.v<1.)||!q.point.allFinite()||!q.normal.allFinite())
                throw std::invalid_argument("shared half-jump backend requires face-interior trace points");
            points.push_back(space.geometry().to_reference(q.point));
            normals.push_back(space.geometry().normal_to_reference(q.normal));
        }
        fitted_value=space.evaluation_matrix(points);
        fitted_normal=normal_matrix(space,points,normals);
        // Transfer/trace coverage was checked before the expensive factorization.
        extension=std::make_unique<SharedFieldExtension3D>(make_shared_field_fit_matrices_3d(space),options.fit);
        known_a=Eigen::VectorXd::Zero(density_fit.rows());known_b=known_a;
        known_trace_a=Eigen::VectorXd::Zero(layout.traces.size());known_trace_b=known_trace_a;
        for(int q=0;q<known_a.size();++q) {
            const auto data=problem.evaluate(samples.world_points[q]);
            if(neumann) known_b[q]=data.gradient.dot(samples.world_normals[q]);else known_a[q]=data.value;
        }
        for(int q=0;q<known_trace_a.size();++q) {
            const auto& trace=layout.traces[q];const auto data=problem.evaluate(trace.point);
            if(neumann) known_trace_b[q]=data.gradient.dot(trace.normal);else known_trace_a[q]=data.value;
        }
        descriptor={layout.reference_raw_dofs,grid.num_dofs(),static_cast<int>(layout.traces.size()),true,
            trace93_density_layout_id_3d(layout),trace93_trace_layout_id_3d(layout)};
        setup_seconds=seconds(setup_start);
    }
};

SharedFieldCorrectionBackend3D::SharedFieldCorrectionBackend3D(const Trace93Case3D& problem,
    const Trace93DensityLayout3D& layout,const CartesianGrid3D& grid,const Eigen::VectorXi& labels,SharedFieldBackendOptions3D options)
    :impl_(std::make_unique<Impl>(problem,layout,grid,labels,options)) {}
SharedFieldCorrectionBackend3D::~SharedFieldCorrectionBackend3D()=default;
const CorrectionBackendDescriptor3D& SharedFieldCorrectionBackend3D::descriptor() const{return impl_->descriptor;}

CorrectionEvaluation3D SharedFieldCorrectionBackend3D::evaluate(const Eigen::VectorXd& c,ApplyPart3D part) {
    auto& p=*impl_;
    if(c.size()!=p.descriptor.raw_coefficient_count||!c.allFinite()) throw std::invalid_argument("shared backend density size/finite check failed");
    if(part!=ApplyPart3D::Homogeneous&&part!=ApplyPart3D::WithPrescribedData) throw std::invalid_argument("invalid shared apply part");
    Eigen::VectorXd a=Eigen::VectorXd::Zero(p.density_fit.rows()),b=a;
    TracePair3D requested{Eigen::VectorXd::Zero(p.descriptor.trace_count),Eigen::VectorXd::Zero(p.descriptor.trace_count)};
    if(p.neumann) {a=p.density_fit*c;requested.value=p.density_trace*c;}
    else {b=p.density_fit*c;requested.normal=p.density_trace*c;}
    if(part==ApplyPart3D::WithPrescribedData) {a+=p.known_a;b+=p.known_b;requested.value+=p.known_trace_a;requested.normal+=p.known_trace_b;}
    auto start=Clock::now();Eigen::VectorXd coefficients=p.extension->solve(a,b);p.fit_seconds+=seconds(start);
    start=Clock::now();auto transferred=p.transfer.evaluate(coefficients);p.transfer_seconds+=seconds(start);
    CorrectionEvaluation3D out;out.rhs_for_delta=std::move(transferred.rhs_for_delta);out.raw_correction=std::move(transferred.raw_correction);
    out.requested_jump=std::move(requested);out.fitted_jump=TracePair3D{p.fitted_value*coefficients,p.fitted_normal*coefficients};
    p.last_a=std::move(a);p.last_b=std::move(b);p.last_coefficients=std::move(coefficients);
    out.evaluation_id=++p.evaluation_id;return out;
}

TracePair3D SharedFieldCorrectionBackend3D::observe_grid(const Eigen::VectorXd& U) const {return impl_->transfer.observe_grid(U);}

CorrectionDiagnosticSnapshot3D SharedFieldCorrectionBackend3D::snapshot_last_evaluation(std::uint64_t id) const {
    const auto& p=*impl_;
    if(id==0||id!=p.evaluation_id) throw std::invalid_argument("shared snapshot requested with stale evaluation ID");
    CorrectionDiagnosticSnapshot3D out;out.evaluation_id=id;out.field_coefficients=p.last_coefficients;
    out.scalar_diagnostics=p.transfer.diagnostics();
    auto& d=out.scalar_diagnostics;const auto& setup=p.extension->setup_diagnostics();
    const auto fit=p.extension->diagnostics(p.last_coefficients,p.last_a,p.last_b);
    d["field_coefficients"]=p.space.coefficient_count();d["field_active_cells"]=double(p.space.active_cells().size());
    d["field_H"]=p.space.spacing();d["field_W"]=p.space.options().width*p.space.options().h;
    d["field_surface_samples"]=double(p.space.surface_samples().reference_points.size());d["field_volume_samples"]=double(p.space.volume_samples().reference_points.size());
    d["field_matrix_nnz"]=double(setup.matrix_nnz);d["field_gram_nnz"]=double(setup.gram_nnz);d["field_factor_nnz"]=double(setup.factor_nnz);
    d["field_estimated_bytes"]=double(setup.estimated_bytes);d["field_factorization_count"]=setup.factorization_count;
    d["field_fit_setup_seconds"]=setup.setup_seconds;d["field_backend_setup_seconds"]=p.setup_seconds;
    d["field_fit_calls"]=double(p.evaluation_id);d["field_fit_seconds"]=p.fit_seconds;d["field_transfer_seconds"]=p.transfer_seconds;
    d["jump_fit_linf"]=fit.value_linf;d["normal_jump_fit_linf"]=fit.normal_linf;d["pde_fit_linf"]=fit.pde_linf;
    d["jump_fit_rms"]=fit.value_rms;d["normal_jump_fit_rms"]=fit.normal_rms;d["pde_fit_rms"]=fit.pde_rms;
    d["field_normalized_optimality"]=fit.normalized_optimality;d["off_surface_exact_samples_used"]=0.;d["correction_path_queries"]=0.;
    d["field_solve_backward_residual"]=p.extension->last_solve_backward_residual();
    for(const auto& item:d) if(!std::isfinite(item.second)) throw std::runtime_error("nonfinite shared field diagnostic: "+item.first);
    return out;
}

std::unique_ptr<SharedFieldCorrectionBackend3D> make_trace93_shared_field_backend_3d(
    const Trace93Case3D& problem,const Trace93DensityLayout3D& layout,const CartesianGrid3D& grid,
    const Eigen::VectorXi& labels,SharedFieldBackendOptions3D options) {
    return std::make_unique<SharedFieldCorrectionBackend3D>(problem,layout,grid,labels,options);
}
} // namespace kfbim::app3d
