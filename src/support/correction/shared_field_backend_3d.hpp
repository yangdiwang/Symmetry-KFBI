#pragma once
#include "src/support/correction/shared_field_extension_3d.hpp"
#include "src/support/trace/shared_field_transfer_3d.hpp"
#include "src/support/density/trace93_density_layout_3d.hpp"
#include "src/support/geometry/trace93_case_3d.hpp"
#include <memory>

namespace kfbim::app3d {
struct SharedFieldBackendOptions3D {
    double ratio=4.,width=4.;
    SharedFieldFitOptions3D fit;
    SharedFieldRestrictMode3D restrict_mode=SharedFieldRestrictMode3D::Staged;
};

// All prescribed data and density rows are sampled once at construction.
// The backend owns its numerical plans and does not retain case/layout/grid.
class SharedFieldCorrectionBackend3D final : public ICorrectionBackend3D {
public:
    SharedFieldCorrectionBackend3D(const Trace93Case3D& problem,
        const Trace93DensityLayout3D& layout,const CartesianGrid3D& grid,
        const Eigen::VectorXi& labels,SharedFieldBackendOptions3D options={});
    ~SharedFieldCorrectionBackend3D();
    const CorrectionBackendDescriptor3D& descriptor() const override;
    CorrectionEvaluation3D evaluate(const Eigen::VectorXd&,ApplyPart3D) override;
    TracePair3D observe_grid(const Eigen::VectorXd&) const override;
    CorrectionDiagnosticSnapshot3D snapshot_last_evaluation(std::uint64_t) const override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::unique_ptr<SharedFieldCorrectionBackend3D> make_trace93_shared_field_backend_3d(
    const Trace93Case3D&,const Trace93DensityLayout3D&,const CartesianGrid3D&,
    const Eigen::VectorXi&,SharedFieldBackendOptions3D options={});
} // namespace kfbim::app3d
