#pragma once
#include "src/support/trace/spread_restrict_resource_3d.hpp"
#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace kfbim::app3d {
enum class ApplyPart3D { Homogeneous, WithPrescribedData };
enum class ExteriorTarget3D { RawTrace, InputJumpHalf };
struct TracePair3D { Eigen::VectorXd value, normal; };
struct CorrectionEvaluation3D {
    std::uint64_t evaluation_id=0;
    Eigen::VectorXd rhs_for_delta;
    TracePair3D raw_correction;
    std::optional<TracePair3D> requested_jump, fitted_jump;
};
struct CorrectionBackendDescriptor3D {
    int raw_coefficient_count=0, full_grid_size=0, trace_count=0;
    bool supports_input_jump_half=false;
    std::string density_layout_id, trace_layout_id;
};
struct CorrectionDiagnosticSnapshot3D {
    std::uint64_t evaluation_id=0;
    std::optional<Eigen::VectorXd> field_coefficients;
    std::map<std::string,double> scalar_diagnostics;
};
class ICorrectionBackend3D {
public:
    virtual ~ICorrectionBackend3D()=default;
    virtual const CorrectionBackendDescriptor3D& descriptor()const=0;
    virtual CorrectionEvaluation3D evaluate(const Eigen::VectorXd&,ApplyPart3D)=0;
    virtual TracePair3D observe_grid(const Eigen::VectorXd&)const=0;
    virtual CorrectionDiagnosticSnapshot3D snapshot_last_evaluation(std::uint64_t)const=0;
};
// Owns all matrices. Observation is independent of the last evaluation.
class ResourceCorrectionBackend3D final:public ICorrectionBackend3D {
public:
    ResourceCorrectionBackend3D(ResourceBvpOperators3D,std::string density_layout_id,std::string trace_layout_id);
    const CorrectionBackendDescriptor3D& descriptor()const override{return descriptor_;}
    CorrectionEvaluation3D evaluate(const Eigen::VectorXd&,ApplyPart3D)override;
    TracePair3D observe_grid(const Eigen::VectorXd&)const override;
    CorrectionDiagnosticSnapshot3D snapshot_last_evaluation(std::uint64_t)const override;
private:
    ResourceBvpOperators3D operators_;
    CorrectionBackendDescriptor3D descriptor_;
    CorrectionDiagnosticSnapshot3D last_;
};
} // namespace kfbim::app3d
