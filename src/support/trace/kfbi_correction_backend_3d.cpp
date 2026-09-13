#include "kfbi_correction_backend_3d.hpp"
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>
namespace kfbim::app3d {
ResourceCorrectionBackend3D::ResourceCorrectionBackend3D(ResourceBvpOperators3D op,
    std::string density_id,std::string trace_id):operators_(std::move(op)) {
    auto& d=descriptor_;d.raw_coefficient_count=static_cast<int>(operators_.S.cols());
    d.full_grid_size=static_cast<int>(operators_.S.rows());
    d.trace_count=static_cast<int>(operators_.restrict.Rg_value.rows());
    d.density_layout_id=std::move(density_id);d.trace_layout_id=std::move(trace_id);
    const auto& r=operators_.restrict;
    const auto shape=[](const auto& m,int rows,int cols){return m.rows()==rows&&m.cols()==cols;};
    if(d.raw_coefficient_count<1||d.full_grid_size<1||d.trace_count<1||d.density_layout_id.empty()||d.trace_layout_id.empty()||
       operators_.known_spread.size()!=d.full_grid_size||
       !shape(r.Rg_value,d.trace_count,d.full_grid_size)||!shape(r.Rg_normal,d.trace_count,d.full_grid_size)||
       !shape(r.exterior.Rc_value,d.trace_count,d.raw_coefficient_count)||
       !shape(r.exterior.Rc_normal,d.trace_count,d.raw_coefficient_count)||
       r.exterior.known_value.size()!=d.trace_count||r.exterior.known_normal.size()!=d.trace_count||
       !shape(operators_.trace_basis,d.trace_count,d.raw_coefficient_count))
        throw std::invalid_argument("ResourceCorrectionBackend3D: inconsistent dimensions or empty layout ID");
}
CorrectionEvaluation3D ResourceCorrectionBackend3D::evaluate(const Eigen::VectorXd& c,ApplyPart3D part){
    if(c.size()!=descriptor_.raw_coefficient_count||!c.allFinite())throw std::invalid_argument("correction coefficient shape/value");
    if(part!=ApplyPart3D::Homogeneous&&part!=ApplyPart3D::WithPrescribedData)throw std::invalid_argument("unknown apply part");
    if(last_.evaluation_id==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("correction evaluation ID overflow");
    const auto start=std::chrono::steady_clock::now();
    CorrectionEvaluation3D result;result.evaluation_id=last_.evaluation_id+1;
    result.rhs_for_delta=operators_.S*c;
    const auto& r=operators_.restrict.exterior;
    result.raw_correction.value=r.Rc_value*c;result.raw_correction.normal=r.Rc_normal*c;
    if(part==ApplyPart3D::WithPrescribedData){result.rhs_for_delta+=operators_.known_spread;
        result.raw_correction.value+=r.known_value;result.raw_correction.normal+=r.known_normal;}
    result.rhs_for_delta=-result.rhs_for_delta;
    last_.evaluation_id=result.evaluation_id;
    last_.scalar_diagnostics["correction_evaluation_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    return result;
}
TracePair3D ResourceCorrectionBackend3D::observe_grid(const Eigen::VectorXd& u)const{
    if(u.size()!=descriptor_.full_grid_size||!u.allFinite())throw std::invalid_argument("correction grid shape/value");
    return {operators_.restrict.Rg_value*u,operators_.restrict.Rg_normal*u};
}
CorrectionDiagnosticSnapshot3D ResourceCorrectionBackend3D::snapshot_last_evaluation(std::uint64_t id)const{
    if(id==0||id!=last_.evaluation_id)throw std::invalid_argument("snapshot ID is not the last correction evaluation");
    return last_;
}
}
