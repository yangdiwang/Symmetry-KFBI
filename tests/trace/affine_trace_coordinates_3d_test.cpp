#include "src/support/trace/affine_trace_coordinates_3d.hpp"
#include "src/support/density/trace93_density_layout_3d.hpp"
#include <iostream>
#include <stdexcept>
using namespace kfbim::app3d;
int main(){try {
    Trace93DensityLayout3D l;l.reference_raw_dofs=2;l.Z=Eigen::MatrixXd::Identity(2,2);
    l.particular=Eigen::VectorXd::Constant(2,3);l.weights=Eigen::VectorXd::Ones(3);
    Eigen::MatrixXd B(3,2);B<<1,0,0,1,1,1;l.trace_basis=B.sparseView();
    Eigen::MatrixXd e(1,2);e<<2,0;l.edge_basis=e.sparseView();
    Eigen::MatrixXd v(1,2);v<<0,3;l.vertex_basis=v.sparseView();
    Eigen::MatrixXd et(1,3);et<<1,0,0;l.edge_trace=et.sparseView();
    Eigen::MatrixXd vt(1,3);vt<<0,1,0;l.vertex_trace=vt.sparseView();
    l.edge_weights=Eigen::VectorXd::Constant(1,0.5);l.vertex_weights=Eigen::VectorXd::Constant(1,2);
    const auto density_id=trace93_density_layout_id_3d(l),trace_id=trace93_trace_layout_id_3d(l);
    auto altered=l;altered.Z(0,0)=2;
    if(trace93_density_layout_id_3d(altered)==density_id)throw std::runtime_error("density coordinate semantics absent from ID");
    altered=l;altered.edge_weights[0]=1;
    if(trace93_trace_layout_id_3d(altered)==trace_id)throw std::runtime_error("trace projection semantics absent from ID");
    altered=l;
    if(trace93_density_layout_id_3d(altered)!=density_id||trace93_trace_layout_id_3d(altered)!=trace_id)throw std::runtime_error("IDs depend on object address");
    Trace93AffineTraceCoordinates3D c(l,"density-A","trace-A");
    Eigen::VectorXd t(3);t<<2,3,5;
    // M = [[4,1],[1,20]], r = [9,26], hence M^-1 r = [154,95]/79.
    Eigen::VectorXd expected(2);expected<<154.0/79,95.0/79;
    if((c.project(t)-expected).norm()>1e-13)throw std::runtime_error("edge/vertex projection lost");
    l.Z.setZero();l.particular.setZero();
    if((c.lift_homogeneous(expected)-expected).norm()!=0||c.particular()[0]!=3)throw std::runtime_error("coordinate lifetime");
    bool threw=false;try{c.project(Eigen::VectorXd::Zero(2));}catch(const std::invalid_argument&){threw=true;}
    if(!threw)throw std::runtime_error("trace shape unchecked");
    std::cout<<"affine_trace_coordinates_3d_test passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
