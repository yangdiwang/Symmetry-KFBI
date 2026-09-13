#include "src/support/trace/jump_identity_closure_3d.hpp"
#include <iostream>
#include <stdexcept>
using namespace kfbim::app3d;
int main() { try {
    TracePair3D raw{Eigen::VectorXd::Constant(2,3),Eigen::VectorXd::Constant(2,7)};
    CorrectionEvaluation3D c;
    c.requested_jump=TracePair3D{Eigen::VectorXd::Constant(2,5),Eigen::VectorXd::Constant(2,11)};
    c.fitted_jump=TracePair3D{Eigen::VectorXd::Constant(2,1),Eigen::VectorXd::Constant(2,3)};
    const auto closed=close_exterior_target_3d(raw,c,ExteriorTarget3D::InputJumpHalf);
    if((closed.value.array()!=1).any()||(closed.normal.array()!=3).any()) throw std::runtime_error("half jump sign/value");
    if((raw.value.array()!=3).any()) throw std::runtime_error("raw mutated");
    c.fitted_jump.reset();
    if((close_exterior_target_3d(raw,c,ExteriorTarget3D::RawTrace).normal-raw.normal).norm()!=0) throw std::runtime_error("raw target");
    bool threw=false;try {close_exterior_target_3d(raw,c,ExteriorTarget3D::InputJumpHalf);}catch(const std::invalid_argument&){threw=true;}
    if(!threw)throw std::runtime_error("missing jump accepted");
    c.fitted_jump=TracePair3D{Eigen::VectorXd::Zero(1),Eigen::VectorXd::Zero(2)};
    threw=false;try {close_exterior_target_3d(raw,c,ExteriorTarget3D::InputJumpHalf);}catch(const std::invalid_argument&){threw=true;}
    if(!threw)throw std::runtime_error("jump size mismatch accepted");
    std::cout<<"jump_identity_closure_3d_test passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
