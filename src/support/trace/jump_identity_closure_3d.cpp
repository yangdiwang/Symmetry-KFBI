#include "jump_identity_closure_3d.hpp"
#include <stdexcept>
namespace kfbim::app3d {
TracePair3D close_exterior_target_3d(const TracePair3D& raw,const CorrectionEvaluation3D& correction,ExteriorTarget3D target){
    if(raw.value.size()!=raw.normal.size()||!raw.value.allFinite()||!raw.normal.allFinite())
        throw std::invalid_argument("raw trace shape/value");
    if(target==ExteriorTarget3D::RawTrace)return raw;
    if(target!=ExteriorTarget3D::InputJumpHalf)throw std::invalid_argument("unknown exterior target");
    if(!correction.requested_jump||!correction.fitted_jump)throw std::invalid_argument("input jump half requires requested and fitted jumps");
    const auto valid=[&](const TracePair3D& p){return p.value.size()==raw.value.size()&&p.normal.size()==raw.normal.size()&&p.value.allFinite()&&p.normal.allFinite();};
    if(!valid(*correction.requested_jump)||!valid(*correction.fitted_jump))throw std::invalid_argument("jump shape/value mismatch");
    return {raw.value+0.5*(correction.fitted_jump->value-correction.requested_jump->value),
        raw.normal+0.5*(correction.fitted_jump->normal-correction.requested_jump->normal)};
}
}
