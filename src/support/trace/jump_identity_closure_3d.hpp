#pragma once
#include "src/support/trace/kfbi_correction_backend_3d.hpp"
namespace kfbim::app3d {
TracePair3D close_exterior_target_3d(const TracePair3D&,const CorrectionEvaluation3D&,ExteriorTarget3D);
}
