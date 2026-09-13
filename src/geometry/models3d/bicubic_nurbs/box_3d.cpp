#include "construction.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d::bicubic_nurbs_detail {
void box(BicubicNurbsModel3D& c)
{
    const std::vector<Eigen::Vector2d> boundary{{-.5,-.5},{.5,-.5},{.5,.5},{-.5,.5}};
    const std::vector<PrismCell> cells{{"whole",-.5,.5,-.5,.5}};
    extrude_prism(c, boundary, cells);
    c.surface.exact_inside=[](const Vector& x) {
        constexpr double eps=5e-13, a=.5;
        if(!(x.z()>z0+eps && x.z()<z1-eps
            && x.x()>-a+eps && x.x()<a-eps && x.y()>-a+eps && x.y()<a-eps)) return false;
        return true;
    };
}

} // namespace kfbim::app3d::bicubic_nurbs_detail
