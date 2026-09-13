#include "construction.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d::bicubic_nurbs_detail {
void l_prism(BicubicNurbsModel3D& c)
{
    const std::vector<Eigen::Vector2d> boundary{{-.5,-.5},{-.1,-.5},{.5,-.5},{.5,-.1},
         {-.1,-.1},{-.1,.5},{-.5,.5},{-.5,-.1}};
    const std::vector<PrismCell> cells{{"lower_left",-.5,-.1,-.5,-.1},
         {"lower_right",-.1,.5,-.5,-.1},{"upper_left",-.5,-.1,-.1,.5}};
    extrude_prism(c, boundary, cells);
    c.surface.exact_inside=[](const Vector& x) {
        constexpr double eps=5e-13, a=.5;
        if(!(x.z()>z0+eps && x.z()<z1-eps
            && x.x()>-a+eps && x.x()<a-eps && x.y()>-a+eps && x.y()<a-eps)) return false;
        return x.x()<-.1-eps || x.y()<-.1-eps;
    };
}

} // namespace kfbim::app3d::bicubic_nurbs_detail
