#include "construction.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d::bicubic_nurbs_detail {
void u_prism(BicubicNurbsModel3D& c)
{
    const std::vector<Eigen::Vector2d> boundary{{-.55,-.55},{-.22,-.55},{.22,-.55},{.55,-.55},
         {.55,-.1},{.55,.55},{.22,.55},{.22,-.1},{-.22,-.1},
         {-.22,.55},{-.55,.55},{-.55,-.1}};
    const std::vector<PrismCell> cells{{"bottom_left",-.55,-.22,-.55,-.1},
         {"bottom_middle",-.22,.22,-.55,-.1},{"bottom_right",.22,.55,-.55,-.1},
         {"upper_left",-.55,-.22,-.1,.55},{"upper_right",.22,.55,-.1,.55}};
    extrude_prism(c, boundary, cells);
    c.surface.exact_inside=[](const Vector& x) {
        constexpr double eps=5e-13, a=.55;
        if(!(x.z()>z0+eps && x.z()<z1-eps
            && x.x()>-a+eps && x.x()<a-eps && x.y()>-a+eps && x.y()<a-eps)) return false;
        return x.y()<-.1-eps || x.x()<-.22-eps || x.x()>.22+eps;
    };
}

} // namespace kfbim::app3d::bicubic_nurbs_detail
