#include "mac_grid_2d.hpp"

namespace kfbim {

MACGrid2D::MACGrid2D(std::array<double, 2> origin,
                      std::array<double, 2> spacing,
                      std::array<int, 2>    num_cells)
    : p_grid_ (origin, spacing, num_cells, DofLayout2D::CellCenter),
      ux_grid_(origin, spacing, num_cells, DofLayout2D::FaceX),
      uy_grid_(origin, spacing, num_cells, DofLayout2D::FaceY)
{}

} // namespace kfbim
