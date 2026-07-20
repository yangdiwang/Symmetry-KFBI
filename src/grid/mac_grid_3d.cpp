#include "mac_grid_3d.hpp"

namespace kfbim {

MACGrid3D::MACGrid3D(std::array<double, 3> origin,
                      std::array<double, 3> spacing,
                      std::array<int, 3>    num_cells)
    : p_grid_ (origin, spacing, num_cells, DofLayout3D::CellCenter),
      ux_grid_(origin, spacing, num_cells, DofLayout3D::FaceX),
      uy_grid_(origin, spacing, num_cells, DofLayout3D::FaceY),
      uz_grid_(origin, spacing, num_cells, DofLayout3D::FaceZ)
{}

} // namespace kfbim
