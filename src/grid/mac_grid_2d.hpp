#pragma once

#include "cartesian_grid_2d.hpp"

namespace kfbim {

class MACGrid2D {
public:
    MACGrid2D(std::array<double, 2> origin,
              std::array<double, 2> spacing,
              std::array<int, 2>    num_cells);

    const CartesianGrid2D& pressure_grid()   const { return p_grid_; }
    const CartesianGrid2D& velocity_grid_x() const { return ux_grid_; }
    const CartesianGrid2D& velocity_grid_y() const { return uy_grid_; }

private:
    CartesianGrid2D p_grid_;   // CellCenter
    CartesianGrid2D ux_grid_;  // FaceX
    CartesianGrid2D uy_grid_;  // FaceY
};

} // namespace kfbim
