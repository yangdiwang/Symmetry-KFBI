#pragma once

#include "cartesian_grid_3d.hpp"

namespace kfbim {

// Staggered MAC grid for 3D Stokes/Navier-Stokes.
// Composes four CartesianGrid3D descriptors sharing the same cell decomposition.
// Data for each component (ux, uy, uz, p) is stored externally as Eigen::VectorXd.
class MACGrid3D {
public:
    // num_cells: number of pressure cells in each direction
    MACGrid3D(std::array<double, 3> origin,
              std::array<double, 3> spacing,
              std::array<int, 3>    num_cells);

    const CartesianGrid3D& pressure_grid()   const { return p_grid_; }
    const CartesianGrid3D& velocity_grid_x() const { return ux_grid_; }
    const CartesianGrid3D& velocity_grid_y() const { return uy_grid_; }
    const CartesianGrid3D& velocity_grid_z() const { return uz_grid_; }

private:
    CartesianGrid3D p_grid_;   // CellCenter
    CartesianGrid3D ux_grid_;  // FaceX
    CartesianGrid3D uy_grid_;  // FaceY
    CartesianGrid3D uz_grid_;  // FaceZ
};

} // namespace kfbim
