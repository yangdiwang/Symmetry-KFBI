#pragma once

#include <array>
#include "dof_layout.hpp"
#include "i_cartesian_grid.hpp"

namespace kfbim {

// Uniform scalar grid in 3D. Implements ICartesianGrid3D for use by BulkSolver
// and higher layers. Also exposes (i,j,k) indexing for stencil construction
// and LocalCauchySolver, which need structured grid access.
// Does not own field data — data is stored externally as Eigen::VectorXd.
// Constructor takes number of cells; DOF count is derived from layout.
class CartesianGrid3D : public ICartesianGrid3D {
public:
    // num_cells: number of cells in each direction (not DOFs)
    // spacing:   cell size in each direction (hx, hy, hz)
    CartesianGrid3D(std::array<double, 3> origin,
                    std::array<double, 3> spacing,
                    std::array<int, 3>    num_cells,
                    DofLayout3D           layout);

    // ICartesianGrid3D interface (used by BulkSolver and above)
    int                   num_dofs()         const override;
    std::array<double, 3> coord(int idx)     const override;
    std::array<int, 6>    neighbors(int idx) const override;

    // structured access — available only on uniform grids (used by stencil builders
    // and LocalCauchySolver, which need (i,j,k) layout knowledge)
    std::array<double, 3> coord(int i, int j, int k) const;
    int                   index(int i, int j, int k) const;
    std::array<int, 3>    dof_dims()                 const;

    std::array<double, 3> origin()    const { return origin_; }
    std::array<double, 3> spacing()   const { return spacing_; }
    std::array<int, 3>    num_cells() const { return num_cells_; }
    DofLayout3D           layout()    const { return layout_; }

private:
    std::array<double, 3> origin_;
    std::array<double, 3> spacing_;
    std::array<int, 3>    num_cells_;
    DofLayout3D           layout_;
};

} // namespace kfbim
