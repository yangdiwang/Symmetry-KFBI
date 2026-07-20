#pragma once

#include <array>
#include "dof_layout.hpp"
#include "i_cartesian_grid.hpp"

namespace kfbim {

class CartesianGrid2D : public ICartesianGrid2D {
public:
    CartesianGrid2D(std::array<double, 2> origin,
                    std::array<double, 2> spacing,
                    std::array<int, 2>    num_cells,
                    DofLayout2D           layout);

    // ICartesianGrid2D interface
    int                   num_dofs()         const override;
    std::array<double, 2> coord(int idx)     const override;
    std::array<int, 4>    neighbors(int idx) const override;

    // structured access for stencil builders and LocalCauchySolver
    std::array<double, 2> coord(int i, int j) const;
    int                   index(int i, int j) const;
    std::array<int, 2>    dof_dims()          const;

    std::array<double, 2> origin()    const { return origin_; }
    std::array<double, 2> spacing()   const { return spacing_; }
    std::array<int, 2>    num_cells() const { return num_cells_; }
    DofLayout2D           layout()    const { return layout_; }

private:
    std::array<double, 2> origin_;
    std::array<double, 2> spacing_;
    std::array<int, 2>    num_cells_;
    DofLayout2D           layout_;
};

} // namespace kfbim
