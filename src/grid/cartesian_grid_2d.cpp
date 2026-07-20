#include "cartesian_grid_2d.hpp"

namespace kfbim {

CartesianGrid2D::CartesianGrid2D(std::array<double, 2> origin,
                                  std::array<double, 2> spacing,
                                  std::array<int, 2>    num_cells,
                                  DofLayout2D           layout)
    : origin_(origin), spacing_(spacing), num_cells_(num_cells), layout_(layout)
{}

// ---------------------------------------------------------------------------
// DOF layout geometry
// ---------------------------------------------------------------------------

// Number of DOF nodes in each direction.
// FaceX adds one node in x (the right face of the last cell).
// FaceY adds one node in y.  Node adds one in both.
std::array<int, 2> CartesianGrid2D::dof_dims() const {
    int nx = num_cells_[0], ny = num_cells_[1];
    switch (layout_) {
    case DofLayout2D::CellCenter: return {nx,     ny    };
    case DofLayout2D::FaceX:      return {nx + 1, ny    };
    case DofLayout2D::FaceY:      return {nx,     ny + 1};
    case DofLayout2D::Node:       return {nx + 1, ny + 1};
    }
    return {0, 0}; // unreachable
}

int CartesianGrid2D::num_dofs() const {
    auto d = dof_dims();
    return d[0] * d[1];
}

// Flat index: i varies fastest (x-direction).
// idx = j * dof_dims[0] + i
int CartesianGrid2D::index(int i, int j) const {
    return j * dof_dims()[0] + i;
}

std::array<double, 2> CartesianGrid2D::coord(int i, int j) const {
    double ox = origin_[0], oy = origin_[1];
    double dx = spacing_[0], dy = spacing_[1];
    switch (layout_) {
    case DofLayout2D::CellCenter: return {ox + (i + 0.5) * dx, oy + (j + 0.5) * dy};
    case DofLayout2D::FaceX:      return {ox +  i        * dx, oy + (j + 0.5) * dy};
    case DofLayout2D::FaceY:      return {ox + (i + 0.5) * dx, oy +  j        * dy};
    case DofLayout2D::Node:       return {ox +  i        * dx, oy +  j        * dy};
    }
    return {0.0, 0.0};
}

// ---------------------------------------------------------------------------
// ICartesianGrid2D interface
// ---------------------------------------------------------------------------

std::array<double, 2> CartesianGrid2D::coord(int idx) const {
    int nx = dof_dims()[0];
    return coord(idx % nx, idx / nx);
}

// Returns neighbors in order [−x, +x, −y, +y]; -1 at domain boundary.
std::array<int, 4> CartesianGrid2D::neighbors(int idx) const {
    auto d  = dof_dims();
    int nx  = d[0], ny = d[1];
    int i   = idx % nx;
    int j   = idx / nx;
    return {
        (i > 0)      ? j * nx + (i - 1) : -1,
        (i < nx - 1) ? j * nx + (i + 1) : -1,
        (j > 0)      ? (j - 1) * nx + i : -1,
        (j < ny - 1) ? (j + 1) * nx + i : -1
    };
}

} // namespace kfbim
