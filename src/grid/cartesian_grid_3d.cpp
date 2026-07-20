#include "cartesian_grid_3d.hpp"

namespace kfbim {

CartesianGrid3D::CartesianGrid3D(std::array<double, 3> origin,
                                  std::array<double, 3> spacing,
                                  std::array<int, 3>    num_cells,
                                  DofLayout3D           layout)
    : origin_(origin), spacing_(spacing), num_cells_(num_cells), layout_(layout)
{}

// ---------------------------------------------------------------------------
// DOF layout geometry
// ---------------------------------------------------------------------------

std::array<int, 3> CartesianGrid3D::dof_dims() const {
    int nx = num_cells_[0], ny = num_cells_[1], nz = num_cells_[2];
    switch (layout_) {
    case DofLayout3D::CellCenter: return {nx,     ny,     nz    };
    case DofLayout3D::FaceX:      return {nx + 1, ny,     nz    };
    case DofLayout3D::FaceY:      return {nx,     ny + 1, nz    };
    case DofLayout3D::FaceZ:      return {nx,     ny,     nz + 1};
    case DofLayout3D::Node:       return {nx + 1, ny + 1, nz + 1};
    }
    return {0, 0, 0};
}

int CartesianGrid3D::num_dofs() const {
    auto d = dof_dims();
    return d[0] * d[1] * d[2];
}

// Flat index: i varies fastest, then j, then k.
// idx = k * (nx_d * ny_d) + j * nx_d + i
int CartesianGrid3D::index(int i, int j, int k) const {
    auto d = dof_dims();
    return k * (d[0] * d[1]) + j * d[0] + i;
}

std::array<double, 3> CartesianGrid3D::coord(int i, int j, int k) const {
    double ox = origin_[0], oy = origin_[1], oz = origin_[2];
    double dx = spacing_[0], dy = spacing_[1], dz = spacing_[2];
    switch (layout_) {
    case DofLayout3D::CellCenter:
        return {ox + (i + 0.5) * dx, oy + (j + 0.5) * dy, oz + (k + 0.5) * dz};
    case DofLayout3D::FaceX:
        return {ox +  i        * dx, oy + (j + 0.5) * dy, oz + (k + 0.5) * dz};
    case DofLayout3D::FaceY:
        return {ox + (i + 0.5) * dx, oy +  j        * dy, oz + (k + 0.5) * dz};
    case DofLayout3D::FaceZ:
        return {ox + (i + 0.5) * dx, oy + (j + 0.5) * dy, oz +  k        * dz};
    case DofLayout3D::Node:
        return {ox +  i        * dx, oy +  j        * dy, oz +  k        * dz};
    }
    return {0.0, 0.0, 0.0};
}

// ---------------------------------------------------------------------------
// ICartesianGrid3D interface
// ---------------------------------------------------------------------------

std::array<double, 3> CartesianGrid3D::coord(int idx) const {
    auto d  = dof_dims();
    int  nxy = d[0] * d[1];
    int  k   = idx / nxy;
    int  rem = idx % nxy;
    int  j   = rem / d[0];
    int  i   = rem % d[0];
    return coord(i, j, k);
}

// Returns neighbors in order [−x, +x, −y, +y, −z, +z]; -1 at domain boundary.
std::array<int, 6> CartesianGrid3D::neighbors(int idx) const {
    auto d   = dof_dims();
    int  nx  = d[0], ny = d[1], nz = d[2];
    int  nxy = nx * ny;
    int  k   = idx / nxy;
    int  rem = idx % nxy;
    int  j   = rem / nx;
    int  i   = rem % nx;
    return {
        (i > 0)      ? idx - 1   : -1,
        (i < nx - 1) ? idx + 1   : -1,
        (j > 0)      ? idx - nx  : -1,
        (j < ny - 1) ? idx + nx  : -1,
        (k > 0)      ? idx - nxy : -1,
        (k < nz - 1) ? idx + nxy : -1
    };
}

} // namespace kfbim
