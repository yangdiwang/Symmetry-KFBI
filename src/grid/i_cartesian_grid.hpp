#pragma once

#include <array>
#include <vector>

namespace kfbim {

// Abstract interface for a scalar Cartesian grid (uniform or adaptive).
// BulkSolver and all layers above depend only on this — not on the concrete type.
// (i,j,k) indexing is intentionally absent here; only flat-index operations are virtual.
class ICartesianGrid2D {
public:
    virtual ~ICartesianGrid2D() = default;

    virtual int                   num_dofs()          const = 0;
    virtual std::array<double, 2> coord(int idx)      const = 0;
    virtual std::array<int, 4>    neighbors(int idx)  const = 0; // ±x, ±y; -1 if boundary
};

class ICartesianGrid3D {
public:
    virtual ~ICartesianGrid3D() = default;

    virtual int                   num_dofs()          const = 0;
    virtual std::array<double, 3> coord(int idx)      const = 0;
    virtual std::array<int, 6>    neighbors(int idx)  const = 0; // ±x, ±y, ±z; -1 if boundary
};

} // namespace kfbim
