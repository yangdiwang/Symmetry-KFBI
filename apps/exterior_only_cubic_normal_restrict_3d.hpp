#pragma once

#include "native_nurbs_surface_3d.hpp"
#include "src/geometry/grid_pair_3d.hpp"
#include "src/grid/cartesian_grid_3d.hpp"

#include <Eigen/Dense>

#include <utility>
#include <vector>

namespace kfbim::app3d {

struct ExteriorOnlyCubicNormalStencil3D {
    std::vector<int> grid_ids;
    Eigen::VectorXd weights;
    double condition = 0.0;
};

[[nodiscard]] ExteriorOnlyCubicNormalStencil3D
build_exterior_only_cubic_normal_stencil_3d(
    const Eigen::Vector3d& center,
    const Eigen::Vector3d& tangent1,
    const Eigen::Vector3d& tangent2,
    const Eigen::Vector3d& normal,
    double h,
    const std::vector<std::pair<int, Eigen::Vector3d>>& exterior_candidates,
    int sample_count = 48);

class ExteriorOnlyCubicNormalRestrict3D {
public:
    ExteriorOnlyCubicNormalRestrict3D(
        const CartesianGrid3D& grid,
        const GridPair3D& grid_pair,
        const SurfaceDofCloud3D& cloud,
        int sample_count = 48,
        int maximum_index_radius = 4);

    [[nodiscard]] Eigen::VectorXd apply(
        const Eigen::VectorXd& potential) const;

    [[nodiscard]]
    const std::vector<ExteriorOnlyCubicNormalStencil3D>& stencils() const
        noexcept;

private:
    int grid_dof_count_ = 0;
    std::vector<ExteriorOnlyCubicNormalStencil3D> stencils_;
};

} // namespace kfbim::app3d
