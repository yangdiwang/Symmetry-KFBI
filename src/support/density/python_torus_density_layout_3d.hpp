#pragma once

#include "src/support/density/native_nurbs_density_space_3d.hpp"

namespace kfbim::app3d {

// Exact representation of the reference's anisotropic cubic space in the
// native isotropic C0 carrier. Knot insertion changes coordinates, not the
// trial functions; no sample fitting or change to the geometry is involved.
struct PythonTorusDensityLayout3D {
    int spans_u = 0, spans_v = 0;
    int reference_raw_dofs = 0;
    Eigen::MatrixXd Z;
    Eigen::VectorXd particular;
    std::vector<NativeDensityGaussPoint3D> traces;
    Eigen::VectorXd weights;
    Eigen::RowVectorXd mean_row;
    double c0_embedding_error = 0.0;
    double mean_nullspace_error = 0.0;
};

[[nodiscard]] Eigen::MatrixXd cubic_midpoint_insertion_matrix_3d(int spans);

// Only for the reference 4-by-4 periodic torus atlas, pid=major*4+minor,
// u=major, v=minor. The carrier must have spans_u+3 coefficients per axis,
// full-edge C0 identification and no subsequent reduction (TopologyBase).
[[nodiscard]] PythonTorusDensityLayout3D build_python_torus_density_layout_3d(
    const NativeNurbsDensitySpace3D& carrier, int spans_u, int spans_v,
    bool mean_free, int trace_order = 4, int mean_order = 8);

} // namespace kfbim::app3d
