#pragma once

#include "rational_bezier_element_3d.hpp"

#include <array>
#include <vector>

namespace kfbim::geometry3d {

// One workspace per concurrent caller. Common degrees use only inline storage;
// larger degrees retain their allocation for subsequent evaluations.
class NurbsPolarEvaluationWorkspace3D {
public:
    static constexpr int fixed_degree = 3;

private:
    friend class NurbsPatchPolarEvaluator3D;
    static constexpr std::size_t fixed_size =
        (fixed_degree + 1) * (fixed_degree + 1) + 2 * (fixed_degree + 1);
    std::array<Eigen::Vector4d, fixed_size> fixed_;
    std::vector<Eigen::Vector4d> dynamic_;
};

// Immutable base-span data: no references to patch or intersector storage.
// The patch argument at evaluation must describe the same patch supplied to
// the constructor; it supplies original span semantics and the reference fallback.
class NurbsPatchPolarEvaluator3D {
public:
    NurbsPatchPolarEvaluator3D(
        const NurbsSurfacePatch3D& patch,
        std::vector<RationalBezierElement3D> base_elements);

    NurbsSurfaceDerivatives3D evaluate_with_derivatives(
        const NurbsSurfacePatch3D& patch, double u, double v,
        NurbsPolarEvaluationWorkspace3D& workspace) const;

private:
    std::vector<RationalBezierElement3D> base_elements_;
    std::vector<int> span_elements_;
    int span_count_u_ = 0;
    int span_count_v_ = 0;
};

} // namespace kfbim::geometry3d
