#pragma once

#include "src/geometry/nurbs_basis.hpp"

#include <array>

namespace kfbim::geometry {

// Analytic B-spline derivatives on the same one-sided knot span selected by
// NurbsBasis1D.  In particular the right endpoint uses the final left span;
// no differencing across a knot or geometry samples are involved.
inline std::vector<double> nonzero_basis_third_derivatives(
    const NurbsBasis1D& basis, double parameter)
{
    const int span = basis.find_span(parameter);
    const int degree = basis.degree();
    const auto& knots = basis.knots();
    const auto indices = basis.active_basis_indices(parameter);
    std::vector<double> result(indices.size(), 0.0);
    if (degree < 3)
        return result;
    const double x = std::clamp(parameter, basis.domain_start(), basis.domain_end());
    auto evaluate = [&](auto&& self, int i, int p, int d) -> double {
        if (d > p || i < 0 || i + p + 1 >= static_cast<int>(knots.size()))
            return 0.0;
        if (p == 0)
            return i == span ? 1.0 : 0.0;
        const double left = knots[static_cast<std::size_t>(i + p)] - knots[static_cast<std::size_t>(i)];
        const double right = knots[static_cast<std::size_t>(i + p + 1)] - knots[static_cast<std::size_t>(i + 1)];
        if (d > 0) {
            return (left == 0.0 ? 0.0 : p * self(self, i, p - 1, d - 1) / left)
                - (right == 0.0 ? 0.0 : p * self(self, i + 1, p - 1, d - 1) / right);
        }
        return (left == 0.0 ? 0.0 : (x - knots[static_cast<std::size_t>(i)])
                      * self(self, i, p - 1, 0) / left)
            + (right == 0.0 ? 0.0 : (knots[static_cast<std::size_t>(i + p + 1)] - x)
                      * self(self, i + 1, p - 1, 0) / right);
    };
    for (std::size_t q = 0; q < indices.size(); ++q)
        result[q] = evaluate(evaluate, indices[q], degree, 3);
    return result;
}

} // namespace kfbim::geometry
