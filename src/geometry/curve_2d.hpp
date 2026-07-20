#pragma once

#include <Eigen/Dense>

namespace kfbim {

// ---------------------------------------------------------------------------
// ICurve2D
//
// Represents a 2D parametric curve for interface generation.
// The curve is parameterized by t in [t_min, t_max].
// ---------------------------------------------------------------------------
class ICurve2D {
public:
    virtual ~ICurve2D() = default;

    // Evaluate the curve position r(t)
    virtual Eigen::Vector2d eval(double t) const = 0;

    // Evaluate the curve derivative r'(t)
    virtual Eigen::Vector2d deriv(double t) const = 0;

    // Parameter bounds
    virtual double t_min() const = 0;
    virtual double t_max() const = 0;
};

} // namespace kfbim
