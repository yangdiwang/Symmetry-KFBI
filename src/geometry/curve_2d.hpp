#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>

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

    // Evaluate r''(t).  Smooth analytic curves can override this.  The
    // default keeps existing curve implementations source-compatible and
    // uses a centered (or endpoint one-sided) difference of r'(t).
    virtual Eigen::Vector2d second_deriv(double t) const
    {
        const double lo = t_min();
        const double hi = t_max();
        const double span = hi - lo;
        const double dt = std::max(
            64.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, std::abs(t)),
            1.0e-5 * std::max(1.0, std::abs(span)));
        if (t - dt >= lo && t + dt <= hi)
            return (deriv(t + dt) - deriv(t - dt)) / (2.0 * dt);
        if (t + 2.0 * dt <= hi) {
            return (-3.0 * deriv(t)
                    + 4.0 * deriv(t + dt)
                    - deriv(t + 2.0 * dt))
                 / (2.0 * dt);
        }
        if (t - 2.0 * dt >= lo) {
            return (3.0 * deriv(t)
                    - 4.0 * deriv(t - dt)
                    + deriv(t - 2.0 * dt))
                 / (2.0 * dt);
        }
        return Eigen::Vector2d::Zero();
    }

    // Parameter bounds
    virtual double t_min() const = 0;
    virtual double t_max() const = 0;
};

} // namespace kfbim
