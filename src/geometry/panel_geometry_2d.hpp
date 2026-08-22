#pragma once

#include <Eigen/Dense>

namespace kfbim {

// Optional smooth panel geometry retained by Interface2D.
//
// The local parameter is the same s in [-1, 1] used by a quadratic
// Lagrange panel.  The tangent and second derivative are derivatives with
// respect to that local parameter.  Interfaces without a provider continue
// to use their positional P2 geometry.
class IPanelGeometry2D {
public:
    virtual ~IPanelGeometry2D() = default;

    virtual int num_panels() const = 0;
    virtual Eigen::Vector2d point(int panel, double local_s) const = 0;
    virtual Eigen::Vector2d tangent(int panel, double local_s) const = 0;
    virtual Eigen::Vector2d second_derivative(
        int panel,
        double local_s) const = 0;
    virtual Eigen::Vector2d normal(int panel, double local_s) const = 0;
};

} // namespace kfbim
