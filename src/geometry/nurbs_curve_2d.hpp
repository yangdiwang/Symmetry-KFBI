#pragma once

#include "curve_2d.hpp"
#include "nurbs_curve.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfbim::geometry2d {

class NurbsCurve2D final : public kfbim::ICurve2D {
public:
    using Vector = Eigen::Vector2d;
    using Curve = kfbim::geometry::NurbsCurve<2>;
    using ControlPointVector = typename Curve::ControlPointVector;
    using ProjectionResult = kfbim::geometry::NurbsProjectionResult2D;

    NurbsCurve2D(kfbim::geometry::NurbsBasis1D basis,
                 ControlPointVector            control_points,
                 std::vector<double>           weights)
        : curve_(std::move(basis), std::move(control_points), std::move(weights))
    {
    }

    explicit NurbsCurve2D(Curve curve)
        : curve_(std::move(curve))
    {
    }

    [[nodiscard]] const Curve& curve() const noexcept { return curve_; }

    [[nodiscard]] const kfbim::geometry::NurbsBasis1D& basis() const noexcept
    {
        return curve_.basis();
    }

    [[nodiscard]] const ControlPointVector& control_points() const noexcept
    {
        return curve_.control_points();
    }

    [[nodiscard]] const std::vector<double>& weights() const noexcept
    {
        return curve_.weights();
    }

    [[nodiscard]] double domain_start() const noexcept { return curve_.domain_start(); }
    [[nodiscard]] double domain_end() const noexcept { return curve_.domain_end(); }

    [[nodiscard]] Vector evaluate(double t) const { return curve_.evaluate(t); }
    [[nodiscard]] Vector derivative(double t) const { return curve_.derivative(t); }
    [[nodiscard]] Vector tangent(double t) const { return curve_.tangent(t); }

    [[nodiscard]] double rational_basis_value(int i, double t) const
    {
        return curve_.rational_basis_value(i, t);
    }

    [[nodiscard]] std::vector<int> active_basis_indices(double t) const
    {
        return curve_.active_basis_indices(t);
    }

    [[nodiscard]] ProjectionResult project(const Vector& x) const
    {
        return curve_.project(x);
    }

    Eigen::Vector2d eval(double t) const override { return evaluate(t); }
    Eigen::Vector2d deriv(double t) const override { return derivative(t); }
    double t_min() const override { return domain_start(); }
    double t_max() const override { return domain_end(); }

    [[nodiscard]] static NurbsCurve2D make_line_segment(const Vector& a,
                                                        const Vector& b)
    {
        return NurbsCurve2D(Curve::make_line_segment(a, b));
    }

    [[nodiscard]] static NurbsCurve2D make_quarter_circle(double radius)
    {
        kfbim::geometry::require_finite(radius, "radius");
        if (radius <= 0.0)
            throw std::invalid_argument("quarter circle radius must be positive");

        const double middle_weight = std::sqrt(2.0) / 2.0;
        ControlPointVector control_points;
        control_points.reserve(3);
        control_points.emplace_back(radius, 0.0);
        control_points.emplace_back(radius, radius);
        control_points.emplace_back(0.0, radius);
        return NurbsCurve2D(
            kfbim::geometry::NurbsBasis1D(2, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}),
            std::move(control_points),
            {1.0, middle_weight, 1.0});
    }

private:
    Curve curve_;
};

} // namespace kfbim::geometry2d
