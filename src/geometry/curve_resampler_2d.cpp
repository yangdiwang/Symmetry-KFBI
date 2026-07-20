#include "curve_resampler_2d.hpp"
#include "p2_curve_2d.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace kfbim {

CurveResampler2D::ArcLengthMap CurveResampler2D::build_arc_length_map(const ICurve2D& curve, int num_samples) {
    ArcLengthMap map;
    map.t_vals.reserve(num_samples);
    map.s_vals.reserve(num_samples);

    double t_min = curve.t_min();
    double t_max = curve.t_max();
    double dt = (t_max - t_min) / (num_samples - 1);

    map.t_vals.push_back(t_min);
    map.s_vals.push_back(0.0);

    double current_s = 0.0;
    for (int i = 1; i < num_samples; ++i) {
        double t0 = t_min + (i - 1) * dt;
        double t1 = t_min + i * dt;
        
        // Simpson's 1/3 rule for integrating |r'(t)|
        double tm = 0.5 * (t0 + t1);
        double v0 = curve.deriv(t0).norm();
        double vm = curve.deriv(tm).norm();
        double v1 = curve.deriv(t1).norm();
        
        double ds = (dt / 6.0) * (v0 + 4.0 * vm + v1);
        current_s += ds;

        map.t_vals.push_back(t1);
        map.s_vals.push_back(current_s);
    }
    map.total_length = current_s;
    
    // Ensure exact bounds
    map.t_vals.back() = t_max;

    return map;
}

double CurveResampler2D::ArcLengthMap::get_t(double s) const {
    if (s <= 0.0) return t_vals.front();
    if (s >= total_length) return t_vals.back();

    auto it = std::lower_bound(s_vals.begin(), s_vals.end(), s);
    int idx = static_cast<int>(std::distance(s_vals.begin(), it));

    if (idx == 0) return t_vals.front();

    double s0 = s_vals[idx - 1];
    double s1 = s_vals[idx];
    double t0 = t_vals[idx - 1];
    double t1 = t_vals[idx];

    double frac = (s - s0) / (s1 - s0);
    return t0 + frac * (t1 - t0);
}

Interface2D CurveResampler2D::discretize(const ICurve2D& curve,
                                          double h,
                                          double target_L_h_ratio)
{
    return discretize_quadratic_lagrange(curve, h, target_L_h_ratio);
}

Interface2D CurveResampler2D::discretize_quadratic_lagrange(
    const ICurve2D& curve,
    double h,
    double target_L_h_ratio)
{
    if (h <= 0.0) {
        throw std::invalid_argument("CurveResampler2D: h must be positive.");
    }
    if (target_L_h_ratio <= 0.0) {
        throw std::invalid_argument("CurveResampler2D: target_L_h_ratio must be positive.");
    }

    ArcLengthMap map = build_arc_length_map(curve, 20000);
    double S = map.total_length;

    int num_panels = std::max(2, static_cast<int>(std::round(S / (target_L_h_ratio * h))));
    double L = S / num_panels;

    const int k = 3;
    const int Nq = 2 * num_panels;

    Eigen::MatrixX2d pts(Nq, 2);
    Eigen::MatrixX2d nml(Nq, 2);
    Eigen::VectorXd  wts = Eigen::VectorXd::Zero(Nq);
    Eigen::MatrixXi  panel_point_indices(num_panels, k);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(num_panels);

    static const double kP2_w[3] = {1.0/3.0, 4.0/3.0, 1.0/3.0};

    for (int p = 0; p < num_panels; ++p) {
        const double s_q = p * L;
        const double t_q = map.get_t(s_q);

        Eigen::Vector2d r = curve.eval(t_q);
        Eigen::Vector2d drdt = curve.deriv(t_q);
        double speed = drdt.norm();

        pts(p, 0) = r.x();
        pts(p, 1) = r.y();
        nml(p, 0) =  drdt.y() / speed;
        nml(p, 1) = -drdt.x() / speed;
    }

    for (int p = 0; p < num_panels; ++p) {
        double s_mid = (p + 0.5) * L;
        double half_L = 0.5 * L;

        const int q_left = p;
        const int q_mid = num_panels + p;
        const int q_right = (p + 1) % num_panels;
        panel_point_indices(p, 0) = q_left;
        panel_point_indices(p, 1) = q_mid;
        panel_point_indices(p, 2) = q_right;

        for (int i = 0; i < k; ++i) {
            double s_q = s_mid + half_L * geometry2d::kP2NodeS[i];
            double t_q = map.get_t(s_q);
            const int q = panel_point_indices(p, i);

            Eigen::Vector2d r = curve.eval(t_q);
            Eigen::Vector2d drdt = curve.deriv(t_q);
            double speed = drdt.norm();

            if (i == 1) {
                pts(q, 0) = r.x();
                pts(q, 1) = r.y();
                nml(q, 0) =  drdt.y() / speed;
                nml(q, 1) = -drdt.x() / speed;
            }
            wts(q) += kP2_w[i] * half_L;
        }
    }

    return Interface2D(std::move(pts), std::move(nml), std::move(wts), k,
                       std::move(panel_point_indices), std::move(comp),
                       PanelNodeLayout2D::QuadraticLagrange);
}

Interface2D CurveResampler2D::discretize_chebyshev_lobatto(
    const ICurve2D& curve,
    double h,
    double target_L_h_ratio)
{
    return discretize_quadratic_lagrange(curve, h, target_L_h_ratio);
}

Interface2D CurveResampler2D::discretize_legacy_gauss(const ICurve2D& curve,
                                                      double h,
                                                      double target_L_h_ratio)
{
    if (h <= 0.0) {
        throw std::invalid_argument("CurveResampler2D: h must be positive.");
    }
    if (target_L_h_ratio <= 3.55) {
        throw std::invalid_argument("CurveResampler2D: target_L_h_ratio must be > 3.55 to guarantee arc_h_ratio > 0.8 for 3-point Gauss-Legendre panels.");
    }

    ArcLengthMap map = build_arc_length_map(curve, 20000); // Dense sampling
    double S = map.total_length;
    
    int num_panels = std::max(1, static_cast<int>(std::round(S / (target_L_h_ratio * h))));
    double L = S / num_panels;

    const int k = 3;
    const int Nq = k * num_panels;

    Eigen::MatrixX2d pts(Nq, 2);
    Eigen::MatrixX2d nml(Nq, 2);
    Eigen::VectorXd  wts(Nq);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(num_panels);

    static const double kGL_s[3] = {-0.7745966692414834, 0.0, +0.7745966692414834};
    static const double kGL_w[3] = {5.0/9.0, 8.0/9.0, 5.0/9.0};

    int q = 0;
    for (int p = 0; p < num_panels; ++p) {
        double s_mid = (p + 0.5) * L;
        double half_L = 0.5 * L;
        
        for (int i = 0; i < k; ++i) {
            double s_q = s_mid + half_L * kGL_s[i];
            double t_q = map.get_t(s_q);

            Eigen::Vector2d r = curve.eval(t_q);
            Eigen::Vector2d drdt = curve.deriv(t_q);
            double speed = drdt.norm();

            pts(q, 0) = r.x();
            pts(q, 1) = r.y();

            // Outward normal (assuming curve is parameterized counter-clockwise around the interior)
            nml(q, 0) =  drdt.y() / speed;
            nml(q, 1) = -drdt.x() / speed;

            // Integration weight in arc-length space is w_i * L / 2
            wts(q) = kGL_w[i] * half_L;

            ++q;
        }
    }

    return Interface2D(std::move(pts), std::move(nml), std::move(wts), k, std::move(comp),
                       PanelNodeLayout2D::LegacyGaussLegendre);
}

} // namespace kfbim
