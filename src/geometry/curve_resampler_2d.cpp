#include "curve_resampler_2d.hpp"
#include "p2_curve_2d.hpp"
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace kfbim {

namespace {

double cross_2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b)
{
    return a.x() * b.y() - a.y() * b.x();
}

int arc_length_map_sample_count()
{
    // Retain the historical value by default.  The opt-in override is useful
    // for diagnosing whether the piecewise-linear inverse arclength map is a
    // resolution floor for very short Cauchy collocation offsets.
    const char* raw =
        std::getenv("KFBIM_CURVE_ARC_LENGTH_MAP_SAMPLES");
    if (raw == nullptr || std::string(raw).empty())
        return 20000;
    const long long samples = std::stoll(raw);
    if (samples < 2
        || samples > static_cast<long long>(
            std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "KFBIM_CURVE_ARC_LENGTH_MAP_SAMPLES must be in [2, INT_MAX]");
    }
    return static_cast<int>(samples);
}

double interpolate_monotone(
    const std::vector<double>& abscissa,
    const std::vector<double>& ordinate,
    double query)
{
    if (abscissa.size() != ordinate.size() || abscissa.empty()) {
        throw std::invalid_argument(
            "CurveResampler2D: invalid monotone interpolation data.");
    }
    if (query <= abscissa.front())
        return ordinate.front();
    if (query >= abscissa.back())
        return ordinate.back();

    const auto it =
        std::lower_bound(abscissa.begin(), abscissa.end(), query);
    const std::size_t hi =
        static_cast<std::size_t>(std::distance(abscissa.begin(), it));
    const std::size_t lo = hi - 1;
    const double width = abscissa[hi] - abscissa[lo];
    if (!(width > 0.0)) {
        throw std::runtime_error(
            "CurveResampler2D: non-increasing interpolation abscissa.");
    }
    const double fraction = (query - abscissa[lo]) / width;
    return ordinate[lo] + fraction * (ordinate[hi] - ordinate[lo]);
}

Eigen::Vector2d unit_tangent(const ICurve2D& curve, double t)
{
    const Eigen::Vector2d derivative = curve.deriv(t);
    const double speed = derivative.norm();
    if (!(speed > 64.0 * std::numeric_limits<double>::epsilon())) {
        throw std::invalid_argument(
            "CurveResampler2D: curve derivative must be nonzero.");
    }
    return derivative / speed;
}

class ArcLengthCurvePanelGeometry2D final : public IPanelGeometry2D {
public:
    ArcLengthCurvePanelGeometry2D(
        std::shared_ptr<const ICurve2D> curve,
        std::vector<double> arc_length_values,
        std::vector<double> parameter_values,
        std::vector<double> panel_edges)
        : curve_(std::move(curve))
        , arc_length_values_(std::move(arc_length_values))
        , parameter_values_(std::move(parameter_values))
        , panel_edges_(std::move(panel_edges))
    {
        if (!curve_ || panel_edges_.size() < 2
            || arc_length_values_.size() != parameter_values_.size()
            || arc_length_values_.empty()) {
            throw std::invalid_argument(
                "CurveResampler2D: invalid retained panel geometry");
        }
    }

    int num_panels() const override
    {
        return static_cast<int>(panel_edges_.size()) - 1;
    }

    Eigen::Vector2d point(int panel, double local_s) const override
    {
        return curve_->eval(parameter(panel, local_s));
    }

    Eigen::Vector2d tangent(int panel, double local_s) const override
    {
        const double t = parameter(panel, local_s);
        return half_panel_length(panel) * unit_tangent(*curve_, t);
    }

    Eigen::Vector2d second_derivative(
        int panel,
        double local_s) const override
    {
        const double t = parameter(panel, local_s);
        const Eigen::Vector2d first = curve_->deriv(t);
        const double speed = first.norm();
        if (!(speed > 64.0 * std::numeric_limits<double>::epsilon())) {
            throw std::invalid_argument(
                "CurveResampler2D: curve derivative must be nonzero");
        }

        const Eigen::Vector2d second = curve_->second_deriv(t);
        const double speed2 = speed * speed;
        const Eigen::Vector2d tangent_arc_derivative =
            second / speed2
            - first * (first.dot(second) / (speed2 * speed2));
        const double half_length = half_panel_length(panel);
        return half_length * half_length * tangent_arc_derivative;
    }

    Eigen::Vector2d normal(int panel, double local_s) const override
    {
        const Eigen::Vector2d unit =
            unit_tangent(*curve_, parameter(panel, local_s));
        return {unit[1], -unit[0]};
    }

private:
    void require_panel(int panel) const
    {
        if (panel < 0 || panel >= num_panels()) {
            throw std::out_of_range(
                "CurveResampler2D: panel geometry index is out of range");
        }
    }

    double half_panel_length(int panel) const
    {
        require_panel(panel);
        return 0.5
             * (panel_edges_[static_cast<std::size_t>(panel + 1)]
                - panel_edges_[static_cast<std::size_t>(panel)]);
    }

    double parameter(int panel, double local_s) const
    {
        require_panel(panel);
        if (!std::isfinite(local_s)) {
            throw std::invalid_argument(
                "CurveResampler2D: panel parameter must be finite");
        }
        const double left =
            panel_edges_[static_cast<std::size_t>(panel)];
        const double right =
            panel_edges_[static_cast<std::size_t>(panel + 1)];
        const double arc_length =
            0.5 * ((1.0 - local_s) * left
                   + (1.0 + local_s) * right);
        return interpolate_monotone(
            arc_length_values_, parameter_values_, arc_length);
    }

    std::shared_ptr<const ICurve2D> curve_;
    std::vector<double> arc_length_values_;
    std::vector<double> parameter_values_;
    std::vector<double> panel_edges_;
};

} // namespace

CurveResampler2D::ArcLengthMap CurveResampler2D::build_arc_length_map(
    const ICurve2D& curve,
    int num_samples)
{
    if (num_samples < 2) {
        throw std::invalid_argument(
            "CurveResampler2D: at least two map samples are required.");
    }
    ArcLengthMap map;
    map.t_vals.reserve(num_samples);
    map.s_vals.reserve(num_samples);
    map.absolute_turn_vals.reserve(num_samples);

    double t_min = curve.t_min();
    double t_max = curve.t_max();
    double dt = (t_max - t_min) / (num_samples - 1);
    if (!(dt > 0.0)) {
        throw std::invalid_argument(
            "CurveResampler2D: curve parameter interval must be positive.");
    }

    map.t_vals.push_back(t_min);
    map.s_vals.push_back(0.0);
    map.absolute_turn_vals.push_back(0.0);

    double current_s = 0.0;
    double current_absolute_turn = 0.0;
    Eigen::Vector2d previous_tangent = unit_tangent(curve, t_min);
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

        const Eigen::Vector2d tangent = unit_tangent(curve, t1);
        current_absolute_turn += std::abs(std::atan2(
            cross_2d(previous_tangent, tangent),
            previous_tangent.dot(tangent)));
        previous_tangent = tangent;

        map.t_vals.push_back(t1);
        map.s_vals.push_back(current_s);
        map.absolute_turn_vals.push_back(current_absolute_turn);
    }
    map.total_length = current_s;
    map.total_absolute_turn = current_absolute_turn;
    
    // Ensure exact bounds
    map.t_vals.back() = t_max;

    return map;
}

double CurveResampler2D::ArcLengthMap::get_t(double s) const {
    return interpolate_monotone(s_vals, t_vals, s);
}

Interface2D CurveResampler2D::discretize(const ICurve2D& curve,
                                          double h,
                                          double target_L_h_ratio)
{
    return discretize_quadratic_lagrange(curve, h, target_L_h_ratio);
}

Interface2D CurveResampler2D::discretize(
    std::shared_ptr<const ICurve2D> curve,
    double h,
    double target_L_h_ratio)
{
    return discretize_quadratic_lagrange(
        std::move(curve), h, target_L_h_ratio);
}

Interface2D CurveResampler2D::discretize_quadratic_lagrange(
    const ICurve2D& curve,
    double h,
    double target_L_h_ratio)
{
    return discretize_quadratic_lagrange_impl(
        curve, nullptr, h, target_L_h_ratio);
}

Interface2D CurveResampler2D::discretize_quadratic_lagrange(
    std::shared_ptr<const ICurve2D> curve,
    double h,
    double target_L_h_ratio)
{
    if (!curve) {
        throw std::invalid_argument(
            "CurveResampler2D: retained curve must be non-null");
    }
    return discretize_quadratic_lagrange_impl(
        *curve, curve, h, target_L_h_ratio);
}

Interface2D CurveResampler2D::discretize_quadratic_lagrange_impl(
    const ICurve2D& curve,
    std::shared_ptr<const ICurve2D> curve_owner,
    double h,
    double target_L_h_ratio)
{
    if (!(h > 0.0) || !std::isfinite(h)) {
        throw std::invalid_argument("CurveResampler2D: h must be positive.");
    }
    if (!(target_L_h_ratio > 0.0)
        || !std::isfinite(target_L_h_ratio)) {
        throw std::invalid_argument("CurveResampler2D: target_L_h_ratio must be positive.");
    }

    ArcLengthMap map =
        build_arc_length_map(curve, arc_length_map_sample_count());
    double S = map.total_length;

    int num_panels = std::max(2, static_cast<int>(std::round(S / (target_L_h_ratio * h))));
    std::vector<double> panel_edges(
        static_cast<std::size_t>(num_panels + 1));
    for (int p = 0; p <= num_panels; ++p) {
        panel_edges[static_cast<std::size_t>(p)] =
            S * static_cast<double>(p) / static_cast<double>(num_panels);
    }
    return build_quadratic_lagrange_interface(
        curve, map, panel_edges, std::move(curve_owner));
}

Interface2D
CurveResampler2D::discretize_quadratic_lagrange_curvature_adaptive(
    const ICurve2D& curve,
    double h,
    double target_L_h_ratio,
    double max_panel_turn,
    CurvaturePanelMonitor2D monitor_combination)
{
    return discretize_quadratic_lagrange_curvature_adaptive_impl(
        curve,
        nullptr,
        h,
        target_L_h_ratio,
        max_panel_turn,
        monitor_combination);
}

Interface2D
CurveResampler2D::discretize_quadratic_lagrange_curvature_adaptive(
    std::shared_ptr<const ICurve2D> curve,
    double h,
    double target_L_h_ratio,
    double max_panel_turn,
    CurvaturePanelMonitor2D monitor_combination)
{
    if (!curve) {
        throw std::invalid_argument(
            "CurveResampler2D: retained curve must be non-null");
    }
    return discretize_quadratic_lagrange_curvature_adaptive_impl(
        *curve,
        curve,
        h,
        target_L_h_ratio,
        max_panel_turn,
        monitor_combination);
}

Interface2D
CurveResampler2D::discretize_quadratic_lagrange_curvature_adaptive_impl(
    const ICurve2D& curve,
    std::shared_ptr<const ICurve2D> curve_owner,
    double h,
    double target_L_h_ratio,
    double max_panel_turn,
    CurvaturePanelMonitor2D monitor_combination)
{
    if (!(h > 0.0) || !std::isfinite(h)) {
        throw std::invalid_argument(
            "CurveResampler2D: h must be positive.");
    }
    if (!(target_L_h_ratio > 0.0)
        || !std::isfinite(target_L_h_ratio)) {
        throw std::invalid_argument(
            "CurveResampler2D: target_L_h_ratio must be positive.");
    }
    if (!(max_panel_turn > 0.0)
        || !std::isfinite(max_panel_turn)) {
        throw std::invalid_argument(
            "CurveResampler2D: max_panel_turn must be positive.");
    }

    const ArcLengthMap map =
        build_arc_length_map(curve, arc_length_map_sample_count());
    std::vector<double> monitor(map.s_vals.size(), 0.0);
    for (std::size_t i = 1; i < monitor.size(); ++i) {
        const double length_increment =
            (map.s_vals[i] - map.s_vals[i - 1])
            / (target_L_h_ratio * h);
        const double turn_increment =
            (map.absolute_turn_vals[i]
             - map.absolute_turn_vals[i - 1])
            / max_panel_turn;
        double monitor_increment = 0.0;
        switch (monitor_combination) {
        case CurvaturePanelMonitor2D::Maximum:
            monitor_increment =
                std::max(length_increment, turn_increment);
            break;
        case CurvaturePanelMonitor2D::Additive:
            monitor_increment =
                length_increment + turn_increment;
            break;
        }
        monitor[i] = monitor[i - 1] + monitor_increment;
    }
    const double total_monitor = monitor.back();
    const int num_panels = std::max(
        2, static_cast<int>(std::ceil(total_monitor)));
    std::vector<double> panel_edges(
        static_cast<std::size_t>(num_panels + 1));
    for (int p = 0; p <= num_panels; ++p) {
        const double target_monitor =
            total_monitor * static_cast<double>(p)
            / static_cast<double>(num_panels);
        panel_edges[static_cast<std::size_t>(p)] =
            interpolate_monotone(monitor, map.s_vals, target_monitor);
    }
    panel_edges.front() = 0.0;
    panel_edges.back() = map.total_length;
    return build_quadratic_lagrange_interface(
        curve, map, panel_edges, std::move(curve_owner));
}

Interface2D CurveResampler2D::build_quadratic_lagrange_interface(
    const ICurve2D& curve,
    const ArcLengthMap& map,
    const std::vector<double>& panel_edges,
    std::shared_ptr<const ICurve2D> curve_owner)
{
    if (panel_edges.size() < 3) {
        throw std::invalid_argument(
            "CurveResampler2D: at least two P2 panels are required.");
    }
    const int num_panels =
        static_cast<int>(panel_edges.size()) - 1;
    if (std::abs(panel_edges.front()) > 1.0e-14
        || std::abs(panel_edges.back() - map.total_length) > 1.0e-12) {
        throw std::invalid_argument(
            "CurveResampler2D: panel edges must span the curve.");
    }

    const int k = 3;
    const int Nq = 2 * num_panels;

    Eigen::MatrixX2d pts(Nq, 2);
    Eigen::MatrixX2d nml(Nq, 2);
    Eigen::VectorXd  wts = Eigen::VectorXd::Zero(Nq);
    Eigen::MatrixXi  panel_point_indices(num_panels, k);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(num_panels);

    static const double kP2_w[3] = {1.0/3.0, 4.0/3.0, 1.0/3.0};

    for (int p = 0; p < num_panels; ++p) {
        const double s_q = panel_edges[static_cast<std::size_t>(p)];
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
        const double s_left =
            panel_edges[static_cast<std::size_t>(p)];
        const double s_right =
            panel_edges[static_cast<std::size_t>(p + 1)];
        if (!(s_right > s_left)) {
            throw std::runtime_error(
                "CurveResampler2D: adaptive panel edges must increase.");
        }
        const double s_mid = 0.5 * (s_left + s_right);
        const double half_L = 0.5 * (s_right - s_left);

        const int q_left = p;
        const int q_mid = num_panels + p;
        const int q_right = (p + 1) % num_panels;
        panel_point_indices(p, 0) = q_left;
        panel_point_indices(p, 1) = q_mid;
        panel_point_indices(p, 2) = q_right;

        for (int i = 0; i < k; ++i) {
            const double s_q =
                s_mid + half_L * geometry2d::kP2NodeS[i];
            const double t_q = map.get_t(s_q);
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

    std::shared_ptr<const IPanelGeometry2D> panel_geometry;
    if (curve_owner) {
        panel_geometry = std::make_shared<ArcLengthCurvePanelGeometry2D>(
            std::move(curve_owner),
            map.s_vals,
            map.t_vals,
            panel_edges);
    }

    return Interface2D(
        std::move(pts),
        std::move(nml),
        std::move(wts),
        k,
        std::move(panel_point_indices),
        std::move(comp),
        PanelNodeLayout2D::QuadraticLagrange,
        std::move(panel_geometry));
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

    ArcLengthMap map =
        build_arc_length_map(curve, arc_length_map_sample_count());
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
