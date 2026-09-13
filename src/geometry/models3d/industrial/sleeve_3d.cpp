#include "src/geometry/models3d/industrial/models_3d.hpp"

#include <array>
#include <cmath>
#include <utility>

namespace kfbim::geometry3d {
namespace {
struct MeridianSegment {
    std::string name;
    std::vector<Eigen::Vector2d> controls; // radius, height
    std::vector<double> weights;
};
}

IndustrialNurbsModel3D make_industrial_sleeve_3d()
{
    const double pi = std::acos(-1.0);
    std::vector<MeridianSegment> profile;
    double volume_integral = 0.0;
    auto line = [&](const std::string& name, Eigen::Vector2d a, Eigen::Vector2d b) {
        profile.push_back({name, {a, b}, {1, 1}});
        volume_integral += (b.y() - a.y())
            * (a.x() * a.x() + a.x() * b.x() + b.x() * b.x()) / 3.0;
    };
    auto arc = [&](const std::string& name, Eigen::Vector2d center, double radius,
                   double first, double last) {
        const double middle = (first + last) / 2, weight = std::cos((last - first) / 2);
        auto direction = [](double a) { return Eigen::Vector2d(std::cos(a), std::sin(a)); };
        profile.push_back({name, {center + radius * direction(first),
            center + radius / weight * direction(middle), center + radius * direction(last)}, {1, weight, 1}});
        // Exact independent meridian formula: V = pi * integral r^2 dz.
        auto primitive = [&](double a) {
            const double s = std::sin(a), c = center.x();
            return radius * c * c * s + c * radius * radius * (a + std::sin(2*a)/2)
                + radius * radius * radius * (s - s*s*s/3);
        };
        volume_integral += primitive(last) - primitive(first);
    };

    // Closed counterclockwise meridian: lead-in chamfer, retaining groove,
    // shoulder root fillet, rounded flange, bore entry radius and bore exit.
    line("lower_annular_face", {0.32,-0.69}, {0.49,-0.69});
    line("outer_lead_chamfer", {0.49,-0.69}, {0.54,-0.64});
    line("lower_barrel", {0.54,-0.64}, {0.54,-0.16});
    line("groove_lower_chamfer", {0.54,-0.16}, {0.505,-0.14});
    line("retaining_groove", {0.505,-0.14}, {0.505,-0.065});
    line("groove_upper_chamfer", {0.505,-0.065}, {0.54,-0.045});
    line("upper_barrel", {0.54,-0.045}, {0.54,0.28});
    arc("shoulder_root_radius", {0.60,0.28}, 0.06, pi, pi/2);
    line("flange_underside", {0.60,0.34}, {0.70,0.34});
    arc("flange_lower_radius", {0.70,0.38}, 0.04, -pi/2, 0);
    line("flange_rim", {0.74,0.38}, {0.74,0.60});
    arc("flange_upper_radius", {0.70,0.60}, 0.04, 0, pi/2);
    line("flange_face", {0.70,0.64}, {0.32,0.64});
    arc("bore_entry_radius", {0.32,0.60}, 0.04, pi/2, pi);
    line("shaft_bore", {0.28,0.60}, {0.28,-0.65});
    line("bore_exit_chamfer", {0.28,-0.65}, {0.32,-0.69});

    std::vector<NurbsSurfacePatch3D> patches;
    std::vector<std::string> names;
    for (const auto& segment : profile) for (int quadrant = 0; quadrant < 4; ++quadrant) {
        const double a = quadrant * pi/2, b = a + pi/2;
        const std::array<double,3> angles{{a, (a+b)/2, b}}, cw{{1, std::sqrt(0.5), 1}};
        std::vector<std::vector<Eigen::Vector3d>> net(3);
        std::vector<std::vector<double>> weights(3);
        for (int i = 0; i < 3; ++i) for (std::size_t j = 0; j < segment.controls.size(); ++j) {
            const auto p = segment.controls[j];
            net[i].push_back({p.x() * std::cos(angles[i]) / cw[i],
                              p.x() * std::sin(angles[i]) / cw[i], p.y()});
            weights[i].push_back(cw[i] * segment.weights[j]);
        }
        const int degree = static_cast<int>(segment.controls.size()) - 1;
        std::vector<double> knots(degree + 1, 0.0);
        knots.insert(knots.end(), degree + 1, 1.0);
        patches.emplace_back(geometry::NurbsBasis1D(2, {0,0,0,1,1,1}),
            geometry::NurbsBasis1D(degree, knots), std::move(net), std::move(weights));
        names.push_back(segment.name + "_" + std::to_string(quadrant));
    }
    return finalize_industrial_model("sleeve",
        "Flanged shaft bushing with retaining groove, lead-in chamfers, shoulder fillet, "
        "rounded flange edges and circular shaft bore; exact rational surfaces of revolution.",
        std::move(patches), std::move(names), pi * volume_integral, 1);
}
} // namespace kfbim::geometry3d
