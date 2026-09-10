#include "src/geometry/industrial_nurbs_models_3d.hpp"
#include "examples/industrial_nurbs/diagnostics.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace kfbim::geometry3d;

void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

void check_part_features(const IndustrialNurbsModel3D& model)
{
    // Check observable part features, independently of face labels/counts.
    double low_z = 1e10, high_z = -1e10, upper_radius = 0, shaft_radius = 0;
    double hub_top = -1e10, rim_top = -1e10;
    for (const auto& patch : model.patches) {
        for (int i = 0; i <= 24; ++i) for (int j = 0; j <= 24; ++j) {
            const auto p = patch.evaluate(i / 24.0, j / 24.0);
            const double radius = p.head<2>().norm();
            low_z = std::min(low_z, p.z()); high_z = std::max(high_z, p.z());
            if (p.z() > 0.38) upper_radius = std::max(upper_radius, radius);
            if (p.z() > 0.02 && p.z() < 0.20) shaft_radius = std::max(shaft_radius, radius);
            if (radius > 0.24 && radius < 0.38) hub_top = std::max(hub_top, p.z());
            if (radius > 0.52 && radius < 0.78) rim_top = std::max(rim_top, p.z());
        }
    }
    if (model.name == "sleeve") {
        require(upper_radius > shaft_radius + 0.12, "sleeve needs a distinct raised shoulder/flange");
        bool curved_profile = false;
        for (const auto& p : model.patches)
            curved_profile = curved_profile || (p.basis_u().degree() >= 2 && p.basis_v().degree() >= 2);
        require(curved_profile, "sleeve needs curved meridian fillets");
    } else if (model.name == "u_bracket") {
        require(model.expected_genus == 2, "bracket needs two ear through-holes");
        require(high_z - low_z > 1.0, "bracket must have upright ears");
    } else if (model.name == "flange") {
        require(model.expected_genus == 9, "flange needs central bore and eight bolt holes");
        require(hub_top > rim_top + 0.10, "flange needs a raised hub/sealing face");
    } else if (model.name == "impeller") {
        require(high_z - low_z > 0.40, "impeller needs a raised hub and upright vanes");
        require(hub_top > rim_top + 0.015, "impeller needs a hub and varying-height blade profile");
        int curved_flanks = 0;
        for (const auto& p : model.patches) for (bool transpose : {false, true}) {
            auto point = [&](double t, double height) {
                return transpose ? p.evaluate(height, t) : p.evaluate(t, height);
            };
            const auto a = point(0,0), b = point(1,0), at = point(0,1), bt = point(1,1);
            if ((a-at).head<2>().norm() > 1e-10 || (b-bt).head<2>().norm() > 1e-10) continue;
            if (std::abs(a.z()) > 1e-10 || std::abs(b.z()) > 1e-10 || at.z() < 0.1 || bt.z() < 0.1) continue;
            if (std::abs(a.head<2>().norm() - b.head<2>().norm()) < 0.4) continue;
            const Eigen::Vector2d chord = (b-a).head<2>();
            const Eigen::Vector2d offset = (point(0.5,0)-a).head<2>();
            const double bend = std::abs(chord.x()*offset.y()-chord.y()*offset.x()) / chord.norm();
            if (bend > 0.03) ++curved_flanks;
        }
        require(curved_flanks == 16, "impeller needs two curved full-height flanks for each of eight vanes");
    }
}

void check_model(const IndustrialNurbsModel3D& model)
{
    require(!model.patches.empty(), "factory must return NURBS patches");
    require(model.patch_names.size() == model.patches.size(), "patch names missing");
    const auto topology = model.geometry_model().validate_closed();
    require(topology.component_count == 1, "expected one boundary component");
    require(model.connections.size() == 2 * model.patches.size(), "unpaired patch edges");
    for (const auto& patch : model.patches) {
        for (const auto& row : patch.weights())
            for (double weight : row) require(weight > 0, "nonpositive weight");
        for (int i = 0; i <= 32; ++i) {
            for (int j = 0; j <= 32; ++j) {
                const auto d = patch.evaluate_with_derivatives(i / 32.0, j / 32.0);
                require(d.point.allFinite(), "nonfinite point");
                require(d.du.cross(d.dv).norm() > 1e-8, "degenerate sampled Jacobian");
            }
        }
    }
    // Missing and inward-facing patches must not be silently accepted.
    auto connections = model.connections;
    connections.pop_back();
    bool missing_rejected = false;
    try {
        NurbsSurfaceModel3D broken(model.patches, std::vector<int>(model.patches.size(), 0), connections);
        broken.validate_closed();
    } catch (const std::exception&) { missing_rejected = true; }
    require(missing_rejected, "missing seam was accepted");
    auto reversed = model.patches;
    auto net = reversed.front().control_net();
    auto weights = reversed.front().weights();
    std::reverse(net.begin(), net.end());
    std::reverse(weights.begin(), weights.end());
    reversed.front() = NurbsSurfacePatch3D(reversed.front().basis_u(), reversed.front().basis_v(), net, weights);
    bool reversed_rejected = false;
    try {
        finalize_industrial_model("reversed", "negative fixture", reversed, model.patch_names, 0, model.expected_genus);
    } catch (const std::exception&) { reversed_rejected = true; }
    require(reversed_rejected, "reversed face was accepted");
    const auto d = industrial_example::inspect(model);
    std::cout << model.name << ": " << model.patches.size() << " native NURBS patches OK; Euler="
              << d.euler_characteristic << ", volume=" << d.volume << ", min_J=" << d.min_jacobian << '\n';
}

int main()
{
    try {
        for (const auto& model : {make_industrial_sleeve_3d(), make_industrial_u_bracket_3d(),
                                  make_industrial_flange_3d(), make_industrial_impeller_3d()}) {
            check_part_features(model);
            check_model(model);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
