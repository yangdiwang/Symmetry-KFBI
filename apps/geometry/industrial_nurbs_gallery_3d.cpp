#include "examples/industrial_nurbs/diagnostics.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace kfbim::geometry3d;

namespace {
void quote(std::ostream& out, const std::string& s)
{
    out << '"';
    for (char c : s) {
        if (c == '"' || c == '\\') out << '\\';
        if (c == '\n') out << "\\n"; else out << c;
    }
    out << '"';
}
void point(std::ostream& out, const Eigen::Vector3d& p)
{
    out << '[' << p.x() << ',' << p.y() << ',' << p.z() << ']';
}
void numbers(std::ostream& out, const std::vector<double>& values)
{
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
}
void write_model(std::ostream& out, const IndustrialNurbsModel3D& m, int samples)
{
    const auto d = industrial_example::inspect(m);
    std::cout << m.name << ": patches=" << d.patch_count << " Euler=" << d.euler_characteristic
              << " volume=" << d.volume << " min_J=" << d.min_jacobian
              << " max_gap=" << d.max_seam_gap << '\n';
    out << "{\"name\":"; quote(out, m.name);
    out << ",\"description\":"; quote(out, m.description);
    out << ",\"expected_genus\":" << m.expected_genus << ",\"expected_volume\":" << m.expected_volume;
    out << ",\"diagnostics\":{\"patch_count\":" << d.patch_count
        << ",\"connection_count\":" << d.connection_count
        << ",\"smooth_connection_count\":" << d.smooth_connection_count
        << ",\"connected_components\":" << d.connected_components
        << ",\"euler_characteristic\":" << d.euler_characteristic
        << ",\"min_jacobian\":" << d.min_jacobian << ",\"max_seam_gap\":" << d.max_seam_gap
        << ",\"area\":" << d.area << ",\"volume\":" << d.volume << ",\"normal_flux\":" << d.normal_flux
        << ",\"volume_quadrature_difference\":" << d.volume_quadrature_difference
        << ",\"relative_volume_error\":" << d.relative_volume_error << "},\"connections\":[";
    for (size_t i = 0; i < m.connections.size(); ++i) {
        if (i) out << ',';
        const auto& c = m.connections[i];
        out << "{\"first\":[" << c.first.patch << ',' << static_cast<int>(c.first.edge)
            << "],\"second\":[" << c.second.patch << ',' << static_cast<int>(c.second.edge)
            << "],\"reversed\":" << (c.reversed ? "true" : "false")
            << ",\"g1\":" << (c.g1 ? "true" : "false") << '}';
    }
    out << "],\"patches\":[";
    for (size_t index = 0; index < m.patches.size(); ++index) {
        if (index) out << ',';
        const auto& p = m.patches[index];
        out << "{\"name\":"; quote(out, m.patch_names[index]);
        out << ",\"degree_u\":" << p.basis_u().degree() << ",\"degree_v\":" << p.basis_v().degree();
        out << ",\"knots_u\":"; numbers(out, p.basis_u().knots());
        out << ",\"knots_v\":"; numbers(out, p.basis_v().knots());
        out << ",\"control_points\":[";
        for (size_t i = 0; i < p.control_net().size(); ++i) {
            if (i) out << ','; out << '[';
            for (size_t j = 0; j < p.control_net()[i].size(); ++j) {
                if (j) out << ','; point(out, p.control_net()[i][j]);
            }
            out << ']';
        }
        out << "],\"weights\":[";
        for (size_t i = 0; i < p.weights().size(); ++i) {
            if (i) out << ','; numbers(out, p.weights()[i]);
        }
        out << "],\"points\":[";
        for (int i = 0; i <= samples; ++i) {
            if (i) out << ','; out << '[';
            for (int j = 0; j <= samples; ++j) {
                if (j) out << ','; point(out, p.evaluate(i / double(samples), j / double(samples)));
            }
            out << ']';
        }
        out << "]}";
    }
    out << "]}";
}
}

int main(int argc, char** argv)
{
    try {
        std::filesystem::path output = "output/industrial_nurbs";
        int samples = 24;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--output" && i + 1 < argc) output = argv[++i];
            else if (option == "--samples" && i + 1 < argc) samples = std::stoi(argv[++i]);
            else throw std::invalid_argument("usage: industrial_nurbs_gallery_3d [--output DIR] [--samples 4..100]");
        }
        if (samples < 4 || samples > 100) throw std::invalid_argument("samples must be in [4,100]");
        const std::vector<IndustrialNurbsModel3D> models{make_industrial_sleeve_3d(), make_industrial_u_bracket_3d(),
            make_industrial_flange_3d(), make_industrial_impeller_3d()};
        // Complete geometry validation before touching an existing result file.
        std::ostringstream payload;
        payload << std::setprecision(17) << "{\"schema_version\":1,\"samples_per_direction\":" << samples << ",\"models\":[";
        for (size_t i = 0; i < models.size(); ++i) {
            if (i) payload << ','; write_model(payload, models[i], samples);
        }
        payload << "]}\n";
        std::filesystem::create_directories(output);
        std::ofstream file(output / "models.json");
        file << payload.str();
        file.close();
        if (!file) throw std::runtime_error("could not write models.json");
        std::cout << "Native geometry and sampled views written to " << (output / "models.json") << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
