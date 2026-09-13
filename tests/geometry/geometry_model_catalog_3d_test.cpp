#include "src/geometry/models3d/catalog_3d.hpp"

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace kfbim::geometry3d;

namespace {
void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
}

int main()
{
    try {
        std::set<std::string> ids;
        for (const auto& entry : available_geometry_models_3d()) {
            require(!entry.id.empty() && !entry.description.empty(), "model metadata is incomplete");
            require(ids.insert(entry.id).second, "model ID must be unique");
        }
        struct Expected { const char* id; int patches; };
        // Bilinear and bicubic NURBS prisms have distinct patch layouts.
        for (const auto& c : {Expected{"native/torus",16},
                              Expected{"native/u_prism",18},
                              Expected{"bicubic_nurbs/u_prism",22},
                              Expected{"bicubic_nurbs/l_prism",14},
                              Expected{"bicubic_nurbs/box",6},
                              Expected{"bicubic_nurbs/solid_cylinder",14}}) {
            require(ids.count(c.id)==1, "known geometry is missing from catalog");
            const auto model=make_geometry_model_3d(c.id);
            require(model.num_patches()==c.patches, "wrong geometry profile or patch layout");
            const auto d=model.validate_closed();
            require(d.uncovered_interval_count==0 && d.multiply_covered_interval_count==0 &&
                    d.position_mismatch_count==0 && d.orientation_mismatch_count==0 &&
                    d.g1_normal_mismatch_count==0, "catalog geometry is not closed and oriented");
        }
        for (const char* id : {"industrial/sleeve","industrial/u_bracket",
                               "industrial/flange","industrial/impeller"})
            require(ids.count(id)==1, "industrial geometry is missing from catalog");
        const auto solid=make_geometry_model_3d("bicubic_nurbs/solid_cylinder");
        const auto p=solid.patch(0).evaluate(0.0,0.0);
        require(std::abs(p.head<2>().norm()-.54)<1e-13, "bicubic NURBS solid cylinder radius changed");
        bool rejected=false;
        try { (void)make_geometry_model_3d("unknown/model"); }
        catch (const std::invalid_argument&) { rejected=true; }
        require(rejected, "unknown geometry must be rejected");
        std::cout << "geometry model catalog: PASS\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
