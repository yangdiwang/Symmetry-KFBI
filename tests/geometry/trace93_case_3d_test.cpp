#include "src/support/geometry/trace93_case_3d.hpp"

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace kfbim::app3d;
namespace {
void require(bool condition,const char* message)
{if(!condition) throw std::runtime_error(message);}
}

int main()
{
    try {
        for(const std::string name:{"box","l","u","cylinder"}) {
            double reference_shift=0;
            // The package registers four rigid transforms; retain identity
            // as an independent reference for the mean in every geometry.
            for(const std::string transform:{"identity","translate","rotate",
                    "rotate_same_translate","rotate_translate"}) {
                const auto c=make_trace93_case_3d(name,transform);
                const std::size_t expected=name=="box"?6:name=="u"?22:14;
                require(c.surface.patches.size()==expected,"trace93 patch count");
                require(c.surface.geometric_connections.size()==2*expected,"trace93 complete manifold edge coverage");
                if(transform=="identity") reference_shift=c.boundary_mean_shift;
                else require(std::abs(c.boundary_mean_shift-reference_shift)<3e-14,"trace93 rigid mean invariance");
                std::set<int> sheets;
                for(std::size_t p=0;p<expected;++p) {
                    const auto& patch=c.surface.patches[p];sheets.insert(c.analysis_patches[p].sheet);
                    require(patch.basis_u().degree()==3 && patch.basis_v().degree()==3,"trace93 geometry not bicubic");
                    // Positive rational weights put the entire patch in the
                    // convex hull of its controls. This checks a bound for
                    // the complete geometry, not just the sampled points.
                    for(const auto& row:patch.weights()) for(double weight:row)
                        require(std::isfinite(weight) && weight>0.0,
                            "trace93 convex-hull bound requires positive finite weights");
                    for(const auto& row:patch.control_net()) for(const auto& point:row)
                        require(point.allFinite() && point.cwiseAbs().maxCoeff()<1.0,
                            "trace93 native control hull is not inside the [-1,1]^3 solve box");
                    for(double u:{0.,.17,.5,.83,1.}) for(double v:{0.,.19,.5,.81,1.}) {
                        const auto analysis=c.analysis_at(static_cast<int>(p),u,v,1);
                        const auto native=c.analysis_to_native_uv(static_cast<int>(p),u,v);
                        require((analysis.d[0][0]-patch.evaluate(native.x(),native.y())).norm()<1e-12,
                            "analysis/native physical geometry mismatch");
                        const auto back=c.world_to_analysis_uv(static_cast<int>(p),analysis.d[0][0],{u,v});
                        require((back-Eigen::Vector2d(u,v)).norm()<2e-10,"analysis inverse chart round trip");
                        const auto native_d=patch.evaluate_with_derivatives(native.x(),native.y());
                        require(analysis.d[1][0].cross(analysis.d[0][1]).normalized().dot(
                            native_d.du.cross(native_d.dv).normalized())>1-2e-12,"analysis/native orientation mismatch");
                    }
                    const double u=.31,v=.57,step=2e-5;
                    const auto d=c.analysis_at(static_cast<int>(p),u,v,4);
                    const auto du0=c.analysis_at(static_cast<int>(p),u-step,v,3);
                    const auto du1=c.analysis_at(static_cast<int>(p),u+step,v,3);
                    const auto dv0=c.analysis_at(static_cast<int>(p),u,v-step,3);
                    const auto dv1=c.analysis_at(static_cast<int>(p),u,v+step,3);
                    for(int i=0;i<4;++i) for(int j=0;j<4-i;++j) {
                        require(((du1.d[i][j]-du0.d[i][j])/(2*step)-d.d[i+1][j]).norm()<1e-7,
                            "trace93 analytic u derivative");
                        require(((dv1.d[i][j]-dv0.d[i][j])/(2*step)-d.d[i][j+1]).norm()<1e-7,
                            "trace93 analytic v derivative");
                    }
                }
                const std::size_t expected_sheets=name=="box"?6:name=="l"?8:name=="u"?10:3;
                require(sheets.size()==expected_sheets,"trace93 physical sheet count");
                if(name=="cylinder") {
                    require(c.surface.patch_names[4]=="top_center" && c.surface.patch_names[9]=="bottom_center",
                        "cylinder Python patch ID convention");
                    require(!c.native_parameters_match_analysis,"circle rational and linear-angle charts conflated");
                    require(c.surface.exact_inside(c.transform.forward_point({0,0,0})),"solid cylinder incorrectly hollow");
                    require(!c.surface.exact_inside(c.transform.forward_point({.55,0,0})),"solid cylinder radius");
                } else {
                    require(c.native_parameters_match_analysis,"affine chart should match native parameter");
                }
                const auto jet=c.evaluate(c.transform.forward_point({.17,-.21,.13}));
                require(std::abs(jet.hessian.trace())<3e-13,"trace93 manufactured field is not harmonic");
            }
        }
        std::cout<<"trace93 exact bicubic geometry / analysis chart cases: PASS\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
