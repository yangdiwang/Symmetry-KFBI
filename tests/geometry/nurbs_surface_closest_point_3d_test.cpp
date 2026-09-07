#include "src/geometry/nurbs_surface_closest_point_3d.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace kfbim::geometry3d;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void contains_distance(const NurbsSurfaceClosestPointResult3D& r, double d) {
    require(r.distance_lower <= d + 1e-13 && r.distance_upper >= d - 1e-13,
            "distance enclosure does not contain known minimum");
}
void plane_and_boundary() {
    const auto plane = NurbsSurfacePatch3D::make_unit_square_xy();
    NurbsSurfaceClosestPointIndex3D index(NurbsSurfaceModel3D({plane}, {0}, {}));
    NurbsSurfaceClosestPointWorkspace3D workspace;
    NurbsSurfaceClosestPointOptions3D options;
    options.localization_tolerance = 1e-5;
    options.max_boxes = 4096;
    const auto r = index.query({0.3, 0.6, 0.7}, options, workspace);
    require(r.status == NurbsSurfaceClosestPointStatus3D::Bounded, "plane distance not bounded");
    require(r.localized, "plane minimum not localized");
    require((r.point - Eigen::Vector3d(0.3,0.6,0)).norm() < 1e-10, "plane closest point");
    require(r.normal_valid && r.normal.z() > 0.999999, "plane normal");
    contains_distance(r, 0.7);
    const auto edge = index.query({1.2,-0.3,0.4}, options, workspace);
    require((edge.point-Eigen::Vector3d(1,0,0)).norm() < 1e-10, "corner constrained minimum");
    contains_distance(edge, std::sqrt(0.29));
    const auto internal_boundary=index.query({0.5,0.5,0.7},options,workspace);
    require(internal_boundary.localized,"artificial cell-boundary minimum was lost");
    contains_distance(internal_boundary,0.7);
}
void cylinder_and_budget() {
    const auto patch = NurbsSurfacePatch3D::make_quarter_cylinder_patch(1.0,-1.0,1.0);
    NurbsSurfaceClosestPointIndex3D index(NurbsSurfaceModel3D({patch},{0},{}));
    NurbsSurfaceClosestPointWorkspace3D workspace;
    NurbsSurfaceClosestPointOptions3D options;
    options.distance_tolerance = 1e-5;
    options.localization_tolerance = 1e-3;
    options.max_boxes = 2048;
    const double a = 0.6;
    const Eigen::Vector3d direction(std::cos(a),std::sin(a),0);
    const auto r = index.query(1.2*direction+Eigen::Vector3d(0,0,0.2),options,workspace);
    contains_distance(r,0.2);
    require((r.point-(direction+Eigen::Vector3d(0,0,0.2))).norm()<1e-7,"cylinder candidate error");
    require(r.normal_valid,"cylinder normal missing");
    options.max_boxes = 1;
    const auto limited = index.query(1.2*direction,options,workspace);
    require(limited.status == NurbsSurfaceClosestPointStatus3D::BudgetExceeded,
            "one-box query must report unfinished global distance search");
    require(!limited.localized,"budget limit must not invent localization");
    contains_distance(limited,0.2);
}
void multiple_patches_and_ties() {
    const auto a = NurbsSurfacePatch3D::make_unit_square_xy();
    const auto b = NurbsSurfacePatch3D::make_bilinear_plane(
        {0,0,2},{1,0,2},{0,1,2},{1,1,2});
    NurbsSurfaceClosestPointIndex3D index(NurbsSurfaceModel3D({a,b},{0,1},{}));
    NurbsSurfaceClosestPointWorkspace3D workspace;
    NurbsSurfaceClosestPointOptions3D options;
    options.max_boxes = 512;
    const auto r = index.query({0.3,0.4,1},options,workspace);
    contains_distance(r,1);
    require(r.status == NurbsSurfaceClosestPointStatus3D::Bounded,"tied distance can still be bounded");
    require(r.separated_near_ties && !r.localized,"distinct equal minima must not appear unique");
    require(r.candidates.size()>=2,"equal minima candidates missing");
    const auto near_b = index.query({0.3,0.4,1.8},options,workspace);
    require(near_b.patch_index==1 && near_b.component==1,"nearest patch/component selection");
    contains_distance(near_b,0.2);
}
void smooth_seam() {
    const auto a=NurbsSurfacePatch3D::make_unit_square_xy();
    const auto b=NurbsSurfacePatch3D::make_bilinear_plane({1,0,0},{2,0,0},{1,1,0},{2,1,0});
    NurbsSurfaceClosestPointIndex3D index(NurbsSurfaceModel3D({a,b},{0,0},{}));
    NurbsSurfaceClosestPointWorkspace3D workspace;
    NurbsSurfaceClosestPointOptions3D options;options.localization_tolerance=1e-5;
    const auto r=index.query({1,0.4,0.3},options,workspace);
    require(r.status==NurbsSurfaceClosestPointStatus3D::Bounded && r.localized,"seam point must remain localized");
    require(!r.separated_near_ties && r.candidates.size()>=2,"coincident seam owners must be retained");
    contains_distance(r,0.3);
}
void stable_deep_derivative_enclosure() {
    // A real N32 Q27 grid node (id 11474) whose incumbent was accurate to
    // 3e-10, but differences of deep interval control points stalled location.
    const auto unshifted=NurbsSurfacePatch3D::make_quarter_cylinder_patch(.55,-.63,.67);
    auto controls=unshifted.control_net();
    for(auto& row:controls)for(auto& p:row)p+=Eigen::Vector3d(.06,-.05,0);
    const NurbsSurfacePatch3D patch(unshifted.basis_u(),unshifted.basis_v(),controls,unshifted.weights());
    NurbsSurfaceClosestPointIndex3D index(NurbsSurfaceModel3D({patch},{0},{}));
    NurbsSurfaceClosestPointWorkspace3D workspace;
    NurbsSurfaceClosestPointOptions3D options;
    options.distance_tolerance=1e-9*.09375;options.localization_tolerance=1e-6*.09375;
    const Eigen::Vector3d a(.65625,.09375,-.5625);
    const auto r=index.query(a,options,workspace);
    const double radius=std::hypot(a.x()-.06,a.y()+.05);
    const Eigen::Vector3d q(.06+.55*(a.x()-.06)/radius,-.05+.55*(a.y()+.05)/radius,a.z());
    require((r.point-q).norm()<1e-8,"deep-bound fixture must have an accurate incumbent");
    contains_distance(r,radius-.55);
    require(r.localized,"accurate real-grid incumbent must achieve requested localization within budget");
    require(r.stats.boxes_visited<2048,"real-grid localization must not exhaust the default budget");
}
}
int main() {
    try {
        plane_and_boundary(); cylinder_and_budget(); multiple_patches_and_ties();smooth_seam();
        stable_deep_derivative_enclosure();
        std::cout << "NURBS surface closest-point tests passed\n"; return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
