#include "src/geometry/models3d/industrial/models_3d.hpp"

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace kfbim::geometry3d {
namespace {

// All profile curves are quadratic Bezier curves in the (y,z) plane.
// Lines are degree-elevated so an exact circular hole and the rounded outer
// profile can bound one untrimmed rational tensor-product face strip.
struct BracketProfileCurve {
    std::array<Eigen::Vector2d, 3> points;
    std::array<double, 3> weights;
};

BracketProfileCurve profile_line(Eigen::Vector2d a, Eigen::Vector2d b)
{
    return {{{a, (a + b) / 2, b}}, {{1, 1, 1}}};
}

BracketProfileCurve pin_circle_arc(double radius, double center_z,
                                   double angle0, double angle1)
{
    const double middle = (angle0 + angle1) / 2;
    const double weight = std::cos((angle1 - angle0) / 2);
    return {{{{radius * std::cos(angle0), center_z + radius * std::sin(angle0)},
              {radius * std::cos(middle) / weight,
               center_z + radius * std::sin(middle) / weight},
              {radius * std::cos(angle1), center_z + radius * std::sin(angle1)}}},
            {{1, weight, 1}}};
}

} // namespace

IndustrialNurbsModel3D make_industrial_u_bracket_3d()
{
    const double pi = std::acos(-1.0);
    // Overall box: 1.70 x 1.00 x 1.34.  The 0.24-thick rectangular foot
    // carries two 0.27-thick ears, leaving a 1.16-wide fork opening.
    const std::array<double, 4> x{{-0.85, -0.58, 0.58, 0.85}};
    const double half_width = 0.50, base_bottom = -0.62, base_top = -0.38;
    const double ear_top = 0.72, corner_radius = 0.14;
    const double pin_center_z = 0.28, pin_radius = 0.20, chamfer = 0.03;
    const double mouth_radius = pin_radius + chamfer;
    const double corner_z = ear_top - corner_radius;
    const double corner_y = half_width - corner_radius;
    const double arc_weight = std::sqrt(0.5);

    // Counterclockwise outline viewed along -x.  Segment zero is the foot
    // attachment; it is present on the ear end faces but is never extruded.
    // That open attachment is closed by the exposed surfaces of the foot.
    const std::array<Eigen::Vector2d, 6> outer_vertices{{
        {-half_width, base_top}, {half_width, base_top},
        {half_width, corner_z}, {corner_y, ear_top},
        {-corner_y, ear_top}, {-half_width, corner_z}}};
    const std::array<BracketProfileCurve, 6> outer{{
        profile_line(outer_vertices[0], outer_vertices[1]),
        profile_line(outer_vertices[1], outer_vertices[2]),
        {{{outer_vertices[2], {half_width, ear_top}, outer_vertices[3]}},
         {{1, arc_weight, 1}}},
        profile_line(outer_vertices[3], outer_vertices[4]),
        {{{outer_vertices[4], {-half_width, ear_top}, outer_vertices[5]}},
         {{1, arc_weight, 1}}},
        profile_line(outer_vertices[5], outer_vertices[0])}};

    // Rays from the bore center to the outline vertices partition each ear
    // face into six regular O-grid strips.  These angles stay below pi/2;
    // every circle weight is strictly positive and no collapsed pole occurs.
    std::array<double, 7> angles{};
    for (int k = 0; k < 6; ++k) {
        angles[k] = std::atan2(outer_vertices[k].y() - pin_center_z,
                              outer_vertices[k].x());
        if (k > 0 && angles[k] <= angles[k - 1]) angles[k] += 2 * pi;
    }
    angles[6] = angles[0] + 2 * pi;

    std::vector<NurbsSurfacePatch3D> patches;
    std::vector<std::string> names;
    auto strip = [&](const BracketProfileCurve& a, double xa,
                     const BracketProfileCurve& b, double xb,
                     const std::string& name) {
        std::vector<std::vector<Eigen::Vector3d>> net(3,
            std::vector<Eigen::Vector3d>(2));
        std::vector<std::vector<double>> weights(3, std::vector<double>(2));
        for (int i = 0; i < 3; ++i) {
            net[i][0] = {xa, a.points[i].x(), a.points[i].y()};
            net[i][1] = {xb, b.points[i].x(), b.points[i].y()};
            weights[i][0] = a.weights[i];
            weights[i][1] = b.weights[i];
        }
        patches.emplace_back(geometry::NurbsBasis1D(2, {0, 0, 0, 1, 1, 1}),
                             geometry::NurbsBasis1D(1, {0, 0, 1, 1}),
                             std::move(net), std::move(weights));
        names.push_back(name);
    };
    auto plane = [&](Eigen::Vector3d a, Eigen::Vector3d b,
                     Eigen::Vector3d c, Eigen::Vector3d d,
                     const std::string& name) {
        patches.push_back(NurbsSurfacePatch3D::make_bilinear_plane(a, b, c, d));
        names.push_back(name);
    };

    for (int ear = 0; ear < 2; ++ear) {
        const double left = x[ear == 0 ? 0 : 2];
        const double right = x[ear == 0 ? 1 : 3];
        const std::string prefix = ear == 0 ? "left_ear_" : "right_ear_";
        for (int k = 0; k < 6; ++k) {
            const auto mouth = pin_circle_arc(mouth_radius, pin_center_z,
                                              angles[k], angles[k + 1]);
            const auto bore = pin_circle_arc(pin_radius, pin_center_z,
                                             angles[k], angles[k + 1]);
            const std::string suffix = "_" + std::to_string(k);
            strip(mouth, left, outer[k], left, prefix + "left_face" + suffix);
            strip(outer[k], right, mouth, right, prefix + "right_face" + suffix);
            if (k != 0)
                strip(outer[k], left, outer[k], right,
                      prefix + "outer_wall" + suffix);
            // Descending axial order points normals into the pin hole.
            // The mouth-to-bore joins are exact 45-degree conical chamfers.
            strip(mouth, right, bore, right - chamfer,
                  prefix + "right_pin_chamfer" + suffix);
            strip(bore, right - chamfer, bore, left + chamfer,
                  prefix + "pin_bore" + suffix);
            strip(bore, left + chamfer, mouth, left,
                  prefix + "left_pin_chamfer" + suffix);
        }
    }

    // Continue every ear attachment corner across the foot.  Dividing its
    // front, back and underside at all four x stations keeps every adjacency
    // a complete [0,1] edge, including the exposed top in the fork opening.
    for (int i = 0; i < 3; ++i) {
        const double a = x[i], b = x[i + 1];
        const std::string suffix = "_" + std::to_string(i);
        plane({b,-half_width,base_bottom}, {a,-half_width,base_bottom},
              {b, half_width,base_bottom}, {a, half_width,base_bottom},
              "base_bottom" + suffix);
        plane({a,-half_width,base_bottom}, {b,-half_width,base_bottom},
              {a,-half_width,base_top}, {b,-half_width,base_top},
              "base_front" + suffix);
        plane({b, half_width,base_bottom}, {a, half_width,base_bottom},
              {b, half_width,base_top}, {a, half_width,base_top},
              "base_back" + suffix);
    }
    plane({x[0], half_width,base_bottom}, {x[0],-half_width,base_bottom},
          {x[0], half_width,base_top}, {x[0],-half_width,base_top}, "base_left");
    plane({x[3],-half_width,base_bottom}, {x[3], half_width,base_bottom},
          {x[3],-half_width,base_top}, {x[3], half_width,base_top}, "base_right");
    plane({x[1],-half_width,base_top}, {x[2],-half_width,base_top},
          {x[1], half_width,base_top}, {x[2], half_width,base_top}, "fork_floor");

    // Independent volume: rectangular foot + two rounded ear extrusions,
    // minus two cylindrical cavities and four conical mouth cavities.
    const double thickness = x[1] - x[0];
    const double ear_area = 2 * half_width * (ear_top - base_top)
        - 2 * corner_radius * corner_radius * (1 - pi / 4);
    const double pin_volume = pi * pin_radius * pin_radius * (thickness - 2 * chamfer)
        + 2 * pi * chamfer * (pin_radius * pin_radius
            + pin_radius * mouth_radius + mouth_radius * mouth_radius) / 3;
    const double foot_volume = (x[3] - x[0]) * 2 * half_width * (base_top - base_bottom);
    return finalize_industrial_model("u_bracket",
        "Upright clevis bracket with two rounded ears, aligned pin bores and chamfered rims",
        std::move(patches), std::move(names),
        foot_volume + 2 * (ear_area * thickness - pin_volume), 2);
}

} // namespace kfbim::geometry3d
