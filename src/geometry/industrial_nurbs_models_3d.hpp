#pragma once

#include "nurbs_surface_model_3d.hpp"

#include <string>
#include <vector>

namespace kfbim::geometry3d {

// The factories construct native rational tensor-product surfaces, without CAD input.
struct IndustrialNurbsModel3D {
    std::string name;
    std::string description;
    std::vector<NurbsSurfacePatch3D> patches;
    std::vector<std::string> patch_names;
    std::vector<NurbsPatchEdgeConnection3D> connections;
    double expected_volume = 0.0; // Zero means no analytic reference is supplied.
    int expected_genus = 0;

    NurbsSurfaceModel3D geometry_model() const;
};

// Pair full edges by their native evaluations; reject missing or ambiguous mates.
// These small, deliberately conforming examples need no trimming or partial edges.
IndustrialNurbsModel3D finalize_industrial_model(
    std::string name, std::string description,
    std::vector<NurbsSurfacePatch3D> patches,
    std::vector<std::string> patch_names,
    double expected_volume, int expected_genus);

IndustrialNurbsModel3D make_industrial_sleeve_3d();
IndustrialNurbsModel3D make_industrial_u_bracket_3d();
IndustrialNurbsModel3D make_industrial_flange_3d();
IndustrialNurbsModel3D make_industrial_impeller_3d();

} // namespace kfbim::geometry3d
