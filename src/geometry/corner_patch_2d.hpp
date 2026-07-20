#pragma once

#include <vector>

#include "../interface/interface_2d.hpp"

namespace kfbim {

struct CornerPatchOptions2D {
    double geometry_tol = 1.0e-12;
    bool require_no_overlap = true;
};

enum class CornerPatchBuildMode2D {
    DominantOpening,
    BothSides,
    ReentrantCircleDominantOpening,
    Circle
};

struct CornerPatchConfig2D {
    bool enabled = false;
    double radius = 0.0;
    std::vector<double> radius_overrides_by_corner;
    CornerPatchBuildMode2D build_mode = CornerPatchBuildMode2D::DominantOpening;
    CornerPatchOptions2D options;
};

std::vector<CornerPatch2D> build_corner_patches_2d(
    const Interface2D& iface,
    double radius,
    CornerPatchBuildMode2D build_mode = CornerPatchBuildMode2D::DominantOpening,
    const CornerPatchOptions2D& options = {});

Interface2D apply_corner_patch_config_2d(
    const Interface2D& iface,
    const CornerPatchConfig2D& config);

Interface2D with_corner_patches_2d(
    const Interface2D& iface,
    std::vector<CornerPatch2D> patches);

bool corner_patch_contains_point_2d(
    const CornerPatch2D& patch,
    const Eigen::Vector2d& point,
    double tolerance = 1.0e-12);

bool corner_patch_sector_contains_point_2d(
    const CornerPatch2D& patch,
    const Eigen::Vector2d& point,
    double tolerance = 1.0e-12);

double corner_patch_radial_signed_distance_2d(
    const CornerPatch2D& patch,
    const Eigen::Vector2d& point);

const char* corner_patch_side_name(CornerPatchSide2D side);

} // namespace kfbim
