#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <Eigen/SVD>

#include "crossing_owner_restrict_3d.hpp"
#include "crossing_cauchy_plan_3d.hpp"
#include "csv_rfc4180.hpp"
#include "direct_coefficient_cauchy_3d.hpp"
#include "dirichlet_rigid_transform_study_3d.hpp"
#include "edge_jump_jet_reduction_3d.hpp"
#include "exterior_only_cubic_normal_restrict_3d.hpp"
#include "geometric_feature_trace_fit_3d.hpp"
#include "harmonic_polynomial_space_3d.hpp"
#include "native_density_resolution_3d.hpp"
#include "native_nurbs_density_space_3d.hpp"
#include "native_endpoint_path_adapter_3d.hpp"
#include "native_nurbs_surface_3d.hpp"
#include "reduced_trace_projection_3d.hpp"
#include "restrict_crossing_selector_3d.hpp"
#include "shared_quadratic_restrict_3d.hpp"
#include "tensor_product_cover_restrict_3d.hpp"
#include "topology_affine_reduction_3d.hpp"
#include "topology_density_constraints_3d.hpp"
#include "topology_mean_free_reduction_3d.hpp"
#include "topology_reachable_target_projection_3d.hpp"
#include "topology_trace_projector_3d.hpp"
#include "src/bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include "src/geometry/grid_pair_3d.hpp"
#include "src/geometry/nurbs_bezier_intersection_3d.hpp"
#include "src/geometry/nurbs_cartesian_domain_3d.hpp"
#include "src/geometry/nurbs_patch_triangulator_3d.hpp"
#include "src/geometry/p2_surface_3d.hpp"
#include "src/gmres/gmres.hpp"
#include "src/grid/cartesian_grid_3d.hpp"
#include "src/potentials/laplace_potential.hpp"
#include "src/transfer/laplace_correction_support.hpp"
#include "src/transfer/laplace_restrict_3d.hpp"
#include "src/transfer/laplace_spread_3d.hpp"

using namespace kfbim;

namespace {

constexpr double kBoxMin = -1.5;
constexpr double kBoxSide = 3.0;
constexpr double kTargetP2NodeSpacingOverH = 1.2;
constexpr int kCauchyValueNeighborCount = 48;
constexpr int kCauchyDerivativeNeighborCount = 28;
constexpr int kNeumannRestrictValueNeighborCount = 24;
constexpr int kNeumannRestrictNormalNeighborCount = 12;
#ifndef KFBIM_3D_CAUCHY_POLYNOMIAL_DEGREE
#define KFBIM_3D_CAUCHY_POLYNOMIAL_DEGREE 3
#endif
#ifndef KFBIM_3D_RESTRICT_GRID_DEGREE
#define KFBIM_3D_RESTRICT_GRID_DEGREE 3
#endif
#ifndef KFBIM_3D_RESTRICT_NORMAL_DEGREE
#define KFBIM_3D_RESTRICT_NORMAL_DEGREE 3
#endif
constexpr int kCauchyPolynomialDegree =
    KFBIM_3D_CAUCHY_POLYNOMIAL_DEGREE;
constexpr int kRestrictGridDegree =
    KFBIM_3D_RESTRICT_GRID_DEGREE;
constexpr int kRestrictNormalDegree =
    KFBIM_3D_RESTRICT_NORMAL_DEGREE;
constexpr int kRestrictGridStencilSize = kRestrictGridDegree + 1;
constexpr int kRestrictGridPointCount =
    kRestrictGridStencilSize
    * kRestrictGridStencilSize
    * kRestrictGridStencilSize;
constexpr int kRestrictGridLeftOffset = kRestrictGridDegree / 2;
constexpr int kRestrictNormalLayerCount = kRestrictNormalDegree + 1;
constexpr int kRestrictNormalSampleCount =
    2 * kRestrictNormalLayerCount;
constexpr int kJointNormalCoefficientCount =
    2 * kRestrictNormalDegree;
constexpr int kGlobalTraceCoefficientCount = 4;

bool selected_native_endpoint_support_path()
{
    const char* raw = std::getenv("KFBIM_3D_SUPPORT_PATH");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "legacy")
        return false;
    if (std::string(raw) == "native_certified")
        return true;
    throw std::invalid_argument(
        "KFBIM_3D_SUPPORT_PATH must be legacy or native_certified");
}

std::filesystem::path support_path_output_root_3d(std::filesystem::path root)
{
    // Keep opt-in certification runs away from existing convergence baselines.
    if (selected_native_endpoint_support_path())
        root /= "native_endpoint";
    return root;
}

std::filesystem::path application_output_root_3d()
{
    if (const char* override_root =
            std::getenv("KFBIM_3D_OUTPUT_ROOT")) {
        if (*override_root != '\0')
            return support_path_output_root_3d(std::filesystem::path(override_root));
    }
#ifdef KFBIM_APP_OUTPUT_DIR
    return support_path_output_root_3d(std::filesystem::path(KFBIM_APP_OUTPUT_DIR));
#else
    return support_path_output_root_3d(std::filesystem::path("output"));
#endif
}

static_assert(kCauchyPolynomialDegree >= 1,
              "Cauchy polynomial degree must be positive");
static_assert(kRestrictGridDegree >= 1,
              "grid interpolation degree must be positive");
static_assert(kRestrictNormalDegree >= 2,
              "normal trace degree must be at least quadratic");

using GeometryKind = app3d::GeometryKind3D;
using NativeNurbsSurface3D = app3d::NativeNurbsSurface3D;
using SurfaceDof = app3d::SurfaceDof3D;
using SurfaceDofCloud = app3d::SurfaceDofCloud3D;
using SurfacePatchInfo = app3d::SurfaceDofPatch3D;

enum class CauchyStencilPolicy3D {
    G1Nearest,
    TopologicalNearest,
    SamePatch,
    BalancedPatches
};

enum class SolveSelection3D {
    Both,
    NeumannOnly,
    DirichletNormalOnly
};

enum class DensityIterationMode3D {
    SurfaceSamples,
    ReducedCoefficients
};

enum class ExteriorNormalRestrictMode3D {
    JointTricubicCauchy,
    JointTricubicCrossingOwner,
    GlobalCubicExteriorBranchCrossingOwner,
    SharedQuadraticGridlineCauchy,
    SharedQ10CubicGridlineCauchy,
    Q27Cover3AllEventCauchy,
    Q64Cover4AllEventCauchy,
    ExteriorOnlyHarmonicCubic
};

enum class NeumannCompatibilityMode3D {
    OperatorFluxCorrection,
    TraceBorderLegacy
};

enum class CauchyValueTraceFitMode3D {
    GeometricFeature,
    LegacyJoint
};

// The reduced-coefficient Neumann operator uses the panel-centred value jet
// only when it restricts the Cartesian interface solution back to the
// surface.  At a C0 feature, using the G1 patch-owned (one-sided) fit avoids
// coupling unrelated higher derivatives across the sharp seam.  This switch
// deliberately does not alter the crossing-local spread construction.
enum class NeumannRestrictValueJetMode3D {
    C0OneSided,
    FeatureConstrained
};

std::string approximation_scheme_name()
{
    if (kCauchyPolynomialDegree == 3
        && kRestrictGridDegree == 3
        && kRestrictNormalDegree == 3) {
        return "all_cubic";
    }
    if (kCauchyPolynomialDegree == 4
        && kRestrictGridDegree == 4
        && kRestrictNormalDegree == 4) {
        return "all_quartic";
    }
    return "mixed_c" + std::to_string(kCauchyPolynomialDegree)
         + "_g" + std::to_string(kRestrictGridDegree)
         + "_n" + std::to_string(kRestrictNormalDegree);
}

std::string cauchy_policy_name(CauchyStencilPolicy3D policy)
{
    switch (policy) {
    case CauchyStencilPolicy3D::G1Nearest:
        return "g1_nearest";
    case CauchyStencilPolicy3D::TopologicalNearest:
        return "topological_nearest";
    case CauchyStencilPolicy3D::SamePatch:
        return "same_patch";
    case CauchyStencilPolicy3D::BalancedPatches:
        return "balanced_patches";
    }
    throw std::runtime_error("unknown 3D Cauchy stencil policy");
}

CauchyStencilPolicy3D selected_cauchy_policy()
{
    const char* raw = std::getenv("KFBIM_3D_CAUCHY_POLICY");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "g1_nearest") {
        return CauchyStencilPolicy3D::G1Nearest;
    }
    if (std::string(raw) == "same_patch")
        return CauchyStencilPolicy3D::SamePatch;
    if (std::string(raw) == "topological_nearest")
        return CauchyStencilPolicy3D::TopologicalNearest;
    if (std::string(raw) == "balanced_patches")
        return CauchyStencilPolicy3D::BalancedPatches;
    throw std::invalid_argument(
        "KFBIM_3D_CAUCHY_POLICY must be g1_nearest, topological_nearest, "
        "same_patch, or balanced_patches");
}

std::string cauchy_value_trace_fit_mode_name(
    CauchyValueTraceFitMode3D mode)
{
    switch (mode) {
    case CauchyValueTraceFitMode3D::GeometricFeature:
        return "geometric_feature";
    case CauchyValueTraceFitMode3D::LegacyJoint:
        return "legacy_joint";
    }
    throw std::runtime_error("unknown Cauchy value-trace fit mode");
}

CauchyValueTraceFitMode3D selected_cauchy_value_trace_fit_mode()
{
    const char* raw = std::getenv("KFBIM_3D_FEATURE_TRACE_FIT");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "geometric_feature"
        || std::string(raw) == "geometric") {
        return CauchyValueTraceFitMode3D::GeometricFeature;
    }
    if (std::string(raw) == "legacy"
        || std::string(raw) == "legacy_joint") {
        return CauchyValueTraceFitMode3D::LegacyJoint;
    }
    throw std::invalid_argument(
        "KFBIM_3D_FEATURE_TRACE_FIT must be geometric_feature or "
        "legacy_joint");
}

std::string neumann_restrict_value_jet_mode_name(
    NeumannRestrictValueJetMode3D mode)
{
    switch (mode) {
    case NeumannRestrictValueJetMode3D::C0OneSided:
        return "c0_one_sided";
    case NeumannRestrictValueJetMode3D::FeatureConstrained:
        return "feature_constrained";
    }
    throw std::runtime_error("unknown Neumann restrict value-jet mode");
}

NeumannRestrictValueJetMode3D default_neumann_restrict_value_jet_mode()
{
#ifdef KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT
    return NeumannRestrictValueJetMode3D::C0OneSided;
#else
    // The legacy sample-DOF formulation does not use the separated
    // coefficient restrict path, so retain its historical default label.
    return NeumannRestrictValueJetMode3D::FeatureConstrained;
#endif
}

NeumannRestrictValueJetMode3D selected_neumann_restrict_value_jet_mode()
{
    const char* raw = std::getenv("KFBIM_3D_NEUMANN_RESTRICT_VALUE_JET");
    if (raw == nullptr || std::string(raw).empty())
        return default_neumann_restrict_value_jet_mode();
    const std::string value(raw);
    if (value == "c0_one_sided" || value == "one_sided"
        || value == "legacy_one_sheet" || value == "legacy_joint") {
        return NeumannRestrictValueJetMode3D::C0OneSided;
    }
    if (value == "feature_constrained" || value == "geometric_feature"
        || value == "geometric") {
        return NeumannRestrictValueJetMode3D::FeatureConstrained;
    }
    throw std::invalid_argument(
        "KFBIM_3D_NEUMANN_RESTRICT_VALUE_JET must be c0_one_sided "
        "or feature_constrained");
}

std::string neumann_compatibility_mode_name(
    NeumannCompatibilityMode3D mode)
{
    switch (mode) {
    case NeumannCompatibilityMode3D::OperatorFluxCorrection:
        return "operator_flux";
    case NeumannCompatibilityMode3D::TraceBorderLegacy:
        return "trace_border_legacy";
    }
    throw std::runtime_error("unknown 3D Neumann compatibility mode");
}

NeumannCompatibilityMode3D selected_neumann_compatibility_mode()
{
    const char* raw = std::getenv("KFBIM_3D_NEUMANN_COMPATIBILITY");
    if (raw == nullptr || std::string(raw).empty()) {
#ifdef KFBIM_3D_TOPOLOGY_AFFINE_DEFAULT
        return NeumannCompatibilityMode3D::TraceBorderLegacy;
#else
        return NeumannCompatibilityMode3D::OperatorFluxCorrection;
#endif
    }
    if (std::string(raw) == "operator_flux")
        return NeumannCompatibilityMode3D::OperatorFluxCorrection;
    if (std::string(raw) == "trace_border_legacy")
        return NeumannCompatibilityMode3D::TraceBorderLegacy;
    throw std::invalid_argument(
        "KFBIM_3D_NEUMANN_COMPATIBILITY must be operator_flux "
        "or trace_border_legacy");
}

bool selected_neumann_galerkin_trace_projection()
{
    const char* raw = std::getenv("KFBIM_3D_NEUMANN_TRACE_PROJECTION");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "least_squares") {
        return false;
    }
    if (std::string(raw) == "galerkin"
        || std::string(raw) == "weighted_transpose") {
        return true;
    }
    throw std::invalid_argument(
        "KFBIM_3D_NEUMANN_TRACE_PROJECTION must be least_squares "
        "or galerkin");
}

bool selected_neumann_trace_mass_coordinates()
{
    const char* raw = std::getenv(
        "KFBIM_3D_NEUMANN_DENSITY_COORDINATES");
    if (raw == nullptr || std::string(raw).empty()) {
#ifdef KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT
        return true;
#else
        return false;
#endif
    }
    if (std::string(raw) == "c0_euclidean")
        return false;
    if (std::string(raw) == "trace_mass"
        || std::string(raw) == "mass_whitened") {
        return true;
    }
    throw std::invalid_argument(
        "KFBIM_3D_NEUMANN_DENSITY_COORDINATES must be c0_euclidean "
        "or trace_mass");
}

bool selected_neumann_projected_border_elimination()
{
    const char* raw = std::getenv("KFBIM_3D_NEUMANN_BORDER_SOLVER");
    if (raw == nullptr || std::string(raw).empty()) {
#ifdef KFBIM_3D_TOPOLOGY_AFFINE_DEFAULT
        return false;
#elif defined(KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT)
        // An explicit diagnostic switch back to the legacy Euclidean
        // density coordinates should remain runnable without requiring a
        // second environment override.
        return selected_neumann_trace_mass_coordinates();
#else
        return false;
#endif
    }
    if (std::string(raw) == "augmented")
        return false;
    if (std::string(raw) == "mean_free_pivot_elimination"
        || std::string(raw) == "mean_free_pivot"
        || std::string(raw) == "mean_free_reduction") {
        return false;
    }
    if (std::string(raw) == "projected_elimination"
        || std::string(raw) == "projected") {
        return true;
    }
    if (std::string(raw) == "householder_elimination"
        || std::string(raw) == "householder") {
        return false;
    }
    throw std::invalid_argument(
        "KFBIM_3D_NEUMANN_BORDER_SOLVER must be augmented, "
        "projected_elimination, householder_elimination, or "
        "mean_free_pivot_elimination");
}

bool selected_neumann_householder_border_elimination()
{
    const char* raw = std::getenv("KFBIM_3D_NEUMANN_BORDER_SOLVER");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "augmented"
        || std::string(raw) == "projected_elimination"
        || std::string(raw) == "projected"
        || std::string(raw) == "mean_free_pivot_elimination"
        || std::string(raw) == "mean_free_pivot"
        || std::string(raw) == "mean_free_reduction") {
        return false;
    }
    if (std::string(raw) == "householder_elimination"
        || std::string(raw) == "householder") {
        return true;
    }
    throw std::invalid_argument(
        "KFBIM_3D_NEUMANN_BORDER_SOLVER must be augmented, "
        "projected_elimination, householder_elimination, or "
        "mean_free_pivot_elimination");
}

bool selected_neumann_mean_free_pivot_elimination()
{
    const char* raw = std::getenv("KFBIM_3D_NEUMANN_BORDER_SOLVER");
    if (raw == nullptr || std::string(raw).empty()) {
#ifdef KFBIM_3D_TOPOLOGY_AFFINE_DEFAULT
        return true;
#else
        return false;
#endif
    }
    const std::string value(raw);
    if (value == "mean_free_pivot_elimination"
        || value == "mean_free_pivot"
        || value == "mean_free_reduction") {
        return true;
    }
    if (value == "augmented" || value == "projected_elimination"
        || value == "projected" || value == "householder_elimination"
        || value == "householder") {
        return false;
    }
    throw std::invalid_argument(
        "KFBIM_3D_NEUMANN_BORDER_SOLVER must be augmented, "
        "projected_elimination, householder_elimination, or "
        "mean_free_pivot_elimination");
}

bool selected_neumann_native_gauss_trace_sampling()
{
    const char* raw = std::getenv("KFBIM_3D_NEUMANN_TRACE_SAMPLING");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "panel_centers") {
        return false;
    }
    if (std::string(raw) == "native_gauss"
        || std::string(raw) == "density_gauss") {
        return true;
    }
    throw std::invalid_argument(
        "KFBIM_3D_NEUMANN_TRACE_SAMPLING must be panel_centers "
        "or native_gauss");
}

enum class NeumannEdgeJumpJetMode3D {
    Disabled,
    StrongFeatureMortar,
    TopologyAffineLocalSvd,
    TopologyAmbientGradientLocalSvd
};

NeumannEdgeJumpJetMode3D selected_neumann_edge_jump_jet_mode()
{
    const char* raw = std::getenv("KFBIM_3D_NEUMANN_EDGE_JUMP_JET");
    if (raw == nullptr || std::string(raw).empty()) {
#ifdef KFBIM_3D_TOPOLOGY_AFFINE_DEFAULT
        return NeumannEdgeJumpJetMode3D::TopologyAffineLocalSvd;
#else
        return NeumannEdgeJumpJetMode3D::Disabled;
#endif
    }
    if (std::string(raw) == "disabled" || std::string(raw) == "off")
        return NeumannEdgeJumpJetMode3D::Disabled;
    if (std::string(raw) == "strong_feature_mortar"
        || std::string(raw) == "strong" || std::string(raw) == "mortar") {
        return NeumannEdgeJumpJetMode3D::StrongFeatureMortar;
    }
    if (std::string(raw) == "topology_affine_local_svd"
        || std::string(raw) == "topology_affine"
        || std::string(raw) == "local_affine_svd") {
        return NeumannEdgeJumpJetMode3D::TopologyAffineLocalSvd;
    }
    if (std::string(raw) == "topology_ambient_gradient_local_svd"
        || std::string(raw) == "topology_ambient_gradient"
        || std::string(raw) == "ambient_gradient") {
        return NeumannEdgeJumpJetMode3D::
            TopologyAmbientGradientLocalSvd;
    }
    throw std::invalid_argument(
        "KFBIM_3D_NEUMANN_EDGE_JUMP_JET must be disabled, "
        "strong_feature_mortar, topology_affine_local_svd, or "
        "topology_ambient_gradient_local_svd");
}

std::string neumann_edge_jump_jet_mode_name()
{
    switch (selected_neumann_edge_jump_jet_mode()) {
    case NeumannEdgeJumpJetMode3D::Disabled:
        return "disabled";
    case NeumannEdgeJumpJetMode3D::StrongFeatureMortar:
        return "strong_feature_mortar";
    case NeumannEdgeJumpJetMode3D::TopologyAffineLocalSvd:
        return "topology_affine_local_svd";
    case NeumannEdgeJumpJetMode3D::TopologyAmbientGradientLocalSvd:
        return "topology_ambient_gradient_local_svd";
    }
    throw std::runtime_error("unknown Neumann edge jump-jet mode");
}

enum class DirichletJumpSpaceMode3D {
    LegacySampleFit,
    AnalyticJ0AffineJ1
};

constexpr DirichletJumpSpaceMode3D default_dirichlet_jump_space_mode()
{
#ifdef KFBIM_3D_TOPOLOGY_AFFINE_DEFAULT
    return DirichletJumpSpaceMode3D::AnalyticJ0AffineJ1;
#else
    return DirichletJumpSpaceMode3D::LegacySampleFit;
#endif
}

DirichletJumpSpaceMode3D selected_dirichlet_jump_space_mode()
{
    const char* raw = std::getenv("KFBIM_3D_DIRICHLET_JUMP_SPACE");
    if (raw == nullptr || std::string(raw).empty())
        return default_dirichlet_jump_space_mode();
    const std::string value(raw);
    if (value == "legacy_sample_fit" || value == "legacy"
        || value == "sample_fit") {
        return DirichletJumpSpaceMode3D::LegacySampleFit;
    }
    if (value == "analytic_j0_affine_j1" || value == "analytic_affine"
        || value == "topology_affine") {
        return DirichletJumpSpaceMode3D::AnalyticJ0AffineJ1;
    }
    throw std::invalid_argument(
        "KFBIM_3D_DIRICHLET_JUMP_SPACE must be legacy_sample_fit or "
        "analytic_j0_affine_j1");
}

std::string dirichlet_jump_space_mode_name(
    DirichletJumpSpaceMode3D mode)
{
    switch (mode) {
    case DirichletJumpSpaceMode3D::LegacySampleFit:
        return "legacy_sample_fit";
    case DirichletJumpSpaceMode3D::AnalyticJ0AffineJ1:
        return "analytic_j0_affine_j1";
    }
    throw std::runtime_error("unknown Dirichlet jump-space mode");
}

enum class DirichletFeatureCouplingMode3D {
    BrokenSheets,
    AmbientGradientAffineMortar
};

DirichletFeatureCouplingMode3D selected_dirichlet_feature_coupling_mode()
{
    const char* raw = std::getenv("KFBIM_3D_DIRICHLET_FEATURE_COUPLING");
    if (raw == nullptr || std::string(raw).empty())
        return DirichletFeatureCouplingMode3D::BrokenSheets;
    const std::string value(raw);
    if (value == "broken_sheets" || value == "broken"
        || value == "disabled") {
        return DirichletFeatureCouplingMode3D::BrokenSheets;
    }
    if (value == "ambient_gradient_affine_mortar"
        || value == "ambient_gradient" || value == "affine_mortar") {
        return DirichletFeatureCouplingMode3D::
            AmbientGradientAffineMortar;
    }
    throw std::invalid_argument(
        "KFBIM_3D_DIRICHLET_FEATURE_COUPLING must be broken_sheets or "
        "ambient_gradient_affine_mortar");
}

std::string dirichlet_feature_coupling_mode_name(
    DirichletFeatureCouplingMode3D mode)
{
    switch (mode) {
    case DirichletFeatureCouplingMode3D::BrokenSheets:
        return "broken_sheets";
    case DirichletFeatureCouplingMode3D::AmbientGradientAffineMortar:
        return "ambient_gradient_affine_mortar";
    }
    throw std::runtime_error("unknown Dirichlet feature-coupling mode");
}

std::string neumann_density_coordinates_name()
{
    return selected_neumann_trace_mass_coordinates()
        ? "trace_mass"
        : "c0_euclidean";
}

std::string neumann_border_solver_name()
{
    if (selected_neumann_mean_free_pivot_elimination())
        return "mean_free_pivot_elimination";
    if (selected_neumann_householder_border_elimination())
        return "householder_elimination";
    if (selected_neumann_projected_border_elimination())
        return "projected_elimination";
    return "augmented";
}

std::string neumann_trace_sampling_name()
{
    return selected_neumann_native_gauss_trace_sampling()
        ? "native_gauss"
        : "panel_centers";
}

std::string neumann_output_mode_name()
{
    const std::string sampling =
        selected_neumann_native_gauss_trace_sampling() ? "ng" : "pc";
    const std::string coordinates =
        selected_neumann_trace_mass_coordinates() ? "tm" : "c0";
    std::string border = "aug";
    if (selected_neumann_mean_free_pivot_elimination())
        border = "mf";
    else if (selected_neumann_householder_border_elimination())
        border = "hh";
    else if (selected_neumann_projected_border_elimination())
        border = "proj";
    return "nm_" + sampling + '_' + coordinates + '_' + border;
}

std::string trace_restrict_mode_name(ExteriorNormalRestrictMode3D mode)
{
    switch (mode) {
    case ExteriorNormalRestrictMode3D::JointTricubicCauchy:
        return "joint_tricubic_cauchy";
    case ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner:
        return "joint_tricubic_crossing_owner";
    case ExteriorNormalRestrictMode3D::
            GlobalCubicExteriorBranchCrossingOwner:
        return "global_cubic_exterior_branch_crossing_owner";
    case ExteriorNormalRestrictMode3D::SharedQuadraticGridlineCauchy:
        return "shared_quadratic_gridline_cauchy";
    case ExteriorNormalRestrictMode3D::SharedQ10CubicGridlineCauchy:
        return "shared_q10_cubic_gridline_cauchy";
    case ExteriorNormalRestrictMode3D::Q27Cover3AllEventCauchy:
        return "q27_cover3_all_event_cauchy";
    case ExteriorNormalRestrictMode3D::Q64Cover4AllEventCauchy:
        return "q64_cover4_all_event_cauchy";
    case ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic:
        return "exterior_only_harmonic_cubic";
    }
    throw std::runtime_error("unknown 3D trace restrict mode");
}

ExteriorNormalRestrictMode3D selected_neumann_trace_restrict_mode()
{
    const char* raw = std::getenv("KFBIM_3D_NEUMANN_TRACE_RESTRICT");
    if (raw == nullptr || std::string(raw).empty()) {
#ifdef KFBIM_3D_TOPOLOGY_AFFINE_DEFAULT
        return ExteriorNormalRestrictMode3D::
            Q27Cover3AllEventCauchy;
#elif defined(KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT)
        return ExteriorNormalRestrictMode3D::
            GlobalCubicExteriorBranchCrossingOwner;
#else
        return ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner;
#endif
    }
    if (std::string(raw) == "joint_tricubic_crossing_owner"
        || std::string(raw) == "crossing_owner") {
        return ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner;
    }
    if (std::string(raw) == "joint_tricubic_cauchy"
        || std::string(raw) == "legacy") {
        return ExteriorNormalRestrictMode3D::JointTricubicCauchy;
    }
    if (std::string(raw) == "global_cubic_exterior_branch_crossing_owner"
        || std::string(raw) == "global_cubic_exterior_branch"
        || std::string(raw) == "global_exterior_branch") {
        return ExteriorNormalRestrictMode3D::
            GlobalCubicExteriorBranchCrossingOwner;
    }
    if (std::string(raw) == "shared_quadratic_gridline_cauchy"
        || std::string(raw) == "shared_quadratic"
        || std::string(raw) == "quadratic_gridline") {
        return ExteriorNormalRestrictMode3D::
            SharedQuadraticGridlineCauchy;
    }
    if (std::string(raw) == "shared_q10_cubic_gridline_cauchy"
        || std::string(raw) == "shared_q10_cubic"
        || std::string(raw) == "topology_affine_q10") {
        return ExteriorNormalRestrictMode3D::
            SharedQ10CubicGridlineCauchy;
    }
    if (std::string(raw) == "q27_cover3_all_event_cauchy"
        || std::string(raw) == "q27_cover3"
        || std::string(raw) == "topology_affine_q27") {
        return ExteriorNormalRestrictMode3D::Q27Cover3AllEventCauchy;
    }
    if (std::string(raw) == "q64_cover4_all_event_cauchy"
        || std::string(raw) == "q64_cover4"
        || std::string(raw) == "topology_affine_q64") {
        return ExteriorNormalRestrictMode3D::Q64Cover4AllEventCauchy;
    }
    throw std::invalid_argument(
        "KFBIM_3D_NEUMANN_TRACE_RESTRICT must be "
        "joint_tricubic_crossing_owner (or crossing_owner), "
        "global_cubic_exterior_branch_crossing_owner, or "
        "shared_quadratic_gridline_cauchy (or shared_quadratic), "
        "shared_q10_cubic_gridline_cauchy, q27_cover3_all_event_cauchy, "
        "q64_cover4_all_event_cauchy, or "
        "joint_tricubic_cauchy (or legacy)");
}

bool trace_restrict_uses_crossing_owner(ExteriorNormalRestrictMode3D mode)
{
    return mode == ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner
        || mode == ExteriorNormalRestrictMode3D::
               GlobalCubicExteriorBranchCrossingOwner;
}

bool trace_restrict_uses_shared_quadratic(
    ExteriorNormalRestrictMode3D mode)
{
    return mode == ExteriorNormalRestrictMode3D::
               SharedQuadraticGridlineCauchy
        || mode == ExteriorNormalRestrictMode3D::
               SharedQ10CubicGridlineCauchy;
}

bool trace_restrict_uses_tensor_cover(
    ExteriorNormalRestrictMode3D mode)
{
    return mode == ExteriorNormalRestrictMode3D::Q27Cover3AllEventCauchy
        || mode == ExteriorNormalRestrictMode3D::Q64Cover4AllEventCauchy;
}

bool trace_restrict_uses_all_event_path(
    ExteriorNormalRestrictMode3D mode)
{
    return trace_restrict_uses_shared_quadratic(mode)
        || trace_restrict_uses_tensor_cover(mode);
}

bool trace_restrict_uses_topology_affine_cubic(
    ExteriorNormalRestrictMode3D mode)
{
    return mode == ExteriorNormalRestrictMode3D::
        SharedQ10CubicGridlineCauchy;
}

ExteriorNormalRestrictMode3D selected_dirichlet_normal_restrict_mode()
{
    const char* raw =
        std::getenv("KFBIM_3D_DIRICHLET_NORMAL_RESTRICT");
    if (raw == nullptr || std::string(raw).empty()) {
#ifdef KFBIM_3D_TOPOLOGY_AFFINE_DEFAULT
        return ExteriorNormalRestrictMode3D::
            Q64Cover4AllEventCauchy;
#else
        return ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner;
#endif
    }
    if (std::string(raw) == "joint_tricubic_crossing_owner"
        || std::string(raw) == "crossing_owner") {
        return ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner;
    }
    if (std::string(raw) == "joint_tricubic_cauchy"
        || std::string(raw) == "legacy") {
        return ExteriorNormalRestrictMode3D::JointTricubicCauchy;
    }
    if (std::string(raw) == "shared_quadratic_gridline_cauchy"
        || std::string(raw) == "shared_quadratic"
        || std::string(raw) == "quadratic_gridline") {
        return ExteriorNormalRestrictMode3D::
            SharedQuadraticGridlineCauchy;
    }
    if (std::string(raw) == "shared_q10_cubic_gridline_cauchy"
        || std::string(raw) == "shared_q10_cubic"
        || std::string(raw) == "topology_affine_q10") {
        return ExteriorNormalRestrictMode3D::
            SharedQ10CubicGridlineCauchy;
    }
    if (std::string(raw) == "q27_cover3_all_event_cauchy"
        || std::string(raw) == "q27_cover3"
        || std::string(raw) == "topology_affine_q27") {
        return ExteriorNormalRestrictMode3D::Q27Cover3AllEventCauchy;
    }
    if (std::string(raw) == "q64_cover4_all_event_cauchy"
        || std::string(raw) == "q64_cover4"
        || std::string(raw) == "topology_affine_q64") {
        return ExteriorNormalRestrictMode3D::Q64Cover4AllEventCauchy;
    }
    throw std::invalid_argument(
        "KFBIM_3D_DIRICHLET_NORMAL_RESTRICT must be "
        "joint_tricubic_crossing_owner (or crossing_owner) or "
        "shared_quadratic_gridline_cauchy (or shared_quadratic), "
        "shared_q10_cubic_gridline_cauchy, q27_cover3_all_event_cauchy, "
        "q64_cover4_all_event_cauchy, or "
        "joint_tricubic_cauchy (or legacy)");
}

SolveSelection3D selected_solve_selection()
{
    const char* raw = std::getenv("KFBIM_3D_SOLVE_SELECTION");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "both") {
        return SolveSelection3D::Both;
    }
    if (std::string(raw) == "neumann_only")
        return SolveSelection3D::NeumannOnly;
    if (std::string(raw) == "dirichlet_normal_only")
        return SolveSelection3D::DirichletNormalOnly;
    throw std::invalid_argument(
        "KFBIM_3D_SOLVE_SELECTION must be both, neumann_only, "
        "or dirichlet_normal_only");
}

const char* solve_selection_name(SolveSelection3D selection)
{
    switch (selection) {
    case SolveSelection3D::Both:
        return "both";
    case SolveSelection3D::NeumannOnly:
        return "neumann_only";
    case SolveSelection3D::DirichletNormalOnly:
        return "dirichlet_normal_only";
    }
    throw std::runtime_error("unknown 3D solve selection");
}

const char* density_iteration_mode_name(DensityIterationMode3D mode)
{
    switch (mode) {
    case DensityIterationMode3D::SurfaceSamples:
        return "surface_samples";
    case DensityIterationMode3D::ReducedCoefficients:
        return "reduced_coefficients";
    }
    throw std::runtime_error("unknown 3D density iteration mode");
}

DensityIterationMode3D selected_density_iteration_mode()
{
    const char* raw = std::getenv("KFBIM_3D_DENSITY_MODE");
    if (raw == nullptr || std::string(raw).empty()) {
#ifdef KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT
        return DensityIterationMode3D::ReducedCoefficients;
#else
        return DensityIterationMode3D::SurfaceSamples;
#endif
    }
    if (std::string(raw) == "coefficient"
        || std::string(raw) == "coefficients"
        || std::string(raw) == "reduced_coefficients") {
        return DensityIterationMode3D::ReducedCoefficients;
    }
    if (std::string(raw) == "surface_samples"
        || std::string(raw) == "point_dofs") {
        return DensityIterationMode3D::SurfaceSamples;
    }
    throw std::invalid_argument(
        "KFBIM_3D_DENSITY_MODE must be reduced_coefficients or "
        "surface_samples");
}

app3d::DirichletRigidStudyCase3D selected_ordinary_rigid_case()
{
    const char* raw = std::getenv("KFBIM_3D_RIGID_CASE");
    const std::string requested =
        raw == nullptr || std::string(raw).empty() ? "baseline" : raw;
    const std::vector<app3d::DirichletRigidStudyCase3D> cases =
        app3d::make_l_prism_dirichlet_rigid_study_cases_3d();
    const auto found = std::find_if(
        cases.begin(), cases.end(), [&](const auto& study_case) {
            return study_case.id == requested;
        });
    if (found != cases.end())
        return *found;

    std::ostringstream message;
    message << "KFBIM_3D_RIGID_CASE must name one of";
    for (const auto& study_case : cases)
        message << ' ' << study_case.id;
    throw std::invalid_argument(message.str());
}

int optional_density_coefficient_override()
{
    const char* raw = std::getenv("KFBIM_3D_DENSITY_COEFFICIENTS");
    if (raw == nullptr || std::string(raw).empty())
        return 0;
    const std::string text(raw);
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "KFBIM_3D_DENSITY_COEFFICIENTS must be an integer >= 4");
    }
    if (consumed != text.size() || value < 4)
        throw std::invalid_argument(
            "KFBIM_3D_DENSITY_COEFFICIENTS must be an integer >= 4");
    return value;
}

int positive_environment_integer(const char* name, int default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr)
        return default_value;

    const std::string text(raw);
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name)
                                    + " must be a positive integer");
    }
    if (consumed != text.size() || value <= 0) {
        throw std::invalid_argument(std::string(name)
                                    + " must be a positive integer");
    }
    return value;
}

double positive_environment_double(const char* name, double default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr)
        return default_value;

    const std::string text(raw);
    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name)
                                    + " must be a positive finite number");
    }
    if (consumed != text.size() || !std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(std::string(name)
                                    + " must be a positive finite number");
    }
    return value;
}

geometry3d::PathCertificationBudget3D selected_native_endpoint_budget()
{
    geometry3d::PathCertificationBudget3D budget;
    budget.max_candidate_regions = positive_environment_integer(
        "KFBIM_3D_NATIVE_PATH_MAX_CANDIDATES",
        static_cast<int>(budget.max_candidate_regions));
    budget.max_subdivision_nodes = positive_environment_integer(
        "KFBIM_3D_NATIVE_PATH_MAX_BOXES",
        static_cast<int>(budget.max_subdivision_nodes));
    budget.max_newton_steps = positive_environment_integer(
        "KFBIM_3D_NATIVE_PATH_MAX_NEWTON",
        static_cast<int>(budget.max_newton_steps));
    budget.max_precision_bits = static_cast<unsigned>(positive_environment_integer(
        "KFBIM_3D_NATIVE_PATH_MAX_BITS", static_cast<int>(budget.max_precision_bits)));
    return budget;
}

struct GeometryBundle {
    std::string name;
    std::string description;
    NativeNurbsSurface3D native_surface;
    Interface3D correction_interface;
    Interface3D crossing_interface;
    std::vector<geometry3d::NurbsParamTriangle3D> correction_triangles;
    std::vector<geometry3d::NurbsParamTriangle3D> geometry_triangles;
    std::vector<geometry3d::NurbsPatchGeometricEdge3D>
        feature_edge_segments;
    int feature_edges = 0;
    int feature_vertices = 0;
    std::function<bool(const Eigen::Vector3d&)> exact_inside;
};

struct SurfaceCloudDiagnostics {
    double area = 0.0;
    double area_relative_error = 0.0;
    double normal_error = 0.0;
    double nearest_spacing_min_over_h = 0.0;
    double nearest_spacing_mean_over_h = 0.0;
    double nearest_spacing_max_over_h = 0.0;
};

struct CauchyStencil {
    std::vector<int> value_ids;
    std::vector<int> derivative_ids;
    double radius_over_h = 0.0;
    int incident_patch_count = 0;
    int value_patch_imbalance = 0;
    int derivative_patch_imbalance = 0;
};

struct CauchyStencilSet {
    CauchyStencilPolicy3D policy = CauchyStencilPolicy3D::G1Nearest;
    int value_count = 0;
    int derivative_count = 0;
    int value_count_min = 0;
    int value_count_max = 0;
    int derivative_count_min = 0;
    int derivative_count_max = 0;
    std::vector<CauchyStencil> rows;
    double radius_max_over_h = 0.0;
    double radius_mean_over_h = 0.0;
    int incident_patch_count_min = 0;
    int incident_patch_count_max = 0;
    int value_patch_imbalance_max = 0;
    int derivative_patch_imbalance_max = 0;
};

struct SharedQuadraticRestrictDiagnostics3D {
    std::size_t side_plans = 0;
    std::size_t unique_gridline_crossings = 0;
    std::size_t wrong_side_nodes = 0;
    std::size_t segment_queries = 0;
    std::size_t multiple_root_selections = 0;
    std::size_t trace_endpoint_selections = 0;
    std::size_t feature_root_selections = 0;
    std::size_t unreliable_root_selections = 0;
    std::size_t segment_fallbacks = 0;
    std::size_t no_g1_gridline_fallbacks = 0;
    std::size_t gridline_ties = 0;
    double interpolation_condition_max = 0.0;
    double gridline_distance_max_over_h = 0.0;
    double cauchy_condition_max = 0.0;
};

struct SolveMetrics3D {
    std::string formulation;
    std::string compatibility_mode;
    std::string trace_restrict_mode = "not_applicable";
    std::string restrict_value_jet = "not_applicable";
    std::string trace_sampling = "not_applicable";
    std::string density_coordinates = "not_applicable";
    std::string border_solver = "not_applicable";
    std::string edge_jump_jet = "disabled";
    std::string known_value_jump = "not_applicable";
    std::string dirichlet_jump_space = "not_applicable";
    int edge_feature_edges = 0;
    int edge_constraint_rows = 0;
    int edge_constraint_rank = 0;
    int edge_reduced_dofs = 0;
    double edge_constraint_residual_linf = 0.0;
    double edge_nullspace_residual_linf = 0.0;
    double feature_vertex_residual_linf = 0.0;
    double feature_edge_residual_linf = 0.0;
    double feature_discarded_residual_linf = 0.0;
    double trace_projection_leakage_relative_l2 = 0.0;
    double edge_normal_fit_linf = 0.0;
    double edge_target_projection_linf = 0.0;
    double feature_target_requested_linf = 0.0;
    double feature_target_requested_l2 = 0.0;
    double feature_target_projected_linf = 0.0;
    double feature_target_projected_l2 = 0.0;
    double feature_target_projection_l2 = 0.0;
    double feature_target_projection_relative_l2 = 0.0;
    double feature_target_reachable_certification_linf = 0.0;
    int feature_target_projection_components = 0;
    bool feature_target_constraints_exact = true;
    std::string reduction_scheme = "legacy_native";
    std::string particular_solver = "not_applicable";
    int topology_constraint_rows = 0;
    int topology_constraint_rank = 0;
    // c_full = A0 c_base and c_base = p + E z.
    int topology_base_coordinates = 0;
    int topology_full_coordinates = 0;
    int topology_pre_mean_coordinates = 0;
    int topology_reduced_coordinates = 0;
    int topology_block_count = 0;
    double topology_particular_residual_linf = 0.0;
    double topology_homogeneous_residual_linf = 0.0;
    int topology_projector_rank = 0;
    double topology_projector_condition = 0.0;
    double topology_projector_pb_error = 0.0;
    int topology_trace_samples = 0;
    int topology_trace_oversampling_margin = 0;
    double topology_trace_oversampling_ratio = 0.0;
    double topology_trace_closure_linf = 0.0;
    int mean_free_pivot_index = -1;
    double mean_free_pivot_moment = 0.0;
    double mean_free_relative_observability = 0.0;
    double mean_free_coordinate_condition = 1.0;
    double mean_free_particular_residual = 0.0;
    double mean_free_homogeneous_residual = 0.0;
    double mean_free_constant_projection_linf = 0.0;
    double mean_free_intrinsic_mean = 0.0;
    std::vector<app3d::AffineEliminationRecord3D> topology_block_records;
    int iterations = 0;
    bool converged = false;
    bool physical_converged = false;
    double seconds = 0.0;
    double gmres_tolerance = 0.0;
    double gmres_relative_residual = 0.0;
    double operator_residual_linf = 0.0;
    double exterior_condition_linf = 0.0;
    double lagrange_multiplier = 0.0;
    double operator_compatibility_correction = 0.0;
    double total_flux_correction = 0.0;
    double border_column_linf = 0.0;
    double exterior_trace_weighted_mean = 0.0;
    double exterior_trace_demeaned_linf = 0.0;
    double augmented_trace_closure_linf = 0.0;
    double compatibility_residual = 0.0;
    double boundary_residual_linf = 0.0;
    double route_mismatch_linf = 0.0;
    double density_linf = 0.0;
    double density_l2 = 0.0;
    double interior_linf = 0.0;
    double interior_l2 = 0.0;
    double exterior_bulk_linf = 0.0;
    double exterior_bulk_l2 = 0.0;
    double compatibility_mean_removed = 0.0;
    double data_weighted_mean = 0.0;
    double density_weighted_mean = 0.0;
    double constant_shift = 0.0;
};

struct ReadinessResult {
    std::string geometry;
    std::string cauchy_policy;
    std::string cauchy_value_trace_fit;
    int N = 0;
    double h = 0.0;
    int correction_panels = 0;
    int correction_dofs = 0;
    int crossing_panels = 0;
    int feature_edges = 0;
    int feature_vertices = 0;
    int interior_nodes = 0;
    int exterior_nodes = 0;
    int label_mismatches = 0;
    int crossings = 0;
    int exact_crossings = 0;
    int gap_crossings = 0;
    int endpoint_crossings = 0;
    int nurbs_patches = 0;
    int bezier_elements = 0;
    int acceleration_leaves = 0;
    double maximum_query_element_extent = 0.0;
    std::size_t candidate_grid_edges = 0;
    int triangle_seed_hits = 0;
    int triangle_seed_misses_recovered = 0;
    int subdivision_boxes = 0;
    int newton_attempts = 0;
    int newton_iterations = 0;
    int maximum_subdivision_depth = 0;
    int terminal_certificate_boxes = 0;
    int maximum_terminal_certificate_depth = 0;
    int closest_point_attempts = 0;
    int closest_point_iterations = 0;
    int closest_point_roots_recovered = 0;
    int closest_point_terminal_misses = 0;
    int closest_point_failures = 0;
    int seam_deduplications = 0;
    int sample_seed_candidates = 0;
    int sample_seeds_accepted = 0;
    int sample_seed_roots_recovered = 0;
    int maximum_sample_seeds_per_element = 0;
    int stationary_solve_attempts = 0;
    int stationary_solve_converged = 0;
    int stationary_witnesses = 0;
    int stationary_protected_root_pairs = 0;
    int ambiguous_root_clusters = 0;
    int non_g1_topology_merges = 0;
    int high_degree_fallbacks = 0;
    std::size_t interface_x = 0;
    std::size_t interface_y = 0;
    std::size_t interface_z = 0;
    std::size_t multi_crossing_edges = 0;
    std::size_t even_parity_interface_edges = 0;
    std::size_t odd_parity_interface_edges = 0;
    std::size_t ambiguous_parity_edges = 0;
    std::size_t ambiguous_label_changing_edges = 0;
    std::size_t targeted_retries = 0;
    std::size_t targeted_retries_resolved = 0;
    std::size_t targeted_retries_unsafe = 0;
    std::size_t correction_safe_edges = 0;
    std::size_t unsafe_label_changing_edges = 0;
    int maximum_targeted_retry_subdivision_depth = 0;
    std::size_t endpoint_parity_fallbacks = 0;
    std::size_t endpoint_classification_queries = 0;
    std::size_t component_parity_toggles = 0;
    std::size_t barrier_x = 0;
    std::size_t barrier_y = 0;
    std::size_t barrier_z = 0;
    int grid_components = 0;
    int box_exterior_components = 0;
    int representative_queries = 0;
    double nurbs_geometry_tolerance = 0.0;
    double nurbs_root_residual_max = 0.0;
    int triangle_fallback_crossings = 0;
    int correction_nodes = 0;
    double correction_area = 0.0;
    double crossing_area = 0.0;
    double normal_error = 0.0;
    double min_box_margin_over_h = 0.0;
    double constant_A1_linf = 0.0;
    double constant_exterior_trace_linf = 0.0;
    double constant_interior_trace_linf = 0.0;
    double constant_bulk_linf = 0.0;
    double harmonic_constant_exterior_trace_linf = 0.0;
    double harmonic_constant_interior_trace_linf = 0.0;
    double harmonic_constant_exterior_normal_linf = 0.0;
    double harmonic_constant_interior_normal_linf = 0.0;
    double harmonic_constant_bulk_linf = 0.0;
    int surface_patches = 0;
    int surface_dofs = 0;
    double surface_dof_area = 0.0;
    double surface_area_relative_error = 0.0;
    double surface_spacing_min_over_h = 0.0;
    double surface_spacing_mean_over_h = 0.0;
    double surface_spacing_max_over_h = 0.0;
    std::string density_iteration_mode = "surface_samples";
    bool automatic_density = false;
    int density_coefficients_per_direction = 0;
    int density_observability_cap = 0;
    double density_orientation_integral = 0.0;
    double density_predicted_crossings = 0.0;
    double density_target_crossings_per_cell = 0.0;
    int value_density_raw_coefficients = 0;
    int value_density_c0_coefficients = 0;
    int value_density_reduced_coefficients = 0;
    int normal_density_raw_coefficients = 0;
    int normal_density_c0_coefficients = 0;
    int normal_density_reduced_coefficients = 0;
    int value_density_c0_seams = 0;
    int value_density_c1_seams = 0;
    int normal_density_broken_seams = 0;
    int normal_density_c1_seams = 0;
    double value_density_constraint_residual = 0.0;
    double normal_density_constraint_residual = 0.0;
    double value_density_constant_error = 0.0;
    double normal_density_constant_error = 0.0;
    double value_projection_condition = 0.0;
    double normal_projection_condition = 0.0;
    double value_projection_pb_error = 0.0;
    double normal_projection_pb_error = 0.0;
    double value_coordinate_condition = 0.0;
    double normal_coordinate_condition = 0.0;
    int cauchy_value_neighbors = 0;
    int cauchy_derivative_neighbors = 0;
    int cauchy_value_neighbors_min = 0;
    int cauchy_value_neighbors_max = 0;
    int cauchy_derivative_neighbors_min = 0;
    int cauchy_derivative_neighbors_max = 0;
    double cauchy_radius_max_over_h = 0.0;
    double cauchy_radius_mean_over_h = 0.0;
    int cauchy_incident_patches_min = 0;
    int cauchy_incident_patches_max = 0;
    int cauchy_value_patch_imbalance_max = 0;
    int cauchy_derivative_patch_imbalance_max = 0;
    double cauchy_condition_median = 0.0;
    double cauchy_condition_p95 = 0.0;
    double cauchy_condition_max = 0.0;
    double crossing_cauchy_condition_median = 0.0;
    double crossing_cauchy_condition_p95 = 0.0;
    double crossing_cauchy_condition_max = 0.0;
    std::size_t crossing_cauchy_unique_plans = 0;
    std::size_t crossing_cauchy_svd_count = 0;
    SharedQuadraticRestrictDiagnostics3D shared_quadratic_restrict;
    int feature_trace_segments = 0;
    int feature_trace_vertices = 0;
    int feature_trace_active_maps = 0;
    int feature_trace_edge_maps = 0;
    int feature_trace_vertex_maps = 0;
    int feature_trace_maximum_features_per_map = 0;
    int feature_trace_maximum_sheets_per_map = 0;
    double feature_trace_constraint_residual_max = 0.0;
    double feature_trace_condition_median = 0.0;
    double feature_trace_condition_p95 = 0.0;
    double feature_trace_condition_max = 0.0;
    SolveMetrics3D neumann;
    SolveMetrics3D dirichlet_normal;
};

bool is_power_of_two(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

GeometryKind parse_geometry(const std::string& name)
{
    if (name == "torus" || name == "ring")
        return GeometryKind::Torus;
    if (name == "cylinder" || name == "hollow_cylinder"
        || name == "pipe" || name == "capped_cylinder")
        return GeometryKind::HollowCylinder;
    if (name == "l_prism" || name == "l-prism" || name == "lprism")
        return GeometryKind::LPrism;
    if (name == "u_prism" || name == "u-prism" || name == "uprism")
        return GeometryKind::UPrism;
    throw std::invalid_argument(
        "geometry must be torus, cylinder, l_prism, u_prism, or all");
}

SurfaceCloudDiagnostics validate_surface_dofs(const SurfaceDofCloud& cloud,
                                              double h)
{
    if (cloud.patches.empty() || cloud.dofs.size() < 2)
        throw std::runtime_error("surface panel-center cloud is empty");
    std::vector<int> patch_counts(cloud.patches.size(), 0);
    SurfaceCloudDiagnostics result;
    for (std::size_t i = 0; i < cloud.patches.size(); ++i) {
        const SurfacePatchInfo& patch = cloud.patches[i];
        if (patch.smooth_patch_ids.empty()
            || std::find(patch.smooth_patch_ids.begin(),
                         patch.smooth_patch_ids.end(),
                         static_cast<int>(i)) == patch.smooth_patch_ids.end()) {
            throw std::runtime_error(
                "surface patch smooth adjacency must include itself");
        }
        for (int adjacent : patch.smooth_patch_ids) {
            if (adjacent < 0 || adjacent >= static_cast<int>(cloud.patches.size()))
                throw std::runtime_error("surface patch has invalid adjacency");
        }
    }
    for (const SurfaceDof& dof : cloud.dofs) {
        if (dof.patch_id < 0
            || dof.patch_id >= static_cast<int>(cloud.patches.size()))
            throw std::runtime_error("surface DOF has invalid patch ownership");
        if (!dof.point.allFinite() || !dof.normal.allFinite()
            || !std::isfinite(dof.weight) || !(dof.weight > 0.0))
            throw std::runtime_error("surface DOF contains NaN/Inf");
        ++patch_counts[static_cast<std::size_t>(dof.patch_id)];
        result.area += dof.weight;
        result.normal_error = std::max(
            result.normal_error, std::abs(dof.normal.norm() - 1.0));
    }
    if (std::find(patch_counts.begin(), patch_counts.end(), 0) != patch_counts.end())
        throw std::runtime_error("at least one surface patch has no center DOF");
    result.area_relative_error = std::abs(result.area - cloud.expected_area)
                               / cloud.expected_area;

    double spacing_sum = 0.0;
    double spacing_min = std::numeric_limits<double>::infinity();
    double spacing_max = 0.0;
    for (std::size_t i = 0; i < cloud.dofs.size(); ++i) {
        double nearest_sq = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < cloud.dofs.size(); ++j) {
            if (i == j)
                continue;
            nearest_sq = std::min(
                nearest_sq,
                (cloud.dofs[i].point - cloud.dofs[j].point).squaredNorm());
        }
        const double spacing = std::sqrt(nearest_sq) / h;
        spacing_sum += spacing;
        spacing_min = std::min(spacing_min, spacing);
        spacing_max = std::max(spacing_max, spacing);
    }
    result.nearest_spacing_min_over_h = spacing_min;
    result.nearest_spacing_mean_over_h = spacing_sum
        / static_cast<double>(cloud.dofs.size());
    result.nearest_spacing_max_over_h = spacing_max;
    return result;
}

std::vector<int> select_cauchy_dofs(CauchyStencilPolicy3D policy,
                                    const NativeNurbsSurface3D& surface,
                                    const SurfaceDofCloud& cloud,
                                    int center,
                                    int count)
{
    switch (policy) {
    case CauchyStencilPolicy3D::G1Nearest:
        return app3d::nearest_g1_cauchy_dofs(
            surface, cloud, center, count);
    case CauchyStencilPolicy3D::TopologicalNearest:
        return app3d::nearest_topological_cauchy_dofs(
            surface, cloud, center, count);
    case CauchyStencilPolicy3D::SamePatch:
        return app3d::nearest_same_patch_cauchy_dofs(cloud, center, count);
    case CauchyStencilPolicy3D::BalancedPatches:
        return app3d::balanced_topological_cauchy_dofs(
            surface, cloud, center, count);
    }
    throw std::runtime_error("unknown Cauchy stencil selection policy");
}

int selected_patch_imbalance(const SurfaceDofCloud& cloud,
                             const std::vector<int>& ids)
{
    std::map<int, int> counts;
    for (int q : ids)
        ++counts[cloud.dofs[static_cast<std::size_t>(q)].patch_id];
    if (counts.empty())
        return 0;
    int minimum = std::numeric_limits<int>::max();
    int maximum = 0;
    for (const auto& item : counts) {
        minimum = std::min(minimum, item.second);
        maximum = std::max(maximum, item.second);
    }
    return maximum - minimum;
}

CauchyStencilSet build_cauchy_stencils(const NativeNurbsSurface3D& surface,
                                       const SurfaceDofCloud& cloud,
                                       double h,
                                       int requested_value_count,
                                       int requested_derivative_count,
                                       CauchyStencilPolicy3D policy)
{
    if (requested_value_count <= 0 || requested_derivative_count < 0
        || requested_derivative_count > requested_value_count) {
        throw std::invalid_argument("invalid Cauchy stencil sizes");
    }

    CauchyStencilSet result;
    result.policy = policy;
    result.value_count = requested_value_count;
    result.derivative_count = requested_derivative_count;
    result.value_count_min = std::numeric_limits<int>::max();
    result.derivative_count_min = std::numeric_limits<int>::max();
    result.rows.reserve(cloud.dofs.size());
    result.incident_patch_count_min = std::numeric_limits<int>::max();

    for (int center = 0; center < static_cast<int>(cloud.dofs.size()); ++center) {
        const SurfaceDof& target = cloud.dofs[static_cast<std::size_t>(center)];
        CauchyStencil stencil;
        stencil.value_ids = select_cauchy_dofs(
            policy, surface, cloud, center, result.value_count);
        stencil.derivative_ids = select_cauchy_dofs(
            policy, surface, cloud, center, result.derivative_count);
        if (std::find(stencil.value_ids.begin(), stencil.value_ids.end(), center)
            == stencil.value_ids.end()) {
            throw std::runtime_error("Cauchy value stencil does not contain its center");
        }
        if (std::find(stencil.derivative_ids.begin(),
                      stencil.derivative_ids.end(), center)
            == stencil.derivative_ids.end()) {
            throw std::runtime_error(
                "Cauchy normal stencil does not contain its center");
        }
        double radius_sq = 0.0;
        for (int q : stencil.value_ids) {
            radius_sq = std::max(
                radius_sq,
                (cloud.dofs[static_cast<std::size_t>(q)].point - target.point)
                    .squaredNorm());
        }
        for (int q : stencil.derivative_ids) {
            radius_sq = std::max(
                radius_sq,
                (cloud.dofs[static_cast<std::size_t>(q)].point - target.point)
                    .squaredNorm());
        }
        stencil.radius_over_h = std::sqrt(radius_sq) / h;
        std::set<int> selected_patches;
        for (int q : stencil.value_ids)
            selected_patches.insert(cloud.dofs[static_cast<std::size_t>(q)].patch_id);
        for (int q : stencil.derivative_ids)
            selected_patches.insert(cloud.dofs[static_cast<std::size_t>(q)].patch_id);
        stencil.incident_patch_count = static_cast<int>(selected_patches.size());
        stencil.value_patch_imbalance =
            selected_patch_imbalance(cloud, stencil.value_ids);
        stencil.derivative_patch_imbalance =
            selected_patch_imbalance(cloud, stencil.derivative_ids);

        result.radius_max_over_h = std::max(
            result.radius_max_over_h, stencil.radius_over_h);
        result.radius_mean_over_h += stencil.radius_over_h;
        result.incident_patch_count_min = std::min(
            result.incident_patch_count_min, stencil.incident_patch_count);
        result.incident_patch_count_max = std::max(
            result.incident_patch_count_max, stencil.incident_patch_count);
        result.value_count_min = std::min(
            result.value_count_min, static_cast<int>(stencil.value_ids.size()));
        result.value_count_max = std::max(
            result.value_count_max, static_cast<int>(stencil.value_ids.size()));
        result.derivative_count_min = std::min(
            result.derivative_count_min,
            static_cast<int>(stencil.derivative_ids.size()));
        result.derivative_count_max = std::max(
            result.derivative_count_max,
            static_cast<int>(stencil.derivative_ids.size()));
        result.value_patch_imbalance_max = std::max(
            result.value_patch_imbalance_max, stencil.value_patch_imbalance);
        result.derivative_patch_imbalance_max = std::max(
            result.derivative_patch_imbalance_max,
            stencil.derivative_patch_imbalance);
        result.rows.push_back(std::move(stencil));
    }
    result.radius_mean_over_h /= static_cast<double>(result.rows.size());
    return result;
}

using app3d::HarmonicPolynomialSpace3D;
using app3d::svd_pseudoinverse_3d;
using app3d::svd_pseudoinverse_from_decomposition_3d;

struct CauchyFitMap3D {
    Eigen::MatrixXd value_map;
    Eigen::MatrixXd normal_map;
    double condition = 0.0;
};

class PanelCenterCauchyFit3D {
public:
    PanelCenterCauchyFit3D(const NativeNurbsSurface3D& surface,
                           const SurfaceDofCloud& cloud,
                           const CauchyStencilSet& stencils,
                           double h,
                           int degree = 4,
                           CauchyValueTraceFitMode3D value_trace_mode =
                               CauchyValueTraceFitMode3D::GeometricFeature,
                           int local_value_count = 0,
                           int local_normal_count = 0,
                           bool build_maps = true)
        : cloud_(cloud)
        , stencils_(stencils)
        , h_(h)
        , space_(degree)
        , maps_enabled_(build_maps)
    {
        if (stencils_.rows.size() != cloud_.dofs.size())
            throw std::invalid_argument("Cauchy stencil count does not match surface DOFs");
        if ((local_value_count == 0) != (local_normal_count == 0)
            || local_value_count < 0 || local_normal_count < 0) {
            throw std::invalid_argument(
                "local restrict Cauchy counts must both be zero or positive");
        }
        if (local_value_count > 0
            && local_value_count + local_normal_count < dimension()) {
            throw std::invalid_argument(
                "local restrict Cauchy fit has too few conditions");
        }
        local_value_count_ = local_value_count;
        local_normal_count_ = local_normal_count;
        center_cauchy_constraints_.resize(2, dimension());
        center_cauchy_constraints_.row(0) =
            space_.basis(0.0, 0.0, 0.0).transpose();
        center_cauchy_constraints_.row(1) =
            space_.gradient(0.0, 0.0, 0.0).row(2);
        const Eigen::Matrix2d center_cauchy_gram =
            center_cauchy_constraints_
            * center_cauchy_constraints_.transpose();
        if (!center_cauchy_gram.allFinite()
            || std::abs(center_cauchy_gram.determinant()) < 1.0e-13) {
            throw std::runtime_error(
                "center Cauchy constraint modes are rank deficient");
        }
        center_cauchy_right_inverse_ =
            center_cauchy_constraints_.transpose()
            * center_cauchy_gram.inverse();
        if ((center_cauchy_constraints_ * center_cauchy_right_inverse_
             - Eigen::Matrix2d::Identity()).cwiseAbs().maxCoeff()
            > 2.0e-13) {
            throw std::runtime_error(
                "center Cauchy constraint right inverse is inaccurate");
        }
        if (maps_enabled_) {
            maps_.reserve(cloud_.dofs.size());
            local_maps_.reserve(
                local_value_count_ > 0 ? cloud_.dofs.size() : 0);
            for (int center = 0;
                 center < static_cast<int>(cloud_.dofs.size());
                 ++center) {
                maps_.push_back(build_map(center));
                if (local_value_count_ > 0) {
                    local_maps_.push_back(build_map(
                        center, local_value_count_, local_normal_count_));
                }
            }
            if (value_trace_mode
                == CauchyValueTraceFitMode3D::GeometricFeature) {
                feature_trace_fit_ =
                    std::make_unique<app3d::GeometricFeatureTraceFit3D>(
                        surface, cloud, h, degree);
            }
        }
    }

    int dimension() const { return space_.dimension(); }
    bool maps_enabled() const noexcept { return maps_enabled_; }
    std::vector<double> condition_values() const
    {
        std::vector<double> result;
        result.reserve(maps_.size());
        for (const CauchyFitMap3D& map : maps_)
            result.push_back(map.condition);
        return result;
    }
    std::vector<double> local_condition_values() const
    {
        std::vector<double> result;
        result.reserve(local_maps_.size());
        for (const CauchyFitMap3D& map : local_maps_)
            result.push_back(map.condition);
        return result;
    }
    const HarmonicPolynomialSpace3D& space() const { return space_; }
    app3d::GeometricFeatureTraceSummary3D feature_trace_summary() const
    {
        return feature_trace_fit_
            ? feature_trace_fit_->summary()
            : app3d::GeometricFeatureTraceSummary3D{};
    }
    std::vector<double> feature_trace_condition_values() const
    {
        return feature_trace_fit_
            ? feature_trace_fit_->condition_values()
            : std::vector<double>{};
    }

    Eigen::MatrixXd coefficients(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        bool use_feature_value_jet = true,
        bool anchor_center_cauchy = false,
        bool use_local_restrict_fit = false) const
    {
        const int size = static_cast<int>(cloud_.dofs.size());
        if (value_jump.size() != size || normal_jump.size() != size)
            throw std::invalid_argument("jump data size does not match surface DOFs");
        if (!maps_enabled_) {
            throw std::runtime_error(
                "legacy panel-centred Cauchy maps are disabled for the "
                "direct all-event cover pipeline");
        }
        Eigen::MatrixXd result(size, dimension());
        if (use_local_restrict_fit && local_maps_.empty()) {
            throw std::runtime_error(
                "local restrict Cauchy fit was not initialized");
        }
        for (int center = 0; center < size; ++center) {
            const CauchyStencil& stencil =
                stencils_.rows[static_cast<std::size_t>(center)];
            const int value_count = use_local_restrict_fit
                ? local_value_count_
                : static_cast<int>(stencil.value_ids.size());
            const int normal_count = use_local_restrict_fit
                ? local_normal_count_
                : static_cast<int>(stencil.derivative_ids.size());
            Eigen::VectorXd values(value_count);
            Eigen::VectorXd normals(normal_count);
            for (int k = 0; k < value_count; ++k)
                values[k] = value_jump[stencil.value_ids[static_cast<std::size_t>(k)]];
            for (int k = 0; k < normal_count; ++k)
                normals[k] = normal_jump[stencil.derivative_ids[static_cast<std::size_t>(k)]];
            const CauchyFitMap3D& map = (use_local_restrict_fit
                ? local_maps_ : maps_)[static_cast<std::size_t>(center)];
            Eigen::VectorXd value_coefficients;
            if (!use_local_restrict_fit && use_feature_value_jet
                && feature_trace_fit_
                && feature_trace_fit_->active(center)) {
                value_coefficients =
                    feature_trace_fit_->coefficients(center, value_jump);
            } else {
                value_coefficients = map.value_map * values;
            }
            Eigen::VectorXd coefficients =
                value_coefficients + map.normal_map * normals;
            if (anchor_center_cauchy) {
                // The local least-squares fit supplies tangential and higher
                // derivatives, while the density itself supplies the exact
                // zeroth and first normal Cauchy data.  Constant and z are
                // harmonic modes, so these two corrections preserve the
                // harmonic polynomial and leave every higher derivative
                // unchanged.
                Eigen::Vector2d target;
                target << value_jump[center], h_ * normal_jump[center];
                coefficients += center_cauchy_right_inverse_
                    * (target - center_cauchy_constraints_ * coefficients);
            }
            result.row(center) = coefficients.transpose();
        }
        return result;
    }

private:
    Eigen::Vector3d local_coordinate(const SurfaceDof& center,
                                     const Eigen::Vector3d& point) const
    {
        const Eigen::Vector3d displacement = (point - center.point) / h_;
        return {displacement.dot(center.tangent1),
                displacement.dot(center.tangent2),
                displacement.dot(center.normal)};
    }

    CauchyFitMap3D build_map(
        int center_id,
        int requested_value_count = 0,
        int requested_normal_count = 0) const
    {
        const SurfaceDof& center = cloud_.dofs[static_cast<std::size_t>(center_id)];
        const CauchyStencil& stencil =
            stencils_.rows[static_cast<std::size_t>(center_id)];
        const int value_count = requested_value_count > 0
            ? requested_value_count
            : static_cast<int>(stencil.value_ids.size());
        const int normal_count = requested_normal_count > 0
            ? requested_normal_count
            : static_cast<int>(stencil.derivative_ids.size());
        if (value_count > static_cast<int>(stencil.value_ids.size())
            || normal_count
                   > static_cast<int>(stencil.derivative_ids.size())) {
            throw std::invalid_argument(
                "local restrict Cauchy count exceeds the available stencil");
        }
        const int rows = value_count + normal_count;
        if (rows < dimension()) {
            throw std::runtime_error(
                "Cauchy fit has fewer conditions than harmonic coefficients");
        }
        Eigen::MatrixXd design(rows, dimension());
        Eigen::VectorXd sqrt_weights(rows);

        for (int k = 0; k < value_count; ++k) {
            const SurfaceDof& sample = cloud_.dofs[static_cast<std::size_t>(
                stencil.value_ids[static_cast<std::size_t>(k)])];
            const Eigen::Vector3d xi = local_coordinate(center, sample.point);
            design.row(k) = space_.basis(xi.x(), xi.y(), xi.z()).transpose();
            sqrt_weights[k] = std::sqrt(
                1.0 / std::pow(0.35 + xi.norm(), 2.0));
        }
        for (int k = 0; k < normal_count; ++k) {
            const SurfaceDof& sample = cloud_.dofs[static_cast<std::size_t>(
                stencil.derivative_ids[static_cast<std::size_t>(k)])];
            const Eigen::Vector3d xi = local_coordinate(center, sample.point);
            const Eigen::Vector3d normal_components(
                sample.normal.dot(center.tangent1),
                sample.normal.dot(center.tangent2),
                sample.normal.dot(center.normal));
            design.row(value_count + k) =
                normal_components.transpose()
                * space_.gradient(xi.x(), xi.y(), xi.z());
            sqrt_weights[value_count + k] = std::sqrt(
                0.85 / std::pow(0.35 + xi.norm(), 2.0));
        }

        const Eigen::MatrixXd weighted = sqrt_weights.asDiagonal() * design;
        Eigen::JacobiSVD<Eigen::MatrixXd> condition_svd(
            weighted, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorXd singular = condition_svd.singularValues();
        if (singular.size() != dimension() || singular.size() == 0
            || !singular.allFinite() || !(singular[0] > 0.0)
            || !(singular[singular.size() - 1] > 3.0e-12 * singular[0])) {
            throw std::runtime_error(
                "Cauchy fit is rank deficient at surface DOF "
                + std::to_string(center_id));
        }
        CauchyFitMap3D result;
        result.condition = singular[0] / singular[singular.size() - 1];
        const Eigen::MatrixXd pinv =
            svd_pseudoinverse_from_decomposition_3d(condition_svd, 3.0e-12);
        result.value_map.resize(dimension(), value_count);
        result.normal_map.resize(dimension(), normal_count);
        for (int k = 0; k < value_count; ++k)
            result.value_map.col(k) = pinv.col(k) * sqrt_weights[k];
        for (int k = 0; k < normal_count; ++k) {
            result.normal_map.col(k) =
                pinv.col(value_count + k) * sqrt_weights[value_count + k] * h_;
        }
        return result;
    }

    const SurfaceDofCloud& cloud_;
    const CauchyStencilSet& stencils_;
    double h_ = 0.0;
    HarmonicPolynomialSpace3D space_;
    bool maps_enabled_ = true;
    Eigen::MatrixXd center_cauchy_constraints_;
    Eigen::MatrixXd center_cauchy_right_inverse_;
    std::vector<CauchyFitMap3D> maps_;
    std::vector<CauchyFitMap3D> local_maps_;
    int local_value_count_ = 0;
    int local_normal_count_ = 0;
    std::unique_ptr<app3d::GeometricFeatureTraceFit3D>
        feature_trace_fit_;
};

GeometryBundle make_geometry(
    GeometryKind kind,
    double h,
    const app3d::RigidTransform3D& transform)
{
    const NativeNurbsSurface3D original_surface =
        app3d::make_native_nurbs_surface_3d(kind);
    NativeNurbsSurface3D native_surface =
        app3d::transform_native_nurbs_surface_3d(original_surface, transform);
    for (const auto& patch : native_surface.patches) {
        for (const auto& row : patch.control_net()) {
            for (const Eigen::Vector3d& point : row) {
                if (!point.allFinite()
                    || !(point.array().minCoeff() > kBoxMin)
                    || !(point.array().maxCoeff() < kBoxMin + kBoxSide)) {
                    throw std::runtime_error(
                        "transformed NURBS control point is outside the fixed box");
                }
            }
        }
    }
    geometry3d::NurbsPatchTriangulatorOptions3D options;
    options.edge_buffer_factor = 0.5;
    options.edge_sample_step_factor = 1.0;
    options.min_edge_samples = 4;
    options.edge_length_quadrature_samples = 16;
    options.H_factor = 2.0 * kTargetP2NodeSpacingOverH;
    options.max_edge_over_H = 1.10;
    options.metric_ratio_tolerance = 1.50;
    options.max_depth = 6;
    geometry3d::NurbsPatchTriangulation3D triangulation =
        geometry3d::triangulate_nurbs_surface_patches_3d(
            native_surface.patches, h, options);
    std::string name = native_surface.name;
    std::string description = native_surface.description;
    std::function<bool(const Eigen::Vector3d&)> exact_inside =
        native_surface.exact_inside;
    const int feature_edges = triangulation.summary.num_feature_edges;
    const int feature_vertices = triangulation.summary.num_feature_vertices;
    return {std::move(name),
            std::move(description),
            std::move(native_surface),
            std::move(triangulation.interface),
            std::move(triangulation.geometry_interface),
            std::move(triangulation.triangles),
            std::move(triangulation.geometry_triangles),
            std::move(triangulation.feature_edges),
            feature_edges,
            feature_vertices,
            std::move(exact_inside)};
}

Eigen::Vector3d grid_point(const CartesianGrid3D& grid, int node)
{
    const std::array<double, 3> x = grid.coord(node);
    return {x[0], x[1], x[2]};
}

std::vector<int> panel_nurbs_patch_indices(
    const std::vector<geometry3d::NurbsParamTriangle3D>& triangles)
{
    std::vector<int> result;
    result.reserve(triangles.size());
    for (const geometry3d::NurbsParamTriangle3D& triangle : triangles)
        result.push_back(triangle.patch_index);
    return result;
}

std::array<double, kRestrictGridStencilSize>
grid_lagrange_weights(double fraction)
{
    std::array<double, kRestrictGridStencilSize> weights{};
    weights.fill(1.0);
    for (int i = 0; i < kRestrictGridStencilSize; ++i) {
        const double node_i =
            static_cast<double>(i - kRestrictGridLeftOffset);
        for (int j = 0; j < kRestrictGridStencilSize; ++j) {
            if (i != j) {
                const double node_j =
                    static_cast<double>(j - kRestrictGridLeftOffset);
                weights[static_cast<std::size_t>(i)] *=
                    (fraction - node_j) / (node_i - node_j);
            }
        }
    }
    return weights;
}

double scalar_integer_power(double value, int power)
{
    double result = 1.0;
    for (int q = 0; q < power; ++q)
        result *= value;
    return result;
}

constexpr std::array<double, kRestrictNormalLayerCount>
make_normal_layers()
{
    std::array<double, kRestrictNormalLayerCount> layers{};
    for (int layer = 0; layer < kRestrictNormalLayerCount; ++layer) {
        layers[static_cast<std::size_t>(layer)] =
            static_cast<double>(2 * layer + 1) / 5.0;
    }
    return layers;
}

struct HarmonicJetField3D {
    Eigen::VectorXd potential;
    Eigen::MatrixXd coefficients;
    // Restrict-only representation of the value jump.  It is recovered
    // directly from the reduced density on a 3x3 patch-owned stencil at
    // radius 0.35h.  The normal jump remains in the separated harmonic
    // coefficients below, so the Cartesian spread is unchanged.
    Eigen::MatrixXd restrict_value_jets;
    Eigen::MatrixXd restrict_normal_coefficients;
    Eigen::VectorXd direct_value_c0;
    Eigen::VectorXd direct_normal_c0;
    app3d::KnownDirichletJetCallback3D direct_known_value_jet;
    app3d::KnownNeumannJetCallback3D direct_known_normal_jet;
    bool direct_coefficient_cauchy = false;
};

constexpr int kLocalRestrictSampleCount = 9;

struct LocalRestrictDensitySample3D {
    int patch = -1;
    double u = 0.0;
    double v = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
};

struct LocalRestrictCauchyPlan3D {
    app3d::LocalOrthonormalFrame3D frame;
    app3d::TangentGraphHessian3D graph_hessian;
    std::array<LocalRestrictDensitySample3D,
               kLocalRestrictSampleCount> samples;
    Eigen::Matrix<double, 6, kLocalRestrictSampleCount> value_recovery =
        Eigen::Matrix<double, 6, kLocalRestrictSampleCount>::Zero();
    double value_condition = 0.0;
};

struct HarmonicCrossingRow3D {
    int rhs_node = -1;
    int center_dof = -1;
    double scale = 0.0;
    Eigen::VectorXd evaluation;
};

// Crossing-centred Cauchy row used by the coefficient formulation.  The
// stored weights already include both the local harmonic recovery and its
// evaluation at the Cartesian correction node.  Consequently a spread row
// reads only the density basis values named by value_ids/normal_ids; it does
// not route the unknown through a panel-centre Cauchy polynomial.
struct HarmonicCrossingDirectRow3D {
    int rhs_node = -1;
    double scale = 0.0;
    std::vector<int> value_ids;
    std::vector<int> normal_ids;
    Eigen::VectorXd value_weights;
    Eigen::VectorXd normal_weights;
    double condition = 0.0;
};

// Transient per-undirected-edge plan.  The dense maps are used to compose
// the directed rows immediately and are not retained by the pipeline.
struct HarmonicCrossingDirectPlan3D {
    SurfaceDof frame;
    std::vector<int> value_ids;
    std::vector<int> normal_ids;
    Eigen::MatrixXd value_map;
    Eigen::MatrixXd normal_map;
    double condition = 0.0;
};

struct HarmonicTraceCorrectionTerm3D {
    int owner_dof = -1;
    Eigen::VectorXd evaluation;
    Eigen::Matrix<double, 1, 6> value_jet_evaluation =
        Eigen::Matrix<double, 1, 6>::Zero();
};

constexpr std::size_t kRestrictOwnerDecisionKindCount =
    static_cast<std::size_t>(
        app3d::RestrictOwnerDecisionKind3D::AmbiguousEdgeFallback)
    + std::size_t{1};

struct HarmonicTraceOwnerAuditTerm3D {
    int grid_node = -1;
    double interpolation_weight = 0.0;
    int owner_dof = -1;
    int crossing_patch = -1;
    double crossing_u = 0.0;
    double crossing_v = 0.0;
    double segment_parameter = 0.0;
    double residual = 0.0;
    double transversality = 0.0;
};

struct HarmonicTraceSample3D {
    std::array<int, kRestrictGridPointCount> grid_ids{};
    std::array<double, kRestrictGridPointCount> weights{};
    Eigen::VectorXd legacy_correction_evaluation;
    Eigen::Matrix<double, 1, 6> legacy_value_jet_evaluation =
        Eigen::Matrix<double, 1, 6>::Zero();
    std::vector<HarmonicTraceCorrectionTerm3D> owner_corrections;
    int wrong_side_node_count = 0;
    double wrong_side_sum_abs_weight = 0.0;
    std::array<int, kRestrictOwnerDecisionKindCount> owner_decision_counts{};
    std::array<double, kRestrictOwnerDecisionKindCount>
        owner_decision_sum_abs_weights{};
    int owner_unresolved_fallback_count = 0;
    double owner_unresolved_fallback_sum_abs_weight = 0.0;
    int owner_unrelated_coincidence_fallback_count = 0;
    double owner_unrelated_coincidence_fallback_sum_abs_weight = 0.0;
    std::size_t owner_geometry_query_count = 0;
    std::vector<HarmonicTraceOwnerAuditTerm3D> owner_reroute_terms;
};

// One Cauchy correction row evaluated at a support grid node.  The row is
// crossing-centred and already composed with the local value/normal recovery,
// so restrict apply only reads surface jump samples and performs dot products.
struct SharedQuadraticGridlineCorrection3D {
    int stencil_node = -1;
    std::uint64_t native_event_id = 0;
    std::uint64_t native_proof_id = 0;
    // Branch change for this one physical event along the oriented
    // support-node -> trace path: inside_after - inside_before.  Keeping the
    // sign on the event is essential when a path intersects the interface
    // two or more times.
    int continuation_sign = 0;
    std::vector<int> value_ids;
    std::vector<int> normal_ids;
    // Stored maps recover the crossing-centred Cauchy polynomial directly
    // from the surface density samples.  This is deliberately independent
    // of the panel-centred restrict jets used by older routes.
    Eigen::VectorXd value_weights;
    Eigen::VectorXd normal_weights;
    P2CrossingOwner3D direct_crossing_owner;
    int direct_grid_node = -1;
    app3d::NativeDensityC0Stencil3D direct_value_row;
    app3d::NativeDensityC0Stencil3D direct_normal_coefficient_row;
    app3d::ValueCauchyWeightRow3D direct_known_value_row =
        app3d::ValueCauchyWeightRow3D::Zero();
    app3d::NormalCauchyWeightRow3D direct_normal_row =
        app3d::NormalCauchyWeightRow3D::Zero();
    app3d::NativeSurfaceParameterJet3D direct_geometry;
    app3d::LocalOrthonormalFrame3D direct_frame;
    int direct_patch = -1;
    double direct_u = 0.0;
    double direct_v = 0.0;
    bool direct_coefficient_ready = false;
    bool direct_normal_coefficient_ready = false;
    bool direct_known_value_ready = false;
    double cauchy_condition = 0.0;
    double gridline_distance_over_h = 0.0;
};

struct DirectCoefficientSpreadRow3D {
    int rhs_node = -1;
    double scale = 0.0;
    app3d::NativeDensityC0Stencil3D value_row;
    app3d::NormalCauchyWeightRow3D normal_row =
        app3d::NormalCauchyWeightRow3D::Zero();
    app3d::NativeSurfaceParameterJet3D geometry;
    app3d::LocalOrthonormalFrame3D frame;
    int patch = -1;
    double u = 0.0;
    double v = 0.0;
    std::vector<int> legacy_normal_ids;
    Eigen::VectorXd legacy_normal_weights;
};

struct DirectCoefficientNormalSpreadRow3D {
    int rhs_node = -1;
    double scale = 0.0;
    app3d::NativeDensityC0Stencil3D normal_row;
    app3d::ValueCauchyWeightRow3D known_value_row =
        app3d::ValueCauchyWeightRow3D::Zero();
    app3d::NativeSurfaceParameterJet3D geometry;
    app3d::LocalOrthonormalFrame3D frame;
    int patch = -1;
    double u = 0.0;
    double v = 0.0;
    std::vector<int> legacy_value_ids;
    Eigen::VectorXd legacy_value_weights;
};

// The three normal-layer queries on one side share these ten Cartesian nodes
// and one complete-quadratic interpolation polynomial.
struct SharedQuadraticTraceSidePlan3D {
    app3d::SharedQuadraticRestrictStencil3D interpolation;
    std::vector<SharedQuadraticGridlineCorrection3D> corrections;
};

// One tensor-product Cartesian cover at the trace point.  The interpolation
// weights act directly at the interface; there are no auxiliary normal
// layers and no a1/h profile recovery.  Interior and exterior plans share
// the same cover but retain independent all-event branch continuations.
struct TensorProductCoverTraceBranchPlan3D {
    app3d::TensorProductCoverRestrictStencil3D interpolation;
    std::vector<SharedQuadraticGridlineCorrection3D> corrections;
};

struct SharedQuadraticGridlineCrossing3D {
    int lo = -1;
    int hi = -1;
    geometry3d::GridEdgeEventId3D event_id;
    P2CrossingOwner3D owner;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
};

struct RestrictOwnerSampleDiagnostics3D {
    int target_dof = -1;
    int target_patch = -1;
    int side = -1;
    int layer = -1;
    int wrong_side_count = 0;
    double wrong_side_sum_abs_weight = 0.0;
    std::array<int, kRestrictOwnerDecisionKindCount> decision_counts{};
    std::array<double, kRestrictOwnerDecisionKindCount>
        decision_sum_abs_weights{};
    int unresolved_fallback_count = 0;
    double unresolved_fallback_sum_abs_weight = 0.0;
    int unrelated_coincidence_fallback_count = 0;
    double unrelated_coincidence_fallback_sum_abs_weight = 0.0;
    std::size_t geometry_query_count = 0;
};

struct RestrictOwnerAuditRecord3D {
    int target_dof = -1;
    int target_patch = -1;
    int side = -1;
    int layer = -1;
    int grid_node = -1;
    double interpolation_weight = 0.0;
    int owner_dof = -1;
    int owner_patch = -1;
    int crossing_patch = -1;
    double crossing_u = 0.0;
    double crossing_v = 0.0;
    double segment_parameter = 0.0;
    double residual = 0.0;
    double transversality = 0.0;
};

class PanelCenterHarmonicJetKFBI3D {
public:
    PanelCenterHarmonicJetKFBI3D(const CartesianGrid3D& grid,
                                 const GridPair3D& grid_pair,
                                 const NativeNurbsSurface3D& native_surface,
                                 const std::vector<geometry3d::NurbsParamTriangle3D>&
                                     correction_triangles,
                                 const std::vector<geometry3d::NurbsParamTriangle3D>&
                                     geometry_triangles,
                                 const SurfaceDofCloud& cloud,
                                 const CauchyStencilSet& stencils,
                                 bool build_exterior_only_restrict = false,
                                 bool build_crossing_owner_restrict = false,
                                 int local_restrict_value_count = 0,
                                 int local_restrict_normal_count = 0,
                                 bool build_local_quadratic_restrict = false,
                                 bool build_shared_quadratic_restrict = false,
                                 bool build_standard_trace_restrict = true,
                                 app3d::SharedQuadraticNormalProfile3D
                                     shared_normal_profile =
                                         app3d::SharedQuadraticNormalProfile3D::
                                             LegacySplitQuadratic,
                                 bool build_q27_cover3_restrict = false,
                                 bool build_q64_cover4_restrict = false,
                                 bool use_native_endpoint_paths = false)
        : grid_(grid)
        , grid_pair_(grid_pair)
        , native_surface_(native_surface)
        , correction_triangles_(correction_triangles)
        , geometry_triangles_(geometry_triangles)
        , cloud_(cloud)
        , stencils_(stencils)
        , shared_normal_profile_(shared_normal_profile)
        , native_endpoint_paths_(use_native_endpoint_paths)
        , h_(grid.spacing()[0])
        , fit_(native_surface, cloud, stencils, h_,
               kCauchyPolynomialDegree,
               selected_cauchy_value_trace_fit_mode(),
               local_restrict_value_count,
               local_restrict_normal_count,
               !((build_shared_quadratic_restrict
                  || build_q27_cover3_restrict
                  || build_q64_cover4_restrict)
                 && !build_standard_trace_restrict
                 && selected_density_iteration_mode()
                        == DensityIterationMode3D::ReducedCoefficients))
        , bulk_(grid, ZfftBcType::Dirichlet, 0.0, 2)
        , correction_support_(build_laplace_correction_support_3d(
              grid_pair, "PanelCenterHarmonicJetKFBI3D"))
    {
        if (native_endpoint_paths_) {
            if (!build_q27_cover3_restrict && !build_q64_cover4_restrict)
                throw std::invalid_argument(
                    "native endpoint support paths require a Q27/Q64 cover route");
            native_endpoint_budget_ = selected_native_endpoint_budget();
        }
        const auto spacing = grid_.spacing();
        if (std::abs(spacing[0] - spacing[1]) > 1.0e-13
            || std::abs(spacing[0] - spacing[2]) > 1.0e-13) {
            throw std::invalid_argument(
                "harmonic-jet KFBI3D requires an isotropic Cartesian grid");
        }
        if (build_exterior_only_restrict) {
            exterior_only_restrict_ =
                std::make_unique<app3d::ExteriorOnlyCubicNormalRestrict3D>(
                    grid_, grid_pair_, cloud_);
        }
        build_crossing_rows(
            selected_density_iteration_mode()
            == DensityIterationMode3D::ReducedCoefficients);
        if (build_local_quadratic_restrict)
            build_local_quadratic_restrict_plans();
        geometry3d::NurbsSurfaceIntersectorOptions3D restrict_options;
        restrict_options.maximum_element_extent = 2.0 * h_;
        restrict_options.local_max_subdivision_depth = 4;
        if (build_crossing_owner_restrict) {
            const geometry3d::NurbsSurfaceIntersector3D intersector(
                native_surface_.geometry_model(), restrict_options);
            build_trace_templates(&intersector);
            crossing_owner_templates_built_ = true;
        } else if (build_standard_trace_restrict) {
            build_trace_templates(nullptr);
        }
        if (build_shared_quadratic_restrict
            || build_q27_cover3_restrict
            || build_q64_cover4_restrict) {
            // Unlike the legacy route, all-event path continuation needs the
            // complete ordered root set.  Give endpoint queries a dedicated
            // deep subdivision budget and never accept per-element partials.
            restrict_options.local_max_subdivision_depth = 48;
            // Canonicalize one physical feature event into one crossing and
            // retain all incident patch owners on that crossing.  Distinct
            // (even extremely close) roots must remain distinct events.
            restrict_options.preserve_non_g1_root_owners = false;
            restrict_options.collect_unresolved_regions = true;
            const geometry3d::NurbsSurfaceIntersector3D shared_intersector(
                native_surface_.geometry_model(), restrict_options);
            if (build_shared_quadratic_restrict) {
                build_shared_quadratic_trace_templates(shared_intersector);
                shared_quadratic_templates_built_ = true;
            }
            if (build_q27_cover3_restrict) {
                build_tensor_product_cover_trace_templates(
                    shared_intersector,
                    app3d::TensorProductCoverKind3D::Q27Cover3);
                q27_cover3_templates_built_ = true;
            }
            if (build_q64_cover4_restrict) {
                build_tensor_product_cover_trace_templates(
                    shared_intersector,
                    app3d::TensorProductCoverKind3D::Q64Cover4);
                q64_cover4_templates_built_ = true;
            }
            // Dense recovery maps are needed only while composing retained
            // all-event correction rows.  Both cover families must be built
            // before this common cache is released.
            shared_quadratic_cauchy_cache_.clear();
            all_event_exact_cauchy_cache_.clear();
            all_event_intersection_cache_.clear();
            native_endpoint_path_cache_.clear();
        }
        if (build_standard_trace_restrict) {
            build_joint_trace_fit();
            build_global_trace_fit();
        }
        if (build_shared_quadratic_restrict)
            build_shared_quadratic_joint_trace_fit();
    }

    int surface_size() const
    {
        return static_cast<int>(cloud_.dofs.size());
    }

    double surface_area() const
    {
        double area = 0.0;
        for (const SurfaceDof& dof : cloud_.dofs)
            area += dof.weight;
        return area;
    }

    const SurfaceDofCloud& surface() const { return cloud_; }

    Eigen::Matrix<double, 1, 6> restrict_value_jet_row(
        int owner_dof,
        const Eigen::Vector3d& target) const
    {
        if (owner_dof < 0 || owner_dof >= surface_size()
            || local_restrict_plans_.size()
                   != static_cast<std::size_t>(surface_size())) {
            throw std::invalid_argument(
                "local value-jet row received an invalid owner or an "
                "uninitialized restrict plan");
        }
        const SurfaceDof& owner =
            cloud_.dofs[static_cast<std::size_t>(owner_dof)];
        const LocalRestrictCauchyPlan3D& plan =
            local_restrict_plans_[static_cast<std::size_t>(owner_dof)];
        return app3d::pweights_general_3d(
                   target - owner.point,
                   plan.frame,
                   plan.graph_hessian)
            .w0;
    }

    Eigen::VectorXd restrict_normal_cauchy_row(
        int owner_dof,
        const Eigen::Vector3d& target) const
    {
        if (owner_dof < 0 || owner_dof >= surface_size()) {
            throw std::invalid_argument(
                "local normal-Cauchy row received an invalid owner");
        }
        const Eigen::Vector3d xi = local_coordinate(owner_dof, target);
        return fit_.space().basis(xi.x(), xi.y(), xi.z());
    }

    int restrict_cauchy_dimension() const noexcept
    {
        return fit_.dimension();
    }

    Eigen::VectorXd solve_external_correction_rhs(
        const Eigen::VectorXd& correction_rhs) const
    {
        if (correction_rhs.size() != grid_.num_dofs())
            throw std::invalid_argument(
                "external Cauchy correction has the wrong grid size");
        Eigen::VectorXd potential;
        bulk_.solve(-correction_rhs, potential);
        return potential;
    }

    std::vector<double> cauchy_condition_values() const
    {
        return fit_.condition_values();
    }

    std::vector<double> local_restrict_condition_values() const
    {
        return fit_.local_condition_values();
    }

    std::vector<double> local_quadratic_restrict_condition_values() const
    {
        std::vector<double> result;
        result.reserve(local_restrict_plans_.size());
        for (const LocalRestrictCauchyPlan3D& plan
             : local_restrict_plans_) {
            result.push_back(plan.value_condition);
        }
        return result;
    }

    Eigen::MatrixXd recover_local_quadratic_value_jets(
        const app3d::NativeNurbsDensitySpace3D& density,
        const Eigen::VectorXd& c0_coefficients) const
    {
        if (local_restrict_plans_.size()
            != static_cast<std::size_t>(surface_size())) {
            throw std::runtime_error(
                "local quadratic Neumann restrict was not initialized");
        }
        if (density.patch_count()
                != static_cast<int>(native_surface_.patches.size())
            || c0_coefficients.size()
                   != density.c0_coefficient_count()) {
            throw std::invalid_argument(
                "local quadratic restrict received incompatible density data");
        }
        Eigen::MatrixXd result(surface_size(), 6);
        for (int center = 0; center < surface_size(); ++center) {
            const LocalRestrictCauchyPlan3D& plan =
                local_restrict_plans_[static_cast<std::size_t>(center)];
            Eigen::Matrix<double, kLocalRestrictSampleCount, 1> values;
            for (int sample = 0;
                 sample < kLocalRestrictSampleCount;
                 ++sample) {
                const LocalRestrictDensitySample3D& point =
                    plan.samples[static_cast<std::size_t>(sample)];
                values[sample] = density.evaluate_c0(
                    point.patch, point.u, point.v, c0_coefficients);
            }
            result.row(center) =
                (plan.value_recovery * values).transpose();
            const SurfaceDof& owner =
                cloud_.dofs[static_cast<std::size_t>(center)];
            result(center, 0) = density.evaluate_c0(
                owner.patch_id, owner.u, owner.v, c0_coefficients);
        }
        if (!result.allFinite()) {
            throw std::runtime_error(
                "local quadratic restrict recovered NaN/Inf value jets");
        }
        return result;
    }

    app3d::GeometricFeatureTraceSummary3D feature_trace_summary() const
    {
        return fit_.feature_trace_summary();
    }

    std::vector<double> feature_trace_condition_values() const
    {
        return fit_.feature_trace_condition_values();
    }

    std::vector<double> exterior_only_restrict_condition_values() const
    {
        if (!exterior_only_restrict_)
            throw std::runtime_error(
                "exterior-only normal restrict was not initialized");
        std::vector<double> result;
        result.reserve(exterior_only_restrict_->stencils().size());
        for (const auto& stencil : exterior_only_restrict_->stencils())
            result.push_back(stencil.condition);
        return result;
    }

    std::vector<int> joint_trace_wrong_side_node_counts() const
    {
        std::vector<int> result(static_cast<std::size_t>(surface_size()), 0);
        for (int center = 0; center < surface_size(); ++center) {
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0;
                     layer < kRestrictNormalLayerCount;
                     ++layer) {
                    result[static_cast<std::size_t>(center)] +=
                        trace_samples_[trace_sample_index(center, side, layer)]
                            .wrong_side_node_count;
                }
            }
        }
        return result;
    }

    std::vector<RestrictOwnerSampleDiagnostics3D>
    restrict_owner_sample_diagnostics() const
    {
        if (!crossing_owner_templates_built_) {
            throw std::runtime_error(
                "crossing-owner normal restrict was not initialized");
        }
        std::vector<RestrictOwnerSampleDiagnostics3D> result;
        result.reserve(trace_samples_.size());
        for (int center = 0; center < surface_size(); ++center) {
            const int target_patch =
                cloud_.dofs[static_cast<std::size_t>(center)].patch_id;
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0;
                     layer < kRestrictNormalLayerCount;
                     ++layer) {
                    const HarmonicTraceSample3D& sample = trace_samples_[
                        trace_sample_index(center, side, layer)];
                    RestrictOwnerSampleDiagnostics3D item;
                    item.target_dof = center;
                    item.target_patch = target_patch;
                    item.side = side;
                    item.layer = layer;
                    item.wrong_side_count = sample.wrong_side_node_count;
                    item.wrong_side_sum_abs_weight =
                        sample.wrong_side_sum_abs_weight;
                    item.decision_counts = sample.owner_decision_counts;
                    item.decision_sum_abs_weights =
                        sample.owner_decision_sum_abs_weights;
                    item.unresolved_fallback_count =
                        sample.owner_unresolved_fallback_count;
                    item.unresolved_fallback_sum_abs_weight =
                        sample.owner_unresolved_fallback_sum_abs_weight;
                    item.unrelated_coincidence_fallback_count =
                        sample.owner_unrelated_coincidence_fallback_count;
                    item.unrelated_coincidence_fallback_sum_abs_weight =
                        sample.
                            owner_unrelated_coincidence_fallback_sum_abs_weight;
                    item.geometry_query_count =
                        sample.owner_geometry_query_count;
                    result.push_back(std::move(item));
                }
            }
        }
        return result;
    }

    std::vector<RestrictOwnerAuditRecord3D>
    restrict_owner_audit_records() const
    {
        if (!crossing_owner_templates_built_) {
            throw std::runtime_error(
                "crossing-owner normal restrict was not initialized");
        }
        std::vector<RestrictOwnerAuditRecord3D> result;
        for (int center = 0; center < surface_size(); ++center) {
            const int target_patch =
                cloud_.dofs[static_cast<std::size_t>(center)].patch_id;
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0;
                     layer < kRestrictNormalLayerCount;
                     ++layer) {
                    const HarmonicTraceSample3D& sample = trace_samples_[
                        trace_sample_index(center, side, layer)];
                    for (const HarmonicTraceOwnerAuditTerm3D& term
                         : sample.owner_reroute_terms) {
                        if (term.owner_dof < 0
                            || term.owner_dof >= surface_size()) {
                            throw std::logic_error(
                                "crossing-owner audit has invalid owner DOF");
                        }
                        RestrictOwnerAuditRecord3D item;
                        item.target_dof = center;
                        item.target_patch = target_patch;
                        item.side = side;
                        item.layer = layer;
                        item.grid_node = term.grid_node;
                        item.interpolation_weight =
                            term.interpolation_weight;
                        item.owner_dof = term.owner_dof;
                        item.owner_patch = cloud_.dofs[
                            static_cast<std::size_t>(term.owner_dof)].patch_id;
                        item.crossing_patch = term.crossing_patch;
                        item.crossing_u = term.crossing_u;
                        item.crossing_v = term.crossing_v;
                        item.segment_parameter = term.segment_parameter;
                        item.residual = term.residual;
                        item.transversality = term.transversality;
                        result.push_back(std::move(item));
                    }
                }
            }
        }
        return result;
    }

    std::size_t restrict_owner_geometry_query_count() const
    {
        std::size_t result = 0;
        for (const HarmonicTraceSample3D& sample : trace_samples_)
            result += sample.owner_geometry_query_count;
        return result;
    }

    const SharedQuadraticRestrictDiagnostics3D&
    shared_quadratic_restrict_diagnostics() const
    {
        if (!shared_quadratic_templates_built_
            && !q27_cover3_templates_built_
            && !q64_cover4_templates_built_) {
            throw std::runtime_error(
                "all-event Cauchy restrict was not initialized");
        }
        return shared_quadratic_diagnostics_;
    }

    HarmonicJetField3D evaluate(const Eigen::VectorXd& value_jump,
                                const Eigen::VectorXd& normal_jump) const
    {
        HarmonicJetField3D result;
        result.coefficients = fit_.coefficients(value_jump, normal_jump);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid_.num_dofs());
        for (const HarmonicCrossingRow3D& row : crossing_rows_) {
            rhs[row.rhs_node] += row.scale
                * row.evaluation.dot(
                    result.coefficients.row(row.center_dof).transpose());
        }
        // The spread correction is assembled for Delta_h, while the project
        // bulk solver accepts the right-hand side of -Delta_h.
        bulk_.solve(-rhs, result.potential);
        return result;
    }

    HarmonicJetField3D evaluate_crossing_local(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        const std::function<double(int)>& value_at_surface_dof,
        const std::function<double(int)>& normal_at_surface_dof,
        bool use_feature_value_jet = true,
        bool anchor_center_cauchy = false,
        bool use_local_restrict_fit = false) const
    {
        if (value_jump.size() != surface_size()
            || normal_jump.size() != surface_size()
            || !value_at_surface_dof || !normal_at_surface_dof) {
            throw std::invalid_argument(
                "crossing-local Cauchy evaluation received incompatible data");
        }
        HarmonicJetField3D result;
        // Panel-centred jets are retained only for the later exterior
        // restriction corrections.  Cartesian spread corrections below use
        // the independent crossing-centred direct rows.
        if (fit_.maps_enabled()) {
            result.coefficients = fit_.coefficients(
                value_jump, normal_jump, use_feature_value_jet,
                anchor_center_cauchy, use_local_restrict_fit);
            if (use_local_restrict_fit) {
                result.restrict_normal_coefficients = fit_.coefficients(
                    Eigen::VectorXd::Zero(surface_size()),
                    normal_jump,
                    false,
                    anchor_center_cauchy,
                    true);
            }
        } else {
            // An all-event direct-cover trace reconstructs every continuation
            // from the jump arrays below and never consumes panel-centred jets.
            result.coefficients = Eigen::MatrixXd::Zero(
                surface_size(), fit_.dimension());
            if (use_local_restrict_fit) {
                result.restrict_normal_coefficients = result.coefficients;
            }
        }
        if (direct_crossing_rows_.size() != crossing_rows_.size()) {
            throw std::runtime_error(
                "crossing-local spread rows were not requested at setup");
        }
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid_.num_dofs());
        for (const HarmonicCrossingDirectRow3D& row
             : direct_crossing_rows_) {
            double correction = 0.0;
            for (int k = 0; k < static_cast<int>(row.value_ids.size()); ++k) {
                correction += row.value_weights[k]
                    * value_at_surface_dof(
                        row.value_ids[static_cast<std::size_t>(k)]);
            }
            for (int k = 0; k < static_cast<int>(row.normal_ids.size()); ++k) {
                correction += row.normal_weights[k]
                    * normal_at_surface_dof(
                        row.normal_ids[static_cast<std::size_t>(k)]);
            }
            rhs[row.rhs_node] += row.scale * correction;
        }
        bulk_.solve(-rhs, result.potential);
        return result;
    }

    void bind_direct_coefficient_value_density(
        const app3d::NativeNurbsDensitySpace3D& density)
    {
        if (density.patch_count()
            != static_cast<int>(native_surface_.patches.size())) {
            throw std::invalid_argument(
                "direct coefficient Cauchy density has incompatible patches");
        }
        if (direct_crossing_rows_.size()
            != correction_support_.crossing_ops.size()) {
            throw std::runtime_error(
                "direct coefficient Cauchy requires crossing-local setup");
        }

        struct CachedPlan {
            app3d::DirectCoefficientValueJetPlan3D value;
            app3d::NativeSurfaceParameterJet3D geometry;
        };
        std::map<std::tuple<int, long long, long long>, CachedPlan> cache;
        const auto plan_for = [&](const P2CrossingOwner3D& owner)
            -> const CachedPlan& {
            if (owner.nurbs_patch_index < 0
                || !owner.nurbs_parameter.allFinite()
                || !owner.crossing_point.allFinite()) {
                throw std::runtime_error(
                    "direct coefficient Cauchy crossing has no exact NURBS owner");
            }
            const int patch_index = owner.nurbs_patch_index;
            const auto& patch = density.surface().patches[
                static_cast<std::size_t>(patch_index)];
            double u = (owner.nurbs_parameter.x() - patch.domain_start_u())
                     / (patch.domain_end_u() - patch.domain_start_u());
            double v = (owner.nurbs_parameter.y() - patch.domain_start_v())
                     / (patch.domain_end_v() - patch.domain_start_v());
            u = std::clamp(u, 0.0, 1.0);
            v = std::clamp(v, 0.0, 1.0);
            const auto key = std::make_tuple(
                patch_index,
                std::llround(u * 1.0e12),
                std::llround(v * 1.0e12));
            auto found = cache.find(key);
            if (found != cache.end())
                return found->second;

            CachedPlan cached;
            cached.geometry = app3d::native_surface_parameter_jet_3d(
                density, patch_index, u, v);
            Eigen::Vector3d normal = cached.geometry.normal;
            if (owner.crossing_normal.allFinite()
                && owner.crossing_normal.norm() > 0.5
                && normal.dot(owner.crossing_normal) < 0.0) {
                normal = -normal;
            }
            const app3d::LocalOrthonormalFrame3D frame =
                app3d::make_local_orthonormal_frame_3d(
                    normal, cached.geometry.x_u);
            cached.value = app3d::build_direct_coefficient_value_jet_plan_3d(
                density, patch_index, u, v, cached.geometry, frame);
            return cache.emplace(key, std::move(cached)).first->second;
        };

        direct_coefficient_spread_rows_.clear();
        direct_coefficient_spread_rows_.reserve(
            correction_support_.crossing_ops.size());
        for (std::size_t index = 0;
             index < correction_support_.crossing_ops.size(); ++index) {
            const LaplaceCrossingCorrectionOp& op =
                correction_support_.crossing_ops[index];
            const P2CrossingOwner3D owner =
                laplace_crossing_owner_3d(grid_pair_, op);
            const CachedPlan& cached = plan_for(owner);
            const app3d::CauchyPolynomialWeights3D weights =
                app3d::cauchy_polynomial_weights_3d(
                    cached.geometry.point,
                    cached.value.frame,
                    cached.value.graph_hessian,
                    grid_point(grid_, op.correction_node));
            DirectCoefficientSpreadRow3D row;
            row.rhs_node = op.rhs_node;
            row.scale = static_cast<double>(op.side_delta)
                      * op.stencil_weight;
            row.value_row = cached.value.compose_value_row(weights.w0);
            row.normal_row = weights.w1;
            row.geometry = cached.geometry;
            row.frame = cached.value.frame;
            row.patch = cached.value.patch;
            row.u = cached.value.u;
            row.v = cached.value.v;
            row.legacy_normal_ids = direct_crossing_rows_[index].normal_ids;
            row.legacy_normal_weights =
                direct_crossing_rows_[index].normal_weights;
            direct_coefficient_spread_rows_.push_back(std::move(row));
        }

        const auto bind_cover_corrections = [&](auto& plans) {
          for (auto& side : plans) {
            for (SharedQuadraticGridlineCorrection3D& correction :
                 side.corrections) {
                const CachedPlan& cached =
                    plan_for(correction.direct_crossing_owner);
                const app3d::CauchyPolynomialWeights3D weights =
                    app3d::cauchy_polynomial_weights_3d(
                        cached.geometry.point,
                        cached.value.frame,
                        cached.value.graph_hessian,
                        grid_point(grid_, correction.direct_grid_node));
                correction.direct_value_row =
                    cached.value.compose_value_row(weights.w0);
                correction.direct_normal_row = weights.w1;
                correction.direct_geometry = cached.geometry;
                correction.direct_frame = cached.value.frame;
                correction.direct_patch = cached.value.patch;
                correction.direct_u = cached.value.u;
                correction.direct_v = cached.value.v;
                correction.direct_coefficient_ready = true;
            }
          }
        };
        bind_cover_corrections(shared_quadratic_plans_);
        bind_cover_corrections(q27_cover3_plans_);
        bind_cover_corrections(q64_cover4_plans_);
        direct_coefficient_density_ = &density;
    }

    bool has_direct_coefficient_value_density(
        const app3d::NativeNurbsDensitySpace3D& density) const noexcept
    {
        return direct_coefficient_density_ == &density
            && direct_coefficient_spread_rows_.size()
                   == correction_support_.crossing_ops.size();
    }

    HarmonicJetField3D evaluate_direct_coefficient_crossing_local(
        const app3d::NativeNurbsDensitySpace3D& density,
        const Eigen::VectorXd& value_c0,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        const app3d::KnownNeumannJetCallback3D& known_normal_jet = {},
        bool anchor_center_cauchy = false,
        bool use_local_restrict_fit = false) const
    {
        if (!has_direct_coefficient_value_density(density)
            || value_c0.size() != density.c0_coefficient_count()
            || value_jump.size() != surface_size()
            || normal_jump.size() != surface_size()) {
            throw std::invalid_argument(
                "direct coefficient crossing evaluation received incompatible data");
        }
        HarmonicJetField3D result;
        // All-event direct covers never read this legacy panel-centred
        // representation.
        // Retain only the known normal part for compatibility with diagnostic
        // callers and deliberately do not fit the unknown value density.
        if (fit_.maps_enabled()) {
            result.coefficients = fit_.coefficients(
                Eigen::VectorXd::Zero(surface_size()),
                normal_jump, false, anchor_center_cauchy,
                use_local_restrict_fit);
        } else {
            result.coefficients = Eigen::MatrixXd::Zero(
                surface_size(), fit_.dimension());
        }
        if (use_local_restrict_fit) {
            result.restrict_normal_coefficients = result.coefficients;
        }
        result.direct_value_c0 = value_c0;
        result.direct_known_normal_jet = known_normal_jet;
        result.direct_coefficient_cauchy = true;

        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid_.num_dofs());
        for (const DirectCoefficientSpreadRow3D& row :
             direct_coefficient_spread_rows_) {
            double correction = row.value_row.dot(value_c0);
            if (known_normal_jet) {
                const app3d::NormalJet3D normal_jet = known_normal_jet(
                    row.patch, row.u, row.v, row.geometry, row.frame);
                correction += row.normal_row.dot(normal_jet);
            } else {
                for (int q = 0;
                     q < static_cast<int>(row.legacy_normal_ids.size()); ++q) {
                    correction += row.legacy_normal_weights[q]
                        * normal_jump[row.legacy_normal_ids[
                            static_cast<std::size_t>(q)]];
                }
            }
            rhs[row.rhs_node] += row.scale * correction;
        }
        bulk_.solve(-rhs, result.potential);
        return result;
    }

    void bind_direct_coefficient_normal_density(
        const app3d::NativeNurbsDensitySpace3D& density)
    {
        if (density.patch_count()
            != static_cast<int>(native_surface_.patches.size())) {
            throw std::invalid_argument(
                "direct normal coefficient Cauchy density has incompatible patches");
        }
        if (direct_crossing_rows_.size()
            != correction_support_.crossing_ops.size()) {
            throw std::runtime_error(
                "direct normal coefficient Cauchy requires crossing-local setup");
        }

        struct CachedPlan {
            app3d::DirectCoefficientNormalJetPlan3D normal;
            app3d::NativeSurfaceParameterJet3D geometry;
        };
        std::map<std::tuple<int, long long, long long>, CachedPlan> cache;
        const auto plan_for = [&](const P2CrossingOwner3D& owner)
            -> const CachedPlan& {
            if (owner.nurbs_patch_index < 0
                || !owner.nurbs_parameter.allFinite()
                || !owner.crossing_point.allFinite()) {
                throw std::runtime_error(
                    "direct normal coefficient crossing has no exact NURBS owner");
            }
            const int patch_index = owner.nurbs_patch_index;
            const auto& patch = density.surface().patches[
                static_cast<std::size_t>(patch_index)];
            double u = (owner.nurbs_parameter.x() - patch.domain_start_u())
                     / (patch.domain_end_u() - patch.domain_start_u());
            double v = (owner.nurbs_parameter.y() - patch.domain_start_v())
                     / (patch.domain_end_v() - patch.domain_start_v());
            u = std::clamp(u, 0.0, 1.0);
            v = std::clamp(v, 0.0, 1.0);
            const auto key = std::make_tuple(
                patch_index,
                std::llround(u * 1.0e12),
                std::llround(v * 1.0e12));
            auto found = cache.find(key);
            if (found != cache.end())
                return found->second;

            CachedPlan cached;
            cached.geometry = app3d::native_surface_parameter_jet_3d(
                density, patch_index, u, v);
            Eigen::Vector3d normal = cached.geometry.normal;
            if (owner.crossing_normal.allFinite()
                && owner.crossing_normal.norm() > 0.5
                && normal.dot(owner.crossing_normal) < 0.0) {
                normal = -normal;
            }
            const app3d::LocalOrthonormalFrame3D frame =
                app3d::make_local_orthonormal_frame_3d(
                    normal, cached.geometry.x_u);
            cached.normal =
                app3d::build_direct_coefficient_normal_jet_plan_3d(
                    density, patch_index, u, v, cached.geometry, frame);
            return cache.emplace(key, std::move(cached)).first->second;
        };

        direct_coefficient_normal_spread_rows_.clear();
        direct_coefficient_normal_spread_rows_.reserve(
            correction_support_.crossing_ops.size());
        for (std::size_t index = 0;
             index < correction_support_.crossing_ops.size(); ++index) {
            const LaplaceCrossingCorrectionOp& op =
                correction_support_.crossing_ops[index];
            const P2CrossingOwner3D owner =
                laplace_crossing_owner_3d(grid_pair_, op);
            const CachedPlan& cached = plan_for(owner);
            const app3d::CauchyPolynomialWeights3D weights =
                app3d::cauchy_polynomial_weights_3d(
                    cached.geometry.point,
                    cached.normal.frame,
                    cached.normal.graph_hessian,
                    grid_point(grid_, op.correction_node));
            DirectCoefficientNormalSpreadRow3D row;
            row.rhs_node = op.rhs_node;
            row.scale = static_cast<double>(op.side_delta)
                      * op.stencil_weight;
            row.normal_row = cached.normal.compose_normal_row(weights.w1);
            row.known_value_row = weights.w0;
            row.geometry = cached.geometry;
            row.frame = cached.normal.frame;
            row.patch = cached.normal.patch;
            row.u = cached.normal.u;
            row.v = cached.normal.v;
            row.legacy_value_ids = direct_crossing_rows_[index].value_ids;
            row.legacy_value_weights = direct_crossing_rows_[index].value_weights;
            direct_coefficient_normal_spread_rows_.push_back(std::move(row));
        }

        const auto bind_cover_corrections = [&](auto& plans) {
          for (auto& side : plans) {
            for (SharedQuadraticGridlineCorrection3D& correction :
                 side.corrections) {
                const CachedPlan& cached =
                    plan_for(correction.direct_crossing_owner);
                const app3d::CauchyPolynomialWeights3D weights =
                    app3d::cauchy_polynomial_weights_3d(
                        cached.geometry.point,
                        cached.normal.frame,
                        cached.normal.graph_hessian,
                        grid_point(grid_, correction.direct_grid_node));
                correction.direct_normal_coefficient_row =
                    cached.normal.compose_normal_row(weights.w1);
                correction.direct_known_value_row = weights.w0;
                correction.direct_geometry = cached.geometry;
                correction.direct_frame = cached.normal.frame;
                correction.direct_patch = cached.normal.patch;
                correction.direct_u = cached.normal.u;
                correction.direct_v = cached.normal.v;
                correction.direct_normal_coefficient_ready = true;
                correction.direct_known_value_ready = true;
            }
          }
        };
        bind_cover_corrections(shared_quadratic_plans_);
        bind_cover_corrections(q27_cover3_plans_);
        bind_cover_corrections(q64_cover4_plans_);
        direct_coefficient_normal_density_ = &density;
    }

    bool has_direct_coefficient_normal_density(
        const app3d::NativeNurbsDensitySpace3D& density) const noexcept
    {
        return direct_coefficient_normal_density_ == &density
            && direct_coefficient_normal_spread_rows_.size()
                   == correction_support_.crossing_ops.size();
    }

    HarmonicJetField3D evaluate_direct_coefficient_normal_crossing_local(
        const app3d::NativeNurbsDensitySpace3D& density,
        const Eigen::VectorXd& normal_c0,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        bool use_feature_value_jet = true,
        bool anchor_center_cauchy = false,
        bool use_local_restrict_fit = false,
        const app3d::KnownDirichletJetCallback3D& known_value_jet = {}) const
    {
        if (!has_direct_coefficient_normal_density(density)
            || normal_c0.size() != density.c0_coefficient_count()
            || value_jump.size() != surface_size()
            || normal_jump.size() != surface_size()) {
            throw std::invalid_argument(
                "direct normal coefficient crossing evaluation received incompatible data");
        }
        HarmonicJetField3D result;
        // An analytic known-J0 callback is the topology-native affine route:
        // it is evaluated at each exact crossing and never enters the panel
        // sample fit.  The homogeneous topology matvec supplies J0 == 0 and
        // likewise needs no fit.  Other empty-callback calls retain the
        // legacy A/B sampled-data route.
        if (!fit_.maps_enabled()
            || known_value_jet || value_jump.isZero(0.0)) {
            result.coefficients = Eigen::MatrixXd::Zero(
                surface_size(), fit_.dimension());
        } else {
            result.coefficients = fit_.coefficients(
                value_jump, Eigen::VectorXd::Zero(surface_size()),
                use_feature_value_jet, anchor_center_cauchy,
                use_local_restrict_fit);
        }
        result.direct_normal_c0 = normal_c0;
        result.direct_known_value_jet = known_value_jet;
        result.direct_coefficient_cauchy = true;

        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid_.num_dofs());
        for (const DirectCoefficientNormalSpreadRow3D& row :
             direct_coefficient_normal_spread_rows_) {
            double correction = row.normal_row.dot(normal_c0);
            if (known_value_jet) {
                const app3d::ValueJet3D value_jet = known_value_jet(
                    row.patch, row.u, row.v, row.geometry, row.frame);
                if (!value_jet.allFinite()) {
                    throw std::runtime_error(
                        "known Dirichlet crossing jet is not finite");
                }
                correction += row.known_value_row.dot(value_jet);
            } else {
                for (int q = 0;
                     q < static_cast<int>(row.legacy_value_ids.size()); ++q) {
                    correction += row.legacy_value_weights[q]
                        * value_jump[row.legacy_value_ids[
                            static_cast<std::size_t>(q)]];
                }
            }
            rhs[row.rhs_node] += row.scale * correction;
        }
        bulk_.solve(-rhs, result.potential);
        return result;
    }

    std::vector<double> crossing_local_condition_values() const
    {
        std::vector<double> result;
        result.reserve(direct_crossing_rows_.size());
        for (const HarmonicCrossingDirectRow3D& row : direct_crossing_rows_)
            result.push_back(row.condition);
        return result;
    }

    std::size_t crossing_local_unique_plan_count() const noexcept
    {
        return direct_crossing_unique_plan_count_;
    }

    std::size_t crossing_local_svd_count() const noexcept
    {
        return direct_crossing_svd_count_;
    }

    HarmonicJetField3D field_from_grid_and_jumps(
        const Eigen::VectorXd& potential,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        if (potential.size() != grid_.num_dofs()
            || value_jump.size() != surface_size()
            || normal_jump.size() != surface_size()) {
            throw std::invalid_argument(
                "exact-grid field received incompatible sizes");
        }
        return {potential, fit_.coefficients(value_jump, normal_jump)};
    }

    Eigen::VectorXd exterior_trace(const HarmonicJetField3D& field,
                                   const Eigen::VectorXd& value_jump,
                                   const Eigen::VectorXd& normal_jump) const
    {
        return exterior_trace(
            field, value_jump, normal_jump,
            ExteriorNormalRestrictMode3D::JointTricubicCauchy);
    }

    Eigen::VectorXd exterior_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        ExteriorNormalRestrictMode3D mode) const
    {
        if (trace_restrict_uses_tensor_cover(mode)) {
            return recover_tensor_product_cover_trace(
                field, value_jump, normal_jump, false, false, mode);
        }
        if (trace_restrict_uses_shared_quadratic(mode)) {
            return recover_shared_quadratic_joint_trace(
                shared_quadratic_continued_samples(
                    field, value_jump, normal_jump, false), false);
        }
        if (mode == ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic) {
            throw std::invalid_argument(
                "exterior-only harmonic cubic restrict provides only "
                "a normal trace");
        }
        const Eigen::MatrixXd samples = continued_samples(
            field, value_jump, normal_jump, false, mode);
        return recover_trace(
            samples,
            mode == ExteriorNormalRestrictMode3D::
                        GlobalCubicExteriorBranchCrossingOwner
                ? global_c0_weights_ : c0_weights_,
            1.0);
    }

    Eigen::VectorXd interior_trace(const HarmonicJetField3D& field,
                                   const Eigen::VectorXd& value_jump,
                                   const Eigen::VectorXd& normal_jump) const
    {
        return interior_trace(
            field, value_jump, normal_jump,
            ExteriorNormalRestrictMode3D::JointTricubicCauchy);
    }

    Eigen::VectorXd interior_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        ExteriorNormalRestrictMode3D mode) const
    {
        if (trace_restrict_uses_tensor_cover(mode)) {
            return recover_tensor_product_cover_trace(
                field, value_jump, normal_jump, true, false, mode);
        }
        if (trace_restrict_uses_shared_quadratic(mode)) {
            return recover_shared_quadratic_joint_trace(
                shared_quadratic_continued_samples(
                    field, value_jump, normal_jump, true), false);
        }
        if (mode == ExteriorNormalRestrictMode3D::
                GlobalCubicExteriorBranchCrossingOwner) {
            return recover_trace(
                continued_samples(
                    field, value_jump, normal_jump, true, mode),
                global_c0_weights_, 1.0);
        }
        if (mode == ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic) {
            throw std::invalid_argument(
                "exterior-only harmonic cubic restrict provides only "
                "a normal trace");
        }
        return recover_trace(
            continued_samples(field, value_jump, normal_jump, true, mode),
            c0_weights_, 1.0);
    }

    Eigen::VectorXd exterior_normal_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        return exterior_normal_trace(
            field, value_jump, normal_jump,
            ExteriorNormalRestrictMode3D::JointTricubicCauchy);
    }

    Eigen::VectorXd exterior_normal_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        ExteriorNormalRestrictMode3D mode) const
    {
        if (trace_restrict_uses_tensor_cover(mode)) {
            return recover_tensor_product_cover_trace(
                field, value_jump, normal_jump, false, true, mode);
        }
        if (trace_restrict_uses_shared_quadratic(mode)) {
            return recover_shared_quadratic_joint_trace(
                shared_quadratic_continued_samples(
                    field, value_jump, normal_jump, false), true);
        }
        if (mode == ExteriorNormalRestrictMode3D::
                GlobalCubicExteriorBranchCrossingOwner) {
            throw std::invalid_argument(
                "global exterior-branch restrict currently provides only "
                "the Neumann exterior value trace");
        }
        if (mode == ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic) {
            if (!exterior_only_restrict_) {
                throw std::runtime_error(
                    "exterior-only normal restrict was not initialized");
            }
            return exterior_only_restrict_->apply(field.potential);
        }
        return recover_trace(
            continued_samples(field, value_jump, normal_jump, false, mode),
            c1_weights_, 1.0 / h_);
    }

    Eigen::VectorXd interior_normal_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        return interior_normal_trace(
            field, value_jump, normal_jump,
            ExteriorNormalRestrictMode3D::JointTricubicCauchy);
    }

    Eigen::VectorXd interior_normal_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        ExteriorNormalRestrictMode3D mode) const
    {
        if (trace_restrict_uses_tensor_cover(mode)) {
            return recover_tensor_product_cover_trace(
                field, value_jump, normal_jump, true, true, mode);
        }
        if (trace_restrict_uses_shared_quadratic(mode)) {
            return recover_shared_quadratic_joint_trace(
                shared_quadratic_continued_samples(
                    field, value_jump, normal_jump, true), true);
        }
        if (mode == ExteriorNormalRestrictMode3D::
                GlobalCubicExteriorBranchCrossingOwner) {
            throw std::invalid_argument(
                "global exterior-branch restrict currently provides only "
                "the Neumann exterior value trace");
        }
        return recover_trace(
            continued_samples(field, value_jump, normal_jump, true, mode),
            c1_weights_, 1.0 / h_);
    }

private:
    double evaluate_all_event_cauchy_correction(
        const SharedQuadraticGridlineCorrection3D& correction,
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        double cauchy_value = 0.0;
        if (field.direct_coefficient_cauchy) {
            if (field.direct_known_value_jet) {
                if (!correction.direct_known_value_ready) {
                    throw std::runtime_error(
                        "all-event restrict analytic Dirichlet correction was not bound");
                }
                const app3d::ValueJet3D value_jet =
                    field.direct_known_value_jet(
                        correction.direct_patch,
                        correction.direct_u,
                        correction.direct_v,
                        correction.direct_geometry,
                        correction.direct_frame);
                if (!value_jet.allFinite()) {
                    throw std::runtime_error(
                        "all-event restrict known Dirichlet jet is not finite");
                }
                cauchy_value +=
                    correction.direct_known_value_row.dot(value_jet);
            } else if (field.direct_value_c0.size() != 0) {
                if (!correction.direct_coefficient_ready) {
                    throw std::runtime_error(
                        "all-event restrict direct value correction was not bound");
                }
                cauchy_value +=
                    correction.direct_value_row.dot(field.direct_value_c0);
            } else {
                for (int q = 0;
                     q < static_cast<int>(correction.value_ids.size()); ++q) {
                    cauchy_value += correction.value_weights[q]
                        * value_jump[correction.value_ids[
                            static_cast<std::size_t>(q)]];
                }
            }
            if (field.direct_known_normal_jet) {
                const app3d::NormalJet3D normal_jet =
                    field.direct_known_normal_jet(
                        correction.direct_patch,
                        correction.direct_u,
                        correction.direct_v,
                        correction.direct_geometry,
                        correction.direct_frame);
                cauchy_value +=
                    correction.direct_normal_row.dot(normal_jet);
            } else if (field.direct_normal_c0.size() != 0) {
                if (!correction.direct_normal_coefficient_ready) {
                    throw std::runtime_error(
                        "all-event restrict direct normal correction was not bound");
                }
                cauchy_value +=
                    correction.direct_normal_coefficient_row.dot(
                        field.direct_normal_c0);
            } else {
                for (int q = 0;
                     q < static_cast<int>(correction.normal_ids.size()); ++q) {
                    cauchy_value += correction.normal_weights[q]
                        * normal_jump[correction.normal_ids[
                            static_cast<std::size_t>(q)]];
                }
            }
            return cauchy_value;
        }

        for (int q = 0;
             q < static_cast<int>(correction.value_ids.size()); ++q) {
            cauchy_value += correction.value_weights[q]
                * value_jump[correction.value_ids[static_cast<std::size_t>(q)]];
        }
        for (int q = 0;
             q < static_cast<int>(correction.normal_ids.size()); ++q) {
            cauchy_value += correction.normal_weights[q]
                * normal_jump[
                    correction.normal_ids[static_cast<std::size_t>(q)]];
        }
        return cauchy_value;
    }

    Eigen::VectorXd recover_tensor_product_cover_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        bool interior_branch,
        bool normal_derivative,
        ExteriorNormalRestrictMode3D mode) const
    {
        const bool q27 =
            mode == ExteriorNormalRestrictMode3D::Q27Cover3AllEventCauchy;
        const bool q64 =
            mode == ExteriorNormalRestrictMode3D::Q64Cover4AllEventCauchy;
        if ((!q27 && !q64)
            || (q27 && !q27_cover3_templates_built_)
            || (q64 && !q64_cover4_templates_built_)) {
            throw std::runtime_error(
                "requested tensor-product cover restrict was not initialized");
        }
        const auto& plans = q27 ? q27_cover3_plans_ : q64_cover4_plans_;
        const int size = surface_size();
        if (plans.size() != static_cast<std::size_t>(2 * size)
            || field.potential.size() != grid_.num_dofs()
            || value_jump.size() != size || normal_jump.size() != size) {
            throw std::invalid_argument(
                "tensor-product cover restrict received incompatible data");
        }

        Eigen::VectorXd result(size);
        const int branch_index = interior_branch ? 0 : 1;
        for (int center = 0; center < size; ++center) {
            const auto& plan = plans[static_cast<std::size_t>(
                2 * center + branch_index)];
            Eigen::VectorXd continued_nodes(plan.interpolation.grid_ids.size());
            for (int q = 0; q < continued_nodes.size(); ++q) {
                continued_nodes[q] = field.potential[
                    plan.interpolation.grid_ids[static_cast<std::size_t>(q)]];
            }
            for (const auto& correction : plan.corrections) {
                if (correction.stencil_node < 0
                    || correction.stencil_node >= continued_nodes.size()) {
                    throw std::logic_error(
                        "tensor-product cover correction has an invalid slot");
                }
                continued_nodes[correction.stencil_node] +=
                    static_cast<double>(correction.continuation_sign)
                    * evaluate_all_event_cauchy_correction(
                        correction, field, value_jump, normal_jump);
            }
            result[center] =
                (normal_derivative
                    ? plan.interpolation.normal_weights
                    : plan.interpolation.value_weights)
                    .dot(continued_nodes);
        }
        if (!result.allFinite()) {
            throw std::runtime_error(
                "tensor-product cover restrict produced NaN/Inf");
        }
        return result;
    }

    Eigen::Matrix<double, Eigen::Dynamic,
                  app3d::kSharedQuadraticRestrictQueryCount3D>
    shared_quadratic_samples(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        app3d::QuadraticRestrictNormalSide3D side) const
    {
        if (!shared_quadratic_templates_built_) {
            throw std::runtime_error(
                "shared quadratic gridline-Cauchy restrict was not initialized");
        }
        const int size = surface_size();
        if (field.potential.size() != grid_.num_dofs()
            || value_jump.size() != size || normal_jump.size() != size) {
            throw std::invalid_argument(
                "shared quadratic restrict received incompatible field or jump sizes");
        }
        Eigen::Matrix<double, Eigen::Dynamic,
                      app3d::kSharedQuadraticRestrictQueryCount3D>
            samples(size, app3d::kSharedQuadraticRestrictQueryCount3D);
        const int side_index =
            side == app3d::QuadraticRestrictNormalSide3D::Interior ? 0 : 1;
        for (int center = 0; center < size; ++center) {
            const SharedQuadraticTraceSidePlan3D& plan =
                shared_quadratic_plans_[static_cast<std::size_t>(
                    center * 2 + side_index)];
            Eigen::Matrix<double,
                          app3d::kSharedQuadraticRestrictNodeCount3D,
                          1> continued_nodes;
            for (int q = 0;
                 q < app3d::kSharedQuadraticRestrictNodeCount3D; ++q) {
                continued_nodes[q] = field.potential[
                    plan.interpolation.grid_ids[static_cast<std::size_t>(q)]];
            }
            for (const SharedQuadraticGridlineCorrection3D& correction
                 : plan.corrections) {
                double cauchy_value = 0.0;
                if (field.direct_coefficient_cauchy) {
                    if (field.direct_known_value_jet) {
                        if (!correction.direct_known_value_ready) {
                            throw std::runtime_error(
                                "shared Q10 analytic known Dirichlet correction was not bound");
                        }
                        const app3d::ValueJet3D value_jet =
                            field.direct_known_value_jet(
                                correction.direct_patch,
                                correction.direct_u,
                                correction.direct_v,
                                correction.direct_geometry,
                                correction.direct_frame);
                        if (!value_jet.allFinite()) {
                            throw std::runtime_error(
                                "shared Q10 known Dirichlet jet is not finite");
                        }
                        cauchy_value += correction.direct_known_value_row.dot(
                            value_jet);
                    } else if (field.direct_value_c0.size() != 0) {
                        if (!correction.direct_coefficient_ready) {
                            throw std::runtime_error(
                                "shared Q10 direct value coefficient correction was not bound");
                        }
                        cauchy_value += correction.direct_value_row.dot(
                            field.direct_value_c0);
                    } else {
                        for (int q = 0;
                             q < static_cast<int>(
                                     correction.value_ids.size()); ++q) {
                            cauchy_value += correction.value_weights[q]
                                * value_jump[correction.value_ids[
                                    static_cast<std::size_t>(q)]];
                        }
                    }
                    if (field.direct_known_normal_jet) {
                        const app3d::NormalJet3D normal_jet =
                            field.direct_known_normal_jet(
                                correction.direct_patch,
                                correction.direct_u,
                                correction.direct_v,
                                correction.direct_geometry,
                                correction.direct_frame);
                        cauchy_value +=
                            correction.direct_normal_row.dot(normal_jet);
                    } else if (field.direct_normal_c0.size() != 0) {
                        if (!correction.direct_normal_coefficient_ready) {
                            throw std::runtime_error(
                                "shared Q10 direct normal coefficient correction was not bound");
                        }
                        cauchy_value +=
                            correction.direct_normal_coefficient_row.dot(
                                field.direct_normal_c0);
                    } else {
                        for (int q = 0;
                             q < static_cast<int>(
                                     correction.normal_ids.size()); ++q) {
                            cauchy_value += correction.normal_weights[q]
                                * normal_jump[correction.normal_ids[
                                    static_cast<std::size_t>(q)]];
                        }
                    }
                } else {
                    for (int q = 0;
                         q < static_cast<int>(correction.value_ids.size()); ++q) {
                        cauchy_value += correction.value_weights[q]
                            * value_jump[correction.value_ids[
                                static_cast<std::size_t>(q)]];
                    }
                    for (int q = 0;
                         q < static_cast<int>(correction.normal_ids.size()); ++q) {
                        cauchy_value += correction.normal_weights[q]
                            * normal_jump[correction.normal_ids[
                                static_cast<std::size_t>(q)]];
                    }
                }
                continued_nodes[correction.stencil_node] +=
                    static_cast<double>(correction.continuation_sign)
                    * cauchy_value;
            }
            samples.row(center) =
                (plan.interpolation.weights * continued_nodes).transpose();
        }
        if (!samples.allFinite()) {
            throw std::runtime_error(
                "shared quadratic restrict produced non-finite samples");
        }
        return samples;
    }

    Eigen::Matrix<double, Eigen::Dynamic,
                  2 * app3d::kSharedQuadraticRestrictQueryCount3D>
    shared_quadratic_continued_samples(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        bool interior_continuation) const
    {
        constexpr int layers =
            app3d::kSharedQuadraticRestrictQueryCount3D;
        const auto interior = shared_quadratic_samples(
            field, value_jump, normal_jump,
            app3d::QuadraticRestrictNormalSide3D::Interior);
        const auto exterior = shared_quadratic_samples(
            field, value_jump, normal_jump,
            app3d::QuadraticRestrictNormalSide3D::Exterior);
        Eigen::Matrix<double, Eigen::Dynamic, 2 * layers> result(
            surface_size(), 2 * layers);
        result.leftCols(layers) = interior;
        result.rightCols(layers) = exterior;
        const auto exterior_layers = app3d::shared_quadratic_signed_rho_3d(
            app3d::QuadraticRestrictNormalSide3D::Exterior,
            shared_normal_profile_);
        for (int center = 0; center < surface_size(); ++center) {
            for (int layer = 0; layer < layers; ++layer) {
                const double rho =
                    exterior_layers[static_cast<std::size_t>(layer)];
                if (interior_continuation) {
                    // At rho>0: W^- = W^+ + [W] + rho*h[W_n].
                    result(center, layers + layer) += value_jump[center]
                        + rho * h_ * normal_jump[center];
                } else {
                    // At rho<0: W^+ = W^- - [W] - rho*h[W_n].
                    // rho below is the positive layer magnitude.
                    result(center, layer) -= value_jump[center];
                    result(center, layer) +=
                        rho * h_ * normal_jump[center];
                }
            }
        }
        return result;
    }

    Eigen::VectorXd recover_shared_quadratic_joint_trace(
        const Eigen::Matrix<double, Eigen::Dynamic,
                            2 * app3d::kSharedQuadraticRestrictQueryCount3D>&
            samples,
        bool normal_derivative) const
    {
        const Eigen::Matrix<double, 1,
                            2 * app3d::kSharedQuadraticRestrictQueryCount3D>&
            weights = normal_derivative
            ? shared_quadratic_joint_c1_weights_
            : shared_quadratic_joint_c0_weights_;
        Eigen::VectorXd result(samples.rows());
        for (int center = 0; center < samples.rows(); ++center)
            result[center] = weights.dot(samples.row(center));
        if (normal_derivative)
            result /= h_;
        return result;
    }

    Eigen::MatrixXd continued_samples(const HarmonicJetField3D& field,
                                      const Eigen::VectorXd& value_jump,
                                      const Eigen::VectorXd& normal_jump,
                                      bool interior_continuation,
                                      ExteriorNormalRestrictMode3D mode =
                                          ExteriorNormalRestrictMode3D::
                                              JointTricubicCauchy) const
    {
        const bool use_crossing_owner =
            trace_restrict_uses_crossing_owner(mode);
        const bool use_global_exterior_branch =
            mode == ExteriorNormalRestrictMode3D::
                        GlobalCubicExteriorBranchCrossingOwner;
        if (use_crossing_owner && !crossing_owner_templates_built_) {
            throw std::runtime_error(
                "crossing-owner normal restrict was not initialized");
        }
        const int size = surface_size();
        const bool use_local_quadratic_restrict =
            use_global_exterior_branch
            && field.restrict_value_jets.rows() == size;
        if (field.potential.size() != grid_.num_dofs()
            || field.coefficients.rows() != size
            || value_jump.size() != size
            || normal_jump.size() != size) {
            throw std::invalid_argument(
                "exterior trace received incompatible field or jump sizes");
        }
        if ((field.restrict_value_jets.size() != 0
             && (field.restrict_value_jets.rows() != size
                 || field.restrict_value_jets.cols() != 6))
            || (use_local_quadratic_restrict
                && (field.restrict_normal_coefficients.rows() != size
                    || field.restrict_normal_coefficients.cols()
                           != fit_.dimension()))) {
            throw std::invalid_argument(
                "local quadratic restrict received malformed field jets");
        }

        Eigen::MatrixXd samples = Eigen::MatrixXd::Zero(
            size, kRestrictNormalSampleCount);
        for (int center = 0; center < size; ++center) {
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0;
                     layer < kRestrictNormalLayerCount;
                     ++layer) {
                    const HarmonicTraceSample3D& sample = trace_samples_[
                        trace_sample_index(center, side, layer)];
                    double value = 0.0;
                    for (int q = 0; q < kRestrictGridPointCount; ++q) {
                        value += sample.weights[static_cast<std::size_t>(q)]
                               * field.potential[
                                   sample.grid_ids[static_cast<std::size_t>(q)]];
                    }
                    if (use_crossing_owner) {
                        for (const HarmonicTraceCorrectionTerm3D& term
                             : sample.owner_corrections) {
                            if (use_local_quadratic_restrict) {
                                value += term.value_jet_evaluation.dot(
                                    field.restrict_value_jets.row(
                                        term.owner_dof));
                                value += term.evaluation.dot(
                                    field.restrict_normal_coefficients.row(
                                        term.owner_dof).transpose());
                            } else {
                                value += term.evaluation.dot(
                                    field.coefficients.row(
                                        term.owner_dof).transpose());
                            }
                        }
                    } else {
                        if (use_local_quadratic_restrict) {
                            value += sample.legacy_value_jet_evaluation.dot(
                                field.restrict_value_jets.row(center));
                            value +=
                                sample.legacy_correction_evaluation.dot(
                                    field.restrict_normal_coefficients.row(
                                        center).transpose());
                        } else {
                            value += sample.legacy_correction_evaluation.dot(
                                field.coefficients.row(center).transpose());
                        }
                    }
                    samples(
                        center,
                        kRestrictNormalLayerCount * side + layer) = value;
                }
            }

            if (use_global_exterior_branch) {
                if (kRestrictGridDegree < kCauchyPolynomialDegree) {
                    throw std::runtime_error(
                        "global exterior-branch restrict requires the "
                        "Cartesian interpolation degree to be at least the "
                        "Cauchy polynomial degree");
                }
                // The negative templates are on the interior branch and the
                // positive templates are on the exterior branch.  Switch the
                // opposite half-ray with the complete, anchored patch-owned
                // Cauchy polynomial.  This supplies two independent global
                // reconstructions, so the jump route remains a real audit.
                for (int layer = 0;
                     layer < kRestrictNormalLayerCount;
                     ++layer) {
                    double negative_jump = 0.0;
                    double positive_jump = 0.0;
                    if (use_local_quadratic_restrict) {
                        const SurfaceDof& dof = cloud_.dofs[
                            static_cast<std::size_t>(center)];
                        const LocalRestrictCauchyPlan3D& plan =
                            local_restrict_plans_[
                                static_cast<std::size_t>(center)];
                        const double distance =
                            normal_layers_[static_cast<std::size_t>(layer)]
                            * h_;
                        negative_jump = app3d::pweights_general_3d(
                            -distance * dof.normal,
                            plan.frame,
                            plan.graph_hessian).w0.dot(
                                field.restrict_value_jets.row(center));
                        positive_jump = app3d::pweights_general_3d(
                            distance * dof.normal,
                            plan.frame,
                            plan.graph_hessian).w0.dot(
                                field.restrict_value_jets.row(center));
                        negative_jump +=
                            negative_normal_ray_basis_.row(layer).dot(
                                field.restrict_normal_coefficients.row(center));
                        positive_jump +=
                            positive_normal_ray_basis_.row(layer).dot(
                                field.restrict_normal_coefficients.row(center));
                    }
                    if (interior_continuation) {
                        samples(
                            center,
                            kRestrictNormalLayerCount + layer) +=
                                use_local_quadratic_restrict
                                    ? positive_jump
                                    : positive_normal_ray_basis_.row(layer).dot(
                                          field.coefficients.row(center));
                    } else {
                        samples(center, layer) -=
                            use_local_quadratic_restrict
                                ? negative_jump
                                : negative_normal_ray_basis_.row(layer).dot(
                                      field.coefficients.row(center));
                    }
                }
                continue;
            }

            if (interior_continuation) {
                // At rho=+tau*h, W^- = W^+ + [W] + rho[W_n].
                for (int layer = 0;
                     layer < kRestrictNormalLayerCount;
                     ++layer) {
                    samples(
                        center,
                        kRestrictNormalLayerCount + layer) += value_jump[center];
                    samples(
                        center,
                        kRestrictNormalLayerCount + layer) +=
                        normal_layers_[static_cast<std::size_t>(layer)]
                        * h_ * normal_jump[center];
                }
            } else {
                // At rho=-tau*h, W^+ = W^- - [W] - rho[W_n].
                for (int layer = 0;
                     layer < kRestrictNormalLayerCount;
                     ++layer) {
                    samples(center, layer) -= value_jump[center];
                    samples(center, layer) +=
                        normal_layers_[static_cast<std::size_t>(layer)]
                        * h_ * normal_jump[center];
                }
            }
        }
        return samples;
    }

    Eigen::VectorXd recover_trace(
        const Eigen::MatrixXd& samples,
        const std::array<double, kRestrictNormalSampleCount>& weights,
        double scale) const
    {
        Eigen::VectorXd trace(samples.rows());
        for (int center = 0; center < samples.rows(); ++center) {
            double value = 0.0;
            for (int q = 0; q < kRestrictNormalSampleCount; ++q)
                value += weights[static_cast<std::size_t>(q)]
                       * samples(center, q);
            trace[center] = scale * value;
        }
        return trace;
    }

    std::size_t trace_sample_index(int center, int side, int layer) const
    {
        return static_cast<std::size_t>(
            (center * 2 + side) * kRestrictNormalLayerCount + layer);
    }

    Eigen::Vector3d local_coordinate(int center,
                                     const Eigen::Vector3d& point) const
    {
        const SurfaceDof& dof = cloud_.dofs[static_cast<std::size_t>(center)];
        const Eigen::Vector3d displacement = (point - dof.point) / h_;
        return {displacement.dot(dof.tangent1),
                displacement.dot(dof.tangent2),
                displacement.dot(dof.normal)};
    }

    int surface_dof_for_crossing(const P2CrossingOwner3D& owner,
                                 const Eigen::Vector3d& point) const
    {
        int patch = owner.nurbs_patch_index;
        Eigen::Vector2d uv = owner.nurbs_parameter;
        if (patch >= 0) {
            if (patch >= static_cast<int>(native_surface_.patches.size())) {
                throw std::runtime_error(
                    "native crossing patch index is outside the readiness NURBS surface");
            }
            if (!uv.allFinite()) {
                throw std::runtime_error(
                    "native crossing parameters are not finite");
            }
        }
        if (patch < 0) {
            const geometry3d::NurbsParamTriangle3D* triangle = nullptr;
            Eigen::Vector3d barycentric = Eigen::Vector3d::Zero();
            if (owner.geometry_panel_index >= 0
                && owner.geometry_panel_index
                       < static_cast<int>(geometry_triangles_.size())) {
                triangle = &geometry_triangles_[
                    static_cast<std::size_t>(owner.geometry_panel_index)];
                barycentric = owner.geometry_barycentric;
            } else if (owner.panel_index >= 0
                       && owner.panel_index
                              < static_cast<int>(correction_triangles_.size())) {
                triangle = &correction_triangles_[
                    static_cast<std::size_t>(owner.panel_index)];
                barycentric = owner.barycentric;
            }
            if (triangle == nullptr) {
                throw std::runtime_error(
                    "crossing has no native NURBS triangle owner");
            }
            patch = triangle->patch_index;
            uv = app3d::interpolate_triangle_parameter(
                *triangle, barycentric);
        }
        const std::array<int, 4> candidates =
            app3d::parameter_dof_candidates_2x2(
                native_surface_, cloud_, patch, uv.x(), uv.y());
        const geometry3d::NurbsSurfaceDerivatives3D derivatives =
            native_surface_.patches[
                static_cast<std::size_t>(patch)]
                .evaluate_with_derivatives(uv.x(), uv.y());
        Eigen::Vector3d reference_normal =
            derivatives.du.cross(derivatives.dv).normalized();
        int nearest = -1;
        double best = std::numeric_limits<double>::infinity();
        for (int pass = 0; pass < 2 && nearest < 0; ++pass) {
            for (int q : candidates) {
                const SurfaceDof& dof = cloud_.dofs[static_cast<std::size_t>(q)];
                if (pass == 0 && dof.normal.dot(reference_normal) < 0.50) {
                    continue;
                }
                const double distance = (dof.point - point).squaredNorm();
                if (distance < best) {
                    best = distance;
                    nearest = q;
                }
            }
        }
        if (nearest < 0)
            throw std::runtime_error(
                "failed to associate a crossing with four parameter DOFs");
        return nearest;
    }

    SurfaceDof crossing_frame(const P2CrossingOwner3D& owner,
                              const Eigen::Vector3d& hit,
                              int fallback_center) const
    {
        SurfaceDof result =
            cloud_.dofs[static_cast<std::size_t>(fallback_center)];
        result.point = hit;
        if (owner.nurbs_patch_index < 0
            || owner.nurbs_patch_index
                   >= static_cast<int>(native_surface_.patches.size())
            || !owner.nurbs_parameter.allFinite()) {
            return result;
        }
        const geometry3d::NurbsSurfaceDerivatives3D derivatives =
            native_surface_.patches[
                static_cast<std::size_t>(owner.nurbs_patch_index)]
                .evaluate_with_derivatives(
                    owner.nurbs_parameter.x(), owner.nurbs_parameter.y());
        Eigen::Vector3d normal = derivatives.du.cross(derivatives.dv);
        const double normal_norm = normal.norm();
        if (!(normal_norm > 1.0e-14) || !std::isfinite(normal_norm))
            throw std::runtime_error(
                "crossing-local Cauchy frame has a degenerate normal");
        normal /= normal_norm;
        if (owner.crossing_normal.allFinite()
            && owner.crossing_normal.norm() > 0.5
            && normal.dot(owner.crossing_normal) < 0.0) {
            normal = -normal;
        }
        Eigen::Vector3d tangent1 =
            derivatives.du - derivatives.du.dot(normal) * normal;
        if (tangent1.norm() <= 1.0e-14) {
            tangent1 = derivatives.dv
                     - derivatives.dv.dot(normal) * normal;
        }
        const double tangent_norm = tangent1.norm();
        if (!(tangent_norm > 1.0e-14) || !std::isfinite(tangent_norm))
            throw std::runtime_error(
                "crossing-local Cauchy frame has a degenerate tangent");
        tangent1 /= tangent_norm;
        Eigen::Vector3d tangent2 = normal.cross(tangent1).normalized();
        result.normal = normal;
        result.tangent1 = tangent1;
        result.tangent2 = tangent2;
        result.patch_id = owner.nurbs_patch_index;
        result.u = owner.nurbs_parameter.x();
        result.v = owner.nurbs_parameter.y();
        return result;
    }

    Eigen::Vector3d local_coordinate(
        const SurfaceDof& frame,
        const Eigen::Vector3d& point) const
    {
        const Eigen::Vector3d displacement = (point - frame.point) / h_;
        return {displacement.dot(frame.tangent1),
                displacement.dot(frame.tangent2),
                displacement.dot(frame.normal)};
    }

    std::vector<int> crossing_cauchy_dofs(
        const Eigen::Vector3d& crossing,
        int owner_patch,
        int requested_count) const
    {
        if (requested_count == 0)
            return {};
        if (owner_patch < 0
            || owner_patch >= static_cast<int>(native_surface_.patches.size())) {
            throw std::runtime_error(
                "crossing-local Cauchy fit has no native owner patch");
        }

        // A crossing at a sharp C0 seam has two legitimate one-sided jets.
        // Use the canonical intersection owner and traverse G1 connections
        // only.  Thus cylinder side-patch seams share samples, whereas cap
        // rims and L-prism creases never mix normal data across the corner.
        std::vector<unsigned char> visited(
            native_surface_.patches.size(), 0);
        visited[static_cast<std::size_t>(owner_patch)] = 1;
        std::vector<int> included{owner_patch};
        std::vector<int> frontier{owner_patch};
        int candidate_count =
            cloud_.patches[static_cast<std::size_t>(owner_patch)].dof_count();
        bool expanded_one_ring = false;
        while ((!expanded_one_ring || candidate_count < requested_count)
               && !frontier.empty()) {
            std::vector<int> next;
            for (int patch : frontier) {
                for (int neighbor : app3d::smooth_patch_neighbors_3d(
                         native_surface_, patch)) {
                    if (neighbor < 0
                        || neighbor
                               >= static_cast<int>(native_surface_.patches.size())) {
                        throw std::runtime_error(
                            "crossing-local G1 topology has an invalid patch");
                    }
                    if (!visited[static_cast<std::size_t>(neighbor)]) {
                        visited[static_cast<std::size_t>(neighbor)] = 1;
                        next.push_back(neighbor);
                    }
                }
            }
            std::sort(next.begin(), next.end());
            next.erase(std::unique(next.begin(), next.end()), next.end());
            for (int patch : next) {
                included.push_back(patch);
                candidate_count += cloud_.patches[
                    static_cast<std::size_t>(patch)].dof_count();
            }
            frontier = std::move(next);
            expanded_one_ring = true;
        }

        std::vector<std::pair<double, int>> candidates;
        candidates.reserve(static_cast<std::size_t>(candidate_count));
        for (int patch : included) {
            const auto& tensor =
                cloud_.patches[static_cast<std::size_t>(patch)];
            for (int local = 0; local < tensor.dof_count(); ++local) {
                const int sample = tensor.first_dof + local;
                candidates.emplace_back(
                    (cloud_.dofs[static_cast<std::size_t>(sample)].point
                     - crossing).squaredNorm(),
                    sample);
            }
        }
        const int selected = std::min(
            requested_count, static_cast<int>(candidates.size()));
        if (selected <= 0)
            throw std::runtime_error(
                "crossing-local Cauchy neighborhood is empty");
        const auto candidate_less = [](const auto& lhs, const auto& rhs) {
            if (lhs.first < rhs.first)
                return true;
            if (rhs.first < lhs.first)
                return false;
            return lhs.second < rhs.second;
        };
        // Keep the legacy complete ordering in this low-risk grouping patch.
        // A partial-selection optimization should be validated independently.
        std::sort(candidates.begin(), candidates.end(), candidate_less);
        std::vector<int> result;
        result.reserve(static_cast<std::size_t>(selected));
        for (int k = 0; k < selected; ++k)
            result.push_back(candidates[static_cast<std::size_t>(k)].second);
        return result;
    }

    HarmonicCrossingDirectPlan3D build_direct_crossing_plan(
        const P2CrossingOwner3D& owner,
        const Eigen::Vector3d& hit,
        int center) const
    {
        HarmonicCrossingDirectPlan3D result;
        result.frame = crossing_frame(owner, hit, center);
        result.value_ids = crossing_cauchy_dofs(
            hit, result.frame.patch_id, stencils_.value_count);
        result.normal_ids = crossing_cauchy_dofs(
            hit, result.frame.patch_id, stencils_.derivative_count);
        const int value_count = static_cast<int>(result.value_ids.size());
        const int normal_count = static_cast<int>(result.normal_ids.size());
        const int rows = value_count + normal_count;
        const int dimension = fit_.dimension();
        if (rows < dimension)
            throw std::runtime_error(
                "crossing-local Cauchy fit has too few conditions");

        Eigen::MatrixXd design(rows, dimension);
        Eigen::VectorXd sqrt_weights(rows);
        for (int k = 0; k < value_count; ++k) {
            const SurfaceDof& sample = cloud_.dofs[static_cast<std::size_t>(
                result.value_ids[static_cast<std::size_t>(k)])];
            const Eigen::Vector3d xi =
                local_coordinate(result.frame, sample.point);
            design.row(k) = fit_.space()
                                .basis(xi.x(), xi.y(), xi.z())
                                .transpose();
            sqrt_weights[k] = std::sqrt(
                1.0 / std::pow(0.35 + xi.norm(), 2.0));
        }
        for (int k = 0; k < normal_count; ++k) {
            const SurfaceDof& sample = cloud_.dofs[static_cast<std::size_t>(
                result.normal_ids[static_cast<std::size_t>(k)])];
            const Eigen::Vector3d xi =
                local_coordinate(result.frame, sample.point);
            const Eigen::Vector3d normal_components(
                sample.normal.dot(result.frame.tangent1),
                sample.normal.dot(result.frame.tangent2),
                sample.normal.dot(result.frame.normal));
            design.row(value_count + k) =
                normal_components.transpose()
                * fit_.space().gradient(xi.x(), xi.y(), xi.z());
            sqrt_weights[value_count + k] = std::sqrt(
                0.85 / std::pow(0.35 + xi.norm(), 2.0));
        }

        const Eigen::MatrixXd weighted = sqrt_weights.asDiagonal() * design;
        Eigen::JacobiSVD<Eigen::MatrixXd> condition_svd(
            weighted, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorXd singular = condition_svd.singularValues();
        if (singular.size() != dimension || singular.size() == 0
            || !singular.allFinite() || !(singular[0] > 0.0)
            || !(singular[singular.size() - 1]
                 > 3.0e-12 * singular[0])) {
            throw std::runtime_error(
                "crossing-local Cauchy fit is rank deficient");
        }
        const Eigen::MatrixXd pinv =
            svd_pseudoinverse_from_decomposition_3d(condition_svd, 3.0e-12);
        result.value_map.resize(dimension, value_count);
        result.normal_map.resize(dimension, normal_count);
        for (int k = 0; k < value_count; ++k)
            result.value_map.col(k) = pinv.col(k) * sqrt_weights[k];
        for (int k = 0; k < normal_count; ++k) {
            result.normal_map.col(k) =
                pinv.col(value_count + k)
                * sqrt_weights[value_count + k] * h_;
        }
        result.condition = singular[0] / singular[singular.size() - 1];
        return result;
    }

    HarmonicCrossingDirectRow3D compose_direct_crossing_row(
        const LaplaceCrossingCorrectionOp& op,
        const HarmonicCrossingDirectPlan3D& plan) const
    {
        const Eigen::Vector3d xi = local_coordinate(
            plan.frame, grid_point(grid_, op.correction_node));
        const Eigen::VectorXd evaluation =
            fit_.space().basis(xi.x(), xi.y(), xi.z());

        HarmonicCrossingDirectRow3D result;
        result.rhs_node = op.rhs_node;
        result.scale =
            static_cast<double>(op.side_delta) * op.stencil_weight;
        result.value_ids = plan.value_ids;
        result.normal_ids = plan.normal_ids;
        result.value_weights = plan.value_map.transpose() * evaluation;
        result.normal_weights = plan.normal_map.transpose() * evaluation;
        result.condition = plan.condition;
        return result;
    }

    SharedQuadraticGridlineCorrection3D
    compose_shared_quadratic_correction(
        int stencil_slot,
        int grid_node,
        const HarmonicCrossingDirectPlan3D& plan,
        double distance_over_h) const
    {
        const Eigen::Vector3d xi = local_coordinate(
            plan.frame, grid_point(grid_, grid_node));
        const Eigen::VectorXd evaluation =
            fit_.space().basis(xi.x(), xi.y(), xi.z());
        SharedQuadraticGridlineCorrection3D result;
        result.stencil_node = stencil_slot;
        result.value_ids = plan.value_ids;
        result.normal_ids = plan.normal_ids;
        result.value_weights = plan.value_map.transpose() * evaluation;
        result.normal_weights = plan.normal_map.transpose() * evaluation;
        result.cauchy_condition = plan.condition;
        result.gridline_distance_over_h = distance_over_h;
        return result;
    }

    app3d::G1PatchTopology3D shared_quadratic_g1_topology() const
    {
        app3d::G1PatchTopology3D result;
        result.surface_component_by_patch = native_surface_.patch_components;
        result.g1_neighbors.resize(native_surface_.patches.size());
        if ((!native_surface_.smooth_neighbors.empty()
             && native_surface_.smooth_neighbors.size()
                    != native_surface_.patches.size())
            || native_surface_.patch_components.size()
                   != native_surface_.patches.size()) {
            throw std::runtime_error(
                "shared quadratic restrict requires complete native G1 topology");
        }
        for (int patch = 0;
             patch < static_cast<int>(native_surface_.patches.size()); ++patch) {
            result.g1_neighbors[static_cast<std::size_t>(patch)] =
                app3d::smooth_patch_neighbors_3d(native_surface_, patch);
        }
        return result;
    }

    void build_shared_quadratic_gridline_catalog()
    {
        std::set<std::tuple<int, int, int>> seen;
        for (const LaplaceCrossingCorrectionOp& op
             : correction_support_.crossing_ops) {
            const int lo = std::min(op.rhs_node, op.correction_node);
            const int hi = std::max(op.rhs_node, op.correction_node);
            const int ordinal = has_grid_edge_event_3d(op)
                ? op.grid_edge_event.ordinal : -1;
            if (!seen.emplace(lo, hi, ordinal).second)
                continue;
            const P2CrossingOwner3D owner =
                laplace_crossing_owner_3d(grid_pair_, op);
            if (owner.status != P2CrossingOwnerStatus3D::ExactIntersection) {
                throw std::runtime_error(
                    "shared quadratic restrict refuses a non-exact grid-edge event owner");
            }
            if (owner.nurbs_patch_index < 0
                || owner.surface_component < 0
                || !owner.crossing_point.allFinite()) {
                throw std::runtime_error(
                    "shared quadratic restrict requires exact native gridline crossings");
            }
            SharedQuadraticGridlineCrossing3D crossing;
            crossing.lo = lo;
            crossing.hi = hi;
            crossing.event_id = op.grid_edge_event;
            crossing.owner = owner;
            crossing.point = owner.crossing_point;
            shared_quadratic_gridline_crossings_.push_back(
                std::move(crossing));
        }
        if (shared_quadratic_gridline_crossings_.empty()) {
            throw std::runtime_error(
                "shared quadratic restrict found no Cartesian gridline crossings");
        }
        shared_quadratic_gridline_records_.reserve(
            shared_quadratic_gridline_crossings_.size());
        for (int q = 0;
             q < static_cast<int>(
                     shared_quadratic_gridline_crossings_.size()); ++q) {
            const SharedQuadraticGridlineCrossing3D& crossing =
                shared_quadratic_gridline_crossings_[
                    static_cast<std::size_t>(q)];
            shared_quadratic_gridline_records_.push_back({
                q,
                crossing.owner.nurbs_patch_index,
                crossing.owner.surface_component,
                crossing.point});
        }
        shared_quadratic_diagnostics_.unique_gridline_crossings =
            shared_quadratic_gridline_crossings_.size();
    }

    std::vector<SharedQuadraticGridlineCorrection3D>
    build_all_event_trace_corrections(
        int center,
        bool desired_inside,
        const std::vector<int>& grid_ids,
        const geometry3d::NurbsSurfaceIntersector3D& intersector,
        const app3d::G1PatchTopology3D& topology) const
    {
        const SurfaceDof& dof =
            cloud_.dofs[static_cast<std::size_t>(center)];
        if (grid_ids.empty()) {
            throw std::invalid_argument(
                "all-event trace correction received an empty cover");
        }
        std::vector<SharedQuadraticGridlineCorrection3D> result;
        const double tie_tolerance = std::max(
            16.0 * intersector.geometry_tolerance(), 1.0e-12 * h_);
        for (int stencil_node = 0;
             stencil_node < static_cast<int>(grid_ids.size());
             ++stencil_node) {
            const int node = grid_ids[static_cast<std::size_t>(stencil_node)];
            const bool node_inside = grid_pair_.domain_label(node) > 0;
            if (node_inside != desired_inside)
                ++shared_quadratic_diagnostics_.wrong_side_nodes;
            const Eigen::Vector3d node_point = grid_point(grid_, node);
            geometry3d::NurbsSurfaceIntersectionResult3D intersection;
            app3d::SegmentPhysicalEventSequence3D sequence;
            if (native_endpoint_paths_) {
                const std::pair<int, int> cache_key{center, node};
                auto cached = native_endpoint_path_cache_.find(cache_key);
                if (cached == native_endpoint_path_cache_.end()) {
                    geometry3d::NativeEndpointQueryOptions3D options;
                    options.point_tolerance = intersector.geometry_tolerance();
                    options.expected_start_inside = node_inside;
                    ++shared_quadratic_diagnostics_.segment_queries;
                    auto path = intersector.intersect_segment_to_native_endpoint(
                        node_point, {dof.patch_id, dof.u, dof.v},
                        options, native_endpoint_budget_);
                    if (path.status != geometry3d::PathStatus3D::Certified) {
                        ++shared_quadratic_diagnostics_.segment_fallbacks;
                        std::ostringstream message;
                        message << std::setprecision(17)
                                << "native endpoint support path failed; center="
                                << center << " grid_node=" << node
                                << " stencil_slot=" << stencil_node
                                << " side=" << (desired_inside ? "interior" : "exterior")
                                << " status="
                                << geometry3d::native_endpoint_path_status_name_3d(path.status)
                                << " reason=" << path.diagnostics.reason
                                << " message=" << path.message << '\n' << path.dump;
                        throw std::runtime_error(message.str());
                    }
                    // Failure dumps are retained above. Successful certificates
                    // are reused by both sides without retaining verbose text
                    // for every cover node at N=64/128.
                    path.dump.clear();
                    path.dump.shrink_to_fit();
                    cached = native_endpoint_path_cache_.emplace(
                        cache_key, std::move(path)).first;
                }
                auto selected = app3d::select_native_endpoint_path_events_3d(
                    cached->second, node_inside, desired_inside);
                intersection = std::move(selected.intersection);
                sequence = std::move(selected.sequence);
            } else {
            // Legacy explicit A/B route: it is never an automatic fallback
            // from the native certificate path.
            const Eigen::Vector3d support_to_trace = dof.point - node_point;
            const double support_to_trace_length = support_to_trace.norm();
            if (!std::isfinite(support_to_trace_length)
                || support_to_trace_length <= tie_tolerance) {
                throw std::runtime_error(
                    "shared quadratic restrict has a degenerate support-to-trace segment");
            }
            const Eigen::Vector3d segment_direction =
                support_to_trace / support_to_trace_length;
            const auto& endpoint_patch = native_surface_.patches[
                static_cast<std::size_t>(dof.patch_id)];
            const double parameter_scale = std::max(
                {1.0,
                 endpoint_patch.domain_end_u()
                     - endpoint_patch.domain_start_u(),
                 endpoint_patch.domain_end_v()
                     - endpoint_patch.domain_start_v()});
            // Match the intersector's native owner resolution.  This is only
            // an endpoint-identity test; distinct nearby roots with different
            // owners remain separate physical events.
            const double endpoint_identity_parameter_tolerance =
                16.0e-12 * parameter_scale;

            // Make the known trace endpoint a strict interior root of the
            // certified query.  Thus the old retreat gap is covered without
            // asking the intersector to certify a root on its own endpoint.
            // All roots beyond the trace point are classified explicitly and
            // excluded only because they lie outside the support-to-trace
            // path; any unresolved region anywhere in this extended query
            // still fails closed.
            const double endpoint_extension = 4.0 * tie_tolerance;
            const Eigen::Vector3d certified_end =
                dof.point + endpoint_extension * segment_direction;
            const std::pair<int, int> cache_key{center, node};
            auto cached_intersection =
                all_event_intersection_cache_.find(cache_key);
            if (cached_intersection == all_event_intersection_cache_.end()) {
                ++shared_quadratic_diagnostics_.segment_queries;
                cached_intersection = all_event_intersection_cache_.emplace(
                    cache_key,
                    intersector.intersect_segment(node_point, certified_end))
                                          .first;
            }
            // Interior/exterior branches and nested Q27/Q64 covers use the
            // identical geometric support path.  Cache its certified raw
            // event set once; branch continuation and endpoint handling below
            // remain independent.
            const geometry3d::NurbsSurfaceIntersectionResult3D&
                certified_intersection = cached_intersection->second;
            const app3d::SegmentEndpointPartition3D endpoint_partition =
                app3d::partition_certified_segment_endpoint_3d(
                    certified_intersection, node_point, dof.point,
                    certified_end, dof.patch_id, dof.u, dof.v,
                    tie_tolerance,
                    endpoint_identity_parameter_tolerance);
            if (endpoint_partition.kind
                != app3d::SegmentEndpointPartitionKind3D::Certified) {
                ++shared_quadratic_diagnostics_.segment_fallbacks;
                std::ostringstream message;
                message << std::setprecision(17)
                        << "shared quadratic restrict requires a complete "
                           "extended support-to-trace root set with one "
                           "native exact endpoint"
                        << "; center=" << center
                        << " side="
                        << (desired_inside ? "interior" : "exterior")
                        << " stencil_slot=" << stencil_node
                        << " grid_node=" << node
                        << " partition_kind="
                        << static_cast<int>(endpoint_partition.kind)
                        << " unresolved="
                        << certified_intersection.diagnostics.
                               unresolved_candidates
                        << " ambiguous_clusters="
                        << certified_intersection.diagnostics.
                               ambiguous_root_clusters
                        << " endpoint_extension=" << endpoint_extension;
                throw std::runtime_error(message.str());
            }
            intersection.diagnostics = certified_intersection.diagnostics;
            intersection.overlap_detected =
                certified_intersection.overlap_detected;
            intersection.crossings.reserve(
                endpoint_partition.open_crossing_indices.size());
            for (int crossing_index :
                 endpoint_partition.open_crossing_indices) {
                geometry3d::NurbsSurfaceCrossing3D crossing =
                    certified_intersection.crossings[
                        static_cast<std::size_t>(crossing_index)];
                const double distance_along =
                    (crossing.point - node_point).dot(segment_direction);
                if (!std::isfinite(distance_along)
                    || !(distance_along < support_to_trace_length)) {
                    ++shared_quadratic_diagnostics_.segment_fallbacks;
                    throw std::runtime_error(
                        "shared quadratic restrict endpoint partition retained a non-open event");
                }
                crossing.edge_parameter =
                    distance_along / support_to_trace_length;
                intersection.crossings.push_back(std::move(crossing));
            }
            sequence =
                app3d::select_certified_segment_physical_events_3d(
                    intersection, node_point, dof.point, tie_tolerance,
                    native_surface_.exact_inside);
            app3d::OpenSegmentContinuation3D continuation =
                app3d::evaluate_open_segment_continuation_3d(
                    sequence, node_inside, desired_inside);
            if (continuation.kind
                    != app3d::OpenSegmentContinuationKind3D::Complete
                && continuation.kind
                    != app3d::OpenSegmentContinuationKind3D::
                           ExactEndpointRequired) {
                ++shared_quadratic_diagnostics_.segment_fallbacks;
                std::ostringstream message;
                message
                    << "shared quadratic restrict could not certify the full "
                       "support-to-trace event sequence (kind="
                    << static_cast<int>(sequence.kind)
                    << ", continuation_kind="
                    << static_cast<int>(continuation.kind)
                    << ", start_inside=" << node_inside
                    << ", desired_inside=" << desired_inside
                    << ", roots=" << sequence.input_crossing_count
                    << ", physical_events="
                    << sequence.physical_event_count
                    << ", partition_open="
                    << endpoint_partition.open_crossing_indices.size()
                    << ", partition_post="
                    << endpoint_partition.post_endpoint_crossing_indices.size()
                    << ", overlap=" << intersection.overlap_detected
                    << ", unresolved="
                    << intersection.diagnostics.unresolved_candidates
                    << ", ambiguous_clusters="
                    << intersection.diagnostics.ambiguous_root_clusters
                    << ", center=" << center
                    << ", side="
                    << (desired_inside ? "interior" : "exterior")
                    << ", stencil_slot=" << stencil_node
                    << ", grid_node=" << node;
                for (const auto& event : sequence.events) {
                    message << "; event(t=" << event.edge_parameter
                            << ",sign=" << event.continuation_sign
                            << ",feature=" << event.feature_edge_contact
                            << ')';
                }
                message << ')';
                throw std::runtime_error(message.str());
            }

            // The exact trace endpoint is a one-sided limiting event.  It is
            // needed only when the certified open-segment events have not
            // already continued the support value onto the requested side.
            // In particular, an equal-label double crossing contributes both
            // interior events and must not receive an extra endpoint toggle.
            if (continuation.exact_endpoint_required()) {
                const double endpoint_boundary_parameter_tolerance =
                    64.0 * std::numeric_limits<double>::epsilon()
                    * parameter_scale;
                const bool endpoint_on_patch_boundary =
                    std::abs(dof.u - endpoint_patch.domain_start_u())
                            <= endpoint_boundary_parameter_tolerance
                    || std::abs(dof.u - endpoint_patch.domain_end_u())
                            <= endpoint_boundary_parameter_tolerance
                    || std::abs(dof.v - endpoint_patch.domain_start_v())
                            <= endpoint_boundary_parameter_tolerance
                    || std::abs(dof.v - endpoint_patch.domain_end_v())
                            <= endpoint_boundary_parameter_tolerance;
                if (endpoint_on_patch_boundary) {
                    ++shared_quadratic_diagnostics_.segment_fallbacks;
                    throw std::runtime_error(
                        "shared quadratic restrict refuses a trace-endpoint event on a patch boundary without a complete incident-owner catalog");
                }
                geometry3d::NurbsSurfaceCrossing3D exact_endpoint;
                exact_endpoint.patch_index = dof.patch_id;
                exact_endpoint.component = native_surface_.patch_components[
                    static_cast<std::size_t>(dof.patch_id)];
                exact_endpoint.u = dof.u;
                exact_endpoint.v = dof.v;
                exact_endpoint.edge_parameter = 1.0;
                exact_endpoint.point = dof.point;
                exact_endpoint.normal = dof.normal;
                exact_endpoint.residual = 0.0;
                exact_endpoint.transversality =
                    std::abs(dof.normal.dot(segment_direction));
                // The exact point removes location error but cannot make a
                // grazing endpoint orientable, so tangent paths fail closed.
                exact_endpoint.reliable_transversality_tolerance = std::max(
                    1.0e-10,
                    tie_tolerance / support_to_trace_length);
                intersection.crossings.push_back(std::move(exact_endpoint));
                sequence = app3d::select_certified_segment_physical_events_3d(
                    intersection, node_point, dof.point, tie_tolerance,
                    native_surface_.exact_inside);
                // This second sequence now includes the closed trace endpoint
                // at t=1, so the half-open continuation helper deliberately
                // does not apply.  The ordered event loop below validates all
                // binary transitions and the final requested side.
                if (sequence.kind
                        != app3d::SegmentPhysicalEventSequenceKind3D::
                               Certified
                    || sequence.events.empty()) {
                    ++shared_quadratic_diagnostics_.segment_fallbacks;
                    throw std::runtime_error(
                        "shared quadratic restrict could not certify the conditional trace-endpoint event");
                }
                const auto& endpoint_event = sequence.events.back();
                const double endpoint_parameter_tolerance =
                    tie_tolerance / support_to_trace_length;
                if (std::abs(endpoint_event.edge_parameter - 1.0)
                            > endpoint_parameter_tolerance
                    || endpoint_event.continuation_sign
                           != continuation.required_endpoint_sign) {
                    ++shared_quadratic_diagnostics_.segment_fallbacks;
                    throw std::runtime_error(
                        "shared quadratic restrict conditional endpoint has the wrong event orientation");
                }
            }
            } // explicit legacy support-path route
            if (sequence.events.size() > 1)
                ++shared_quadratic_diagnostics_.multiple_root_selections;

            int continuation_state = node_inside ? 1 : 0;
            for (const app3d::SegmentPhysicalCrossingEvent3D& event
                 : sequence.events) {
                const int next_state =
                    continuation_state + event.continuation_sign;
                if (next_state < 0 || next_state > 1) {
                    ++shared_quadratic_diagnostics_.segment_fallbacks;
                    throw std::runtime_error(
                        "shared quadratic restrict event signs do not form a valid inside/outside path");
                }
                continuation_state = next_state;
                if (native_endpoint_paths_ ? event.edge_parameter == 1.0
                    : (event.point - dof.point).norm() <= tie_tolerance)
                    ++shared_quadratic_diagnostics_.trace_endpoint_selections;
                if (event.crossing_indices.size() != 1) {
                    ++shared_quadratic_diagnostics_.segment_fallbacks;
                    throw std::logic_error(
                        "shared quadratic restrict received a noncanonical physical event");
                }
                const auto& physical_root = intersection.crossings[
                    static_cast<std::size_t>(
                        event.crossing_indices.front())];
                const std::size_t physical_owner_count =
                    physical_root.owners.empty()
                    ? std::size_t{1}
                    : physical_root.owners.size();
                if (event.feature_edge_contact
                    || physical_owner_count > 1) {
                    ++shared_quadratic_diagnostics_.feature_root_selections;
                }
                if (event.continuation_sign == 0) {
                    // exact_inside certified a physical C0 contact without
                    // an inside/outside branch change.  Keep it in the
                    // ordered all-event audit, but it carries no jump and
                    // therefore contributes no correction row.
                    continue;
                }

                const auto make_owner =
                    [&](const geometry3d::NurbsSurfaceRootOwner3D* owner) {
                        P2CrossingOwner3D result_owner;
                        result_owner.edge_parameter =
                            physical_root.edge_parameter;
                        result_owner.nurbs_patch_index = owner == nullptr
                            ? physical_root.patch_index
                            : owner->patch_index;
                        result_owner.nurbs_parameter = owner == nullptr
                            ? Eigen::Vector2d(
                                  physical_root.u, physical_root.v)
                            : Eigen::Vector2d(owner->u, owner->v);
                        result_owner.crossing_point = physical_root.point;
                        result_owner.crossing_normal = owner == nullptr
                            ? physical_root.normal : owner->normal;
                        result_owner.surface_component =
                            physical_root.component;
                        result_owner.crossing_residual = owner == nullptr
                            ? physical_root.residual : owner->residual;
                        result_owner.status =
                            P2CrossingOwnerStatus3D::ExactIntersection;
                        return result_owner;
                    };
                const auto make_owner_root =
                    [&](const geometry3d::NurbsSurfaceRootOwner3D* owner) {
                        geometry3d::NurbsSurfaceCrossing3D candidate =
                            physical_root;
                        candidate.owners.clear();
                        if (owner != nullptr) {
                            candidate.patch_index = owner->patch_index;
                            candidate.u = owner->u;
                            candidate.v = owner->v;
                            candidate.point = owner->point;
                            candidate.normal = owner->normal;
                            candidate.residual = owner->residual;
                            candidate.transversality =
                                owner->transversality;
                            candidate.feature_edge_contact =
                                owner->feature_edge_contact;
                            candidate.reliable_transversality_tolerance =
                                owner->reliable_transversality_tolerance;
                        }
                        return candidate;
                    };

                P2CrossingOwner3D exact_owner;
                app3d::GridlineCrossingSelection3D selected_gridline;
                bool owner_selected = false;
                double selected_transversality = -1.0;
                const auto consider_owner =
                    [&](const geometry3d::NurbsSurfaceRootOwner3D* owner) {
                        const auto owner_root = make_owner_root(owner);
                        const auto candidate_gridline =
                            app3d::select_nearest_gridline_crossing_on_g1_sheet_3d(
                                owner_root, topology,
                                shared_quadratic_gridline_records_,
                                tie_tolerance);
                        if (candidate_gridline.kind
                            != app3d::GridlineCrossingSelectionKind3D::
                                   Selected) {
                            return;
                        }
                        const bool closer = !owner_selected
                            || candidate_gridline.distance_to_segment_root
                                   < selected_gridline.
                                             distance_to_segment_root
                                         - tie_tolerance;
                        const bool equally_close_and_more_transverse =
                            owner_selected
                            && std::abs(
                                   candidate_gridline.
                                       distance_to_segment_root
                                   - selected_gridline.
                                         distance_to_segment_root)
                                   <= tie_tolerance
                            && owner_root.transversality
                                   > selected_transversality + 1.0e-13;
                        if (closer || equally_close_and_more_transverse) {
                            owner_selected = true;
                            exact_owner = make_owner(owner);
                            selected_gridline = candidate_gridline;
                            selected_transversality =
                                owner_root.transversality;
                        }
                    };

                // Multi-owner C0 events need a deterministic one-sided sheet
                // choice.  A smooth single-owner event needs no anchor unless
                // it is eligible for the optional nearest-grid optimization.
                const bool nearest_gridline_eligible =
                    sequence.events.size() == 1
                    && !event.feature_edge_contact
                    && physical_owner_count == 1;
                if (physical_root.owners.empty()) {
                    if (nearest_gridline_eligible)
                        consider_owner(nullptr);
                    if (!owner_selected) {
                        exact_owner = make_owner(nullptr);
                        owner_selected = true;
                    }
                } else if (physical_root.owners.size() == 1) {
                    if (nearest_gridline_eligible)
                        consider_owner(&physical_root.owners.front());
                    if (!owner_selected) {
                        exact_owner = make_owner(
                            &physical_root.owners.front());
                        owner_selected = true;
                    }
                } else {
                    for (const auto& owner : physical_root.owners)
                        consider_owner(&owner);
                    if (!owner_selected) {
                        ++shared_quadratic_diagnostics_.
                            no_g1_gridline_fallbacks;
                        throw std::runtime_error(
                            "shared quadratic restrict could not select a G1 sheet for a C0 physical event");
                    }
                }

                const HarmonicCrossingDirectPlan3D* cauchy_plan = nullptr;
                P2CrossingOwner3D correction_owner = exact_owner;
                double correction_distance_over_h =
                    (node_point - event.point).norm() / h_;
                if (nearest_gridline_eligible
                    && selected_gridline.kind
                           == app3d::GridlineCrossingSelectionKind3D::
                                  Selected) {
                    if (selected_gridline.nearest_tie_count > 1)
                        ++shared_quadratic_diagnostics_.gridline_ties;
                    const SharedQuadraticGridlineCrossing3D& crossing =
                        shared_quadratic_gridline_crossings_[
                            static_cast<std::size_t>(
                                selected_gridline.payload_index)];
                    const int crossing_center = surface_dof_for_crossing(
                        crossing.owner, crossing.point);
                    const std::tuple<int, int, int> crossing_key{
                        crossing.lo, crossing.hi,
                        crossing.event_id.ordinal};
                    auto cached =
                        shared_quadratic_cauchy_cache_.find(crossing_key);
                    if (cached == shared_quadratic_cauchy_cache_.end()) {
                        cached = shared_quadratic_cauchy_cache_.emplace(
                            crossing_key,
                            build_direct_crossing_plan(
                                crossing.owner, crossing.point,
                                crossing_center))
                                     .first;
                    }
                    cauchy_plan = &cached->second;
                    correction_owner = crossing.owner;
                    correction_distance_over_h =
                        selected_gridline.distance_to_segment_root / h_;
                } else {
                    const int crossing_center = surface_dof_for_crossing(
                        exact_owner, event.point);
                    // Repeated interior/exterior and nested Q27/Q64 plans see
                    // the exact same certified owner parameters.  Reuse only
                    // bit-for-bit identical parameter pairs; no tolerance
                    // merge is allowed, so nearby physical roots remain
                    // distinct events.
                    const std::tuple<int, double, double> exact_key{
                        exact_owner.nurbs_patch_index,
                        exact_owner.nurbs_parameter.x(),
                        exact_owner.nurbs_parameter.y()};
                    auto cached = all_event_exact_cauchy_cache_.find(
                        exact_key);
                    if (cached == all_event_exact_cauchy_cache_.end()) {
                        cached = all_event_exact_cauchy_cache_.emplace(
                            exact_key,
                            build_direct_crossing_plan(
                                exact_owner, event.point, crossing_center))
                                     .first;
                    }
                    cauchy_plan = &cached->second;
                }
                SharedQuadraticGridlineCorrection3D correction =
                    compose_shared_quadratic_correction(
                        stencil_node, node, *cauchy_plan,
                        correction_distance_over_h);
                correction.continuation_sign = event.continuation_sign;
                correction.native_event_id = event.native_event_id;
                correction.native_proof_id = event.native_proof_id;
                correction.direct_crossing_owner = correction_owner;
                correction.direct_grid_node = node;
                result.push_back(std::move(correction));
                if (result.back().stencil_node != stencil_node
                    || grid_ids[static_cast<std::size_t>(
                           result.back().stencil_node)]
                           != node) {
                    throw std::logic_error(
                        "shared quadratic correction confused a local stencil slot with a global grid node");
                }
                shared_quadratic_diagnostics_.
                    gridline_distance_max_over_h = std::max(
                        shared_quadratic_diagnostics_.
                            gridline_distance_max_over_h,
                        correction_distance_over_h);
                shared_quadratic_diagnostics_.cauchy_condition_max =
                    std::max(
                        shared_quadratic_diagnostics_.cauchy_condition_max,
                        cauchy_plan->condition);
            }
            if (continuation_state != (desired_inside ? 1 : 0)) {
                ++shared_quadratic_diagnostics_.segment_fallbacks;
                throw std::runtime_error(
                    "shared quadratic restrict all-event path ends on the wrong branch");
            }
        }
        return result;
    }

    SharedQuadraticTraceSidePlan3D
    build_shared_quadratic_side_plan(
        int center,
        app3d::QuadraticRestrictNormalSide3D side,
        const geometry3d::NurbsSurfaceIntersector3D& intersector,
        const app3d::G1PatchTopology3D& topology) const
    {
        const SurfaceDof& dof =
            cloud_.dofs[static_cast<std::size_t>(center)];
        const auto queries = app3d::quadratic_restrict_normal_query_points_3d(
            dof.point, dof.normal, h_, side, shared_normal_profile_);
        SharedQuadraticTraceSidePlan3D result;
        result.interpolation =
            app3d::build_shared_quadratic_restrict_stencil_3d(grid_, queries);
        const std::vector<int> grid_ids(
            result.interpolation.grid_ids.begin(),
            result.interpolation.grid_ids.end());
        result.corrections = build_all_event_trace_corrections(
            center,
            side == app3d::QuadraticRestrictNormalSide3D::Interior,
            grid_ids, intersector, topology);
        return result;
    }

    TensorProductCoverTraceBranchPlan3D
    build_tensor_product_cover_branch_plan(
        int center,
        bool desired_inside,
        app3d::TensorProductCoverKind3D kind,
        const geometry3d::NurbsSurfaceIntersector3D& intersector,
        const app3d::G1PatchTopology3D& topology) const
    {
        const SurfaceDof& dof =
            cloud_.dofs[static_cast<std::size_t>(center)];
        TensorProductCoverTraceBranchPlan3D result;
        result.interpolation =
            app3d::build_tensor_product_cover_restrict_stencil_3d(
                grid_, dof.point, dof.normal, kind);
        result.corrections = build_all_event_trace_corrections(
            center, desired_inside, result.interpolation.grid_ids,
            intersector, topology);
        return result;
    }

    void build_shared_quadratic_trace_templates(
        const geometry3d::NurbsSurfaceIntersector3D& intersector)
    {
        if (shared_quadratic_gridline_crossings_.empty())
            build_shared_quadratic_gridline_catalog();
        const app3d::G1PatchTopology3D topology =
            shared_quadratic_g1_topology();
        shared_quadratic_plans_.reserve(
            static_cast<std::size_t>(2 * surface_size()));
        for (int center = 0; center < surface_size(); ++center) {
            for (const app3d::QuadraticRestrictNormalSide3D side : {
                     app3d::QuadraticRestrictNormalSide3D::Interior,
                     app3d::QuadraticRestrictNormalSide3D::Exterior}) {
                shared_quadratic_plans_.push_back(
                    build_shared_quadratic_side_plan(
                        center, side, intersector, topology));
                shared_quadratic_diagnostics_.interpolation_condition_max =
                    std::max(
                        shared_quadratic_diagnostics_.interpolation_condition_max,
                        shared_quadratic_plans_.back()
                            .interpolation.condition);
            }
        }
        shared_quadratic_diagnostics_.side_plans =
            shared_quadratic_plans_.size();
        if (shared_quadratic_diagnostics_.segment_fallbacks != 0
            || shared_quadratic_diagnostics_.no_g1_gridline_fallbacks != 0) {
            throw std::runtime_error(
                "shared quadratic restrict setup used a forbidden geometry fallback");
        }
    }

    void build_tensor_product_cover_trace_templates(
        const geometry3d::NurbsSurfaceIntersector3D& intersector,
        app3d::TensorProductCoverKind3D kind)
    {
        if (shared_quadratic_gridline_crossings_.empty())
            build_shared_quadratic_gridline_catalog();
        const app3d::G1PatchTopology3D topology =
            shared_quadratic_g1_topology();
        std::vector<TensorProductCoverTraceBranchPlan3D>& plans =
            kind == app3d::TensorProductCoverKind3D::Q27Cover3
            ? q27_cover3_plans_ : q64_cover4_plans_;
        plans.reserve(static_cast<std::size_t>(2 * surface_size()));
        for (int center = 0; center < surface_size(); ++center) {
            for (const bool desired_inside : {true, false}) {
                plans.push_back(build_tensor_product_cover_branch_plan(
                    center, desired_inside, kind, intersector, topology));
                const auto& interpolation = plans.back().interpolation;
                shared_quadratic_diagnostics_.interpolation_condition_max =
                    std::max(
                        shared_quadratic_diagnostics_.
                            interpolation_condition_max,
                        std::max(interpolation.value_weight_l1,
                                 interpolation.scaled_normal_weight_l1));
            }
        }
        shared_quadratic_diagnostics_.side_plans += plans.size();
        if (shared_quadratic_diagnostics_.segment_fallbacks != 0
            || shared_quadratic_diagnostics_.no_g1_gridline_fallbacks != 0) {
            throw std::runtime_error(
                "tensor-product cover setup used a forbidden geometry fallback");
        }
    }

    void build_crossing_rows(bool build_direct_rows)
    {
        direct_crossing_unique_plan_count_ = 0;
        direct_crossing_svd_count_ = 0;
        crossing_rows_.reserve(correction_support_.crossing_ops.size());
        if (!build_direct_rows) {
            // Preserve the legacy surface-sample route unchanged.
            for (const LaplaceCrossingCorrectionOp& op
                 : correction_support_.crossing_ops) {
                const P2CrossingOwner3D owner =
                    laplace_crossing_owner_3d(grid_pair_, op);
                const Eigen::Vector3d a = grid_point(grid_, op.rhs_node);
                const Eigen::Vector3d b =
                    grid_point(grid_, op.correction_node);
                const double phase =
                    std::max(0.0, std::min(1.0, owner.edge_parameter));
                const Eigen::Vector3d hit = owner.nurbs_patch_index >= 0
                    ? owner.crossing_point
                    : a + phase * (b - a);

                const int center = surface_dof_for_crossing(owner, hit);
                const Eigen::Vector3d xi = local_coordinate(
                    center, grid_point(grid_, op.correction_node));
                HarmonicCrossingRow3D row;
                row.rhs_node = op.rhs_node;
                row.center_dof = center;
                row.scale = static_cast<double>(op.side_delta)
                          * op.stencil_weight;
                row.evaluation =
                    fit_.space().basis(xi.x(), xi.y(), xi.z());
                crossing_rows_.push_back(std::move(row));
            }
            return;
        }

        struct CrossingPlanKey {
            int lo = -1;
            int hi = -1;
            LaplaceCrossingKind kind = LaplaceCrossingKind::InterfaceJump;
            int patch = -1;
            int event_ordinal = -1;
        };
        struct CrossingOpRef {
            CrossingPlanKey key;
            std::size_t op_index = 0;
        };
        const auto make_key = [](const LaplaceCrossingCorrectionOp& op) {
            return CrossingPlanKey{
                std::min(op.rhs_node, op.correction_node),
                std::max(op.rhs_node, op.correction_node),
                op.kind,
                op.patch,
                has_grid_edge_event_3d(op)
                    ? op.grid_edge_event.ordinal : -1};
        };
        const auto key_less = [](const CrossingPlanKey& lhs,
                                 const CrossingPlanKey& rhs) {
            return std::tie(lhs.lo, lhs.hi, lhs.kind, lhs.patch,
                            lhs.event_ordinal)
                 < std::tie(rhs.lo, rhs.hi, rhs.kind, rhs.patch,
                            rhs.event_ordinal);
        };
        const auto key_equal = [](const CrossingPlanKey& lhs,
                                  const CrossingPlanKey& rhs) {
            return lhs.lo == rhs.lo && lhs.hi == rhs.hi
                && lhs.kind == rhs.kind && lhs.patch == rhs.patch
                && lhs.event_ordinal == rhs.event_ordinal;
        };

        std::vector<CrossingOpRef> refs;
        refs.reserve(correction_support_.crossing_ops.size());
        for (std::size_t op_index = 0;
             op_index < correction_support_.crossing_ops.size();
             ++op_index) {
            refs.push_back({
                make_key(correction_support_.crossing_ops[op_index]),
                op_index});
        }
        std::sort(
            refs.begin(), refs.end(),
            [&](const CrossingOpRef& lhs, const CrossingOpRef& rhs) {
                if (key_less(lhs.key, rhs.key))
                    return true;
                if (key_less(rhs.key, lhs.key))
                    return false;
                return lhs.op_index < rhs.op_index;
            });

        const bool verify_grouped_plans =
            std::getenv("KFBIM_3D_VERIFY_CROSSING_PLAN") != nullptr;
        std::size_t verified_rows = 0;
        std::size_t verification_id_mismatches = 0;
        std::size_t verification_center_mismatches = 0;
        double verification_hit_max = 0.0;
        double verification_frame_max = 0.0;
        double verification_map_max = 0.0;
        double verification_weight_max = 0.0;
        double verification_legacy_evaluation_max = 0.0;
        double verification_condition_max = 0.0;
        const auto max_abs_difference = [](const auto& lhs,
                                           const auto& rhs) {
            if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols())
                return std::numeric_limits<double>::infinity();
            return lhs.size() == 0
                ? 0.0
                : (lhs - rhs).cwiseAbs().maxCoeff();
        };

        crossing_rows_.resize(correction_support_.crossing_ops.size());
        direct_crossing_rows_.resize(correction_support_.crossing_ops.size());
        std::size_t group_begin = 0;
        while (group_begin < refs.size()) {
            std::size_t group_end = group_begin + 1;
            while (group_end < refs.size()
                   && key_equal(refs[group_begin].key,
                                refs[group_end].key)) {
                ++group_end;
            }

            const CrossingPlanKey& key = refs[group_begin].key;
            const LaplaceCrossingCorrectionOp& representative_op =
                correction_support_.crossing_ops[
                    refs[group_begin].op_index];
            const P2CrossingOwner3D owner =
                laplace_crossing_owner_3d(
                    grid_pair_, representative_op);
            const Eigen::Vector3d a = grid_point(grid_, key.lo);
            const Eigen::Vector3d b = grid_point(grid_, key.hi);
            const double phase =
                std::max(0.0, std::min(1.0, owner.edge_parameter));
            // edge_parameter belongs to the canonical (lo, hi) orientation;
            // never reuse it with a reversed directed op.
            const Eigen::Vector3d hit = owner.nurbs_patch_index >= 0
                ? owner.crossing_point
                : a + phase * (b - a);
            const int center = surface_dof_for_crossing(owner, hit);
            const HarmonicCrossingDirectPlan3D plan =
                build_direct_crossing_plan(owner, hit, center);
            ++direct_crossing_unique_plan_count_;
            // Rank/condition diagnostics and pseudoinverse share one SVD.
            ++direct_crossing_svd_count_;

            for (std::size_t pos = group_begin; pos < group_end; ++pos) {
                const std::size_t op_index = refs[pos].op_index;
                const LaplaceCrossingCorrectionOp& op =
                    correction_support_.crossing_ops[op_index];
                const Eigen::Vector3d xi = local_coordinate(
                    center, grid_point(grid_, op.correction_node));
                HarmonicCrossingRow3D row;
                row.rhs_node = op.rhs_node;
                row.center_dof = center;
                row.scale = static_cast<double>(op.side_delta)
                          * op.stencil_weight;
                row.evaluation =
                    fit_.space().basis(xi.x(), xi.y(), xi.z());
                crossing_rows_[op_index] = std::move(row);
                direct_crossing_rows_[op_index] =
                    compose_direct_crossing_row(op, plan);

                if (verify_grouped_plans) {
                    // Rebuild exactly as the former per-directed-op loop did,
                    // in the same process and against the same geometry.  This
                    // isolates grouping equivalence from independent geometry
                    // setup and OpenMP ordering effects.
                    const P2CrossingOwner3D reference_owner =
                        laplace_crossing_owner_3d(grid_pair_, op);
                    const Eigen::Vector3d reference_a =
                        grid_point(grid_, op.rhs_node);
                    const Eigen::Vector3d reference_b =
                        grid_point(grid_, op.correction_node);
                    const double reference_phase = std::max(
                        0.0,
                        std::min(1.0, reference_owner.edge_parameter));
                    const Eigen::Vector3d reference_hit =
                        reference_owner.nurbs_patch_index >= 0
                        ? reference_owner.crossing_point
                        : reference_a
                            + reference_phase * (reference_b - reference_a);
                    const int reference_center = surface_dof_for_crossing(
                        reference_owner, reference_hit);
                    const HarmonicCrossingDirectPlan3D reference_plan =
                        build_direct_crossing_plan(
                            reference_owner, reference_hit, reference_center);
                    const HarmonicCrossingDirectRow3D reference_direct_row =
                        compose_direct_crossing_row(op, reference_plan);
                    const Eigen::Vector3d reference_legacy_xi =
                        local_coordinate(
                            reference_center,
                            grid_point(grid_, op.correction_node));
                    const Eigen::VectorXd reference_legacy_evaluation =
                        fit_.space().basis(
                            reference_legacy_xi.x(),
                            reference_legacy_xi.y(),
                            reference_legacy_xi.z());

                    ++verified_rows;
                    if (plan.value_ids != reference_plan.value_ids
                        || plan.normal_ids != reference_plan.normal_ids) {
                        ++verification_id_mismatches;
                    }
                    if (center != reference_center)
                        ++verification_center_mismatches;
                    verification_hit_max = std::max(
                        verification_hit_max,
                        (hit - reference_hit).cwiseAbs().maxCoeff());
                    verification_frame_max = std::max(
                        verification_frame_max,
                        std::max({
                            (plan.frame.point - reference_plan.frame.point)
                                .cwiseAbs().maxCoeff(),
                            (plan.frame.normal - reference_plan.frame.normal)
                                .cwiseAbs().maxCoeff(),
                            (plan.frame.tangent1
                             - reference_plan.frame.tangent1)
                                .cwiseAbs().maxCoeff(),
                            (plan.frame.tangent2
                             - reference_plan.frame.tangent2)
                                .cwiseAbs().maxCoeff()}));
                    verification_map_max = std::max(
                        verification_map_max,
                        std::max(
                            max_abs_difference(
                                plan.value_map, reference_plan.value_map),
                            max_abs_difference(
                                plan.normal_map, reference_plan.normal_map)));
                    verification_weight_max = std::max(
                        verification_weight_max,
                        std::max(
                            max_abs_difference(
                                direct_crossing_rows_[op_index].value_weights,
                                reference_direct_row.value_weights),
                            max_abs_difference(
                                direct_crossing_rows_[op_index].normal_weights,
                                reference_direct_row.normal_weights)));
                    verification_legacy_evaluation_max = std::max(
                        verification_legacy_evaluation_max,
                        max_abs_difference(
                            crossing_rows_[op_index].evaluation,
                            reference_legacy_evaluation));
                    verification_condition_max = std::max(
                        verification_condition_max,
                        std::abs(plan.condition - reference_plan.condition));
                }
            }
            group_begin = group_end;
        }
        if (verify_grouped_plans) {
            std::cerr << std::setprecision(17)
                      << "crossing-plan reference rows=" << verified_rows
                      << " id_mismatches="
                      << verification_id_mismatches
                      << " center_mismatches="
                      << verification_center_mismatches
                      << " hit_max=" << verification_hit_max
                      << " frame_max=" << verification_frame_max
                      << " map_max=" << verification_map_max
                      << " weight_max=" << verification_weight_max
                      << " legacy_evaluation_max="
                      << verification_legacy_evaluation_max
                      << " condition_max=" << verification_condition_max
                      << std::endl;
            if (verification_id_mismatches != 0
                || verification_center_mismatches != 0
                || verification_hit_max != 0.0
                || verification_frame_max != 0.0
                || verification_map_max != 0.0
                || verification_weight_max != 0.0
                || verification_legacy_evaluation_max != 0.0
                || verification_condition_max != 0.0) {
                throw std::runtime_error(
                    "grouped crossing plan differs from directed reference");
            }
            if (std::getenv("KFBIM_3D_VERIFY_CROSSING_PLAN_ONLY")
                != nullptr) {
                throw std::runtime_error(
                    "crossing-plan verification-only stop");
            }
        }
    }

    static std::array<double, 3> local_parameter_triplet(
        double center,
        double requested_step,
        double lower,
        double upper)
    {
        if (!std::isfinite(center) || !std::isfinite(requested_step)
            || !std::isfinite(lower) || !std::isfinite(upper)
            || !(requested_step > 0.0) || !(upper > lower)
            || center < lower || center > upper) {
            throw std::invalid_argument(
                "invalid local quadratic restrict parameter interval");
        }
        const double left = center - lower;
        const double right = upper - center;
        if (left >= requested_step && right >= requested_step) {
            return {center - requested_step,
                    center,
                    center + requested_step};
        }
        if (right >= left) {
            const double step = std::min(requested_step, 0.5 * right);
            if (!(step > 1.0e-12 * (upper - lower)))
                throw std::runtime_error(
                    "local quadratic restrict has no forward patch room");
            return {center, center + step, center + 2.0 * step};
        }
        const double step = std::min(requested_step, 0.5 * left);
        if (!(step > 1.0e-12 * (upper - lower)))
            throw std::runtime_error(
                "local quadratic restrict has no backward patch room");
        return {center - 2.0 * step, center - step, center};
    }

    void build_local_quadratic_restrict_plans()
    {
        const double delta = 0.35 * h_;
        local_restrict_plans_.reserve(
            static_cast<std::size_t>(surface_size()));
        for (int center = 0; center < surface_size(); ++center) {
            const SurfaceDof& dof =
                cloud_.dofs[static_cast<std::size_t>(center)];
            if (dof.patch_id < 0
                || dof.patch_id
                       >= static_cast<int>(native_surface_.patches.size())) {
                throw std::runtime_error(
                    "local quadratic restrict has an invalid owner patch");
            }
            const geometry3d::NurbsSurfacePatch3D& patch =
                native_surface_.patches[
                    static_cast<std::size_t>(dof.patch_id)];
            const geometry3d::NurbsSurfaceDerivatives3D derivatives =
                patch.evaluate_with_derivatives(dof.u, dof.v);
            if (!(derivatives.du.norm() > 1.0e-13)
                || !(derivatives.dv.norm() > 1.0e-13)) {
                throw std::runtime_error(
                    "local quadratic restrict has a degenerate patch Jacobian");
            }

            LocalRestrictCauchyPlan3D plan;
            plan.frame = app3d::make_local_orthonormal_frame_3d(
                dof.normal, dof.tangent1);
            const std::array<double, 3> us = local_parameter_triplet(
                dof.u,
                delta / derivatives.du.norm(),
                patch.domain_start_u(),
                patch.domain_end_u());
            const std::array<double, 3> vs = local_parameter_triplet(
                dof.v,
                delta / derivatives.dv.norm(),
                patch.domain_start_v(),
                patch.domain_end_v());

            app3d::SurfacePointMatrix3D sample_points(
                kLocalRestrictSampleCount, 3);
            Eigen::MatrixXd graph_design(kLocalRestrictSampleCount, 3);
            Eigen::VectorXd graph_height(kLocalRestrictSampleCount);
            int sample = 0;
            for (double v : vs) {
                for (double u : us) {
                    LocalRestrictDensitySample3D& local =
                        plan.samples[static_cast<std::size_t>(sample)];
                    local.patch = dof.patch_id;
                    local.u = u;
                    local.v = v;
                    local.point = patch.evaluate(u, v);
                    sample_points.row(sample) = local.point.transpose();
                    const Eigen::Vector3d displacement =
                        local.point - dof.point;
                    const double s = displacement.dot(plan.frame.tangent1);
                    const double t = displacement.dot(plan.frame.tangent2);
                    graph_height[sample] =
                        displacement.dot(plan.frame.normal);
                    graph_design.row(sample) <<
                        0.5 * s * s, s * t, 0.5 * t * t;
                    ++sample;
                }
            }

            app3d::JetRecoveryOptions3D options;
            options.coordinate_scale = delta;
            const app3d::JetRecoveryMatrix3D recovery =
                app3d::build_value_jet_recovery_3d(
                    dof.point, plan.frame, sample_points, options);
            plan.value_recovery = recovery.matrix;
            plan.value_condition =
                recovery.diagnostics.condition_number;

            const Eigen::Vector3d hessian =
                svd_pseudoinverse_3d(graph_design, 3.0e-12)
                * graph_height;
            plan.graph_hessian.h11 = hessian[0];
            plan.graph_hessian.h12 = hessian[1];
            plan.graph_hessian.h22 = hessian[2];
            if (!plan.graph_hessian.all_finite()
                || !std::isfinite(plan.value_condition)) {
                throw std::runtime_error(
                    "local quadratic restrict plan contains NaN/Inf");
            }
            local_restrict_plans_.push_back(std::move(plan));
        }
    }

    HarmonicTraceSample3D build_trace_sample(int center,
                                             bool desired_inside,
                                             double signed_layer,
                                             const geometry3d::
                                                 NurbsSurfaceIntersector3D*
                                                 restrict_intersector) const
    {
        const SurfaceDof& dof = cloud_.dofs[static_cast<std::size_t>(center)];
        const Eigen::Vector3d query =
            dof.point + signed_layer * h_ * dof.normal;
        const std::array<double, 3> origin = grid_.origin();
        const std::array<int, 3> dims = grid_.dof_dims();
        std::array<int, 3> start_index{};
        std::array<std::array<double, kRestrictGridStencilSize>, 3>
            axis_weights{};
        for (int axis = 0; axis < 3; ++axis) {
            const double coordinate =
                (query[axis] - origin[static_cast<std::size_t>(axis)]) / h_;
            if (coordinate < 0.0
                || coordinate
                   > static_cast<double>(dims[static_cast<std::size_t>(axis)] - 1)) {
                throw std::runtime_error(
                    "normal-layer sample exits the embedding box");
            }
            const int floor_index = static_cast<int>(std::floor(coordinate));
            const int start = std::max(
                0,
                std::min(
                    floor_index - kRestrictGridLeftOffset,
                    dims[static_cast<std::size_t>(axis)]
                        - kRestrictGridStencilSize));
            start_index[static_cast<std::size_t>(axis)] = start;
            const int reference = start + kRestrictGridLeftOffset;
            const double fraction = coordinate
                - static_cast<double>(reference);
            axis_weights[static_cast<std::size_t>(axis)] =
                grid_lagrange_weights(fraction);
        }

        // Tensor-product interpolation uses degree+1 Cartesian nodes on each
        // axis; the default is 4^3 tricubic and the quartic target uses 5^3.
        HarmonicTraceSample3D result;
        result.legacy_correction_evaluation =
            Eigen::VectorXd::Zero(fit_.dimension());
        std::map<int, Eigen::VectorXd> owner_evaluations;
        std::map<int, Eigen::Matrix<double, 1, 6>>
            owner_value_jet_evaluations;
        int q = 0;
        for (int iz = 0; iz < kRestrictGridStencilSize; ++iz) {
            for (int iy = 0; iy < kRestrictGridStencilSize; ++iy) {
                for (int ix = 0; ix < kRestrictGridStencilSize; ++ix) {
                    const int i = start_index[0] + ix;
                    const int j = start_index[1] + iy;
                    const int k = start_index[2] + iz;
                    const int node = grid_.index(i, j, k);
                    const double weight = axis_weights[0][static_cast<std::size_t>(ix)]
                        * axis_weights[1][static_cast<std::size_t>(iy)]
                        * axis_weights[2][static_cast<std::size_t>(iz)];
                    result.grid_ids[static_cast<std::size_t>(q)] = node;
                    result.weights[static_cast<std::size_t>(q)] = weight;
                    const bool node_inside = grid_pair_.domain_label(node) > 0;
                    if (node_inside != desired_inside) {
                        ++result.wrong_side_node_count;
                        const double absolute_weight = std::abs(weight);
                        result.wrong_side_sum_abs_weight += absolute_weight;
                        const Eigen::Vector3d node_point =
                            grid_point(grid_, node);
                        const Eigen::Vector3d target_xi =
                            local_coordinate(center, node_point);
                        const double correction_sign = desired_inside ? 1.0 : -1.0;
                        result.legacy_correction_evaluation +=
                            correction_sign * weight
                            * fit_.space().basis(
                                target_xi.x(), target_xi.y(), target_xi.z());
                        if (!local_restrict_plans_.empty()) {
                            const LocalRestrictCauchyPlan3D& target_plan =
                                local_restrict_plans_[
                                    static_cast<std::size_t>(center)];
                            result.legacy_value_jet_evaluation +=
                                correction_sign * weight
                                * app3d::pweights_general_3d(
                                      node_point - dof.point,
                                      target_plan.frame,
                                      target_plan.graph_hessian).w0;
                        }

                        if (restrict_intersector != nullptr) {
                            ++result.owner_geometry_query_count;
                            int owner = center;
                            try {
                                const geometry3d::
                                    NurbsSurfaceIntersectionResult3D intersection =
                                        restrict_intersector->intersect_segment(
                                            query, node_point);
                                const app3d::RestrictOwnerDecision3D decision =
                                    app3d::select_restrict_correction_owner_3d(
                                        center, query, node_point,
                                        native_surface_, cloud_, intersection);
                                owner = decision.owner_dof;
                                const std::size_t kind =
                                    static_cast<std::size_t>(decision.kind);
                                if (kind
                                    >= result.owner_decision_counts.size()) {
                                    throw std::logic_error(
                                        "crossing-owner decision kind is invalid");
                                }
                                ++result.owner_decision_counts[kind];
                                result.owner_decision_sum_abs_weights[kind] +=
                                    absolute_weight;
                                if (decision.kind == app3d::
                                        RestrictOwnerDecisionKind3D::
                                            ForeignNonG1SingleCrossing) {
                                    if (intersection.crossings.size() != 1) {
                                        throw std::logic_error(
                                            "foreign crossing-owner decision "
                                            "does not have one crossing");
                                    }
                                    const geometry3d::NurbsSurfaceCrossing3D&
                                        root = intersection.crossings.front();
                                    HarmonicTraceOwnerAuditTerm3D audit;
                                    audit.grid_node = node;
                                    audit.interpolation_weight = weight;
                                    audit.owner_dof = owner;
                                    audit.crossing_patch = root.patch_index;
                                    audit.crossing_u = root.u;
                                    audit.crossing_v = root.v;
                                    audit.segment_parameter =
                                        root.edge_parameter;
                                    audit.residual = root.residual;
                                    audit.transversality =
                                        root.transversality;
                                    result.owner_reroute_terms.push_back(
                                        std::move(audit));
                                }
                            } catch (const geometry3d::
                                         UnresolvedNurbsIntersectionCandidate3D&) {
                                ++result.owner_unresolved_fallback_count;
                                result.
                                    owner_unresolved_fallback_sum_abs_weight +=
                                        absolute_weight;
                            } catch (const std::runtime_error& error) {
                                if (std::string(error.what())
                                    != "coincident roots on unrelated NURBS patches") {
                                    throw;
                                }
                                ++result.
                                    owner_unrelated_coincidence_fallback_count;
                                result.
                                    owner_unrelated_coincidence_fallback_sum_abs_weight +=
                                        absolute_weight;
                            }

                            const Eigen::Vector3d owner_xi =
                                local_coordinate(owner, node_point);
                            Eigen::VectorXd& owner_evaluation =
                                owner_evaluations[owner];
                            if (owner_evaluation.size() == 0) {
                                owner_evaluation =
                                    Eigen::VectorXd::Zero(fit_.dimension());
                            }
                            owner_evaluation += correction_sign * weight
                                * fit_.space().basis(
                                    owner_xi.x(), owner_xi.y(), owner_xi.z());
                            if (!local_restrict_plans_.empty()) {
                                auto inserted = owner_value_jet_evaluations
                                    .emplace(
                                        owner,
                                        Eigen::Matrix<double, 1, 6>::Zero());
                                const SurfaceDof& owner_dof = cloud_.dofs[
                                    static_cast<std::size_t>(owner)];
                                const LocalRestrictCauchyPlan3D& owner_plan =
                                    local_restrict_plans_[
                                        static_cast<std::size_t>(owner)];
                                inserted.first->second +=
                                    correction_sign * weight
                                    * app3d::pweights_general_3d(
                                          node_point - owner_dof.point,
                                          owner_plan.frame,
                                          owner_plan.graph_hessian).w0;
                            }
                        }
                    }
                    ++q;
                }
            }
        }
        result.owner_corrections.reserve(owner_evaluations.size());
        for (auto& owner_evaluation : owner_evaluations) {
            HarmonicTraceCorrectionTerm3D term;
            term.owner_dof = owner_evaluation.first;
            term.evaluation = std::move(owner_evaluation.second);
            if (!local_restrict_plans_.empty()) {
                const auto found = owner_value_jet_evaluations.find(
                    term.owner_dof);
                if (found == owner_value_jet_evaluations.end()) {
                    throw std::logic_error(
                        "local quadratic owner correction is missing");
                }
                term.value_jet_evaluation = found->second;
            }
            result.owner_corrections.push_back(std::move(term));
        }
        if (restrict_intersector != nullptr) {
            int classified_count =
                result.owner_unresolved_fallback_count
                + result.owner_unrelated_coincidence_fallback_count;
            for (int count : result.owner_decision_counts)
                classified_count += count;
            if (classified_count != result.wrong_side_node_count
                || result.owner_geometry_query_count
                       != static_cast<std::size_t>(
                           result.wrong_side_node_count)) {
                throw std::logic_error(
                    "crossing-owner restrict did not classify every "
                    "wrong-side support node");
            }
            const std::size_t foreign_kind = static_cast<std::size_t>(
                app3d::RestrictOwnerDecisionKind3D::
                    ForeignNonG1SingleCrossing);
            if (result.owner_reroute_terms.size()
                != static_cast<std::size_t>(
                    result.owner_decision_counts[foreign_kind])) {
                throw std::logic_error(
                    "crossing-owner reroute audit count is inconsistent");
            }
        }
        return result;
    }

    void build_trace_templates(
        const geometry3d::NurbsSurfaceIntersector3D* restrict_intersector)
    {
        trace_samples_.reserve(
            static_cast<std::size_t>(
                surface_size() * 2 * kRestrictNormalLayerCount));
        for (int center = 0; center < surface_size(); ++center) {
            for (int side = 0; side < 2; ++side) {
                const bool desired_inside = side == 0;
                const double sign = desired_inside ? -1.0 : 1.0;
                for (double layer : normal_layers_)
                    trace_samples_.push_back(
                        build_trace_sample(
                            center, desired_inside, sign * layer,
                            restrict_intersector));
            }
        }
    }

    void build_joint_trace_fit()
    {
        Eigen::MatrixXd design = Eigen::MatrixXd::Zero(
            kRestrictNormalSampleCount,
            kJointNormalCoefficientCount);
        for (int q = 0; q < kRestrictNormalLayerCount; ++q) {
            const double xm = -normal_layers_[static_cast<std::size_t>(q)];
            const double xp = normal_layers_[static_cast<std::size_t>(q)];
            design(q, 0) = 1.0;
            design(q, 1) = xm;
            design(kRestrictNormalLayerCount + q, 0) = 1.0;
            design(kRestrictNormalLayerCount + q, 1) = xp;
            for (int power = 2;
                 power <= kRestrictNormalDegree;
                 ++power) {
                design(q, power) = scalar_integer_power(xm, power);
                design(
                    kRestrictNormalLayerCount + q,
                    kRestrictNormalDegree + power - 1) =
                    scalar_integer_power(xp, power);
            }
        }
        const Eigen::MatrixXd pinv = svd_pseudoinverse_3d(design, 1.0e-13);
        for (int q = 0; q < kRestrictNormalSampleCount; ++q) {
            c0_weights_[static_cast<std::size_t>(q)] = pinv(0, q);
            c1_weights_[static_cast<std::size_t>(q)] = pinv(1, q);
        }
    }

    void build_global_trace_fit()
    {
        Eigen::MatrixXd design(
            kRestrictNormalSampleCount, kGlobalTraceCoefficientCount);
        negative_normal_ray_basis_.resize(
            kRestrictNormalLayerCount, fit_.dimension());
        positive_normal_ray_basis_.resize(
            kRestrictNormalLayerCount, fit_.dimension());
        for (int layer = 0;
             layer < kRestrictNormalLayerCount;
             ++layer) {
            const double tau =
                normal_layers_[static_cast<std::size_t>(layer)];
            negative_normal_ray_basis_.row(layer) =
                fit_.space().basis(0.0, 0.0, -tau).transpose();
            positive_normal_ray_basis_.row(layer) =
                fit_.space().basis(0.0, 0.0, tau).transpose();
            for (int side = 0; side < 2; ++side) {
                const double rho = side == 0 ? -tau : tau;
                const int row = side * kRestrictNormalLayerCount + layer;
                for (int power = 0;
                     power < kGlobalTraceCoefficientCount;
                     ++power) {
                    design(row, power) =
                        scalar_integer_power(rho, power);
                }
            }
        }
        const Eigen::MatrixXd pinv =
            svd_pseudoinverse_3d(design, 1.0e-13);
        for (int q = 0; q < kRestrictNormalSampleCount; ++q) {
            global_c0_weights_[static_cast<std::size_t>(q)] = pinv(0, q);
        }
        if (!negative_normal_ray_basis_.allFinite()
            || !positive_normal_ray_basis_.allFinite()) {
            throw std::runtime_error(
                "global exterior-branch ray basis is not finite");
        }
        for (int power = 0;
             power < kGlobalTraceCoefficientCount;
             ++power) {
            double moment = 0.0;
            for (int layer = 0;
                 layer < kRestrictNormalLayerCount;
                 ++layer) {
                const double tau =
                    normal_layers_[static_cast<std::size_t>(layer)];
                moment += global_c0_weights_[
                              static_cast<std::size_t>(layer)]
                        * scalar_integer_power(-tau, power);
                moment += global_c0_weights_[static_cast<std::size_t>(
                              kRestrictNormalLayerCount + layer)]
                        * scalar_integer_power(tau, power);
            }
            const double expected = power == 0 ? 1.0 : 0.0;
            if (!std::isfinite(moment)
                || std::abs(moment - expected) > 2.0e-13) {
                throw std::runtime_error(
                    "global exterior-branch trace weights do not reproduce "
                    "cubic moments");
            }
        }
    }

    void build_shared_quadratic_joint_trace_fit()
    {
        constexpr int layers =
            app3d::kSharedQuadraticRestrictQueryCount3D;
        Eigen::Matrix<double, 2 * layers, 4> design;
        design.setZero();
        const auto positive = app3d::shared_quadratic_signed_rho_3d(
            app3d::QuadraticRestrictNormalSide3D::Exterior,
            shared_normal_profile_);
        for (int layer = 0; layer < layers; ++layer) {
            const double tau = positive[static_cast<std::size_t>(layer)];
            if (shared_normal_profile_
                == app3d::SharedQuadraticNormalProfile3D::
                       TopologyAffineCubic) {
                const double negative_tau = -tau;
                design(layer, 0) = 1.0;
                design(layer, 1) = negative_tau;
                design(layer, 2) = negative_tau * negative_tau;
                design(layer, 3) = negative_tau * negative_tau
                    * negative_tau;
                design(layers + layer, 0) = 1.0;
                design(layers + layer, 1) = tau;
                design(layers + layer, 2) = tau * tau;
                design(layers + layer, 3) = tau * tau * tau;
            } else {
                design(layer, 0) = 1.0;
                design(layer, 1) = -tau;
                design(layer, 2) = tau * tau;
                design(layers + layer, 0) = 1.0;
                design(layers + layer, 1) = tau;
                design(layers + layer, 3) = tau * tau;
            }
        }
        const Eigen::MatrixXd pinv =
            svd_pseudoinverse_3d(design, 1.0e-13);
        shared_quadratic_joint_c0_weights_ = pinv.row(0);
        shared_quadratic_joint_c1_weights_ = pinv.row(1);
        const Eigen::Matrix<double, 4, 4> moments = pinv * design;
        if ((moments - Eigen::Matrix4d::Identity())
                .cwiseAbs().maxCoeff() > 3.0e-13) {
            throw std::runtime_error(
                "shared quadratic joint normal fit does not reproduce moments");
        }
    }

    const CartesianGrid3D& grid_;
    const GridPair3D& grid_pair_;
    const NativeNurbsSurface3D& native_surface_;
    const std::vector<geometry3d::NurbsParamTriangle3D>& correction_triangles_;
    const std::vector<geometry3d::NurbsParamTriangle3D>& geometry_triangles_;
    const SurfaceDofCloud& cloud_;
    const CauchyStencilSet& stencils_;
    app3d::SharedQuadraticNormalProfile3D shared_normal_profile_ =
        app3d::SharedQuadraticNormalProfile3D::LegacySplitQuadratic;
    bool crossing_owner_templates_built_ = false;
    bool shared_quadratic_templates_built_ = false;
    bool q27_cover3_templates_built_ = false;
    bool q64_cover4_templates_built_ = false;
    bool native_endpoint_paths_ = false;
    geometry3d::PathCertificationBudget3D native_endpoint_budget_;
    double h_ = 0.0;
    PanelCenterCauchyFit3D fit_;
    std::unique_ptr<app3d::ExteriorOnlyCubicNormalRestrict3D>
        exterior_only_restrict_;
    LaplaceFftBulkSolverZfft3D bulk_;
    LaplaceCorrectionSupport3D correction_support_;
    std::vector<HarmonicCrossingRow3D> crossing_rows_;
    std::vector<HarmonicCrossingDirectRow3D> direct_crossing_rows_;
    const app3d::NativeNurbsDensitySpace3D* direct_coefficient_density_ =
        nullptr;
    std::vector<DirectCoefficientSpreadRow3D>
        direct_coefficient_spread_rows_;
    const app3d::NativeNurbsDensitySpace3D*
        direct_coefficient_normal_density_ = nullptr;
    std::vector<DirectCoefficientNormalSpreadRow3D>
        direct_coefficient_normal_spread_rows_;
    std::size_t direct_crossing_unique_plan_count_ = 0;
    std::size_t direct_crossing_svd_count_ = 0;
    std::vector<HarmonicTraceSample3D> trace_samples_;
    std::vector<SharedQuadraticTraceSidePlan3D> shared_quadratic_plans_;
    std::vector<TensorProductCoverTraceBranchPlan3D> q27_cover3_plans_;
    std::vector<TensorProductCoverTraceBranchPlan3D> q64_cover4_plans_;
    std::vector<SharedQuadraticGridlineCrossing3D>
        shared_quadratic_gridline_crossings_;
    std::vector<app3d::CartesianGridlineCrossingRecord3D>
        shared_quadratic_gridline_records_;
    mutable std::map<std::tuple<int, int, int>, HarmonicCrossingDirectPlan3D>
        shared_quadratic_cauchy_cache_;
    mutable std::map<
        std::tuple<int, double, double>, HarmonicCrossingDirectPlan3D>
        all_event_exact_cauchy_cache_;
    mutable std::map<
        std::pair<int, int>,
        geometry3d::NurbsSurfaceIntersectionResult3D>
        all_event_intersection_cache_;
    // The immutable pipeline owns the model and grid: (center,node) therefore
    // fixes native endpoint, start point, direction, precision and budget.
    mutable std::map<std::pair<int, int>, geometry3d::NativeEndpointPathResult3D>
        native_endpoint_path_cache_;
    mutable SharedQuadraticRestrictDiagnostics3D
        shared_quadratic_diagnostics_;
    std::vector<LocalRestrictCauchyPlan3D> local_restrict_plans_;
    const std::array<double, kRestrictNormalLayerCount> normal_layers_ =
        make_normal_layers();
    std::array<double, kRestrictNormalSampleCount> c0_weights_{};
    std::array<double, kRestrictNormalSampleCount> c1_weights_{};
    std::array<double, kRestrictNormalSampleCount> global_c0_weights_{};
    Eigen::MatrixXd negative_normal_ray_basis_;
    Eigen::MatrixXd positive_normal_ray_basis_;
    Eigen::Matrix<double, 1,
                  2 * app3d::kSharedQuadraticRestrictQueryCount3D>
        shared_quadratic_joint_c0_weights_ =
            Eigen::Matrix<double, 1,
                          2 * app3d::kSharedQuadraticRestrictQueryCount3D>::Zero();
    Eigen::Matrix<double, 1,
                  2 * app3d::kSharedQuadraticRestrictQueryCount3D>
        shared_quadratic_joint_c1_weights_ =
            Eigen::Matrix<double, 1,
                          2 * app3d::kSharedQuadraticRestrictQueryCount3D>::Zero();
};

struct NativeDensityCrossingTarget3D {
    int rhs_node = -1;
    double scale = 0.0;
    Eigen::Matrix<double, 1, 6> value_row =
        Eigen::Matrix<double, 1, 6>::Zero();
};

struct NativeDensityCrossingPlan3D {
    std::array<app3d::NativeDensityC0Stencil3D,
               kLocalRestrictSampleCount> value_samples;
    app3d::NativeDensityC0Stencil3D center_sample;
    Eigen::Matrix<double, 6, kLocalRestrictSampleCount> value_recovery =
        Eigen::Matrix<double, 6, kLocalRestrictSampleCount>::Zero();
    std::vector<NativeDensityCrossingTarget3D> targets;
    double value_condition = 0.0;
};

// Value-jump spread assembled from the same native density that is iterated
// by GMRES.  Each unique Cartesian/surface crossing owns a one-sided 3x3
// density jet; directed finite-difference correction rows reuse that jet.
class NativeDensityValueSpread3D {
public:
    NativeDensityValueSpread3D(
        const CartesianGrid3D& grid,
        const GridPair3D& grid_pair,
        const PanelCenterHarmonicJetKFBI3D& pipeline,
        const app3d::NativeNurbsDensitySpace3D& density)
        : grid_(grid)
        , grid_pair_(grid_pair)
        , pipeline_(pipeline)
        , density_(density)
        , h_(grid.spacing()[0])
    {
        const LaplaceCorrectionSupport3D support =
            build_laplace_correction_support_3d(
                grid_pair_, "NativeDensityValueSpread3D");
        std::map<std::tuple<int, int, int>, std::size_t> plan_by_event;
        plans_.reserve(support.crossing_ops.size() / 2);
        for (const LaplaceCrossingCorrectionOp& op : support.crossing_ops) {
            const int lo = std::min(op.rhs_node, op.correction_node);
            const int hi = std::max(op.rhs_node, op.correction_node);
            const std::tuple<int, int, int> key{
                lo, hi,
                has_grid_edge_event_3d(op)
                    ? op.grid_edge_event.ordinal : -1};
            auto found = plan_by_event.find(key);
            if (found == plan_by_event.end()) {
                const P2CrossingOwner3D owner =
                    laplace_crossing_owner_3d(grid_pair_, op);
                plans_.push_back(build_plan(owner));
                const std::size_t index = plans_.size() - 1;
                found = plan_by_event.emplace(key, index).first;
            }
            NativeDensityCrossingPlan3D& plan =
                plans_[found->second];
            const P2CrossingOwner3D owner =
                laplace_crossing_owner_3d(grid_pair_, op);
            const Eigen::Vector3d hit = owner.crossing_point;
            const NativeCrossingFrame3D frame = crossing_frame(owner);
            NativeDensityCrossingTarget3D target;
            target.rhs_node = op.rhs_node;
            target.scale = static_cast<double>(op.side_delta)
                         * op.stencil_weight;
            target.value_row = app3d::pweights_general_3d(
                grid_point(grid_, op.correction_node) - hit,
                frame.frame,
                frame.graph_hessian).w0;
            plan.targets.push_back(std::move(target));
        }
        if (plans_.empty())
            throw std::runtime_error(
                "native density spread found no Cartesian crossings");
    }

    Eigen::VectorXd evaluate(
        const Eigen::Ref<const Eigen::VectorXd>& value_c0) const
    {
        if (value_c0.size() != density_.c0_coefficient_count())
            throw std::invalid_argument(
                "native density spread received the wrong C0 size");
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid_.num_dofs());
        for (const NativeDensityCrossingPlan3D& plan : plans_) {
            Eigen::Matrix<double, kLocalRestrictSampleCount, 1> samples;
            for (int q = 0; q < kLocalRestrictSampleCount; ++q) {
                samples[q] = plan.value_samples[static_cast<std::size_t>(q)]
                    .dot(value_c0);
            }
            Eigen::Matrix<double, 6, 1> jet =
                plan.value_recovery * samples;
            jet[0] = plan.center_sample.dot(value_c0);
            for (const NativeDensityCrossingTarget3D& target : plan.targets)
                rhs[target.rhs_node] +=
                    target.scale * target.value_row.dot(jet);
        }
        return pipeline_.solve_external_correction_rhs(rhs);
    }

    std::vector<double> value_condition_values() const
    {
        std::vector<double> result;
        result.reserve(plans_.size());
        for (const NativeDensityCrossingPlan3D& plan : plans_)
            result.push_back(plan.value_condition);
        return result;
    }

private:
    struct NativeCrossingFrame3D {
        app3d::LocalOrthonormalFrame3D frame;
        app3d::TangentGraphHessian3D graph_hessian;
        int patch = -1;
        double u = 0.0;
        double v = 0.0;
    };

    static std::array<double, 3> parameter_triplet(
        double center,
        double requested_step)
    {
        const double left = center;
        const double right = 1.0 - center;
        if (left >= requested_step && right >= requested_step)
            return {center - requested_step, center, center + requested_step};
        if (right >= left) {
            const double step = std::min(requested_step, 0.5 * right);
            if (!(step > 1.0e-12))
                throw std::runtime_error(
                    "native crossing jet has no forward patch room");
            return {center, center + step, center + 2.0 * step};
        }
        const double step = std::min(requested_step, 0.5 * left);
        if (!(step > 1.0e-12))
            throw std::runtime_error(
                "native crossing jet has no backward patch room");
        return {center - 2.0 * step, center - step, center};
    }

    NativeCrossingFrame3D crossing_frame(
        const P2CrossingOwner3D& owner) const
    {
        if (owner.nurbs_patch_index < 0
            || owner.nurbs_patch_index >= density_.patch_count()
            || !owner.nurbs_parameter.allFinite()
            || !owner.crossing_point.allFinite()) {
            throw std::runtime_error(
                "native density crossing has no exact NURBS owner");
        }
        NativeCrossingFrame3D result;
        result.patch = owner.nurbs_patch_index;
        const auto& patch = density_.surface().patches[
            static_cast<std::size_t>(result.patch)];
        result.u = (owner.nurbs_parameter.x() - patch.domain_start_u())
                 / (patch.domain_end_u() - patch.domain_start_u());
        result.v = (owner.nurbs_parameter.y() - patch.domain_start_v())
                 / (patch.domain_end_v() - patch.domain_start_v());
        result.u = std::clamp(result.u, 0.0, 1.0);
        result.v = std::clamp(result.v, 0.0, 1.0);
        const app3d::NativeDensitySurfaceEvaluation3D geometry =
            density_.geometry(result.patch, result.u, result.v);
        Eigen::Vector3d normal = geometry.normal;
        if (owner.crossing_normal.allFinite()
            && owner.crossing_normal.norm() > 0.5
            && normal.dot(owner.crossing_normal) < 0.0) {
            normal = -normal;
        }
        result.frame = app3d::make_local_orthonormal_frame_3d(
            normal, geometry.tangents.col(0));

        const double delta = 0.35 * h_;
        const std::array<double, 3> us = parameter_triplet(
            result.u, delta / geometry.tangents.col(0).norm());
        const std::array<double, 3> vs = parameter_triplet(
            result.v, delta / geometry.tangents.col(1).norm());
        Eigen::MatrixXd graph_design(kLocalRestrictSampleCount, 3);
        Eigen::VectorXd graph_height(kLocalRestrictSampleCount);
        int sample = 0;
        for (double v : vs) {
            for (double u : us) {
                const Eigen::Vector3d point =
                    density_.geometry(result.patch, u, v).point;
                const Eigen::Vector3d displacement =
                    point - owner.crossing_point;
                const double s = displacement.dot(result.frame.tangent1);
                const double t = displacement.dot(result.frame.tangent2);
                graph_height[sample] =
                    displacement.dot(result.frame.normal);
                graph_design.row(sample) <<
                    0.5 * s * s, s * t, 0.5 * t * t;
                ++sample;
            }
        }
        const Eigen::Vector3d hessian =
            svd_pseudoinverse_3d(graph_design, 3.0e-12) * graph_height;
        result.graph_hessian.h11 = hessian[0];
        result.graph_hessian.h12 = hessian[1];
        result.graph_hessian.h22 = hessian[2];
        return result;
    }

    NativeDensityCrossingPlan3D build_plan(
        const P2CrossingOwner3D& owner) const
    {
        const NativeCrossingFrame3D data = crossing_frame(owner);
        const app3d::NativeDensitySurfaceEvaluation3D geometry =
            density_.geometry(data.patch, data.u, data.v);
        const double delta = 0.35 * h_;
        const std::array<double, 3> us = parameter_triplet(
            data.u, delta / geometry.tangents.col(0).norm());
        const std::array<double, 3> vs = parameter_triplet(
            data.v, delta / geometry.tangents.col(1).norm());
        NativeDensityCrossingPlan3D result;
        result.center_sample = density_.c0_basis_stencil(
            data.patch, data.u, data.v);
        app3d::SurfacePointMatrix3D sample_points(
            kLocalRestrictSampleCount, 3);
        int sample = 0;
        for (double v : vs) {
            for (double u : us) {
                result.value_samples[static_cast<std::size_t>(sample)] =
                    density_.c0_basis_stencil(data.patch, u, v);
                sample_points.row(sample) =
                    density_.geometry(data.patch, u, v).point.transpose();
                ++sample;
            }
        }
        app3d::JetRecoveryOptions3D options;
        options.coordinate_scale = delta;
        const app3d::JetRecoveryMatrix3D recovery =
            app3d::build_value_jet_recovery_3d(
                owner.crossing_point, data.frame, sample_points, options);
        result.value_recovery = recovery.matrix;
        result.value_condition = recovery.diagnostics.condition_number;
        return result;
    }

    const CartesianGrid3D& grid_;
    const GridPair3D& grid_pair_;
    const PanelCenterHarmonicJetKFBI3D& pipeline_;
    const app3d::NativeNurbsDensitySpace3D& density_;
    double h_ = 0.0;
    std::vector<NativeDensityCrossingPlan3D> plans_;
};

struct NativeGaussOwnerCorrection3D {
    int owner_dof = -1;
    Eigen::Matrix<double, 1, 6> value_row =
        Eigen::Matrix<double, 1, 6>::Zero();
    Eigen::VectorXd normal_row;
};

struct NativeGaussBranchCorrection3D {
    Eigen::Matrix<double, 1, 6> target_value_row =
        Eigen::Matrix<double, 1, 6>::Zero();
    Eigen::VectorXd target_normal_row;
    std::vector<NativeGaussOwnerCorrection3D> foreign_rows;
};

struct NativeGaussValueTracePlan3D {
    int trace_index = -1;
    int proxy_dof = -1;
    std::array<app3d::NativeDensityC0Stencil3D,
               kLocalRestrictSampleCount> value_samples;
    app3d::NativeDensityC0Stencil3D target_value_sample;
    Eigen::Matrix<double, 6, kLocalRestrictSampleCount> value_recovery =
        Eigen::Matrix<double, 6, kLocalRestrictSampleCount>::Zero();
    std::array<NativeGaussBranchCorrection3D, 2> branch;
    double value_condition = 0.0;
};

// Direct restriction at the native density quadrature points.  The same
// Gauss rule is subsequently used by the reduced weighted-LS projector, so
// the Neumann trial and test spaces form one consistent discrete pairing.
// Index 0 of branch[] is the exterior continuation and index 1 is interior.
class NativeGaussExteriorValueRestrict3D {
public:
    NativeGaussExteriorValueRestrict3D(
        const CartesianGrid3D& grid,
        const GridPair3D& grid_pair,
        const PanelCenterHarmonicJetKFBI3D& pipeline,
        const app3d::NativeNurbsDensitySpace3D& density)
        : grid_(grid)
        , grid_pair_(grid_pair)
        , pipeline_(pipeline)
        , density_(density)
        , h_(grid.spacing()[0])
        , trace_weights_(make_global_trace_weights())
    {
        if (density_.options().field != app3d::NativeDensityField3D::ValueTrace)
            throw std::invalid_argument(
                "native Gauss value restrict requires a value density space");
        if (density_.patch_count()
                != static_cast<int>(pipeline_.surface().patches.size())
            || !(h_ > 0.0)) {
            throw std::invalid_argument(
                "native Gauss value restrict received incompatible geometry");
        }
        geometry3d::NurbsSurfaceIntersectorOptions3D options;
        options.maximum_element_extent = 2.0 * h_;
        options.local_max_subdivision_depth = 4;
        const geometry3d::NurbsSurfaceIntersector3D intersector(
            density_.surface().geometry_model(), options);
        const auto& points = density_.trace_points();
        plans_.reserve(points.size());
        for (int trace = 0; trace < static_cast<int>(points.size()); ++trace)
            plans_.push_back(build_plan(trace, intersector));
    }

    int trace_count() const noexcept
    {
        return static_cast<int>(plans_.size());
    }

    std::vector<double> value_condition_values() const
    {
        std::vector<double> result;
        result.reserve(plans_.size());
        for (const NativeGaussValueTracePlan3D& plan : plans_)
            result.push_back(plan.value_condition);
        return result;
    }

    Eigen::VectorXd exterior_value_trace(
        const HarmonicJetField3D& field,
        const Eigen::Ref<const Eigen::VectorXd>& value_c0) const
    {
        return value_trace(field, value_c0, false);
    }

    Eigen::VectorXd interior_value_trace(
        const HarmonicJetField3D& field,
        const Eigen::Ref<const Eigen::VectorXd>& value_c0) const
    {
        return value_trace(field, value_c0, true);
    }

private:
    static std::array<double, kRestrictNormalSampleCount>
    make_global_trace_weights()
    {
        Eigen::MatrixXd design(
            kRestrictNormalSampleCount, kGlobalTraceCoefficientCount);
        const auto layers = make_normal_layers();
        for (int side = 0; side < 2; ++side) {
            for (int layer = 0; layer < kRestrictNormalLayerCount; ++layer) {
                const double rho = (side == 0 ? -1.0 : 1.0)
                    * layers[static_cast<std::size_t>(layer)];
                const int row = side * kRestrictNormalLayerCount + layer;
                for (int power = 0;
                     power < kGlobalTraceCoefficientCount;
                     ++power) {
                    design(row, power) = scalar_integer_power(rho, power);
                }
            }
        }
        const Eigen::MatrixXd inverse =
            svd_pseudoinverse_3d(design, 1.0e-13);
        std::array<double, kRestrictNormalSampleCount> result{};
        for (int row = 0; row < kRestrictNormalSampleCount; ++row)
            result[static_cast<std::size_t>(row)] = inverse(0, row);
        return result;
    }

    static std::array<double, 3> parameter_triplet(
        double center,
        double requested_step)
    {
        if (!std::isfinite(center) || !std::isfinite(requested_step)
            || center < 0.0 || center > 1.0 || !(requested_step > 0.0)) {
            throw std::invalid_argument(
                "invalid native Gauss local parameter interval");
        }
        const double left = center;
        const double right = 1.0 - center;
        if (left >= requested_step && right >= requested_step)
            return {center - requested_step, center, center + requested_step};
        if (right >= left) {
            const double step = std::min(requested_step, 0.5 * right);
            if (!(step > 1.0e-12))
                throw std::runtime_error(
                    "native Gauss restrict has no forward patch room");
            return {center, center + step, center + 2.0 * step};
        }
        const double step = std::min(requested_step, 0.5 * left);
        if (!(step > 1.0e-12))
            throw std::runtime_error(
                "native Gauss restrict has no backward patch room");
        return {center - 2.0 * step, center - step, center};
    }

    int proxy_dof(const app3d::NativeDensityGaussPoint3D& point) const
    {
        const auto& patch = density_.surface().patches[
            static_cast<std::size_t>(point.patch)];
        const double native_u = patch.domain_start_u()
            + point.u * (patch.domain_end_u() - patch.domain_start_u());
        const double native_v = patch.domain_start_v()
            + point.v * (patch.domain_end_v() - patch.domain_start_v());
        const std::array<int, 4> candidates =
            app3d::parameter_dof_candidates_2x2(
                density_.surface(), pipeline_.surface(), point.patch,
                native_u, native_v);
        int result = -1;
        double best = std::numeric_limits<double>::infinity();
        for (int candidate : candidates) {
            const SurfaceDof& dof = pipeline_.surface().dofs[
                static_cast<std::size_t>(candidate)];
            if (dof.patch_id != point.patch)
                continue;
            const double distance = (dof.point - point.point).squaredNorm();
            if (distance < best) {
                best = distance;
                result = candidate;
            }
        }
        if (result < 0) {
            // At a patch boundary a resolved lattice candidate can belong to
            // the smoothly connected patch.  Retain the nearest candidate;
            // Gauss targets themselves never lie on that boundary.
            for (int candidate : candidates) {
                const SurfaceDof& dof = pipeline_.surface().dofs[
                    static_cast<std::size_t>(candidate)];
                const double distance =
                    (dof.point - point.point).squaredNorm();
                if (distance < best) {
                    best = distance;
                    result = candidate;
                }
            }
        }
        if (result < 0)
            throw std::runtime_error(
                "native Gauss restrict could not choose a proxy DOF");
        return result;
    }

    NativeGaussValueTracePlan3D build_plan(
        int trace_index,
        const geometry3d::NurbsSurfaceIntersector3D& intersector) const
    {
        const auto& point = density_.trace_points()[
            static_cast<std::size_t>(trace_index)];
        const app3d::NativeDensitySurfaceEvaluation3D geometry =
            density_.geometry(point.patch, point.u, point.v);
        const double delta = 0.35 * h_;
        const app3d::LocalOrthonormalFrame3D frame =
            app3d::make_local_orthonormal_frame_3d(
                point.normal, geometry.tangents.col(0));
        const std::array<double, 3> us = parameter_triplet(
            point.u, delta / geometry.tangents.col(0).norm());
        const std::array<double, 3> vs = parameter_triplet(
            point.v, delta / geometry.tangents.col(1).norm());

        NativeGaussValueTracePlan3D plan;
        plan.trace_index = trace_index;
        plan.proxy_dof = proxy_dof(point);
        plan.target_value_sample = density_.c0_basis_stencil(
            point.patch, point.u, point.v);
        app3d::SurfacePointMatrix3D sample_points(
            kLocalRestrictSampleCount, 3);
        Eigen::MatrixXd graph_design(kLocalRestrictSampleCount, 3);
        Eigen::VectorXd graph_height(kLocalRestrictSampleCount);
        int sample = 0;
        for (double v : vs) {
            for (double u : us) {
                const app3d::NativeDensitySurfaceEvaluation3D local =
                    density_.geometry(point.patch, u, v);
                plan.value_samples[static_cast<std::size_t>(sample)] =
                    density_.c0_basis_stencil(point.patch, u, v);
                sample_points.row(sample) = local.point.transpose();
                const Eigen::Vector3d displacement =
                    local.point - point.point;
                const double s = displacement.dot(frame.tangent1);
                const double t = displacement.dot(frame.tangent2);
                graph_height[sample] = displacement.dot(frame.normal);
                graph_design.row(sample) <<
                    0.5 * s * s, s * t, 0.5 * t * t;
                ++sample;
            }
        }
        app3d::JetRecoveryOptions3D recovery_options;
        recovery_options.coordinate_scale = delta;
        const app3d::JetRecoveryMatrix3D recovery =
            app3d::build_value_jet_recovery_3d(
                point.point, frame, sample_points, recovery_options);
        plan.value_recovery = recovery.matrix;
        plan.value_condition = recovery.diagnostics.condition_number;
        const Eigen::Vector3d hessian =
            svd_pseudoinverse_3d(graph_design, 3.0e-12) * graph_height;
        app3d::TangentGraphHessian3D graph_hessian;
        graph_hessian.h11 = hessian[0];
        graph_hessian.h12 = hessian[1];
        graph_hessian.h22 = hessian[2];
        if (!graph_hessian.all_finite()
            || !std::isfinite(plan.value_condition)) {
            throw std::runtime_error(
                "native Gauss restrict local plan contains NaN/Inf");
        }

        const int dimension = pipeline_.restrict_cauchy_dimension();
        for (NativeGaussBranchCorrection3D& correction : plan.branch)
            correction.target_normal_row = Eigen::VectorXd::Zero(dimension);
        std::array<std::map<int, NativeGaussOwnerCorrection3D>, 2>
            foreign_rows;
        const auto layers = make_normal_layers();
        for (int side = 0; side < 2; ++side) {
            const double sign = side == 0 ? -1.0 : 1.0;
            for (int layer = 0; layer < kRestrictNormalLayerCount; ++layer) {
                const int trace_row =
                    side * kRestrictNormalLayerCount + layer;
                const Eigen::Vector3d query = point.point
                    + sign * layers[static_cast<std::size_t>(layer)]
                          * h_ * point.normal;
                std::array<int, 3> start{};
                std::array<std::array<double, kRestrictGridStencilSize>, 3>
                    axis_weights{};
                interpolation_axes(query, start, axis_weights);
                for (int iz = 0; iz < kRestrictGridStencilSize; ++iz) {
                    for (int iy = 0; iy < kRestrictGridStencilSize; ++iy) {
                        for (int ix = 0; ix < kRestrictGridStencilSize; ++ix) {
                            const int node = grid_.index(
                                start[0] + ix,
                                start[1] + iy,
                                start[2] + iz);
                            const bool node_inside =
                                grid_pair_.domain_label(node) > 0;
                            // An inside node must be continued to exterior;
                            // an outside node must be continued to interior.
                            const int branch = node_inside ? 0 : 1;
                            const double continuation_sign =
                                node_inside ? -1.0 : 1.0;
                            const double interpolation =
                                axis_weights[0][static_cast<std::size_t>(ix)]
                                * axis_weights[1][static_cast<std::size_t>(iy)]
                                * axis_weights[2][static_cast<std::size_t>(iz)];
                            const double weight = continuation_sign
                                * trace_weights_[static_cast<std::size_t>(
                                      trace_row)]
                                * interpolation;
                            if (weight == 0.0)
                                continue;
                            const Eigen::Vector3d node_point =
                                grid_point(grid_, node);
                            int owner = plan.proxy_dof;
                            bool foreign = false;
                            try {
                                const auto intersection =
                                    intersector.intersect_segment(
                                        query, node_point);
                                const app3d::RestrictOwnerDecision3D decision =
                                    app3d::select_restrict_correction_owner_3d(
                                        plan.proxy_dof, query, node_point,
                                        density_.surface(),
                                        pipeline_.surface(), intersection);
                                if (decision.kind == app3d::
                                        RestrictOwnerDecisionKind3D::
                                            ForeignNonG1SingleCrossing) {
                                    owner = decision.owner_dof;
                                    foreign = true;
                                }
                            } catch (const geometry3d::
                                         UnresolvedNurbsIntersectionCandidate3D&) {
                                // Conservative target-sheet fallback.
                            } catch (const std::runtime_error& error) {
                                if (std::string(error.what())
                                    != "coincident roots on unrelated NURBS patches") {
                                    throw;
                                }
                            }
                            NativeGaussBranchCorrection3D& correction =
                                plan.branch[static_cast<std::size_t>(branch)];
                            if (!foreign) {
                                correction.target_value_row += weight
                                    * app3d::pweights_general_3d(
                                          node_point - point.point,
                                          frame,
                                          graph_hessian)
                                          .w0;
                                correction.target_normal_row.noalias() +=
                                    weight
                                    * pipeline_.restrict_normal_cauchy_row(
                                          plan.proxy_dof, node_point);
                            } else {
                                NativeGaussOwnerCorrection3D& row =
                                    foreign_rows[static_cast<std::size_t>(
                                        branch)][owner];
                                if (row.owner_dof < 0) {
                                    row.owner_dof = owner;
                                    row.normal_row =
                                        Eigen::VectorXd::Zero(dimension);
                                }
                                row.value_row += weight
                                    * pipeline_.restrict_value_jet_row(
                                          owner, node_point);
                                row.normal_row.noalias() += weight
                                    * pipeline_.restrict_normal_cauchy_row(
                                          owner, node_point);
                            }
                        }
                    }
                }
            }
        }
        for (int branch = 0; branch < 2; ++branch) {
            auto& destination =
                plan.branch[static_cast<std::size_t>(branch)].foreign_rows;
            destination.reserve(
                foreign_rows[static_cast<std::size_t>(branch)].size());
            for (auto& entry : foreign_rows[static_cast<std::size_t>(branch)])
                destination.push_back(std::move(entry.second));
        }
        return plan;
    }

    void interpolation_axes(
        const Eigen::Vector3d& query,
        std::array<int, 3>& start,
        std::array<std::array<double, kRestrictGridStencilSize>, 3>&
            axis_weights) const
    {
        const std::array<double, 3> origin = grid_.origin();
        const std::array<int, 3> dims = grid_.dof_dims();
        for (int axis = 0; axis < 3; ++axis) {
            const double coordinate =
                (query[axis] - origin[static_cast<std::size_t>(axis)]) / h_;
            if (coordinate < 0.0
                || coordinate > static_cast<double>(
                       dims[static_cast<std::size_t>(axis)] - 1)) {
                throw std::runtime_error(
                    "native Gauss normal layer exits the embedding box");
            }
            const int floor_index =
                static_cast<int>(std::floor(coordinate));
            start[static_cast<std::size_t>(axis)] = std::max(
                0,
                std::min(
                    floor_index - kRestrictGridLeftOffset,
                    dims[static_cast<std::size_t>(axis)]
                        - kRestrictGridStencilSize));
            const int reference =
                start[static_cast<std::size_t>(axis)]
                + kRestrictGridLeftOffset;
            axis_weights[static_cast<std::size_t>(axis)] =
                grid_lagrange_weights(
                    coordinate - static_cast<double>(reference));
        }
    }

    double raw_trace_value(
        const NativeGaussValueTracePlan3D& plan,
        const Eigen::VectorXd& potential) const
    {
        const auto& point = density_.trace_points()[
            static_cast<std::size_t>(plan.trace_index)];
        const auto layers = make_normal_layers();
        double result = 0.0;
        for (int side = 0; side < 2; ++side) {
            const double sign = side == 0 ? -1.0 : 1.0;
            for (int layer = 0; layer < kRestrictNormalLayerCount; ++layer) {
                const int trace_row =
                    side * kRestrictNormalLayerCount + layer;
                const Eigen::Vector3d query = point.point
                    + sign * layers[static_cast<std::size_t>(layer)]
                          * h_ * point.normal;
                std::array<int, 3> start{};
                std::array<std::array<double, kRestrictGridStencilSize>, 3>
                    axis_weights{};
                interpolation_axes(query, start, axis_weights);
                double layer_value = 0.0;
                for (int iz = 0; iz < kRestrictGridStencilSize; ++iz) {
                    for (int iy = 0; iy < kRestrictGridStencilSize; ++iy) {
                        for (int ix = 0; ix < kRestrictGridStencilSize; ++ix) {
                            const int node = grid_.index(
                                start[0] + ix,
                                start[1] + iy,
                                start[2] + iz);
                            layer_value +=
                                axis_weights[0][static_cast<std::size_t>(ix)]
                                * axis_weights[1][static_cast<std::size_t>(iy)]
                                * axis_weights[2][static_cast<std::size_t>(iz)]
                                * potential[node];
                        }
                    }
                }
                result += trace_weights_[static_cast<std::size_t>(trace_row)]
                    * layer_value;
            }
        }
        return result;
    }

    Eigen::VectorXd value_trace(
        const HarmonicJetField3D& field,
        const Eigen::Ref<const Eigen::VectorXd>& value_c0,
        bool interior) const
    {
        if (field.potential.size() != grid_.num_dofs()
            || value_c0.size() != density_.c0_coefficient_count()) {
            throw std::invalid_argument(
                "native Gauss restrict received incompatible field data");
        }
        if (field.restrict_normal_coefficients.rows()
                != pipeline_.surface_size()
            || field.restrict_normal_coefficients.cols()
                   != pipeline_.restrict_cauchy_dimension()) {
            throw std::invalid_argument(
                "native Gauss restrict requires separated normal Cauchy data");
        }
        Eigen::VectorXd result(trace_count());
        for (int trace = 0; trace < trace_count(); ++trace) {
            const NativeGaussValueTracePlan3D& plan =
                plans_[static_cast<std::size_t>(trace)];
            Eigen::Matrix<double, kLocalRestrictSampleCount, 1> samples;
            for (int q = 0; q < kLocalRestrictSampleCount; ++q) {
                samples[q] = plan.value_samples[static_cast<std::size_t>(q)]
                    .dot(value_c0);
            }
            Eigen::Matrix<double, 6, 1> value_jet =
                plan.value_recovery * samples;
            value_jet[0] = plan.target_value_sample.dot(value_c0);
            const NativeGaussBranchCorrection3D& correction =
                plan.branch[static_cast<std::size_t>(interior ? 1 : 0)];
            double value = raw_trace_value(plan, field.potential)
                + correction.target_value_row.dot(value_jet)
                + correction.target_normal_row.dot(
                      field.restrict_normal_coefficients.row(
                          plan.proxy_dof));
            for (const NativeGaussOwnerCorrection3D& foreign
                 : correction.foreign_rows) {
                if (field.restrict_value_jets.rows()
                        != pipeline_.surface_size()
                    || field.restrict_value_jets.cols() != 6) {
                    throw std::invalid_argument(
                        "native Gauss foreign C0 correction requires panel "
                        "value jets");
                }
                value += foreign.value_row.dot(
                    field.restrict_value_jets.row(foreign.owner_dof));
                value += foreign.normal_row.dot(
                    field.restrict_normal_coefficients.row(
                        foreign.owner_dof));
            }
            result[trace] = value;
        }
        if (!result.allFinite())
            throw std::runtime_error(
                "native Gauss restrict produced NaN/Inf");
        return result;
    }

    const CartesianGrid3D& grid_;
    const GridPair3D& grid_pair_;
    const PanelCenterHarmonicJetKFBI3D& pipeline_;
    const app3d::NativeNurbsDensitySpace3D& density_;
    double h_ = 0.0;
    std::array<double, kRestrictNormalSampleCount> trace_weights_{};
    std::vector<NativeGaussValueTracePlan3D> plans_;
};

// The native density reduction is an elimination basis c=R*a.  Iterating in
// a directly would make GMRES depend on the arbitrary pivot choices used to
// eliminate seam constraints.  These coordinates use y=(R^T R)^(1/2)*a, so
// c=Q*y with Q^T Q=I in the merged C0 coefficient metric.
class NativeOrthonormalReducedCoordinates3D {
public:
    explicit NativeOrthonormalReducedCoordinates3D(int identity_size)
        : size_(identity_size), low_rank_(true)
    {
        if (identity_size <= 0)
            throw std::invalid_argument("invalid identity coordinate size");
        low_rank_vectors_.resize(identity_size, 0);
        square_root_updates_.resize(0);
        inverse_square_root_updates_.resize(0);
    }

    explicit NativeOrthonormalReducedCoordinates3D(
        const Eigen::MatrixXd& reduction)
    {
        if (reduction.rows() < reduction.cols() || reduction.cols() == 0)
            throw std::invalid_argument("invalid native reduced density basis");
        size_ = static_cast<int>(reduction.cols());
        if (reduction.cols() > 1000) {
            build_low_rank_transform(reduction);
            return;
        }
        const Eigen::MatrixXd gram = reduction.transpose() * reduction;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(gram);
        if (eigensolver.info() != Eigen::Success)
            throw std::runtime_error(
                "could not diagonalize native density coordinate Gram matrix");
        eigenvectors_ = eigensolver.eigenvectors();
        const Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
        const double largest = eigenvalues.maxCoeff();
        const double smallest = eigenvalues.minCoeff();
        if (!(smallest > 128.0 * std::numeric_limits<double>::epsilon()
                             * largest)) {
            throw std::runtime_error(
                "native density elimination basis is numerically rank deficient");
        }
        square_roots_ = eigenvalues.array().sqrt();
        inverse_square_roots_ = square_roots_.array().inverse();
        condition_ = std::sqrt(largest / smallest);
    }

    Eigen::VectorXd to_native(const Eigen::VectorXd& orthonormal) const
    {
        check_size(orthonormal);
        if (low_rank_) {
            const Eigen::VectorXd projected =
                low_rank_vectors_.transpose() * orthonormal;
            return orthonormal
                 + low_rank_vectors_
                       * (inverse_square_root_updates_.array()
                          * projected.array()).matrix();
        }
        return eigenvectors_
             * (inverse_square_roots_.array()
                * orthonormal.array()).matrix();
    }

    Eigen::VectorXd from_native(const Eigen::VectorXd& native) const
    {
        check_size(native);
        if (low_rank_) {
            const Eigen::VectorXd projected =
                low_rank_vectors_.transpose() * native;
            return native
                 + low_rank_vectors_
                       * (square_root_updates_.array()
                          * projected.array()).matrix();
        }
        return (square_roots_.array()
                * (eigenvectors_.transpose() * native).array()).matrix();
    }

    Eigen::VectorXd dual_from_native(
        const Eigen::VectorXd& native_dual) const
    {
        check_size(native_dual);
        if (low_rank_)
            return to_native(native_dual);
        return (inverse_square_roots_.array()
                * (eigenvectors_.transpose() * native_dual).array())
            .matrix();
    }

    double condition() const { return condition_; }

private:
    void check_size(const Eigen::VectorXd& values) const
    {
        if (values.size() != size_)
            throw std::invalid_argument(
                "native density coordinate vector has the wrong size");
    }

    void build_low_rank_transform(const Eigen::MatrixXd& reduction)
    {
        const double scale = std::max(1.0, reduction.cwiseAbs().maxCoeff());
        const double tolerance =
            128.0 * std::numeric_limits<double>::epsilon() * scale;
        std::vector<unsigned char> identity_columns(
            static_cast<std::size_t>(size_), 0);
        std::vector<int> remainder_rows;
        remainder_rows.reserve(
            static_cast<std::size_t>(reduction.rows() - reduction.cols()));
        for (Eigen::Index row = 0; row < reduction.rows(); ++row) {
            int active_count = 0;
            int active_column = -1;
            double active_value = 0.0;
            for (Eigen::Index column = 0; column < reduction.cols(); ++column) {
                const double value = reduction(row, column);
                if (std::abs(value) <= tolerance)
                    continue;
                ++active_count;
                active_column = static_cast<int>(column);
                active_value = value;
                if (active_count > 1)
                    break;
            }
            const bool unused_identity =
                active_count == 1
                && std::abs(active_value - 1.0) <= tolerance
                && !identity_columns[static_cast<std::size_t>(active_column)];
            if (unused_identity)
                identity_columns[static_cast<std::size_t>(active_column)] = 1;
            else
                remainder_rows.push_back(static_cast<int>(row));
        }
        if (std::find(identity_columns.begin(), identity_columns.end(), 0)
                != identity_columns.end()
            || remainder_rows.size()
                   != static_cast<std::size_t>(
                       reduction.rows() - reduction.cols())) {
            throw std::runtime_error(
                "large native density reduction does not expose identity free rows");
        }

        if (remainder_rows.empty()) {
            low_rank_vectors_.resize(size_, 0);
            square_root_updates_.resize(0);
            inverse_square_root_updates_.resize(0);
            condition_ = 1.0;
            low_rank_ = true;
            return;
        }

        Eigen::MatrixXd eliminated(
            static_cast<Eigen::Index>(remainder_rows.size()),
            reduction.cols());
        for (Eigen::Index row = 0; row < eliminated.rows(); ++row)
            eliminated.row(row) = reduction.row(
                remainder_rows[static_cast<std::size_t>(row)]);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            eliminated, Eigen::ComputeThinV);
        const Eigen::VectorXd singular_values = svd.singularValues();
        if (singular_values.size() == 0 || !singular_values.allFinite())
            throw std::runtime_error(
                "could not factor native density low-rank Gram update");
        const double largest = singular_values[0] * singular_values[0];
        const double singular_tolerance =
            std::sqrt(256.0 * std::numeric_limits<double>::epsilon()
                      * std::max(1.0, largest));
        int positive_rank = 0;
        for (Eigen::Index i = 0; i < singular_values.size(); ++i)
            if (singular_values[i] > singular_tolerance)
                ++positive_rank;
        low_rank_vectors_ = svd.matrixV().leftCols(positive_rank);
        square_root_updates_.resize(positive_rank);
        inverse_square_root_updates_.resize(positive_rank);
        for (int column = 0; column < positive_rank; ++column) {
            const double sigma = singular_values[column];
            const double root = std::sqrt(1.0 + sigma * sigma);
            square_root_updates_[column] = root - 1.0;
            inverse_square_root_updates_[column] = 1.0 / root - 1.0;
        }
        condition_ = std::sqrt(1.0 + largest);
        low_rank_ = true;
    }

    Eigen::MatrixXd eigenvectors_;
    Eigen::VectorXd square_roots_;
    Eigen::VectorXd inverse_square_roots_;
    Eigen::MatrixXd low_rank_vectors_;
    Eigen::VectorXd square_root_updates_;
    Eigen::VectorXd inverse_square_root_updates_;
    int size_ = 0;
    bool low_rank_ = false;
    double condition_ = 1.0;
};

class NativeDensityTransfer3D {
public:
    NativeDensityTransfer3D(
        const NativeNurbsSurface3D& surface,
        const SurfaceDofCloud& samples,
        int coefficients_per_direction,
        app3d::NativeDensityField3D field,
        app3d::NativeDensityReductionBackend3D reduction_backend =
            app3d::NativeDensityReductionBackend3D::Legacy)
        : density_(
              surface,
              make_density_options(
                  coefficients_per_direction, field, reduction_backend))
        , sample_stencils_(make_sample_stencils(density_, samples))
        , c0_design_(make_c0_design(density_, sample_stencils_))
        , weights_(make_weights(samples))
        , projection_(make_projection(density_, c0_design_, weights_))
        , coordinates_(make_coordinates(density_))
    {
        if (projection_.rank() != density_.reduced_coefficient_count())
            throw std::runtime_error(
                "native density trace design is not full column rank");
        if (selected_neumann_native_gauss_trace_sampling()) {
            gauss_projection_ =
                make_gauss_projection(density_);
        }
        const Eigen::VectorXd ones = Eigen::VectorXd::Ones(sample_count());
        constant_orthonormal_ = fit_orthonormal(ones);
        constant_sample_error_ =
            (evaluate_orthonormal(constant_orthonormal_) - ones)
                .lpNorm<Eigen::Infinity>();
    }

    int coefficient_count() const
    {
        return density_.reduced_coefficient_count();
    }

    int sample_count() const
    {
        return static_cast<int>(weights_.size());
    }

    int gauss_trace_count() const
    {
        return static_cast<int>(density_.trace_points().size());
    }

    Eigen::VectorXd evaluate_native(const Eigen::VectorXd& native) const
    {
        if (native.size() != coefficient_count())
            throw std::invalid_argument(
                "native reduced density vector has the wrong size");
        return c0_design_ * expand_native_c0(native);
    }

    Eigen::VectorXd evaluate_orthonormal(
        const Eigen::VectorXd& orthonormal) const
    {
        return evaluate_native(coordinates_.to_native(orthonormal));
    }

    Eigen::VectorXd evaluate_trace_mass(
        const Eigen::VectorXd& mass_coordinates) const
    {
        return evaluate_native(
            projection_.trace_mass_to_native(mass_coordinates));
    }

    Eigen::VectorXd evaluate_gauss_trace_mass(
        const Eigen::VectorXd& mass_coordinates) const
    {
        return evaluate_native_at_gauss(
            gauss_projection().trace_mass_to_native(mass_coordinates));
    }

    Eigen::VectorXd evaluate_panel_gauss_trace_mass(
        const Eigen::VectorXd& mass_coordinates) const
    {
        return evaluate_native(
            gauss_projection().trace_mass_to_native(mass_coordinates));
    }

    Eigen::VectorXd expand_c0_orthonormal(
        const Eigen::VectorXd& orthonormal) const
    {
        return expand_native_c0(coordinates_.to_native(orthonormal));
    }

    Eigen::VectorXd expand_c0_trace_mass(
        const Eigen::VectorXd& mass_coordinates) const
    {
        return expand_native_c0(
            projection_.trace_mass_to_native(mass_coordinates));
    }

    Eigen::VectorXd expand_c0_gauss_trace_mass(
        const Eigen::VectorXd& mass_coordinates) const
    {
        return expand_native_c0(
            gauss_projection().trace_mass_to_native(mass_coordinates));
    }

    Eigen::VectorXd evaluate_native_at_gauss(
        const Eigen::VectorXd& native) const
    {
        if (native.size() != coefficient_count())
            throw std::invalid_argument(
                "native Gauss density vector has the wrong size");
        return density_.trace_c0_design()
             * expand_native_c0(native);
    }

    Eigen::VectorXd evaluate_c0_at_gauss(
        const Eigen::VectorXd& c0_coefficients) const
    {
        if (c0_coefficients.size() != density_.c0_coefficient_count())
            throw std::invalid_argument(
                "native Gauss C0 density vector has the wrong size");
        return density_.trace_c0_design() * c0_coefficients;
    }

    Eigen::VectorXd evaluate_c0_samples(
        const Eigen::VectorXd& c0_coefficients) const
    {
        if (c0_coefficients.size() != density_.c0_coefficient_count())
            throw std::invalid_argument(
                "C0 density vector has the wrong size");
        return c0_design_ * c0_coefficients;
    }

    double evaluate_c0_at_sample(
        int sample,
        const Eigen::VectorXd& c0_coefficients) const
    {
        if (sample < 0 || sample >= sample_count()
            || c0_coefficients.size() != density_.c0_coefficient_count()) {
            throw std::invalid_argument(
                "direct density sample evaluation received incompatible data");
        }
        return sample_stencils_[static_cast<std::size_t>(sample)]
            .dot(c0_coefficients);
    }

    Eigen::VectorXd fit_native(const Eigen::VectorXd& samples) const
    {
        if (samples.size() != sample_count())
            throw std::invalid_argument(
                "native density fit received the wrong sample count");
        return projection_.apply(samples);
    }

    Eigen::VectorXd fit_orthonormal(const Eigen::VectorXd& samples) const
    {
        return coordinates_.from_native(fit_native(samples));
    }

    Eigen::VectorXd fit_trace_mass(const Eigen::VectorXd& samples) const
    {
        return projection_.native_to_trace_mass(fit_native(samples));
    }

    Eigen::VectorXd fit_gauss_orthonormal(
        const Eigen::VectorXd& samples) const
    {
        if (samples.size() != gauss_trace_count())
            throw std::invalid_argument(
                "native Gauss density fit received the wrong sample count");
        return coordinates_.from_native(gauss_projection().apply(samples));
    }

    Eigen::VectorXd fit_gauss_trace_mass(
        const Eigen::VectorXd& samples) const
    {
        if (samples.size() != gauss_trace_count())
            throw std::invalid_argument(
                "native Gauss density fit received the wrong sample count");
        return gauss_projection().native_to_trace_mass(
            gauss_projection().apply(samples));
    }

    Eigen::VectorXd galerkin_orthonormal(
        const Eigen::VectorXd& samples) const
    {
        if (samples.size() != sample_count()) {
            throw std::invalid_argument(
                "native Galerkin trace received the wrong sample count");
        }
        const Eigen::VectorXd weighted =
            (weights_.array() * samples.array()).matrix();
        const Eigen::VectorXd c0_dual = c0_design_.transpose() * weighted;
        Eigen::VectorXd native_dual = c0_dual;
        if (!density_.uses_identity_reduction()) {
            native_dual =
                density_.reduction_matrix().transpose() * c0_dual;
        }
        return coordinates_.dual_from_native(native_dual);
    }

    double weighted_mean(const Eigen::VectorXd& samples) const
    {
        if (samples.size() != sample_count())
            throw std::invalid_argument(
                "native density mean received the wrong sample count");
        return weights_.dot(samples) / weights_.sum();
    }

    double gauss_weighted_mean(const Eigen::VectorXd& samples) const
    {
        if (samples.size() != gauss_trace_count())
            throw std::invalid_argument(
                "native Gauss mean received the wrong sample count");
        return density_.trace_weights().dot(samples)
             / density_.surface_area();
    }

    double c0_weighted_mean(const Eigen::VectorXd& c0_coefficients) const
    {
        if (c0_coefficients.size() != density_.c0_coefficient_count())
            throw std::invalid_argument(
                "native C0 mean received the wrong coefficient count");
        return density_.c0_mass_row().dot(c0_coefficients)
             / density_.surface_area();
    }

    const app3d::NativeNurbsDensitySpace3D& density() const
    {
        return density_;
    }

    const app3d::ReducedTraceProjection3D& projection() const
    {
        return projection_;
    }

    const app3d::ReducedTraceProjection3D& gauss_projection() const
    {
        if (!gauss_projection_)
            throw std::runtime_error(
                "native Gauss projection was not initialized");
        return *gauss_projection_;
    }

    const Eigen::SparseMatrix<double>& c0_design() const noexcept
    {
        return c0_design_;
    }

    const Eigen::VectorXd& weights() const noexcept
    {
        return weights_;
    }

    double coordinate_condition() const
    {
        return coordinates_.condition();
    }

    double constant_sample_error() const
    {
        return constant_sample_error_;
    }

private:
    static app3d::ReducedTraceProjection3D make_projection(
        const app3d::NativeNurbsDensitySpace3D& density,
        const Eigen::SparseMatrix<double>& c0_design,
        const Eigen::VectorXd& weights)
    {
        if (density.uses_identity_reduction())
            return app3d::ReducedTraceProjection3D(c0_design, weights);
        return app3d::ReducedTraceProjection3D(
            c0_design, density.reduction_matrix(), weights);
    }

    static NativeOrthonormalReducedCoordinates3D make_coordinates(
        const app3d::NativeNurbsDensitySpace3D& density)
    {
        if (density.uses_identity_reduction())
            return NativeOrthonormalReducedCoordinates3D(
                density.c0_coefficient_count());
        return NativeOrthonormalReducedCoordinates3D(
            density.reduction_matrix());
    }

    static std::unique_ptr<app3d::ReducedTraceProjection3D>
    make_gauss_projection(
        const app3d::NativeNurbsDensitySpace3D& density)
    {
        if (density.uses_identity_reduction()) {
            return std::make_unique<app3d::ReducedTraceProjection3D>(
                density.trace_c0_design(), density.trace_weights());
        }
        return std::make_unique<app3d::ReducedTraceProjection3D>(
            density.trace_c0_design(), density.reduction_matrix(),
            density.trace_weights());
    }

    Eigen::VectorXd expand_native_c0(
        const Eigen::Ref<const Eigen::VectorXd>& native) const
    {
        if (density_.uses_identity_reduction())
            return Eigen::VectorXd(native);
        return density_.reduction_matrix() * native;
    }

    static app3d::NativeNurbsDensityOptions3D make_density_options(
        int coefficients_per_direction,
        app3d::NativeDensityField3D field,
        app3d::NativeDensityReductionBackend3D reduction_backend)
    {
        app3d::NativeNurbsDensityOptions3D options;
        options.field = field;
        options.reduction_backend = reduction_backend;
        options.coefficients_per_direction = coefficients_per_direction;
        // One cubic element has four basis functions in each direction.
        // A 3-point rule integrates its mass matrix but does not provide an
        // injective point-evaluation map (3 < 4).  Use the minimal
        // unisolvent order only for this coarsest density space.
        options.trace_gauss_order =
            selected_neumann_native_gauss_trace_sampling()
                && coefficients_per_direction == 4
            ? 4
            : 3;
        return options;
    }

    static std::vector<app3d::NativeDensityC0Stencil3D>
    make_sample_stencils(
        const app3d::NativeNurbsDensitySpace3D& density,
        const SurfaceDofCloud& samples)
    {
        std::vector<app3d::NativeDensityC0Stencil3D> result;
        result.reserve(samples.dofs.size());
        for (const SurfaceDof& sample : samples.dofs) {
            result.push_back(density.c0_basis_stencil(
                sample.patch_id, sample.u, sample.v));
        }
        return result;
    }

    static Eigen::SparseMatrix<double> make_c0_design(
        const app3d::NativeNurbsDensitySpace3D& density,
        const std::vector<app3d::NativeDensityC0Stencil3D>& sample_stencils)
    {
        Eigen::SparseMatrix<double> result(
            static_cast<int>(sample_stencils.size()),
            density.c0_coefficient_count());
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(sample_stencils.size() * 16);
        // The same rows are retained for sample evaluation.  Reuse them here
        // without repeating B-spline evaluation or changing COO entry order.
        for (int row = 0; row < static_cast<int>(sample_stencils.size()); ++row) {
            const app3d::NativeDensityC0Stencil3D& stencil =
                sample_stencils[static_cast<std::size_t>(row)];
            for (int q = 0; q < stencil.count; ++q) {
                triplets.emplace_back(
                    row,
                    stencil.indices[static_cast<std::size_t>(q)],
                    stencil.weights[static_cast<std::size_t>(q)]);
            }
        }
        result.setFromTriplets(triplets.begin(), triplets.end());
        result.makeCompressed();
        return result;
    }

    static Eigen::VectorXd make_weights(const SurfaceDofCloud& samples)
    {
        Eigen::VectorXd result(static_cast<int>(samples.dofs.size()));
        for (int q = 0; q < result.size(); ++q) {
            result[q] = samples.dofs[static_cast<std::size_t>(q)].weight;
            if (!(result[q] > 0.0) || !std::isfinite(result[q]))
                throw std::runtime_error(
                    "native density trace sample has invalid weight");
        }
        return result;
    }

    app3d::NativeNurbsDensitySpace3D density_;
    std::vector<app3d::NativeDensityC0Stencil3D> sample_stencils_;
    Eigen::SparseMatrix<double> c0_design_;
    Eigen::VectorXd weights_;
    app3d::ReducedTraceProjection3D projection_;
    std::unique_ptr<app3d::ReducedTraceProjection3D> gauss_projection_;
    NativeOrthonormalReducedCoordinates3D coordinates_;
    Eigen::VectorXd constant_orthonormal_;
    double constant_sample_error_ = 0.0;
};

class FunctionOperator3D final : public IKFBIOperator {
public:
    using Function =
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;

    FunctionOperator3D(int size, Function function)
        : size_(size), function_(std::move(function))
    {}

    int problem_size() const override { return size_; }

    void apply(const Eigen::VectorXd& x,
               Eigen::VectorXd& y) const override
    {
        if (x.size() != size_)
            throw std::invalid_argument(
                "coefficient KFBI operator input has the wrong size");
        function_(x, y);
    }

private:
    int size_ = 0;
    Function function_;
};

HarmonicJetField3D evaluate_coefficient_crossing_field(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const Eigen::VectorXd& value_samples,
    const Eigen::VectorXd& normal_samples,
    bool use_feature_value_jet,
    bool anchor_center_cauchy,
    bool use_local_restrict_fit = false,
    const NativeDensityTransfer3D* value_transfer = nullptr,
    const Eigen::VectorXd* value_c0 = nullptr,
    const NativeDensityTransfer3D* normal_transfer = nullptr,
    const Eigen::VectorXd* normal_c0 = nullptr,
    const app3d::KnownNeumannJetCallback3D& known_normal_jet = {},
    const app3d::KnownDirichletJetCallback3D& known_value_jet = {})
{
    if ((value_transfer == nullptr) != (value_c0 == nullptr)
        || (normal_transfer == nullptr) != (normal_c0 == nullptr)) {
        throw std::invalid_argument(
            "coefficient crossing field has an incomplete density evaluator");
    }
    auto value_at = [&](int sample) {
        return value_transfer != nullptr
            ? value_transfer->evaluate_c0_at_sample(sample, *value_c0)
            : value_samples[sample];
    };
    auto normal_at = [&](int sample) {
        return normal_transfer != nullptr
            ? normal_transfer->evaluate_c0_at_sample(sample, *normal_c0)
            : normal_samples[sample];
    };
    HarmonicJetField3D result;
    if (value_transfer != nullptr
        && pipeline.has_direct_coefficient_value_density(
               value_transfer->density())) {
        result = pipeline.evaluate_direct_coefficient_crossing_local(
            value_transfer->density(), *value_c0,
            value_samples, normal_samples, known_normal_jet,
            anchor_center_cauchy, use_local_restrict_fit);
    } else if (normal_transfer != nullptr
               && pipeline.has_direct_coefficient_normal_density(
                      normal_transfer->density())) {
        result = pipeline.evaluate_direct_coefficient_normal_crossing_local(
            normal_transfer->density(), *normal_c0,
            value_samples, normal_samples,
            use_feature_value_jet, anchor_center_cauchy,
            use_local_restrict_fit, known_value_jet);
    } else {
        result = pipeline.evaluate_crossing_local(
            value_samples, normal_samples, value_at, normal_at,
            use_feature_value_jet, anchor_center_cauchy,
            use_local_restrict_fit);
    }
    if (use_local_restrict_fit && !result.direct_coefficient_cauchy) {
        if (value_transfer != nullptr) {
            result.restrict_value_jets =
                pipeline.recover_local_quadratic_value_jets(
                    value_transfer->density(), *value_c0);
        } else if (value_samples.isZero(0.0)) {
            result.restrict_value_jets =
                Eigen::MatrixXd::Zero(pipeline.surface_size(), 6);
        } else {
            throw std::invalid_argument(
                "local quadratic coefficient restrict requires direct "
                "value-density coefficients");
        }
    }
    return result;
}

class ExteriorZeroTraceOperator3D final : public IKFBIOperator {
public:
    ExteriorZeroTraceOperator3D(
        const PanelCenterHarmonicJetKFBI3D& pipeline,
        NeumannCompatibilityMode3D compatibility_mode,
        ExteriorNormalRestrictMode3D trace_restrict_mode)
        : pipeline_(pipeline)
        , compatibility_mode_(compatibility_mode)
        , trace_restrict_mode_(trace_restrict_mode)
        , border_column_(Eigen::VectorXd::Ones(pipeline.surface_size()))
    {
        if (compatibility_mode_
            == NeumannCompatibilityMode3D::OperatorFluxCorrection) {
            const int size = pipeline_.surface_size();
            const Eigen::VectorXd zero_value = Eigen::VectorXd::Zero(size);
            const Eigen::VectorXd unit_normal = Eigen::VectorXd::Ones(size);
            const HarmonicJetField3D unit_field =
                pipeline_.evaluate(zero_value, unit_normal);
            // The scalar unknown c corrects prescribed data as g_h-c, so the
            // bordered column is the physical operator response -B_h 1.
            border_column_ = -pipeline_.exterior_trace(
                unit_field, zero_value, unit_normal, trace_restrict_mode_);
        }
    }

    int problem_size() const override
    {
        return pipeline_.surface_size() + 1;
    }

    void apply(const Eigen::VectorXd& unknown,
               Eigen::VectorXd& result) const override
    {
        if (unknown.size() != problem_size())
            throw std::invalid_argument("exterior-zero operator input has wrong size");
        const int size = pipeline_.surface_size();
        const Eigen::VectorXd value_jump = unknown.head(size);
        const Eigen::VectorXd zero_normal = Eigen::VectorXd::Zero(size);
        const HarmonicJetField3D field =
            pipeline_.evaluate(value_jump, zero_normal);
        const Eigen::VectorXd trace =
            pipeline_.exterior_trace(
                field, value_jump, zero_normal, trace_restrict_mode_);
        result.resize(size + 1);
        result.head(size) = trace + unknown[size] * border_column_;
        double weighted_mean = 0.0;
        for (int q = 0; q < size; ++q) {
            weighted_mean += pipeline_.surface().dofs[static_cast<std::size_t>(q)].weight
                           * value_jump[q];
        }
        result[size] = weighted_mean / pipeline_.surface_area();
    }

    Eigen::VectorXd right_hand_side(
        const Eigen::VectorXd& prescribed_normal_jump) const
    {
        const int size = pipeline_.surface_size();
        if (prescribed_normal_jump.size() != size)
            throw std::invalid_argument("prescribed Neumann data has wrong size");
        const Eigen::VectorXd zero_value = Eigen::VectorXd::Zero(size);
        const HarmonicJetField3D field =
            pipeline_.evaluate(zero_value, prescribed_normal_jump);
        Eigen::VectorXd result = Eigen::VectorXd::Zero(size + 1);
        result.head(size) = -pipeline_.exterior_trace(
            field, zero_value, prescribed_normal_jump, trace_restrict_mode_);
        return result;
    }

    const Eigen::VectorXd& border_column() const
    {
        return border_column_;
    }

private:
    const PanelCenterHarmonicJetKFBI3D& pipeline_;
    NeumannCompatibilityMode3D compatibility_mode_ =
        NeumannCompatibilityMode3D::OperatorFluxCorrection;
    ExteriorNormalRestrictMode3D trace_restrict_mode_ =
        ExteriorNormalRestrictMode3D::JointTricubicCauchy;
    Eigen::VectorXd border_column_;
};

struct ExteriorZeroTraceSolution3D {
    Eigen::VectorXd value_jump;
    Eigen::VectorXd normal_jump;
    Eigen::VectorXd potential;
    Eigen::MatrixXd coefficients;
    Eigen::VectorXd augmented_residual;
    std::vector<double> gmres_residuals;
    NeumannCompatibilityMode3D compatibility_mode =
        NeumannCompatibilityMode3D::OperatorFluxCorrection;
    double lagrange_multiplier = 0.0;
    double operator_compatibility_correction = 0.0;
    double border_column_linf = 0.0;
    int iterations = 0;
    bool converged = false;
};

ExteriorZeroTraceSolution3D solve_exterior_zero_trace_neumann_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const Eigen::VectorXd& prescribed_normal_jump,
    NeumannCompatibilityMode3D compatibility_mode,
    ExteriorNormalRestrictMode3D trace_restrict_mode,
    double tolerance,
    int restart,
    int max_iterations)
{
    ExteriorZeroTraceOperator3D op(
        pipeline, compatibility_mode, trace_restrict_mode);
    const Eigen::VectorXd rhs = op.right_hand_side(prescribed_normal_jump);
    Eigen::VectorXd augmented_unknown = Eigen::VectorXd::Zero(op.problem_size());
    GMRES gmres(max_iterations, tolerance, restart);
    ExteriorZeroTraceSolution3D result;
    result.iterations = gmres.solve(op, rhs, augmented_unknown);
    result.converged = gmres.converged();
    result.gmres_residuals = gmres.residuals();
    const int size = pipeline.surface_size();
    result.compatibility_mode = compatibility_mode;
    result.value_jump = augmented_unknown.head(size);
    result.normal_jump = prescribed_normal_jump;
    if (compatibility_mode
        == NeumannCompatibilityMode3D::OperatorFluxCorrection) {
        result.operator_compatibility_correction = augmented_unknown[size];
        result.normal_jump.array() -=
            result.operator_compatibility_correction;
    } else {
        result.lagrange_multiplier = augmented_unknown[size];
    }
    result.border_column_linf =
        op.border_column().lpNorm<Eigen::Infinity>();
    const HarmonicJetField3D field =
        pipeline.evaluate(result.value_jump, result.normal_jump);
    result.potential = field.potential;
    result.coefficients = field.coefficients;
    Eigen::VectorXd applied;
    op.apply(augmented_unknown, applied);
    result.augmented_residual = applied - rhs;
    return result;
}

class ExteriorNormalTraceOperator3D final : public IKFBIOperator {
public:
    ExteriorNormalTraceOperator3D(
        const PanelCenterHarmonicJetKFBI3D& pipeline,
        ExteriorNormalRestrictMode3D mode)
        : pipeline_(pipeline)
        , mode_(mode)
    {}

    int problem_size() const override
    {
        return pipeline_.surface_size();
    }

    void apply(const Eigen::VectorXd& normal_jump,
               Eigen::VectorXd& result) const override
    {
        if (normal_jump.size() != problem_size()) {
            throw std::invalid_argument(
                "exterior-normal operator input has wrong size");
        }
        const Eigen::VectorXd zero_value =
            Eigen::VectorXd::Zero(problem_size());
        const HarmonicJetField3D field =
            pipeline_.evaluate(zero_value, normal_jump);
        result = pipeline_.exterior_normal_trace(
            field, zero_value, normal_jump, mode_);
    }

    Eigen::VectorXd right_hand_side(
        const Eigen::VectorXd& prescribed_value_jump) const
    {
        if (prescribed_value_jump.size() != problem_size()) {
            throw std::invalid_argument(
                "prescribed Dirichlet data has wrong size");
        }
        const Eigen::VectorXd zero_normal =
            Eigen::VectorXd::Zero(problem_size());
        const HarmonicJetField3D field =
            pipeline_.evaluate(prescribed_value_jump, zero_normal);
        return -pipeline_.exterior_normal_trace(
            field, prescribed_value_jump, zero_normal, mode_);
    }

private:
    const PanelCenterHarmonicJetKFBI3D& pipeline_;
    ExteriorNormalRestrictMode3D mode_;
};

struct ExteriorNormalTraceSolution3D {
    Eigen::VectorXd normal_jump;
    Eigen::VectorXd potential;
    Eigen::MatrixXd coefficients;
    Eigen::VectorXd operator_residual;
    std::vector<double> gmres_residuals;
    int iterations = 0;
    bool converged = false;
};

ExteriorNormalTraceSolution3D solve_exterior_zero_normal_dirichlet_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const Eigen::VectorXd& prescribed_value_jump,
    ExteriorNormalRestrictMode3D mode,
    double tolerance,
    int restart,
    int max_iterations)
{
    ExteriorNormalTraceOperator3D op(pipeline, mode);
    const Eigen::VectorXd rhs = op.right_hand_side(prescribed_value_jump);
    Eigen::VectorXd normal_jump = Eigen::VectorXd::Zero(op.problem_size());
    GMRES gmres(max_iterations, tolerance, restart);
    ExteriorNormalTraceSolution3D result;
    result.iterations = gmres.solve(op, rhs, normal_jump);
    result.converged = gmres.converged();
    result.gmres_residuals = gmres.residuals();
    result.normal_jump = normal_jump;
    const HarmonicJetField3D field =
        pipeline.evaluate(prescribed_value_jump, normal_jump);
    result.potential = field.potential;
    result.coefficients = field.coefficients;
    Eigen::VectorXd applied;
    op.apply(normal_jump, applied);
    result.operator_residual = applied - rhs;
    return result;
}

struct CoefficientExteriorZeroTraceSolution3D {
    Eigen::VectorXd orthonormal_value_coefficients;
    Eigen::VectorXd value_c0_coefficients;
    Eigen::VectorXd value_jump;
    Eigen::VectorXd normal_jump;
    Eigen::VectorXd potential;
    Eigen::MatrixXd cauchy_coefficients;
    Eigen::MatrixXd restrict_value_jets;
    Eigen::MatrixXd restrict_normal_coefficients;
    Eigen::VectorXd augmented_residual;
    std::vector<double> gmres_residuals;
    NeumannCompatibilityMode3D compatibility_mode =
        NeumannCompatibilityMode3D::OperatorFluxCorrection;
    double lagrange_multiplier = 0.0;
    double operator_compatibility_correction = 0.0;
    double border_column_linf = 0.0;
    std::string edge_jump_jet = "disabled";
    int edge_feature_edges = 0;
    int edge_constraint_rows = 0;
    int edge_constraint_rank = 0;
    int edge_reduced_dofs = 0;
    double edge_constraint_residual_linf = 0.0;
    double edge_nullspace_residual_linf = 0.0;
    double edge_normal_fit_linf = 0.0;
    double edge_target_projection_linf = 0.0;
    double feature_target_requested_linf = 0.0;
    double feature_target_requested_l2 = 0.0;
    double feature_target_projected_linf = 0.0;
    double feature_target_projected_l2 = 0.0;
    double feature_target_projection_l2 = 0.0;
    double feature_target_projection_relative_l2 = 0.0;
    double feature_target_reachable_certification_linf = 0.0;
    int feature_target_projection_components = 0;
    bool feature_target_constraints_exact = true;
    std::string reduction_scheme = "legacy_native";
    std::string particular_solver = "not_applicable";
    int topology_constraint_rows = 0;
    int topology_constraint_rank = 0;
    // c_full = A0 c_base and c_base = p + E z.
    int topology_base_coordinates = 0;
    int topology_full_coordinates = 0;
    int topology_pre_mean_coordinates = 0;
    int topology_reduced_coordinates = 0;
    int topology_block_count = 0;
    double topology_particular_residual_linf = 0.0;
    double topology_homogeneous_residual_linf = 0.0;
    int topology_projector_rank = 0;
    double topology_projector_condition = 0.0;
    double topology_projector_pb_error = 0.0;
    int topology_trace_samples = 0;
    int topology_trace_oversampling_margin = 0;
    double topology_trace_oversampling_ratio = 0.0;
    double topology_trace_closure_linf = 0.0;
    int mean_free_pivot_index = -1;
    double mean_free_pivot_moment = 0.0;
    double mean_free_relative_observability = 0.0;
    double mean_free_coordinate_condition = 1.0;
    double mean_free_particular_residual = 0.0;
    double mean_free_homogeneous_residual = 0.0;
    double mean_free_constant_projection_linf = 0.0;
    double mean_free_intrinsic_mean = 0.0;
    std::vector<app3d::AffineEliminationRecord3D> topology_block_records;
    int iterations = 0;
    bool converged = false;
};

CoefficientExteriorZeroTraceSolution3D
solve_exterior_zero_trace_neumann_coefficients_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const NativeDensityTransfer3D& value_transfer,
    const NativeDensityTransfer3D* normal_transfer,
    const NativeGaussExteriorValueRestrict3D* gauss_restrict,
    const NativeDensityValueSpread3D* native_value_spread,
    const Eigen::VectorXd& prescribed_normal_jump,
    NeumannCompatibilityMode3D compatibility_mode,
    ExteriorNormalRestrictMode3D trace_restrict_mode,
    NeumannRestrictValueJetMode3D restrict_value_jet_mode,
    double tolerance,
    int restart,
    int max_iterations,
    const app3d::KnownNeumannJetCallback3D& known_normal_jet = {})
{
    const int sample_size = pipeline.surface_size();
    const int coefficient_size = value_transfer.coefficient_count();
    if (value_transfer.sample_count() != sample_size
        || prescribed_normal_jump.size() != sample_size) {
        throw std::invalid_argument(
            "coefficient Neumann solve received incompatible sample sizes");
    }
    const Eigen::VectorXd zero_value = Eigen::VectorXd::Zero(sample_size);
    const Eigen::VectorXd zero_normal = Eigen::VectorXd::Zero(sample_size);
    const Eigen::VectorXd zero_value_c0 = Eigen::VectorXd::Zero(
        value_transfer.density().c0_coefficient_count());
    const bool use_gauss_restrict = gauss_restrict != nullptr;
    if ((native_value_spread != nullptr) != use_gauss_restrict) {
        throw std::invalid_argument(
            "native density spread and Gauss restrict must be enabled "
            "together");
    }
    if (use_gauss_restrict
        && gauss_restrict->trace_count()
               != value_transfer.gauss_trace_count()) {
        throw std::invalid_argument(
            "native Gauss restrict and density projection sizes differ");
    }
    const bool use_feature_value_jet = restrict_value_jet_mode
        == NeumannRestrictValueJetMode3D::FeatureConstrained;
    const bool anchor_center_cauchy = trace_restrict_mode
        == ExteriorNormalRestrictMode3D::
               GlobalCubicExteriorBranchCrossingOwner;
    const bool use_local_restrict_fit = anchor_center_cauchy;
    const bool use_galerkin_projection =
        selected_neumann_galerkin_trace_projection();
    const bool use_trace_mass_coordinates =
        selected_neumann_trace_mass_coordinates();
    const NeumannEdgeJumpJetMode3D edge_jump_jet_mode =
        selected_neumann_edge_jump_jet_mode();
    const bool use_edge_jump_jet = edge_jump_jet_mode
        == NeumannEdgeJumpJetMode3D::StrongFeatureMortar;
    const bool use_topology_ambient_gradient = edge_jump_jet_mode
        == NeumannEdgeJumpJetMode3D::TopologyAmbientGradientLocalSvd;
    const bool use_topology_affine = edge_jump_jet_mode
            == NeumannEdgeJumpJetMode3D::TopologyAffineLocalSvd
        || use_topology_ambient_gradient;
    const bool use_mean_free_pivot =
        selected_neumann_mean_free_pivot_elimination();
    if (use_mean_free_pivot && !use_topology_affine) {
        throw std::invalid_argument(
            "mean_free_pivot_elimination is implemented only for the "
            "topology-affine density spaces");
    }
    if (use_galerkin_projection && use_trace_mass_coordinates) {
        throw std::invalid_argument(
            "Galerkin trace projection and trace-mass density coordinates "
            "cannot be enabled together");
    }
    if ((use_edge_jump_jet || use_topology_affine)
        && !use_trace_mass_coordinates) {
        throw std::invalid_argument(
            "feature-edge affine constraints require trace_mass "
            "density coordinates");
    }
    if ((use_edge_jump_jet || use_topology_affine)
        && normal_transfer == nullptr) {
        throw std::invalid_argument(
            "feature-edge affine constraints require the broken "
            "normal-density space");
    }
    if (use_topology_affine
        && compatibility_mode
               != NeumannCompatibilityMode3D::TraceBorderLegacy) {
        throw std::invalid_argument(
            "topology-affine modes keep the prescribed, pre-mean-removed "
            "Neumann data unchanged; select trace_border_legacy (the legacy "
            "configuration name now denotes this no-flux-correction mode)");
    }
    if (use_topology_affine && use_gauss_restrict) {
        throw std::invalid_argument(
            "topology-affine modes currently use the panel-center "
            "independent geometric test set; native_gauss is a separate "
            "projection backend");
    }
    const auto project_trace = [&](const Eigen::VectorXd& trace) {
        if (use_gauss_restrict) {
            return use_trace_mass_coordinates
                ? value_transfer.fit_gauss_trace_mass(trace)
                : value_transfer.fit_gauss_orthonormal(trace);
        }
        if (use_trace_mass_coordinates)
            return value_transfer.fit_trace_mass(trace);
        return use_galerkin_projection
            ? value_transfer.galerkin_orthonormal(trace)
            : value_transfer.fit_orthonormal(trace);
    };
    const auto expand_c0_coordinates = [&](
        const Eigen::VectorXd& coordinates) {
        return use_trace_mass_coordinates
            ? (use_gauss_restrict
                   ? value_transfer.expand_c0_gauss_trace_mass(coordinates)
                   : value_transfer.expand_c0_trace_mass(coordinates))
            : value_transfer.expand_c0_orthonormal(coordinates);
    };
    const auto evaluate_coordinates = [&](
        const Eigen::VectorXd& coordinates) {
        return use_trace_mass_coordinates
            ? (use_gauss_restrict
                   ? value_transfer.evaluate_panel_gauss_trace_mass(
                         coordinates)
                   : value_transfer.evaluate_trace_mass(coordinates))
            : value_transfer.evaluate_orthonormal(coordinates);
    };
    const auto restrict_exterior = [&](
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_c0,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) {
        return use_gauss_restrict
            ? gauss_restrict->exterior_value_trace(field, value_c0)
            : pipeline.exterior_trace(
                  field, value_jump, normal_jump, trace_restrict_mode);
    };
    const auto evaluate_value_field = [&](
        const Eigen::VectorXd& value_c0,
        const Eigen::VectorXd& value_jump) {
        if (!use_gauss_restrict) {
            return evaluate_coefficient_crossing_field(
                pipeline, value_jump, zero_normal,
                use_feature_value_jet, anchor_center_cauchy,
                use_local_restrict_fit,
                &value_transfer, &value_c0);
        }
        HarmonicJetField3D field;
        field.potential = native_value_spread->evaluate(value_c0);
        field.coefficients = Eigen::MatrixXd::Zero(
            sample_size, pipeline.restrict_cauchy_dimension());
        field.restrict_value_jets =
            pipeline.recover_local_quadratic_value_jets(
                value_transfer.density(), value_c0);
        field.restrict_normal_coefficients = Eigen::MatrixXd::Zero(
            sample_size, pipeline.restrict_cauchy_dimension());
        return field;
    };

    const auto apply_trace_block = [&](
        const Eigen::VectorXd& coordinates,
        Eigen::VectorXd& result) {
        const Eigen::VectorXd value_c0 =
            expand_c0_coordinates(coordinates);
        const Eigen::VectorXd value_jump =
            value_transfer.evaluate_c0_samples(value_c0);
        const HarmonicJetField3D field =
            evaluate_value_field(value_c0, value_jump);
        const Eigen::VectorXd trace = restrict_exterior(
            field, value_c0, value_jump, zero_normal);
        result = project_trace(trace);
    };

    Eigen::VectorXd border_samples;
    Eigen::VectorXd border_coefficients;
    Eigen::VectorXd rhs;
    std::unique_ptr<FunctionOperator3D> op;
    if (!use_topology_affine) {
        border_samples = Eigen::VectorXd::Ones(
            use_gauss_restrict ? gauss_restrict->trace_count() : sample_size);
        if (compatibility_mode
            == NeumannCompatibilityMode3D::OperatorFluxCorrection) {
            const Eigen::VectorXd unit_normal =
                Eigen::VectorXd::Ones(sample_size);
            const HarmonicJetField3D unit_field =
                evaluate_coefficient_crossing_field(
                    pipeline, zero_value, unit_normal,
                    use_feature_value_jet, anchor_center_cauchy,
                    use_local_restrict_fit);
            border_samples = -restrict_exterior(
                unit_field, zero_value_c0, zero_value, unit_normal);
        }
        border_coefficients = project_trace(border_samples);
        op = std::make_unique<FunctionOperator3D>(
            coefficient_size + 1,
            [&](const Eigen::VectorXd& augmented_unknown,
                Eigen::VectorXd& image) {
                const Eigen::VectorXd value_c0 = expand_c0_coordinates(
                    augmented_unknown.head(coefficient_size));
                const Eigen::VectorXd value_jump =
                    value_transfer.evaluate_c0_samples(value_c0);
                image.resize(coefficient_size + 1);
                Eigen::VectorXd trace_block;
                apply_trace_block(
                    augmented_unknown.head(coefficient_size), trace_block);
                image.head(coefficient_size) = trace_block
                    + augmented_unknown[coefficient_size]
                          * border_coefficients;
                image[coefficient_size] =
                    (use_edge_jump_jet || use_gauss_restrict)
                    ? value_transfer.c0_weighted_mean(value_c0)
                    : value_transfer.weighted_mean(value_jump);
            });

        const HarmonicJetField3D data_field =
            evaluate_coefficient_crossing_field(
                pipeline, zero_value, prescribed_normal_jump,
                use_feature_value_jet, anchor_center_cauchy,
                use_local_restrict_fit);
        const Eigen::VectorXd data_trace = restrict_exterior(
            data_field, zero_value_c0, zero_value,
            prescribed_normal_jump);
        rhs = Eigen::VectorXd::Zero(coefficient_size + 1);
        rhs.head(coefficient_size) = -project_trace(data_trace);
    }

    Eigen::VectorXd unknown = Eigen::VectorXd::Zero(coefficient_size + 1);
    Eigen::VectorXd topology_augmented_residual;
    Eigen::VectorXd topology_final_c0;
    Eigen::VectorXd topology_final_jump;
    CoefficientExteriorZeroTraceSolution3D result;
    result.edge_jump_jet = use_topology_affine
        ? (use_topology_ambient_gradient
               ? "topology_ambient_gradient_local_svd"
               : "topology_affine_local_svd")
        : (use_edge_jump_jet ? "strong_feature_mortar" : "disabled");
    result.reduction_scheme = use_topology_affine
        ? "topology_affine_local_svd"
        : "legacy_native";
    const bool use_projected_border =
        selected_neumann_projected_border_elimination();
    const bool use_householder_border =
        selected_neumann_householder_border_elimination();
    if (use_topology_affine && !use_mean_free_pivot) {
        throw std::invalid_argument(
            "topology-affine modes eliminate the global density mean "
            "before GMRES; select KFBIM_3D_NEUMANN_BORDER_SOLVER="
            "mean_free_pivot_elimination");
    }
    if (use_topology_affine) {
        using app3d::ConstraintBlock3D;
        using app3d::ConstraintSystem3D;
        using app3d::SparseMatrixCSR3D;

        const int value_base_size =
            value_transfer.density().c0_coefficient_count();
        const int normal_base_size =
            normal_transfer->density().c0_coefficient_count();
        if (coefficient_size != value_base_size
            || normal_transfer->coefficient_count() != normal_base_size) {
            throw std::runtime_error(
                "topology-affine transfers must expose their BaseOnly C0 "
                "coordinates without a legacy reduction");
        }
        SparseMatrixCSR3D base_identity(value_base_size, value_base_size);
        base_identity.setIdentity();
        base_identity.makeCompressed();
        SparseMatrixCSR3D normal_identity(
            normal_base_size, normal_base_size);
        normal_identity.setIdentity();
        normal_identity.makeCompressed();

        // The known normal data first enters its own smooth-sheet space.
        // Feature seams remain broken here, so no false equality is imposed
        // between densities defined with different face normals.
        const app3d::TopologyDensityConstraintPlan3D normal_plan =
            app3d::make_topology_density_constraint_plan_3d(
                normal_transfer->density());
        app3d::AffineReductionOptions3D normal_reduction_options;
        normal_reduction_options.schedule =
            app3d::AffineEliminationSchedule3D::StagedVertexThenEdge;
        const app3d::AffineReduction3D normal_reduction =
            app3d::affine_eliminate_local_svd_3d(
                normal_plan.system, normal_plan.blocks,
                normal_reduction_options);
        const app3d::TopologyTraceProjector3D normal_projector(
            normal_transfer->c0_design(), normal_identity,
            normal_reduction, normal_transfer->weights());
        const Eigen::VectorXd normal_mass_coordinates =
            normal_projector.project(prescribed_normal_jump);
        const Eigen::VectorXd normal_c0 =
            normal_projector.lift_full_c0(normal_mass_coordinates);
        result.edge_normal_fit_linf =
            (normal_transfer->evaluate_c0_samples(normal_c0)
             - prescribed_normal_jump).lpNorm<Eigen::Infinity>();

        const app3d::TopologyDensityConstraintPlan3D value_plan =
            app3d::make_topology_density_constraint_plan_3d(
                value_transfer.density());
        app3d::AffineReductionOptions3D reduction_options;
        reduction_options.consistency_tolerance = 2.0e-8;
        reduction_options.invariant_tolerance = 2.0e-8;
        reduction_options.schedule =
            app3d::AffineEliminationSchedule3D::StagedVertexThenEdge;
        // Fix the homogeneous value-topology space first.  The known g_N
        // feature target is then projected only onto the jet values reachable
        // from that space.  prescribed_normal_jump itself remains untouched
        // in the physical spread and base-field evaluation below.
        const app3d::AffineReduction3D value_topology_reduction =
            app3d::affine_eliminate_local_svd_3d(
                value_plan.system, value_plan.blocks, reduction_options);
        const app3d::TopologyFeatureJumpJetOperators3D feature =
            app3d::make_topology_feature_jump_jet_operators_3d(
                value_transfer.density(), normal_transfer->density(), {}, {},
                use_topology_ambient_gradient
                    ? app3d::TopologyNeumannFeatureC1Form3D::AmbientGradient
                    : app3d::TopologyNeumannFeatureC1Form3D::ConormalSolved);
        ConstraintSystem3D feature_system =
            feature.bind_normal_target(normal_c0);
        const app3d::ReachableConstraintTargetProjection3D
            feature_target_projection =
                app3d::project_constraint_target_to_reachable_space_3d(
                    value_topology_reduction, feature_system,
                    feature.blocks);
        feature_system = feature_target_projection.projected_system;

        const Eigen::Index topology_rows = value_plan.system.C.rows();
        const Eigen::Index feature_rows = feature_system.C.rows();
        ConstraintSystem3D combined;
        combined.C.resize(
            topology_rows + feature_rows, value_base_size);
        combined.d = Eigen::VectorXd::Zero(topology_rows + feature_rows);
        combined.d.head(topology_rows) = value_plan.system.d;
        combined.d.tail(feature_rows) = feature_system.d;
        combined.meta = value_plan.system.meta;
        combined.meta.reserve(static_cast<std::size_t>(
            topology_rows + feature_rows));
        combined.meta.insert(combined.meta.end(),
                             feature_system.meta.begin(),
                             feature_system.meta.end());
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<std::size_t>(
            value_plan.system.C.nonZeros()
            + feature_system.C.nonZeros()));
        for (int row = 0; row < value_plan.system.C.outerSize(); ++row) {
            for (SparseMatrixCSR3D::InnerIterator entry(
                     value_plan.system.C, row); entry; ++entry) {
                triplets.emplace_back(
                    entry.row(), entry.col(), entry.value());
            }
        }
        for (int row = 0; row < feature_system.C.outerSize(); ++row) {
            for (SparseMatrixCSR3D::InnerIterator entry(
                     feature_system.C, row); entry; ++entry) {
                triplets.emplace_back(topology_rows + entry.row(),
                                      entry.col(), entry.value());
            }
        }
        combined.C.setFromTriplets(triplets.begin(), triplets.end());
        combined.C.makeCompressed();
        combined.validate();

        std::vector<ConstraintBlock3D> blocks = value_plan.blocks;
        for (ConstraintBlock3D block : feature.blocks) {
            for (Eigen::Index& row : block.row_ids)
                row += topology_rows;
            blocks.push_back(std::move(block));
        }

        const app3d::AffineReduction3D reduction =
            app3d::affine_eliminate_local_svd_3d(
                std::move(combined), std::move(blocks), reduction_options);
        if (reduction.reduced_size() <= 1) {
            throw std::runtime_error(
                "topology Neumann space has no nontrivial coordinate after "
                "the global mean degree is removed");
        }

        // The topology constraints first produce c=c_p+Gz.  Enforce the
        // Neumann gauge in that coefficient space before constructing the
        // trace projector.  Using exactly the same panel weights and test
        // design as the projector makes the final K-1 test space orthogonal
        // to the constant trace in the discrete boundary L2 product.
        Eigen::VectorXd projected_mean_dual =
            value_transfer.c0_design().transpose()
            * value_transfer.weights();
        const double projected_area = value_transfer.weights().sum();
        if (!(projected_area > 0.0) || !std::isfinite(projected_area)) {
            throw std::runtime_error(
                "topology mean-free projection has invalid surface area");
        }
        projected_mean_dual /= projected_area;
        const app3d::TopologyMeanFreeReduction3D mean_free_reduction =
            app3d::eliminate_topology_mean_3d(
                reduction.particular_base(), reduction.homogeneous_base(),
                projected_mean_dual);
        if (mean_free_reduction.reduced_coordinate_count()
                != reduction.reduced_size() - 1) {
            throw std::runtime_error(
                "topology mean elimination did not remove exactly one "
                "coordinate");
        }
        const app3d::TopologyTraceProjector3D topology_projector(
            value_transfer.c0_design(), mean_free_reduction.particular(),
            mean_free_reduction.homogeneous(), value_transfer.weights());

        const Eigen::VectorXd particular_c0 =
            topology_projector.particular_c0();
        const Eigen::VectorXd particular_jump =
            value_transfer.evaluate_c0_samples(particular_c0);
        const HarmonicJetField3D base_field =
            evaluate_coefficient_crossing_field(
                pipeline, particular_jump, prescribed_normal_jump,
                use_feature_value_jet, anchor_center_cauchy,
                use_local_restrict_fit,
                &value_transfer, &particular_c0,
                nullptr, nullptr, known_normal_jet);
        const Eigen::VectorXd base_trace = restrict_exterior(
            base_field, particular_c0, particular_jump,
            prescribed_normal_jump);
        const Eigen::VectorXd constant_projected =
            topology_projector.project(Eigen::VectorXd::Ones(
                static_cast<Eigen::Index>(base_trace.size())));
        const Eigen::VectorXd intrinsic_mean_dual =
            value_transfer.density().c0_mass_row().transpose()
            / value_transfer.density().surface_area();
        const int reduced_size =
            topology_projector.reduced_coordinate_count();

        const auto apply_topology_homogeneous = [&] (
            const Eigen::VectorXd& mass_coordinates,
            Eigen::VectorXd& projected_trace) {
            const Eigen::VectorXd value_c0 =
                topology_projector.lift_homogeneous_c0(mass_coordinates);
            const Eigen::VectorXd value_jump =
                value_transfer.evaluate_c0_samples(value_c0);
            const HarmonicJetField3D field =
                evaluate_value_field(value_c0, value_jump);
            const Eigen::VectorXd trace = restrict_exterior(
                field, value_c0, value_jump, zero_normal);
            projected_trace = topology_projector.project(trace);
        };

        FunctionOperator3D topology_op(
            reduced_size,
            [&](const Eigen::VectorXd& mean_free_unknown,
                Eigen::VectorXd& image) {
                apply_topology_homogeneous(mean_free_unknown, image);
            });
        const Eigen::VectorXd topology_rhs =
            -topology_projector.project(base_trace);
        Eigen::VectorXd topology_unknown = Eigen::VectorXd::Zero(
            reduced_size);
        GMRES gmres(max_iterations, tolerance, restart);
        result.iterations = gmres.solve(
            topology_op, topology_rhs, topology_unknown);
        result.converged = gmres.converged();
        result.gmres_residuals = gmres.residuals();

        const Eigen::VectorXd final_c0 = topology_projector.lift_full_c0(
            topology_unknown);
        unknown.head(coefficient_size) =
            value_transfer.projection().native_to_trace_mass(final_c0);
        unknown[coefficient_size] = 0.0;
        Eigen::VectorXd topology_image;
        topology_op.apply(topology_unknown, topology_image);
        const Eigen::VectorXd topology_operator_residual =
            topology_image - topology_rhs;
        // Preserve the established residual schema for downstream reports:
        // the first K-1 entries are the actual square GMRES residual and the
        // final audit entry is the already-eliminated mean constraint.
        topology_augmented_residual.resize(reduced_size + 1);
        topology_augmented_residual.head(reduced_size) =
            topology_operator_residual;
        topology_augmented_residual[reduced_size] =
            projected_mean_dual.dot(final_c0);
        const Eigen::VectorXd final_jump =
            value_transfer.evaluate_c0_samples(final_c0);
        topology_final_c0 = final_c0;
        topology_final_jump = final_jump;
        const HarmonicJetField3D final_field =
            evaluate_coefficient_crossing_field(
                pipeline, final_jump, prescribed_normal_jump,
                use_feature_value_jet, anchor_center_cauchy,
                use_local_restrict_fit,
                &value_transfer, &final_c0,
                nullptr, nullptr, known_normal_jet);
        const Eigen::VectorXd final_trace = restrict_exterior(
            final_field, final_c0, final_jump, prescribed_normal_jump);
        const Eigen::VectorXd expected_topology_head =
            topology_projector.project(final_trace);
        result.topology_trace_closure_linf =
            (topology_augmented_residual.head(reduced_size)
             - expected_topology_head).lpNorm<Eigen::Infinity>();

        result.edge_feature_edges = feature.feature_edge_count();
        result.edge_constraint_rows =
            static_cast<int>(reduction.constraints().row_count());
        result.edge_constraint_rank =
            static_cast<int>(reduction.base_size()
                             - reduction.reduced_size());
        // Edge/topology diagnostics stop before the separate global mean
        // rank-one reduction; topology_reduced_coordinates below is the
        // actual final GMRES dimension.
        result.edge_reduced_dofs =
            static_cast<int>(reduction.reduced_size());
        result.edge_constraint_residual_linf =
            (reduction.constraints().C * final_c0
             - reduction.constraints().d).lpNorm<Eigen::Infinity>();
        result.edge_nullspace_residual_linf =
            reduction.homogeneous_residual_linf();
        result.edge_target_projection_linf =
            feature_target_projection.projection_linf;
        result.feature_target_requested_linf =
            feature_target_projection.requested_linf;
        result.feature_target_requested_l2 =
            feature_target_projection.requested_l2;
        result.feature_target_projected_linf =
            feature_target_projection.projected_linf;
        result.feature_target_projected_l2 =
            feature_target_projection.projected_l2;
        result.feature_target_projection_l2 =
            feature_target_projection.projection_l2;
        result.feature_target_projection_relative_l2 =
            feature_target_projection.relative_projection_l2;
        result.feature_target_reachable_certification_linf =
            feature_target_projection.reachable_certification_linf;
        result.feature_target_projection_components = static_cast<int>(
            feature_target_projection.records.size());
        result.feature_target_constraints_exact =
            feature_target_projection.constraints_exact;
        result.topology_constraint_rows =
            static_cast<int>(reduction.constraints().row_count());
        result.topology_constraint_rank = result.edge_constraint_rank;
        result.topology_base_coordinates =
            static_cast<int>(reduction.base_size());
        result.topology_full_coordinates =
            value_transfer.density().raw_coefficient_count();
        result.topology_pre_mean_coordinates =
            static_cast<int>(reduction.reduced_size());
        result.topology_reduced_coordinates = reduced_size;
        result.topology_block_count =
            static_cast<int>(reduction.records().size());
        result.particular_solver =
            "component_local_target_projection+sequential_local_scaled_svd"
            "+global_mean_pivot";
        result.topology_block_records = reduction.records();
        result.topology_particular_residual_linf =
            reduction.particular_residual_linf();
        result.topology_homogeneous_residual_linf =
            reduction.homogeneous_residual_linf();
        result.topology_projector_rank = topology_projector.rank();
        result.topology_projector_condition = topology_projector.condition();
        result.topology_projector_pb_error =
            topology_projector.pb_max_error();
        result.topology_trace_samples = topology_projector.sample_count();
        result.topology_trace_oversampling_margin =
            result.topology_trace_samples - reduced_size;
        result.topology_trace_oversampling_ratio =
            static_cast<double>(result.topology_trace_samples)
            / static_cast<double>(reduced_size);
        const app3d::TopologyMeanFreeReductionDiagnostics3D& mean_diag =
            mean_free_reduction.diagnostics();
        result.mean_free_pivot_index =
            static_cast<int>(mean_diag.pivot_coordinate);
        result.mean_free_pivot_moment = mean_diag.pivot_moment;
        result.mean_free_relative_observability =
            mean_diag.relative_observability;
        result.mean_free_coordinate_condition =
            mean_diag.coordinate_map_condition;
        result.mean_free_particular_residual =
            mean_diag.particular_mean_residual;
        result.mean_free_homogeneous_residual =
            mean_diag.homogeneous_mean_residual;
        result.mean_free_constant_projection_linf =
            constant_projected.lpNorm<Eigen::Infinity>();
        result.mean_free_intrinsic_mean =
            intrinsic_mean_dual.dot(final_c0);
    } else if (use_edge_jump_jet) {
        const app3d::NativeFeatureEdgeJumpJetConstraints3D edge_constraints(
            value_transfer.density(), normal_transfer->density());
        if (edge_constraints.constraint_count() <= 0) {
            throw std::runtime_error(
                "strong feature-edge jump-jet mode found no feature-edge "
                "constraints");
        }

        const app3d::ReducedTraceProjection3D& active_projection =
            use_gauss_restrict
            ? value_transfer.gauss_projection()
            : value_transfer.projection();
        Eigen::MatrixXd edge_matrix(
            edge_constraints.constraint_count(), coefficient_size);
        for (int row = 0; row < edge_constraints.constraint_count(); ++row) {
            edge_matrix.row(row) =
                active_projection.native_dual_to_trace_mass(
                    edge_constraints.value_matrix().row(row).transpose())
                    .transpose();
        }
        const Eigen::VectorXd normal_native =
            normal_transfer->fit_native(prescribed_normal_jump);
        result.edge_normal_fit_linf = (
            normal_transfer->evaluate_native(normal_native)
            - prescribed_normal_jump).lpNorm<Eigen::Infinity>();
        const Eigen::VectorXd unit_normal_native =
            normal_transfer->fit_native(
                Eigen::VectorXd::Ones(sample_size));
        const Eigen::VectorXd raw_edge_data_target =
            edge_constraints.normal_target(normal_native);
        Eigen::VectorXd raw_edge_lambda_target = Eigen::VectorXd::Zero(
            edge_constraints.constraint_count());
        if (compatibility_mode
            == NeumannCompatibilityMode3D::OperatorFluxCorrection) {
            raw_edge_lambda_target =
                -edge_constraints.normal_target(unit_normal_native);
        }
        const Eigen::VectorXd native_mean_row =
            value_transfer.density().mass_row().transpose()
            / value_transfer.density().surface_area();
        const Eigen::VectorXd coordinate_mean_row =
            active_projection.native_dual_to_trace_mass(native_mean_row);

        // The broken normal-density fit can make algebraically dependent
        // mortar equations disagree by its approximation error, especially
        // where several L-prism patches represent one physical edge.  Project
        // both affine targets onto range(D restricted to the intrinsic
        // mean-zero space).  This retains every independent edge constraint
        // and distributes only the unavoidable redundant-data defect.
        Eigen::MatrixXd raw_targets(
            edge_constraints.constraint_count(), 2);
        raw_targets.col(0) = raw_edge_data_target;
        raw_targets.col(1) = raw_edge_lambda_target;
        const app3d::EdgeJumpJetTargetProjection3D target_projection =
            app3d::project_edge_jump_jet_targets_to_mean_zero_range_3d(
                edge_matrix, raw_targets, coordinate_mean_row);
        if (target_projection.mean_null_residual_linf > 1.0e-8
            || target_projection.mean_null_identity_residual > 1.0e-8
            || target_projection.lift_mean_residual_linf > 1.0e-8) {
            throw std::runtime_error(
                "feature-edge compatible-target projection failed its "
                "mean-null audit");
        }
        result.edge_target_projection_linf =
            target_projection.normalized_projection_linf;
        const Eigen::VectorXd edge_data_target =
            target_projection.targets.col(0);
        const Eigen::VectorXd edge_lambda_target =
            target_projection.targets.col(1);
        const app3d::EdgeJumpJetAffineReduction3D reduction(
            edge_matrix, edge_data_target, edge_lambda_target,
            coordinate_mean_row, 0.0);

        Eigen::VectorXd applied_data_lift;
        apply_trace_block(reduction.data_lift(), applied_data_lift);
        Eigen::VectorXd applied_lambda_lift;
        apply_trace_block(reduction.lambda_lift(), applied_lambda_lift);
        const Eigen::VectorXd effective_border =
            reduction.effective_border(
                border_coefficients, applied_lambda_lift);
        const Eigen::VectorXd shifted_rhs =
            rhs.head(coefficient_size) - applied_data_lift;
        const Eigen::VectorXd reduced_rhs =
            reduction.project_and_test(shifted_rhs, effective_border);
        FunctionOperator3D reduced_op(
            reduction.reduced_size(),
            [&](const Eigen::VectorXd& reduced,
                Eigen::VectorXd& tested) {
                Eigen::VectorXd image;
                apply_trace_block(
                    reduction.homogeneous_coordinates(reduced), image);
                tested = reduction.project_and_test(
                    image, effective_border);
            });
        Eigen::VectorXd reduced = Eigen::VectorXd::Zero(
            reduction.reduced_size());
        GMRES gmres(max_iterations, tolerance, restart);
        result.iterations = gmres.solve(
            reduced_op, reduced_rhs, reduced);
        result.converged = gmres.converged();
        result.gmres_residuals = gmres.residuals();

        const Eigen::VectorXd homogeneous =
            reduction.homogeneous_coordinates(reduced);
        Eigen::VectorXd applied_homogeneous;
        apply_trace_block(homogeneous, applied_homogeneous);
        unknown[coefficient_size] = reduction.recover_lambda(
            shifted_rhs - applied_homogeneous, effective_border);
        unknown.head(coefficient_size) = reduction.expand(
            reduced, unknown[coefficient_size]);

        result.edge_feature_edges = edge_constraints.feature_edge_count();
        result.edge_constraint_rows = edge_constraints.constraint_count();
        result.edge_constraint_rank = reduction.constraint_rank();
        result.edge_reduced_dofs = reduction.reduced_size();
        result.edge_nullspace_residual_linf =
            reduction.nullspace_constraint_residual();
        result.edge_constraint_residual_linf = (
            edge_matrix * unknown.head(coefficient_size)
            - edge_data_target
            - unknown[coefficient_size] * edge_lambda_target)
                .lpNorm<Eigen::Infinity>();
    } else if (use_householder_border) {
        if (!use_trace_mass_coordinates) {
            throw std::invalid_argument(
                "Householder Neumann border elimination requires "
                "trace_mass density coordinates");
        }
        if (coefficient_size < 2)
            throw std::runtime_error(
                "Householder Neumann elimination has too few coefficients");
        const double border_norm = border_coefficients.norm();
        const Eigen::VectorXd trace_ones = Eigen::VectorXd::Ones(
            static_cast<Eigen::Index>(border_samples.size()));
        const Eigen::VectorXd compatibility_row =
            project_trace(trace_ones) / value_transfer.density().surface_area();
        const double compatibility_norm = compatibility_row.norm();
        if (!(border_norm > 1.0e-14)
            || !(compatibility_norm > 1.0e-14)
            || !std::isfinite(border_norm)
            || !std::isfinite(compatibility_norm)) {
            throw std::runtime_error(
                "Householder Neumann border data are singular");
        }
        const Eigen::VectorXd border_unit =
            border_coefficients / border_norm;
        const Eigen::VectorXd compatibility_unit =
            compatibility_row / compatibility_norm;
        const auto reflector = [&](const Eigen::VectorXd& unit) {
            Eigen::VectorXd direction =
                Eigen::VectorXd::Zero(coefficient_size);
            direction[coefficient_size - 1] = 1.0;
            direction -= unit;
            const double norm = direction.norm();
            if (norm > 32.0 * std::numeric_limits<double>::epsilon())
                direction /= norm;
            else
                direction.setZero();
            return direction;
        };
        const Eigen::VectorXd domain_reflector =
            reflector(compatibility_unit);
        const Eigen::VectorXd range_reflector =
            reflector(border_unit);
        const auto apply_reflector = [](
            const Eigen::VectorXd& direction,
            Eigen::VectorXd values) {
            if (!direction.isZero(0.0))
                values.noalias() -=
                    2.0 * direction.dot(values) * direction;
            return values;
        };
        FunctionOperator3D reduced_op(
            coefficient_size - 1,
            [&](const Eigen::VectorXd& reduced,
                Eigen::VectorXd& tested) {
                Eigen::VectorXd embedded =
                    Eigen::VectorXd::Zero(coefficient_size);
                embedded.head(coefficient_size - 1) = reduced;
                const Eigen::VectorXd constrained =
                    apply_reflector(domain_reflector, std::move(embedded));
                Eigen::VectorXd image;
                apply_trace_block(constrained, image);
                image = apply_reflector(
                    range_reflector, std::move(image));
                tested = image.head(coefficient_size - 1);
            });
        const Eigen::VectorXd tested_rhs = apply_reflector(
            range_reflector, rhs.head(coefficient_size));
        const Eigen::VectorXd reduced_rhs =
            tested_rhs.head(coefficient_size - 1);
        Eigen::VectorXd reduced =
            Eigen::VectorXd::Zero(coefficient_size - 1);
        GMRES gmres(max_iterations, tolerance, restart);
        result.iterations =
            gmres.solve(reduced_op, reduced_rhs, reduced);
        result.converged = gmres.converged();
        result.gmres_residuals = gmres.residuals();
        Eigen::VectorXd embedded =
            Eigen::VectorXd::Zero(coefficient_size);
        embedded.head(coefficient_size - 1) = reduced;
        unknown.head(coefficient_size) = apply_reflector(
            domain_reflector, std::move(embedded));
        Eigen::VectorXd trace_image;
        apply_trace_block(
            unknown.head(coefficient_size), trace_image);
        unknown[coefficient_size] = border_coefficients.dot(
            rhs.head(coefficient_size) - trace_image)
            / border_coefficients.squaredNorm();
    } else if (use_projected_border) {
        if (!use_trace_mass_coordinates) {
            throw std::invalid_argument(
                "projected Neumann border elimination requires "
                "trace_mass density coordinates");
        }
        const double border_norm = border_coefficients.norm();
        if (!(border_norm > 1.0e-14)
            || !std::isfinite(border_norm)) {
            throw std::runtime_error(
                "projected Neumann border column is singular");
        }
        const Eigen::VectorXd trace_ones = Eigen::VectorXd::Ones(
            static_cast<Eigen::Index>(border_samples.size()));
        const Eigen::VectorXd compatibility_row =
            project_trace(trace_ones) / value_transfer.density().surface_area();
        const double compatibility_norm = compatibility_row.norm();
        if (!(compatibility_norm > 1.0e-14)
            || !std::isfinite(compatibility_norm)) {
            throw std::runtime_error(
                "projected Neumann compatibility row is singular");
        }
        const Eigen::VectorXd border_unit =
            border_coefficients / border_norm;
        const Eigen::VectorXd compatibility_unit =
            compatibility_row / compatibility_norm;
        FunctionOperator3D projected_op(
            coefficient_size,
            [&](const Eigen::VectorXd& z, Eigen::VectorXd& projected) {
                const double constrained_component =
                    compatibility_unit.dot(z);
                const Eigen::VectorXd constrained =
                    z - constrained_component * compatibility_unit;
                Eigen::VectorXd image;
                apply_trace_block(constrained, image);
                projected = image
                    - border_unit * border_unit.dot(image)
                    + constrained_component * border_unit;
            });
        const Eigen::VectorXd projected_rhs =
            rhs.head(coefficient_size)
            - border_unit
                  * border_unit.dot(rhs.head(coefficient_size));
        Eigen::VectorXd z = Eigen::VectorXd::Zero(coefficient_size);
        GMRES gmres(max_iterations, tolerance, restart);
        result.iterations = gmres.solve(projected_op, projected_rhs, z);
        result.converged = gmres.converged();
        result.gmres_residuals = gmres.residuals();
        unknown.head(coefficient_size) =
            z - compatibility_unit.dot(z) * compatibility_unit;
        Eigen::VectorXd trace_image;
        apply_trace_block(
            unknown.head(coefficient_size), trace_image);
        unknown[coefficient_size] = border_coefficients.dot(
            rhs.head(coefficient_size) - trace_image)
            / border_coefficients.squaredNorm();
    } else {
        GMRES gmres(max_iterations, tolerance, restart);
        result.iterations = gmres.solve(*op, rhs, unknown);
        result.converged = gmres.converged();
        result.gmres_residuals = gmres.residuals();
    }
    result.compatibility_mode = compatibility_mode;
    result.orthonormal_value_coefficients =
        unknown.head(coefficient_size);
    result.value_jump = use_topology_affine
        ? topology_final_jump
        : evaluate_coordinates(result.orthonormal_value_coefficients);
    result.normal_jump = prescribed_normal_jump;
    if (compatibility_mode
        == NeumannCompatibilityMode3D::OperatorFluxCorrection) {
        result.operator_compatibility_correction =
            unknown[coefficient_size];
        result.normal_jump.array() -=
            result.operator_compatibility_correction;
    } else if (use_topology_affine) {
        result.lagrange_multiplier =
            std::numeric_limits<double>::quiet_NaN();
    } else {
        result.lagrange_multiplier = unknown[coefficient_size];
    }
    result.border_column_linf = use_topology_affine
        ? std::numeric_limits<double>::quiet_NaN()
        : border_samples.lpNorm<Eigen::Infinity>();
    result.value_c0_coefficients = use_topology_affine
        ? topology_final_c0
        : expand_c0_coordinates(result.orthonormal_value_coefficients);
    HarmonicJetField3D field;
    if (use_gauss_restrict) {
        const HarmonicJetField3D value_field = evaluate_value_field(
            result.value_c0_coefficients, result.value_jump);
        const HarmonicJetField3D normal_field =
            evaluate_coefficient_crossing_field(
                pipeline, zero_value, result.normal_jump,
                use_feature_value_jet, anchor_center_cauchy,
                use_local_restrict_fit);
        field.potential = value_field.potential + normal_field.potential;
        field.coefficients = normal_field.coefficients;
        field.restrict_value_jets = value_field.restrict_value_jets;
        field.restrict_normal_coefficients =
            normal_field.restrict_normal_coefficients;
    } else {
        field = evaluate_coefficient_crossing_field(
            pipeline, result.value_jump, result.normal_jump,
            use_feature_value_jet, anchor_center_cauchy,
            use_local_restrict_fit,
            &value_transfer, &result.value_c0_coefficients,
            nullptr, nullptr, known_normal_jet);
    }
    result.potential = field.potential;
    result.cauchy_coefficients = field.coefficients;
    result.restrict_value_jets = field.restrict_value_jets;
    result.restrict_normal_coefficients =
        field.restrict_normal_coefficients;
    if (use_topology_affine) {
        result.augmented_residual = topology_augmented_residual;
    } else {
        Eigen::VectorXd applied;
        op->apply(unknown, applied);
        result.augmented_residual = applied - rhs;
    }
    return result;
}

struct CoefficientExteriorNormalTraceSolution3D {
    Eigen::VectorXd orthonormal_normal_coefficients;
    Eigen::VectorXd normal_c0_coefficients;
    Eigen::VectorXd normal_jump;
    Eigen::VectorXd potential;
    Eigen::MatrixXd cauchy_coefficients;
    Eigen::MatrixXd restrict_value_jets;
    Eigen::MatrixXd restrict_normal_coefficients;
    Eigen::VectorXd operator_residual;
    std::vector<double> gmres_residuals;
    std::string reduction_scheme = "legacy_native";
    std::string particular_solver = "not_applicable";
    std::string known_value_jump = "sample_fit";
    std::string jump_space = "legacy_sample_fit";
    std::string feature_constraint_mode = "broken_sheets";
    int feature_edge_count = 0;
    int feature_constraint_rows = 0;
    int feature_constraint_rank = 0;
    double feature_constraint_residual_linf = 0.0;
    double feature_vertex_residual_linf = 0.0;
    double feature_edge_residual_linf = 0.0;
    double feature_discarded_residual_linf = 0.0;
    double trace_projection_leakage_relative_l2 = 0.0;
    int topology_constraint_rows = 0;
    int topology_constraint_rank = 0;
    // c_full = A0 c_base and c_base = p + E z.
    int topology_base_coordinates = 0;
    int topology_full_coordinates = 0;
    int topology_reduced_coordinates = 0;
    int topology_block_count = 0;
    double topology_particular_residual_linf = 0.0;
    double topology_homogeneous_residual_linf = 0.0;
    int topology_projector_rank = 0;
    double topology_projector_condition = 0.0;
    double topology_projector_pb_error = 0.0;
    int topology_trace_samples = 0;
    int topology_trace_oversampling_margin = 0;
    double topology_trace_oversampling_ratio = 0.0;
    double topology_trace_closure_linf = 0.0;
    std::vector<app3d::AffineEliminationRecord3D> topology_block_records;
    int iterations = 0;
    bool converged = false;
};

CoefficientExteriorNormalTraceSolution3D
solve_exterior_zero_normal_dirichlet_coefficients_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const NativeDensityTransfer3D& normal_transfer,
    const Eigen::VectorXd& prescribed_value_jump,
    const app3d::KnownDirichletValueGradientHessianCallback3D&
        known_dirichlet,
    ExteriorNormalRestrictMode3D mode,
    double tolerance,
    int restart,
    int max_iterations)
{
    const int sample_size = pipeline.surface_size();
    const int coefficient_size = normal_transfer.coefficient_count();
    if (normal_transfer.sample_count() != sample_size
        || prescribed_value_jump.size() != sample_size) {
        throw std::invalid_argument(
            "coefficient Dirichlet solve received incompatible sample sizes");
    }
    const Eigen::VectorXd zero_normal = Eigen::VectorXd::Zero(sample_size);
    const bool use_topology_affine =
        selected_dirichlet_jump_space_mode()
        == DirichletJumpSpaceMode3D::AnalyticJ0AffineJ1;
    if (use_topology_affine) {
        using app3d::ConstraintBlock3D;
        using app3d::ConstraintSystem3D;
        using app3d::SparseMatrixCSR3D;

        if (!known_dirichlet) {
            throw std::invalid_argument(
                "topology-affine Dirichlet requires analytic g_D value, "
                "gradient, and Hessian data");
        }
        if (!pipeline.has_direct_coefficient_normal_density(
                normal_transfer.density())) {
            throw std::invalid_argument(
                "topology-affine Dirichlet requires direct normal-density "
                "coefficient Cauchy rows");
        }
        if (!trace_restrict_uses_all_event_path(mode)) {
            throw std::invalid_argument(
                "topology-affine analytic Dirichlet requires an all-event "
                "normal-trace route");
        }
        const app3d::KnownDirichletJetCallback3D known_value_jet =
            [known_dirichlet](
                int patch,
                double u,
                double v,
                const app3d::NativeSurfaceParameterJet3D& geometry,
                const app3d::LocalOrthonormalFrame3D& frame) {
                return app3d::evaluate_known_dirichlet_jet_3d(
                    known_dirichlet, patch, u, v, geometry, frame);
            };
        const int base_size =
            normal_transfer.density().c0_coefficient_count();
        SparseMatrixCSR3D identity(base_size, base_size);
        identity.setIdentity();
        identity.makeCompressed();
        const app3d::TopologyDensityConstraintPlan3D topology_plan =
            app3d::make_topology_density_constraint_plan_3d(
                normal_transfer.density());
        app3d::TopologyDirichletFeatureConstraintOptions3D feature_options;
        const DirichletFeatureCouplingMode3D feature_coupling_mode =
            selected_dirichlet_feature_coupling_mode();
        feature_options.enabled = feature_coupling_mode
            == DirichletFeatureCouplingMode3D::
                   AmbientGradientAffineMortar;
        const app3d::TopologyDirichletFeatureConstraintPlan3D
            feature_candidates =
            app3d::make_topology_dirichlet_feature_constraint_plan_3d(
                normal_transfer.density(), known_dirichlet,
                feature_options);

        app3d::AffineReductionOptions3D reduction_options;
        reduction_options.schedule =
            app3d::AffineEliminationSchedule3D::StagedVertexThenEdge;
        app3d::TopologyDirichletUnisolventConstraintPlan3D feature_plan;
        Eigen::Index topology_only_reduced_size = base_size;
        if (feature_candidates.constraint_count() > 0) {
            const app3d::AffineReduction3D topology_only_reduction =
                app3d::affine_eliminate_local_svd_3d(
                    topology_plan.system, topology_plan.blocks,
                    reduction_options);
            topology_only_reduced_size =
                topology_only_reduction.reduced_size();
            feature_plan =
                app3d::select_topology_dirichlet_unisolvent_constraints_3d(
                    feature_candidates,
                    topology_only_reduction.homogeneous_base(),
                    reduction_options.relative_rank_tolerance);
        } else {
            feature_plan.system = feature_candidates.system;
            feature_plan.blocks = feature_candidates.blocks;
        }

        const Eigen::Index topology_rows = topology_plan.system.C.rows();
        const Eigen::Index feature_rows = feature_plan.system.C.rows();
        ConstraintSystem3D combined;
        combined.C.resize(topology_rows + feature_rows, base_size);
        combined.d = Eigen::VectorXd::Zero(topology_rows + feature_rows);
        combined.d.head(topology_rows) = topology_plan.system.d;
        combined.d.tail(feature_rows) = feature_plan.system.d;
        combined.meta = topology_plan.system.meta;
        combined.meta.reserve(static_cast<std::size_t>(
            topology_rows + feature_rows));
        combined.meta.insert(combined.meta.end(),
                             feature_plan.system.meta.begin(),
                             feature_plan.system.meta.end());
        std::vector<Eigen::Triplet<double>> combined_triplets;
        combined_triplets.reserve(static_cast<std::size_t>(
            topology_plan.system.C.nonZeros()
            + feature_plan.system.C.nonZeros()));
        for (int row = 0; row < topology_plan.system.C.outerSize(); ++row) {
            for (SparseMatrixCSR3D::InnerIterator entry(
                     topology_plan.system.C, row); entry; ++entry) {
                combined_triplets.emplace_back(
                    entry.row(), entry.col(), entry.value());
            }
        }
        for (int row = 0; row < feature_plan.system.C.outerSize(); ++row) {
            for (SparseMatrixCSR3D::InnerIterator entry(
                     feature_plan.system.C, row); entry; ++entry) {
                combined_triplets.emplace_back(
                    topology_rows + entry.row(), entry.col(), entry.value());
            }
        }
        combined.C.setFromTriplets(
            combined_triplets.begin(), combined_triplets.end());
        combined.C.makeCompressed();
        combined.validate();

        std::vector<ConstraintBlock3D> blocks = topology_plan.blocks;
        for (ConstraintBlock3D block : feature_plan.blocks) {
            for (Eigen::Index& row : block.row_ids)
                row += topology_rows;
            blocks.push_back(std::move(block));
        }
        const app3d::AffineReduction3D reduction =
            app3d::affine_eliminate_local_svd_3d(
                std::move(combined), std::move(blocks), reduction_options);
        int feature_constraint_rank = 0;
        if (feature_rows > 0) {
            feature_constraint_rank = static_cast<int>(
                topology_only_reduced_size - reduction.reduced_size());
            if (feature_constraint_rank != feature_rows) {
                throw std::runtime_error(
                    "unisolvent Dirichlet feature rows did not reduce the "
                    "topology coordinate space one-for-one");
            }
        }
        const app3d::TopologyTraceProjector3D projector(
            normal_transfer.c0_design(), identity, reduction,
            normal_transfer.weights());
        const int reduced_size = projector.reduced_coordinate_count();
        FunctionOperator3D topology_op(
            reduced_size,
            [&](const Eigen::VectorXd& mass_coordinates,
                Eigen::VectorXd& projected) {
                const Eigen::VectorXd normal_c0 =
                    projector.lift_homogeneous_c0(mass_coordinates);
                const Eigen::VectorXd normal_jump =
                    normal_transfer.evaluate_c0_samples(normal_c0);
                const HarmonicJetField3D field =
                    evaluate_coefficient_crossing_field(
                        pipeline, zero_normal, normal_jump, true, false,
                        false, nullptr, nullptr,
                        &normal_transfer, &normal_c0);
                const Eigen::VectorXd full_trace =
                    pipeline.exterior_normal_trace(
                        field, zero_normal, normal_jump, mode);
                projected = projector.project(full_trace);
            });
        const Eigen::VectorXd particular_c0 = projector.particular_c0();
        const Eigen::VectorXd particular_normal =
            normal_transfer.evaluate_c0_samples(particular_c0);
        const HarmonicJetField3D data_field =
            evaluate_coefficient_crossing_field(
                pipeline, prescribed_value_jump, particular_normal,
                true, false, false,
                nullptr, nullptr, &normal_transfer, &particular_c0,
                {}, known_value_jet);
        const Eigen::VectorXd data_trace =
            pipeline.exterior_normal_trace(
                data_field, prescribed_value_jump, particular_normal, mode);
        const Eigen::VectorXd rhs = -projector.project(data_trace);
        Eigen::VectorXd mass_coordinates = Eigen::VectorXd::Zero(
            reduced_size);
        GMRES gmres(max_iterations, tolerance, restart);
        CoefficientExteriorNormalTraceSolution3D result;
        result.iterations = gmres.solve(
            topology_op, rhs, mass_coordinates);
        result.converged = gmres.converged();
        result.gmres_residuals = gmres.residuals();
        const Eigen::VectorXd final_c0 =
            projector.lift_full_c0(mass_coordinates);
        result.normal_c0_coefficients = final_c0;
        result.normal_jump =
            normal_transfer.evaluate_c0_samples(final_c0);
        const HarmonicJetField3D field =
            evaluate_coefficient_crossing_field(
                pipeline, prescribed_value_jump, result.normal_jump,
                true, false, false,
                nullptr, nullptr, &normal_transfer, &final_c0,
                {}, known_value_jet);
        result.potential = field.potential;
        result.cauchy_coefficients = field.coefficients;
        result.restrict_value_jets = field.restrict_value_jets;
        result.restrict_normal_coefficients =
            field.restrict_normal_coefficients;
        Eigen::VectorXd applied;
        topology_op.apply(mass_coordinates, applied);
        result.operator_residual = applied - rhs;
        const Eigen::VectorXd final_trace =
            pipeline.exterior_normal_trace(
                field, prescribed_value_jump, result.normal_jump, mode);
        const Eigen::VectorXd expected_projected_residual =
            projector.project(final_trace);
        result.topology_trace_closure_linf =
            (result.operator_residual - expected_projected_residual)
                .lpNorm<Eigen::Infinity>();
        result.known_value_jump = "analytic_boundary_jet";
        result.jump_space = "analytic_j0_affine_j1";
        result.feature_constraint_mode =
            dirichlet_feature_coupling_mode_name(feature_coupling_mode);
        result.feature_edge_count = feature_candidates.feature_edge_count();
        result.feature_constraint_rows =
            feature_candidates.constraint_count();
        result.feature_constraint_rank = feature_constraint_rank;
        const Eigen::VectorXd feature_residual =
            feature_plan.system.C * final_c0 - feature_plan.system.d;
        result.feature_constraint_residual_linf =
            feature_residual.size() == 0
                ? 0.0 : feature_residual.lpNorm<Eigen::Infinity>();
        for (const ConstraintBlock3D& block : feature_plan.blocks) {
            double block_residual = 0.0;
            for (Eigen::Index row : block.row_ids) {
                block_residual = std::max(
                    block_residual,
                    std::abs(feature_residual[row]));
            }
            if (block.kind == app3d::ConstraintBlockKind3D::Vertex) {
                result.feature_vertex_residual_linf = std::max(
                    result.feature_vertex_residual_linf, block_residual);
            } else {
                result.feature_edge_residual_linf = std::max(
                    result.feature_edge_residual_linf, block_residual);
            }
        }
        if (feature_candidates.constraint_count() > 0) {
            const Eigen::VectorXd candidate_residual =
                feature_candidates.system.C * final_c0
                - feature_candidates.system.d;
            std::vector<bool> retained(
                static_cast<std::size_t>(candidate_residual.size()), false);
            for (Eigen::Index row :
                 feature_plan.retained_candidate_rows) {
                retained[static_cast<std::size_t>(row)] = true;
            }
            for (Eigen::Index row = 0; row < candidate_residual.size(); ++row) {
                if (!retained[static_cast<std::size_t>(row)]) {
                    result.feature_discarded_residual_linf = std::max(
                        result.feature_discarded_residual_linf,
                        std::abs(candidate_residual[row]));
                }
            }
        }
        const Eigen::VectorXd projected_trace_c0 =
            projector.lift_homogeneous_c0(expected_projected_residual);
        const Eigen::VectorXd represented_trace =
            normal_transfer.c0_design() * projected_trace_c0;
        const Eigen::VectorXd leakage = final_trace - represented_trace;
        const double full_trace_weighted_l2 = std::sqrt(
            (normal_transfer.weights().array()
             * final_trace.array().square()).sum());
        const double leakage_weighted_l2 = std::sqrt(
            (normal_transfer.weights().array()
             * leakage.array().square()).sum());
        result.trace_projection_leakage_relative_l2 =
            leakage_weighted_l2
            / std::max(full_trace_weighted_l2,
                       std::numeric_limits<double>::epsilon());
        result.reduction_scheme = "topology_affine_local_svd";
        result.topology_constraint_rows =
            static_cast<int>(reduction.constraints().row_count());
        result.topology_constraint_rank =
            static_cast<int>(reduction.base_size()
                             - reduction.reduced_size());
        result.topology_base_coordinates =
            static_cast<int>(reduction.base_size());
        result.topology_full_coordinates =
            normal_transfer.density().raw_coefficient_count();
        result.topology_reduced_coordinates = reduced_size;
        result.topology_block_count =
            static_cast<int>(reduction.records().size());
        result.particular_solver = "sequential_local_scaled_svd";
        result.topology_block_records = reduction.records();
        result.topology_particular_residual_linf =
            reduction.particular_residual_linf();
        result.topology_homogeneous_residual_linf =
            reduction.homogeneous_residual_linf();
        result.topology_projector_rank = projector.rank();
        result.topology_projector_condition = projector.condition();
        result.topology_projector_pb_error = projector.pb_max_error();
        result.topology_trace_samples = projector.sample_count();
        result.topology_trace_oversampling_margin =
            result.topology_trace_samples - reduced_size;
        result.topology_trace_oversampling_ratio =
            static_cast<double>(result.topology_trace_samples)
            / static_cast<double>(reduced_size);
        return result;
    }
    FunctionOperator3D op(
        coefficient_size,
        [&](const Eigen::VectorXd& normal_coefficients,
            Eigen::VectorXd& result) {
            const Eigen::VectorXd normal_c0 =
                normal_transfer.expand_c0_orthonormal(
                    normal_coefficients);
            const Eigen::VectorXd normal_jump =
                normal_transfer.evaluate_c0_samples(normal_c0);
            const HarmonicJetField3D field =
                evaluate_coefficient_crossing_field(
                    pipeline, zero_normal, normal_jump, true, false,
                    false,
                    nullptr, nullptr, &normal_transfer, &normal_c0);
            const Eigen::VectorXd trace =
                pipeline.exterior_normal_trace(
                    field, zero_normal, normal_jump, mode);
            result = normal_transfer.fit_orthonormal(trace);
        });

    const HarmonicJetField3D data_field =
        evaluate_coefficient_crossing_field(
            pipeline, prescribed_value_jump, zero_normal, true, false);
    const Eigen::VectorXd data_trace = pipeline.exterior_normal_trace(
        data_field, prescribed_value_jump, zero_normal, mode);
    const Eigen::VectorXd rhs =
        -normal_transfer.fit_orthonormal(data_trace);
    Eigen::VectorXd unknown = Eigen::VectorXd::Zero(coefficient_size);
    GMRES gmres(max_iterations, tolerance, restart);
    CoefficientExteriorNormalTraceSolution3D result;
    result.iterations = gmres.solve(op, rhs, unknown);
    result.converged = gmres.converged();
    result.gmres_residuals = gmres.residuals();
    result.orthonormal_normal_coefficients = unknown;
    result.normal_jump = normal_transfer.evaluate_orthonormal(unknown);
    const Eigen::VectorXd result_normal_c0 =
        normal_transfer.expand_c0_orthonormal(
            result.orthonormal_normal_coefficients);
    result.normal_c0_coefficients = result_normal_c0;
    const HarmonicJetField3D field =
        evaluate_coefficient_crossing_field(
            pipeline, prescribed_value_jump, result.normal_jump,
            true, false,
            false,
            nullptr, nullptr, &normal_transfer, &result_normal_c0);
    result.potential = field.potential;
    result.cauchy_coefficients = field.coefficients;
    result.restrict_value_jets = field.restrict_value_jets;
    result.restrict_normal_coefficients = field.restrict_normal_coefficients;
    Eigen::VectorXd applied;
    op.apply(unknown, applied);
    result.operator_residual = applied - rhs;
    return result;
}

double vector_linf(const Eigen::VectorXd& values)
{
    return values.size() == 0 ? 0.0 : values.lpNorm<Eigen::Infinity>();
}

double vector_rms(const Eigen::VectorXd& values)
{
    return values.size() == 0
        ? 0.0
        : std::sqrt(values.squaredNorm() / static_cast<double>(values.size()));
}

struct ConditionStatistics3D {
    double median = 0.0;
    double p95 = 0.0;
    double maximum = 0.0;
};

ConditionStatistics3D summarize_conditions(std::vector<double> values)
{
    if (values.empty())
        throw std::invalid_argument("Cauchy condition list is empty");
    if (!std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value) && value > 0.0;
        })) {
        throw std::runtime_error("Cauchy condition list contains invalid values");
    }
    std::sort(values.begin(), values.end());
    ConditionStatistics3D result;
    result.median = values[(values.size() - 1) / 2];
    const std::size_t p95_index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(
            std::ceil(0.95 * static_cast<double>(values.size()))) - 1);
    result.p95 = values[p95_index];
    result.maximum = values.back();
    return result;
}

double surface_weighted_mean(const SurfaceDofCloud& surface,
                             const Eigen::VectorXd& values)
{
    if (values.size() != static_cast<int>(surface.dofs.size()))
        throw std::invalid_argument("surface weighted mean received wrong size");
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    for (int q = 0; q < values.size(); ++q) {
        const double weight = surface.dofs[static_cast<std::size_t>(q)].weight;
        weighted_sum += weight * values[q];
        weight_sum += weight;
    }
    return weighted_sum / weight_sum;
}

[[maybe_unused]] app3d::KnownDirichletValueGradientHessianCallback3D
manufactured_dirichlet_data_callback_3d(
    const app3d::RigidTransform3D& transform)
{
    return [transform](
               int,
               double,
               double,
               const Eigen::Vector3d& point,
               const Eigen::Vector3d&) {
        app3d::KnownDirichletValueGradientHessian3D result;
        result.value = app3d::transformed_manufactured_harmonic_value_3d(
            transform, point);
        result.ambient_gradient =
            app3d::transformed_manufactured_harmonic_gradient_3d(
                transform, point);
        result.ambient_hessian =
            app3d::transformed_manufactured_harmonic_hessian_3d(
                transform, point);
        return result;
    };
}

[[maybe_unused]] app3d::KnownDirichletJetCallback3D
manufactured_dirichlet_jet_callback_3d(
    const app3d::RigidTransform3D& transform)
{
    const app3d::KnownDirichletValueGradientHessianCallback3D data_callback =
        manufactured_dirichlet_data_callback_3d(transform);
    return [data_callback](
               int patch,
               double u,
               double v,
               const app3d::NativeSurfaceParameterJet3D& geometry,
               const app3d::LocalOrthonormalFrame3D& frame) {
        return app3d::evaluate_known_dirichlet_jet_3d(
            data_callback, patch, u, v, geometry, frame);
    };
}

app3d::KnownNeumannJetCallback3D
manufactured_neumann_jet_callback_3d(
    const app3d::RigidTransform3D& transform,
    double compatibility_mean_removed)
{
    return [transform, compatibility_mean_removed](
               int,
               double,
               double,
               const app3d::NativeSurfaceParameterJet3D& geometry,
               const app3d::LocalOrthonormalFrame3D& frame) {
        app3d::NativeSurfaceNormalParameterJet3D normal_jet =
            app3d::surface_normal_parameter_jet_3d(geometry);
        if (normal_jet.normal.dot(frame.normal) < 0.0) {
            normal_jet.normal = -normal_jet.normal;
            normal_jet.normal_u = -normal_jet.normal_u;
            normal_jet.normal_v = -normal_jet.normal_v;
        }
        const Eigen::Vector3d gradient =
            app3d::transformed_manufactured_harmonic_gradient_3d(
                transform, geometry.point);
        const Eigen::Matrix3d hessian =
            app3d::transformed_manufactured_harmonic_hessian_3d(
                transform, geometry.point);
        Eigen::Vector2d parameter_gradient;
        parameter_gradient.x() =
            (hessian * geometry.x_u).dot(normal_jet.normal)
            + gradient.dot(normal_jet.normal_u);
        parameter_gradient.y() =
            (hessian * geometry.x_v).dot(normal_jet.normal)
            + gradient.dot(normal_jet.normal_v);
        const Eigen::Vector2d tangent_gradient =
            app3d::parameter_gradient_to_tangent_gradient_3d(
                geometry, frame, parameter_gradient);
        app3d::NormalJet3D result;
        result << gradient.dot(normal_jet.normal)
                    - compatibility_mean_removed,
            tangent_gradient.x(), tangent_gradient.y();
        return result;
    };
}

SolveMetrics3D run_neumann_case(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const app3d::RigidTransform3D& transform,
    int gmres_max_iterations,
    double gmres_tolerance,
    NeumannCompatibilityMode3D compatibility_mode,
    ExteriorNormalRestrictMode3D trace_restrict_mode)
{
    const int size = pipeline.surface_size();
    Eigen::VectorXd exact_trace(size);
    Eigen::VectorXd normal_data(size);
    for (int q = 0; q < size; ++q) {
        const SurfaceDof& dof =
            pipeline.surface().dofs[static_cast<std::size_t>(q)];
        exact_trace[q] = app3d::transformed_manufactured_harmonic_value_3d(
            transform, dof.point);
        normal_data[q] =
            app3d::transformed_manufactured_harmonic_gradient_3d(
                transform, dof.point).dot(dof.normal);
    }
    // Enforce the discrete compatibility condition.  The exact flux has zero
    // continuous mean; the tiny quadrature defect is otherwise amplified by
    // the first-kind exterior-trace equation.
    const double compatibility_mean_removed = surface_weighted_mean(
        pipeline.surface(), normal_data);
    normal_data.array() -= compatibility_mean_removed;
    const auto solve_start = std::chrono::steady_clock::now();
    const ExteriorZeroTraceSolution3D solution =
        solve_exterior_zero_trace_neumann_3d(
            pipeline,
            normal_data,
            compatibility_mode,
            trace_restrict_mode,
            gmres_tolerance,
            80,
            gmres_max_iterations);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    const HarmonicJetField3D field{
        solution.potential, solution.coefficients};
    const Eigen::VectorXd direct_exterior = pipeline.exterior_trace(
        field, solution.value_jump, solution.normal_jump,
        trace_restrict_mode);
    const Eigen::VectorXd exterior_from_jump = pipeline.interior_trace(
        field, solution.value_jump, solution.normal_jump,
        trace_restrict_mode)
        - solution.value_jump;

    SolveMetrics3D result;
    result.formulation =
        compatibility_mode
            == NeumannCompatibilityMode3D::OperatorFluxCorrection
        ? "neumann_exterior_zero_value_trace_operator_flux"
        : "neumann_exterior_zero_value_trace_legacy";
    result.compatibility_mode =
        neumann_compatibility_mode_name(compatibility_mode);
    result.trace_restrict_mode =
        trace_restrict_mode_name(trace_restrict_mode);
    result.restrict_value_jet = "configured_sample_dof";
    result.iterations = solution.iterations;
    result.converged = solution.converged;
    result.seconds = seconds;
    result.gmres_tolerance = gmres_tolerance;
    result.gmres_relative_residual = solution.gmres_residuals.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : solution.gmres_residuals.back();
    result.operator_residual_linf = vector_linf(
        solution.augmented_residual.head(size));
    result.exterior_condition_linf = vector_linf(direct_exterior);
    result.lagrange_multiplier = solution.lagrange_multiplier;
    result.operator_compatibility_correction =
        solution.operator_compatibility_correction;
    result.total_flux_correction =
        compatibility_mean_removed
        + solution.operator_compatibility_correction;
    result.border_column_linf = solution.border_column_linf;
    result.exterior_trace_weighted_mean = surface_weighted_mean(
        pipeline.surface(), direct_exterior);
    const Eigen::VectorXd demeaned_exterior =
        direct_exterior.array() - result.exterior_trace_weighted_mean;
    result.exterior_trace_demeaned_linf = vector_linf(demeaned_exterior);
    Eigen::VectorXd expected_augmented_residual = direct_exterior;
    if (compatibility_mode
        == NeumannCompatibilityMode3D::TraceBorderLegacy) {
        expected_augmented_residual.array() +=
            solution.lagrange_multiplier;
    }
    result.augmented_trace_closure_linf = vector_linf(
        solution.augmented_residual.head(size)
        - expected_augmented_residual);
    result.route_mismatch_linf = vector_linf(
        direct_exterior - exterior_from_jump);
    result.compatibility_mean_removed = compatibility_mean_removed;
    result.data_weighted_mean = surface_weighted_mean(
        pipeline.surface(), solution.normal_jump);
    result.density_weighted_mean = surface_weighted_mean(
        pipeline.surface(), solution.value_jump);
    const double physical_tolerance = 10.0 * gmres_tolerance;
    result.physical_converged =
        result.converged
        && result.exterior_condition_linf <= physical_tolerance
        && (compatibility_mode
                == NeumannCompatibilityMode3D::OperatorFluxCorrection
            || std::abs(result.lagrange_multiplier)
                   <= physical_tolerance)
        && result.augmented_trace_closure_linf <= physical_tolerance;

    exact_trace.array() -= surface_weighted_mean(
        pipeline.surface(), exact_trace);
    const Eigen::VectorXd density_error = solution.value_jump - exact_trace;
    result.density_linf = vector_linf(density_error);
    result.density_l2 = vector_rms(density_error);

    double shift_sum = 0.0;
    int interior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) <= 0)
            continue;
        shift_sum += app3d::transformed_manufactured_harmonic_value_3d(
                         transform, grid_point(grid, node))
                   - solution.potential[node];
        ++interior_count;
    }
    result.constant_shift = shift_sum / static_cast<double>(interior_count);
    double interior_error_sq = 0.0;
    double exterior_error_sq = 0.0;
    int exterior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) > 0) {
            const double error = solution.potential[node]
                               + result.constant_shift
                               - app3d::transformed_manufactured_harmonic_value_3d(
                                     transform, grid_point(grid, node));
            result.interior_linf = std::max(
                result.interior_linf, std::abs(error));
            interior_error_sq += error * error;
        } else {
            const double error = solution.potential[node];
            result.exterior_bulk_linf = std::max(
                result.exterior_bulk_linf, std::abs(error));
            exterior_error_sq += error * error;
            ++exterior_count;
        }
    }
    result.interior_l2 = std::sqrt(
        interior_error_sq / static_cast<double>(interior_count));
    result.exterior_bulk_l2 = std::sqrt(
        exterior_error_sq / static_cast<double>(exterior_count));
    return result;
}

SolveMetrics3D run_dirichlet_normal_case(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const app3d::RigidTransform3D& transform,
    int gmres_max_iterations,
    double gmres_tolerance,
    ExteriorNormalRestrictMode3D mode =
        ExteriorNormalRestrictMode3D::JointTricubicCauchy,
    std::vector<double>* residual_history = nullptr)
{
    const int size = pipeline.surface_size();
    Eigen::VectorXd value_data(size);
    Eigen::VectorXd exact_normal(size);
    for (int q = 0; q < size; ++q) {
        const SurfaceDof& dof =
            pipeline.surface().dofs[static_cast<std::size_t>(q)];
        value_data[q] = app3d::transformed_manufactured_harmonic_value_3d(
            transform, dof.point);
        exact_normal[q] =
            app3d::transformed_manufactured_harmonic_gradient_3d(
                transform, dof.point).dot(dof.normal);
    }

    const auto solve_start = std::chrono::steady_clock::now();
    const ExteriorNormalTraceSolution3D solution =
        solve_exterior_zero_normal_dirichlet_3d(
            pipeline, value_data,
            mode, gmres_tolerance, 0, gmres_max_iterations);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    const HarmonicJetField3D field{
        solution.potential, solution.coefficients};
    const Eigen::VectorXd direct_exterior_normal =
        pipeline.exterior_normal_trace(
            field, value_data, solution.normal_jump, mode);
    const Eigen::VectorXd exterior_normal_from_jump =
        pipeline.interior_normal_trace(
            field, value_data, solution.normal_jump, mode)
        - solution.normal_jump;
    const Eigen::VectorXd boundary_residual = pipeline.interior_trace(
        field, value_data, solution.normal_jump, mode) - value_data;

    SolveMetrics3D result;
    result.formulation = "dirichlet_exterior_zero_normal_trace";
    result.compatibility_mode = "not_applicable";
    result.trace_restrict_mode = trace_restrict_mode_name(mode);
    result.known_value_jump = "sample_fit";
    result.dirichlet_jump_space = "legacy_sample_fit";
    result.iterations = solution.iterations;
    result.converged = solution.converged;
    result.seconds = seconds;
    result.gmres_tolerance = gmres_tolerance;
    result.gmres_relative_residual = solution.gmres_residuals.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : solution.gmres_residuals.back();
    result.operator_residual_linf = vector_linf(solution.operator_residual);
    result.exterior_condition_linf = vector_linf(direct_exterior_normal);
    result.boundary_residual_linf = vector_linf(boundary_residual);
    result.route_mismatch_linf = vector_linf(
        direct_exterior_normal - exterior_normal_from_jump);
    result.data_weighted_mean = surface_weighted_mean(
        pipeline.surface(), value_data);
    result.density_weighted_mean = surface_weighted_mean(
        pipeline.surface(), solution.normal_jump);
    result.physical_converged =
        result.converged
        && result.exterior_condition_linf <= 10.0 * gmres_tolerance;

    const Eigen::VectorXd density_error = solution.normal_jump - exact_normal;
    result.density_linf = vector_linf(density_error);
    result.density_l2 = vector_rms(density_error);

    double interior_error_sq = 0.0;
    double exterior_error_sq = 0.0;
    int interior_count = 0;
    int exterior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) > 0) {
            const double error = solution.potential[node]
                               - app3d::transformed_manufactured_harmonic_value_3d(
                                     transform, grid_point(grid, node));
            result.interior_linf = std::max(
                result.interior_linf, std::abs(error));
            interior_error_sq += error * error;
            ++interior_count;
        } else {
            const double error = solution.potential[node];
            result.exterior_bulk_linf = std::max(
                result.exterior_bulk_linf, std::abs(error));
            exterior_error_sq += error * error;
            ++exterior_count;
        }
    }
    result.interior_l2 = std::sqrt(
        interior_error_sq / static_cast<double>(interior_count));
    result.exterior_bulk_l2 = std::sqrt(
        exterior_error_sq / static_cast<double>(exterior_count));
    if (residual_history != nullptr)
        *residual_history = solution.gmres_residuals;
    return result;
}

SolveMetrics3D run_neumann_coefficient_case(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const NativeDensityTransfer3D& value_transfer,
    const NativeDensityTransfer3D* normal_transfer,
    const app3d::RigidTransform3D& transform,
    int gmres_max_iterations,
    double gmres_tolerance,
    NeumannCompatibilityMode3D compatibility_mode,
    ExteriorNormalRestrictMode3D trace_restrict_mode,
    NeumannRestrictValueJetMode3D restrict_value_jet_mode)
{
    const int size = pipeline.surface_size();
    Eigen::VectorXd exact_trace(size);
    Eigen::VectorXd normal_data(size);
    for (int q = 0; q < size; ++q) {
        const SurfaceDof& dof =
            pipeline.surface().dofs[static_cast<std::size_t>(q)];
        exact_trace[q] = app3d::transformed_manufactured_harmonic_value_3d(
            transform, dof.point);
        normal_data[q] =
            app3d::transformed_manufactured_harmonic_gradient_3d(
                transform, dof.point).dot(dof.normal);
    }
    const double compatibility_mean_removed = surface_weighted_mean(
        pipeline.surface(), normal_data);
    normal_data.array() -= compatibility_mean_removed;
    const app3d::KnownNeumannJetCallback3D known_normal_jet =
        pipeline.has_direct_coefficient_value_density(
            value_transfer.density())
        ? manufactured_neumann_jet_callback_3d(
              transform, compatibility_mean_removed)
        : app3d::KnownNeumannJetCallback3D{};

    std::unique_ptr<NativeGaussExteriorValueRestrict3D> gauss_restrict;
    std::unique_ptr<NativeDensityValueSpread3D> native_value_spread;
    if (selected_neumann_native_gauss_trace_sampling()) {
        if (trace_restrict_mode
            != ExteriorNormalRestrictMode3D::
                   GlobalCubicExteriorBranchCrossingOwner) {
            throw std::invalid_argument(
                "native Gauss Neumann sampling requires the global "
                "exterior-branch restrict mode");
        }
        gauss_restrict =
            std::make_unique<NativeGaussExteriorValueRestrict3D>(
                grid, grid_pair, pipeline, value_transfer.density());
        native_value_spread =
            std::make_unique<NativeDensityValueSpread3D>(
                grid, grid_pair, pipeline, value_transfer.density());
    }

    const auto solve_start = std::chrono::steady_clock::now();
    const CoefficientExteriorZeroTraceSolution3D solution =
        solve_exterior_zero_trace_neumann_coefficients_3d(
            pipeline, value_transfer, normal_transfer,
            gauss_restrict.get(),
            native_value_spread.get(), normal_data, compatibility_mode,
            trace_restrict_mode, restrict_value_jet_mode,
            gmres_tolerance, 80,
            gmres_max_iterations, known_normal_jet);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    HarmonicJetField3D field{
        solution.potential,
        solution.cauchy_coefficients,
        solution.restrict_value_jets,
        solution.restrict_normal_coefficients};
    if (pipeline.has_direct_coefficient_value_density(
            value_transfer.density())) {
        field.direct_value_c0 = solution.value_c0_coefficients;
        field.direct_known_normal_jet = known_normal_jet;
        field.direct_coefficient_cauchy = true;
    }
    const Eigen::VectorXd direct_exterior = gauss_restrict
        ? gauss_restrict->exterior_value_trace(
              field, solution.value_c0_coefficients)
        : pipeline.exterior_trace(
              field, solution.value_jump, solution.normal_jump,
              trace_restrict_mode);
    const Eigen::VectorXd trace_value_jump = gauss_restrict
        ? value_transfer.evaluate_c0_at_gauss(
              solution.value_c0_coefficients)
        : solution.value_jump;
    const Eigen::VectorXd exterior_from_jump = gauss_restrict
        ? gauss_restrict->interior_value_trace(
              field, solution.value_c0_coefficients)
              - trace_value_jump
        : pipeline.interior_trace(
              field, solution.value_jump, solution.normal_jump,
              trace_restrict_mode)
              - solution.value_jump;

    SolveMetrics3D result;
    result.formulation = solution.reduction_scheme
            == "topology_affine_local_svd"
        ? "neumann_topology_affine_mean_free_exterior_zero_value_trace"
        : (compatibility_mode
            == NeumannCompatibilityMode3D::OperatorFluxCorrection
        ? "neumann_coefficient_exterior_zero_value_trace_operator_flux"
        : "neumann_coefficient_exterior_zero_value_trace_legacy");
    result.compatibility_mode = solution.reduction_scheme
            == "topology_affine_local_svd"
        ? "prescribed_flux_mean_removed+mean_free_density"
        : neumann_compatibility_mode_name(compatibility_mode);
    result.trace_restrict_mode =
        trace_restrict_mode_name(trace_restrict_mode);
    result.restrict_value_jet = solution.reduction_scheme
            == "topology_affine_local_svd"
        ? "direct_coefficient_cauchy"
        : neumann_restrict_value_jet_mode_name(
              restrict_value_jet_mode);
    result.trace_sampling = neumann_trace_sampling_name();
    result.density_coordinates = neumann_density_coordinates_name();
    result.border_solver = solution.reduction_scheme
            == "topology_affine_local_svd"
        ? "mean_free_pivot_elimination"
        : (solution.edge_jump_jet == "disabled"
               ? neumann_border_solver_name()
               : "edge_jump_jet_projection");
    result.edge_jump_jet = solution.edge_jump_jet;
    result.edge_feature_edges = solution.edge_feature_edges;
    result.edge_constraint_rows = solution.edge_constraint_rows;
    result.edge_constraint_rank = solution.edge_constraint_rank;
    result.edge_reduced_dofs = solution.edge_reduced_dofs;
    result.edge_constraint_residual_linf =
        solution.edge_constraint_residual_linf;
    result.edge_nullspace_residual_linf =
        solution.edge_nullspace_residual_linf;
    result.edge_normal_fit_linf = solution.edge_normal_fit_linf;
    result.edge_target_projection_linf =
        solution.edge_target_projection_linf;
    result.feature_target_requested_linf =
        solution.feature_target_requested_linf;
    result.feature_target_requested_l2 =
        solution.feature_target_requested_l2;
    result.feature_target_projected_linf =
        solution.feature_target_projected_linf;
    result.feature_target_projected_l2 =
        solution.feature_target_projected_l2;
    result.feature_target_projection_l2 =
        solution.feature_target_projection_l2;
    result.feature_target_projection_relative_l2 =
        solution.feature_target_projection_relative_l2;
    result.feature_target_reachable_certification_linf =
        solution.feature_target_reachable_certification_linf;
    result.feature_target_projection_components =
        solution.feature_target_projection_components;
    result.feature_target_constraints_exact =
        solution.feature_target_constraints_exact;
    result.reduction_scheme = solution.reduction_scheme;
    result.particular_solver = solution.particular_solver;
    result.topology_constraint_rows = solution.topology_constraint_rows;
    result.topology_constraint_rank = solution.topology_constraint_rank;
    result.topology_base_coordinates =
        solution.topology_base_coordinates;
    result.topology_full_coordinates =
        solution.topology_full_coordinates;
    result.topology_pre_mean_coordinates =
        solution.topology_pre_mean_coordinates;
    result.topology_reduced_coordinates =
        solution.topology_reduced_coordinates;
    result.topology_block_count = solution.topology_block_count;
    result.topology_particular_residual_linf =
        solution.topology_particular_residual_linf;
    result.topology_homogeneous_residual_linf =
        solution.topology_homogeneous_residual_linf;
    result.topology_projector_rank = solution.topology_projector_rank;
    result.topology_projector_condition =
        solution.topology_projector_condition;
    result.topology_projector_pb_error =
        solution.topology_projector_pb_error;
    result.topology_trace_samples = solution.topology_trace_samples;
    result.topology_trace_oversampling_margin =
        solution.topology_trace_oversampling_margin;
    result.topology_trace_oversampling_ratio =
        solution.topology_trace_oversampling_ratio;
    result.topology_trace_closure_linf =
        solution.topology_trace_closure_linf;
    result.mean_free_pivot_index = solution.mean_free_pivot_index;
    result.mean_free_pivot_moment = solution.mean_free_pivot_moment;
    result.mean_free_relative_observability =
        solution.mean_free_relative_observability;
    result.mean_free_coordinate_condition =
        solution.mean_free_coordinate_condition;
    result.mean_free_particular_residual =
        solution.mean_free_particular_residual;
    result.mean_free_homogeneous_residual =
        solution.mean_free_homogeneous_residual;
    result.mean_free_constant_projection_linf =
        solution.mean_free_constant_projection_linf;
    result.mean_free_intrinsic_mean = solution.mean_free_intrinsic_mean;
    result.topology_block_records = solution.topology_block_records;
    result.iterations = solution.iterations;
    result.converged = solution.converged;
    result.seconds = seconds;
    result.gmres_tolerance = gmres_tolerance;
    result.gmres_relative_residual = solution.gmres_residuals.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : solution.gmres_residuals.back();
    const Eigen::Index augmented_head_size =
        solution.augmented_residual.size() - 1;
    result.operator_residual_linf = vector_linf(
        solution.augmented_residual.head(augmented_head_size));
    result.compatibility_residual = std::abs(
        solution.augmented_residual[augmented_head_size]);
    result.exterior_condition_linf = vector_linf(direct_exterior);
    result.lagrange_multiplier = solution.lagrange_multiplier;
    result.operator_compatibility_correction =
        solution.operator_compatibility_correction;
    result.total_flux_correction =
        compatibility_mean_removed
        + solution.operator_compatibility_correction;
    result.border_column_linf = solution.border_column_linf;
    result.exterior_trace_weighted_mean = gauss_restrict
        ? value_transfer.gauss_weighted_mean(direct_exterior)
        : surface_weighted_mean(pipeline.surface(), direct_exterior);
    result.exterior_trace_demeaned_linf = vector_linf(
        (direct_exterior.array()
         - result.exterior_trace_weighted_mean).matrix());
    const bool use_galerkin_projection =
        selected_neumann_galerkin_trace_projection();
    const bool use_trace_mass_coordinates =
        selected_neumann_trace_mass_coordinates();
    const auto project_trace = [&](const Eigen::VectorXd& trace) {
        if (gauss_restrict) {
            return use_trace_mass_coordinates
                ? value_transfer.fit_gauss_trace_mass(trace)
                : value_transfer.fit_gauss_orthonormal(trace);
        }
        if (use_trace_mass_coordinates)
            return value_transfer.fit_trace_mass(trace);
        return use_galerkin_projection
            ? value_transfer.galerkin_orthonormal(trace)
            : value_transfer.fit_orthonormal(trace);
    };
    Eigen::VectorXd expected_coefficient_residual =
        project_trace(direct_exterior);
    if (solution.reduction_scheme != "topology_affine_local_svd"
        && compatibility_mode
               == NeumannCompatibilityMode3D::TraceBorderLegacy) {
        expected_coefficient_residual += solution.lagrange_multiplier
            * project_trace(Eigen::VectorXd::Ones(
                  static_cast<Eigen::Index>(direct_exterior.size())));
    }
    result.augmented_trace_closure_linf =
        solution.reduction_scheme == "topology_affine_local_svd"
        ? solution.topology_trace_closure_linf
        : vector_linf(
              solution.augmented_residual.head(augmented_head_size)
              - expected_coefficient_residual);
    result.route_mismatch_linf = vector_linf(
        direct_exterior - exterior_from_jump);
    result.compatibility_mean_removed = compatibility_mean_removed;
    result.data_weighted_mean = surface_weighted_mean(
        pipeline.surface(), solution.normal_jump);
    result.density_weighted_mean = solution.reduction_scheme
            == "topology_affine_local_svd"
        ? surface_weighted_mean(pipeline.surface(), solution.value_jump)
        : ((solution.edge_jump_jet != "disabled" || gauss_restrict)
        ? value_transfer.c0_weighted_mean(
              solution.value_c0_coefficients)
        : surface_weighted_mean(pipeline.surface(), solution.value_jump));
    result.physical_converged = result.converged
        && result.operator_residual_linf <= 10.0 * gmres_tolerance
        && result.exterior_condition_linf <= 10.0 * gmres_tolerance
        && result.augmented_trace_closure_linf <= 10.0 * gmres_tolerance
        && result.compatibility_residual <= 10.0 * gmres_tolerance
        && std::abs(result.density_weighted_mean)
               <= 10.0 * gmres_tolerance
        && (result.edge_jump_jet == "disabled"
            || result.edge_constraint_residual_linf
                   <= 10.0 * gmres_tolerance);

    exact_trace.array() -= surface_weighted_mean(
        pipeline.surface(), exact_trace);
    const Eigen::VectorXd density_error =
        solution.value_jump - exact_trace;
    result.density_linf = vector_linf(density_error);
    result.density_l2 = vector_rms(density_error);

    double shift_sum = 0.0;
    int interior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) <= 0)
            continue;
        shift_sum += app3d::transformed_manufactured_harmonic_value_3d(
                         transform, grid_point(grid, node))
                   - solution.potential[node];
        ++interior_count;
    }
    result.constant_shift = shift_sum / static_cast<double>(interior_count);
    double interior_error_sq = 0.0;
    double exterior_error_sq = 0.0;
    int exterior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) > 0) {
            const double error = solution.potential[node]
                + result.constant_shift
                - app3d::transformed_manufactured_harmonic_value_3d(
                    transform, grid_point(grid, node));
            result.interior_linf = std::max(
                result.interior_linf, std::abs(error));
            interior_error_sq += error * error;
        } else {
            const double error = solution.potential[node];
            result.exterior_bulk_linf = std::max(
                result.exterior_bulk_linf, std::abs(error));
            exterior_error_sq += error * error;
            ++exterior_count;
        }
    }
    result.interior_l2 = std::sqrt(
        interior_error_sq / static_cast<double>(interior_count));
    result.exterior_bulk_l2 = std::sqrt(
        exterior_error_sq / static_cast<double>(exterior_count));
    return result;
}

SolveMetrics3D run_dirichlet_normal_coefficient_case(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const NativeDensityTransfer3D& normal_transfer,
    const app3d::RigidTransform3D& transform,
    int gmres_max_iterations,
    double gmres_tolerance,
    ExteriorNormalRestrictMode3D mode)
{
    const int size = pipeline.surface_size();
    Eigen::VectorXd value_data(size);
    Eigen::VectorXd exact_normal(size);
    for (int q = 0; q < size; ++q) {
        const SurfaceDof& dof =
            pipeline.surface().dofs[static_cast<std::size_t>(q)];
        value_data[q] = app3d::transformed_manufactured_harmonic_value_3d(
            transform, dof.point);
        exact_normal[q] =
            app3d::transformed_manufactured_harmonic_gradient_3d(
                transform, dof.point).dot(dof.normal);
    }
    const app3d::KnownDirichletValueGradientHessianCallback3D
        known_dirichlet = manufactured_dirichlet_data_callback_3d(transform);
    const app3d::KnownDirichletJetCallback3D known_value_jet =
        manufactured_dirichlet_jet_callback_3d(transform);

    const auto solve_start = std::chrono::steady_clock::now();
    const CoefficientExteriorNormalTraceSolution3D solution =
        solve_exterior_zero_normal_dirichlet_coefficients_3d(
            pipeline, normal_transfer, value_data, known_dirichlet, mode,
            gmres_tolerance, 0, gmres_max_iterations);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    HarmonicJetField3D field{
        solution.potential,
        solution.cauchy_coefficients,
        solution.restrict_value_jets,
        solution.restrict_normal_coefficients};
    if (pipeline.has_direct_coefficient_normal_density(
            normal_transfer.density())) {
        field.direct_normal_c0 = solution.normal_c0_coefficients;
        if (solution.known_value_jump == "analytic_boundary_jet")
            field.direct_known_value_jet = known_value_jet;
        field.direct_coefficient_cauchy = true;
    }
    const Eigen::VectorXd direct_exterior_normal =
        pipeline.exterior_normal_trace(
            field, value_data, solution.normal_jump, mode);
    const Eigen::VectorXd exterior_normal_from_jump =
        pipeline.interior_normal_trace(
            field, value_data, solution.normal_jump, mode)
        - solution.normal_jump;
    const Eigen::VectorXd boundary_residual = pipeline.interior_trace(
        field, value_data, solution.normal_jump, mode) - value_data;

    SolveMetrics3D result;
    result.formulation =
        "dirichlet_coefficient_exterior_zero_normal_trace";
    result.compatibility_mode = "not_applicable";
    result.trace_restrict_mode = trace_restrict_mode_name(mode);
    result.trace_sampling = "panel_centers";
    result.density_coordinates = solution.reduction_scheme
            == "topology_affine_local_svd"
        ? "trace_mass"
        : "c0_orthonormal";
    result.border_solver = "not_applicable";
    result.known_value_jump = solution.known_value_jump;
    result.dirichlet_jump_space = solution.jump_space;
    result.edge_jump_jet = solution.feature_constraint_mode;
    result.edge_feature_edges = solution.feature_edge_count;
    result.edge_constraint_rows = solution.feature_constraint_rows;
    result.edge_constraint_rank = solution.feature_constraint_rank;
    result.edge_reduced_dofs = solution.topology_reduced_coordinates;
    result.edge_constraint_residual_linf =
        solution.feature_constraint_residual_linf;
    result.edge_nullspace_residual_linf =
        solution.topology_homogeneous_residual_linf;
    result.feature_vertex_residual_linf =
        solution.feature_vertex_residual_linf;
    result.feature_edge_residual_linf =
        solution.feature_edge_residual_linf;
    result.feature_discarded_residual_linf =
        solution.feature_discarded_residual_linf;
    result.trace_projection_leakage_relative_l2 =
        solution.trace_projection_leakage_relative_l2;
    result.reduction_scheme = solution.reduction_scheme;
    result.particular_solver = solution.particular_solver;
    result.topology_constraint_rows = solution.topology_constraint_rows;
    result.topology_constraint_rank = solution.topology_constraint_rank;
    result.topology_base_coordinates =
        solution.topology_base_coordinates;
    result.topology_full_coordinates =
        solution.topology_full_coordinates;
    result.topology_reduced_coordinates =
        solution.topology_reduced_coordinates;
    result.topology_block_count = solution.topology_block_count;
    result.topology_particular_residual_linf =
        solution.topology_particular_residual_linf;
    result.topology_homogeneous_residual_linf =
        solution.topology_homogeneous_residual_linf;
    result.topology_projector_rank = solution.topology_projector_rank;
    result.topology_projector_condition =
        solution.topology_projector_condition;
    result.topology_projector_pb_error =
        solution.topology_projector_pb_error;
    result.topology_trace_samples = solution.topology_trace_samples;
    result.topology_trace_oversampling_margin =
        solution.topology_trace_oversampling_margin;
    result.topology_trace_oversampling_ratio =
        solution.topology_trace_oversampling_ratio;
    result.topology_trace_closure_linf =
        solution.topology_trace_closure_linf;
    result.topology_block_records = solution.topology_block_records;
    result.iterations = solution.iterations;
    result.converged = solution.converged;
    result.seconds = seconds;
    result.gmres_tolerance = gmres_tolerance;
    result.gmres_relative_residual = solution.gmres_residuals.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : solution.gmres_residuals.back();
    result.operator_residual_linf =
        vector_linf(solution.operator_residual);
    result.exterior_condition_linf =
        vector_linf(direct_exterior_normal);
    result.boundary_residual_linf = vector_linf(boundary_residual);
    result.route_mismatch_linf = vector_linf(
        direct_exterior_normal - exterior_normal_from_jump);
    result.data_weighted_mean = surface_weighted_mean(
        pipeline.surface(), value_data);
    result.density_weighted_mean = surface_weighted_mean(
        pipeline.surface(), solution.normal_jump);
    result.physical_converged = result.converged
        && result.exterior_condition_linf <= 10.0 * gmres_tolerance
        && result.edge_constraint_residual_linf
               <= 10.0 * gmres_tolerance;
    const Eigen::VectorXd density_error =
        solution.normal_jump - exact_normal;
    result.density_linf = vector_linf(density_error);
    result.density_l2 = vector_rms(density_error);

    double interior_error_sq = 0.0;
    double exterior_error_sq = 0.0;
    int interior_count = 0;
    int exterior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) > 0) {
            const double error = solution.potential[node]
                - app3d::transformed_manufactured_harmonic_value_3d(
                    transform, grid_point(grid, node));
            result.interior_linf = std::max(
                result.interior_linf, std::abs(error));
            interior_error_sq += error * error;
            ++interior_count;
        } else {
            const double error = solution.potential[node];
            result.exterior_bulk_linf = std::max(
                result.exterior_bulk_linf, std::abs(error));
            exterior_error_sq += error * error;
            ++exterior_count;
        }
    }
    result.interior_l2 = std::sqrt(
        interior_error_sq / static_cast<double>(interior_count));
    result.exterior_bulk_l2 = std::sqrt(
        exterior_error_sq / static_cast<double>(exterior_count));
    return result;
}

double validate_interface(const Interface3D& iface)
{
    if (iface.panel_node_layout() != PanelNodeLayout3D::QuadraticLagrange
        || iface.points_per_panel() != 6
        || iface.num_panels() <= 0
        || iface.num_points() <= 0
        || iface.weights().sum() <= 0.0) {
        throw std::runtime_error("invalid P2 surface interface");
    }
    double normal_error = 0.0;
    for (int q = 0; q < iface.num_points(); ++q) {
        const double norm = iface.normals().row(q).norm();
        if (!std::isfinite(norm))
            throw std::runtime_error("surface contains a non-finite normal");
        normal_error = std::max(normal_error, std::abs(norm - 1.0));
    }
    if (normal_error > 1.0e-9)
        throw std::runtime_error("surface normals are not unit length");
    return normal_error;
}

std::ofstream open_output_file(const std::filesystem::path& path)
{
    std::ofstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open output file: " + path.string());
    return stream;
}

void write_surface_files(const std::filesystem::path& output_dir,
                         const GeometryBundle& geometry,
                         int N)
{
    std::filesystem::create_directories(output_dir);
    const std::string stem = geometry.name + "_N" + std::to_string(N);
    const Interface3D& surface = geometry.crossing_interface;
    std::ofstream obj = open_output_file(
        output_dir / (stem + "_crossing_surface.obj"));
    obj << std::setprecision(17);
    obj << "# " << geometry.description << '\n';
    for (int v = 0; v < surface.num_vertices(); ++v)
        obj << "v " << surface.vertices()(v, 0) << ' '
            << surface.vertices()(v, 1) << ' '
            << surface.vertices()(v, 2) << '\n';
    for (int p = 0; p < surface.num_panels(); ++p)
        obj << "f " << surface.panels()(p, 0) + 1 << ' '
            << surface.panels()(p, 1) + 1 << ' '
            << surface.panels()(p, 2) + 1 << '\n';

    const Interface3D& iface = geometry.correction_interface;
    std::ofstream csv = open_output_file(
        output_dir / (stem + "_correction_dofs.csv"));
    csv << std::setprecision(17);
    csv << "q,x,y,z,nx,ny,nz,weight\n";
    for (int q = 0; q < iface.num_points(); ++q)
        csv << q << ','
            << iface.points()(q, 0) << ',' << iface.points()(q, 1) << ','
            << iface.points()(q, 2) << ',' << iface.normals()(q, 0) << ','
            << iface.normals()(q, 1) << ',' << iface.normals()(q, 2) << ','
            << iface.weights()[q] << '\n';
}

void write_panel_center_files(const std::filesystem::path& output_dir,
                              const std::string& geometry_name,
                              int N,
                              const SurfaceDofCloud& cloud,
                              const CauchyStencilSet& stencils)
{
    std::filesystem::create_directories(output_dir);
    const std::string stem = geometry_name + "_N" + std::to_string(N);

    std::ofstream patch_csv = open_output_file(
        output_dir / (stem + "_surface_patches.csv"));
    patch_csv << "patch_id,patch_name,nu,nv,smooth_patch_ids\n";
    for (int patch_id = 0;
         patch_id < static_cast<int>(cloud.patches.size()); ++patch_id) {
        const SurfacePatchInfo& patch =
            cloud.patches[static_cast<std::size_t>(patch_id)];
        patch_csv << patch_id << ',' << patch.name << ','
                  << patch.nu << ',' << patch.nv << ',';
        for (std::size_t j = 0; j < patch.smooth_patch_ids.size(); ++j) {
            if (j != 0)
                patch_csv << ';';
            patch_csv << patch.smooth_patch_ids[j];
        }
        patch_csv << '\n';
    }

    std::ofstream dof_csv = open_output_file(
        output_dir / (stem + "_surface_panel_center_dofs.csv"));
    dof_csv << std::setprecision(17);
    dof_csv << "q,patch_id,patch_name,i,j,u,v,"
               "x,y,z,nx,ny,nz,t1x,t1y,t1z,t2x,t2y,t2z,weight\n";
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q) {
        const SurfaceDof& dof = cloud.dofs[static_cast<std::size_t>(q)];
        dof_csv << q << ',' << dof.patch_id << ','
                << cloud.patches[static_cast<std::size_t>(dof.patch_id)].name
                << ',' << dof.i << ',' << dof.j << ','
                << dof.u << ',' << dof.v << ','
                << dof.point.x() << ',' << dof.point.y() << ',' << dof.point.z() << ','
                << dof.normal.x() << ',' << dof.normal.y() << ',' << dof.normal.z() << ','
                << dof.tangent1.x() << ',' << dof.tangent1.y() << ','
                << dof.tangent1.z() << ',' << dof.tangent2.x() << ','
                << dof.tangent2.y() << ',' << dof.tangent2.z() << ','
                << dof.weight << '\n';
    }

    std::ofstream stencil_csv = open_output_file(
        output_dir / (stem + "_cauchy_stencils.csv"));
    stencil_csv << std::setprecision(17);
    stencil_csv << "q,patch_id,radius_over_h,incident_patch_count,"
                   "value_count,normal_count,value_patch_imbalance,"
                   "normal_patch_imbalance";
    for (int k = 0; k < stencils.value_count; ++k)
        stencil_csv << ",value_" << k;
    for (int k = 0; k < stencils.derivative_count; ++k)
        stencil_csv << ",normal_" << k;
    stencil_csv << '\n';
    for (int q = 0; q < static_cast<int>(stencils.rows.size()); ++q) {
        const CauchyStencil& stencil =
            stencils.rows[static_cast<std::size_t>(q)];
        stencil_csv << q << ','
                    << cloud.dofs[static_cast<std::size_t>(q)].patch_id << ','
                    << stencil.radius_over_h << ','
                    << stencil.incident_patch_count << ','
                    << stencil.value_ids.size() << ','
                    << stencil.derivative_ids.size() << ','
                    << stencil.value_patch_imbalance << ','
                    << stencil.derivative_patch_imbalance;
        for (int id : stencil.value_ids)
            stencil_csv << ',' << id;
        for (int k = static_cast<int>(stencil.value_ids.size());
             k < stencils.value_count; ++k) {
            stencil_csv << ',';
        }
        for (int id : stencil.derivative_ids)
            stencil_csv << ',' << id;
        for (int k = static_cast<int>(stencil.derivative_ids.size());
             k < stencils.derivative_count; ++k) {
            stencil_csv << ',';
        }
        stencil_csv << '\n';
    }
}

ReadinessResult run_readiness_case(GeometryKind kind,
                                   int N,
                                   const std::filesystem::path& output_dir,
                                   CauchyStencilPolicy3D cauchy_policy,
                                   int cauchy_value_count,
                                   int cauchy_normal_count,
                                   int local_restrict_value_count,
                                   int local_restrict_normal_count,
                                   int gmres_max_iterations,
                                   double gmres_tolerance,
                                   NeumannCompatibilityMode3D
                                       neumann_compatibility_mode,
                                   ExteriorNormalRestrictMode3D
                                       neumann_trace_restrict_mode,
                                   NeumannRestrictValueJetMode3D
                                       neumann_restrict_value_jet_mode,
                                   ExteriorNormalRestrictMode3D
                                       dirichlet_normal_restrict_mode,
                                   const app3d::RigidTransform3D& transform,
                                   SolveSelection3D solve_selection)
{
    const double h = kBoxSide / static_cast<double>(N);
    CartesianGrid3D grid({kBoxMin, kBoxMin, kBoxMin},
                         {h, h, h},
                         {N, N, N},
                         DofLayout3D::Node);
    GeometryBundle geometry = make_geometry(kind, h, transform);
    const auto domain = std::make_shared<const
        geometry3d::NurbsCartesianDomain3D>(
            grid, geometry.native_surface.geometry_model());
    const SurfaceDofCloud surface_dofs =
        app3d::make_native_surface_dofs_3d(geometry.native_surface, h);
    const SurfaceCloudDiagnostics surface_diagnostics =
        validate_surface_dofs(surface_dofs, h);
    const CauchyStencilSet cauchy_stencils = build_cauchy_stencils(
        geometry.native_surface,
        surface_dofs,
        h,
        cauchy_value_count,
        cauchy_normal_count,
        cauchy_policy);
    if (solve_selection != SolveSelection3D::DirichletNormalOnly) {
        write_surface_files(output_dir, geometry, N);
        write_panel_center_files(
            output_dir, geometry.name, N, surface_dofs, cauchy_stencils);
    }

    ReadinessResult result;
    result.geometry = geometry.name;
    result.cauchy_policy = cauchy_policy_name(cauchy_policy);
    result.cauchy_value_trace_fit = cauchy_value_trace_fit_mode_name(
        selected_cauchy_value_trace_fit_mode());
    result.N = N;
    result.h = h;
    result.correction_panels = geometry.correction_interface.num_panels();
    result.correction_dofs = geometry.correction_interface.num_points();
    result.crossing_panels = geometry.crossing_interface.num_panels();
    result.feature_edges = geometry.feature_edges;
    result.feature_vertices = geometry.feature_vertices;
    result.correction_area = geometry.correction_interface.weights().sum();
    result.crossing_area = geometry.crossing_interface.weights().sum();
    result.normal_error = validate_interface(geometry.correction_interface);
    validate_interface(geometry.crossing_interface);
    result.surface_patches = static_cast<int>(surface_dofs.patches.size());
    result.surface_dofs = static_cast<int>(surface_dofs.dofs.size());
    result.surface_dof_area = surface_diagnostics.area;
    result.surface_area_relative_error = surface_diagnostics.area_relative_error;
    result.surface_spacing_min_over_h =
        surface_diagnostics.nearest_spacing_min_over_h;
    result.surface_spacing_mean_over_h =
        surface_diagnostics.nearest_spacing_mean_over_h;
    result.surface_spacing_max_over_h =
        surface_diagnostics.nearest_spacing_max_over_h;
    result.cauchy_value_neighbors = cauchy_stencils.value_count;
    result.cauchy_derivative_neighbors = cauchy_stencils.derivative_count;
    result.cauchy_value_neighbors_min = cauchy_stencils.value_count_min;
    result.cauchy_value_neighbors_max = cauchy_stencils.value_count_max;
    result.cauchy_derivative_neighbors_min =
        cauchy_stencils.derivative_count_min;
    result.cauchy_derivative_neighbors_max =
        cauchy_stencils.derivative_count_max;
    result.cauchy_radius_max_over_h = cauchy_stencils.radius_max_over_h;
    result.cauchy_radius_mean_over_h = cauchy_stencils.radius_mean_over_h;
    result.cauchy_incident_patches_min =
        cauchy_stencils.incident_patch_count_min;
    result.cauchy_incident_patches_max =
        cauchy_stencils.incident_patch_count_max;
    result.cauchy_value_patch_imbalance_max =
        cauchy_stencils.value_patch_imbalance_max;
    result.cauchy_derivative_patch_imbalance_max =
        cauchy_stencils.derivative_patch_imbalance_max;

    const auto dims = grid.dof_dims();
    Eigen::Vector3d grid_min(kBoxMin, kBoxMin, kBoxMin);
    Eigen::Vector3d grid_max(
        kBoxMin + static_cast<double>(dims[0] - 1) * h,
        kBoxMin + static_cast<double>(dims[1] - 1) * h,
        kBoxMin + static_cast<double>(dims[2] - 1) * h);
    const geometry3d::NurbsAabb3D& surface_bounds = domain->surface_bounds();
    const double min_margin = std::min(
        (surface_bounds.lower - grid_min).minCoeff(),
        (grid_max - surface_bounds.upper).minCoeff());
    result.min_box_margin_over_h = min_margin / h;
    if (result.min_box_margin_over_h < 2.0)
        throw std::runtime_error("surface is too close to the Cartesian box boundary");

    GridPair3D grid_pair(grid,
                         geometry.correction_interface,
                         geometry.crossing_interface,
                         domain,
                         panel_nurbs_patch_indices(
                             geometry.correction_triangles));
    const geometry3d::NurbsCartesianDomainDiagnostics3D&
        native_diagnostics = grid_pair.nurbs_domain_diagnostics();
    result.nurbs_patches = native_diagnostics.nurbs_patch_count;
    result.bezier_elements = native_diagnostics.bezier_element_count;
    result.acceleration_leaves = native_diagnostics.acceleration_leaf_count;
    result.maximum_query_element_extent =
        native_diagnostics.maximum_query_element_extent;
    result.candidate_grid_edges =
        native_diagnostics.candidate_grid_edge_count;
    result.triangle_seed_hits =
        native_diagnostics.intersections.triangle_seed_hits;
    result.triangle_seed_misses_recovered =
        native_diagnostics.intersections.triangle_seed_misses_recovered;
    result.subdivision_boxes =
        native_diagnostics.intersections.subdivision_boxes;
    result.newton_attempts =
        native_diagnostics.intersections.newton_attempts;
    result.newton_iterations =
        native_diagnostics.intersections.newton_iterations;
    result.maximum_subdivision_depth =
        native_diagnostics.intersections.maximum_subdivision_depth_reached;
    result.terminal_certificate_boxes =
        native_diagnostics.intersections.terminal_certificate_boxes;
    result.maximum_terminal_certificate_depth =
        native_diagnostics.intersections
            .maximum_terminal_certificate_depth_reached;
    result.closest_point_attempts =
        native_diagnostics.intersections.closest_point_attempts;
    result.closest_point_iterations =
        native_diagnostics.intersections.closest_point_iterations;
    result.closest_point_roots_recovered =
        native_diagnostics.intersections.roots_recovered_by_closest_point;
    result.closest_point_terminal_misses =
        native_diagnostics.intersections.terminal_misses_by_closest_point;
    result.closest_point_failures =
        native_diagnostics.intersections.closest_point_failures;
    result.seam_deduplications =
        native_diagnostics.intersections.seam_deduplications;
    result.sample_seed_candidates =
        native_diagnostics.intersections.sample_seed_candidates;
    result.sample_seeds_accepted =
        native_diagnostics.intersections.sample_seeds_accepted;
    result.sample_seed_roots_recovered =
        native_diagnostics.intersections.roots_recovered_by_sample_seed;
    result.maximum_sample_seeds_per_element =
        native_diagnostics.intersections.maximum_sample_seeds_per_element;
    result.stationary_solve_attempts =
        native_diagnostics.intersections.stationary_solve_attempts;
    result.stationary_solve_converged =
        native_diagnostics.intersections.stationary_solve_converged;
    result.stationary_witnesses =
        native_diagnostics.intersections.stationary_witnesses;
    result.stationary_protected_root_pairs =
        native_diagnostics.intersections
            .root_pairs_protected_by_stationary_witness;
    result.ambiguous_root_clusters =
        native_diagnostics.intersections.ambiguous_root_clusters;
    result.non_g1_topology_merges =
        native_diagnostics.intersections.non_g1_topology_merges;
    result.high_degree_fallbacks =
        native_diagnostics.intersections.high_degree_fallbacks;
    result.interface_x = native_diagnostics.interface_edge_counts[0];
    result.interface_y = native_diagnostics.interface_edge_counts[1];
    result.interface_z = native_diagnostics.interface_edge_counts[2];
    result.multi_crossing_edges =
        native_diagnostics.multi_crossing_edge_count;
    result.even_parity_interface_edges =
        native_diagnostics.even_parity_interface_edge_count;
    result.odd_parity_interface_edges =
        native_diagnostics.odd_parity_interface_edge_count;
    result.ambiguous_parity_edges =
        native_diagnostics.ambiguous_parity_edge_count;
    result.ambiguous_label_changing_edges =
        native_diagnostics.ambiguous_label_changing_edge_count;
    result.targeted_retries =
        native_diagnostics.targeted_retry_count;
    result.targeted_retries_resolved =
        native_diagnostics.targeted_retry_resolved_count;
    result.targeted_retries_unsafe =
        native_diagnostics.targeted_retry_unsafe_count;
    result.correction_safe_edges =
        native_diagnostics.correction_safe_edge_count;
    result.unsafe_label_changing_edges =
        native_diagnostics.unsafe_label_changing_edge_count;
    result.maximum_targeted_retry_subdivision_depth =
        native_diagnostics.targeted_retry_intersections
            .maximum_subdivision_depth_reached;
    result.endpoint_parity_fallbacks =
        native_diagnostics.endpoint_parity_fallback_count;
    result.endpoint_classification_queries =
        native_diagnostics.endpoint_classification_query_count;
    result.component_parity_toggles =
        native_diagnostics.component_parity_toggle_count;
    result.barrier_x = native_diagnostics.barrier_edge_counts[0];
    result.barrier_y = native_diagnostics.barrier_edge_counts[1];
    result.barrier_z = native_diagnostics.barrier_edge_counts[2];
    result.grid_components = native_diagnostics.grid_component_count;
    result.box_exterior_components =
        native_diagnostics.box_exterior_component_count;
    result.representative_queries =
        native_diagnostics.representative_query_count;
    result.nurbs_geometry_tolerance = domain->geometry_tolerance();
    result.nurbs_root_residual_max =
        native_diagnostics.maximum_root_residual;
    if (result.unsafe_label_changing_edges != 0)
        throw std::runtime_error("unsafe native label-changing edge remains");
    if (result.maximum_subdivision_depth > 4)
        throw std::runtime_error("primary NURBS query exceeded depth four");
    if (result.maximum_targeted_retry_subdivision_depth > 10)
        throw std::runtime_error("targeted NURBS retry exceeded depth ten");
    if (result.nurbs_root_residual_max
        > result.nurbs_geometry_tolerance) {
        throw std::runtime_error(
            "native NURBS root residual exceeds geometry tolerance");
    }
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const bool numerical_inside = grid_pair.domain_label(n) > 0;
        const auto coordinate = grid.coord(n);
        const Eigen::Vector3d point(
            coordinate[0], coordinate[1], coordinate[2]);
        const bool exact_inside = geometry.exact_inside(point);
        if (numerical_inside)
            ++result.interior_nodes;
        else
            ++result.exterior_nodes;
        if (numerical_inside != exact_inside) {
            ++result.label_mismatches;
            if (result.label_mismatches <= 5) {
                std::cout << "  [label mismatch] x=("
                          << point.x() << ',' << point.y() << ',' << point.z()
                          << ") numerical=" << numerical_inside
                          << " exact=" << exact_inside << '\n';
            }
        }
    }
    if (result.interior_nodes == 0 || result.exterior_nodes == 0)
        throw std::runtime_error("domain labeling did not produce both sides");
    if (result.label_mismatches != 0)
        throw std::runtime_error("native NURBS grid labels are not exact");

    const LaplaceCorrectionSupport3D support =
        build_laplace_correction_support_3d(
            grid_pair, "neumann exterior-zero-trace 3D geometry readiness");
    result.crossings = static_cast<int>(support.crossing_ops.size());
    result.correction_nodes = static_cast<int>(support.correction_nodes.size());
    if (result.crossings == 0
        || support.restrict_stencils.size()
           != static_cast<std::size_t>(geometry.correction_interface.num_points())) {
        throw std::runtime_error("failed to build fixed crossing/restrict routes");
    }

    std::set<std::tuple<int, int, int>> crossing_events;
    bool all_native_owners_exact = true;
    for (const LaplaceCrossingCorrectionOp& op : support.crossing_ops) {
        const std::tuple<int, int, int> event_key{
            std::min(op.rhs_node, op.correction_node),
            std::max(op.rhs_node, op.correction_node),
            has_grid_edge_event_3d(op)
                ? op.grid_edge_event.ordinal : -1};
        if (!crossing_events.insert(event_key).second)
            continue;
        const P2CrossingOwner3D owner =
            laplace_crossing_owner_3d(grid_pair, op);
        if (owner.status == P2CrossingOwnerStatus3D::ExactIntersection)
            ++result.exact_crossings;
        else if (owner.status == P2CrossingOwnerStatus3D::GapFallback)
            ++result.gap_crossings;
        else
            ++result.endpoint_crossings;

        // surface_dof_for_crossing uses legacy triangle/barycentric ownership
        // exactly when the owner has no native NURBS patch.
        if (owner.nurbs_patch_index < 0)
            ++result.triangle_fallback_crossings;
        if (owner.status != P2CrossingOwnerStatus3D::ExactIntersection
            || owner.nurbs_patch_index < 0
            || !owner.nurbs_parameter.allFinite()
            || !owner.crossing_point.allFinite()) {
            all_native_owners_exact = false;
        }
    }
    if (result.gap_crossings != 0)
        throw std::runtime_error(
            "native readiness crossing owner used gap fallback");
    if (result.triangle_fallback_crossings != 0)
        throw std::runtime_error(
            "native readiness crossing owner used legacy triangle/barycentric fallback");
    if (!all_native_owners_exact)
        throw std::runtime_error(
            "native readiness crossing owner is not exact");

    LaplaceQuadraticPatchCenterSpread3D spread(grid_pair, 0.0);
    LaplaceFftBulkSolverZfft3D bulk(
        grid, ZfftBcType::Dirichlet, 0.0, 2);
    LaplaceQuadraticPatchCenterRestrict3D restrict_op(grid_pair, 2);
    LaplacePotentialEval3D potentials(spread, bulk, restrict_op);

    std::vector<LaplaceJumpData3D> jumps(
        static_cast<std::size_t>(geometry.correction_interface.num_points()));
    for (LaplaceJumpData3D& jump : jumps) {
        jump.u_jump = 1.0;
        jump.un_jump = 0.0;
        jump.rhs_derivs = Eigen::VectorXd::Zero(1);
    }
    const LaplacePotentialEvalResult3D probe = potentials.evaluate(
        jumps, Eigen::VectorXd::Zero(grid.num_dofs()));
    for (int q = 0; q < probe.u_avg.size(); ++q) {
        const double interior_trace = probe.u_avg[q] + 0.5;
        const double exterior_trace = probe.u_avg[q] - 0.5;
        result.constant_A1_linf = std::max(
            result.constant_A1_linf, std::abs(1.0 - interior_trace));
        result.constant_exterior_trace_linf = std::max(
            result.constant_exterior_trace_linf, std::abs(exterior_trace));
        result.constant_interior_trace_linf = std::max(
            result.constant_interior_trace_linf, std::abs(interior_trace - 1.0));
    }
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const double expected = grid_pair.domain_label(n) > 0 ? 1.0 : 0.0;
        result.constant_bulk_linf = std::max(
            result.constant_bulk_linf, std::abs(probe.u_bulk[n] - expected));
    }
    if (!std::isfinite(result.constant_A1_linf)
        || !std::isfinite(result.constant_bulk_linf)) {
        throw std::runtime_error("constant value-jump readiness probe produced NaN/Inf");
    }

    PanelCenterHarmonicJetKFBI3D harmonic_pipeline(
        grid,
        grid_pair,
        geometry.native_surface,
        geometry.correction_triangles,
        geometry.geometry_triangles,
        surface_dofs,
        cauchy_stencils,
        false,
        (solve_selection != SolveSelection3D::DirichletNormalOnly
         && trace_restrict_uses_crossing_owner(
                neumann_trace_restrict_mode))
            || (solve_selection != SolveSelection3D::NeumannOnly
                && trace_restrict_uses_crossing_owner(
                       dirichlet_normal_restrict_mode)),
        solve_selection != SolveSelection3D::DirichletNormalOnly
                && neumann_trace_restrict_mode
                       == ExteriorNormalRestrictMode3D::
                              GlobalCubicExteriorBranchCrossingOwner
            ? local_restrict_value_count : 0,
        solve_selection != SolveSelection3D::DirichletNormalOnly
                && neumann_trace_restrict_mode
                       == ExteriorNormalRestrictMode3D::
                              GlobalCubicExteriorBranchCrossingOwner
            ? local_restrict_normal_count : 0,
        solve_selection != SolveSelection3D::DirichletNormalOnly
            && neumann_trace_restrict_mode
                   == ExteriorNormalRestrictMode3D::
                          GlobalCubicExteriorBranchCrossingOwner,
        (solve_selection != SolveSelection3D::DirichletNormalOnly
         && trace_restrict_uses_shared_quadratic(
                neumann_trace_restrict_mode))
            || (solve_selection != SolveSelection3D::NeumannOnly
                && trace_restrict_uses_shared_quadratic(
                       dirichlet_normal_restrict_mode)),
        (solve_selection != SolveSelection3D::DirichletNormalOnly
         && !trace_restrict_uses_all_event_path(
                neumann_trace_restrict_mode))
            || (solve_selection != SolveSelection3D::NeumannOnly
                && !trace_restrict_uses_all_event_path(
                       dirichlet_normal_restrict_mode)),
        ((solve_selection != SolveSelection3D::DirichletNormalOnly
          && trace_restrict_uses_topology_affine_cubic(
                 neumann_trace_restrict_mode))
         || (solve_selection != SolveSelection3D::NeumannOnly
             && trace_restrict_uses_topology_affine_cubic(
                    dirichlet_normal_restrict_mode)))
            ? app3d::SharedQuadraticNormalProfile3D::TopologyAffineCubic
            : app3d::SharedQuadraticNormalProfile3D::LegacySplitQuadratic,
        (solve_selection != SolveSelection3D::DirichletNormalOnly
         && neumann_trace_restrict_mode
                == ExteriorNormalRestrictMode3D::Q27Cover3AllEventCauchy)
            || (solve_selection != SolveSelection3D::NeumannOnly
                && dirichlet_normal_restrict_mode
                       == ExteriorNormalRestrictMode3D::
                              Q27Cover3AllEventCauchy),
        (solve_selection != SolveSelection3D::DirichletNormalOnly
         && neumann_trace_restrict_mode
                == ExteriorNormalRestrictMode3D::Q64Cover4AllEventCauchy)
            || (solve_selection != SolveSelection3D::NeumannOnly
                && dirichlet_normal_restrict_mode
                       == ExteriorNormalRestrictMode3D::
                              Q64Cover4AllEventCauchy),
        selected_native_endpoint_support_path());
    const std::vector<double> panel_cauchy_conditions =
        harmonic_pipeline.cauchy_condition_values();
    if (!panel_cauchy_conditions.empty()) {
        const ConditionStatistics3D condition_statistics =
            summarize_conditions(panel_cauchy_conditions);
        result.cauchy_condition_median = condition_statistics.median;
        result.cauchy_condition_p95 = condition_statistics.p95;
        result.cauchy_condition_max = condition_statistics.maximum;
    }
    const std::vector<double> crossing_conditions =
        harmonic_pipeline.crossing_local_condition_values();
    result.crossing_cauchy_unique_plans =
        harmonic_pipeline.crossing_local_unique_plan_count();
    result.crossing_cauchy_svd_count =
        harmonic_pipeline.crossing_local_svd_count();
    if (!crossing_conditions.empty()) {
        const ConditionStatistics3D crossing_condition_statistics =
            summarize_conditions(crossing_conditions);
        result.crossing_cauchy_condition_median =
            crossing_condition_statistics.median;
        result.crossing_cauchy_condition_p95 =
            crossing_condition_statistics.p95;
        result.crossing_cauchy_condition_max =
            crossing_condition_statistics.maximum;
    }
    if ((solve_selection != SolveSelection3D::DirichletNormalOnly
         && trace_restrict_uses_all_event_path(
                neumann_trace_restrict_mode))
        || (solve_selection != SolveSelection3D::NeumannOnly
            && trace_restrict_uses_all_event_path(
                   dirichlet_normal_restrict_mode))) {
        result.shared_quadratic_restrict =
            harmonic_pipeline.shared_quadratic_restrict_diagnostics();
    }
    const app3d::GeometricFeatureTraceSummary3D feature_trace_summary =
        harmonic_pipeline.feature_trace_summary();
    result.feature_trace_segments = feature_trace_summary.feature_segments;
    result.feature_trace_vertices = feature_trace_summary.feature_vertices;
    result.feature_trace_active_maps = feature_trace_summary.active_maps;
    result.feature_trace_edge_maps = feature_trace_summary.edge_maps;
    result.feature_trace_vertex_maps = feature_trace_summary.vertex_maps;
    result.feature_trace_maximum_features_per_map =
        feature_trace_summary.maximum_features_per_map;
    result.feature_trace_maximum_sheets_per_map =
        feature_trace_summary.maximum_sheets_per_map;
    result.feature_trace_constraint_residual_max =
        feature_trace_summary.maximum_constraint_residual;
    const std::vector<double> feature_trace_conditions =
        harmonic_pipeline.feature_trace_condition_values();
    if (!feature_trace_conditions.empty()) {
        const ConditionStatistics3D feature_condition_statistics =
            summarize_conditions(feature_trace_conditions);
        result.feature_trace_condition_median =
            feature_condition_statistics.median;
        result.feature_trace_condition_p95 =
            feature_condition_statistics.p95;
        result.feature_trace_condition_max =
            feature_condition_statistics.maximum;
    }
    const Eigen::VectorXd constant_value =
        Eigen::VectorXd::Ones(harmonic_pipeline.surface_size());
    const Eigen::VectorXd zero_normal =
        Eigen::VectorXd::Zero(harmonic_pipeline.surface_size());
    const DensityIterationMode3D density_mode =
        selected_density_iteration_mode();
    const bool global_neumann_restrict =
        solve_selection != SolveSelection3D::DirichletNormalOnly
        && neumann_trace_restrict_mode
               == ExteriorNormalRestrictMode3D::
                      GlobalCubicExteriorBranchCrossingOwner;
    const bool use_neumann_feature_value_jet =
        neumann_restrict_value_jet_mode
        == NeumannRestrictValueJetMode3D::FeatureConstrained;
    const HarmonicJetField3D harmonic_probe =
        density_mode == DensityIterationMode3D::ReducedCoefficients
        ? harmonic_pipeline.evaluate_crossing_local(
              constant_value, zero_normal,
              [](int) { return 1.0; },
              [](int) { return 0.0; },
              use_neumann_feature_value_jet,
              global_neumann_restrict,
              global_neumann_restrict)
        : harmonic_pipeline.evaluate(constant_value, zero_normal);
    if (solve_selection != SolveSelection3D::DirichletNormalOnly) {
        result.harmonic_constant_exterior_trace_linf =
            harmonic_pipeline.exterior_trace(
                harmonic_probe, constant_value, zero_normal,
                neumann_trace_restrict_mode).lpNorm<Eigen::Infinity>();
        result.harmonic_constant_interior_trace_linf =
            (harmonic_pipeline.interior_trace(
                 harmonic_probe, constant_value, zero_normal,
                 neumann_trace_restrict_mode).array() - 1.0)
            .matrix().lpNorm<Eigen::Infinity>();
    }
    if (solve_selection != SolveSelection3D::NeumannOnly) {
        result.harmonic_constant_exterior_normal_linf =
            harmonic_pipeline.exterior_normal_trace(
                harmonic_probe, constant_value, zero_normal,
                dirichlet_normal_restrict_mode).lpNorm<Eigen::Infinity>();
        result.harmonic_constant_interior_normal_linf =
            harmonic_pipeline.interior_normal_trace(
                harmonic_probe, constant_value, zero_normal,
                dirichlet_normal_restrict_mode).lpNorm<Eigen::Infinity>();
    }
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const double expected = grid_pair.domain_label(n) > 0 ? 1.0 : 0.0;
        result.harmonic_constant_bulk_linf = std::max(
            result.harmonic_constant_bulk_linf,
            std::abs(harmonic_probe.potential[n] - expected));
    }
    if (!std::isfinite(result.harmonic_constant_exterior_trace_linf)
        || !std::isfinite(result.harmonic_constant_exterior_normal_linf)
        || !std::isfinite(result.harmonic_constant_bulk_linf)) {
        throw std::runtime_error(
            "harmonic-jet constant probe produced NaN/Inf");
    }
    result.density_iteration_mode =
        density_iteration_mode_name(density_mode);
    if (density_mode == DensityIterationMode3D::ReducedCoefficients) {
        const app3d::NativeDensityResolution3D density_choice =
            app3d::choose_native_density_resolution_3d(
                geometry.native_surface, h);
        const int override_coefficients =
            optional_density_coefficient_override();
        int observability_cap = std::numeric_limits<int>::max();
        for (const SurfacePatchInfo& patch : surface_dofs.patches)
            observability_cap = std::min(
                observability_cap, std::min(patch.nu, patch.nv));
        if (observability_cap < 4)
            throw std::runtime_error(
                "surface sampling cannot observe a cubic density basis");
        const int coefficients_per_direction = override_coefficients > 0
            ? override_coefficients
            : std::min(density_choice.coefficients_per_direction,
                       observability_cap);
        if (coefficients_per_direction > observability_cap) {
            throw std::invalid_argument(
                "KFBIM_3D_DENSITY_COEFFICIENTS exceeds the per-patch "
                "panel-center observability cap");
        }
        result.automatic_density = override_coefficients == 0;
        result.density_coefficients_per_direction =
            coefficients_per_direction;
        result.density_observability_cap = observability_cap;
        result.density_orientation_integral =
            density_choice.orientation_integral;
        result.density_predicted_crossings =
            density_choice.predicted_crossings;
        result.density_target_crossings_per_cell =
            density_choice.target_crossings_per_cell;

        std::unique_ptr<NativeDensityTransfer3D> value_transfer;
        std::unique_ptr<NativeDensityTransfer3D> normal_transfer;
        const NeumannEdgeJumpJetMode3D selected_neumann_feature_mode =
            selected_neumann_edge_jump_jet_mode();
        const bool use_neumann_topology_affine_density =
               selected_neumann_feature_mode
                   == NeumannEdgeJumpJetMode3D::TopologyAffineLocalSvd
            || selected_neumann_feature_mode
                   == NeumannEdgeJumpJetMode3D::
                          TopologyAmbientGradientLocalSvd;
        const bool use_dirichlet_analytic_affine_density =
            solve_selection != SolveSelection3D::NeumannOnly
            && selected_dirichlet_jump_space_mode()
                   == DirichletJumpSpaceMode3D::AnalyticJ0AffineJ1;
        const app3d::NativeDensityReductionBackend3D value_backend =
            use_neumann_topology_affine_density
            ? app3d::NativeDensityReductionBackend3D::TopologyBase
            : app3d::NativeDensityReductionBackend3D::Legacy;
        const app3d::NativeDensityReductionBackend3D normal_backend =
            (use_neumann_topology_affine_density
             || use_dirichlet_analytic_affine_density)
            ? app3d::NativeDensityReductionBackend3D::TopologyBase
            : app3d::NativeDensityReductionBackend3D::Legacy;
        const bool need_edge_normal_space =
            solve_selection != SolveSelection3D::DirichletNormalOnly
            && selected_neumann_edge_jump_jet_mode()
                   != NeumannEdgeJumpJetMode3D::Disabled;
        if (solve_selection != SolveSelection3D::DirichletNormalOnly) {
            value_transfer = std::make_unique<NativeDensityTransfer3D>(
                geometry.native_surface, surface_dofs,
                coefficients_per_direction,
                app3d::NativeDensityField3D::ValueTrace,
                value_backend);
            const auto& density = value_transfer->density();
            result.value_density_raw_coefficients =
                density.raw_coefficient_count();
            result.value_density_c0_coefficients =
                density.c0_coefficient_count();
            result.value_density_reduced_coefficients =
                density.reduced_coefficient_count();
            result.value_density_c0_seams =
                density.strong_c0_seam_count();
            result.value_density_c1_seams =
                density.weak_c1_seam_count();
            result.value_density_constraint_residual =
                density.reduction_constraint_residual();
            result.value_density_constant_error = std::max(
                density.constant_reproduction_error(),
                value_transfer->constant_sample_error());
            result.value_projection_condition =
                value_transfer->projection().condition();
            result.value_projection_pb_error =
                value_transfer->projection().pb_max_error();
            result.value_coordinate_condition =
                value_transfer->coordinate_condition();
            if (use_neumann_topology_affine_density) {
                harmonic_pipeline.bind_direct_coefficient_value_density(
                    density);
            }
        }
        if (solve_selection != SolveSelection3D::NeumannOnly
            || need_edge_normal_space) {
            normal_transfer = std::make_unique<NativeDensityTransfer3D>(
                geometry.native_surface, surface_dofs,
                coefficients_per_direction,
                app3d::NativeDensityField3D::NormalTrace,
                normal_backend);
            const auto& density = normal_transfer->density();
            result.normal_density_raw_coefficients =
                density.raw_coefficient_count();
            result.normal_density_c0_coefficients =
                density.c0_coefficient_count();
            result.normal_density_reduced_coefficients =
                density.reduced_coefficient_count();
            result.normal_density_broken_seams =
                density.broken_seam_count();
            result.normal_density_c1_seams =
                density.weak_c1_seam_count();
            result.normal_density_constraint_residual =
                density.reduction_constraint_residual();
            result.normal_density_constant_error = std::max(
                density.constant_reproduction_error(),
                normal_transfer->constant_sample_error());
            result.normal_projection_condition =
                normal_transfer->projection().condition();
            result.normal_projection_pb_error =
                normal_transfer->projection().pb_max_error();
            result.normal_coordinate_condition =
                normal_transfer->coordinate_condition();
            if (use_neumann_topology_affine_density
                || use_dirichlet_analytic_affine_density) {
                harmonic_pipeline.bind_direct_coefficient_normal_density(
                    density);
            }
        }
        if (solve_selection != SolveSelection3D::DirichletNormalOnly) {
            result.neumann = run_neumann_coefficient_case(
                grid, grid_pair, harmonic_pipeline, *value_transfer,
                normal_transfer.get(), transform,
                gmres_max_iterations, gmres_tolerance,
                neumann_compatibility_mode,
                neumann_trace_restrict_mode,
                neumann_restrict_value_jet_mode);
        }
        if (solve_selection != SolveSelection3D::NeumannOnly) {
            result.dirichlet_normal =
                run_dirichlet_normal_coefficient_case(
                    grid, grid_pair, harmonic_pipeline, *normal_transfer,
                    transform, gmres_max_iterations, gmres_tolerance,
                    dirichlet_normal_restrict_mode);
        }
    } else {
        if (solve_selection != SolveSelection3D::DirichletNormalOnly) {
            result.neumann = run_neumann_case(
                grid, grid_pair, harmonic_pipeline, transform,
                gmres_max_iterations,
                gmres_tolerance,
                neumann_compatibility_mode,
                neumann_trace_restrict_mode);
        }
        if (solve_selection != SolveSelection3D::NeumannOnly) {
            result.dirichlet_normal = run_dirichlet_normal_case(
                grid, grid_pair, harmonic_pipeline, transform,
                gmres_max_iterations,
                gmres_tolerance,
                dirichlet_normal_restrict_mode);
        }
    }

    std::cout << "[ready] " << geometry.name << " - " << geometry.description << '\n'
              << "domain_label_mode=nurbs_barrier_components barriers="
              << result.barrier_x << '/' << result.barrier_y << '/'
              << result.barrier_z << " components=" << result.grid_components
              << '/' << result.box_exterior_components
              << " queries=" << result.representative_queries
              << " root_residual=" << result.nurbs_root_residual_max
              << " gap_crossings=" << result.gap_crossings
              << " triangle_fallback_crossings="
              << result.triangle_fallback_crossings
              << " correction_safe_edges=" << result.correction_safe_edges
              << " unsafe_label_changing_edges="
              << result.unsafe_label_changing_edges
              << " targeted_retries=" << result.targeted_retries << '/'
              << result.targeted_retries_resolved << '/'
              << result.targeted_retries_unsafe
              << " targeted_retry_depth="
              << result.maximum_targeted_retry_subdivision_depth << '\n'
              << "  interface/barrier="
              << result.interface_x << '/' << result.interface_y << '/'
              << result.interface_z << " / "
              << result.barrier_x << '/' << result.barrier_y << '/'
              << result.barrier_z
              << " multi=" << result.multi_crossing_edges
              << " even/odd=" << result.even_parity_interface_edges << '/'
              << result.odd_parity_interface_edges
              << " ambiguous/label-changing/fallback="
              << result.ambiguous_parity_edges << '/'
              << result.ambiguous_label_changing_edges << '/'
              << result.endpoint_parity_fallbacks
              << " component_toggles=" << result.component_parity_toggles
              << '\n'
              << "  NURBS query elements=" << result.acceleration_leaves
              << " max_extent=" << result.maximum_query_element_extent
              << " local_depth=" << result.maximum_subdivision_depth
              << " certificate_boxes/depth="
              << result.terminal_certificate_boxes << '/'
              << result.maximum_terminal_certificate_depth
              << " closest attempts/iterations/recovered/misses/failures="
              << result.closest_point_attempts << '/'
              << result.closest_point_iterations << '/'
              << result.closest_point_roots_recovered << '/'
              << result.closest_point_terminal_misses << '/'
              << result.closest_point_failures << '\n'
              << "  sample seeds candidates/accepted/recovered/max="
              << result.sample_seed_candidates << '/'
              << result.sample_seeds_accepted << '/'
              << result.sample_seed_roots_recovered << '/'
              << result.maximum_sample_seeds_per_element
              << " stationary attempts/converged/witnesses/protected="
              << result.stationary_solve_attempts << '/'
              << result.stationary_solve_converged << '/'
              << result.stationary_witnesses << '/'
              << result.stationary_protected_root_pairs
              << " clusters/nonG1/high_degree="
              << result.ambiguous_root_clusters << '/'
              << result.non_g1_topology_merges << '/'
              << result.high_degree_fallbacks << '\n'
              << "  panel-center surface patches/dofs="
              << result.surface_patches << '/' << result.surface_dofs
              << " area=" << result.surface_dof_area
              << " area_rel_err=" << result.surface_area_relative_error << '\n'
              << "  density iteration=" << result.density_iteration_mode
              << " automatic/ncoef=" << result.automatic_density << '/'
              << result.density_coefficients_per_direction
              << " observability_cap=" << result.density_observability_cap
              << " orientation_integral/predicted_crossings/target="
              << result.density_orientation_integral << '/'
              << result.density_predicted_crossings << '/'
              << result.density_target_crossings_per_cell << '\n'
              << "  value density raw/C0/reduced="
              << result.value_density_raw_coefficients << '/'
              << result.value_density_c0_coefficients << '/'
              << result.value_density_reduced_coefficients
              << " C0/C1 seams=" << result.value_density_c0_seams << '/'
              << result.value_density_c1_seams
              << " constraint/constant/PB="
              << result.value_density_constraint_residual << '/'
              << result.value_density_constant_error << '/'
              << result.value_projection_pb_error
              << " projection/coordinate cond="
              << result.value_projection_condition << '/'
              << result.value_coordinate_condition << '\n'
              << "  normal density raw/C0/reduced="
              << result.normal_density_raw_coefficients << '/'
              << result.normal_density_c0_coefficients << '/'
              << result.normal_density_reduced_coefficients
              << " broken/C1 seams="
              << result.normal_density_broken_seams << '/'
              << result.normal_density_c1_seams
              << " constraint/constant/PB="
              << result.normal_density_constraint_residual << '/'
              << result.normal_density_constant_error << '/'
              << result.normal_projection_pb_error
              << " projection/coordinate cond="
              << result.normal_projection_condition << '/'
              << result.normal_coordinate_condition << '\n'
              << "  nearest surface spacing/h min/mean/max="
              << result.surface_spacing_min_over_h << '/'
              << result.surface_spacing_mean_over_h << '/'
              << result.surface_spacing_max_over_h << '\n'
              << "  Cauchy policy=" << result.cauchy_policy
              << " requested values/normals="
              << result.cauchy_value_neighbors << '/'
              << result.cauchy_derivative_neighbors
              << " actual values min/max="
              << result.cauchy_value_neighbors_min << '/'
              << result.cauchy_value_neighbors_max
              << " normals min/max="
              << result.cauchy_derivative_neighbors_min << '/'
              << result.cauchy_derivative_neighbors_max
              << " radius/h mean/max=" << result.cauchy_radius_mean_over_h
              << '/' << result.cauchy_radius_max_over_h
              << " selected patches min/max="
              << result.cauchy_incident_patches_min << '/'
              << result.cauchy_incident_patches_max
              << " imbalance value/normal max="
              << result.cauchy_value_patch_imbalance_max << '/'
              << result.cauchy_derivative_patch_imbalance_max << '\n'
              << "  Cauchy condition median/p95/max="
              << result.cauchy_condition_median << '/'
              << result.cauchy_condition_p95 << '/'
              << result.cauchy_condition_max << '\n'
              << "  crossing-local Cauchy condition median/p95/max="
              << result.crossing_cauchy_condition_median << '/'
              << result.crossing_cauchy_condition_p95 << '/'
              << result.crossing_cauchy_condition_max << '\n'
              << "  crossing-local unique plans/SVDs="
              << result.crossing_cauchy_unique_plans << '/'
              << result.crossing_cauchy_svd_count << '\n'
              << "  shared-quadratic side-plans/gridline-crossings="
              << result.shared_quadratic_restrict.side_plans << '/'
              << result.shared_quadratic_restrict.unique_gridline_crossings
              << " wrong-side/queries/multi-root="
              << result.shared_quadratic_restrict.wrong_side_nodes << '/'
              << result.shared_quadratic_restrict.segment_queries << '/'
              << result.shared_quadratic_restrict.multiple_root_selections
              << " endpoint="
              << result.shared_quadratic_restrict.trace_endpoint_selections
              << " feature/unreliable roots="
              << result.shared_quadratic_restrict.feature_root_selections
              << '/'
              << result.shared_quadratic_restrict.unreliable_root_selections
              << " fallbacks(segment/G1)="
              << result.shared_quadratic_restrict.segment_fallbacks << '/'
              << result.shared_quadratic_restrict.no_g1_gridline_fallbacks
              << " gridline ties="
              << result.shared_quadratic_restrict.gridline_ties
              << " interpolation/Cauchy cond max="
              << result.shared_quadratic_restrict.interpolation_condition_max
              << '/'
              << result.shared_quadratic_restrict.cauchy_condition_max
              << " gridline distance/h max="
              << result.shared_quadratic_restrict
                     .gridline_distance_max_over_h << '\n'
              << "  value-trace fit=" << result.cauchy_value_trace_fit
              << " feature segments/vertices="
              << result.feature_trace_segments << '/'
              << result.feature_trace_vertices
              << " active edge/vertex maps="
              << result.feature_trace_active_maps << ' '
              << result.feature_trace_edge_maps << '/'
              << result.feature_trace_vertex_maps
              << " max features/sheets per map="
              << result.feature_trace_maximum_features_per_map << '/'
              << result.feature_trace_maximum_sheets_per_map
              << " constraint residual max="
              << result.feature_trace_constraint_residual_max
              << " condition median/p95/max="
              << result.feature_trace_condition_median << '/'
              << result.feature_trace_condition_p95 << '/'
              << result.feature_trace_condition_max << '\n'
              << "  P2 correction panels/dofs=" << result.correction_panels
              << '/' << result.correction_dofs
              << " crossing_panels=" << result.crossing_panels
              << " area(correction/full)=" << result.correction_area
              << '/' << result.crossing_area << '\n'
              << "  features edges/vertices=" << result.feature_edges
              << '/' << result.feature_vertices
              << " grid inside/outside=" << result.interior_nodes
              << '/' << result.exterior_nodes
              << " label_mismatches=" << result.label_mismatches
              << " box_margin/h=" << result.min_box_margin_over_h << '\n'
              << "  fixed routes crossing_ops/correction_nodes="
              << result.crossings << '/' << result.correction_nodes
              << " owners exact/gap/endpoint=" << result.exact_crossings
              << '/' << result.gap_crossings
              << '/' << result.endpoint_crossings << '\n'
              << "  constant [u]=1 probe: |A1|inf="
              << result.constant_A1_linf
              << " exterior_trace=" << result.constant_exterior_trace_linf
              << " interior_trace_err=" << result.constant_interior_trace_linf
              << " bulk_err=" << result.constant_bulk_linf << '\n'
              << "  selected-degree harmonic-jet [u]=1 probe: "
                 "exterior/interior trace="
              << result.harmonic_constant_exterior_trace_linf << '/'
              << result.harmonic_constant_interior_trace_linf
              << " exterior/interior normal="
              << result.harmonic_constant_exterior_normal_linf << '/'
              << result.harmonic_constant_interior_normal_linf
              << " bulk_err=" << result.harmonic_constant_bulk_linf << '\n';
    if (solve_selection != SolveSelection3D::DirichletNormalOnly) {
        std::cout << "  [neumann] gmres/sample_condition/iterations="
              << result.neumann.converged << '/'
              << result.neumann.physical_converged << '/'
              << result.neumann.iterations
              << " gmres=" << result.neumann.gmres_relative_residual
              << " operator=" << result.neumann.operator_residual_linf
              << " exterior_trace=" << result.neumann.exterior_condition_linf
              << " compatibility_mode="
              << result.neumann.compatibility_mode
              << " trace_restrict="
              << result.neumann.trace_restrict_mode
              << " restrict_value_jet="
              << result.neumann.restrict_value_jet
              << " trace_sampling="
              << result.neumann.trace_sampling
              << " density_coordinates="
              << result.neumann.density_coordinates
              << " border_solver="
              << result.neumann.border_solver
              << " edge_jump_jet="
              << result.neumann.edge_jump_jet
              << " edge_features/rows/rank/reduced_dofs="
              << result.neumann.edge_feature_edges << '/'
              << result.neumann.edge_constraint_rows << '/'
              << result.neumann.edge_constraint_rank << '/'
              << result.neumann.edge_reduced_dofs
              << " edge_constraint/nullspace="
              << result.neumann.edge_constraint_residual_linf << '/'
              << result.neumann.edge_nullspace_residual_linf
              << " edge_normal_fit="
              << result.neumann.edge_normal_fit_linf
              << " edge_target_projection="
              << result.neumann.edge_target_projection_linf
              << " feature_target_projection_l2/relative/cert/exact="
              << result.neumann.feature_target_projection_l2 << '/'
              << result.neumann.feature_target_projection_relative_l2 << '/'
              << result.neumann
                     .feature_target_reachable_certification_linf << '/'
              << result.neumann.feature_target_constraints_exact
              << " reduction=" << result.neumann.reduction_scheme
              << " particular_solver=" << result.neumann.particular_solver
              << " topology rows/rank/base/full/pre_mean/reduced/blocks="
              << result.neumann.topology_constraint_rows << '/'
              << result.neumann.topology_constraint_rank << '/'
              << result.neumann.topology_base_coordinates << '/'
              << result.neumann.topology_full_coordinates << '/'
              << result.neumann.topology_pre_mean_coordinates << '/'
              << result.neumann.topology_reduced_coordinates << '/'
              << result.neumann.topology_block_count
              << " topology Cp/CE/PB/closure="
              << result.neumann.topology_particular_residual_linf << '/'
              << result.neumann.topology_homogeneous_residual_linf << '/'
              << result.neumann.topology_projector_pb_error << '/'
              << result.neumann.topology_trace_closure_linf
              << " topology projector rank/condition="
              << result.neumann.topology_projector_rank << '/'
              << result.neumann.topology_projector_condition
              << " trace samples/dofs/margin/ratio="
              << result.neumann.topology_trace_samples << '/'
              << result.neumann.topology_reduced_coordinates << '/'
              << result.neumann.topology_trace_oversampling_margin << '/'
              << result.neumann.topology_trace_oversampling_ratio
              << " mean_free pivot/moment/observable/condition="
              << result.neumann.mean_free_pivot_index << '/'
              << result.neumann.mean_free_pivot_moment << '/'
              << result.neumann.mean_free_relative_observability << '/'
              << result.neumann.mean_free_coordinate_condition
              << " mean_free particular/homogeneous/P1/intrinsic="
              << result.neumann.mean_free_particular_residual << '/'
              << result.neumann.mean_free_homogeneous_residual << '/'
              << result.neumann.mean_free_constant_projection_linf << '/'
              << result.neumann.mean_free_intrinsic_mean
              << " lambda=" << result.neumann.lagrange_multiplier
              << " flux_correction/total="
              << result.neumann.operator_compatibility_correction << '/'
              << result.neumann.total_flux_correction
              << " border_linf=" << result.neumann.border_column_linf
              << " trace_mean/demeaned="
              << result.neumann.exterior_trace_weighted_mean << '/'
              << result.neumann.exterior_trace_demeaned_linf
              << " closure="
              << result.neumann.augmented_trace_closure_linf
              << " compatibility_residual="
              << result.neumann.compatibility_residual
              << " route_mismatch=" << result.neumann.route_mismatch_linf
              << " density_linf/l2=" << result.neumann.density_linf << '/'
              << result.neumann.density_l2
              << " interior_linf/l2=" << result.neumann.interior_linf << '/'
              << result.neumann.interior_l2
              << " exterior_bulk_linf/l2="
              << result.neumann.exterior_bulk_linf << '/'
              << result.neumann.exterior_bulk_l2
              << " flux_mean_removed/post="
              << result.neumann.compatibility_mean_removed << '/'
              << result.neumann.data_weighted_mean
              << " seconds=" << result.neumann.seconds << '\n';
    }
    if (solve_selection != SolveSelection3D::NeumannOnly) {
        std::cout << "  [dirichlet-normal] gmres/sample_condition/iterations="
              << result.dirichlet_normal.converged << '/'
              << result.dirichlet_normal.physical_converged << '/'
              << result.dirichlet_normal.iterations
              << " gmres="
              << result.dirichlet_normal.gmres_relative_residual
              << " operator="
              << result.dirichlet_normal.operator_residual_linf
              << " exterior_normal="
              << result.dirichlet_normal.exterior_condition_linf
              << " interior_value_res="
              << result.dirichlet_normal.boundary_residual_linf
              << " route_mismatch="
              << result.dirichlet_normal.route_mismatch_linf
              << " jump_space="
              << result.dirichlet_normal.dirichlet_jump_space
              << " known_J0="
              << result.dirichlet_normal.known_value_jump
              << " feature_mode/edges/candidate/retained/discarded="
              << result.dirichlet_normal.edge_jump_jet << '/'
              << result.dirichlet_normal.edge_feature_edges << '/'
              << result.dirichlet_normal.edge_constraint_rows << '/'
              << result.dirichlet_normal.edge_constraint_rank << '/'
              << (result.dirichlet_normal.edge_constraint_rows
                  - result.dirichlet_normal.edge_constraint_rank)
              << " feature_residual/star/edge="
              << result.dirichlet_normal.edge_constraint_residual_linf << '/'
              << result.dirichlet_normal.feature_vertex_residual_linf << '/'
              << result.dirichlet_normal.feature_edge_residual_linf
              << " discarded_feature_defect="
              << result.dirichlet_normal.feature_discarded_residual_linf
              << " projection_leakage="
              << result.dirichlet_normal
                     .trace_projection_leakage_relative_l2
              << " reduction="
              << result.dirichlet_normal.reduction_scheme
              << " particular_solver="
              << result.dirichlet_normal.particular_solver
              << " topology rows/rank/base/full/reduced/blocks="
              << result.dirichlet_normal.topology_constraint_rows << '/'
              << result.dirichlet_normal.topology_constraint_rank << '/'
              << result.dirichlet_normal.topology_base_coordinates << '/'
              << result.dirichlet_normal.topology_full_coordinates << '/'
              << result.dirichlet_normal.topology_reduced_coordinates << '/'
              << result.dirichlet_normal.topology_block_count
              << " topology Cp/CE/PB/closure="
              << result.dirichlet_normal.topology_particular_residual_linf
              << '/'
              << result.dirichlet_normal.topology_homogeneous_residual_linf
              << '/'
              << result.dirichlet_normal.topology_projector_pb_error
              << '/'
              << result.dirichlet_normal.topology_trace_closure_linf
              << " topology projector rank/condition="
              << result.dirichlet_normal.topology_projector_rank << '/'
              << result.dirichlet_normal.topology_projector_condition
              << " trace samples/dofs/margin/ratio="
              << result.dirichlet_normal.topology_trace_samples << '/'
              << result.dirichlet_normal.topology_reduced_coordinates << '/'
              << result.dirichlet_normal.topology_trace_oversampling_margin
              << '/'
              << result.dirichlet_normal.topology_trace_oversampling_ratio
              << " density_linf/l2="
              << result.dirichlet_normal.density_linf << '/'
              << result.dirichlet_normal.density_l2
              << " interior_linf/l2="
              << result.dirichlet_normal.interior_linf << '/'
              << result.dirichlet_normal.interior_l2
              << " exterior_bulk_linf/l2="
              << result.dirichlet_normal.exterior_bulk_linf << '/'
              << result.dirichlet_normal.exterior_bulk_l2
              << " seconds=" << result.dirichlet_normal.seconds << '\n';
    }
    return result;
}

void write_summary(const std::filesystem::path& output_dir,
                   const std::vector<ReadinessResult>& results)
{
    std::filesystem::create_directories(output_dir);
    if (results.empty())
        return;
    auto write_csv = [&](const std::filesystem::path& path,
                         const std::vector<ReadinessResult>& rows) {
        std::ofstream csv = open_output_file(path);
        csv << std::setprecision(17);
        csv << "geometry,cauchy_policy,N,h,correction_panels,correction_dofs,crossing_panels,"
               "feature_edges,feature_vertices,interior_nodes,exterior_nodes,"
               "label_mismatches,crossing_ops,exact_crossings,gap_crossings,"
               "endpoint_crossings,correction_nodes,correction_area,crossing_area,"
               "normal_error,box_margin_over_h,A1_linf,exterior_trace_linf,"
               "interior_trace_error,bulk_error,surface_patches,surface_dofs,"
               "harmonic_exterior_trace_linf,harmonic_interior_trace_linf,"
               "harmonic_exterior_normal_linf,harmonic_interior_normal_linf,"
               "harmonic_bulk_linf,"
               "surface_dof_area,surface_area_relative_error,"
               "surface_spacing_min_over_h,surface_spacing_mean_over_h,"
               "surface_spacing_max_over_h,cauchy_value_neighbors,"
               "cauchy_derivative_neighbors,cauchy_value_neighbors_min,"
               "cauchy_value_neighbors_max,cauchy_derivative_neighbors_min,"
               "cauchy_derivative_neighbors_max,cauchy_radius_max_over_h,"
               "cauchy_radius_mean_over_h,cauchy_incident_patches_min,"
               "cauchy_incident_patches_max,cauchy_value_patch_imbalance_max,"
               "cauchy_derivative_patch_imbalance_max,cauchy_condition_median,"
               "cauchy_condition_p95,cauchy_condition_max,"
               "crossing_cauchy_condition_median,"
               "crossing_cauchy_condition_p95,"
               "crossing_cauchy_condition_max,"
               "crossing_cauchy_unique_plans,crossing_cauchy_svd_count,"
               "shared_quadratic_side_plans,shared_quadratic_gridline_crossings,"
               "shared_quadratic_wrong_side_nodes,shared_quadratic_segment_queries,"
               "shared_quadratic_multiple_roots,shared_quadratic_endpoint_roots,"
               "shared_quadratic_feature_roots,"
               "shared_quadratic_unreliable_roots,shared_quadratic_segment_fallbacks,"
               "shared_quadratic_g1_fallbacks,shared_quadratic_gridline_ties,"
               "shared_quadratic_interpolation_condition_max,"
               "shared_quadratic_cauchy_condition_max,"
               "shared_quadratic_gridline_distance_max_over_h,"
               "feature_trace_segments,feature_trace_vertices,"
               "feature_trace_active_maps,feature_trace_edge_maps,"
               "feature_trace_vertex_maps,"
               "feature_trace_maximum_features_per_map,"
               "feature_trace_maximum_sheets_per_map,"
               "feature_trace_constraint_residual_max,"
               "feature_trace_condition_median,"
               "feature_trace_condition_p95,feature_trace_condition_max,"
               "nurbs_patches,"
               "bezier_elements,acceleration_leaves,maximum_query_element_extent,"
               "candidate_grid_edges,"
               "triangle_seed_hits,triangle_seed_misses_recovered,subdivision_boxes,"
               "newton_attempts,newton_iterations,maximum_subdivision_depth,"
               "terminal_certificate_boxes,maximum_terminal_certificate_depth,"
               "closest_point_attempts,closest_point_iterations,"
               "closest_point_roots_recovered,closest_point_terminal_misses,"
               "closest_point_failures,seam_deduplications,"
               "sample_seed_candidates,sample_seeds_accepted,"
               "sample_seed_roots_recovered,maximum_sample_seeds_per_element,"
               "stationary_solve_attempts,stationary_solve_converged,"
               "stationary_witnesses,stationary_protected_root_pairs,"
               "ambiguous_root_clusters,non_g1_topology_merges,"
               "high_degree_fallbacks,interface_x,interface_y,interface_z,"
               "multi_crossing_edges,even_parity_interface_edges,"
               "odd_parity_interface_edges,ambiguous_parity_edges,"
               "ambiguous_label_changing_edges,targeted_retries,"
               "targeted_retries_resolved,targeted_retries_unsafe,"
               "correction_safe_edges,unsafe_label_changing_edges,"
               "maximum_targeted_retry_subdivision_depth,"
               "endpoint_parity_fallbacks,endpoint_classification_queries,"
               "component_parity_toggles,"
               "barrier_x,barrier_y,barrier_z,grid_components,"
               "box_exterior_components,representative_queries,"
               "nurbs_geometry_tolerance,nurbs_root_residual_max,"
               "triangle_fallback_crossings,density_iteration_mode,"
               "automatic_density,density_coefficients_per_direction,"
               "density_observability_cap,"
               "density_orientation_integral,density_predicted_crossings,"
               "density_target_crossings_per_cell,"
               "value_density_raw_coefficients,value_density_c0_coefficients,"
               "value_density_reduced_coefficients,"
               "normal_density_raw_coefficients,normal_density_c0_coefficients,"
               "normal_density_reduced_coefficients,value_density_c0_seams,"
               "value_density_c1_seams,normal_density_broken_seams,"
               "normal_density_c1_seams,value_density_constraint_residual,"
               "normal_density_constraint_residual,value_density_constant_error,"
               "normal_density_constant_error,value_projection_condition,"
               "normal_projection_condition,value_projection_pb_error,"
               "normal_projection_pb_error,value_coordinate_condition,"
               "normal_coordinate_condition,value_trace_fit\n";
        for (const ReadinessResult& row : rows)
            csv << row.geometry << ',' << row.cauchy_policy << ','
                << row.N << ',' << row.h << ','
                << row.correction_panels << ',' << row.correction_dofs << ','
                << row.crossing_panels << ',' << row.feature_edges << ','
                << row.feature_vertices << ',' << row.interior_nodes << ','
                << row.exterior_nodes << ',' << row.label_mismatches << ','
                << row.crossings << ',' << row.exact_crossings << ','
                << row.gap_crossings << ',' << row.endpoint_crossings << ','
                << row.correction_nodes << ',' << row.correction_area << ','
                << row.crossing_area << ',' << row.normal_error << ','
                << row.min_box_margin_over_h << ',' << row.constant_A1_linf << ','
                << row.constant_exterior_trace_linf << ','
                << row.constant_interior_trace_linf << ','
                << row.constant_bulk_linf << ',' << row.surface_patches << ','
                << row.surface_dofs << ','
                << row.harmonic_constant_exterior_trace_linf << ','
                << row.harmonic_constant_interior_trace_linf << ','
                << row.harmonic_constant_exterior_normal_linf << ','
                << row.harmonic_constant_interior_normal_linf << ','
                << row.harmonic_constant_bulk_linf << ','
                << row.surface_dof_area << ','
                << row.surface_area_relative_error << ','
                << row.surface_spacing_min_over_h << ','
                << row.surface_spacing_mean_over_h << ','
                << row.surface_spacing_max_over_h << ','
                << row.cauchy_value_neighbors << ','
                << row.cauchy_derivative_neighbors << ','
                << row.cauchy_value_neighbors_min << ','
                << row.cauchy_value_neighbors_max << ','
                << row.cauchy_derivative_neighbors_min << ','
                << row.cauchy_derivative_neighbors_max << ','
                << row.cauchy_radius_max_over_h << ','
                << row.cauchy_radius_mean_over_h << ','
                << row.cauchy_incident_patches_min << ','
                << row.cauchy_incident_patches_max << ','
                << row.cauchy_value_patch_imbalance_max << ','
                << row.cauchy_derivative_patch_imbalance_max << ','
                << row.cauchy_condition_median << ','
                << row.cauchy_condition_p95 << ','
                << row.cauchy_condition_max << ','
                << row.crossing_cauchy_condition_median << ','
                << row.crossing_cauchy_condition_p95 << ','
                << row.crossing_cauchy_condition_max << ','
                << row.crossing_cauchy_unique_plans << ','
                << row.crossing_cauchy_svd_count << ','
                << row.shared_quadratic_restrict.side_plans << ','
                << row.shared_quadratic_restrict.unique_gridline_crossings << ','
                << row.shared_quadratic_restrict.wrong_side_nodes << ','
                << row.shared_quadratic_restrict.segment_queries << ','
                << row.shared_quadratic_restrict.multiple_root_selections << ','
                << row.shared_quadratic_restrict.trace_endpoint_selections << ','
                << row.shared_quadratic_restrict.feature_root_selections << ','
                << row.shared_quadratic_restrict.unreliable_root_selections << ','
                << row.shared_quadratic_restrict.segment_fallbacks << ','
                << row.shared_quadratic_restrict.no_g1_gridline_fallbacks << ','
                << row.shared_quadratic_restrict.gridline_ties << ','
                << row.shared_quadratic_restrict.interpolation_condition_max << ','
                << row.shared_quadratic_restrict.cauchy_condition_max << ','
                << row.shared_quadratic_restrict.gridline_distance_max_over_h << ','
                << row.feature_trace_segments << ','
                << row.feature_trace_vertices << ','
                << row.feature_trace_active_maps << ','
                << row.feature_trace_edge_maps << ','
                << row.feature_trace_vertex_maps << ','
                << row.feature_trace_maximum_features_per_map << ','
                << row.feature_trace_maximum_sheets_per_map << ','
                << row.feature_trace_constraint_residual_max << ','
                << row.feature_trace_condition_median << ','
                << row.feature_trace_condition_p95 << ','
                << row.feature_trace_condition_max << ','
                << row.nurbs_patches << ',' << row.bezier_elements << ','
                << row.acceleration_leaves << ','
                << row.maximum_query_element_extent << ','
                << row.candidate_grid_edges << ','
                << row.triangle_seed_hits << ','
                << row.triangle_seed_misses_recovered << ','
                << row.subdivision_boxes << ',' << row.newton_attempts << ','
                << row.newton_iterations << ','
                << row.maximum_subdivision_depth << ','
                << row.terminal_certificate_boxes << ','
                << row.maximum_terminal_certificate_depth << ','
                << row.closest_point_attempts << ','
                << row.closest_point_iterations << ','
                << row.closest_point_roots_recovered << ','
                << row.closest_point_terminal_misses << ','
                << row.closest_point_failures << ','
                << row.seam_deduplications << ','
                << row.sample_seed_candidates << ','
                << row.sample_seeds_accepted << ','
                << row.sample_seed_roots_recovered << ','
                << row.maximum_sample_seeds_per_element << ','
                << row.stationary_solve_attempts << ','
                << row.stationary_solve_converged << ','
                << row.stationary_witnesses << ','
                << row.stationary_protected_root_pairs << ','
                << row.ambiguous_root_clusters << ','
                << row.non_g1_topology_merges << ','
                << row.high_degree_fallbacks << ','
                << row.interface_x << ',' << row.interface_y << ','
                << row.interface_z << ',' << row.multi_crossing_edges << ','
                << row.even_parity_interface_edges << ','
                << row.odd_parity_interface_edges << ','
                << row.ambiguous_parity_edges << ','
                << row.ambiguous_label_changing_edges << ','
                << row.targeted_retries << ','
                << row.targeted_retries_resolved << ','
                << row.targeted_retries_unsafe << ','
                << row.correction_safe_edges << ','
                << row.unsafe_label_changing_edges << ','
                << row.maximum_targeted_retry_subdivision_depth << ','
                << row.endpoint_parity_fallbacks << ','
                << row.endpoint_classification_queries << ','
                << row.component_parity_toggles << ','
                << row.barrier_x << ',' << row.barrier_y << ','
                << row.barrier_z << ',' << row.grid_components << ','
                << row.box_exterior_components << ','
                << row.representative_queries << ','
                << row.nurbs_geometry_tolerance << ','
                << row.nurbs_root_residual_max << ','
                << row.triangle_fallback_crossings << ','
                << row.density_iteration_mode << ','
                << (row.automatic_density ? 1 : 0) << ','
                << row.density_coefficients_per_direction << ','
                << row.density_observability_cap << ','
                << row.density_orientation_integral << ','
                << row.density_predicted_crossings << ','
                << row.density_target_crossings_per_cell << ','
                << row.value_density_raw_coefficients << ','
                << row.value_density_c0_coefficients << ','
                << row.value_density_reduced_coefficients << ','
                << row.normal_density_raw_coefficients << ','
                << row.normal_density_c0_coefficients << ','
                << row.normal_density_reduced_coefficients << ','
                << row.value_density_c0_seams << ','
                << row.value_density_c1_seams << ','
                << row.normal_density_broken_seams << ','
                << row.normal_density_c1_seams << ','
                << row.value_density_constraint_residual << ','
                << row.normal_density_constraint_residual << ','
                << row.value_density_constant_error << ','
                << row.normal_density_constant_error << ','
                << row.value_projection_condition << ','
                << row.normal_projection_condition << ','
                << row.value_projection_pb_error << ','
                << row.normal_projection_pb_error << ','
                << row.value_coordinate_condition << ','
                << row.normal_coordinate_condition << ','
                << row.cauchy_value_trace_fit << '\n';
    };
    write_csv(output_dir / "geometry_readiness.csv", results);
    std::set<int> levels;
    for (const ReadinessResult& row : results)
        levels.insert(row.N);
    for (int level : levels) {
        std::vector<ReadinessResult> level_rows;
        for (const ReadinessResult& row : results) {
            if (row.N == level)
                level_rows.push_back(row);
        }
        write_csv(output_dir / (
            "geometry_readiness_N" + std::to_string(level) + ".csv"),
            level_rows);
    }
}

double observed_order(double coarse_error,
                      double fine_error,
                      double coarse_h,
                      double fine_h)
{
    if (!(coarse_error > 0.0) || !(fine_error > 0.0)
        || !(coarse_h > fine_h)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log(coarse_error / fine_error)
         / std::log(coarse_h / fine_h);
}

void write_solve_summaries(const std::filesystem::path& output_dir,
                           const std::vector<ReadinessResult>& results,
                           SolveSelection3D solve_selection)
{
    using MetricSelector =
        std::function<const SolveMetrics3D&(const ReadinessResult&)>;
    auto write_csv = [&](const std::filesystem::path& path,
                         const MetricSelector& select_metric) {
        std::vector<const ReadinessResult*> rows;
        rows.reserve(results.size());
        for (const ReadinessResult& result : results)
            rows.push_back(&result);
        std::sort(rows.begin(), rows.end(),
                  [](const ReadinessResult* a, const ReadinessResult* b) {
                      return std::tie(a->geometry, a->N)
                           < std::tie(b->geometry, b->N);
                  });

        std::ofstream csv = open_output_file(path);
        csv << std::setprecision(17);
        csv << "geometry,cauchy_policy,N,h,dofs,density_iteration_mode,"
               "density_coefficients_per_direction,formulation,"
               "approximation_scheme,cauchy_degree,"
               "restrict_grid_degree,restrict_normal_degree,"
               "compatibility_mode,trace_restrict_mode,restrict_value_jet,"
               "trace_sampling,density_coordinates,border_solver,"
               "edge_jump_jet,edge_feature_edges,edge_constraint_rows,"
               "edge_constraint_rank,edge_reduced_dofs,"
               "edge_constraint_residual_linf,"
               "edge_nullspace_residual_linf,edge_normal_fit_linf,"
               "edge_target_projection_linf,"
               "feature_target_requested_linf,feature_target_requested_l2,"
               "feature_target_projected_linf,feature_target_projected_l2,"
               "feature_target_projection_l2,"
               "feature_target_projection_relative_l2,"
               "feature_target_reachable_certification_linf,"
               "feature_target_projection_components,"
               "feature_target_constraints_exact,"
               "reduction_scheme,particular_solver,"
               "topology_constraint_rows,topology_constraint_rank,"
               "topology_full_coordinates,topology_reduced_coordinates,"
               "topology_block_count,topology_particular_residual_linf,"
               "topology_homogeneous_residual_linf,"
               "topology_projector_rank,topology_projector_condition,"
               "topology_projector_pb_error,topology_trace_closure_linf,"
               "iterations,converged,physical_converged,"
               "seconds,gmres_tolerance,"
               "gmres_relative_residual,operator_residual_linf,"
               "exterior_condition_linf,lagrange_multiplier,"
               "operator_compatibility_correction,total_flux_correction,"
               "border_column_linf,exterior_trace_weighted_mean,"
               "exterior_trace_demeaned_linf,augmented_trace_closure_linf,"
               "compatibility_residual,"
               "boundary_residual_linf,"
               "route_mismatch_linf,data_weighted_mean,density_weighted_mean,"
               "compatibility_mean_removed,"
               "density_linf,density_l2,density_order_linf,density_order_l2,"
               "interior_linf,interior_l2,interior_order_linf,"
               "interior_order_l2,exterior_bulk_linf,exterior_bulk_l2,"
               "exterior_bulk_order_linf,exterior_bulk_order_l2,"
               "constant_shift,value_trace_fit,"
               "topology_base_coordinates,topology_pre_mean_coordinates,"
               "mean_free_pivot_index,mean_free_pivot_moment,"
               "mean_free_relative_observability,"
               "mean_free_coordinate_condition,"
               "mean_free_particular_residual,"
               "mean_free_homogeneous_residual,"
               "mean_free_constant_projection_linf,"
               "mean_free_intrinsic_mean,known_value_jump,"
               "feature_vertex_residual_linf,feature_edge_residual_linf,"
               "trace_projection_leakage_relative_l2,"
               "dirichlet_jump_space,topology_trace_samples,"
               "topology_trace_final_dofs,"
               "topology_trace_oversampling_margin,"
               "topology_trace_oversampling_ratio,"
               "feature_discarded_residual_linf\n";
        std::map<std::string, const ReadinessResult*> previous;
        for (const ReadinessResult* row : rows) {
            const SolveMetrics3D& metric = select_metric(*row);
            int iteration_dofs = row->surface_dofs;
            if (row->density_iteration_mode == "reduced_coefficients") {
                iteration_dofs = metric.formulation.find("neumann") == 0
                    ? row->value_density_reduced_coefficients
                    : row->normal_density_reduced_coefficients;
            }
            if (metric.edge_reduced_dofs > 0)
                iteration_dofs = metric.edge_reduced_dofs;
            if (metric.topology_reduced_coordinates > 0)
                iteration_dofs = metric.topology_reduced_coordinates;
            double density_order_linf =
                std::numeric_limits<double>::quiet_NaN();
            double density_order_l2 =
                std::numeric_limits<double>::quiet_NaN();
            double interior_order_linf =
                std::numeric_limits<double>::quiet_NaN();
            double interior_order_l2 =
                std::numeric_limits<double>::quiet_NaN();
            double exterior_order_linf =
                std::numeric_limits<double>::quiet_NaN();
            double exterior_order_l2 =
                std::numeric_limits<double>::quiet_NaN();
            const auto found = previous.find(row->geometry);
            if (found != previous.end()) {
                const ReadinessResult& coarse = *found->second;
                const SolveMetrics3D& coarse_metric = select_metric(coarse);
                density_order_linf = observed_order(
                    coarse_metric.density_linf, metric.density_linf,
                    coarse.h, row->h);
                density_order_l2 = observed_order(
                    coarse_metric.density_l2, metric.density_l2,
                    coarse.h, row->h);
                interior_order_linf = observed_order(
                    coarse_metric.interior_linf, metric.interior_linf,
                    coarse.h, row->h);
                interior_order_l2 = observed_order(
                    coarse_metric.interior_l2, metric.interior_l2,
                    coarse.h, row->h);
                exterior_order_linf = observed_order(
                    coarse_metric.exterior_bulk_linf,
                    metric.exterior_bulk_linf, coarse.h, row->h);
                exterior_order_l2 = observed_order(
                    coarse_metric.exterior_bulk_l2,
                    metric.exterior_bulk_l2, coarse.h, row->h);
            }
            csv << row->geometry << ',' << row->cauchy_policy << ','
                << row->N << ',' << row->h << ','
                << iteration_dofs << ',' << row->density_iteration_mode << ','
                << row->density_coefficients_per_direction << ','
                << metric.formulation << ','
                << approximation_scheme_name() << ','
                << kCauchyPolynomialDegree << ','
                << kRestrictGridDegree << ','
                << kRestrictNormalDegree << ','
                << metric.compatibility_mode << ','
                << metric.trace_restrict_mode << ','
                << metric.restrict_value_jet << ','
                << metric.trace_sampling << ','
                << metric.density_coordinates << ','
                << metric.border_solver << ','
                << metric.edge_jump_jet << ','
                << metric.edge_feature_edges << ','
                << metric.edge_constraint_rows << ','
                << metric.edge_constraint_rank << ','
                << metric.edge_reduced_dofs << ','
                << metric.edge_constraint_residual_linf << ','
                << metric.edge_nullspace_residual_linf << ','
                << metric.edge_normal_fit_linf << ','
                << metric.edge_target_projection_linf << ','
                << metric.feature_target_requested_linf << ','
                << metric.feature_target_requested_l2 << ','
                << metric.feature_target_projected_linf << ','
                << metric.feature_target_projected_l2 << ','
                << metric.feature_target_projection_l2 << ','
                << metric.feature_target_projection_relative_l2 << ','
                << metric.feature_target_reachable_certification_linf << ','
                << metric.feature_target_projection_components << ','
                << metric.feature_target_constraints_exact << ','
                << metric.reduction_scheme << ','
                << metric.particular_solver << ','
                << metric.topology_constraint_rows << ','
                << metric.topology_constraint_rank << ','
                << metric.topology_full_coordinates << ','
                << metric.topology_reduced_coordinates << ','
                << metric.topology_block_count << ','
                << metric.topology_particular_residual_linf << ','
                << metric.topology_homogeneous_residual_linf << ','
                << metric.topology_projector_rank << ','
                << metric.topology_projector_condition << ','
                << metric.topology_projector_pb_error << ','
                << metric.topology_trace_closure_linf << ','
                << metric.iterations << ',' << metric.converged << ','
                << metric.physical_converged << ',' << metric.seconds << ','
                << metric.gmres_tolerance << ','
                << metric.gmres_relative_residual << ','
                << metric.operator_residual_linf << ','
                << metric.exterior_condition_linf << ','
                << metric.lagrange_multiplier << ','
                << metric.operator_compatibility_correction << ','
                << metric.total_flux_correction << ','
                << metric.border_column_linf << ','
                << metric.exterior_trace_weighted_mean << ','
                << metric.exterior_trace_demeaned_linf << ','
                << metric.augmented_trace_closure_linf << ','
                << metric.compatibility_residual << ','
                << metric.boundary_residual_linf << ','
                << metric.route_mismatch_linf << ','
                << metric.data_weighted_mean << ','
                << metric.density_weighted_mean << ','
                << metric.compatibility_mean_removed << ','
                << metric.density_linf << ',' << metric.density_l2 << ','
                << density_order_linf << ',' << density_order_l2 << ','
                << metric.interior_linf << ',' << metric.interior_l2 << ','
                << interior_order_linf << ',' << interior_order_l2 << ','
                << metric.exterior_bulk_linf << ','
                << metric.exterior_bulk_l2 << ','
                << exterior_order_linf << ',' << exterior_order_l2 << ','
                << metric.constant_shift << ','
                << row->cauchy_value_trace_fit << ','
                // Keep the established CSV column offsets stable: new
                // diagnostics are appended rather than inserted mid-schema.
                << metric.topology_base_coordinates << ','
                << metric.topology_pre_mean_coordinates << ','
                << metric.mean_free_pivot_index << ','
                << metric.mean_free_pivot_moment << ','
                << metric.mean_free_relative_observability << ','
                << metric.mean_free_coordinate_condition << ','
                << metric.mean_free_particular_residual << ','
                << metric.mean_free_homogeneous_residual << ','
                << metric.mean_free_constant_projection_linf << ','
                << metric.mean_free_intrinsic_mean << ','
                << metric.known_value_jump << ','
                << metric.feature_vertex_residual_linf << ','
                << metric.feature_edge_residual_linf << ','
                << metric.trace_projection_leakage_relative_l2 << ','
                << metric.dirichlet_jump_space << ','
                << metric.topology_trace_samples << ','
                << metric.topology_reduced_coordinates << ','
                << metric.topology_trace_oversampling_margin << ','
                << metric.topology_trace_oversampling_ratio << ','
                << metric.feature_discarded_residual_linf << '\n';
            previous[row->geometry] = row;
        }
    };

    if (solve_selection != SolveSelection3D::DirichletNormalOnly) {
        write_csv(output_dir / "neumann_results.csv",
                  [](const ReadinessResult& row) -> const SolveMetrics3D& {
                      return row.neumann;
                  });
    }
    if (solve_selection != SolveSelection3D::NeumannOnly) {
        write_csv(output_dir / "dirichlet_normal_results.csv",
                  [](const ReadinessResult& row) -> const SolveMetrics3D& {
                      return row.dirichlet_normal;
                  });
    }

    std::ofstream blocks = open_output_file(
        output_dir / "topology_block_records.csv");
    blocks << std::setprecision(17);
    blocks << "geometry,N,formulation,reduction_scheme,block_index,kind,key,"
              "source_keys,rows,active_rows,support_cols,rank,local_nullity,"
              "remaining,residual,normalized_consistency,"
              "homogeneous_residual,processed_particular_residual_linf,"
              "processed_homogeneous_residual_linf,merge_retry_count\n";
    const auto write_block_records = [&blocks](
        const ReadinessResult& row,
        const SolveMetrics3D& metric) {
        for (std::size_t index = 0;
             index < metric.topology_block_records.size(); ++index) {
            const auto& record = metric.topology_block_records[index];
            std::string sources;
            for (std::size_t q = 0; q < record.source_keys.size(); ++q) {
                if (q > 0)
                    sources += '|';
                sources += record.source_keys[q];
            }
            blocks << row.geometry << ',' << row.N << ','
                   << metric.formulation << ',' << metric.reduction_scheme
                   << ',' << index << ','
                   << (record.kind == app3d::ConstraintBlockKind3D::Vertex
                           ? "vertex" : "edge")
                   << ',' << app3d::rfc4180_csv_field(record.key)
                   << ',' << app3d::rfc4180_csv_field(sources) << ','
                   << record.rows << ',' << record.active_rows << ','
                   << record.support_cols << ',' << record.rank << ','
                   << record.local_nullity << ',' << record.remaining << ','
                   << record.residual << ','
                   << record.normalized_consistency << ','
                   << record.homogeneous_residual << ','
                   << record.processed_particular_residual_linf << ','
                   << record.processed_homogeneous_residual_linf << ','
                   << record.merge_retry_count << '\n';
        }
    };
    for (const ReadinessResult& row : results) {
        if (solve_selection != SolveSelection3D::DirichletNormalOnly)
            write_block_records(row, row.neumann);
        if (solve_selection != SolveSelection3D::NeumannOnly)
            write_block_records(row, row.dirichlet_normal);
    }
}

#ifdef KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT
void write_coefficient_convergence_results(
    const std::filesystem::path& output_dir,
    const std::vector<ReadinessResult>& results)
{
    std::filesystem::create_directories(output_dir);
    std::vector<const ReadinessResult*> rows;
    rows.reserve(results.size());
    for (const ReadinessResult& result : results)
        rows.push_back(&result);
    std::sort(rows.begin(), rows.end(),
              [](const ReadinessResult* a, const ReadinessResult* b) {
                  return std::tie(a->geometry, a->N)
                       < std::tie(b->geometry, b->N);
              });

    std::ofstream csv = open_output_file(
        output_dir / "coefficient_convergence_results.csv");
    csv << std::setprecision(17);
    csv << "geometry,N,h,ncoef,neumann_restrict_value_jet,"
           "surface_samples,value_coefficients,"
           "normal_coefficients,neumann_reduction_scheme,"
           "neumann_topology_coordinates,neumann_iterations,"
           "neumann_gmres_residual,"
           "neumann_exterior_trace_linf,neumann_density_linf,"
           "neumann_interior_linf,dirichlet_reduction_scheme,"
           "dirichlet_topology_coordinates,dirichlet_iterations,"
           "dirichlet_gmres_residual,dirichlet_exterior_normal_linf,"
           "dirichlet_density_linf,dirichlet_interior_linf\n";
    for (const ReadinessResult* row : rows) {
        csv << row->geometry << ',' << row->N << ',' << row->h << ','
            << row->density_coefficients_per_direction << ','
            << row->neumann.restrict_value_jet << ','
            << row->surface_dofs << ','
            << row->value_density_reduced_coefficients << ','
            << row->normal_density_reduced_coefficients << ','
            << row->neumann.reduction_scheme << ','
            << row->neumann.topology_reduced_coordinates << ','
            << row->neumann.iterations << ','
            << row->neumann.gmres_relative_residual << ','
            << row->neumann.exterior_condition_linf << ','
            << row->neumann.density_linf << ','
            << row->neumann.interior_linf << ','
            << row->dirichlet_normal.reduction_scheme << ','
            << row->dirichlet_normal.topology_reduced_coordinates << ','
            << row->dirichlet_normal.iterations << ','
            << row->dirichlet_normal.gmres_relative_residual << ','
            << row->dirichlet_normal.exterior_condition_linf << ','
            << row->dirichlet_normal.density_linf << ','
            << row->dirichlet_normal.interior_linf << '\n';
    }
}
#endif

using CriterionStatus3D = app3d::RigidStudyCriterionStatus3D;

const char* criterion_status_name(CriterionStatus3D status)
{
    switch (status) {
    case CriterionStatus3D::Pass:
        return "pass";
    case CriterionStatus3D::Fail:
        return "fail";
    case CriterionStatus3D::NotEvaluated:
        return "not_evaluated";
    }
    throw std::runtime_error("unknown rigid-study criterion status");
}

struct RigidStudyRow3D {
    app3d::DirichletRigidStudyCase3D study_case;
    ReadinessResult readiness;
    double total_seconds = 0.0;
    double observed_order = std::numeric_limits<double>::quiet_NaN();
    double baseline_error_ratio = std::numeric_limits<double>::quiet_NaN();
    double baseline_iteration_ratio = std::numeric_limits<double>::quiet_NaN();
    CriterionStatus3D gmres_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D monotone_error_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D order_64_128_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D baseline_ratio_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D geometry_diagnostics_pass =
        CriterionStatus3D::NotEvaluated;
    CriterionStatus3D overall_pass = CriterionStatus3D::NotEvaluated;
};

struct RigidStudyAcceptance3D {
    std::string case_id;
    CriterionStatus3D gmres_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D monotone_error_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D order_64_128_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D baseline_ratio_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D geometry_diagnostics_pass =
        CriterionStatus3D::NotEvaluated;
    CriterionStatus3D overall_pass = CriterionStatus3D::NotEvaluated;
};

bool rigid_geometry_diagnostics_pass(const ReadinessResult& result)
{
    return result.label_mismatches == 0
        && result.unsafe_label_changing_edges == 0
        && result.gap_crossings == 0
        && result.endpoint_crossings == 0
        && result.triangle_fallback_crossings == 0;
}

bool is_rigid_acceptance_level(int N)
{
    return N == 32 || N == 64 || N == 128;
}

const RigidStudyRow3D* find_rigid_study_row(
    const std::vector<RigidStudyRow3D>& rows,
    const std::string& case_id,
    int N)
{
    const auto found = std::find_if(
        rows.begin(), rows.end(), [&](const RigidStudyRow3D& row) {
            return row.study_case.id == case_id && row.readiness.N == N;
        });
    return found == rows.end() ? nullptr : &*found;
}

double rigid_observed_order(const RigidStudyRow3D& previous,
                            const RigidStudyRow3D& current)
{
    const double previous_error =
        previous.readiness.dirichlet_normal.interior_linf;
    const double current_error =
        current.readiness.dirichlet_normal.interior_linf;
    if (!(previous_error > 0.0) || !(current_error > 0.0)
        || current.readiness.N <= previous.readiness.N) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log(previous_error / current_error)
         / std::log(static_cast<double>(current.readiness.N)
                    / previous.readiness.N);
}

std::vector<RigidStudyAcceptance3D> update_rigid_study_criteria(
    std::vector<RigidStudyRow3D>& rows,
    const std::vector<app3d::DirichletRigidStudyCase3D>& cases,
    bool require_complete_acceptance)
{
    for (RigidStudyRow3D& row : rows) {
        const RigidStudyRow3D* previous = nullptr;
        for (const RigidStudyRow3D& candidate : rows) {
            if (candidate.study_case.id == row.study_case.id
                && candidate.readiness.N < row.readiness.N
                && (previous == nullptr
                    || candidate.readiness.N > previous->readiness.N)) {
                previous = &candidate;
            }
        }
        row.observed_order = previous == nullptr
            ? std::numeric_limits<double>::quiet_NaN()
            : rigid_observed_order(*previous, row);

        const RigidStudyRow3D* baseline = find_rigid_study_row(
            rows, "baseline", row.readiness.N);
        if (baseline != nullptr
            && baseline->readiness.dirichlet_normal.interior_linf > 0.0) {
            row.baseline_error_ratio =
                row.readiness.dirichlet_normal.interior_linf
                / baseline->readiness.dirichlet_normal.interior_linf;
        }
        if (baseline != nullptr
            && baseline->readiness.dirichlet_normal.iterations > 0) {
            row.baseline_iteration_ratio =
                static_cast<double>(row.readiness.dirichlet_normal.iterations)
                / baseline->readiness.dirichlet_normal.iterations;
        }
        row.gmres_pass = row.readiness.dirichlet_normal.converged
                      && row.readiness.dirichlet_normal.iterations <= 80
            ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
        if (is_rigid_acceptance_level(row.readiness.N)) {
            row.baseline_ratio_pass = std::isfinite(row.baseline_error_ratio)
                                   && row.baseline_error_ratio <= 3.0
                ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
        } else {
            row.baseline_ratio_pass = CriterionStatus3D::NotEvaluated;
        }
        row.geometry_diagnostics_pass =
            rigid_geometry_diagnostics_pass(row.readiness)
            ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
    }

    std::vector<RigidStudyAcceptance3D> acceptance;
    acceptance.reserve(cases.size());
    for (const auto& study_case : cases) {
        std::vector<RigidStudyRow3D*> case_rows;
        for (RigidStudyRow3D& row : rows) {
            if (row.study_case.id == study_case.id)
                case_rows.push_back(&row);
        }

        RigidStudyAcceptance3D item;
        item.case_id = study_case.id;
        if (!case_rows.empty()) {
            const bool gmres_pass = std::all_of(
                case_rows.begin(), case_rows.end(), [](const auto* row) {
                    return row->gmres_pass == CriterionStatus3D::Pass;
                });
            item.gmres_pass = gmres_pass ? CriterionStatus3D::Pass
                                        : CriterionStatus3D::Fail;
            std::vector<RigidStudyRow3D*> acceptance_level_rows;
            std::copy_if(
                case_rows.begin(), case_rows.end(),
                std::back_inserter(acceptance_level_rows), [](const auto* row) {
                    return is_rigid_acceptance_level(row->readiness.N);
                });
            if (!acceptance_level_rows.empty()) {
                const bool baseline_pass = std::all_of(
                    acceptance_level_rows.begin(), acceptance_level_rows.end(),
                    [](const auto* row) {
                        return row->baseline_ratio_pass
                            == CriterionStatus3D::Pass;
                    });
                item.baseline_ratio_pass = baseline_pass
                    ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
            }
            const bool geometry_pass = std::all_of(
                case_rows.begin(), case_rows.end(), [](const auto* row) {
                    return row->geometry_diagnostics_pass
                        == CriterionStatus3D::Pass;
                });
            item.geometry_diagnostics_pass = geometry_pass
                ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
        }

        const RigidStudyRow3D* row32 = find_rigid_study_row(
            rows, study_case.id, 32);
        const RigidStudyRow3D* row64 = find_rigid_study_row(
            rows, study_case.id, 64);
        const RigidStudyRow3D* row128 = find_rigid_study_row(
            rows, study_case.id, 128);
        if (row32 != nullptr && row64 != nullptr && row128 != nullptr) {
            const double error32 = row32->readiness.dirichlet_normal.interior_linf;
            const double error64 = row64->readiness.dirichlet_normal.interior_linf;
            const double error128 = row128->readiness.dirichlet_normal.interior_linf;
            item.monotone_error_pass = error32 > error64 && error64 > error128
                ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
        }
        if (row64 != nullptr && row128 != nullptr) {
            const double order = rigid_observed_order(*row64, *row128);
            item.order_64_128_pass = std::isfinite(order) && order >= 1.8
                ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
        }
        item.overall_pass = app3d::combine_rigid_study_criteria_3d(
            {item.gmres_pass,
             item.monotone_error_pass,
             item.order_64_128_pass,
             item.baseline_ratio_pass,
             item.geometry_diagnostics_pass},
            !case_rows.empty(),
            require_complete_acceptance);

        for (RigidStudyRow3D* row : case_rows) {
            row->monotone_error_pass = item.monotone_error_pass;
            row->order_64_128_pass = item.order_64_128_pass;
            row->overall_pass = item.overall_pass;
        }
        acceptance.push_back(std::move(item));
    }
    return acceptance;
}

void write_rigid_study_results(
    const std::filesystem::path& output_dir,
    const std::vector<RigidStudyRow3D>& rows)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream csv = open_output_file(
        output_dir / "rigid_transform_results.csv");
    csv << std::setprecision(17) << std::boolalpha;
    csv << "case_id,geometry,cauchy_policy,N,h,"
           "rotation_axis_x,rotation_axis_y,rotation_axis_z,"
           "rotation_angle_degrees,rotation_center_x,rotation_center_y,"
           "rotation_center_z,translation_x,translation_y,translation_z,"
           "rotation_00,rotation_01,rotation_02,rotation_10,rotation_11,"
           "rotation_12,rotation_20,rotation_21,rotation_22,surface_dofs,"
           "interior_linf,interior_l2,observed_order,gmres_converged,"
           "gmres_iterations,gmres_relative_residual,solve_seconds,"
           "total_seconds,operator_residual_linf,exterior_condition_linf,"
           "boundary_residual_linf,route_mismatch_linf,density_linf,"
           "density_l2,exterior_bulk_linf,exterior_bulk_l2,"
           "cauchy_value_neighbors,cauchy_derivative_neighbors,"
           "cauchy_condition_median,cauchy_condition_p95,"
           "cauchy_condition_max,label_mismatches,"
           "unsafe_label_changing_edges,gap_crossings,endpoint_crossings,"
           "triangle_fallback_crossings,targeted_retries_unsafe,"
           "ambiguous_parity_edges,ambiguous_label_changing_edges,"
           "endpoint_parity_fallbacks,baseline_error_ratio,"
           "baseline_iteration_ratio,gmres_pass,monotone_error_pass,"
           "order_64_128_pass,baseline_ratio_pass,"
           "geometry_diagnostics_pass,overall_pass\n";
    for (const RigidStudyRow3D& row : rows) {
        const auto& transform = row.study_case.transform;
        const auto& result = row.readiness;
        const auto& metric = result.dirichlet_normal;
        const Eigen::Matrix3d& rotation = transform.rotation();
        csv << row.study_case.id << ',' << result.geometry << ','
            << result.cauchy_policy << ',' << result.N << ',' << result.h
            << ',' << row.study_case.rotation_axis.x()
            << ',' << row.study_case.rotation_axis.y()
            << ',' << row.study_case.rotation_axis.z()
            << ',' << row.study_case.rotation_angle_degrees
            << ',' << transform.center().x()
            << ',' << transform.center().y()
            << ',' << transform.center().z()
            << ',' << transform.translation().x()
            << ',' << transform.translation().y()
            << ',' << transform.translation().z();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
                csv << ',' << rotation(i, j);
        }
        csv << ',' << result.surface_dofs
            << ',' << metric.interior_linf
            << ',' << metric.interior_l2
            << ',' << row.observed_order
            << ',' << metric.converged
            << ',' << metric.iterations
            << ',' << metric.gmres_relative_residual
            << ',' << metric.seconds
            << ',' << row.total_seconds
            << ',' << metric.operator_residual_linf
            << ',' << metric.exterior_condition_linf
            << ',' << metric.boundary_residual_linf
            << ',' << metric.route_mismatch_linf
            << ',' << metric.density_linf
            << ',' << metric.density_l2
            << ',' << metric.exterior_bulk_linf
            << ',' << metric.exterior_bulk_l2
            << ',' << result.cauchy_value_neighbors
            << ',' << result.cauchy_derivative_neighbors
            << ',' << result.cauchy_condition_median
            << ',' << result.cauchy_condition_p95
            << ',' << result.cauchy_condition_max
            << ',' << result.label_mismatches
            << ',' << result.unsafe_label_changing_edges
            << ',' << result.gap_crossings
            << ',' << result.endpoint_crossings
            << ',' << result.triangle_fallback_crossings
            << ',' << result.targeted_retries_unsafe
            << ',' << result.ambiguous_parity_edges
            << ',' << result.ambiguous_label_changing_edges
            << ',' << result.endpoint_parity_fallbacks
            << ',' << row.baseline_error_ratio
            << ',' << row.baseline_iteration_ratio
            << ',' << criterion_status_name(row.gmres_pass)
            << ',' << criterion_status_name(row.monotone_error_pass)
            << ',' << criterion_status_name(row.order_64_128_pass)
            << ',' << criterion_status_name(row.baseline_ratio_pass)
            << ',' << criterion_status_name(row.geometry_diagnostics_pass)
            << ',' << criterion_status_name(row.overall_pass) << '\n';
    }
}

void write_rigid_study_acceptance(
    const std::filesystem::path& output_dir,
    const std::vector<RigidStudyAcceptance3D>& acceptance)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream csv = open_output_file(
        output_dir / "rigid_transform_acceptance.csv");
    csv << "case_id,gmres_pass,monotone_error_pass,order_64_128_pass,"
           "baseline_ratio_pass,geometry_diagnostics_pass,overall_pass\n";
    for (const RigidStudyAcceptance3D& item : acceptance) {
        csv << item.case_id
            << ',' << criterion_status_name(item.gmres_pass)
            << ',' << criterion_status_name(item.monotone_error_pass)
            << ',' << criterion_status_name(item.order_64_128_pass)
            << ',' << criterion_status_name(item.baseline_ratio_pass)
            << ',' << criterion_status_name(item.geometry_diagnostics_pass)
            << ',' << criterion_status_name(item.overall_pass) << '\n';
    }
}

int run_dirichlet_rigid_study(
    std::vector<int> levels,
    CauchyStencilPolicy3D cauchy_policy,
    int cauchy_value_count,
    int cauchy_normal_count,
    int gmres_max_iterations,
    double gmres_tolerance)
{
    if (selected_density_iteration_mode()
        != DensityIterationMode3D::ReducedCoefficients) {
        throw std::invalid_argument(
            "--rigid-study requires "
            "KFBIM_3D_DENSITY_MODE=reduced_coefficients");
    }
    if (cauchy_policy != CauchyStencilPolicy3D::G1Nearest
        || cauchy_value_count != kCauchyValueNeighborCount
        || cauchy_normal_count != kCauchyDerivativeNeighborCount) {
        throw std::invalid_argument(
            "--rigid-study requires KFBIM_3D_CAUCHY_POLICY=g1_nearest, "
            "KFBIM_3D_CAUCHY_VALUE_COUNT=48, and "
            "KFBIM_3D_CAUCHY_NORMAL_COUNT=28");
    }
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());

    const bool require_complete_acceptance =
        std::binary_search(levels.begin(), levels.end(), 32)
        && std::binary_search(levels.begin(), levels.end(), 64)
        && std::binary_search(levels.begin(), levels.end(), 128);

    const std::filesystem::path output_dir =
        application_output_root_3d()
        / "dirichlet_rigid_transform_stability_3d";
    const std::vector<app3d::DirichletRigidStudyCase3D> cases =
        app3d::make_l_prism_dirichlet_rigid_study_cases_3d();
    std::vector<RigidStudyRow3D> rows;
    rows.reserve(levels.size() * cases.size());

    std::cout << "KFBI3D Dirichlet rigid-transform stability study\n"
              << "  geometry=l_prism formulation=dirichlet_normal_only\n"
              << "  cauchy_policy=" << cauchy_policy_name(cauchy_policy)
              << " cauchy_counts=" << cauchy_value_count << '/'
              << cauchy_normal_count << '\n'
              << "  gmres_max_iterations=" << gmres_max_iterations
              << " tolerance=" << gmres_tolerance << '\n'
              << "  levels=";
    for (std::size_t index = 0; index < levels.size(); ++index) {
        if (index != 0)
            std::cout << ',';
        std::cout << levels[index];
    }
    std::cout << " cases=" << cases.size() << '\n';

    for (int N : levels) {
        for (const auto& study_case : cases) {
            const auto case_start = std::chrono::steady_clock::now();
            ReadinessResult result = run_readiness_case(
                GeometryKind::LPrism,
                N,
                output_dir,
                cauchy_policy,
                cauchy_value_count,
                cauchy_normal_count,
                kNeumannRestrictValueNeighborCount,
                kNeumannRestrictNormalNeighborCount,
                gmres_max_iterations,
                gmres_tolerance,
                NeumannCompatibilityMode3D::OperatorFluxCorrection,
                ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner,
                NeumannRestrictValueJetMode3D::FeatureConstrained,
                ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner,
                study_case.transform,
                SolveSelection3D::DirichletNormalOnly);
            const double total_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - case_start).count();
            rows.push_back({study_case, std::move(result), total_seconds});
            const std::vector<RigidStudyAcceptance3D> acceptance =
                update_rigid_study_criteria(
                    rows, cases, require_complete_acceptance);
            write_rigid_study_results(output_dir, rows);
            write_rigid_study_acceptance(output_dir, acceptance);

            const RigidStudyRow3D& row = rows.back();
            const SolveMetrics3D& metric = row.readiness.dirichlet_normal;
            std::cout << "[rigid-study] case=" << row.study_case.id
                      << " N=" << row.readiness.N
                      << " dirichlet_converged/iterations="
                      << metric.converged << '/' << metric.iterations
                      << " interior_linf=" << metric.interior_linf
                      << " order=" << row.observed_order
                      << " baseline_error_ratio="
                      << row.baseline_error_ratio
                      << " solve_seconds=" << metric.seconds
                      << " total_seconds=" << row.total_seconds << '\n';
            if (!metric.converged) {
                std::cerr << "error: 3D rigid-study GMRES did not converge: "
                          << "case=" << row.study_case.id
                          << " N=" << row.readiness.N
                          << " iterations=" << metric.iterations
                          << " final_relative_residual="
                          << std::scientific << std::setprecision(17)
                          << metric.gmres_relative_residual << '\n';
                return 1;
            }
        }
    }

    const std::vector<RigidStudyAcceptance3D> acceptance =
        update_rigid_study_criteria(
            rows, cases, require_complete_acceptance);
    write_rigid_study_results(output_dir, rows);
    write_rigid_study_acceptance(output_dir, acceptance);
    bool all_pass = true;
    for (const RigidStudyAcceptance3D& item : acceptance) {
        std::cout << "[rigid-acceptance] case=" << item.case_id
                  << " gmres=" << criterion_status_name(item.gmres_pass)
                  << " monotone="
                  << criterion_status_name(item.monotone_error_pass)
                  << " order_64_128="
                  << criterion_status_name(item.order_64_128_pass)
                  << " baseline_ratio="
                  << criterion_status_name(item.baseline_ratio_pass)
                  << " geometry="
                  << criterion_status_name(item.geometry_diagnostics_pass)
                  << " overall=" << criterion_status_name(item.overall_pass)
                  << '\n';
        all_pass = all_pass && item.overall_pass == CriterionStatus3D::Pass;
    }
    std::cout << "Rigid-transform study output: " << output_dir.string()
              << '\n';
    if (!all_pass) {
        std::cerr << "error: one or more rigid-study acceptance criteria failed\n";
        return 1;
    }
    return 0;
}

struct NormalRestrictNorms3D {
    double linf = 0.0;
    double weighted_rms = 0.0;
};

struct CommonRhsGmresProbe3D {
    bool converged = false;
    int iterations = 0;
    double final_residual = 0.0;
    std::vector<double> residuals;
};

double residual_contraction(
    const std::vector<double>& residuals,
    std::size_t begin,
    std::size_t end)
{
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    if (begin >= end || end >= residuals.size())
        return invalid;
    const double first = residuals[begin];
    const double last = residuals[end];
    if (!std::isfinite(first) || !std::isfinite(last)
        || !(first > 0.0) || last < 0.0) {
        return invalid;
    }
    if (last == 0.0)
        return 0.0;
    return std::exp(
        (std::log(last) - std::log(first))
        / static_cast<double>(end - begin));
}

double worst_five_step_contraction(const std::vector<double>& residuals)
{
    double worst = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t begin = 0; begin + 5 < residuals.size(); ++begin) {
        const double candidate =
            residual_contraction(residuals, begin, begin + 5);
        if (std::isfinite(candidate)
            && (!std::isfinite(worst) || candidate > worst)) {
            worst = candidate;
        }
    }
    return worst;
}

struct NormalRestrictRouteProbe3D {
    bool complete = false;
    std::string route;
    ConditionStatistics3D conditions;
    Eigen::VectorXd exact_grid_error;
    Eigen::VectorXd smooth_grid_error;
    Eigen::VectorXd exact_equation_residual;
    NormalRestrictNorms3D exact_grid_norms;
    NormalRestrictNorms3D smooth_grid_norms;
    NormalRestrictNorms3D exact_equation_norms;
    SolveMetrics3D physical;
    std::vector<double> physical_residuals;
    CommonRhsGmresProbe3D common;
    double physical_rho_10_30 =
        std::numeric_limits<double>::quiet_NaN();
    double physical_worst_rho_5 =
        std::numeric_limits<double>::quiet_NaN();
    double common_rho_10_30 = std::numeric_limits<double>::quiet_NaN();
    double common_worst_rho_5 = std::numeric_limits<double>::quiet_NaN();
    double seconds = 0.0;
};

struct NormalRestrictCaseProbe3D {
    std::string case_id;
    int N = 0;
    double h = 0.0;
    std::vector<int> patch_ids;
    std::vector<Eigen::Vector3d> points;
    std::vector<double> weights;
    std::vector<double> feature_edge_distance_over_h;
    std::vector<int> wrong_side_support_count;
    std::vector<RestrictOwnerSampleDiagnostics3D> owner_diagnostics;
    std::vector<RestrictOwnerAuditRecord3D> owner_audit_records;
    std::size_t owner_geometry_query_count = 0;
    double setup_seconds = 0.0;
    double pipeline_setup_seconds = 0.0;
    std::array<NormalRestrictRouteProbe3D, 3> routes;
};

const char* normal_restrict_route_name(ExteriorNormalRestrictMode3D mode)
{
    switch (mode) {
    case ExteriorNormalRestrictMode3D::JointTricubicCauchy:
        return "joint_tricubic_cauchy";
    case ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner:
        return "joint_tricubic_crossing_owner";
    case ExteriorNormalRestrictMode3D::
            GlobalCubicExteriorBranchCrossingOwner:
        throw std::invalid_argument(
            "global exterior-branch route is value-trace only");
    case ExteriorNormalRestrictMode3D::SharedQuadraticGridlineCauchy:
        return "shared_quadratic_gridline_cauchy";
    case ExteriorNormalRestrictMode3D::SharedQ10CubicGridlineCauchy:
        throw std::invalid_argument(
            "shared Q10 cubic route is not part of the legacy causal probe");
    case ExteriorNormalRestrictMode3D::Q27Cover3AllEventCauchy:
    case ExteriorNormalRestrictMode3D::Q64Cover4AllEventCauchy:
        throw std::invalid_argument(
            "direct tensor-cover routes are not part of the legacy causal probe");
    case ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic:
        return "exterior_only_harmonic_cubic";
    }
    throw std::runtime_error("unknown exterior-normal restrict mode");
}

int normal_restrict_route_index(ExteriorNormalRestrictMode3D mode)
{
    switch (mode) {
    case ExteriorNormalRestrictMode3D::JointTricubicCauchy:
        return 0;
    case ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner:
        return 1;
    case ExteriorNormalRestrictMode3D::
            GlobalCubicExteriorBranchCrossingOwner:
        throw std::invalid_argument(
            "global exterior-branch route is value-trace only");
    case ExteriorNormalRestrictMode3D::SharedQuadraticGridlineCauchy:
    case ExteriorNormalRestrictMode3D::SharedQ10CubicGridlineCauchy:
    case ExteriorNormalRestrictMode3D::Q27Cover3AllEventCauchy:
    case ExteriorNormalRestrictMode3D::Q64Cover4AllEventCauchy:
        throw std::invalid_argument(
            "all-event cover routes are not part of the legacy causal probe");
    case ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic:
        return 2;
    }
    throw std::runtime_error("unknown exterior-normal restrict mode");
}

NormalRestrictNorms3D normal_restrict_weighted_norms(
    const Eigen::VectorXd& values,
    const std::vector<double>& weights,
    const std::vector<int>* selected = nullptr)
{
    if (values.size() != static_cast<int>(weights.size()))
        throw std::invalid_argument("normal-restrict norm size mismatch");
    NormalRestrictNorms3D result;
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    const auto accumulate = [&](int q) {
        const double value = values[q];
        const double weight = weights[static_cast<std::size_t>(q)];
        if (!std::isfinite(value) || !std::isfinite(weight) || weight <= 0.0)
            throw std::runtime_error("normal-restrict norm received invalid data");
        result.linf = std::max(result.linf, std::abs(value));
        weighted_sum += weight * value * value;
        weight_sum += weight;
    };
    if (selected == nullptr) {
        for (int q = 0; q < values.size(); ++q)
            accumulate(q);
    } else {
        for (int q : *selected)
            accumulate(q);
    }
    if (weight_sum > 0.0)
        result.weighted_rms = std::sqrt(weighted_sum / weight_sum);
    return result;
}

double point_segment_distance_3d(
    const Eigen::Vector3d& point,
    const geometry3d::NurbsPatchGeometricEdge3D& edge)
{
    const Eigen::Vector3d direction = edge.end - edge.start;
    const double length_squared = direction.squaredNorm();
    if (!(length_squared > 0.0))
        return (point - edge.start).norm();
    const double phase = std::max(
        0.0, std::min(1.0, (point - edge.start).dot(direction) / length_squared));
    return (point - (edge.start + phase * direction)).norm();
}

Eigen::VectorXd make_common_normal_restrict_rhs(int size)
{
    Eigen::VectorXd rhs(size);
    for (int q = 0; q < size; ++q) {
        const double index = static_cast<double>(q + 1);
        rhs[q] = std::sin(0.73 * index) + 0.25 * std::cos(0.19 * index);
    }
    const double norm = rhs.norm();
    if (!(norm > 0.0) || !std::isfinite(norm))
        throw std::runtime_error("common normal-restrict RHS is invalid");
    rhs /= norm;
    return rhs;
}

CommonRhsGmresProbe3D run_common_normal_restrict_gmres(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    ExteriorNormalRestrictMode3D mode,
    const Eigen::VectorXd& rhs)
{
    ExteriorNormalTraceOperator3D op(pipeline, mode);
    Eigen::VectorXd unknown = Eigen::VectorXd::Zero(op.problem_size());
    GMRES gmres(160, 2.0e-10, 0);
    CommonRhsGmresProbe3D result;
    result.iterations = gmres.solve(op, rhs, unknown);
    result.converged = gmres.converged();
    result.residuals = gmres.residuals();
    result.final_residual = result.residuals.empty()
        ? 0.0 : result.residuals.back();
    return result;
}

void write_normal_restrict_probe_outputs(
    const std::filesystem::path& output_dir,
    const std::vector<NormalRestrictCaseProbe3D>& cases)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream summary = open_output_file(output_dir / "summary.csv");
    summary << std::setprecision(17)
            << "case_id,N,route,restrict_condition_median,"
               "restrict_condition_p95,restrict_condition_max,"
               "exact_grid_linf,exact_grid_wrms,smooth_grid_linf,"
               "smooth_grid_wrms,exact_equation_linf,exact_equation_wrms,"
               "physical_converged,physical_iterations,"
               "physical_final_residual,physical_interior_linf,"
               "physical_rho_10_30,physical_worst_rho_5,"
               "common_converged,common_iterations,common_final_residual,"
               "common_rho_10_30,common_worst_rho_5,"
               "setup_seconds,pipeline_setup_seconds,"
               "route_seconds\n";
    std::ofstream residuals = open_output_file(
        output_dir / "gmres_residuals.csv");
    residuals << std::setprecision(17)
              << "case_id,N,route,solve_kind,iteration,relative_residual\n";
    for (const NormalRestrictCaseProbe3D& probe_case : cases) {
        for (const NormalRestrictRouteProbe3D& route : probe_case.routes) {
            if (!route.complete)
                continue;
            const double physical_final = route.physical_residuals.empty()
                ? 0.0 : route.physical_residuals.back();
            summary << probe_case.case_id << ',' << probe_case.N << ','
                    << route.route << ',' << route.conditions.median << ','
                    << route.conditions.p95 << ',' << route.conditions.maximum
                    << ',' << route.exact_grid_norms.linf << ','
                    << route.exact_grid_norms.weighted_rms << ','
                    << route.smooth_grid_norms.linf << ','
                    << route.smooth_grid_norms.weighted_rms << ','
                    << route.exact_equation_norms.linf << ','
                    << route.exact_equation_norms.weighted_rms << ','
                    << route.physical.converged << ','
                    << route.physical.iterations << ',' << physical_final << ','
                    << route.physical.interior_linf << ','
                    << route.physical_rho_10_30 << ','
                    << route.physical_worst_rho_5 << ','
                    << route.common.converged << ',' << route.common.iterations
                    << ',' << route.common.final_residual << ','
                    << route.common_rho_10_30 << ','
                    << route.common_worst_rho_5 << ','
                    << probe_case.setup_seconds << ','
                    << probe_case.pipeline_setup_seconds << ','
                    << route.seconds << '\n';
            for (std::size_t k = 0; k < route.physical_residuals.size(); ++k)
                residuals << probe_case.case_id << ',' << probe_case.N << ','
                          << route.route << ",physical," << k << ','
                          << route.physical_residuals[k] << '\n';
            for (std::size_t k = 0; k < route.common.residuals.size(); ++k)
                residuals << probe_case.case_id << ',' << probe_case.N << ','
                          << route.route << ",common," << k << ','
                          << route.common.residuals[k] << '\n';
        }
    }
}

void write_normal_restrict_probe_localization(
    const std::filesystem::path& output_dir,
    const std::vector<NormalRestrictCaseProbe3D>& cases)
{
    std::ofstream dofs = open_output_file(output_dir / "dof_diagnostics.csv");
    dofs << std::setprecision(17)
         << "case_id,N,route,dof,patch_id,x,y,z,weight,"
            "feature_edge_distance_over_h,wrong_side_support_count,"
            "exact_grid_error,smooth_grid_error,"
            "exact_equation_residual\n";
    std::ofstream bins = open_output_file(output_dir / "edge_distance_bins.csv");
    bins << std::setprecision(17)
         << "case_id,N,route,distance_bin,count,"
            "exact_grid_linf,exact_grid_wrms,"
            "exact_equation_linf,exact_equation_wrms\n";
    const std::array<double, 5> upper{{
        1.0, 2.0, 4.0, 8.0,
        std::numeric_limits<double>::infinity()}};
    const std::array<const char*, 5> labels{{
        "0_1", "1_2", "2_4", "4_8", "8_inf"}};
    for (const NormalRestrictCaseProbe3D& probe_case : cases) {
        const int size = static_cast<int>(probe_case.weights.size());
        std::array<std::vector<int>, 5> members;
        for (int q = 0; q < size; ++q) {
            const double distance = probe_case.feature_edge_distance_over_h[
                static_cast<std::size_t>(q)];
            int bin = 0;
            while (bin < 4 && !(distance < upper[static_cast<std::size_t>(bin)]))
                ++bin;
            members[static_cast<std::size_t>(bin)].push_back(q);
        }
        for (const NormalRestrictRouteProbe3D& route : probe_case.routes) {
            if (!route.complete)
                continue;
            for (int q = 0; q < size; ++q) {
                const Eigen::Vector3d& point =
                    probe_case.points[static_cast<std::size_t>(q)];
                dofs << probe_case.case_id << ',' << probe_case.N << ','
                     << route.route << ',' << q << ','
                     << probe_case.patch_ids[static_cast<std::size_t>(q)] << ','
                     << point.x() << ',' << point.y() << ',' << point.z() << ','
                     << probe_case.weights[static_cast<std::size_t>(q)] << ','
                     << probe_case.feature_edge_distance_over_h[
                            static_cast<std::size_t>(q)] << ','
                     << probe_case.wrong_side_support_count[
                            static_cast<std::size_t>(q)] << ','
                     << route.exact_grid_error[q] << ','
                     << route.smooth_grid_error[q] << ','
                     << route.exact_equation_residual[q] << '\n';
            }
            for (int bin = 0; bin < 5; ++bin) {
                const std::vector<int>& selected =
                    members[static_cast<std::size_t>(bin)];
                const NormalRestrictNorms3D grid_norms =
                    normal_restrict_weighted_norms(
                        route.exact_grid_error, probe_case.weights, &selected);
                const NormalRestrictNorms3D equation_norms =
                    normal_restrict_weighted_norms(
                        route.exact_equation_residual,
                        probe_case.weights, &selected);
                bins << probe_case.case_id << ',' << probe_case.N << ','
                     << route.route << ','
                     << labels[static_cast<std::size_t>(bin)] << ','
                     << selected.size() << ','
                     << grid_norms.linf << ',' << grid_norms.weighted_rms << ','
                     << equation_norms.linf << ','
                     << equation_norms.weighted_rms << '\n';
            }
        }
    }
}

const char* restrict_owner_decision_kind_name(std::size_t kind)
{
    switch (static_cast<app3d::RestrictOwnerDecisionKind3D>(kind)) {
    case app3d::RestrictOwnerDecisionKind3D::TargetSideNode:
        return "target_side_node";
    case app3d::RestrictOwnerDecisionKind3D::TargetOrG1SingleCrossing:
        return "target_or_g1_single_crossing";
    case app3d::RestrictOwnerDecisionKind3D::ForeignNonG1SingleCrossing:
        return "foreign_non_g1_single_crossing";
    case app3d::RestrictOwnerDecisionKind3D::NoCrossingFallback:
        return "no_crossing_fallback";
    case app3d::RestrictOwnerDecisionKind3D::MultipleCrossingFallback:
        return "multiple_crossing_fallback";
    case app3d::RestrictOwnerDecisionKind3D::DegenerateCrossingFallback:
        return "degenerate_crossing_fallback";
    case app3d::RestrictOwnerDecisionKind3D::AmbiguousEdgeFallback:
        return "ambiguous_edge_fallback";
    }
    throw std::logic_error("unknown crossing-owner decision kind");
}

void write_restrict_owner_probe_outputs(
    const std::filesystem::path& output_dir,
    const std::vector<NormalRestrictCaseProbe3D>& cases)
{
    std::ofstream summary =
        open_output_file(output_dir / "restrict_owner_summary.csv");
    summary << std::setprecision(17)
            << "case_id,N,target_dof,target_patch,side,layer,decision_kind,"
               "count,sum_abs_weight,wrong_side_count,"
               "wrong_side_sum_abs_weight,geometry_query_count\n";
    std::ofstream terms =
        open_output_file(output_dir / "restrict_owner_terms.csv");
    terms << std::setprecision(17)
          << "case_id,N,target_dof,target_patch,side,layer,grid_node,weight,"
             "owner_dof,owner_patch,crossing_patch,crossing_u,crossing_v,"
             "segment_parameter,residual,transversality\n";
    for (const NormalRestrictCaseProbe3D& probe_case : cases) {
        for (const RestrictOwnerSampleDiagnostics3D& item
             : probe_case.owner_diagnostics) {
            for (std::size_t kind = 0;
                 kind < kRestrictOwnerDecisionKindCount; ++kind) {
                summary << probe_case.case_id << ',' << probe_case.N << ','
                        << item.target_dof << ',' << item.target_patch << ','
                        << item.side << ',' << item.layer << ','
                        << restrict_owner_decision_kind_name(kind) << ','
                        << item.decision_counts[kind] << ','
                        << item.decision_sum_abs_weights[kind] << ','
                        << item.wrong_side_count << ','
                        << item.wrong_side_sum_abs_weight << ','
                        << item.geometry_query_count << '\n';
            }
            summary << probe_case.case_id << ',' << probe_case.N << ','
                    << item.target_dof << ',' << item.target_patch << ','
                    << item.side << ',' << item.layer
                    << ",unresolved_exception_fallback,"
                    << item.unresolved_fallback_count << ','
                    << item.unresolved_fallback_sum_abs_weight << ','
                    << item.wrong_side_count << ','
                    << item.wrong_side_sum_abs_weight << ','
                    << item.geometry_query_count << '\n';
            summary << probe_case.case_id << ',' << probe_case.N << ','
                    << item.target_dof << ',' << item.target_patch << ','
                    << item.side << ',' << item.layer
                    << ",unrelated_coincidence_fallback,"
                    << item.unrelated_coincidence_fallback_count << ','
                    << item.unrelated_coincidence_fallback_sum_abs_weight << ','
                    << item.wrong_side_count << ','
                    << item.wrong_side_sum_abs_weight << ','
                    << item.geometry_query_count << '\n';
        }
        for (const RestrictOwnerAuditRecord3D& item
             : probe_case.owner_audit_records) {
            terms << probe_case.case_id << ',' << probe_case.N << ','
                  << item.target_dof << ',' << item.target_patch << ','
                  << item.side << ',' << item.layer << ',' << item.grid_node
                  << ',' << item.interpolation_weight << ',' << item.owner_dof
                  << ',' << item.owner_patch << ',' << item.crossing_patch
                  << ',' << item.crossing_u << ',' << item.crossing_v << ','
                  << item.segment_parameter << ',' << item.residual << ','
                  << item.transversality << '\n';
        }
    }
}

void write_all_normal_restrict_probe_outputs(
    const std::filesystem::path& output_dir,
    const std::vector<NormalRestrictCaseProbe3D>& cases)
{
    write_normal_restrict_probe_outputs(output_dir, cases);
    write_normal_restrict_probe_localization(output_dir, cases);
}

const NormalRestrictCaseProbe3D* find_normal_restrict_probe_case(
    const std::vector<NormalRestrictCaseProbe3D>& cases,
    const std::string& case_id,
    int N)
{
    for (const NormalRestrictCaseProbe3D& probe_case : cases) {
        if (probe_case.case_id == case_id && probe_case.N == N)
            return &probe_case;
    }
    return nullptr;
}

bool normal_restrict_hypothesis_supported(
    const std::vector<NormalRestrictCaseProbe3D>& cases,
    const std::vector<int>& levels)
{
    const std::array<std::string, 2> rotated_ids{{
        "rot_axis123_17deg", "rot_axis123_17deg_t_xyz_1"}};
    bool supported = true;
    for (int N : levels) {
        const NormalRestrictCaseProbe3D* baseline =
            find_normal_restrict_probe_case(cases, "baseline", N);
        if (baseline == nullptr
            || !baseline->routes[0].complete
            || !baseline->routes[1].complete) {
            supported = false;
            continue;
        }
        const NormalRestrictRouteProbe3D& baseline_legacy =
            baseline->routes[0];
        const NormalRestrictRouteProbe3D& baseline_owner =
            baseline->routes[1];
        const int baseline_physical_margin = std::max(
            5, static_cast<int>(std::ceil(
                   0.10 * static_cast<double>(
                       std::max(1, baseline_legacy.physical.iterations)))));
        const int baseline_common_margin = std::max(
            5, static_cast<int>(std::ceil(
                   0.10 * static_cast<double>(
                       std::max(1, baseline_legacy.common.iterations)))));
        const bool baseline_behavior_stable =
            std::abs(baseline_owner.physical.iterations
                     - baseline_legacy.physical.iterations)
                <= baseline_physical_margin
            && std::abs(baseline_owner.common.iterations
                        - baseline_legacy.common.iterations)
                <= baseline_common_margin;
        const bool baseline_valid =
            baseline_legacy.physical.converged
            && baseline_legacy.common.converged
            && baseline_owner.physical.converged
            && baseline_owner.common.converged
            && baseline_behavior_stable
            && baseline_owner.physical.interior_linf
               <= 1.25 * std::max(
                    baseline_legacy.physical.interior_linf, 1.0e-14);
        supported = supported && baseline_valid;
        for (const std::string& rotated_id : rotated_ids) {
            const NormalRestrictCaseProbe3D* rotated =
                find_normal_restrict_probe_case(cases, rotated_id, N);
            if (rotated == nullptr
                || !rotated->routes[0].complete
                || !rotated->routes[1].complete) {
                supported = false;
                continue;
            }
            const NormalRestrictRouteProbe3D& legacy = rotated->routes[0];
            const NormalRestrictRouteProbe3D& owner = rotated->routes[1];
            const bool routes_valid =
                legacy.physical.converged && legacy.common.converged
                && owner.physical.converged && owner.common.converged;
            const bool accuracy_stable =
                owner.physical.interior_linf
                <= 1.50 * std::max(
                    legacy.physical.interior_linf, 1.0e-14);
            const bool physical_plateau_shortened =
                owner.physical.iterations + 3
                    <= legacy.physical.iterations
                || (std::isfinite(owner.physical_rho_10_30)
                    && std::isfinite(legacy.physical_rho_10_30)
                    && owner.physical_rho_10_30
                       <= 0.98 * legacy.physical_rho_10_30);
            const bool common_plateau_shortened =
                owner.common.iterations + 3 <= legacy.common.iterations
                || (std::isfinite(owner.common_rho_10_30)
                    && std::isfinite(legacy.common_rho_10_30)
                    && owner.common_rho_10_30
                       <= 0.98 * legacy.common_rho_10_30);
            const bool pair_supported = baseline_valid && routes_valid
                && accuracy_stable
                && (physical_plateau_shortened
                    || common_plateau_shortened);
            supported = supported && pair_supported;
            std::cout << "[restrict-decision] case=" << rotated_id
                      << " N=" << N
                      << " physical_iterations(legacy/owner)="
                      << legacy.physical.iterations << '/'
                      << owner.physical.iterations
                      << " common_iterations(legacy/owner)="
                      << legacy.common.iterations << '/'
                      << owner.common.iterations
                      << " physical_plateau_shortened="
                      << physical_plateau_shortened
                      << " common_plateau_shortened="
                      << common_plateau_shortened
                      << " baseline_behavior_stable="
                      << baseline_behavior_stable
                      << " accuracy_stable=" << accuracy_stable << '\n';
        }
    }
    return supported;
}

int run_normal_restrict_causal_probe(std::vector<int> levels)
{
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    for (int N : levels) {
        if (N < 16 || !is_power_of_two(N))
            throw std::invalid_argument(
                "restrict-probe N must be a power of two and at least 16");
    }
    const std::filesystem::path output_dir =
        application_output_root_3d()
        / "dirichlet_normal_restrict_causal_probe_3d";
    const std::array<std::string, 3> selected_ids{{
        "baseline", "rot_axis123_17deg",
        "rot_axis123_17deg_t_xyz_1"}};
    const std::vector<app3d::DirichletRigidStudyCase3D> all_cases =
        app3d::make_l_prism_dirichlet_rigid_study_cases_3d();
    std::vector<app3d::DirichletRigidStudyCase3D> study_cases;
    for (const std::string& id : selected_ids) {
        const auto found = std::find_if(
            all_cases.begin(), all_cases.end(),
            [&](const app3d::DirichletRigidStudyCase3D& item) {
                return item.id == id;
            });
        if (found == all_cases.end())
            throw std::runtime_error("missing restrict-probe rigid case: " + id);
        study_cases.push_back(*found);
    }

    std::vector<NormalRestrictCaseProbe3D> results;
    write_all_normal_restrict_probe_outputs(output_dir, results);
    write_restrict_owner_probe_outputs(output_dir, results);
    std::cout << "KFBI3D exterior-normal restrict causal probe\n"
              << "  geometry=l_prism cauchy=g1_nearest/degree3/48/28\n"
              << "  gmres_tolerance=2e-10 restart=0 cap=160\n";
    for (int N : levels) {
        const double h = kBoxSide / static_cast<double>(N);
        for (const auto& study_case : study_cases) {
            const auto setup_start = std::chrono::steady_clock::now();
            CartesianGrid3D grid({kBoxMin, kBoxMin, kBoxMin},
                                 {h, h, h}, {N, N, N}, DofLayout3D::Node);
            GeometryBundle geometry = make_geometry(
                GeometryKind::LPrism, h, study_case.transform);
            const auto domain = std::make_shared<const
                geometry3d::NurbsCartesianDomain3D>(
                    grid, geometry.native_surface.geometry_model());
            const SurfaceDofCloud surface_dofs =
                app3d::make_native_surface_dofs_3d(
                    geometry.native_surface, h);
            validate_surface_dofs(surface_dofs, h);
            const CauchyStencilSet cauchy_stencils = build_cauchy_stencils(
                geometry.native_surface, surface_dofs, h,
                kCauchyValueNeighborCount,
                kCauchyDerivativeNeighborCount,
                CauchyStencilPolicy3D::G1Nearest);
            GridPair3D grid_pair(grid,
                                 geometry.correction_interface,
                                 geometry.crossing_interface,
                                 domain,
                                 panel_nurbs_patch_indices(
                                     geometry.correction_triangles));
            for (int node = 0; node < grid.num_dofs(); ++node) {
                const bool numerical_inside = grid_pair.domain_label(node) > 0;
                if (numerical_inside
                    != geometry.exact_inside(grid_point(grid, node))) {
                    throw std::runtime_error(
                        "restrict-probe native NURBS label mismatch");
                }
            }
            const auto pipeline_start = std::chrono::steady_clock::now();
            PanelCenterHarmonicJetKFBI3D pipeline(
                grid, grid_pair, geometry.native_surface,
                geometry.correction_triangles,
                geometry.geometry_triangles,
                surface_dofs, cauchy_stencils, true, true);
            const double pipeline_setup_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - pipeline_start).count();

            NormalRestrictCaseProbe3D probe_case;
            probe_case.case_id = study_case.id;
            probe_case.N = N;
            probe_case.h = h;
            probe_case.wrong_side_support_count =
                pipeline.joint_trace_wrong_side_node_counts();
            probe_case.owner_diagnostics =
                pipeline.restrict_owner_sample_diagnostics();
            probe_case.owner_audit_records =
                pipeline.restrict_owner_audit_records();
            probe_case.owner_geometry_query_count =
                pipeline.restrict_owner_geometry_query_count();
            probe_case.pipeline_setup_seconds = pipeline_setup_seconds;
            if (probe_case.owner_diagnostics.empty()
                || probe_case.owner_geometry_query_count == 0) {
                throw std::runtime_error(
                    "restrict-probe crossing-owner diagnostics are empty");
            }
            std::size_t classified_queries = 0;
            for (const RestrictOwnerSampleDiagnostics3D& diagnostic
                 : probe_case.owner_diagnostics) {
                classified_queries +=
                    static_cast<std::size_t>(diagnostic.wrong_side_count);
            }
            if (classified_queries != probe_case.owner_geometry_query_count)
                throw std::logic_error(
                    "crossing-owner probe query count is inconsistent");
            const int size = pipeline.surface_size();
            probe_case.patch_ids.reserve(static_cast<std::size_t>(size));
            probe_case.points.reserve(static_cast<std::size_t>(size));
            probe_case.weights.reserve(static_cast<std::size_t>(size));
            probe_case.feature_edge_distance_over_h.reserve(
                static_cast<std::size_t>(size));
            if (geometry.feature_edge_segments.empty())
                throw std::runtime_error("L-prism has no feature-edge segments");
            for (const SurfaceDof& dof : pipeline.surface().dofs) {
                probe_case.patch_ids.push_back(dof.patch_id);
                probe_case.points.push_back(dof.point);
                probe_case.weights.push_back(dof.weight);
                double distance = std::numeric_limits<double>::infinity();
                for (const auto& edge : geometry.feature_edge_segments) {
                    distance = std::min(
                        distance, point_segment_distance_3d(dof.point, edge));
                }
                probe_case.feature_edge_distance_over_h.push_back(distance / h);
            }

            Eigen::VectorXd value_data(size);
            Eigen::VectorXd exact_normal(size);
            const Eigen::VectorXd zero_jump = Eigen::VectorXd::Zero(size);
            for (int q = 0; q < size; ++q) {
                const SurfaceDof& dof =
                    pipeline.surface().dofs[static_cast<std::size_t>(q)];
                value_data[q] =
                    app3d::transformed_manufactured_harmonic_value_3d(
                        study_case.transform, dof.point);
                exact_normal[q] =
                    app3d::transformed_manufactured_harmonic_gradient_3d(
                        study_case.transform, dof.point).dot(dof.normal);
            }
            Eigen::VectorXd piecewise_exact(grid.num_dofs());
            Eigen::VectorXd smooth_exact(grid.num_dofs());
            for (int node = 0; node < grid.num_dofs(); ++node) {
                const double value =
                    app3d::transformed_manufactured_harmonic_value_3d(
                        study_case.transform, grid_point(grid, node));
                smooth_exact[node] = value;
                piecewise_exact[node] = grid_pair.domain_label(node) > 0
                    ? value : 0.0;
            }
            const HarmonicJetField3D piecewise_field =
                pipeline.field_from_grid_and_jumps(
                    piecewise_exact, value_data, exact_normal);
            const HarmonicJetField3D smooth_field =
                pipeline.field_from_grid_and_jumps(
                    smooth_exact, zero_jump, zero_jump);
            const Eigen::VectorXd common_rhs =
                make_common_normal_restrict_rhs(size);
            probe_case.setup_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - setup_start).count();
            results.push_back(std::move(probe_case));
            write_restrict_owner_probe_outputs(output_dir, results);
            NormalRestrictCaseProbe3D& stored = results.back();

            const std::array<ExteriorNormalRestrictMode3D, 3> modes{{
                ExteriorNormalRestrictMode3D::JointTricubicCauchy,
                ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner,
                ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic}};
            for (ExteriorNormalRestrictMode3D mode : modes) {
                const auto route_start = std::chrono::steady_clock::now();
                NormalRestrictRouteProbe3D& route = stored.routes[
                    static_cast<std::size_t>(normal_restrict_route_index(mode))];
                route.route = normal_restrict_route_name(mode);
                route.conditions = summarize_conditions(
                    mode == ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic
                        ? pipeline.exterior_only_restrict_condition_values()
                        : pipeline.cauchy_condition_values());
                route.exact_grid_error = pipeline.exterior_normal_trace(
                    piecewise_field, value_data, exact_normal, mode);
                route.smooth_grid_error = pipeline.exterior_normal_trace(
                    smooth_field, zero_jump, zero_jump, mode) - exact_normal;
                ExteriorNormalTraceOperator3D op(pipeline, mode);
                Eigen::VectorXd applied;
                op.apply(exact_normal, applied);
                route.exact_equation_residual =
                    applied - op.right_hand_side(value_data);
                route.exact_grid_norms = normal_restrict_weighted_norms(
                    route.exact_grid_error, stored.weights);
                route.smooth_grid_norms = normal_restrict_weighted_norms(
                    route.smooth_grid_error, stored.weights);
                route.exact_equation_norms = normal_restrict_weighted_norms(
                    route.exact_equation_residual, stored.weights);
                route.physical = run_dirichlet_normal_case(
                    grid, grid_pair, pipeline, study_case.transform, 160,
                    2.0e-10, mode, &route.physical_residuals);
                route.common = run_common_normal_restrict_gmres(
                    pipeline, mode, common_rhs);
                route.physical_rho_10_30 =
                    residual_contraction(route.physical_residuals, 10, 30);
                route.physical_worst_rho_5 =
                    worst_five_step_contraction(route.physical_residuals);
                route.common_rho_10_30 =
                    residual_contraction(route.common.residuals, 10, 30);
                route.common_worst_rho_5 =
                    worst_five_step_contraction(route.common.residuals);
                if (pipeline.restrict_owner_geometry_query_count()
                    != stored.owner_geometry_query_count) {
                    throw std::runtime_error(
                        "GMRES apply performed a crossing-owner geometry query");
                }
                route.seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - route_start).count();
                route.complete = true;
                write_all_normal_restrict_probe_outputs(output_dir, results);
                std::cout << "[restrict-probe] case=" << stored.case_id
                          << " N=" << N << " route=" << route.route
                          << " exact_grid_linf="
                          << route.exact_grid_norms.linf
                          << " smooth_grid_linf="
                          << route.smooth_grid_norms.linf
                          << " equation_linf="
                          << route.exact_equation_norms.linf
                          << " physical_iter/error="
                          << route.physical.iterations << '/'
                          << route.physical.interior_linf
                          << " common_iter=" << route.common.iterations
                          << " seconds=" << route.seconds << '\n';
            }
            const double total_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - setup_start).count();
            std::cout << "[restrict-probe-case] case=" << stored.case_id
                      << " N=" << N << " dofs=" << size
                      << " setup_seconds=" << stored.setup_seconds
                      << " pipeline_setup_seconds="
                      << stored.pipeline_setup_seconds
                      << " total_seconds=" << total_seconds << '\n';
        }
    }
    const bool supported = normal_restrict_hypothesis_supported(results, levels);
    std::cout << "[restrict-hypothesis] "
              << (supported ? "supported" : "not_proven") << '\n'
              << "Restrict probe output: " << output_dir.string() << '\n';
    return 0;
}

void print_usage(const char* executable)
{
#ifdef KFBIM_3D_TOPOLOGY_AFFINE_DEFAULT
    constexpr const char* compiled_defaults =
        "topology_affine_local_svd, Neumann q27_cover3_all_event_cauchy, "
        "Dirichlet q64_cover4_all_event_cauchy, "
        "trace_mass, mean_free_pivot_elimination";
    constexpr const char* compatibility_default = "trace_border_legacy";
    constexpr const char* neumann_restrict_default =
        "q27_cover3_all_event_cauchy";
    constexpr const char* border_solver_default =
        "mean_free_pivot_elimination";
    constexpr const char* edge_jump_jet_default =
        "topology_affine_local_svd";
    constexpr const char* dirichlet_restrict_default =
        "q64_cover4_all_event_cauchy";
    constexpr const char* dirichlet_jump_space_default =
        "analytic_j0_affine_j1";
    constexpr const char* dirichlet_feature_coupling_default =
        "broken_sheets";
#elif defined(KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT)
    constexpr const char* compiled_defaults =
        "legacy_native, global_cubic_exterior_branch_crossing_owner, "
        "trace_mass, projected_elimination";
    constexpr const char* compatibility_default = "operator_flux";
    constexpr const char* neumann_restrict_default =
        "global_cubic_exterior_branch_crossing_owner";
    constexpr const char* border_solver_default = "projected_elimination";
    constexpr const char* edge_jump_jet_default = "disabled";
    constexpr const char* dirichlet_restrict_default =
        "joint_tricubic_crossing_owner";
    constexpr const char* dirichlet_jump_space_default =
        "legacy_sample_fit";
    constexpr const char* dirichlet_feature_coupling_default =
        "broken_sheets";
#else
    constexpr const char* compiled_defaults =
        "surface_samples, joint_tricubic_crossing_owner, c0_euclidean, "
        "augmented";
    constexpr const char* compatibility_default = "operator_flux";
    constexpr const char* neumann_restrict_default =
        "joint_tricubic_crossing_owner";
    constexpr const char* border_solver_default = "augmented";
    constexpr const char* edge_jump_jet_default = "disabled";
    constexpr const char* dirichlet_restrict_default =
        "joint_tricubic_crossing_owner";
    constexpr const char* dirichlet_jump_space_default =
        "legacy_sample_fit";
    constexpr const char* dirichlet_feature_coupling_default =
        "broken_sheets";
#endif
    std::cout
        << "usage: " << executable
        << " [torus|cylinder|l_prism|u_prism|all] [N ...]\n"
        << "       " << executable << " --rigid-study [N ...]\n"
        << "       " << executable << " --restrict-probe [N ...]\n"
        << "  Each N must be a power of two and at least 16 (default: 32).\n"
        << "  Rigid-study default levels: 32, 64, 128.\n"
        << "  Restrict-probe default levels: 32, 64.\n"
        << "  Compiled target defaults: " << compiled_defaults << ".\n"
        << "  KFBIM_3D_SUPPORT_PATH selects legacy (default) or native_certified.\n"
        << "  Native certification requires Q27/Q64 and fails closed; outputs\n"
        << "  use a separate native_endpoint directory. Optional positive\n"
        << "  KFBIM_3D_NATIVE_PATH_MAX_CANDIDATES / MAX_BOXES / MAX_NEWTON /\n"
        << "  MAX_BITS suffixes configure per-path work limits.\n"
        << "  This stage builds native NURBS density data, topology-filtered\n"
        << "  48/28 Cauchy stencils, validates\n"
        << "  fixed transfer routes, and executes the Neumann value-jump and\n"
        << "  Dirichlet normal-jump harmonic-jet GMRES formulations.\n"
        << "  KFBIM_3D_CAUCHY_POLICY selects g1_nearest (default),\n"
        << "  topological_nearest, same_patch, or balanced_patches.\n"
        << "  --rigid-study remains calibrated for and requires g1_nearest.\n"
        << "  KFBIM_3D_CAUCHY_VALUE_COUNT and\n"
        << "  KFBIM_3D_CAUCHY_NORMAL_COUNT select positive stencil counts\n"
        << "  (defaults: 48 and 28; normal count may not exceed value count).\n"
        << "  KFBIM_3D_FEATURE_TRACE_FIT selects geometric_feature "
           "(default)\n"
        << "  or legacy_joint for the value-jump Cauchy polynomial.\n"
        << "  KFBIM_3D_GMRES_MAX_ITERATIONS selects a positive GMRES cap\n"
        << "  for both formulations (default: 80).\n"
        << "  KFBIM_3D_GMRES_TOLERANCE selects a positive GMRES relative\n"
        << "  tolerance for ordinary and rigid-study solves (default: 2e-10).\n"
        << "  KFBIM_3D_NEUMANN_COMPATIBILITY selects operator_flux or\n"
        << "  trace_border_legacy (compiled default: "
        << compatibility_default << ").\n"
        << "  KFBIM_3D_NEUMANN_TRACE_RESTRICT selects\n"
        << "  global_cubic_exterior_branch_crossing_owner,\n"
        << "  joint_tricubic_crossing_owner, shared_quadratic_gridline_cauchy,\n"
        << "  shared_q10_cubic_gridline_cauchy, q27_cover3_all_event_cauchy,\n"
        << "  q64_cover4_all_event_cauchy, or joint_tricubic_cauchy\n"
        << "  (compiled default: " << neumann_restrict_default << ").\n"
        << "  The topology default uses one direct 3x3x3 Q2 cover for the\n"
        << "  Neumann value trace.  Every support-to-trace path processes its\n"
        << "  complete ordered event sequence before interpolation.  The old\n"
        << "  shared-Q10 route remains selectable only as a comparison route.\n"
        << "  The shared-Q10 route\n"
        << "  uses three\n"
        << "  points per normal side with one common 10-node complete-P2\n"
        << "  stencil; opposite-side nodes use the nearest support-to-trace\n"
        << "  root and nearest same-G1-sheet Cartesian crossing Cauchy jet.\n"
        << "  The global route converts\n"
        << "  all normal layers to the exterior branch with the complete\n"
        << "  local\n"
        << "  Cauchy polynomial before one global cubic trace fit.\n"
        << "  KFBIM_3D_NEUMANN_RESTRICT_VALUE_COUNT and\n"
        << "  KFBIM_3D_NEUMANN_RESTRICT_NORMAL_COUNT set the global-route\n"
        << "  local Cauchy counts (defaults: 24 and 12).\n"
        << "  KFBIM_3D_NEUMANN_RESTRICT_VALUE_JET selects c0_one_sided\n"
        << "  (native coefficient default) or feature_constrained.  It\n"
        << "  changes only the panel-centred value jet used by Neumann\n"
        << "  restrict; crossing-local spread and Dirichlet are unchanged.\n"
        << "  KFBIM_3D_NEUMANN_DENSITY_COORDINATES selects trace_mass\n"
        << "  (native coefficient default) or c0_euclidean.  trace_mass\n"
        << "  whitens the reduced density basis in the boundary L2 metric.\n"
        << "  KFBIM_3D_NEUMANN_BORDER_SOLVER selects projected_elimination,\n"
        << "  augmented, householder_elimination, or "
           "mean_free_pivot_elimination\n"
        << "  (compiled default: "
        << border_solver_default << ").  Native coefficient and legacy\n"
        << "  defaults remain projected_elimination and augmented.  The\n"
        << "  topology default removes one area-mean density coordinate,\n"
        << "  rebuilds the trace-mass projector on that K-1 space, and uses\n"
        << "  no augmented mean row, border column, or lambda.  The other\n"
        << "  elimination routes recover and\n"
        << "  audit the original bordered system after GMRES.\n"
        << "  KFBIM_3D_NEUMANN_EDGE_JUMP_JET selects disabled,\n"
        << "  strong_feature_mortar, topology_affine_local_svd, or\n"
        << "  topology_ambient_gradient_local_svd\n"
        << "  (compiled default: " << edge_jump_jet_default << ").  The\n"
        << "  ambient-gradient mode directly equates the two reconstructed\n"
        << "  world gradients in the two directions transverse to the edge;\n"
        << "  it does not divide by the dihedral sine.  The strong mode imposes the known\n"
        << "  Neumann jump-jet relation only at non-G1 feature edges and\n"
        << "  eliminates the intrinsic surface-mean nullspace directly.\n"
        << "  KFBIM_3D_NEUMANN_TRACE_SAMPLING selects panel_centers\n"
        << "  (default) or native_gauss.  native_gauss is an experimental\n"
        << "  diagnostic and is not stable at C0 seams.\n"
        << "  KFBIM_3D_DIRICHLET_NORMAL_RESTRICT selects "
           "joint_tricubic_crossing_owner,\n"
        << "  shared_quadratic_gridline_cauchy, "
           "shared_q10_cubic_gridline_cauchy,\n"
        << "  q27_cover3_all_event_cauchy, q64_cover4_all_event_cauchy,\n"
        << "  or joint_tricubic_cauchy (compiled default: "
        << dirichlet_restrict_default << ")\n"
        << "  for both the operator and RHS.  The topology default directly\n"
        << "  evaluates n dot grad of one 4x4x4 Q3 cover at the interface;\n"
        << "  it does not use Q10 normal layers or an a1/h recovery.\n"
        << "  KFBIM_3D_DIRICHLET_JUMP_SPACE selects legacy_sample_fit or\n"
        << "  analytic_j0_affine_j1 (compiled default: "
        << dirichlet_jump_space_default << ").  The analytic-affine route\n"
        << "  inserts known J0=g_D analytically and iterates only the\n"
        << "  homogeneous part of J1=c_p+Gz; it performs no g_D fit.\n"
        << "  KFBIM_3D_DIRICHLET_FEATURE_COUPLING selects broken_sheets or\n"
        << "  ambient_gradient_affine_mortar (compiled default: "
        << dirichlet_feature_coupling_default << ").  broken_sheets\n"
        << "  keeps J1 independent across physical C0 sheets; the mortar\n"
        << "  option adds the stronger common-ambient-gradient assumption.\n"
        << "  KFBIM_3D_SOLVE_SELECTION selects both (default), "
           "neumann_only,\n"
        << "  or dirichlet_normal_only.\n"
        << "  KFBIM_3D_DENSITY_MODE selects reduced_coefficients or\n"
        << "  surface_samples.  The dedicated native coefficient executable\n"
        << "  defaults to reduced_coefficients; the legacy executable keeps\n"
        << "  surface_samples.\n"
        << "  KFBIM_3D_DENSITY_COEFFICIENTS optionally overrides the\n"
        << "  geometry-only automatic cubic coefficient count (minimum 4).\n"
        << "  KFBIM_3D_RIGID_CASE selects a catalogued rigid transform for\n"
        << "  ordinary geometry runs (default: baseline; for example,\n"
        << "  rot_axis123_17deg_t_xyz_1).\n"
        << "  KFBIM_3D_OUTPUT_TAG appends a diagnostic output subdirectory.\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::cout << std::scientific << std::setprecision(6) << std::unitbuf;
        const bool rigid_study = argc >= 2
                              && std::string(argv[1]) == "--rigid-study";
        const bool restrict_probe = argc >= 2
                                  && std::string(argv[1]) == "--restrict-probe";
        const std::string approximation_scheme =
            approximation_scheme_name();
        std::string selection = "all";
        std::vector<int> levels = rigid_study
            ? std::vector<int>{32, 64, 128}
            : restrict_probe ? std::vector<int>{32, 64}
                             : std::vector<int>{32};
        if (argc >= 2 && !rigid_study && !restrict_probe)
            selection = argv[1];
        if (selection == "--help" || selection == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (restrict_probe && selected_native_endpoint_support_path()) {
            throw std::invalid_argument(
                "--restrict-probe is a legacy joint-tricubic diagnostic; "
                "native_certified requires the Q27/Q64 production routes");
        }
        if (argc >= 3) {
            levels.clear();
            for (int argument = 2; argument < argc; ++argument) {
                const int N = std::stoi(argv[argument]);
                if (N < 16 || !is_power_of_two(N)) {
                    throw std::invalid_argument(
                        "each N must be a power of two and at least 16");
                }
                levels.push_back(N);
            }
        }
        if ((rigid_study || restrict_probe)
            && approximation_scheme != "all_cubic") {
            throw std::invalid_argument(
                "--rigid-study and --restrict-probe require the "
                "all-cubic executable");
        }
        if (restrict_probe)
            return run_normal_restrict_causal_probe(levels);
        const CauchyStencilPolicy3D cauchy_policy = selected_cauchy_policy();
        const CauchyValueTraceFitMode3D cauchy_value_trace_fit_mode =
            selected_cauchy_value_trace_fit_mode();
        const NeumannCompatibilityMode3D neumann_compatibility_mode =
            selected_neumann_compatibility_mode();
        const int cauchy_value_count = positive_environment_integer(
            "KFBIM_3D_CAUCHY_VALUE_COUNT", kCauchyValueNeighborCount);
        const int cauchy_normal_count = positive_environment_integer(
            "KFBIM_3D_CAUCHY_NORMAL_COUNT", kCauchyDerivativeNeighborCount);
        const int gmres_max_iterations = positive_environment_integer(
            "KFBIM_3D_GMRES_MAX_ITERATIONS", 80);
        const double gmres_tolerance = positive_environment_double(
            "KFBIM_3D_GMRES_TOLERANCE", 2.0e-10);
        if (cauchy_normal_count > cauchy_value_count) {
            throw std::invalid_argument(
                "KFBIM_3D_CAUCHY_NORMAL_COUNT may not exceed "
                "KFBIM_3D_CAUCHY_VALUE_COUNT");
        }
        if (rigid_study) {
            return run_dirichlet_rigid_study(
                levels,
                cauchy_policy,
                cauchy_value_count,
                cauchy_normal_count,
                gmres_max_iterations,
                gmres_tolerance);
        }
        const ExteriorNormalRestrictMode3D neumann_trace_restrict_mode =
            selected_neumann_trace_restrict_mode();
        const int local_restrict_value_count =
            positive_environment_integer(
                "KFBIM_3D_NEUMANN_RESTRICT_VALUE_COUNT",
                kNeumannRestrictValueNeighborCount);
        const int local_restrict_normal_count =
            positive_environment_integer(
                "KFBIM_3D_NEUMANN_RESTRICT_NORMAL_COUNT",
                kNeumannRestrictNormalNeighborCount);
        const NeumannRestrictValueJetMode3D
            neumann_restrict_value_jet_mode =
                selected_neumann_restrict_value_jet_mode();
        const ExteriorNormalRestrictMode3D dirichlet_normal_restrict_mode =
            selected_dirichlet_normal_restrict_mode();
        const SolveSelection3D solve_selection = selected_solve_selection();
        const DensityIterationMode3D density_iteration_mode =
            selected_density_iteration_mode();
        const NeumannEdgeJumpJetMode3D neumann_edge_jump_jet_mode =
            selected_neumann_edge_jump_jet_mode();
        const DirichletJumpSpaceMode3D dirichlet_jump_space_mode =
            selected_dirichlet_jump_space_mode();
        const DirichletFeatureCouplingMode3D
            dirichlet_feature_coupling_mode =
                selected_dirichlet_feature_coupling_mode();
        if (solve_selection == SolveSelection3D::Both
            && trace_restrict_uses_shared_quadratic(
                   neumann_trace_restrict_mode)
            && trace_restrict_uses_shared_quadratic(
                   dirichlet_normal_restrict_mode)
            && trace_restrict_uses_topology_affine_cubic(
                   neumann_trace_restrict_mode)
                   != trace_restrict_uses_topology_affine_cubic(
                          dirichlet_normal_restrict_mode)) {
            throw std::invalid_argument(
                "Neumann and Dirichlet shared-Q10 routes must use the same "
                "normal-layer recovery profile in one run");
        }
        if (solve_selection != SolveSelection3D::DirichletNormalOnly
            && neumann_edge_jump_jet_mode
                   != NeumannEdgeJumpJetMode3D::Disabled
            && density_iteration_mode
                   != DensityIterationMode3D::ReducedCoefficients) {
            throw std::invalid_argument(
                "feature-edge affine constraints require "
                "reduced_coefficients density iteration");
        }
        if (solve_selection != SolveSelection3D::NeumannOnly
            && dirichlet_jump_space_mode
                   == DirichletJumpSpaceMode3D::AnalyticJ0AffineJ1
            && density_iteration_mode
                   != DensityIterationMode3D::ReducedCoefficients) {
            throw std::invalid_argument(
                "analytic_j0_affine_j1 Dirichlet requires "
                "reduced_coefficients density iteration");
        }
        if (solve_selection != SolveSelection3D::NeumannOnly
            && dirichlet_jump_space_mode
                   == DirichletJumpSpaceMode3D::AnalyticJ0AffineJ1
            && !trace_restrict_uses_all_event_path(
                   dirichlet_normal_restrict_mode)) {
            throw std::invalid_argument(
                "analytic_j0_affine_j1 Dirichlet requires "
                "an all-event direct-coefficient normal restrict");
        }
        if (solve_selection != SolveSelection3D::NeumannOnly
            && dirichlet_feature_coupling_mode
                   == DirichletFeatureCouplingMode3D::
                          AmbientGradientAffineMortar
            && dirichlet_jump_space_mode
                   != DirichletJumpSpaceMode3D::AnalyticJ0AffineJ1) {
            throw std::invalid_argument(
                "ambient-gradient Dirichlet feature coupling requires "
                "analytic_j0_affine_j1 jump space");
        }
        if (neumann_trace_restrict_mode
                == ExteriorNormalRestrictMode3D::
                       GlobalCubicExteriorBranchCrossingOwner
            && density_iteration_mode
                   != DensityIterationMode3D::ReducedCoefficients) {
            throw std::invalid_argument(
                "global exterior-branch Neumann restrict currently "
                "requires reduced_coefficients density iteration");
        }
        const bool selected_all_event_restrict =
            (solve_selection != SolveSelection3D::DirichletNormalOnly
             && trace_restrict_uses_all_event_path(
                    neumann_trace_restrict_mode))
            || (solve_selection != SolveSelection3D::NeumannOnly
                && trace_restrict_uses_all_event_path(
                       dirichlet_normal_restrict_mode));
        if (selected_all_event_restrict
            && density_iteration_mode
                   != DensityIterationMode3D::ReducedCoefficients) {
            throw std::invalid_argument(
                "all-event Cauchy restrict requires "
                "reduced_coefficients density iteration so spread and "
                "restrict use the same crossing-local Cauchy extension");
        }
        if (neumann_trace_restrict_mode
            == ExteriorNormalRestrictMode3D::
                   GlobalCubicExteriorBranchCrossingOwner) {
            if (local_restrict_value_count > cauchy_value_count
                || local_restrict_normal_count > cauchy_normal_count) {
                throw std::invalid_argument(
                    "Neumann restrict Cauchy counts may not exceed the full "
                    "Cauchy stencil counts");
            }
            constexpr int harmonic_cauchy_dimension =
                (kCauchyPolynomialDegree + 1)
                * (kCauchyPolynomialDegree + 1);
            if (local_restrict_value_count + local_restrict_normal_count
                < harmonic_cauchy_dimension) {
                throw std::invalid_argument(
                    "Neumann restrict Cauchy fit has fewer conditions than "
                    "harmonic coefficients");
            }
        }
        const app3d::DirichletRigidStudyCase3D rigid_case =
            selected_ordinary_rigid_case();

        std::vector<GeometryKind> geometries;
        if (selection == "all") {
#ifdef KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT
            geometries = {GeometryKind::HollowCylinder,
                          GeometryKind::LPrism,
                          GeometryKind::UPrism};
#else
            geometries = {GeometryKind::Torus,
                          GeometryKind::HollowCylinder,
                          GeometryKind::LPrism,
                          GeometryKind::UPrism};
#endif
        } else {
            geometries = {parse_geometry(selection)};
        }

        std::filesystem::path output_dir =
            application_output_root_3d()
#ifdef KFBIM_3D_TOPOLOGY_AFFINE_DEFAULT
            / "kfbi_topology_affine_3d";
#elif defined(KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT)
            / "kfbi_native_c0_exterior_trace_3d";
#else
            / "neumann_exterior_zero_trace_3d";
#endif
#ifdef KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT
        // Keep an explicit point-DOF override away from the executable's
        // canonical coefficient convergence outputs.
        if (density_iteration_mode == DensityIterationMode3D::SurfaceSamples)
            output_dir /= "surface_samples";
#else
        if (density_iteration_mode
            == DensityIterationMode3D::ReducedCoefficients) {
            output_dir /= "reduced_coefficients";
        }
#endif
        if (cauchy_policy != CauchyStencilPolicy3D::G1Nearest)
            output_dir /= cauchy_policy_name(cauchy_policy);
        if (cauchy_value_count != kCauchyValueNeighborCount
            || cauchy_normal_count != kCauchyDerivativeNeighborCount) {
            output_dir /= "v" + std::to_string(cauchy_value_count) + "_n"
                          + std::to_string(cauchy_normal_count);
        }
        if (approximation_scheme != "all_cubic")
            output_dir /= approximation_scheme;
        if (cauchy_value_trace_fit_mode
            != CauchyValueTraceFitMode3D::GeometricFeature) {
            output_dir /= cauchy_value_trace_fit_mode_name(
                cauchy_value_trace_fit_mode);
        }
        if (neumann_trace_restrict_mode
            != ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner) {
            if (neumann_trace_restrict_mode
                == ExteriorNormalRestrictMode3D::
                       GlobalCubicExteriorBranchCrossingOwner) {
                output_dir /= "global_exterior_branch";
            } else if (neumann_trace_restrict_mode
                       == ExteriorNormalRestrictMode3D::
                               SharedQ10CubicGridlineCauchy) {
                output_dir /= "tac_q10";
            } else if (neumann_trace_restrict_mode
                       == ExteriorNormalRestrictMode3D::
                              Q27Cover3AllEventCauchy) {
                output_dir /= "q27_c3";
            } else if (neumann_trace_restrict_mode
                       == ExteriorNormalRestrictMode3D::
                              Q64Cover4AllEventCauchy) {
                output_dir /= "q64_c4";
            } else {
                output_dir /= trace_restrict_mode_name(
                    neumann_trace_restrict_mode);
            }
        }
        if (neumann_trace_restrict_mode
            == ExteriorNormalRestrictMode3D::
                   GlobalCubicExteriorBranchCrossingOwner) {
            output_dir /= "restrict_v"
                        + std::to_string(local_restrict_value_count)
                        + "_n"
                        + std::to_string(local_restrict_normal_count);
        }
        if (neumann_restrict_value_jet_mode
            != default_neumann_restrict_value_jet_mode()) {
            output_dir /= "neumann_"
                        + neumann_restrict_value_jet_mode_name(
                              neumann_restrict_value_jet_mode);
        }
        if (solve_selection != SolveSelection3D::NeumannOnly
            && dirichlet_normal_restrict_mode
            != ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner) {
            if (dirichlet_normal_restrict_mode
                == ExteriorNormalRestrictMode3D::Q64Cover4AllEventCauchy) {
                output_dir /= "d_q64_c4";
            } else if (dirichlet_normal_restrict_mode
                       == ExteriorNormalRestrictMode3D::
                              Q27Cover3AllEventCauchy) {
                output_dir /= "d_q27_c3";
            } else {
                output_dir /= "dirichlet_"
                            + trace_restrict_mode_name(
                                  dirichlet_normal_restrict_mode);
            }
        }
        if (solve_selection != SolveSelection3D::DirichletNormalOnly) {
            // Keep this segment compact for Windows builds rooted in a long
            // project path.  The unabbreviated modes are recorded in both
            // the console report and solve CSV.
            output_dir /= neumann_output_mode_name();
        }
        if (solve_selection != SolveSelection3D::DirichletNormalOnly
            && neumann_edge_jump_jet_mode
                   == NeumannEdgeJumpJetMode3D::StrongFeatureMortar) {
            output_dir /= "edge_jj";
        } else if (solve_selection != SolveSelection3D::DirichletNormalOnly
                   && (neumann_edge_jump_jet_mode
                           == NeumannEdgeJumpJetMode3D::
                                  TopologyAffineLocalSvd
                       || neumann_edge_jump_jet_mode
                           == NeumannEdgeJumpJetMode3D::
                                  TopologyAmbientGradientLocalSvd)) {
            output_dir /= (neumann_edge_jump_jet_mode
                        == NeumannEdgeJumpJetMode3D::
                               TopologyAmbientGradientLocalSvd
                    ? "topology_ambient_gradient"
                    : "topology_affine");
        }
        if (rigid_case.id != "baseline") {
            // The longest catalog ID puts several diagnostic CSV paths at
            // the classic Windows MAX_PATH boundary.  Keep its directory
            // code short while reporting the complete ID in the console.
            output_dir /= rigid_case.id
                    == "rot_axis123_17deg_t_xyz_1"
                ? "rigid_r17_t1"
                : "rigid_" + rigid_case.id;
        }
        const char* output_tag = std::getenv("KFBIM_3D_OUTPUT_TAG");
        if (output_tag != nullptr && !std::string(output_tag).empty())
            output_dir /= output_tag;

        std::cout << "KFBI3D harmonic-jet convergence study\n"
                  << "  support_path="
                  << (selected_native_endpoint_support_path()
                        ? "native_certified" : "legacy") << '\n'
                  << "  Neumann target: exterior value trace = 0\n"
                  << "  Dirichlet target: exterior normal trace = 0\n"
                  << "  current stage: native NURBS surface DOFs + "
                     "selected Cauchy neighborhoods + G1 parameter-owned routes\n"
                  << "  density_iteration="
                  << density_iteration_mode_name(density_iteration_mode)
                  << " (value: feature C0; normal: feature broken; "
                     "G1: C0+weak-C1)\n"
                  << "  rigid_case=" << rigid_case.id << '\n'
                  << "  cauchy_policy=" << cauchy_policy_name(cauchy_policy)
                  << '\n'
                  << "  value_trace_fit="
                  << cauchy_value_trace_fit_mode_name(
                         cauchy_value_trace_fit_mode)
                  << '\n'
                  << "  approximation_scheme=" << approximation_scheme
                  << '\n'
                  << "  neumann_compatibility="
                  << neumann_compatibility_mode_name(
                         neumann_compatibility_mode)
                  << '\n'
                  << "  neumann_trace_restrict="
                  << trace_restrict_mode_name(neumann_trace_restrict_mode)
                  << '\n'
                  << "  neumann_restrict_value_jet="
                  << neumann_restrict_value_jet_mode_name(
                         neumann_restrict_value_jet_mode)
                  << '\n'
                  << "  neumann_trace_sampling="
                  << neumann_trace_sampling_name() << '\n'
                  << "  neumann_density_coordinates="
                  << neumann_density_coordinates_name() << '\n'
                  << "  neumann_border_solver="
                  << neumann_border_solver_name() << '\n'
                  << "  neumann_edge_jump_jet="
                  << neumann_edge_jump_jet_mode_name() << '\n'
                  << "  neumann_restrict_cauchy_counts="
                  << local_restrict_value_count << '/'
                  << local_restrict_normal_count << '\n'
                  << "  dirichlet_normal_restrict="
                  << trace_restrict_mode_name(
                         dirichlet_normal_restrict_mode)
                  << " solve_selection="
                  << solve_selection_name(solve_selection) << '\n'
                  << "  dirichlet_jump_space="
                  << dirichlet_jump_space_mode_name(
                         dirichlet_jump_space_mode)
                  << '\n'
                  << "  dirichlet_feature_coupling="
                  << dirichlet_feature_coupling_mode_name(
                         dirichlet_feature_coupling_mode)
                  << '\n'
                  << "  cauchy_degree=" << kCauchyPolynomialDegree
                  << " restrict_grid_degree=" << kRestrictGridDegree
                  << " restrict_normal_degree=" << kRestrictNormalDegree
                  << '\n'
                  << "  cauchy_counts=" << cauchy_value_count << '/'
                  << cauchy_normal_count << '\n'
                  << "  gmres_max_iterations=" << gmres_max_iterations
                  << " tolerance=" << gmres_tolerance << '\n'
                  << "  levels=";
        for (std::size_t index = 0; index < levels.size(); ++index) {
            if (index != 0)
                std::cout << ',';
            std::cout << levels[index];
        }
        std::cout << '\n';

        std::vector<ReadinessResult> results;
        for (int N : levels) {
            for (GeometryKind geometry : geometries) {
                results.push_back(run_readiness_case(
                    geometry,
                    N,
                    output_dir,
                    cauchy_policy,
                    cauchy_value_count,
                    cauchy_normal_count,
                    local_restrict_value_count,
                    local_restrict_normal_count,
                    gmres_max_iterations,
                    gmres_tolerance,
                    neumann_compatibility_mode,
                    neumann_trace_restrict_mode,
                    neumann_restrict_value_jet_mode,
                    dirichlet_normal_restrict_mode,
                    rigid_case.transform,
                    solve_selection));
                write_summary(output_dir, results);
                write_solve_summaries(output_dir, results, solve_selection);
#ifdef KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT
                if (density_iteration_mode
                    == DensityIterationMode3D::ReducedCoefficients) {
                    write_coefficient_convergence_results(
                        output_dir, results);
                }
#endif

                const ReadinessResult& result = results.back();
                const SolveMetrics3D* failed_solve = nullptr;
                if (solve_selection
                        != SolveSelection3D::DirichletNormalOnly
                    && !result.neumann.converged) {
                    failed_solve = &result.neumann;
                } else if (solve_selection
                               != SolveSelection3D::NeumannOnly
                           && !result.dirichlet_normal.converged) {
                    failed_solve = &result.dirichlet_normal;
                }
                if (failed_solve != nullptr) {
                    std::ostringstream message;
                    message
                        << "3D GMRES did not converge: geometry="
                        << result.geometry
                        << " N=" << result.N
                        << " formulation=" << failed_solve->formulation
                        << " iterations=" << failed_solve->iterations
                        << " final_relative_residual="
                        << std::scientific << std::setprecision(17)
                        << failed_solve->gmres_relative_residual;
                    throw std::runtime_error(message.str());
                }
            }
        }

        bool all_sample_conditions_met = true;
        for (const ReadinessResult& result : results) {
            if (solve_selection != SolveSelection3D::DirichletNormalOnly)
                all_sample_conditions_met = all_sample_conditions_met
                    && result.neumann.physical_converged;
            if (solve_selection != SolveSelection3D::NeumannOnly)
                all_sample_conditions_met = all_sample_conditions_met
                    && result.dirichlet_normal.physical_converged;
        }
        std::cout
            << "Geometry checks and selected GMRES formulations completed.\n";
        if (density_iteration_mode
            == DensityIterationMode3D::ReducedCoefficients) {
            std::cout
                << "  Projected coefficient systems converged; raw "
                   "surface-sample conditions are approximation diagnostics"
                   " (all_met="
                << all_sample_conditions_met << ").\n";
        } else {
            std::cout << "  Raw surface-sample conditions met="
                      << all_sample_conditions_met << ".\n";
        }
        std::cout << "Output: " << output_dir.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
