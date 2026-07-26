#pragma once

#include <apps/harmonic_polynomial_space_3d.hpp>
#include <apps/native_nurbs_surface_3d.hpp>

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace kfbim::app3d {

enum class HarmonicCauchyRoute3D {
    G1ValueG1Normal,
    DirectCrossFaceValue,
    EdgeReconstructedValue
};

enum class LegacySurfaceCauchyPolicy3D {
    G1Nearest,
    TopologicalNearest,
    SamePatch,
    BalancedPatches
};

struct SharedEdgePoint3D {
    int id = -1;
    int connection_id = -1;
    int cell_id = -1;
    int cell_count = 0;
    double fraction = 0.0;
    double quadrature_weight = 0.0;
    std::array<double, 2> native_parameters{{0.0, 0.0}};
    std::array<Eigen::Vector2d, 2> native_uv;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangent = Eigen::Vector3d::Zero();
    Eigen::Matrix3d frame = Eigen::Matrix3d::Identity();
    std::array<std::vector<int>, 2> sector_patch_ids;
};

struct SurfaceNonG1EdgeDistance3D {
    int connection_id = -1;
    double distance = std::numeric_limits<double>::infinity();
    std::array<double, 2> native_parameters{{0.0, 0.0}};
    Eigen::Vector3d closest_point = Eigen::Vector3d::Zero();
    double distance_error_bound = std::numeric_limits<double>::infinity();
};

struct SurfaceNonG1EdgeNeighborhood3D {
    int center_dof = -1;
    double nearest_distance_over_h =
        std::numeric_limits<double>::infinity();
    std::vector<SurfaceNonG1EdgeDistance3D> incident_distances;
    std::vector<int> relevant_connection_ids;
};

struct SurfaceNonG1EdgeNeighborhoodSet3D {
    std::vector<SurfaceNonG1EdgeNeighborhood3D> centers;
    std::size_t geometry_query_count = 0;
    std::uint64_t fingerprint = 0;
};

struct HarmonicCauchyFailure3D {
    std::string stage;
    std::string entity_kind;
    int entity_id = -1;
    int connection_id = -1;
    std::vector<std::vector<int>> incident_sectors;
    std::vector<int> actual_value_counts;
    std::vector<int> actual_normal_counts;
    int required_value_count = 0;
    int required_normal_count = 0;
    int actual_edge_count = 0;
    double value_radius_over_h = 0.0;
    double normal_radius_over_h = 0.0;
    double edge_radius_over_h = 0.0;
    double sigma_max = 0.0;
    double sigma_min = 0.0;
    double condition = 0.0;
    std::string message;
};

class HarmonicCauchyError3D : public std::runtime_error {
public:
    explicit HarmonicCauchyError3D(HarmonicCauchyFailure3D diagnostic);
    [[nodiscard]] const HarmonicCauchyFailure3D& diagnostic() const noexcept;

private:
    HarmonicCauchyFailure3D diagnostic_;
};

struct SharedEdgePointSet3D {
    std::vector<SharedEdgePoint3D> points;
    std::vector<std::vector<int>> point_ids_by_connection;
    double max_position_mismatch = 0.0;
    double min_mapped_tangent_dot = 1.0;
    std::uint64_t fingerprint = 0;
};

struct DirectCrossFaceSelection3D {
    std::vector<int> dof_ids;
    std::vector<int> relevant_connection_ids;
    std::vector<std::vector<int>> sector_patch_ids;
    std::vector<int> sector_sample_counts;
    double nearest_edge_distance_over_h =
        std::numeric_limits<double>::infinity();
};

struct SurfaceEdgePointSelection3D {
    std::vector<int> edge_point_ids;
    std::vector<int> relevant_connection_ids;
    double nearest_edge_distance_over_h =
        std::numeric_limits<double>::infinity();
};

struct EdgeValueMap3D {
    SharedEdgePoint3D point;
    std::array<std::vector<int>, 2> sector_patch_ids;
    std::array<int, 2> value_sector_counts{{0, 0}};
    std::array<int, 2> normal_sector_counts{{0, 0}};
    std::vector<int> value_ids;
    std::vector<int> normal_ids;
    Eigen::RowVectorXd E_value;
    Eigen::RowVectorXd E_normal;
    double value_radius_over_h = 0.0;
    double normal_radius_over_h = 0.0;
    double sigma_max = 0.0;
    double sigma_min = 0.0;
    double condition = 0.0;
};

struct SurfaceCauchyMap3D {
    std::vector<int> value_ids;
    std::vector<int> normal_ids;
    std::vector<int> edge_point_ids;
    std::vector<std::vector<int>> value_sector_patch_ids;
    std::vector<int> value_sector_counts;
    std::vector<std::vector<int>> normal_sector_patch_ids;
    std::vector<int> normal_sector_counts;
    std::vector<int> relevant_connection_ids;
    int incident_patch_count = 0;
    int value_patch_imbalance = 0;
    int normal_patch_imbalance = 0;
    Eigen::MatrixXd M_value;
    Eigen::MatrixXd M_normal;
    Eigen::MatrixXd M_edge;
    double value_radius_over_h = 0.0;
    double normal_radius_over_h = 0.0;
    double edge_radius_over_h = 0.0;
    double nearest_edge_distance_over_h =
        std::numeric_limits<double>::infinity();
    double sigma_max = 0.0;
    double sigma_min = 0.0;
    double condition = 0.0;
};

struct HarmonicCauchyApplyResult3D {
    Eigen::VectorXd edge_values;
    Eigen::MatrixXd coefficients;
    std::size_t runtime_geometry_query_count = 0;
    std::size_t runtime_svd_factorization_count = 0;
    std::uint64_t fingerprint_before = 0;
    std::uint64_t fingerprint_after = 0;
};

struct HarmonicCauchyPreprocessAudit3D {
    std::size_t geometry_query_count = 0;
    std::size_t svd_factorization_count = 0;
    std::uint64_t fingerprint = 0;
};

struct LegacyCauchySummary3D {
    LegacySurfaceCauchyPolicy3D policy =
        LegacySurfaceCauchyPolicy3D::G1Nearest;
    int degree = 0;
    int value_count = 0;
    int normal_count = 0;
    int value_count_min = 0;
    int value_count_max = 0;
    int normal_count_min = 0;
    int normal_count_max = 0;
    double radius_max_over_h = 0.0;
    double radius_mean_over_h = 0.0;
    int incident_patch_count_min = 0;
    int incident_patch_count_max = 0;
    int value_patch_imbalance_max = 0;
    int normal_patch_imbalance_max = 0;
};

[[nodiscard]] SharedEdgePointSet3D make_shared_edge_points_3d(
    const NativeNurbsSurface3D& surface,
    double h,
    int edge_length_parameter_samples = 64);

[[nodiscard]] SurfaceNonG1EdgeNeighborhoodSet3D
build_surface_non_g1_edge_neighborhoods_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h);

[[nodiscard]] DirectCrossFaceSelection3D
select_direct_cross_face_value_dofs_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
    int center_dof,
    int count,
    double h);

[[nodiscard]] SurfaceEdgePointSelection3D
select_surface_edge_points_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const SharedEdgePointSet3D& edge_points,
    const SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
    int center_dof,
    double h,
    int max_points_per_edge = 4);

class HarmonicCauchyFit3D {
public:
    static HarmonicCauchyFit3D build(
        const NativeNurbsSurface3D& surface,
        const SurfaceDofCloud3D& cloud,
        const SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
        double h,
        HarmonicCauchyRoute3D route,
        int degree = 3,
        int value_count = 48,
        int normal_count = 28,
        double relative_svd_cutoff = 3e-12);

    static HarmonicCauchyFit3D build_legacy(
        const NativeNurbsSurface3D& surface,
        const SurfaceDofCloud3D& cloud,
        double h,
        LegacySurfaceCauchyPolicy3D policy,
        int degree,
        int value_count,
        int normal_count,
        double relative_svd_cutoff = 3e-12);

    [[nodiscard]] HarmonicCauchyApplyResult3D apply(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const;
    [[nodiscard]] const HarmonicPolynomialSpace3D& space() const noexcept;
    [[nodiscard]] int degree() const noexcept;
    [[nodiscard]] int value_count() const noexcept;
    [[nodiscard]] int normal_count() const noexcept;
    [[nodiscard]] std::optional<LegacySurfaceCauchyPolicy3D>
    legacy_policy() const noexcept;
    [[nodiscard]] std::optional<LegacyCauchySummary3D>
    legacy_summary() const;
    [[nodiscard]] std::vector<double> condition_values() const;
    [[nodiscard]] const std::vector<EdgeValueMap3D>&
    edge_maps() const noexcept;
    [[nodiscard]] const std::vector<SurfaceCauchyMap3D>&
    surface_maps() const noexcept;
    [[nodiscard]] const HarmonicCauchyPreprocessAudit3D&
    audit() const noexcept;

private:
    explicit HarmonicCauchyFit3D(int degree);

    HarmonicPolynomialSpace3D space_;
    int value_count_ = 0;
    int normal_count_ = 0;
    int cloud_size_ = 0;
    HarmonicCauchyRoute3D route_ = HarmonicCauchyRoute3D::G1ValueG1Normal;
    std::optional<LegacySurfaceCauchyPolicy3D> legacy_policy_;
    std::optional<LegacyCauchySummary3D> legacy_summary_;
    std::vector<EdgeValueMap3D> edge_maps_;
    std::vector<SurfaceCauchyMap3D> surface_maps_;
    HarmonicCauchyPreprocessAudit3D audit_;
};

} // namespace kfbim::app3d
