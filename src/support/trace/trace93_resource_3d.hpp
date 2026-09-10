#pragma once

#include "src/support/trace/trace_first_resource_3d.hpp"

namespace kfbim::app3d {
struct Trace93Case3D;
struct Trace93DensityLayout3D;

// Setup-only catalog over the Python analysis coefficient space. Anchor u/v
// remain native geometry coordinates; they are NEVER density parameters.
// Trace centers use the layout's exact analysis coordinates, and real event
// centers are mapped from their physical point into their owner's chart.
class Trace93PolynomialCatalog3D {
public:
    Trace93PolynomialCatalog3D(const Trace93Case3D& problem,
        const Trace93DensityLayout3D& density,
        const std::vector<RestrictResourceAnchor3D>& traces,
        const std::vector<RestrictResourceAnchor3D>& events,
        bool neumann,double neumann_mean_removed=0.0,bool reuse_rows=true);
    ~Trace93PolynomialCatalog3D();
    Trace93PolynomialCatalog3D(Trace93PolynomialCatalog3D&&) noexcept;
    Trace93PolynomialCatalog3D& operator=(Trace93PolynomialCatalog3D&&) noexcept;
    Trace93PolynomialCatalog3D(const Trace93PolynomialCatalog3D&)=delete;
    Trace93PolynomialCatalog3D& operator=(const Trace93PolynomialCatalog3D&)=delete;
    void prepare(int degree,TraceFirstAnchorRef3D anchor);
    [[nodiscard]] ResourceAffineRow3D evaluate(int degree,TraceFirstAnchorRef3D anchor,
        const Eigen::Vector3d& target,int full_grid_id=-1);
    [[nodiscard]] const TracePolynomialCatalogStatistics3D& statistics() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Strict 93 charts: planar Euclidean graph or cylinder angular/axial/radial
// coordinates. The old extended rational-NURBS chart is not substituted.
// The geometry plan may use Python93LastEvent or the certified all-event
// extension; this function does not modify its event/topology decisions.
[[nodiscard]] ResourceBvpOperators3D build_trace93_bvp_operators_3d(
    const TraceFirstGeometryPlan3D& geometry,const CartesianGrid3D& grid,
    const GridPair3D& pair,const LaplaceCorrectionSupport3D& support,
    const Trace93Case3D& problem,const Trace93DensityLayout3D& density,
    bool neumann,double neumann_mean_removed=0.0,
    TraceFirstBuildStatistics3D* statistics=nullptr);
} // namespace kfbim::app3d
