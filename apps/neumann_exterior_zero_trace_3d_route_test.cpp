#define main neumann_exterior_zero_trace_3d_application_main
#include "neumann_exterior_zero_trace_3d.cpp"
#undef main

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void test_exterior_trace_legacy_and_crossing_owner_routes()
{
    constexpr int N = 16;
    const double h = kBoxSide / static_cast<double>(N);
    CartesianGrid3D grid({kBoxMin, kBoxMin, kBoxMin}, {h, h, h},
                         {N, N, N}, DofLayout3D::Node);
    GeometryBundle geometry = make_geometry(
        GeometryKind::LPrism, h, app3d::RigidTransform3D());
    const auto domain = std::make_shared<const geometry3d::NurbsCartesianDomain3D>(
        grid, geometry.native_surface.geometry_model());
    const SurfaceDofCloud surface_dofs =
        app3d::make_native_surface_dofs_3d(geometry.native_surface, h);
    const CauchyStencilSet cauchy_stencils = build_cauchy_stencils(
        geometry.native_surface, surface_dofs, h,
        kCauchyValueNeighborCount, kCauchyDerivativeNeighborCount,
        CauchyStencilPolicy3D::G1Nearest);
    GridPair3D grid_pair(grid, geometry.correction_interface,
                         geometry.crossing_interface, domain);
    PanelCenterHarmonicJetKFBI3D pipeline(
        grid, grid_pair, geometry.native_surface, geometry.correction_triangles,
        geometry.geometry_triangles, surface_dofs, cauchy_stencils,
        false, true);

    const std::vector<RestrictOwnerAuditRecord3D> reroutes =
        pipeline.restrict_owner_audit_records();
    require(!reroutes.empty(),
            "owner-enabled L-prism fixture has foreign non-G1 reroutes");

    HarmonicJetField3D field;
    field.potential = Eigen::VectorXd::Zero(grid.num_dofs());
    field.coefficients = Eigen::MatrixXd::Zero(
        pipeline.surface_size(),
        app3d::HarmonicPolynomialSpace3D(kCauchyPolynomialDegree).dimension());
    for (int row = 0; row < field.coefficients.rows(); ++row)
        field.coefficients(row, 0) = static_cast<double>(row + 1);
    const Eigen::VectorXd zero_jump =
        Eigen::VectorXd::Zero(pipeline.surface_size());

    const Eigen::VectorXd legacy =
        pipeline.exterior_trace(field, zero_jump, zero_jump);
    const Eigen::VectorXd cauchy = pipeline.exterior_trace(
        field, zero_jump, zero_jump,
        app3d::ExteriorValueRestrictMode3D::JointTricubicCauchy);
    const Eigen::VectorXd crossing = pipeline.exterior_trace(
        field, zero_jump, zero_jump,
        app3d::ExteriorValueRestrictMode3D::JointTricubicCrossingOwner);

    require((legacy - cauchy).lpNorm<Eigen::Infinity>() < 1.0e-14,
            "legacy exterior_trace delegates to explicit Cauchy mode");
    require((crossing - cauchy).lpNorm<Eigen::Infinity>() > 1.0e-12,
            "crossing exterior_trace uses foreign owner corrections");
}

} // namespace

int main()
{
    try {
        test_exterior_trace_legacy_and_crossing_owner_routes();
        std::cout << "Neumann exterior value route integration test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Neumann exterior value route integration test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
