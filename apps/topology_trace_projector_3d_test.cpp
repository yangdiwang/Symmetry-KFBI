#include "topology_trace_projector_3d.hpp"
#include "topology_mean_free_reduction_3d.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kfbim::app3d::AffineReduction3D;
using kfbim::app3d::ConstraintMeta3D;
using kfbim::app3d::ConstraintSystem3D;
using kfbim::app3d::ReducedTraceProjection3D;
using kfbim::app3d::ReducedTraceProjectionBackend3D;
using kfbim::app3d::SparseMatrixCSR3D;
using kfbim::app3d::TopologyTraceProjector3D;
using kfbim::app3d::eliminate_topology_mean_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_near(double actual,
                  double expected,
                  double tolerance,
                  const std::string& message)
{
    if (!std::isfinite(actual)
        || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": error=" + std::to_string(
                std::abs(actual - expected)));
    }
}

struct Fixture {
    SparseMatrixCSR3D test_design;
    SparseMatrixCSR3D base_expansion;
    AffineReduction3D reduction;
    Eigen::VectorXd weights;
};

Fixture make_sparse_fixture()
{
    constexpr int base_size = 8;
    constexpr int c0_size = 12;
    constexpr int reduced_size = 4;
    constexpr int samples = 20;

    std::mt19937 generator(1771U);
    std::uniform_real_distribution<double> random_value(-0.7, 0.7);

    std::vector<Eigen::Triplet<double>> e_entries;
    std::vector<Eigen::Triplet<double>> c_entries;
    Eigen::VectorXd particular(base_size);
    Eigen::VectorXd rhs(reduced_size);
    particular.setZero();
    for (int column = 0; column < reduced_size; ++column) {
        double a = random_value(generator);
        double b = random_value(generator);
        const double norm = std::hypot(a, b);
        a /= norm;
        b /= norm;
        e_entries.emplace_back(column, column, a);
        e_entries.emplace_back(column + reduced_size, column, b);

        // (-b,a) is the independent constraint direction in this pair.
        c_entries.emplace_back(column, column, -b);
        c_entries.emplace_back(column, column + reduced_size, a);
        rhs[column] = 0.15 * static_cast<double>(column + 1);
        particular[column] = -b * rhs[column];
        particular[column + reduced_size] = a * rhs[column];
    }

    SparseMatrixCSR3D homogeneous(base_size, reduced_size);
    homogeneous.setFromTriplets(e_entries.begin(), e_entries.end());
    homogeneous.makeCompressed();

    ConstraintSystem3D constraints;
    constraints.C.resize(reduced_size, base_size);
    constraints.C.setFromTriplets(c_entries.begin(), c_entries.end());
    constraints.C.makeCompressed();
    constraints.d = rhs;
    constraints.meta.resize(reduced_size);
    for (ConstraintMeta3D& meta : constraints.meta)
        meta.kind = "fixture";

    const Eigen::VectorXd particular_error =
        constraints.C * particular - rhs;
    const Eigen::MatrixXd homogeneous_error =
        Eigen::MatrixXd(constraints.C * homogeneous);
    const Eigen::MatrixXd orthogonality_error =
        Eigen::MatrixXd(homogeneous.transpose() * homogeneous)
        - Eigen::MatrixXd::Identity(reduced_size, reduced_size);
    const AffineReduction3D reduction(
        particular,
        homogeneous,
        constraints,
        {},
        particular_error.lpNorm<Eigen::Infinity>(),
        homogeneous_error.cwiseAbs().maxCoeff(),
        orthogonality_error.cwiseAbs().maxCoeff());

    std::vector<Eigen::Triplet<double>> a0_entries;
    for (int column = 0; column < base_size; ++column) {
        a0_entries.emplace_back(column, column, 1.0);
        a0_entries.emplace_back(
            (column + 3) % c0_size, column, 0.08 * random_value(generator));
        a0_entries.emplace_back(
            8 + column % 4, column, 0.35 * random_value(generator));
    }
    SparseMatrixCSR3D base_expansion(c0_size, base_size);
    base_expansion.setFromTriplets(a0_entries.begin(), a0_entries.end());
    base_expansion.makeCompressed();

    std::vector<Eigen::Triplet<double>> test_entries;
    for (int row = 0; row < c0_size; ++row)
        test_entries.emplace_back(row, row, 1.0);
    for (int row = c0_size; row < samples; ++row) {
        for (int entry = 0; entry < 4; ++entry) {
            const int column = (3 * row + 5 * entry) % c0_size;
            test_entries.emplace_back(row, column, random_value(generator));
        }
    }
    SparseMatrixCSR3D test_design(samples, c0_size);
    test_design.setFromTriplets(test_entries.begin(), test_entries.end());
    test_design.makeCompressed();

    Eigen::VectorXd weights(samples);
    for (int row = 0; row < samples; ++row)
        weights[row] = 0.3 + 0.025 * static_cast<double>(row + 1);

    return {test_design, base_expansion, reduction, weights};
}

void test_sparse_projection_and_trace_mass_roundtrip()
{
    const Fixture fixture = make_sparse_fixture();
    TopologyTraceProjector3D projector(
        fixture.test_design,
        fixture.base_expansion,
        fixture.reduction,
        fixture.weights);
    require(projector.backend()
                == ReducedTraceProjectionBackend3D::DenseWeightedQr,
            "small topology projector did not select weighted QR");
    require(projector.gram_nonzeros() == 0,
            "weighted-QR backend unexpectedly formed a dense Gram matrix");

    Eigen::VectorXd native(4);
    native << 0.25, -0.8, 1.1, 0.37;
    const Eigen::VectorXd mass = projector.native_to_trace_mass(native);
    const Eigen::VectorXd expected_c0 =
        projector.homogeneous_c0() * native;
    const Eigen::VectorXd lifted =
        projector.lift_homogeneous_c0(mass);
    require(
        (lifted - expected_c0).lpNorm<Eigen::Infinity>() < 2.0e-13,
        "trace-mass homogeneous lift did not recover G*z");

    const Eigen::VectorXd trace = fixture.test_design * lifted;
    const Eigen::VectorXd recovered_mass = projector.project(trace);
    require(
        (recovered_mass - mass).lpNorm<Eigen::Infinity>() < 2.0e-12,
        "project(Bz*z) did not recover trace-mass coordinates");

    require(projector.rank() == 4, "projector rank diagnostic is wrong");
    require(projector.condition() < 20.0,
            "fixture projector is unexpectedly ill-conditioned");
    require(projector.pb_identity_residual() < 2.0e-13,
            "P*B identity residual is too large");
    require(projector.full_roundtrip_residual() < 2.0e-13,
            "full sparse roundtrip residual is too large");

    // Exercise the sparse R overload directly and verify that requesting the
    // lazy dense compatibility map does not change subsequent apply calls.
    const Eigen::SparseMatrix<double> test_csc(fixture.test_design);
    const Eigen::SparseMatrix<double> homogeneous_csc(
        projector.homogeneous_c0());
    ReducedTraceProjection3D native_projector(
        test_csc, homogeneous_csc, fixture.weights);
    const Eigen::VectorXd recovered_native = native_projector.apply(trace);
    require(
        (recovered_native - native).lpNorm<Eigen::Infinity>() < 2.0e-12,
        "sparse composed projector did not recover native coordinates");

    // The new sparse-reduction route must remain numerically identical to
    // the pre-existing dense-reduction API.
    const Eigen::MatrixXd homogeneous_dense(homogeneous_csc);
    ReducedTraceProjection3D dense_reduction_projector(
        test_csc, homogeneous_dense, fixture.weights);
    require(
        (dense_reduction_projector.apply(trace) - recovered_native)
                .lpNorm<Eigen::Infinity>()
            < 2.0e-13,
        "sparse and dense reduction projector routes disagree");
    require_near(
        native_projector.condition(),
        dense_reduction_projector.condition(),
        2.0e-13,
        "sparse and dense reduction condition diagnostics disagree");

    const Eigen::MatrixXd& dense_projection =
        native_projector.projection_matrix();
    require(
        (dense_projection * trace - native).lpNorm<Eigen::Infinity>()
            < 2.0e-12,
        "lazy sparse composed projection matrix is incorrect");
    require(
        (native_projector.apply(trace) - native).lpNorm<Eigen::Infinity>()
            < 2.0e-12,
        "apply changed after lazy projection materialization");
}

void test_particular_is_separate_and_mean_dual_is_consistent()
{
    const Fixture fixture = make_sparse_fixture();
    TopologyTraceProjector3D projector(
        fixture.test_design,
        fixture.base_expansion,
        fixture.reduction,
        fixture.weights);

    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(4);
    require(projector.lift_homogeneous_c0(zero).norm() == 0.0,
            "affine particular leaked into the homogeneous lift");
    require(
        (projector.lift_full_c0(zero) - projector.particular_c0())
                .lpNorm<Eigen::Infinity>()
            < 1.0e-15,
        "full zero-coordinate lift did not equal the particular vector");
    require(projector.particular_c0().norm() > 0.0,
            "fixture particular vector unexpectedly vanished");

    Eigen::VectorXd mass(4);
    mass << -0.6, 0.2, 0.9, -0.3;
    Eigen::VectorXd mean_dual(projector.c0_coefficient_count());
    for (Eigen::Index row = 0; row < mean_dual.size(); ++row)
        mean_dual[row] = 0.07 * static_cast<double>(row + 1);
    const Eigen::VectorXd mass_dual =
        projector.mean_dual_to_trace_mass(mean_dual);
    const double direct = mean_dual.dot(projector.lift_full_c0(mass));
    const double transformed = projector.particular_mean(mean_dual)
        + mass_dual.dot(mass);
    require_near(direct, transformed, 2.0e-13,
                 "mean dual trace-mass transformation is inconsistent");
}

void test_final_mean_free_space_rebuilds_the_projector()
{
    const Fixture fixture = make_sparse_fixture();
    const TopologyTraceProjector3D original(
        fixture.test_design,
        fixture.base_expansion,
        fixture.reduction,
        fixture.weights);
    Eigen::VectorXd mean_dual =
        fixture.test_design.transpose() * fixture.weights;
    mean_dual /= fixture.weights.sum();
    const auto mean_free = eliminate_topology_mean_3d(
        original.particular_c0(), original.homogeneous_c0(), mean_dual);
    require(mean_free.source_coordinate_count() == 4
                && mean_free.reduced_coordinate_count() == 3,
            "final mean constraint did not remove one trace coordinate");

    const TopologyTraceProjector3D projector(
        fixture.test_design,
        mean_free.particular(),
        mean_free.homogeneous(),
        fixture.weights);
    require(projector.reduced_coordinate_count() == 3
                && projector.rank() == 3,
            "mean-free projector has the wrong final rank");
    require(std::abs(mean_dual.dot(projector.particular_c0())) < 2.0e-14,
            "mean-free projector particular does not satisfy the gauge");
    require((projector.homogeneous_c0().transpose() * mean_dual)
                    .lpNorm<Eigen::Infinity>()
                < 2.0e-14,
            "mean-free projector retained a nonzero mean direction");
    const Eigen::VectorXd projected_constant = projector.project(
        Eigen::VectorXd::Ones(fixture.test_design.rows()));
    require(projected_constant.lpNorm<Eigen::Infinity>() < 2.0e-12,
            "final mean-free residual projector does not annihilate constants");

    Eigen::Vector3d mass;
    mass << 0.35, -0.71, 1.2;
    const Eigen::VectorXd trace = fixture.test_design
        * projector.lift_homogeneous_c0(mass);
    require((projector.project(trace) - mass).lpNorm<Eigen::Infinity>()
                < 2.0e-12,
            "final K-1 trace-mass lift/project roundtrip failed");
}

void test_rank_deficiency_is_rejected()
{
    const Fixture fixture = make_sparse_fixture();
    SparseMatrixCSR3D zero_design(
        fixture.test_design.rows(), fixture.test_design.cols());
    zero_design.makeCompressed();
    bool rejected = false;
    try {
        (void)TopologyTraceProjector3D(
            zero_design,
            fixture.base_expansion,
            fixture.reduction,
            fixture.weights);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "rank-deficient sparse trace design was accepted");
}

void test_final_trace_space_requires_strict_oversampling()
{
    constexpr int coordinates = 3;
    const Eigen::MatrixXd square_dense =
        Eigen::MatrixXd::Identity(coordinates, coordinates);
    const Eigen::SparseMatrix<double> square_sparse =
        square_dense.sparseView();
    const Eigen::VectorXd square_weights =
        Eigen::VectorXd::Ones(coordinates);

    SparseMatrixCSR3D square_design = square_sparse;
    SparseMatrixCSR3D homogeneous = square_sparse;
    bool rejected = false;
    try {
        (void)TopologyTraceProjector3D(
            square_design,
            Eigen::VectorXd::Zero(coordinates),
            homogeneous,
            square_weights);
    } catch (const std::invalid_argument& error) {
        const std::string message(error.what());
        rejected = message.find("strictly more") != std::string::npos
            && message.find("samples=3") != std::string::npos
            && message.find("coordinates=3") != std::string::npos;
    }
    require(rejected,
            "final topology trace space accepted samples == coordinates");

    // ReducedTraceProjection3D is also used by legacy/intermediate density
    // transfers, for which a nonsingular square interpolation remains a
    // supported compatibility route.  The strict gate deliberately belongs
    // to the final topology-affine GMRES space.
    ReducedTraceProjection3D legacy_square(
        square_dense, square_weights, 1.0e-12, true);
    require(legacy_square.rank() == coordinates,
            "strict topology gate leaked into the generic square projector");

    // One additional positive-weight observation is exactly the accepted
    // boundary.  Rank and condition checks still execute after the count
    // gate.
    Eigen::MatrixXd oversampled_dense(coordinates + 1, coordinates);
    oversampled_dense.topRows(coordinates).setIdentity();
    oversampled_dense.bottomRows(1).setConstant(0.25);
    SparseMatrixCSR3D oversampled_design =
        oversampled_dense.sparseView();
    const Eigen::VectorXd oversampled_weights =
        Eigen::VectorXd::Ones(coordinates + 1);
    TopologyTraceProjector3D accepted(
        oversampled_design,
        Eigen::VectorXd::Zero(coordinates),
        homogeneous,
        oversampled_weights);
    require(accepted.sample_count() == coordinates + 1
                && accepted.reduced_coordinate_count() == coordinates
                && accepted.rank() == coordinates,
            "strictly oversampled topology trace space was not accepted");

    // A zero-weight row is not an effective observation.  The established
    // quadrature contract rejects it before the count gate rather than
    // allowing it to pad the sample total.
    Eigen::VectorXd zero_padded_weights = oversampled_weights;
    zero_padded_weights[coordinates] = 0.0;
    bool zero_weight_rejected = false;
    try {
        (void)TopologyTraceProjector3D(
            oversampled_design,
            Eigen::VectorXd::Zero(coordinates),
            homogeneous,
            zero_padded_weights);
    } catch (const std::invalid_argument& error) {
        zero_weight_rejected =
            std::string(error.what()).find("weights positive")
            != std::string::npos;
    }
    require(zero_weight_rejected,
            "a zero-weight pseudo-observation bypassed validation");
}

void test_large_sparse_spd_backend_without_dense_gram()
{
    const int coefficients =
        ReducedTraceProjection3D::sparse_cholesky_coordinate_threshold();
    const int samples = 2 * coefficients - 1;
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(3 * coefficients - 2);
    for (int column = 0; column < coefficients; ++column)
        entries.emplace_back(column, column, 1.0);
    for (int edge = 0; edge + 1 < coefficients; ++edge) {
        const int row = coefficients + edge;
        entries.emplace_back(row, edge, 0.20);
        entries.emplace_back(row, edge + 1, -0.15);
    }
    Eigen::SparseMatrix<double> design(samples, coefficients);
    design.setFromTriplets(entries.begin(), entries.end());
    design.makeCompressed();

    Eigen::VectorXd weights(samples);
    for (int row = 0; row < samples; ++row)
        weights[row] = 0.75 + 0.001 * static_cast<double>(row % 101);

    ReducedTraceProjection3D projector(design, weights);
    require(projector.uses_sparse_cholesky(),
            "large sparse design did not select sparse Cholesky");
    require(projector.rank() == coefficients,
            "large sparse projector rank diagnostic is wrong");
    require(projector.gram_nonzeros() < 4LL * coefficients,
            "large sparse Gram unexpectedly lost its banded sparsity");
    require(projector.factor_nonzeros() < 8LL * coefficients,
            "large sparse factor unexpectedly has excessive fill");
    require(std::isfinite(projector.condition())
                && projector.condition() < 2.0,
            "large sparse projector condition estimate is implausible");
    require(projector.pb_identity_residual() < 2.0e-13,
            "large sparse projector P*B RMS diagnostic is too large");
    require(projector.pb_max_error() < 2.0e-13,
            "large sparse projector P*B max diagnostic is too large");

    Eigen::VectorXd native(coefficients);
    for (int column = 0; column < coefficients; ++column) {
        native[column] = std::sin(0.017 * static_cast<double>(column + 1))
            + 0.2 * std::cos(
                0.071 * static_cast<double>(column + 3));
    }
    const Eigen::VectorXd trace = design * native;
    const Eigen::VectorXd recovered = projector.apply(trace);
    require((recovered - native).lpNorm<Eigen::Infinity>() < 2.0e-12,
            "large sparse weighted projection did not recover coefficients");

    const Eigen::VectorXd mass = projector.native_to_trace_mass(native);
    const Eigen::VectorXd native_roundtrip =
        projector.trace_mass_to_native(mass);
    require((native_roundtrip - native).lpNorm<Eigen::Infinity>() < 2.0e-13,
            "AMD-permuted sparse mass-coordinate roundtrip failed");

    Eigen::SparseMatrix<double> weighted_transpose = design.transpose();
    for (int sample = 0; sample < weighted_transpose.outerSize(); ++sample) {
        for (Eigen::SparseMatrix<double>::InnerIterator entry(
                 weighted_transpose, sample);
             entry; ++entry) {
            entry.valueRef() *= weights[sample];
        }
    }
    const Eigen::SparseMatrix<double> gram = weighted_transpose * design;
    require_near(
        mass.squaredNorm(),
        native.dot(gram * native),
        2.0e-11 * std::max(1.0, mass.squaredNorm()),
        "sparse mass coordinates do not represent the weighted trace norm");

    Eigen::VectorXd native_dual(coefficients);
    for (int row = 0; row < coefficients; ++row)
        native_dual[row] = 0.03 * std::sin(0.11 * (row + 1));
    const Eigen::VectorXd mass_dual =
        projector.native_dual_to_trace_mass(native_dual);
    require_near(
        native_dual.dot(native),
        mass_dual.dot(mass),
        2.0e-12,
        "sparse mass-coordinate dual transformation is inconsistent");

    // The topology production path supplies B0 and sparse E separately.  An
    // identity E is sufficient here to certify backend selection without
    // making either Bz or the reduced Gram dense.
    Eigen::SparseMatrix<double> identity(coefficients, coefficients);
    identity.setIdentity();
    ReducedTraceProjection3D composed(
        design, identity, weights);
    require(composed.uses_sparse_cholesky(),
            "large sparse composed design did not use sparse Cholesky");
    require((composed.apply(trace) - native).lpNorm<Eigen::Infinity>()
                < 2.0e-12,
            "large sparse composed projector did not recover coefficients");

    SparseMatrixCSR3D homogeneous(coefficients, coefficients);
    homogeneous.setIdentity();
    ConstraintSystem3D empty_constraints;
    empty_constraints.C.resize(0, coefficients);
    empty_constraints.d.resize(0);
    AffineReduction3D identity_reduction(
        Eigen::VectorXd::Zero(coefficients),
        homogeneous,
        empty_constraints,
        {},
        0.0,
        0.0,
        0.0);
    SparseMatrixCSR3D test_design = design;
    SparseMatrixCSR3D base_expansion(coefficients, coefficients);
    base_expansion.setIdentity();
    TopologyTraceProjector3D topology_projector(
        test_design,
        base_expansion,
        identity_reduction,
        weights);
    require(topology_projector.uses_sparse_cholesky(),
            "topology projector did not expose its sparse backend");
    require(topology_projector.gram_nonzeros()
                == projector.gram_nonzeros(),
            "topology projector sparse Gram diagnostics disagree");
    require(topology_projector.factor_nonzeros()
                == projector.factor_nonzeros(),
            "topology projector sparse factor diagnostics disagree");
    require((topology_projector.project(trace) - mass)
                .lpNorm<Eigen::Infinity>() < 2.0e-12,
            "topology sparse projector returned wrong mass coordinates");
}

} // namespace

int main()
{
    try {
        test_sparse_projection_and_trace_mass_roundtrip();
        test_particular_is_separate_and_mean_dual_is_consistent();
        test_final_mean_free_space_rebuilds_the_projector();
        test_rank_deficiency_is_rejected();
        test_final_trace_space_requires_strict_oversampling();
        test_large_sparse_spd_backend_without_dense_gram();
        std::cout << "topology_trace_projector_3d_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "topology_trace_projector_3d_test: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
