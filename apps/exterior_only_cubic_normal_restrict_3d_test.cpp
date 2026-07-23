#include "exterior_only_cubic_normal_restrict_3d.hpp"

#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using kfbim::app3d::ExteriorOnlyCubicNormalStencil3D;
using kfbim::app3d::build_exterior_only_cubic_normal_stencil_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class Function>
void require_throws(Function&& function, const std::string& message)
{
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

struct CandidateFixture {
    Eigen::Vector3d center;
    Eigen::Vector3d tangent1;
    Eigen::Vector3d tangent2;
    Eigen::Vector3d normal;
    double h = 0.125;
    std::vector<std::pair<int, Eigen::Vector3d>> candidates;
    std::vector<Eigen::Vector3d> local_by_id;
};

CandidateFixture make_candidates(const Eigen::Matrix3d& rotation,
                                 const Eigen::Vector3d& center)
{
    CandidateFixture fixture;
    fixture.center = center;
    fixture.tangent1 = rotation.col(0);
    fixture.tangent2 = rotation.col(1);
    fixture.normal = rotation.col(2);
    int id = 0;
    for (int iz = -3; iz <= -1; ++iz) {
        for (int iy = -3; iy <= 3; ++iy) {
            for (int ix = -3; ix <= 3; ++ix) {
                const Eigen::Vector3d local(
                    0.41 * static_cast<double>(ix),
                    0.37 * static_cast<double>(iy),
                    0.46 * static_cast<double>(iz));
                const Eigen::Vector3d point =
                    center + fixture.h * rotation * local;
                fixture.candidates.emplace_back(id, point);
                fixture.local_by_id.push_back(local);
                ++id;
            }
        }
    }
    return fixture;
}

double harmonic_cubic(const Eigen::Vector3d& x)
{
    return 1.2 + 0.7 * x.x() - 0.4 * x.y() + 2.3 * x.z()
         + 0.5 * x.x() * x.y()
         + 0.8 * (x.x() * x.x() - x.y() * x.y())
         - 0.35 * (x.x() * x.x() * x.x()
                   - 3.0 * x.x() * x.y() * x.y())
         + 0.27 * x.z() * (x.x() * x.x() - x.y() * x.y());
}

double apply_to_fixture(const ExteriorOnlyCubicNormalStencil3D& stencil,
                        const CandidateFixture& fixture)
{
    double value = 0.0;
    for (int k = 0; k < static_cast<int>(stencil.grid_ids.size()); ++k) {
        const int id = stencil.grid_ids[static_cast<std::size_t>(k)];
        value += stencil.weights[k]
               * harmonic_cubic(
                   fixture.local_by_id[static_cast<std::size_t>(id)]);
    }
    return value;
}

void test_cubic_harmonic_reproduction()
{
    const Eigen::Matrix3d rotation =
        (Eigen::AngleAxisd(0.37, Eigen::Vector3d(1.0, 2.0, 3.0).normalized())
         * Eigen::AngleAxisd(-0.19, Eigen::Vector3d::UnitY()))
            .toRotationMatrix();
    const CandidateFixture fixture =
        make_candidates(rotation, Eigen::Vector3d(0.31, -0.27, 0.14));
    const ExteriorOnlyCubicNormalStencil3D stencil =
        build_exterior_only_cubic_normal_stencil_3d(
            fixture.center,
            fixture.tangent1,
            fixture.tangent2,
            fixture.normal,
            fixture.h,
            fixture.candidates,
            48);
    require(stencil.grid_ids.size() == 48 && stencil.weights.size() == 48,
            "reference restrict uses exactly 48 candidates");
    require(std::isfinite(stencil.condition) && stencil.condition > 1.0,
            "reference restrict reports a finite condition number");
    require(std::abs(apply_to_fixture(stencil, fixture) - 2.3 / fixture.h)
                < 2.0e-10,
            "reference restrict reproduces a cubic harmonic derivative");
}

void test_rigid_invariance_and_determinism()
{
    const CandidateFixture identity =
        make_candidates(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(
            0.51, Eigen::Vector3d(-2.0, 1.0, 4.0).normalized())
            .toRotationMatrix();
    const CandidateFixture moved =
        make_candidates(rotation, Eigen::Vector3d(-0.22, 0.17, 0.39));

    const auto build = [](const CandidateFixture& fixture) {
        return build_exterior_only_cubic_normal_stencil_3d(
            fixture.center,
            fixture.tangent1,
            fixture.tangent2,
            fixture.normal,
            fixture.h,
            fixture.candidates,
            48);
    };
    const ExteriorOnlyCubicNormalStencil3D first = build(identity);
    const ExteriorOnlyCubicNormalStencil3D repeated = build(identity);
    const ExteriorOnlyCubicNormalStencil3D transformed = build(moved);
    require(first.grid_ids == repeated.grid_ids
                && (first.weights - repeated.weights).norm() == 0.0,
            "reference restrict construction is deterministic");
    require(std::abs(apply_to_fixture(first, identity) - 2.3 / identity.h)
                < 2.0e-10,
            "identity reference derivative");
    require(std::abs(apply_to_fixture(transformed, moved) - 2.3 / moved.h)
                < 2.0e-10,
            "rigidly transformed reference derivative");
}

void test_invalid_stencils_are_rejected()
{
    const CandidateFixture fixture =
        make_candidates(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    auto build = [&](const auto& candidates,
                     const Eigen::Vector3d& tangent2,
                     double h,
                     int count) {
        return build_exterior_only_cubic_normal_stencil_3d(
            fixture.center,
            fixture.tangent1,
            tangent2,
            fixture.normal,
            h,
            candidates,
            count);
    };

    std::vector<std::pair<int, Eigen::Vector3d>> insufficient(
        fixture.candidates.begin(), fixture.candidates.begin() + 15);
    require_throws(
        [&] { (void)build(insufficient, fixture.tangent2, fixture.h, 15); },
        "fewer than 16 samples are rejected");
    require_throws(
        [&] { (void)build(fixture.candidates, fixture.normal, fixture.h, 48); },
        "non-orthonormal local frame is rejected");
    require_throws(
        [&] { (void)build(fixture.candidates, fixture.tangent2, 0.0, 48); },
        "non-positive spacing is rejected");

    std::vector<std::pair<int, Eigen::Vector3d>> coplanar;
    int id = 0;
    for (int iy = -4; iy <= 4; ++iy) {
        for (int ix = -4; ix <= 4; ++ix) {
            coplanar.emplace_back(
                id++,
                fixture.center + fixture.h
                    * Eigen::Vector3d(
                        0.3 * static_cast<double>(ix),
                        0.3 * static_cast<double>(iy),
                        -1.0));
        }
    }
    require_throws(
        [&] { (void)build(coplanar, fixture.tangent2, fixture.h, 48); },
        "rank-deficient coplanar samples are rejected");
}

} // namespace

int main()
{
    try {
        test_cubic_harmonic_reproduction();
        test_rigid_invariance_and_determinism();
        test_invalid_stencils_are_rejected();
        std::cout << "3D exterior-only cubic normal restrict tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "3D exterior-only cubic normal restrict test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
