#include "harmonic_trace_correction_3d.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using kfbim::app3d::HarmonicTraceCorrectionMode3D;
using kfbim::app3d::HarmonicTraceOwnerTerm3D;
using kfbim::app3d::apply_harmonic_trace_correction_3d;

void require(bool value, const std::string& message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <class Function>
void require_invalid(Function&& function, const std::string& message)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

Eigen::MatrixXd literal_coefficients()
{
    Eigen::MatrixXd coefficients(3, 2);
    coefficients << 2.0, 3.0,
                    4.0, 1.0,
                    2.0, 5.0;
    return coefficients;
}

void test_selects_center_or_precomputed_crossing_owner_rows()
{
    const Eigen::MatrixXd coefficients = literal_coefficients();
    const Eigen::VectorXd center_evaluation =
        (Eigen::Vector2d() << 1.0, 3.0).finished();
    const std::vector<HarmonicTraceOwnerTerm3D> owner_terms{
        {1, (Eigen::Vector2d() << 3.0, 1.0).finished()},
        {2, (Eigen::Vector2d() << 0.0, 2.0).finished()}};

    const double center_owned = apply_harmonic_trace_correction_3d(
        0, coefficients, center_evaluation, owner_terms,
        HarmonicTraceCorrectionMode3D::CenterOwned);
    const double crossing_owned = apply_harmonic_trace_correction_3d(
        0, coefficients, center_evaluation, owner_terms,
        HarmonicTraceCorrectionMode3D::CrossingOwned);

    require(std::abs(center_owned - 11.0) < 1.0e-14,
            "center-owned correction uses center evaluation");
    require(std::abs(crossing_owned - 23.0) < 1.0e-14,
            "crossing-owned correction uses precomputed owner rows");
}

void test_rejects_invalid_owner_indices()
{
    const Eigen::MatrixXd coefficients = literal_coefficients();
    const Eigen::VectorXd evaluation =
        (Eigen::Vector2d() << 1.0, 3.0).finished();

    require_invalid(
        [&] {
            (void)apply_harmonic_trace_correction_3d(
                -1, coefficients, evaluation, {},
                HarmonicTraceCorrectionMode3D::CenterOwned);
        },
        "negative center index rejected");
    require_invalid(
        [&] {
            (void)apply_harmonic_trace_correction_3d(
                3, coefficients, evaluation, {},
                HarmonicTraceCorrectionMode3D::CenterOwned);
        },
        "out-of-range center index rejected");
    require_invalid(
        [&] {
            (void)apply_harmonic_trace_correction_3d(
                0, coefficients, evaluation,
                {{-1, evaluation}},
                HarmonicTraceCorrectionMode3D::CrossingOwned);
        },
        "negative owner index rejected");
    require_invalid(
        [&] {
            (void)apply_harmonic_trace_correction_3d(
                0, coefficients, evaluation,
                {{3, evaluation}},
                HarmonicTraceCorrectionMode3D::CrossingOwned);
        },
        "out-of-range owner index rejected");
}

void test_rejects_incompatible_evaluation_dimensions()
{
    const Eigen::MatrixXd coefficients = literal_coefficients();
    const Eigen::VectorXd good_evaluation =
        (Eigen::Vector2d() << 1.0, 3.0).finished();
    const Eigen::VectorXd bad_evaluation =
        (Eigen::Vector3d() << 1.0, 2.0, 3.0).finished();

    require_invalid(
        [&] {
            (void)apply_harmonic_trace_correction_3d(
                0, coefficients, bad_evaluation, {},
                HarmonicTraceCorrectionMode3D::CenterOwned);
        },
        "incompatible center evaluation rejected");
    require_invalid(
        [&] {
            (void)apply_harmonic_trace_correction_3d(
                0, coefficients, good_evaluation,
                {{1, bad_evaluation}},
                HarmonicTraceCorrectionMode3D::CrossingOwned);
        },
        "incompatible owner evaluation rejected");
}
} // namespace

int main()
{
    try {
        test_selects_center_or_precomputed_crossing_owner_rows();
        test_rejects_invalid_owner_indices();
        test_rejects_incompatible_evaluation_dimensions();
        std::cout << "harmonic trace correction 3D tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "harmonic trace correction 3D test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
