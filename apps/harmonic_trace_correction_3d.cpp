#include "harmonic_trace_correction_3d.hpp"

#include <cmath>
#include <stdexcept>

namespace kfbim::app3d {

namespace {

void validate_evaluation(const Eigen::VectorXd& evaluation,
                         int dimension,
                         const char* context)
{
    if (evaluation.size() != dimension || !evaluation.allFinite())
        throw std::invalid_argument(context);
}

} // namespace

double apply_harmonic_trace_correction_3d(
    int center_dof,
    const Eigen::MatrixXd& coefficients,
    const Eigen::VectorXd& legacy_evaluation,
    const std::vector<HarmonicTraceCorrectionTermInput3D>& owner_terms,
    TraceCorrectionOwnerMode3D mode)
{
    if (coefficients.rows() <= 0 || coefficients.cols() <= 0) {
        throw std::invalid_argument(
            "harmonic trace correction coefficients are invalid");
    }
    if (center_dof < 0 || center_dof >= coefficients.rows())
        throw std::out_of_range("harmonic trace center DOF is invalid");

    if (mode == TraceCorrectionOwnerMode3D::CenterDof) {
        validate_evaluation(
            legacy_evaluation, coefficients.cols(),
            "legacy harmonic trace evaluation is invalid");
        const double result = legacy_evaluation.dot(
            coefficients.row(center_dof).transpose());
        if (!std::isfinite(result))
            throw std::runtime_error("legacy harmonic trace correction is invalid");
        return result;
    }

    if (mode != TraceCorrectionOwnerMode3D::CrossingOwner)
        throw std::invalid_argument("unknown harmonic trace correction mode");

    double result = 0.0;
    for (const HarmonicTraceCorrectionTermInput3D& term : owner_terms) {
        if (term.owner_dof < 0 || term.owner_dof >= coefficients.rows())
            throw std::out_of_range("harmonic trace owner DOF is invalid");
        validate_evaluation(
            term.evaluation, coefficients.cols(),
            "harmonic trace owner evaluation is invalid");
        result += term.evaluation.dot(
            coefficients.row(term.owner_dof).transpose());
    }
    if (!std::isfinite(result))
        throw std::runtime_error("crossing-owner harmonic trace correction is invalid");
    return result;
}

} // namespace kfbim::app3d
