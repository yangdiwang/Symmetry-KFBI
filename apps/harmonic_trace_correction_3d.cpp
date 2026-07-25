#include "harmonic_trace_correction_3d.hpp"

#include <stdexcept>

namespace kfbim::app3d {
namespace {

void validate_index(int dof, int row_count, const char* description)
{
    if (dof < 0 || dof >= row_count)
        throw std::invalid_argument(description);
}

void validate_evaluation(const Eigen::VectorXd& evaluation,
                         int coefficient_count,
                         const char* description)
{
    if (evaluation.size() != coefficient_count)
        throw std::invalid_argument(description);
}

} // namespace

double apply_harmonic_trace_correction_3d(
    int center_dof,
    const Eigen::MatrixXd& coefficients,
    const Eigen::VectorXd& center_evaluation,
    const std::vector<HarmonicTraceOwnerTerm3D>& owner_terms,
    HarmonicTraceCorrectionMode3D mode)
{
    validate_index(center_dof, coefficients.rows(),
                   "harmonic trace correction center index is invalid");
    validate_evaluation(
        center_evaluation, coefficients.cols(),
        "harmonic trace correction center evaluation has invalid dimension");
    for (const HarmonicTraceOwnerTerm3D& term : owner_terms) {
        validate_index(term.owner_dof, coefficients.rows(),
                       "harmonic trace correction owner index is invalid");
        validate_evaluation(
            term.evaluation, coefficients.cols(),
            "harmonic trace correction owner evaluation has invalid dimension");
    }

    if (mode == HarmonicTraceCorrectionMode3D::CenterOwned) {
        return center_evaluation.dot(
            coefficients.row(center_dof).transpose());
    }
    double result = 0.0;
    for (const HarmonicTraceOwnerTerm3D& term : owner_terms) {
        result += term.evaluation.dot(
            coefficients.row(term.owner_dof).transpose());
    }
    return result;
}

} // namespace kfbim::app3d
