#include "gmres.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfbim {

GMRES::GMRES(int max_iter, double tol, int restart)
    : max_iter_(max_iter), tol_(tol), restart_(restart)
{
    if (max_iter_ <= 0)
        throw std::invalid_argument("GMRES: max_iter must be positive");
    if (tol_ <= 0.0)
        throw std::invalid_argument("GMRES: tol must be positive");
}

int GMRES::solve(const IKFBIOperator& op,
                 const Eigen::VectorXd& rhs,
                 Eigen::VectorXd& x)
{
    converged_ = false;
    residuals_.clear();

    const int n = op.problem_size();
    if (x.size() != n)
        x = Eigen::VectorXd::Zero(n);

    const int m = (restart_ > 0) ? std::min(restart_, max_iter_) : max_iter_;

    Eigen::VectorXd r(n), w(n);

    // Initial residual
    op.apply(x, w);
    r = rhs - w;
    const double beta0 = r.norm();
    if (beta0 < tol_) {
        residuals_.push_back(0.0);
        converged_ = true;
        return 0;
    }
    residuals_.push_back(1.0);

    int total_iters = 0;

    while (total_iters < max_iter_) {
        // Current residual for this cycle
        op.apply(x, w);
        r = rhs - w;
        double beta = r.norm();

        if (beta / beta0 < tol_) {
            converged_ = true;
            break;
        }

        const int cycle_m = std::min(m, max_iter_ - total_iters);

        // Krylov basis: columns 0..cycle_m stored as (n x cycle_m+1) matrix
        Eigen::MatrixXd Q(n, cycle_m + 1);
        // Upper Hessenberg matrix: (cycle_m+1) x cycle_m
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(cycle_m + 1, cycle_m);
        // RHS vector for the least-squares problem
        Eigen::VectorXd g = Eigen::VectorXd::Zero(cycle_m + 1);
        // Givens rotation parameters
        Eigen::VectorXd cs(cycle_m), sn(cycle_m);

        Q.col(0) = r / beta;
        g[0] = beta;

        int j = 0;
        for (; j < cycle_m; ++j) {
            // Arnoldi step
            op.apply(Q.col(j), w);

            for (int i = 0; i <= j; ++i) {
                H(i, j) = Q.col(i).dot(w);
                w -= H(i, j) * Q.col(i);
            }
            // A second modified Gram--Schmidt pass is inexpensive compared
            // with applying the KFBI operator and prevents loss of
            // orthogonality for strongly non-normal interface matrices.
            for (int i = 0; i <= j; ++i) {
                const double correction = Q.col(i).dot(w);
                H(i, j) += correction;
                w -= correction * Q.col(i);
            }
            H(j + 1, j) = w.norm();
            const bool happy_breakdown = H(j + 1, j) < 1.0e-14;
            if (!happy_breakdown)
                Q.col(j + 1) = w / H(j + 1, j);

            // Apply all previous Givens rotations to column j
            for (int i = 0; i < j; ++i) {
                double tmp    =  cs[i] * H(i,     j) + sn[i] * H(i + 1, j);
                H(i + 1, j)  = -sn[i] * H(i,     j) + cs[i] * H(i + 1, j);
                H(i,     j)  = tmp;
            }

            // Compute and apply new Givens rotation
            double rho = std::hypot(H(j, j), H(j + 1, j));
            if (rho <= std::numeric_limits<double>::epsilon()) {
                throw std::runtime_error(
                    "GMRES: singular Hessenberg column during Arnoldi iteration");
            }
            cs[j] =  H(j,     j) / rho;
            sn[j] =  H(j + 1, j) / rho;
            H(j,     j) = rho;
            H(j + 1, j) = 0.0;

            g(j + 1) = -sn[j] * g(j);
            g(j)     =  cs[j] * g(j);

            ++total_iters;

            const double rel_res = std::abs(g(j + 1)) / beta0;
            residuals_.push_back(rel_res);

            if (happy_breakdown || rel_res < tol_) {
                ++j;
                break;
            }
        }

        // Solve upper triangular system H[0:j, 0:j] * y = g[0:j]
        Eigen::VectorXd y =
            H.topLeftCorner(j, j)
             .triangularView<Eigen::Upper>()
             .solve(g.head(j));

        // Update solution
        x += Q.leftCols(j) * y;

        // Compute true residual and record
        op.apply(x, w);
        r = rhs - w;
        residuals_.back() = r.norm() / beta0;

        if (residuals_.back() < tol_) {
            converged_ = true;
            break;
        }

        if (total_iters >= max_iter_)
            break;
    }

    return total_iters;
}

} // namespace kfbim
