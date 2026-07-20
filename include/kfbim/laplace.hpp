#pragma once

#include "src/local_cauchy/edge_patch_laplacian_3d.hpp"

#include "src/bulk_solvers/laplace_zfft_bulk_solver_2d.hpp"
#include "src/bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include "src/local_cauchy/edge_singular_bspline_fit_3d.hpp"
#include "src/local_cauchy/edge_singular_fit_3d.hpp"
#include "src/operators/laplace_bvp_2d.hpp"
#include "src/operators/laplace_bvp_3d.hpp"
#include "src/operators/laplace_neumann_exterior_trace_2d.hpp"
#include "src/operators/laplace_transmission_2d.hpp"
#include "src/operators/laplace_transmission_3d.hpp"
#include "src/potentials/laplace_potential.hpp"
#include "src/transfer/laplace_restrict_2d.hpp"
#include "src/transfer/laplace_restrict_3d.hpp"
#include "src/transfer/laplace_spread_2d.hpp"
#include "src/transfer/laplace_spread_3d.hpp"
