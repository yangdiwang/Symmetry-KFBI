#pragma once

#include "src/support/correction/shared_field_space_3d.hpp"
#include "src/support/trace/kfbi_correction_backend_3d.hpp"
#include "src/grid/cartesian_grid_3d.hpp"
#include <map>

namespace kfbim::app3d {
enum class SharedFieldRestrictMode3D { Staged, Direct };
struct SharedFieldTransferResult3D {
    Eigen::VectorXd rhs_for_delta;
    TracePair3D raw_correction;
};

// Owns fixed sparse maps only. Geometry, labels and the spline space are
// consumed during construction; no geometry/owner callback survives setup.
class SharedFieldTransfer3D {
public:
    using Sparse=Eigen::SparseMatrix<double,Eigen::RowMajor>;
    SharedFieldTransfer3D(const CartesianGrid3D& grid,const Eigen::VectorXi& labels,
        const std::vector<Eigen::Vector3d>& trace_points,
        const std::vector<Eigen::Vector3d>& trace_normals,
        const SharedFieldSpace3D& space,bool neumann,
        SharedFieldRestrictMode3D mode=SharedFieldRestrictMode3D::Staged);
    SharedFieldTransferResult3D evaluate(const Eigen::VectorXd& field_coefficients) const;
    TracePair3D observe_grid(const Eigen::VectorXd& full_solution) const;
    const std::vector<int>& grid_query_ids() const {return grid_ids_;}
    const Sparse& grid_evaluation() const {return Eg_;}
    const Sparse& grid_value_observer() const {return Rv_;}
    const Sparse& grid_normal_observer() const {return Rn_;}
    const std::map<std::string,double>& diagnostics() const {return diagnostics_;}
private:
    int full_size_=0,coefficient_count_=0;
    std::vector<int> grid_ids_;
    Sparse Eg_,Sg_,Jv_,Jn_,Eminus_,Nv_,Nn_,Rv_,Rn_;
    std::map<std::string,double> diagnostics_;
};
} // namespace kfbim::app3d
