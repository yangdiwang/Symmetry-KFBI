#include "src/support/trace/shared_field_transfer_3d.hpp"
#include "src/support/trace/restrict_resource_plan_3d.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace kfbim::app3d {
namespace {
using Triplet=Eigen::Triplet<double>;
using Sparse=SharedFieldTransfer3D::Sparse;
void assemble(Sparse& matrix,int rows,int cols,const std::vector<Triplet>& entries) {
    matrix.resize(rows,cols);matrix.setFromTriplets(entries.begin(),entries.end());matrix.makeCompressed();
}
}

SharedFieldTransfer3D::SharedFieldTransfer3D(const CartesianGrid3D& grid,
    const Eigen::VectorXi& labels,const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,const SharedFieldSpace3D& space,
    bool neumann,SharedFieldRestrictMode3D mode)
    :full_size_(grid.num_dofs()),coefficient_count_(space.coefficient_count())
{
    const auto start=std::chrono::steady_clock::now();
    const double h=grid.spacing()[0];
    if(grid.layout()!=DofLayout3D::Node||labels.size()!=full_size_||points.empty()||points.size()!=normals.size()
       ||std::abs(grid.spacing()[1]-h)>1e-14*h||std::abs(grid.spacing()[2]-h)>1e-14*h
       ||std::abs(space.options().h-h)>1e-14*h)
        throw std::invalid_argument("shared transfer requires a matching uniform node grid and trace arrays");
    if(mode!=SharedFieldRestrictMode3D::Staged&&mode!=SharedFieldRestrictMode3D::Direct)
        throw std::invalid_argument("invalid shared restrict mode");
    for(int i=0;i<labels.size();++i) if(labels[i]!=0&&labels[i]!=1)
        throw std::invalid_argument("shared transfer labels must be zero or one");
    const auto dims=grid.dof_dims();
    const auto interior=[&](int id) {
        const int i=id%dims[0],j=(id/dims[0])%dims[1],k=id/(dims[0]*dims[1]);
        return i>0&&i<dims[0]-1&&j>0&&j<dims[1]-1&&k>0&&k<dims[2]-1;
    };
    for(int id=0;id<full_size_;++id) if(!interior(id)&&labels[id])
        throw std::invalid_argument("shared correction interface must lie strictly inside the zero Dirichlet box");

    // During collection the column is a full grid ID. Remap the single union
    // once below, so every use of D at a node refers to the same evaluation.
    std::vector<Triplet> spread,jv,jn,rv,rn,nv,nn;
    std::vector<Eigen::Vector3d> negative_points;
    std::size_t crossings=0,wrong=0;double max_weight=0.;
    for(int k=1;k<dims[2]-1;++k) for(int j=1;j<dims[1]-1;++j) for(int i=1;i<dims[0]-1;++i) {
        const int a=grid.index(i,j,k);
        for(int axis=0;axis<3;++axis) {
            std::array<int,3> ijk{{i,j,k}};++ijk[axis];
            if(ijk[axis]>=dims[axis]-1) continue;
            const int b=grid.index(ijk[0],ijk[1],ijk[2]);
            if(labels[a]==labels[b]) continue;
            ++crossings;
            // Delta solve: (chi_b-chi_a) D_b/h^2 at a and vice versa.
            // Multiple geometric events with equal endpoint labels telescope
            // exactly for one shared D; they do not create separate owners.
            const double sign=(labels[b]-labels[a])/(h*h);
            spread.emplace_back(a,b,sign);spread.emplace_back(b,a,-sign);
            grid_ids_.push_back(a);grid_ids_.push_back(b);
        }
    }
    const auto cover=neumann?TensorProductCoverKind3D::Q27Cover3:TensorProductCoverKind3D::Q64Cover4;
    for(int q=0;q<static_cast<int>(points.size());++q) {
        if(!points[q].allFinite()||!normals[q].allFinite()||std::abs(normals[q].norm()-1.)>1e-10)
            throw std::invalid_argument("shared transfer requires finite points and unit outward normals");
        const auto stencil=build_shared_side_cover_restrict_stencil_3d(grid,points[q],normals[q],cover,true,1e-13);
        for(const auto& side:stencil.sides) {
            for(int s=0;s<3;++s) max_weight=std::max(max_weight,side.sampling_weights.row(s).norm());
            for(int a=0;a<static_cast<int>(side.grid_ids.size());++a) {
                const int id=side.grid_ids[a];
                const double v=side.value_weights[a],n=side.normal_weights[a];
                if(interior(id)) {rv.emplace_back(q,id,v);rn.emplace_back(q,id,n);}
                const int sign=(mode==SharedFieldRestrictMode3D::Staged&&side.desired_inside?1:0)-labels[id];
                if(sign) {++wrong;grid_ids_.push_back(id);jv.emplace_back(q,id,sign*v);jn.emplace_back(q,id,sign*n);}
            }
            if(mode==SharedFieldRestrictMode3D::Staged&&side.desired_inside) {
                for(int a=0;a<3;++a) {
                    const int id=static_cast<int>(negative_points.size());
                    negative_points.push_back(space.geometry().to_reference(side.sample_points[a]));
                    nv.emplace_back(q,id,-side.value_recovery[a]);
                    nn.emplace_back(q,id,-side.normal_recovery[a]);
                }
            }
        }
    }
    std::sort(grid_ids_.begin(),grid_ids_.end());grid_ids_.erase(std::unique(grid_ids_.begin(),grid_ids_.end()),grid_ids_.end());
    std::vector<Eigen::Vector3d> query_points;query_points.reserve(grid_ids_.size());
    for(int id:grid_ids_) {const auto x=grid.coord(id);query_points.push_back(space.geometry().to_reference({x[0],x[1],x[2]}));}
    // evaluation_matrix validates every cell/support. Coverage errors abort
    // setup; neither an owner fallback nor an implicit zero extension exists.
    Eg_=space.evaluation_matrix(query_points,SharedFieldDerivative3D::Value);
    Eminus_=space.evaluation_matrix(negative_points,SharedFieldDerivative3D::Value);
    const auto remap=[&](std::vector<Triplet>& entries) {
        for(auto& e:entries) e=Triplet(e.row(),static_cast<int>(std::lower_bound(grid_ids_.begin(),grid_ids_.end(),e.col())-grid_ids_.begin()),e.value());
    };
    remap(spread);remap(jv);remap(jn);
    const int nq=static_cast<int>(points.size()),ng=static_cast<int>(grid_ids_.size());
    assemble(Sg_,full_size_,ng,spread);assemble(Jv_,nq,ng,jv);assemble(Jn_,nq,ng,jn);
    assemble(Rv_,nq,full_size_,rv);assemble(Rn_,nq,full_size_,rn);
    assemble(Nv_,nq,static_cast<int>(negative_points.size()),nv);assemble(Nn_,nq,static_cast<int>(negative_points.size()),nn);
    diagnostics_={{"operator_setup_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},
        {"shared_grid_query_nodes",double(ng)},{"negative_sample_queries",double(negative_points.size())},
        {"spread_crossings",double(crossings)},{"trace_count",double(nq)},{"wrong_side_nodes",double(wrong)},
        {"q_stencil",neumann?27.:64.},{"max_weight_l2",max_weight},{"shared_coverage_failures",0.},
        {"first_hit_calls",0.},{"face_owner_calls",0.},{"single_face_fallbacks",0.}};
}

SharedFieldTransferResult3D SharedFieldTransfer3D::evaluate(const Eigen::VectorXd& coefficients) const {
    if(coefficients.size()!=coefficient_count_||!coefficients.allFinite())
        throw std::invalid_argument("shared transfer coefficient size/finite check failed");
    const Eigen::VectorXd d=Eg_*coefficients;
    SharedFieldTransferResult3D out;out.rhs_for_delta=Sg_*d;
    out.raw_correction.value=Jv_*d;out.raw_correction.normal=Jn_*d;
    if(Eminus_.rows()) {
        const Eigen::VectorXd negative=Eminus_*coefficients;
        out.raw_correction.value+=Nv_*negative;out.raw_correction.normal+=Nn_*negative;
    }
    if(!out.rhs_for_delta.allFinite()||!out.raw_correction.value.allFinite()||!out.raw_correction.normal.allFinite())
        throw std::runtime_error("shared transfer produced nonfinite output");
    return out;
}

TracePair3D SharedFieldTransfer3D::observe_grid(const Eigen::VectorXd& solution) const {
    if(solution.size()!=full_size_||!solution.allFinite()) throw std::invalid_argument("shared Rg grid size/finite check failed");
    return {Rv_*solution,Rn_*solution};
}
} // namespace kfbim::app3d
