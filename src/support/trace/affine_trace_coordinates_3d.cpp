#include "affine_trace_coordinates_3d.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
namespace kfbim::app3d {
namespace {
// FNV-1a over canonical little-endian integer / IEEE double words. IDs are
// reproducible fingerprints of ordering and numerical coordinate semantics.
struct Fingerprint {
    std::uint64_t value=14695981039346656037ULL;
    void word(std::uint64_t x){for(int i=0;i<8;++i){value^=(x&255);value*=1099511628211ULL;x>>=8;}}
    void scalar(double x){std::uint64_t bits=0;if(x==0)x=0;std::memcpy(&bits,&x,sizeof bits);word(bits);}
    template<class Derived>void dense(const Eigen::MatrixBase<Derived>& a){word(a.rows());word(a.cols());for(Eigen::Index i=0;i<a.rows();++i)for(Eigen::Index j=0;j<a.cols();++j)scalar(a(i,j));}
    void sparse(const Trace93DensityLayout3D::Sparse& a){word(a.rows());word(a.cols());
        for(int i=0;i<a.outerSize();++i)for(Trace93DensityLayout3D::Sparse::InnerIterator it(a,i);it;++it)if(it.value()!=0){word(it.row());word(it.col());scalar(it.value());}}
    std::string str(const char* prefix)const{std::ostringstream s;s<<prefix<<std::hex<<std::setw(16)<<std::setfill('0')<<value;return s.str();}
};
void traces(Fingerprint& f,const Trace93DensityLayout3D& l){f.word(l.neumann);f.scalar(l.h);f.word(l.traces.size());
    for(const auto& q:l.traces){f.word(q.patch);f.scalar(q.u);f.scalar(q.v);f.dense(q.point);f.dense(q.normal);}
    f.dense(l.weights);f.sparse(l.trace_basis);
}
}
std::string trace93_density_layout_id_3d(const Trace93DensityLayout3D& l){
    Fingerprint f;traces(f,l);f.word(l.reference_raw_dofs);for(const auto& p:l.patch_spans){f.word(p[0]);f.word(p[1]);}
    for(int p:l.patch_offsets)f.word(p);f.dense(l.Z);f.dense(l.particular);
    return f.str("trace93-analysis-density-v1-");
}
std::string trace93_trace_layout_id_3d(const Trace93DensityLayout3D& l){
    Fingerprint f;traces(f,l);f.sparse(l.edge_basis);f.sparse(l.edge_trace);f.dense(l.edge_weights);
    f.sparse(l.vertex_basis);f.sparse(l.vertex_trace);f.dense(l.vertex_weights);return f.str("trace93-analysis-trace-v1-");
}
Trace93AffineTraceCoordinates3D::Trace93AffineTraceCoordinates3D(const Trace93DensityLayout3D& l,std::string density_id,std::string trace_id){
    const auto start=std::chrono::steady_clock::now();
    const int n=l.reference_raw_dofs,m=static_cast<int>(l.trace_basis.rows());
    if(n<1||m<1||l.Z.rows()!=n||l.Z.cols()<1||l.particular.size()!=n||l.weights.size()!=m||l.trace_basis.cols()!=n||
       !l.Z.allFinite()||!l.particular.allFinite()||!l.weights.allFinite()||(l.weights.array()<0).any())
        throw std::invalid_argument("Trace93 affine coordinate dimensions/values");
    const auto check_extra=[&](const auto& basis,const auto& trace,const Eigen::VectorXd& weights){
        if(weights.size()&&(basis.rows()!=weights.size()||basis.cols()!=n||trace.rows()!=weights.size()||trace.cols()!=m||!weights.allFinite()||(weights.array()<0).any()))
            throw std::invalid_argument("Trace93 edge/vertex dimensions/weights");};
    check_extra(l.edge_basis,l.edge_trace,l.edge_weights);check_extra(l.vertex_basis,l.vertex_trace,l.vertex_weights);
    B_=l.trace_basis;Z_=l.Z;particular_=l.particular;weights_=l.weights;
    edge_weights_=l.edge_weights;vertex_weights_=l.vertex_weights;edge_trace_=l.edge_trace;vertex_trace_=l.vertex_trace;
    Ber_=edge_weights_.size()?Eigen::MatrixXd(l.edge_basis*Z_):Eigen::MatrixXd(0,Z_.cols());
    Bvr_=vertex_weights_.size()?Eigen::MatrixXd(l.vertex_basis*Z_):Eigen::MatrixXd(0,Z_.cols());
    auto weighted_B=B_;
    for(int row=0;row<weighted_B.outerSize();++row)
        for(Trace93DensityLayout3D::Sparse::InnerIterator it(weighted_B,row);it;++it)it.valueRef()*=weights_[it.row()];
    Eigen::SparseMatrix<double> mass=B_.transpose()*weighted_B;
    Eigen::MatrixXd M=Z_.transpose()*(mass*Z_);
    if(edge_weights_.size())M+=Ber_.transpose()*edge_weights_.asDiagonal()*Ber_;
    if(vertex_weights_.size())M+=Bvr_.transpose()*vertex_weights_.asDiagonal()*Bvr_;
    M=(0.5*(M+M.transpose())).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(M,Eigen::EigenvaluesOnly);
    if(eig.info()!=Eigen::Success)throw std::runtime_error("projector eigenvalue diagnostic failed");
    if(eig.eigenvalues()[0]<=0){regularization_=std::max(eig.eigenvalues().maxCoeff(),1.0)*1e-13-eig.eigenvalues()[0];M.diagonal().array()+=regularization_;}
    chol_.compute(M);
    if(chol_.info()!=Eigen::Success)throw std::runtime_error("trace Gram matrix not positive definite: refine trace sampling");
    const Eigen::VectorXd pivots=chol_.matrixL().toDenseMatrix().diagonal();pivot_ratio_=pivots.minCoeff()/pivots.maxCoeff();
    if(!std::isfinite(pivot_ratio_)||pivot_ratio_<1e-10)throw std::runtime_error("trace space is insufficiently observable");
    density_id_=density_id.empty()?trace93_density_layout_id_3d(l):std::move(density_id);
    trace_id_=trace_id.empty()?trace93_trace_layout_id_3d(l):std::move(trace_id);
    setup_seconds_=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
Eigen::VectorXd Trace93AffineTraceCoordinates3D::lift_homogeneous(const Eigen::VectorXd& y)const{
    if(y.size()!=Z_.cols()||!y.allFinite())throw std::invalid_argument("reduced coordinates shape/value");return Z_*y;
}
Eigen::VectorXd Trace93AffineTraceCoordinates3D::project(const Eigen::VectorXd& t)const{
    if(t.size()!=B_.rows()||!t.allFinite())throw std::invalid_argument("projection trace shape/value");
    Eigen::VectorXd r=Z_.transpose()*(B_.transpose()*(weights_.array()*t.array()).matrix());
    if(edge_weights_.size())r+=Ber_.transpose()*(edge_weights_.array()*(edge_trace_*t).array()).matrix();
    if(vertex_weights_.size())r+=Bvr_.transpose()*(vertex_weights_.array()*(vertex_trace_*t).array()).matrix();
    return chol_.solve(r);
}
}
