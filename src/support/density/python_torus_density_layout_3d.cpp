#include "src/support/density/python_torus_density_layout_3d.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kfbim::app3d {
namespace {
std::pair<Eigen::VectorXd, Eigen::VectorXd> gauss(int n)
{
    if (n < 1 || n > 32) throw std::invalid_argument("invalid Gauss order");
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
    for (int i = 1; i < n; ++i)
        J(i-1,i) = J(i,i-1) = i / std::sqrt(4.0*i*i-1.0);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(J);
    if (eig.info() != Eigen::Success) throw std::runtime_error("Gauss eigensolve failed");
    return {eig.eigenvalues(), 2.0*eig.eigenvectors().row(0).array().square().matrix().transpose()};
}

Eigen::MatrixXd periodic_space(int spans)
{
    const int nc = spans + 3;
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(8, 4*nc);
    for (int q=0; q<4; ++q) {
        const int a=q*nc+nc-1, b=((q+1)%4)*nc;
        C(2*q,a)=1; C(2*q,b)=-1;
        C(2*q+1,a)=1; C(2*q+1,a-1)=-1;
        C(2*q+1,b+1)=-1; C(2*q+1,b)=1;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeFullV);
    svd.setThreshold(1e-12);
    if (svd.rank()!=8) throw std::runtime_error("periodic cubic constraints lost rank");
    return svd.matrixV().rightCols(4*nc-8);
}
} // namespace

Eigen::MatrixXd cubic_midpoint_insertion_matrix_3d(int spans)
{
    if (spans < 1) throw std::invalid_argument("positive span count required");
    std::vector<double> knots(4,0.0);
    for (int i=1;i<spans;++i) knots.push_back(double(i)/spans);
    knots.insert(knots.end(),4,1.0);
    Eigen::MatrixXd P=Eigen::MatrixXd::Identity(spans+3,spans+3);
    for (int s=0;s<spans;++s) {
        const double x=(s+.5)/spans;
        const int k=int(std::upper_bound(knots.begin(),knots.end(),x)-knots.begin())-1;
        Eigen::MatrixXd Q=Eigen::MatrixXd::Zero(P.rows()+1,P.cols());
        for (int i=0;i<=k-3;++i) Q.row(i)=P.row(i);
        for (int i=k;i<P.rows();++i) Q.row(i+1)=P.row(i);
        for (int i=k-2;i<=k;++i) {
            const double alpha=(x-knots[i])/(knots[i+3]-knots[i]);
            Q.row(i)=alpha*P.row(i)+(1-alpha)*P.row(i-1);
        }
        knots.insert(knots.begin()+k+1,x);
        P=std::move(Q);
    }
    return P;
}

PythonTorusDensityLayout3D build_python_torus_density_layout_3d(
    const NativeNurbsDensitySpace3D& density, int nu, int nv,
    bool mean_free, int trace_order, int mean_order)
{
    if (nu != 2*nv || nv<1 || density.patch_count()!=16
        || density.coefficients_per_direction()!=nu+3
        || !density.uses_identity_reduction())
        throw std::invalid_argument("reference torus needs a 16-patch C0 carrier and nu=2*nv");
    PythonTorusDensityLayout3D out;
    out.spans_u=nu; out.spans_v=nv;
    out.reference_raw_dofs=16*(nu+3)*(nv+3);
    const int nf=nu+3, nc=nv+3;
    const auto Zu=periodic_space(nu);
    const auto Zv=periodic_space(nv);
    const auto P=cubic_midpoint_insertion_matrix_3d(nv);
    Eigen::MatrixXd Zv_fine(4*nf,Zv.cols());
    for(int q=0;q<4;++q) Zv_fine.middleRows(q*nf,nf)=P*Zv.middleRows(q*nc,nc);
    out.Z=Eigen::MatrixXd::Zero(density.c0_coefficient_count(),Zu.cols()*Zv.cols());
    std::vector<bool> assigned(density.c0_coefficient_count(),false);
    const auto& map=density.local_to_c0();
    for(int a=0;a<4;++a) for(int b=0;b<4;++b)
        for(int i=0;i<nf;++i) for(int j=0;j<nf;++j) {
            const int raw=(a*4+b)*nf*nf+i*nf+j;
            const int row=map.at(raw);
            Eigen::RowVectorXd value(out.Z.cols());
            for(int v=0;v<Zv.cols();++v)
                value.segment(v*Zu.cols(),Zu.cols())=Zv_fine(b*nf+j,v)*Zu.row(a*nf+i);
            if(assigned[row]) out.c0_embedding_error=std::max(out.c0_embedding_error,
                (out.Z.row(row)-value).cwiseAbs().maxCoeff());
            else {out.Z.row(row)=value;assigned[row]=true;}
        }
    if(out.c0_embedding_error>2e-12)
        throw std::runtime_error("reference atlas and native C0 ordering do not agree");
    out.mean_row=Eigen::RowVectorXd::Zero(density.c0_coefficient_count());
    const auto integrate=[&](int order, bool keep_trace) {
        const auto rule=gauss(order);
        for(int p=0;p<16;++p) for(int a=0;a<nu;++a) for(int b=0;b<nv;++b)
            for(int i=0;i<order;++i) for(int j=0;j<order;++j) {
                const double u=(a+(rule.first[i]+1)*.5)/nu;
                const double v=(b+(rule.first[j]+1)*.5)/nv;
                const auto g=density.geometry(p,u,v);
                const double w=rule.second[i]*rule.second[j]*g.area_element/(4.0*nu*nv);
                if(keep_trace) {
                    NativeDensityGaussPoint3D q;
                    q.patch=p;q.element_u=a;q.element_v=b;q.u=u;q.v=v;
                    q.point=g.point;q.normal=g.normal;q.surface_weight=w;
                    out.traces.push_back(q);
                } else {
                    const auto r=density.c0_basis_stencil(p,u,v);
                    for(int k=0;k<r.count;++k) out.mean_row[r.indices[k]]+=w*r.weights[k];
                }
            }
    };
    integrate(trace_order,true); integrate(mean_order,false);
    out.weights.resize(out.traces.size());
    for(int i=0;i<out.weights.size();++i) out.weights[i]=out.traces[i].surface_weight;
    if(mean_free) {
        Eigen::VectorXd u=(out.mean_row*out.Z).transpose();
        u.normalize();u[0]+=std::copysign(1.0,u[0]);u.normalize();
        const Eigen::VectorXd z_u=out.Z*u;
        out.Z=(out.Z.rightCols(out.Z.cols()-1)-2*z_u*u.tail(u.size()-1).transpose()).eval();
        out.mean_nullspace_error=(out.mean_row*out.Z).cwiseAbs().maxCoeff();
        if(out.mean_nullspace_error>2e-12) throw std::runtime_error("mean-free reduction failed");
    }
    out.particular=Eigen::VectorXd::Zero(out.Z.rows());
    if(out.traces.size()<=std::size_t(out.Z.cols()))
        throw std::runtime_error("external trace count must exceed reduced DOFs");
    return out;
}
} // namespace kfbim::app3d
