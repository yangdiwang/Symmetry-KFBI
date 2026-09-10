#include "src/support/density/trace93_density_layout_3d.hpp"
#include "src/support/geometry/trace93_case_3d.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace kfbim::app3d {
namespace {
using Row = Eigen::RowVectorXd;
using Sparse = Trace93DensityLayout3D::Sparse;
using Triplet = Eigen::Triplet<double>;
using Edge = PatchEdge3D;
constexpr double eps = std::numeric_limits<double>::epsilon();
const std::array<std::array<int,2>,10> deriv{{{0,0},{1,0},{0,1},{2,0},{1,1},
    {0,2},{3,0},{2,1},{1,2},{0,3}}};

std::pair<Eigen::VectorXd,Eigen::VectorXd> gauss(int n) {
    Eigen::MatrixXd J=Eigen::MatrixXd::Zero(n,n);
    for(int i=1;i<n;++i) J(i-1,i)=J(i,i-1)=i/std::sqrt(4.*i*i-1.);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> e(J);
    if(e.info()!=Eigen::Success) throw std::runtime_error("trace93 Gauss rule failed");
    return {e.eigenvalues(),2*e.eigenvectors().row(0).array().square().matrix().transpose()};
}
std::vector<double> knots(int ne) {
    std::vector<double> k(4,0.);
    for(int i=1;i<ne;++i) k.push_back(double(i)/ne);
    k.insert(k.end(),4,1.);return k;
}
Eigen::MatrixXd basis(int ne,double x) {
    const auto k=knots(ne);x=std::clamp(x,0.,1.);
    Eigen::MatrixXd a=Eigen::MatrixXd::Zero(4,k.size()-1);
    const int span=x==1.?ne+2:int(std::upper_bound(k.begin(),k.end(),x)-k.begin())-1;
    a(0,span)=1.;
    for(int p=1;p<=3;++p) {
        Eigen::MatrixXd b=Eigen::MatrixXd::Zero(4,k.size()-p-1);
        for(int i=0;i<b.cols();++i) {
            const double l=k[i+p]-k[i],r=k[i+p+1]-k[i+1];
            if(l>0) b(0,i)+=(x-k[i])/l*a(0,i);
            if(r>0) b(0,i)+=(k[i+p+1]-x)/r*a(0,i+1);
            for(int d=1;d<=p;++d) {
                if(l>0) b(d,i)+=p/l*a(d-1,i);
                if(r>0) b(d,i)-=p/r*a(d-1,i+1);
            }
        }
        a=std::move(b);
    }
    return a;
}
Eigen::VectorXd greville(int ne) {
    const auto k=knots(ne);Eigen::VectorXd x(ne+3);
    for(int i=0;i<x.size();++i)x[i]=(k[i+1]+k[i+2]+k[i+3])/3.;return x;
}
bool u_edge(Edge e){return e==Edge::UMin||e==Edge::UMax;}
bool low_edge(Edge e){return e==Edge::UMin||e==Edge::VMin;}
Eigen::Vector2d edge_uv(Edge e,double s) {
    switch(e){case Edge::UMin:return {0,s};case Edge::UMax:return {1,s};
        case Edge::VMin:return {s,0};case Edge::VMax:return {s,1};}
    throw std::invalid_argument("invalid trace93 edge");
}
int edge_spans(const Trace93DensityLayout3D& s,int p,Edge e) {
    return s.patch_spans.at(p)[u_edge(e)?1:0];
}
int cid(const Trace93DensityLayout3D& s,int p,int i,int j) {
    return s.patch_offsets.at(p)+j*(s.patch_spans.at(p)[0]+3)+i;
}
std::pair<int,int> edge_ids(const Trace93DensityLayout3D& s,int p,Edge e,int k) {
    const int nu=s.patch_spans[p][0]+3,nv=s.patch_spans[p][1]+3;
    if(e==Edge::UMin)return {cid(s,p,0,k),cid(s,p,1,k)};
    if(e==Edge::UMax)return {cid(s,p,nu-1,k),cid(s,p,nu-2,k)};
    if(e==Edge::VMin)return {cid(s,p,k,0),cid(s,p,k,1)};
    return {cid(s,p,k,nv-1),cid(s,p,k,nv-2)};
}
Eigen::Vector3d inward(const Trace93DensityLayout3D& s,int p,Edge e,double t) {
    const auto uv=edge_uv(e,t);const auto g=s.geometry_jet(p,uv[0],uv[1]);
    Eigen::Vector3d v=u_edge(e)?g.lower.x_u:g.lower.x_v;
    if(!low_edge(e))v=-v;return v.normalized();
}
Eigen::Vector3d tangent(const Trace93DensityLayout3D& s,int p,Edge e,double t) {
    const auto uv=edge_uv(e,t);const auto g=s.geometry_jet(p,uv[0],uv[1]);
    return (u_edge(e)?g.lower.x_v:g.lower.x_u).normalized();
}
double edge_speed(const Trace93DensityLayout3D& s,int p,Edge e,double t) {
    const auto uv=edge_uv(e,t);const auto g=s.geometry_jet(p,uv[0],uv[1]);
    return (u_edge(e)?g.lower.x_v:g.lower.x_u).norm();
}
struct Rows {
    int n;std::vector<Row> rows;std::vector<double> rhs;std::vector<std::string> kinds;
    void add(Row r,double b,const std::string& k) {
        const double norm=r.norm();if(norm<=1e-13)return;
        rows.push_back(r/norm);rhs.push_back(b/norm);kinds.push_back(k);
    }
    Eigen::MatrixXd matrix() const {
        Eigen::MatrixXd C(rows.size(),n);for(int i=0;i<C.rows();++i)C.row(i)=rows[i];return C;
    }
    Eigen::VectorXd values() const {
        Eigen::VectorXd d(rhs.size());for(int i=0;i<d.size();++i)d[i]=rhs[i];return d;
    }
};
struct Reduction {int rank=0;Eigen::VectorXd cp;Eigen::MatrixXd Z;double residual=0;};
Reduction reduce(const Eigen::MatrixXd& C,const Eigen::VectorXd& d,double factor) {
    Reduction r;
    if(C.rows()==0){r.cp=Eigen::VectorXd::Zero(C.cols());r.Z=Eigen::MatrixXd::Identity(C.cols(),C.cols());return r;}
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(C.transpose());
    const auto diagonal=qr.matrixR().diagonal().head(std::min(C.rows(),C.cols())).cwiseAbs().eval();
    const double tol=std::max(C.rows(),C.cols())*eps*(diagonal.size()?diagonal[0]:1.)*factor;
    for(int i=0;i<diagonal.size();++i)if(diagonal[i]>tol)++r.rank;
    const Eigen::MatrixXd Q=qr.householderQ()*Eigen::MatrixXd::Identity(C.cols(),C.cols());
    r.Z=Q.rightCols(C.cols()-r.rank);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(C,Eigen::ComputeThinU|Eigen::ComputeThinV);
    svd.setThreshold(tol);r.cp=svd.solve(d);
    r.residual=(C*r.cp-d).cwiseAbs().maxCoeff();return r;
}
Sparse sparse_rows(const std::vector<Row>& rows,int cols) {
    std::vector<Triplet> t;
    for(int i=0;i<int(rows.size());++i)for(int j=0;j<cols;++j)
        if(std::abs(rows[i][j])>1e-15)t.emplace_back(i,j,rows[i][j]);
    Sparse s(rows.size(),cols);s.setFromTriplets(t.begin(),t.end());return s;
}
int dyadic(double x) {int n=2;while(n<std::max(2,int(std::ceil(x))))n*=2;return n;}

void choose_counts(Trace93DensityLayout3D& s,double factor,bool cylinder) {
    const auto& g=*s.problem;const int np=int(g.analysis_patches.size());
    s.patch_spans.resize(np);s.patch_offsets.resize(np);
    const auto rule=gauss(10);
    for(int p=0;p<np;++p)for(int d=0;d<2;++d) {
        double L=0;
        if(!cylinder) L=d==0?g.analysis_patches[p].Lu:g.analysis_patches[p].Lv;
        else for(int k=0;k<7;++k) {
            double length=0;
            for(int q=0;q<10;++q) {
                const double x=(rule.first[q]+1)*.5,t=double(k)/6;
                const auto a=g.analysis_at(p,d==0?x:t,d==0?t:x,1);
                length+=.5*rule.second[q]*a.d[d==0?1:0][d==0?0:1].norm();
            }
            L=std::max(L,length);
        }
        // Analytic lengths at an exact integer ratio must not refine solely
        // because a rigid rotation introduces a few floating-point ulps.
        const double ratio=L/(factor*s.h);
        const double adjusted=ratio-32*eps*std::max(1.,std::abs(ratio));
        s.patch_spans[p][d]=cylinder?dyadic(adjusted):std::max(2,int(std::ceil(adjusted)));
    }
    if(cylinder) {
        std::vector<int> parent(2*np);std::iota(parent.begin(),parent.end(),0);
        const auto root=[&](int a){while(parent[a]!=a)a=parent[a];return a;};
        for(const auto& c:g.surface.geometric_connections) {
            const int a=root(2*c.first.patch+(u_edge(c.first.edge)?1:0));
            const int b=root(2*c.second.patch+(u_edge(c.second.edge)?1:0));parent[b]=a;
        }
        std::map<int,int> maximum;
        for(int p=0;p<np;++p)for(int d=0;d<2;++d)maximum[root(2*p+d)]=std::max(maximum[root(2*p+d)],s.patch_spans[p][d]);
        for(int p=0;p<np;++p)for(int d=0;d<2;++d)s.patch_spans[p][d]=maximum[root(2*p+d)];
    }
    for(int p=0;p<np;++p) {s.patch_offsets[p]=s.reference_raw_dofs;
        s.reference_raw_dofs+=(s.patch_spans[p][0]+3)*(s.patch_spans[p][1]+3);}
}

void make_traces(Trace93DensityLayout3D& s) {
    const auto rule=gauss(4);std::vector<Triplet> triplets;
    s.mean_row=Row::Zero(s.reference_raw_dofs);
    for(int p=0;p<int(s.patch_spans.size());++p) {
        const int nu=s.patch_spans[p][0],nv=s.patch_spans[p][1];
        for(int a=0;a<nu;++a)for(int b=0;b<nv;++b)for(int i=0;i<4;++i)for(int j=0;j<4;++j) {
            const double u=(a+(rule.first[i]+1)*.5)/nu,v=(b+(rule.first[j]+1)*.5)/nv;
            const auto g=s.geometry_jet(p,u,v);
            NativeDensityGaussPoint3D q;q.patch=p;q.element_u=a;q.element_v=b;q.u=u;q.v=v;
            q.point=g.lower.point;q.normal=g.lower.normal;
            q.surface_weight=rule.second[i]*rule.second[j]*g.lower.x_u.cross(g.lower.x_v).norm()/(4.*nu*nv);
            const auto r=s.basis_stencil(p,u,v);const int row=int(s.traces.size());
            for(int k=0;k<r.count;++k) {triplets.emplace_back(row,r.indices[k],r.weights[k]);
                s.mean_row[r.indices[k]]+=q.surface_weight*r.weights[k];}
            s.traces.push_back(q);
        }
    }
    s.weights.resize(s.traces.size());for(int i=0;i<s.weights.size();++i)s.weights[i]=s.traces[i].surface_weight;
    s.trace_basis.resize(s.traces.size(),s.reference_raw_dofs);s.trace_basis.setFromTriplets(triplets.begin(),triplets.end());
}

void essential_constraints(Trace93DensityLayout3D& s,Rows& rows,bool cylinder) {
    for(const auto& c:s.problem->surface.geometric_connections) {
        const int pa=c.first.patch,pb=c.second.patch;const Edge ea=c.first.edge,eb=c.second.edge;
        const int ne=edge_spans(s,pa,ea);if(ne!=edge_spans(s,pb,eb))throw std::runtime_error("trace93 edge knot mismatch");
        const int nc=ne+3;const auto x=greville(ne);
        if(!cylinder) {
            const auto ga=s.geometry_jet(pa,.5,.5),gb=s.geometry_jet(pb,.5,.5);
            const double aa=3.*s.patch_spans[pa][u_edge(ea)?0:1]/(u_edge(ea)?ga.lower.x_u:ga.lower.x_v).norm();
            const double ab=3.*s.patch_spans[pb][u_edge(eb)?0:1]/(u_edge(eb)?gb.lower.x_u:gb.lower.x_v).norm();
            Eigen::VectorXd va(nc),vb(nc);
            if(!s.neumann&&!c.g1)for(int k=0;k<nc;++k) {
                const auto uv=edge_uv(ea,x[k]);const auto xa=s.geometry_jet(pa,uv[0],uv[1]);
                const auto grad=s.problem->evaluate(xa.lower.point).gradient;
                Eigen::Matrix<double,3,2> A;A.col(0)=ga.lower.normal;A.col(1)=-gb.lower.normal;
                const Eigen::Vector3d ta=grad-grad.dot(ga.lower.normal)*ga.lower.normal;
                const Eigen::Vector3d tb=grad-grad.dot(gb.lower.normal)*gb.lower.normal;
                const Eigen::Vector2d mu=A.colPivHouseholderQr().solve(tb-ta);va[k]=mu[0];vb[k]=mu[1];
            }
            if(!s.neumann&&!c.g1) {
                Eigen::MatrixXd G(nc,nc);for(int k=0;k<nc;++k)G.row(k)=basis(ne,x[k]).row(0);
                // Interpolate known edge data to its affine lifting, never fit
                // an unknown density or its jet from a cloud of samples.
                Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(G);va=qr.solve(va).eval();vb=qr.solve(vb).eval();
            }
            for(int k=0;k<nc;++k) {
                const auto a=edge_ids(s,pa,ea,k),b=edge_ids(s,pb,eb,c.reversed?nc-1-k:k);
                if(c.g1||s.neumann) {Row r=Row::Zero(rows.n);r[a.first]=1;r[b.first]-=1;rows.add(r,0,c.g1?"C0_smooth":"C0_feature");}
                if(c.g1) {Row r=Row::Zero(rows.n);r[a.second]+=aa;r[a.first]-=aa;r[b.second]+=ab;r[b.first]-=ab;rows.add(r,0,"C1_smooth");}
                else if(!s.neumann) {Row arow=Row::Zero(rows.n),brow=arow;arow[a.first]=1;brow[b.first]=1;rows.add(arow,va[k],"feature_value_a");rows.add(brow,vb[k],"feature_value_b");}
            }
        } else for(int k=0;k<nc;++k) {
            const double sa=x[k],sb=c.reversed?1-sa:sa;const auto a=edge_uv(ea,sa),b=edge_uv(eb,sb);
            const auto ga=s.geometry_jet(pa,a[0],a[1]),gb=s.geometry_jet(pb,b[0],b[1]);
            if(c.g1||s.neumann)rows.add(s.basis_row(pa,a[0],a[1])-s.basis_row(pb,b[0],b[1]),0,c.g1?"C0_smooth":"C0_feature");
            if(c.g1) {
                const Eigen::Vector3d m=tangent(s,pa,ea,sa).cross(ga.lower.normal).normalized();
                rows.add(s.directional_row(pa,a[0],a[1],m)-s.directional_row(pb,b[0],b[1],m),0,"C1_smooth");
            } else if(!s.neumann&&k!=0&&k!=nc-1) {
                const auto grad=s.problem->evaluate(ga.lower.point).gradient;
                Eigen::Matrix<double,3,2> A;A.col(0)=ga.lower.normal;A.col(1)=-gb.lower.normal;
                const Eigen::Vector3d ta=grad-grad.dot(ga.lower.normal)*ga.lower.normal;
                const Eigen::Vector3d tb=grad-grad.dot(gb.lower.normal)*gb.lower.normal;
                const Eigen::Vector2d mu=A.colPivHouseholderQr().solve(tb-ta);
                rows.add(s.basis_row(pa,a[0],a[1]),mu[0],"feature_value_a");
                rows.add(s.basis_row(pb,b[0],b[1]),mu[1],"feature_value_b");
            }
        }
    }
    if(s.neumann)rows.add(s.mean_row,0,"mean");
}

void polar_constraints(Trace93DensityLayout3D& s,Rows& rows,bool cylinder) {
    Rows polar{rows.n};const int order=cylinder?12:10;const auto rule=gauss(order);
    for(const auto& c:s.problem->surface.geometric_connections)if(!c.g1) {
        const int pa=c.first.patch,pb=c.second.patch;const Edge ea=c.first.edge,eb=c.second.edge;
        const int ne=edge_spans(s,pa,ea),nc=ne+3;
        Eigen::MatrixXd Ra=Eigen::MatrixXd::Zero(nc,rows.n),Rb=Ra;
        Eigen::VectorXd da=Eigen::VectorXd::Zero(nc),db=da;
        for(int e=0;e<ne;++e)for(int q=0;q<order;++q) {
            const double sa=(e+(rule.first[q]+1)*.5)/ne,sb=c.reversed?1-sa:sa;
            const auto a=edge_uv(ea,sa),b=edge_uv(eb,sb);const auto ga=s.geometry_jet(pa,a[0],a[1]),gb=s.geometry_jet(pb,b[0],b[1]);
            const Eigen::Vector3d tau=tangent(s,pa,ea,sa),ma=inward(s,pa,ea,sa),mb=inward(s,pb,eb,sb);
            Eigen::Matrix3d Q;Q.row(0)=tau.transpose();Q.row(1)=ga.lower.normal.transpose();Q.row(2)=gb.lower.normal.transpose();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(Q);const auto sv=svd.singularValues();
            if(sv[2]<1e-12)throw std::runtime_error("trace93 polar singular feature frame");
            const Eigen::RowVector3d ca=ma.transpose()*Q.inverse(),cb=mb.transpose()*Q.inverse();
            const Row rt=s.directional_row(pa,a[0],a[1],tau);
            const Row ra=s.directional_row(pa,a[0],a[1],ma)-ca[0]*rt;
            const Row rb=s.directional_row(pb,b[0],b[1],mb)-cb[0]*rt;
            const auto grad=s.problem->evaluate(ga.lower.point).gradient;
            const double fa=ca[1]*grad.dot(ga.lower.normal)+ca[2]*grad.dot(gb.lower.normal);
            const double fb=cb[1]*grad.dot(ga.lower.normal)+cb[2]*grad.dot(gb.lower.normal);
            const Eigen::VectorXd w=(rule.second[q]*.5/ne*edge_speed(s,pa,ea,sa))*basis(ne,sa).row(0).transpose();
            Ra+=w*ra;Rb+=w*rb;da+=w*fa;db+=w*fb;
            s.polar.maximum_condition=std::max(s.polar.maximum_condition,sv[0]/sv[2]);
            s.polar.maximum_beta=std::max({s.polar.maximum_beta,std::abs(ca[0]),std::abs(cb[0])});
            s.polar.maximum_point_mismatch=std::max(s.polar.maximum_point_mismatch,(ga.lower.point-gb.lower.point).norm());
        }
        for(int i=0;i<nc;++i)polar.add(Ra.row(i),da[i],"polar_a");
        for(int i=0;i<nc;++i)polar.add(Rb.row(i),db[i],"polar_b");
    }
    if(polar.rows.empty())return;
    const auto essential=reduce(rows.matrix(),rows.values(),80);
    const Eigen::MatrixXd Cp=polar.matrix();const Eigen::VectorXd dp=polar.values();
    const Eigen::MatrixXd A=Cp*essential.Z;const Eigen::VectorXd b=dp-Cp*essential.cp;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeThinU|Eigen::ComputeThinV);
    const auto sv=svd.singularValues();const double tol=std::max(A.rows(),A.cols())*eps*(sv.size()?sv[0]:1.)*80;
    int r=0;for(int i=0;i<sv.size();++i)if(sv[i]>tol)++r;
    int kept=r;double gap=1.;
    for(int i=0;i+1<r;++i)if(sv[i]/sv[i+1]>gap){gap=sv[i]/sv[i+1];kept=i+1;}
    if(gap<10)kept=r;
    s.polar.essential_rank=essential.rank;s.polar.polar_rank=r;s.polar.kept=kept;s.polar.defect=r-kept;
    s.polar.gap_factor=gap;s.polar.singular_value_ratios=r?Eigen::VectorXd(sv.head(r)/sv[0]):Eigen::VectorXd();
    if(r<svd.matrixU().cols())s.polar.left_null_rhs=(svd.matrixU().rightCols(svd.matrixU().cols()-r).transpose()*b).norm();
    const Eigen::MatrixXd W=svd.matrixU().leftCols(kept).transpose();
    const Eigen::MatrixXd Cs=W*Cp;const Eigen::VectorXd ds=W*dp;
    for(int i=0;i<kept;++i)rows.add(Cs.row(i),ds[i],"polar_star");
}

Row extrapolated_trace(const Trace93DensityLayout3D& s,int p,double u,double v) {
    const int nu=s.patch_spans[p][0],nv=s.patch_spans[p][1];
    const int a=std::clamp(int(std::floor(u*nu)),0,nu-1),b=std::clamp(int(std::floor(v*nv)),0,nv-1);
    int base=0;for(int k=0;k<p;++k)base+=16*s.patch_spans[k][0]*s.patch_spans[k][1];base+=16*(a*nv+b);
    Eigen::Vector4d U,V,wu=Eigen::Vector4d::Ones(),wv=wu;
    for(int i=0;i<4;++i){U[i]=std::round(s.traces[base+4*i].u*1e15)/1e15;V[i]=std::round(s.traces[base+i].v*1e15)/1e15;}
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)if(i!=j){wu[i]*=(u-U[j])/(U[i]-U[j]);wv[i]*=(v-V[j])/(V[i]-V[j]);}
    Row row=Row::Zero(s.traces.size());for(int i=0;i<4;++i)for(int j=0;j<4;++j)row[base+4*i+j]=wu[i]*wv[j];return row;
}
void projector_rows(Trace93DensityLayout3D& s,const Trace93DensityOptions3D& options,bool cylinder) {
    std::vector<Row> et,eb,vt,vb;std::vector<double> ew,vw;const auto rule=gauss(4);
    if(s.neumann)for(const auto& c:s.problem->surface.geometric_connections)if(!c.g1) {
        const int pa=c.first.patch,pb=c.second.patch,ne=edge_spans(s,pa,c.first.edge);
        for(int e=0;e<ne;++e)for(int q=0;q<4;++q) {
            const double sa=(e+(rule.first[q]+1)*.5)/ne,sb=c.reversed?1-sa:sa;
            const auto a=edge_uv(c.first.edge,sa),b=edge_uv(c.second.edge,sb);
            et.push_back(.5*(extrapolated_trace(s,pa,a[0],a[1])+extrapolated_trace(s,pb,b[0],b[1])));
            eb.push_back(.5*(s.basis_row(pa,a[0],a[1])+s.basis_row(pb,b[0],b[1])));
            ew.push_back(options.edge_star_weight*rule.second[q]*.5/ne*edge_speed(s,pa,c.first.edge,sa));
        }
    }
    if(!cylinder&&s.neumann) {
        struct Corner {int p;double u,v,area;Eigen::Vector3d n;};
        std::map<std::array<long long,3>,std::vector<Corner>> groups;
        for(int p=0;p<int(s.patch_spans.size());++p)for(double u:{0.,1.})for(double v:{0.,1.}) {
            const auto g=s.geometry_jet(p,u,v);std::array<long long,3> key;
            for(int k=0;k<3;++k)key[k]=std::llround(g.lower.point[k]/1e-9);
            groups[key].push_back({p,u,v,g.lower.x_u.cross(g.lower.x_v).norm()/(s.patch_spans[p][0]*s.patch_spans[p][1]),g.lower.normal});
        }
        for(const auto& [key,corners]:groups) {
            (void)key;Eigen::MatrixXd normals(corners.size(),3);
            for(int i=0;i<normals.rows();++i)normals.row(i)=corners[i].n.transpose();
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(normals);if(svd.singularValues().size()<3||svd.singularValues()[2]<1e-10)continue;
            Row tr=Row::Zero(s.traces.size()),br=Row::Zero(s.reference_raw_dofs);double area=0;
            for(const auto& c:corners){tr+=extrapolated_trace(s,c.p,c.u,c.v);br+=s.basis_row(c.p,c.u,c.v);area+=c.area;}
            vt.push_back(tr/double(corners.size()));vb.push_back(br/double(corners.size()));
            vw.push_back(options.vertex_star_weight*std::pow(s.h,-options.vertex_star_power)*area/corners.size());
        }
    }
    s.edge_trace=sparse_rows(et,s.traces.size());s.edge_basis=sparse_rows(eb,s.reference_raw_dofs);
    s.vertex_trace=sparse_rows(vt,s.traces.size());s.vertex_basis=sparse_rows(vb,s.reference_raw_dofs);
    s.edge_weights.resize(ew.size());for(int i=0;i<s.edge_weights.size();++i)s.edge_weights[i]=ew[i];
    s.vertex_weights.resize(vw.size());for(int i=0;i<s.vertex_weights.size();++i)s.vertex_weights[i]=vw[i];
}
} // namespace

std::array<NativeDensityC0Stencil3D,10> Trace93DensityLayout3D::parameter_cubic_jet_stencils(int p,double u,double v) const {
    const auto& counts=patch_spans.at(p);const auto A=basis(counts[0],u),B=basis(counts[1],v);
    std::array<NativeDensityC0Stencil3D,10> rows;
    for(int j=0;j<B.cols();++j)if(B.col(j).cwiseAbs().sum()>1e-14)
        for(int i=0;i<A.cols();++i)if(A.col(i).cwiseAbs().sum()>1e-14) {
            for(int k=0;k<10;++k) {auto& r=rows[k];if(r.count>=16)throw std::runtime_error("trace93 cubic support overflow");
                r.indices[r.count]=cid(*this,p,i,j);r.weights[r.count]=A(deriv[k][0],i)*B(deriv[k][1],j);++r.count;}
        }
    return rows;
}
NativeDensityC0Stencil3D Trace93DensityLayout3D::basis_stencil(int p,double u,double v) const {return parameter_cubic_jet_stencils(p,u,v)[0];}
Row Trace93DensityLayout3D::basis_row(int p,double u,double v) const {
    Row r=Row::Zero(reference_raw_dofs);const auto b=basis_stencil(p,u,v);for(int i=0;i<b.count;++i)r[b.indices[i]]+=b.weights[i];return r;
}
Row Trace93DensityLayout3D::directional_row(int p,double u,double v,const Eigen::Vector3d& direction) const {
    const auto g=geometry_jet(p,u,v);Eigen::Matrix<double,3,2> T;T.col(0)=g.lower.x_u;T.col(1)=g.lower.x_v;
    const Eigen::Vector2d a=(T.transpose()*T).ldlt().solve(T.transpose()*direction);
    const auto b=parameter_cubic_jet_stencils(p,u,v);Row r=Row::Zero(reference_raw_dofs);
    for(int i=0;i<b[0].count;++i)r[b[0].indices[i]]+=a[0]*b[1].weights[i]+a[1]*b[2].weights[i];return r;
}
NativeSurfaceCubicParameterJet3D Trace93DensityLayout3D::geometry_jet(int p,double u,double v) const {
    if(!problem)throw std::logic_error("trace93 layout has no analysis geometry");
    const auto d=problem->analysis_at(p,u,v,3);NativeSurfaceCubicParameterJet3D g;
    g.lower.point=d.d[0][0];g.lower.x_u=d.d[1][0];g.lower.x_v=d.d[0][1];
    g.lower.x_uu=d.d[2][0];g.lower.x_uv=d.d[1][1];g.lower.x_vv=d.d[0][2];
    const auto n=g.lower.x_u.cross(g.lower.x_v);if(n.norm()<1e-14)throw std::runtime_error("trace93 degenerate analysis chart");
    g.lower.normal=n.normalized();g.x_uuu=d.d[3][0];g.x_uuv=d.d[2][1];g.x_uvv=d.d[1][2];g.x_vvv=d.d[0][3];return g;
}
DirectCoefficientCubicCauchyPlan3D Trace93DensityLayout3D::cauchy_plan(int p,double u,double v) const {
    DirectCoefficientCubicCauchyPlan3D plan;plan.patch=p;plan.u=u;plan.v=v;
    const auto g=geometry_jet(p,u,v);plan.center=g.lower.point;plan.frame=make_local_orthonormal_frame_3d(g.lower.normal,g.lower.x_u);
    const auto map=parameter_to_cubic_cauchy_jet_matrix_3d(g,plan.frame);plan.graph=tangent_graph_third_jet_3d(g,plan.frame);
    plan.closure=build_cubic_cauchy_closure_3d(plan.graph);const auto rows=parameter_cubic_jet_stencils(p,u,v);
    for(int k=0;k<10;++k) {plan.value_rows[k]=rows[0];for(int i=0;i<rows[0].count;++i) {
        double value=0;for(int j=0;j<10;++j)value+=map(k,j)*rows[j].weights[i];plan.value_rows[k].weights[i]=value;}
        if(k<6)plan.normal_rows[k]=plan.value_rows[k];}
    return plan;
}
Trace93DensityLayout3D build_trace93_density_layout_3d(const Trace93Case3D& problem,double h,bool neumann,Trace93DensityOptions3D options) {
    if(!(h>0)||!(options.factor>0)||options.edge_star_weight<0||options.vertex_star_weight<0)
        throw std::invalid_argument("trace93 density options must be positive/nonnegative");
    Trace93DensityLayout3D s;s.problem=&problem;s.h=h;s.neumann=neumann;
    bool cylinder=false;for(const auto& p:problem.analysis_patches)if(!p.planar)cylinder=true;
    choose_counts(s,options.factor,cylinder);make_traces(s);Rows rows{s.reference_raw_dofs};
    essential_constraints(s,rows,cylinder);if(neumann)polar_constraints(s,rows,cylinder);
    s.C=rows.matrix();s.d=rows.values();s.constraint_kinds=rows.kinds;
    const auto reduced=reduce(s.C,s.d,cylinder?40:30);s.rank=reduced.rank;s.Z=reduced.Z;s.particular=reduced.cp;s.constraint_residual=reduced.residual;
    s.nullspace_residual=s.Z.cols()?(s.C*s.Z).cwiseAbs().maxCoeff():0.;
    if(s.traces.size()<=std::size_t(s.Z.cols()))throw std::runtime_error("trace93 exterior samples must exceed independent DOFs");
    if(!s.Z.allFinite()||!s.particular.allFinite())throw std::runtime_error("trace93 nonfinite affine reduction");
    projector_rows(s,options,cylinder);return s;
}
} // namespace kfbim::app3d
