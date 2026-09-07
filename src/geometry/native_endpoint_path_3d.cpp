#include "native_endpoint_path_3d.hpp"
#include "native_nurbs_exact_geometry_3d.hpp"

#include <CGAL/number_utils.h>
#ifdef KFBIM_NATIVE_ENDPOINT_HAS_MPFR
#include <mpfr.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace kfbim::geometry3d {
namespace {

using Q = CGAL::Gmpq;
thread_local unsigned interval_precision = 53;
double down(double x) { return std::nextafter(x, -std::numeric_limits<double>::infinity()); }
double up(double x) { return std::nextafter(x, std::numeric_limits<double>::infinity()); }

struct Stop {
    PathStatus3D status;
    std::string reason;
};
struct NeedPrecision {};

// All wide intervals are immutable after construction. Shared ownership only
// avoids expensive copies of temporary MPFR endpoints; operations never mutate
// an operand. Binary64 operations use outward nextafter and no fast-math.
class I {
public:
    double lo = 0.0, hi = 0.0;
#ifdef KFBIM_NATIVE_ENDPOINT_HAS_MPFR
    struct Wide {
        mpfr_t lo, hi;
        explicit Wide(unsigned p) { mpfr_init2(lo, p); mpfr_init2(hi, p); }
        ~Wide() { mpfr_clear(lo); mpfr_clear(hi); }
    };
    std::shared_ptr<Wide> wide;
#endif
    I() = default;
    explicit I(double x) : I(x, x) {}
    I(double a, double b) : lo(a), hi(b) {
#ifdef KFBIM_NATIVE_ENDPOINT_HAS_MPFR
        if (interval_precision > 53) {
            wide = std::make_shared<Wide>(interval_precision);
            mpfr_set_d(wide->lo, a, MPFR_RNDD);
            mpfr_set_d(wide->hi, b, MPFR_RNDU);
        }
#endif
    }
    explicit I(const Q& x) {
#ifdef KFBIM_NATIVE_ENDPOINT_HAS_MPFR
        if (interval_precision > 53) {
            wide = std::make_shared<Wide>(interval_precision);
            mpfr_set_q(wide->lo, x.mpq(), MPFR_RNDD);
            mpfr_set_q(wide->hi, x.mpq(), MPFR_RNDU);
            sync();
            return;
        }
#endif
        const auto bounds = CGAL::to_interval(x);
        lo = bounds.first; hi = bounds.second;
    }
    bool finite() const { return std::isfinite(lo) && std::isfinite(hi) && lo <= hi; }
    bool zero() const {
        if (!finite()) throw Stop{PathStatus3D::Unresolved, "NonFiniteIntervalPredicate"};
        return lo <= 0.0 && hi >= 0.0;
    }
    double mid() const {
        if (!finite()) throw Stop{PathStatus3D::Unresolved, "NonFiniteIntervalMidpoint"};
        const double value = (std::signbit(lo) == std::signbit(hi))
            ? lo + (hi - lo) * 0.5 : lo * 0.5 + hi * 0.5;
        return std::clamp(value, lo, hi);
    }
    double magnitude() const { return std::max(std::abs(lo), std::abs(hi)); }
    void sync() {
#ifdef KFBIM_NATIVE_ENDPOINT_HAS_MPFR
        if (wide) {
            lo = mpfr_get_d(wide->lo, MPFR_RNDD);
            hi = mpfr_get_d(wide->hi, MPFR_RNDU);
        }
#endif
    }
};

#ifdef KFBIM_NATIVE_ENDPOINT_HAS_MPFR
void get_endpoint(mpfr_ptr out, const I& a, bool upper) {
    if (a.wide) mpfr_set(out, upper ? a.wide->hi : a.wide->lo, upper ? MPFR_RNDU : MPFR_RNDD);
    else mpfr_set_d(out, upper ? a.hi : a.lo, upper ? MPFR_RNDU : MPFR_RNDD);
}
I wide_binary(const I& a, const I& b, char operation) {
    I r(0.0);
    mpfr_t al, ah, bl, bh, tmp;
    mpfr_inits2(interval_precision, al, ah, bl, bh, tmp, (mpfr_ptr)nullptr);
    get_endpoint(al, a, false); get_endpoint(ah, a, true);
    get_endpoint(bl, b, false); get_endpoint(bh, b, true);
    if (operation == '+') {
        mpfr_add(r.wide->lo, al, bl, MPFR_RNDD);
        mpfr_add(r.wide->hi, ah, bh, MPFR_RNDU);
    } else if (operation == '-') {
        mpfr_sub(r.wide->lo, al, bh, MPFR_RNDD);
        mpfr_sub(r.wide->hi, ah, bl, MPFR_RNDU);
    } else {
        mpfr_set_inf(r.wide->lo, +1); mpfr_set_inf(r.wide->hi, -1);
        for (mpfr_srcptr x : {static_cast<mpfr_srcptr>(al), static_cast<mpfr_srcptr>(ah)}) {
            for (mpfr_srcptr y : {static_cast<mpfr_srcptr>(bl), static_cast<mpfr_srcptr>(bh)}) {
                if (operation == '*') mpfr_mul(tmp, x, y, MPFR_RNDD);
                else mpfr_div(tmp, x, y, MPFR_RNDD);
                if (mpfr_less_p(tmp, r.wide->lo)) mpfr_set(r.wide->lo, tmp, MPFR_RNDD);
                if (operation == '*') mpfr_mul(tmp, x, y, MPFR_RNDU);
                else mpfr_div(tmp, x, y, MPFR_RNDU);
                if (mpfr_greater_p(tmp, r.wide->hi)) mpfr_set(r.wide->hi, tmp, MPFR_RNDU);
            }
        }
    }
    mpfr_clears(al, ah, bl, bh, tmp, (mpfr_ptr)nullptr);
    r.sync();
    return r;
}
#endif

I operator+(const I& a, const I& b) {
    if (!a.finite() || !b.finite()) throw Stop{PathStatus3D::Unresolved, "NonFiniteIntervalOperand"};
#ifdef KFBIM_NATIVE_ENDPOINT_HAS_MPFR
    if (interval_precision > 53) return wide_binary(a, b, '+');
#endif
    return I(down(a.lo + b.lo), up(a.hi + b.hi));
}
I operator-(const I& a, const I& b) {
    if (!a.finite() || !b.finite()) throw Stop{PathStatus3D::Unresolved, "NonFiniteIntervalOperand"};
#ifdef KFBIM_NATIVE_ENDPOINT_HAS_MPFR
    if (interval_precision > 53) return wide_binary(a, b, '-');
#endif
    return I(down(a.lo - b.hi), up(a.hi - b.lo));
}
I operator*(const I& a, const I& b) {
    if (!a.finite() || !b.finite()) throw Stop{PathStatus3D::Unresolved, "NonFiniteIntervalOperand"};
#ifdef KFBIM_NATIVE_ENDPOINT_HAS_MPFR
    if (interval_precision > 53) return wide_binary(a, b, '*');
#endif
    const std::array<double, 4> p{{a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi}};
    return I(down(*std::min_element(p.begin(), p.end())), up(*std::max_element(p.begin(), p.end())));
}
I operator/(const I& a, const I& b) {
    if (!a.finite() || !b.finite()) throw Stop{PathStatus3D::Unresolved, "NonFiniteIntervalOperand"};
    if (b.zero()) throw Stop{PathStatus3D::Unresolved, "IntervalDivisionContainsZero"};
#ifdef KFBIM_NATIVE_ENDPOINT_HAS_MPFR
    if (interval_precision > 53) return wide_binary(a, b, '/');
#endif
    const std::array<double, 4> p{{a.lo/b.lo, a.lo/b.hi, a.hi/b.lo, a.hi/b.hi}};
    return I(down(*std::min_element(p.begin(), p.end())), up(*std::max_element(p.begin(), p.end())));
}
I square(const I& a) {
    if (!a.zero()) return a*a;
    const I z = a*a;
    return I(0.0, z.hi);
}
I square_root(const I& a) {
    if (!a.finite()) throw Stop{PathStatus3D::Unresolved, "NonFiniteNormInterval"};
    if (a.lo < 0.0) throw Stop{PathStatus3D::Unresolved, "NegativeNormInterval"};
#ifdef KFBIM_NATIVE_ENDPOINT_HAS_MPFR
    if (interval_precision > 53) {
        I r(0.0);
        mpfr_t lower, upper;
        mpfr_inits2(interval_precision, lower, upper, (mpfr_ptr)nullptr);
        get_endpoint(lower, a, false); get_endpoint(upper, a, true);
        mpfr_sqrt(r.wide->lo, lower, MPFR_RNDD); mpfr_sqrt(r.wide->hi, upper, MPFR_RNDU);
        mpfr_clears(lower, upper, (mpfr_ptr)nullptr); r.sync(); return r;
    }
#endif
    return I(a.lo == 0.0 ? 0.0 : down(std::sqrt(a.lo)), up(std::sqrt(a.hi)));
}

struct Box { double u0, u1, v0, v1; };
Box box_of(const ExactNativeBezierElement3D& e) {
    return {e.parameter_u0,e.parameter_u1,e.parameter_v0,e.parameter_v1};
}
bool contains(const Box& b, double u, double v) { return b.u0 <= u && u <= b.u1 && b.v0 <= v && v <= b.v1; }
bool contains(const Box& a, const Box& b) { return a.u0 <= b.u0 && b.u1 <= a.u1 && a.v0 <= b.v0 && b.v1 <= a.v1; }
Box united(const Box& a, const Box& b) { return {std::min(a.u0,b.u0),std::max(a.u1,b.u1),std::min(a.v0,b.v0),std::max(a.v1,b.v1)}; }
RootParameterBox3D public_box(const Box& b) { return {{b.u0,b.u1},{b.v0,b.v1}}; }
Box private_box(const RootParameterBox3D& b) { return {b.u.lower,b.u.upper,b.v.lower,b.v.upper}; }
I fraction(double x, double a, double b) { return I((Q(x)-Q(a))/(Q(b)-Q(a))); }

template<int N> struct Net {
    int du = 0, dv = 0;
    std::vector<std::array<I,N>> c;
    const std::array<I,N>& at(int u,int v) const { return c[static_cast<std::size_t>(u*(dv+1)+v)]; }
    std::array<I,N>& at(int u,int v) { return c[static_cast<std::size_t>(u*(dv+1)+v)]; }
};

template<int N>
std::pair<Net<N>,Net<N>> split(const Net<N>& net, bool along_u, const I& t) {
    Net<N> a=net,b=net;
    const int degree=along_u?net.du:net.dv;
    const int lines=along_u?net.dv+1:net.du+1;
    const I one(1.0);
    for(int line=0;line<lines;++line) {
        std::vector<std::array<I,N>> row(static_cast<std::size_t>(degree+1));
        for(int j=0;j<=degree;++j) row[static_cast<std::size_t>(j)]=along_u?net.at(j,line):net.at(line,j);
        for(int step=0;step<=degree;++step) {
            if(along_u) { a.at(step,line)=row.front(); b.at(degree-step,line)=row[static_cast<std::size_t>(degree-step)]; }
            else { a.at(line,step)=row.front(); b.at(line,degree-step)=row[static_cast<std::size_t>(degree-step)]; }
            for(int j=0;j<degree-step;++j) for(int k=0;k<N;++k)
                row[static_cast<std::size_t>(j)][k]=(one-t)*row[static_cast<std::size_t>(j)][k]+t*row[static_cast<std::size_t>(j+1)][k];
        }
    }
    return {std::move(a),std::move(b)};
}

template<int N>
Net<N> restrict_net(const Net<N>& original,const Box& full,const Box& target) {
    Net<N> net=original;
    if(target.u1<full.u1) net=split(net,true,fraction(target.u1,full.u0,full.u1)).first;
    if(target.u0>full.u0) net=split(net,true,fraction(target.u0,full.u0,target.u1)).second;
    if(target.v1<full.v1) net=split(net,false,fraction(target.v1,full.v0,full.v1)).first;
    if(target.v0>full.v0) net=split(net,false,fraction(target.v0,full.v0,target.v1)).second;
    return net;
}

template<int N>
std::array<I,N> evaluate(const Net<N>& net,const I& u,const I& v) {
    const auto uleft=split(net,true,u).first;
    const auto vleft=split(uleft,false,v).first;
    return vleft.at(net.du,net.dv);
}
template<int N>
std::array<I,N> evaluate_at(const Net<N>& net,const Box& box,double u,double v) {
    return evaluate(net,fraction(u,box.u0,box.u1),fraction(v,box.v0,box.v1));
}
template<int N>
std::array<I,N> hull(const Net<N>& net) {
    if (net.c.empty()) throw Stop{PathStatus3D::Unresolved,"EmptyBernsteinNet"};
    std::array<I,N> h;
    for(int k=0;k<N;++k) {
        double a=std::numeric_limits<double>::infinity(),b=-a;
        for(const auto& c:net.c) {
            if(!c[k].finite()) throw Stop{PathStatus3D::Unresolved,"NonFiniteBernsteinInterval"};
            a=std::min(a,c[k].lo); b=std::max(b,c[k].hi);
        }
        h[k]=I(a,b);
    }
    return h;
}
template<int N>
Net<N> derivative(const Net<N>& net,const Box& box,bool along_u) {
    Net<N> r;
    const int degree=along_u?net.du:net.dv;
    r.du=along_u?std::max(0,net.du-1):net.du;
    r.dv=along_u?net.dv:std::max(0,net.dv-1);
    r.c.resize(static_cast<std::size_t>((r.du+1)*(r.dv+1)));
    if(degree==0) { for(auto& c:r.c) for(I& v:c) v=I(0.0); return r; }
    const I factor(Q(degree)/(Q(along_u?box.u1:box.v1)-Q(along_u?box.u0:box.v0)));
    for(int u=0;u<=r.du;++u) for(int v=0;v<=r.dv;++v) for(int k=0;k<N;++k)
        r.at(u,v)[k]=factor*((along_u?net.at(u+1,v):net.at(u,v+1))[k]-net.at(u,v)[k]);
    return r;
}

struct Line {
    std::array<Q,3> p,D;
    ExactNativeHomogeneousPoint3D q;
    int dominant=0;
    std::array<int,2> transverse{{1,2}};
};
struct Element {
    const ExactNativeBezierElement3D* exact=nullptr;
    Net<4> h;
    Net<2> g;
    Box full;
    std::size_t id=0;
};
Element make_element(const ExactNativeBezierElement3D& exact,std::size_t id,const Line& line) {
    Element e; e.exact=&exact; e.full=box_of(exact); e.id=id;
    e.h.du=e.g.du=exact.degree_u; e.h.dv=e.g.dv=exact.degree_v;
    e.h.c.resize(exact.controls.size()); e.g.c.resize(exact.controls.size());
    const Q hscale=exact.controls.front()[3];
    std::vector<std::array<Q,2>> gexact(exact.controls.size());
    std::array<Q,2> gscale{{Q(0),Q(0)}};
    for(std::size_t a=0;a<exact.controls.size();++a) {
        const auto& c=exact.controls[a];
        for(int k=0;k<4;++k) e.h.c[a][k]=I(c[k]/hscale);
        for(int j=0;j<2;++j) {
            const int k=line.dominant,l=line.transverse[j];
            gexact[a][j]=line.D[k]*(c[l]-line.p[l]*c[3])-line.D[l]*(c[k]-line.p[k]*c[3]);
            const Q magnitude=gexact[a][j]<Q(0)?-gexact[a][j]:gexact[a][j];
            if(magnitude>gscale[j]) gscale[j]=magnitude;
        }
    }
    for(std::size_t a=0;a<exact.controls.size();++a) for(int j=0;j<2;++j)
        e.g.c[a][j]=I(gscale[j]==Q(0)?Q(0):gexact[a][j]/gscale[j]);
    return e;
}
I t_range(const std::array<I,4>& h,const Line& line) {
    if(!(h[3].lo>0.0)) throw Stop{PathStatus3D::Unresolved,"NonPositiveWeightBound"};
    const int k=line.dominant;
    return I(line.q[3])*(h[k]-I(line.p[k])*h[3])/(h[3]*I(line.D[k]));
}

struct Linearization {
    Eigen::Matrix2d inverse=Eigen::Matrix2d::Zero();
    std::array<std::array<I,2>,2> error;
    double contraction=std::numeric_limits<double>::infinity();
    bool valid=false;
    bool unique=false;
};
Linearization linearization(const Net<2>& net,const Box& box) {
    Linearization r;
    const auto du=hull(derivative(net,box,true)),dv=hull(derivative(net,box,false));
    const std::array<std::array<I,2>,2> jac{{{{du[0],dv[0]}},{{du[1],dv[1]}}}};
    Eigen::Matrix2d midpoint;
    for(int i=0;i<2;++i) for(int j=0;j<2;++j) {
        if(!jac[i][j].finite()) return r;
        midpoint(i,j)=jac[i][j].mid();
    }
    Eigen::FullPivLU<Eigen::Matrix2d> lu(midpoint);
    if(lu.rank()!=2) return r;
    r.inverse=lu.solve(Eigen::Matrix2d::Identity());
    if(!r.inverse.allFinite()) return r;
    r.contraction=0.0;
    for(int i=0;i<2;++i) {
        double row=0.0;
        for(int j=0;j<2;++j) {
            I m(i==j?1.0:0.0);
            for(int k=0;k<2;++k) m=m-I(r.inverse(i,k))*jac[k][j];
            if(!m.finite()) return Linearization{};
            r.error[i][j]=m; row=up(row+m.magnitude());
        }
        r.contraction=std::max(r.contraction,row);
    }
    r.valid=std::isfinite(r.contraction);
    r.unique=r.valid && r.contraction<1.0;
    return r;
}

struct Work {
    const PathCertificationBudget3D& budget;
    PathCertificationDiagnostics3D& diag;
    std::uint64_t nodes=0,proof=0;
    void node(unsigned depth) {
        if(nodes>=budget.max_subdivision_nodes || depth>budget.max_subdivision_depth)
            throw Stop{PathStatus3D::BudgetExceeded,"SubdivisionBudgetExceeded"};
        ++nodes; diag.maximum_depth=std::max(diag.maximum_depth,depth);
    }
    void newton() {
        if(diag.newton_steps>=budget.max_newton_steps)
            throw Stop{PathStatus3D::BudgetExceeded,"NewtonBudgetExceeded"};
        ++diag.newton_steps;
    }
    void relation() {
        if(diag.relation_refinements>=budget.max_relation_refinements)
            throw Stop{PathStatus3D::BudgetExceeded,"RelationBudgetExceeded"};
        ++diag.relation_refinements;
    }
};

std::optional<Eigen::Vector2d> numerical_candidate(const Element& e,const Box& box,Work& work) {
    Eigen::Vector2d x(0.5*box.u0+0.5*box.u1,0.5*box.v0+0.5*box.v1);
    const auto du=derivative(e.g,e.full,true),dv=derivative(e.g,e.full,false);
    for(int iteration=0;iteration<20;++iteration) {
        work.newton();
        const auto f=evaluate_at(e.g,e.full,x.x(),x.y());
        const auto a=evaluate_at(du,e.full,x.x(),x.y()),b=evaluate_at(dv,e.full,x.x(),x.y());
        for(int k=0;k<2;++k) if(!f[k].finite() || !a[k].finite() || !b[k].finite())
            throw Stop{PathStatus3D::Unresolved,"NonFiniteNumericalCandidateInterval"};
        Eigen::Matrix2d J; J<<a[0].mid(),b[0].mid(),a[1].mid(),b[1].mid();
        if(!J.allFinite()) throw Stop{PathStatus3D::Unresolved,"NonFiniteNumericalCandidateJacobian"};
        Eigen::FullPivLU<Eigen::Matrix2d> lu(J);
        if(lu.rank()!=2) return std::nullopt;
        const Eigen::Vector2d step=lu.solve(Eigen::Vector2d(-f[0].mid(),-f[1].mid()));
        if(!step.allFinite()) return std::nullopt;
        const Eigen::Vector2d next(std::clamp(x.x()+step.x(),e.full.u0,e.full.u1),std::clamp(x.y()+step.y(),e.full.v0,e.full.v1));
        if((next-x).lpNorm<Eigen::Infinity>()<=16.0*std::numeric_limits<double>::epsilon()*std::max(1.0,x.lpNorm<Eigen::Infinity>())) {
            x=next; ++work.diag.numerical_candidates; return x;
        }
        x=next;
    }
    ++work.diag.numerical_candidates;
    return x;
}

struct KrawczykResult { bool exists=false; bool excluded=false; Box enclosure{}; double contraction=0.0; };
KrawczykResult krawczyk(const Element& e,const Box& box,double u,double v) {
    KrawczykResult r;
    if(!(box.u0<box.u1 && box.v0<box.v1)) return r;
    const auto g=restrict_net(e.g,e.full,box);
    const auto lin=linearization(g,box);
    if(!lin.valid) return r;
    const auto f=evaluate_at(e.g,e.full,u,v);
    const std::array<I,2> delta{{I(box.u0,box.u1)-I(u),I(box.v0,box.v1)-I(v)}};
    std::array<I,2> K;
    for(int i=0;i<2;++i) {
        I z(i==0?u:v);
        for(int j=0;j<2;++j) z=z-I(lin.inverse(i,j))*f[j]+lin.error[i][j]*delta[j];
        if(!z.finite()) return r;
        K[i]=z;
    }
    // Every zero in X belongs to K(X), even if the contraction bound is >=1.
    // This exclusion is essential for near-parallel line equations: testing
    // their separate Bernstein hulls loses their strong correlation.
    r.excluded=K[0].hi<box.u0 || K[0].lo>box.u1 || K[1].hi<box.v0 || K[1].lo>box.v1;
    r.exists=lin.unique && K[0].lo>box.u0 && K[0].hi<box.u1 && K[1].lo>box.v0 && K[1].hi<box.v1;
    r.enclosure={K[0].lo,K[0].hi,K[1].lo,K[1].hi};
    r.contraction=lin.contraction;
    return r;
}

std::array<I,3> cross(const std::array<I,3>& a,const std::array<I,3>& b) {
    return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};
}
struct GeometryBounds {
    std::array<I,3> point,normal;
    I oriented_dot;
    I t;
};
GeometryBounds geometry_bounds(const Element& e,const Box& box,const Line& line,bool known,double u,double v) {
    const auto hu=derivative(e.h,e.full,true),hv=derivative(e.h,e.full,false);
    std::array<I,4> h,du,dv;
    if(known) {
        h=evaluate_at(e.h,e.full,u,v); du=evaluate_at(hu,e.full,u,v); dv=evaluate_at(hv,e.full,u,v);
        const auto exact_h=evaluate_native_bezier_homogeneous(*e.exact,u,v);
        for(int k=0;k<4;++k) h[k]=I(exact_h[k]/e.exact->controls.front()[3]);
    } else {
        h=hull(restrict_net(e.h,e.full,box));
        du=hull(restrict_net(hu,e.full,box)); dv=hull(restrict_net(hv,e.full,box));
    }
    for(int k=0;k<4;++k) if(!h[k].finite() || !du[k].finite() || !dv[k].finite())
        throw Stop{PathStatus3D::Unresolved,"NonFiniteRootGeometryInterval"};
    if(!(h[3].lo>0.0)) throw Stop{PathStatus3D::Unresolved,"NonPositiveRootWeight"};
    GeometryBounds r;
    std::array<I,3> a,b;
    for(int k=0;k<3;++k) {
        r.point[k]=known?I(line.q[k]/line.q[3]):h[k]/h[3];
        a[k]=du[k]*h[3]-h[k]*du[3];
        b[k]=dv[k]*h[3]-h[k]*dv[3];
    }
    const auto n=cross(a,b);
    I norm2(0.0); r.oriented_dot=I(0.0);
    for(int k=0;k<3;++k) { norm2=norm2+square(n[k]); r.oriented_dot=r.oriented_dot+n[k]*I(line.D[k]); }
    if(!(norm2.lo>0.0)) throw Stop{PathStatus3D::Unresolved,"RootNormalNotRegular"};
    const I norm=square_root(norm2);
    for(int k=0;k<3;++k) r.normal[k]=n[k]/norm;
    r.t=known?I(1.0):t_range(h,line);
    if(!r.oriented_dot.finite() || !r.t.finite())
        throw Stop{PathStatus3D::Unresolved,"NonFiniteRootTransitionInterval"};
    for(int k=0;k<3;++k) if(!r.point[k].finite() || !r.normal[k].finite())
        throw Stop{PathStatus3D::Unresolved,"NonFiniteRootRepresentativeInterval"};
    return r;
}

CertifiedNativePathRoot3D make_root(const Element& e,const Box& unique,const Box& enclosure,
    double contraction,bool known,const NativeSurfaceEndpoint3D& endpoint,const Line& line,
    const NativeEndpointQueryOptions3D& options,Work& work) {
    CertifiedNativePathRoot3D r;
    const double u=known?endpoint.u:0.5*enclosure.u0+0.5*enclosure.u1;
    const double v=known?endpoint.v:0.5*enclosure.v0+0.5*enclosure.v1;
    const auto gb=geometry_bounds(e,enclosure,line,known,u,v);
    if(gb.oriented_dot.zero()) {
        if(interval_precision<work.budget.max_precision_bits) throw NeedPrecision{};
        throw Stop{PathStatus3D::Unresolved,"TransversalityNotProved"};
    }
    r.representative.patch_index=e.exact->patch_index;
    r.representative.component=e.exact->component;
    r.representative.u=u; r.representative.v=v;
    for(int k=0;k<3;++k) {
        r.representative.point[k]=gb.point[k].mid(); r.representative.normal[k]=gb.normal[k].mid();
        r.accuracy.point_error_bound=std::max(r.accuracy.point_error_bound,
            up(std::max(r.representative.point[k]-gb.point[k].lo,gb.point[k].hi-r.representative.point[k])));
        r.accuracy.normal_error_bound=std::max(r.accuracy.normal_error_bound,
            up(std::max(r.representative.normal[k]-gb.normal[k].lo,gb.normal[k].hi-r.representative.normal[k])));
    }
    // Use the native surface at the representative parameters. The exact
    // enclosure also bounds the error of this rounded representative.
    const auto hp=evaluate_native_bezier_homogeneous(*e.exact,u,v);
    for(int k=0;k<3;++k) r.representative.point[k]=CGAL::to_double(hp[k]/hp[3]);
    for(int k=0;k<3;++k) r.accuracy.point_error_bound=std::max(r.accuracy.point_error_bound,
        up(std::max(std::abs(r.representative.point[k]-gb.point[k].lo),std::abs(gb.point[k].hi-r.representative.point[k]))));
    const I sqrt_three=square_root(I(3.0));
    r.accuracy.point_error_bound=(sqrt_three*I(r.accuracy.point_error_bound)).hi;
    r.accuracy.normal_error_bound=(sqrt_three*I(r.accuracy.normal_error_bound)).hi;
    r.accuracy.parameter_error_bound=known?0.0:up(std::max({u-enclosure.u0,enclosure.u1-u,v-enclosure.v0,enclosure.v1-v}));
    r.accuracy.certified=r.accuracy.point_error_bound<=options.point_tolerance
        && r.accuracy.parameter_error_bound<=options.parameter_tolerance
        && r.accuracy.normal_error_bound<=options.normal_tolerance;
    if(!r.accuracy.certified) throw NeedPrecision{};
    r.certificate.kind=known?RootProofKind3D::KnownEndpointUnique:RootProofKind3D::ExistsUnique;
    r.certificate.proof_id=++work.proof; r.certificate.patch_index=e.exact->patch_index;
    r.certificate.source_element=e.id; r.certificate.uniqueness_box=public_box(unique);
    r.certificate.root_enclosure=public_box(enclosure);
    r.certificate.t_interval={gb.t.lo,gb.t.hi}; r.certificate.contraction_bound=contraction;
    r.certificate.precision_bits=interval_precision; r.certificate.existence_proved=true; r.certificate.uniqueness_proved=true;
    r.representative.edge_parameter=gb.t.mid();
    Eigen::Vector3d direction;
    for(int k=0;k<3;++k) direction[k]=CGAL::to_double(line.D[k]/line.q[3]);
    const double direction_norm=direction.stableNorm();
    if(!direction.allFinite() || !std::isfinite(direction_norm) || !(direction_norm>0.0))
        throw Stop{PathStatus3D::Unresolved,"NonFiniteRepresentativeDirection"};
    const Eigen::Vector3d unit_direction=direction/direction_norm;
    r.representative.residual=(r.representative.point-Eigen::Vector3d(CGAL::to_double(line.p[0]),CGAL::to_double(line.p[1]),CGAL::to_double(line.p[2]))-r.representative.edge_parameter*direction).stableNorm();
    r.representative.transversality=std::abs(r.representative.normal.dot(unit_direction));
    if(!r.representative.point.allFinite() || !r.representative.normal.allFinite()
        || !std::isfinite(r.representative.residual) || !std::isfinite(r.representative.transversality))
        throw Stop{PathStatus3D::Unresolved,"NonFiniteRootRepresentative"};
    r.transition.oriented_dot_lower=gb.oriented_dot.lo; r.transition.oriented_dot_upper=gb.oriented_dot.hi;
    r.transition.sign=gb.oriented_dot.hi<0.0?+1:-1;
    r.transition.incoming_inside=r.transition.sign==+1?0:1;
    r.transition.outgoing_inside=1-r.transition.incoming_inside; r.transition.certified=true;
    if(known) ++work.diag.known_endpoint_proofs; else ++work.diag.ordinary_root_proofs;
    return r;
}

ExactNativeHomogeneousPoint3D exact_derivative_at(
    const ExactNativeBezierElement3D& source,double u,double v,bool along_u) {
    auto derivative_element=source;
    const int degree=along_u?source.degree_u:source.degree_v;
    if(degree==0) return {{Q(0),Q(0),Q(0),Q(0)}};
    if(along_u) --derivative_element.degree_u; else --derivative_element.degree_v;
    derivative_element.controls.resize(static_cast<std::size_t>((derivative_element.degree_u+1)*(derivative_element.degree_v+1)));
    const Q scale=Q(degree)/(Q(along_u?source.parameter_u1:source.parameter_v1)-Q(along_u?source.parameter_u0:source.parameter_v0));
    for(int i=0;i<=derivative_element.degree_u;++i) for(int j=0;j<=derivative_element.degree_v;++j) {
        const auto& a=source.controls[static_cast<std::size_t>(i*(source.degree_v+1)+j)];
        const auto& b=source.controls[static_cast<std::size_t>((i+(along_u?1:0))*(source.degree_v+1)+j+(along_u?0:1))];
        auto& c=derivative_element.controls[static_cast<std::size_t>(i*(derivative_element.degree_v+1)+j)];
        for(int k=0;k<4;++k) c[k]=scale*(b[k]-a[k]);
    }
    // Derivative nets need not have positive fourth coordinates. Do not pass
    // them to the public positive-weight rational-geometry evaluator.
    const Q tu=(Q(u)-Q(source.parameter_u0))/(Q(source.parameter_u1)-Q(source.parameter_u0));
    const Q tv=(Q(v)-Q(source.parameter_v0))/(Q(source.parameter_v1)-Q(source.parameter_v0));
    const auto polynomial_value=[](std::vector<ExactNativeHomogeneousPoint3D> row,const Q& t) {
        for(std::size_t n=row.size();n>1;--n) for(std::size_t j=0;j+1<n;++j) for(int k=0;k<4;++k)
            row[j][k]=(Q(1)-t)*row[j][k]+t*row[j+1][k];
        return row.front();
    };
    std::vector<ExactNativeHomogeneousPoint3D> rows;
    for(int i=0;i<=derivative_element.degree_u;++i) {
        std::vector<ExactNativeHomogeneousPoint3D> row;
        for(int j=0;j<=derivative_element.degree_v;++j)
            row.push_back(derivative_element.controls[static_cast<std::size_t>(i*(derivative_element.degree_v+1)+j)]);
        rows.push_back(polynomial_value(std::move(row),tv));
    }
    return polynomial_value(std::move(rows),tu);
}
std::array<Q,3> exact_normal_at(const ExactNativeBezierElement3D& source,double u,double v) {
    const auto h=evaluate_native_bezier_homogeneous(source,u,v);
    const auto hu=exact_derivative_at(source,u,v,true),hv=exact_derivative_at(source,u,v,false);
    std::array<Q,3> a,b;
    for(int k=0;k<3;++k) { a[k]=hu[k]*h[3]-h[k]*hu[3]; b[k]=hv[k]*h[3]-h[k]*hv[3]; }
    return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};
}

struct Cell { std::size_t element; Box box; Net<4> h; Net<2> g; unsigned depth; };

std::optional<CertifiedNativePathRoot3D> known_endpoint_attempt(
    const Element& e,const Cell& cell,const NativeSurfaceEndpoint3D& endpoint,
    const Line& line,const NativeEndpointQueryOptions3D& options,Work& work) {
    if(e.exact->patch_index!=endpoint.patch_index || !contains(e.full,endpoint.u,endpoint.v)) return std::nullopt;
    // A common box may cover this cell even when the known root is just
    // outside the cell. Its at-most-one proof still covers every possible
    // root in the cell; the globally valid endpoint is retained exactly once.
    const Box point{endpoint.u,endpoint.u,endpoint.v,endpoint.v};
    const Box common=united(cell.box,point);
    if(!(common.u0<common.u1 && common.v0<common.v1)) return std::nullopt;
    const auto lin=linearization(restrict_net(e.g,e.full,common),common);
    if(!lin.unique) return std::nullopt;
    return make_root(e,common,point,lin.contraction,true,endpoint,line,options,work);
}

std::optional<CertifiedNativePathRoot3D> ordinary_attempt(
    const Element& e,const Cell& cell,const NativeSurfaceEndpoint3D& endpoint,
    const Line& line,const NativeEndpointQueryOptions3D& options,Work& work) {
    const auto candidate=numerical_candidate(e,cell.box,work);
    if(!candidate) return std::nullopt;
    const double u=candidate->x(),v=candidate->y();
    const auto hp=evaluate_native_bezier_homogeneous(*e.exact,u,v);
    bool at_start=true;
    for(int k=0;k<3;++k) at_start=at_start && hp[k]==line.p[k]*hp[3];
    if(at_start) throw Stop{PathStatus3D::UnsupportedEndpoint,"StartOnBoundary"};
    const double scale=std::max({1.0,e.full.u1-e.full.u0,e.full.v1-e.full.v0});
    const double radius=std::max(64.0*options.parameter_tolerance,256.0*std::numeric_limits<double>::epsilon()*scale);
    for(int expansion=0;expansion<4;++expansion) {
        const double r=std::ldexp(radius,expansion*3);
        const Box isolator{std::max(e.full.u0,u-r),std::min(e.full.u1,u+r),std::max(e.full.v0,v-r),std::min(e.full.v1,v+r)};
        auto proof=krawczyk(e,isolator,u,v);
        if(!proof.exists) continue;
        const Box common=united(cell.box,isolator);
        const auto global=linearization(restrict_net(e.g,e.full,common),common);
        if(!global.unique) continue;
        // Existence in the isolator and at-most-one in the common box cover
        // ALL of cell.box, not merely a small neighborhood of the candidate.
        Box enclosed=proof.enclosure;
        for(int refinement=0;refinement<6;++refinement) {
            if(!(enclosed.u0<enclosed.u1 && enclosed.v0<enclosed.v1)) break;
            const double cu=0.5*enclosed.u0+0.5*enclosed.u1,cv=0.5*enclosed.v0+0.5*enclosed.v1;
            auto improved=krawczyk(e,enclosed,cu,cv);
            if(!improved.exists) break;
            enclosed=improved.enclosure;
        }
        const I T=t_range(hull(restrict_net(e.h,e.full,enclosed)),line);
        if(T.hi<0.0 || T.lo>1.0) {
            auto outside=make_root(e,common,enclosed,global.contraction,false,endpoint,line,options,work);
            return outside;
        }
        if(!(T.lo>0.0 && T.hi<1.0)) {
            if(interval_precision<work.budget.max_precision_bits) throw NeedPrecision{};
            return std::nullopt;
        }
        return make_root(e,common,enclosed,global.contraction,false,endpoint,line,options,work);
    }
    return std::nullopt;
}

RootRelation3D root_relation(const CertifiedNativePathRoot3D& a,
    const CertifiedNativePathRoot3D& b,const std::vector<Element>& elements,Work& work) {
    const auto& A=a.certificate; const auto& B=b.certificate;
    if(A.kind==RootProofKind3D::KnownEndpointUnique && B.kind==RootProofKind3D::KnownEndpointUnique)
        return RootRelation3D::SameRoot;
    if(A.t_interval.upper<B.t_interval.lower || B.t_interval.upper<A.t_interval.lower)
        return RootRelation3D::DistinctRoots;
    if(A.patch_index!=B.patch_index) return RootRelation3D::Unresolved;
    if(contains(private_box(A.uniqueness_box),private_box(B.root_enclosure))
        || contains(private_box(B.uniqueness_box),private_box(A.root_enclosure)))
        return RootRelation3D::SameRoot;
    work.relation();
    if(A.source_element!=B.source_element) return RootRelation3D::Unresolved;
    const auto found=std::find_if(elements.begin(),elements.end(),[&](const Element& e){return e.id==A.source_element;});
    if(found==elements.end()) return RootRelation3D::Unresolved;
    const Box common=united(private_box(A.root_enclosure),private_box(B.root_enclosure));
    if(!(common.u0<common.u1 && common.v0<common.v1)) return RootRelation3D::Unresolved;
    if(linearization(restrict_net(found->g,found->full,common),common).unique)
        return RootRelation3D::SameRoot;
    return RootRelation3D::Unresolved;
}

bool split_along_u(const Cell& cell) {
    double uvariation=0.0,vvariation=0.0;
    for(int i=0;i<cell.g.du;++i) for(int j=0;j<=cell.g.dv;++j) for(int k=0;k<2;++k)
        uvariation=std::max(uvariation,(cell.g.at(i+1,j)[k]-cell.g.at(i,j)[k]).magnitude());
    for(int i=0;i<=cell.g.du;++i) for(int j=0;j<cell.g.dv;++j) for(int k=0;k<2;++k)
        vvariation=std::max(vvariation,(cell.g.at(i,j+1)[k]-cell.g.at(i,j)[k]).magnitude());
    return uvariation>=vvariation;
}

void run_pass(const NurbsSurfaceModel3D& model,const std::vector<ExactNativeBezierElement3D>& exact,
    const std::vector<std::size_t>& candidates,const NativeSurfaceEndpoint3D& endpoint,
    const Line& line,const NativeEndpointQueryOptions3D& options,Work& work,NativeEndpointPathResult3D& result) {
    (void)model;
    result.open_roots.clear(); result.endpoint.reset(); result.post_endpoint_roots.clear();
    result.coverage=PathCoverage3D{}; result.coverage.candidate_regions=candidates.size();
    std::vector<Element> elements; elements.reserve(candidates.size());
    for(std::size_t id:candidates) elements.push_back(make_element(exact[id],id,line));
    std::vector<Cell> stack;
    for(std::size_t i=elements.size();i>0;--i) {
        const auto& e=elements[i-1]; stack.push_back({i-1,e.full,e.h,e.g,0});
    }
    std::vector<CertifiedNativePathRoot3D> roots;
    while(!stack.empty()) {
        Cell cell=std::move(stack.back()); stack.pop_back(); work.node(cell.depth);
        ++result.coverage.visited_regions;
        const auto g=hull(cell.g);
        const auto h=hull(cell.h);
        const I T=t_range(h,line);
        if(!g[0].zero() || !g[1].zero() || T.hi<0.0 || T.lo>1.0) {
            ++result.coverage.excluded_regions; continue;
        }
        const auto& e=elements[cell.element];
        auto root=known_endpoint_attempt(e,cell,endpoint,line,options,work);
        if(!root) {
            const auto box_test=krawczyk(e,cell.box,0.5*cell.box.u0+0.5*cell.box.u1,0.5*cell.box.v0+0.5*cell.box.v1);
            if(box_test.excluded) { ++result.coverage.excluded_regions; continue; }
        }
        if(!root) root=ordinary_attempt(e,cell,endpoint,line,options,work);
        if(root) {
            ++result.coverage.root_regions;
            if(root->certificate.t_interval.upper>=0.0 && root->certificate.t_interval.lower<=1.0)
                roots.push_back(std::move(*root));
            continue;
        }
        if(cell.depth>=20 && interval_precision<128 && options.enable_precision_escalation) throw NeedPrecision{};
        if(cell.depth>=work.budget.max_subdivision_depth)
            throw Stop{PathStatus3D::BudgetExceeded,"SubdivisionDepthExceeded"};
        bool along=split_along_u(cell);
        double mid=along?0.5*cell.box.u0+0.5*cell.box.u1:0.5*cell.box.v0+0.5*cell.box.v1;
        const auto valid_mid=[&](bool u,double m){return u?(cell.box.u0<m&&m<cell.box.u1):(cell.box.v0<m&&m<cell.box.v1);};
        if(!valid_mid(along,mid)) { along=!along; mid=along?0.5*cell.box.u0+0.5*cell.box.u1:0.5*cell.box.v0+0.5*cell.box.v1; }
        if(!valid_mid(along,mid)) throw Stop{PathStatus3D::Unresolved,"NoRepresentableSubdivision"};
        const I f=along?fraction(mid,cell.box.u0,cell.box.u1):fraction(mid,cell.box.v0,cell.box.v1);
        auto hs=split(cell.h,along,f); auto gs=split(cell.g,along,f);
        Box a=cell.box,b=cell.box;
        if(along) { a.u1=mid; b.u0=mid; } else { a.v1=mid; b.v0=mid; }
        ++result.coverage.split_regions;
        stack.push_back({cell.element,b,std::move(hs.second),std::move(gs.second),cell.depth+1});
        stack.push_back({cell.element,a,std::move(hs.first),std::move(gs.first),cell.depth+1});
    }
    result.coverage.complete=true;
    std::sort(roots.begin(),roots.end(),[](const auto& a,const auto& b){
        if(a.certificate.t_interval.lower!=b.certificate.t_interval.lower) return a.certificate.t_interval.lower<b.certificate.t_interval.lower;
        if(a.representative.patch_index!=b.representative.patch_index) return a.representative.patch_index<b.representative.patch_index;
        return a.certificate.proof_id<b.certificate.proof_id;
    });
    std::vector<CertifiedNativePathRoot3D> canonical;
    for(auto& root:roots) {
        bool merged=false;
        for(auto& prior:canonical) {
            const auto relation=root_relation(prior,root,elements,work);
            if(relation==RootRelation3D::Unresolved) throw Stop{PathStatus3D::Unresolved,"RootIdentityNotProved"};
            if(relation==RootRelation3D::SameRoot) {
                if(prior.transition.sign!=root.transition.sign) throw Stop{PathStatus3D::Unresolved,"OwnerTransitionConflict"};
                if(root.certificate.kind==RootProofKind3D::KnownEndpointUnique) prior=std::move(root);
                ++work.diag.same_root_merges; merged=true; break;
            }
        }
        if(!merged) canonical.push_back(std::move(root));
    }
    std::sort(canonical.begin(),canonical.end(),[](const auto& a,const auto& b){return a.certificate.t_interval.lower<b.certificate.t_interval.lower;});
    double previous=-std::numeric_limits<double>::infinity(); std::uint64_t event_id=0;
    for(auto& root:canonical) {
        auto& c=root.certificate;
        if(!(c.t_interval.lower>previous)) throw Stop{PathStatus3D::Unresolved,"RootOrderNotProved"};
        previous=c.t_interval.upper; c.event_id=++event_id;
        if(c.kind==RootProofKind3D::KnownEndpointUnique) {
            if(result.endpoint) throw Stop{PathStatus3D::Unresolved,"MultipleNativeEndpoints"};
            result.endpoint=std::move(root);
        } else {
            if(!(0.0<c.t_interval.lower && c.t_interval.upper<1.0)) throw Stop{PathStatus3D::Unresolved,"OpenRootRangeNotProved"};
            result.open_roots.push_back(std::move(root));
        }
    }
    if(!result.endpoint) throw Stop{PathStatus3D::Unresolved,"MissingEndpointProof"};
    result.coverage.order_certified=true;
    int state=result.endpoint->transition.incoming_inside;
    for(auto i=result.open_roots.rbegin();i!=result.open_roots.rend();++i) {
        if(i->transition.outgoing_inside!=state) throw Stop{PathStatus3D::Unresolved,"TransitionChainMismatch"};
        state=i->transition.incoming_inside;
    }
    if(options.expected_start_inside && state!=static_cast<int>(*options.expected_start_inside))
        throw Stop{PathStatus3D::Unresolved,"StartStateMismatch"};
    result.coverage.certified_start_inside=state;
    result.coverage.transitions_certified=true; result.coverage.representatives_certified=true;
    result.status=PathStatus3D::Certified;
}

std::uint64_t model_fingerprint(const NurbsSurfaceModel3D& model) {
    std::uint64_t hash=1469598103934665603ULL;
    const auto add=[&](double x){std::uint64_t bits=0;std::memcpy(&bits,&x,sizeof(bits));for(int b=0;b<8;++b){hash^=(bits>>(8*b))&255U;hash*=1099511628211ULL;}};
    add(model.num_patches());
    for(int p=0;p<model.num_patches();++p) {
        const auto& patch=model.patch(p); add(p); add(model.patch_component(p));
        for(double knot:patch.basis_u().knots()) add(knot);
        for(double knot:patch.basis_v().knots()) add(knot);
        for(const auto& row:patch.control_net()) for(const auto& c:row) for(int k=0;k<3;++k) add(c[k]);
        for(const auto& row:patch.weights()) for(double w:row) add(w);
    }
    for(const auto& c:model.connections()) {add(c.first.patch);add(static_cast<int>(c.first.edge));add(c.first.begin);add(c.first.end);add(c.second.patch);add(static_cast<int>(c.second.edge));add(c.second.begin);add(c.second.end);add(c.reversed);add(c.g1);}
    return hash;
}
std::string diagnostic_dump(const NurbsSurfaceModel3D& model,const Eigen::Vector3d& start,
    const NativeSurfaceEndpoint3D& endpoint,const NativeEndpointQueryOptions3D& options,
    const PathCertificationBudget3D& budget,const NativeEndpointPathResult3D& result) {
    std::ostringstream out;
    out<<"native_endpoint_path_v1 model_fnv="<<std::hex<<model_fingerprint(model)<<std::dec
       <<" status="<<native_endpoint_path_status_name_3d(result.status)<<" reason="<<result.diagnostics.reason<<'\n';
    out<<std::hexfloat<<"start="<<start.x()<<','<<start.y()<<','<<start.z()<<" endpoint="<<endpoint.patch_index<<','<<endpoint.u<<','<<endpoint.v
       <<" tolerances="<<options.point_tolerance<<','<<options.parameter_tolerance<<','<<options.normal_tolerance<<std::defaultfloat
       <<" budget="<<budget.max_candidate_regions<<','<<budget.max_subdivision_nodes<<','<<budget.max_subdivision_depth<<','<<budget.max_newton_steps<<','<<budget.max_closest_point_evaluations<<','<<budget.max_relation_refinements<<','<<budget.max_precision_escalations<<','<<budget.max_precision_bits<<'\n'
       <<"coverage="<<result.coverage.candidate_regions<<','<<result.coverage.visited_regions<<','<<result.coverage.excluded_regions<<','<<result.coverage.root_regions<<','<<result.coverage.split_regions
       <<" gates="<<result.coverage.complete<<','<<result.coverage.order_certified<<','<<result.coverage.transitions_certified<<','<<result.coverage.representatives_certified
       <<" precision="<<result.diagnostics.highest_precision_bits<<" newton="<<result.diagnostics.newton_steps<<'\n';
    const auto print_root=[&](const CertifiedNativePathRoot3D& root){
        const auto& c=root.certificate; const auto& r=root.representative;
        out<<std::setprecision(17)<<"root proof="<<c.proof_id<<" event="<<c.event_id<<" kind="<<static_cast<int>(c.kind)
           <<" patch="<<r.patch_index<<" uv="<<r.u<<','<<r.v<<" T="<<c.t_interval.lower<<','<<c.t_interval.upper
           <<" residual="<<r.residual<<" contraction="<<c.contraction_bound<<" sign="<<root.transition.sign<<'\n';
    };
    for(const auto& r:result.open_roots) print_root(r); if(result.endpoint) print_root(*result.endpoint);
    if(result.status!=PathStatus3D::Certified) {
        // Failure replay contains source bit patterns, not just a rounded q.
        out<<std::hexfloat;
        for(int p=0;p<model.num_patches();++p) {
            const auto& s=model.patch(p); out<<"patch "<<p<<" component "<<model.patch_component(p)<<" ku";
            for(double k:s.basis_u().knots()) out<<' '<<k; out<<" kv"; for(double k:s.basis_v().knots()) out<<' '<<k; out<<'\n';
            for(std::size_t u=0;u<s.control_net().size();++u) for(std::size_t v=0;v<s.control_net()[u].size();++v) {
                const auto& pnt=s.control_net()[u][v]; out<<"cw "<<u<<' '<<v<<' '<<pnt.x()<<' '<<pnt.y()<<' '<<pnt.z()<<' '<<s.weights()[u][v]<<'\n';
            }
            if(out.tellp()>1024*1024) {out<<"DUMP_TRUNCATED_MODEL_FINGERPRINT_REQUIRES_SOURCE\n";break;}
        }
    }
    return out.str();
}

} // namespace

const char* native_endpoint_path_status_name_3d(PathStatus3D status) noexcept {
    switch(status) {
    case PathStatus3D::Certified:return "Certified";
    case PathStatus3D::Unresolved:return "Unresolved";
    case PathStatus3D::BudgetExceeded:return "BudgetExceeded";
    case PathStatus3D::UnsupportedEndpoint:return "UnsupportedEndpoint";
    case PathStatus3D::DegenerateSegment:return "DegenerateSegment";
    case PathStatus3D::InvalidInput:return "InvalidInput";
    }
    return "InvalidStatus";
}

NativeEndpointPathResult3D certify_native_endpoint_path_3d(
    const NurbsSurfaceModel3D& model,const std::vector<ExactNativeBezierElement3D>& exact,
    const Eigen::Vector3d& start,const NativeSurfaceEndpoint3D& endpoint,
    const NativeEndpointQueryOptions3D& options,const PathCertificationBudget3D& budget,
    const std::vector<NurbsAabb3D>* certified_bounds) {
    NativeEndpointPathResult3D result;
    const unsigned saved_precision=interval_precision;
    try {
        if(!start.allFinite() || !std::isfinite(endpoint.u) || !std::isfinite(endpoint.v)
            || endpoint.patch_index<0 || endpoint.patch_index>=model.num_patches() || exact.empty()
            || !(options.point_tolerance>0.0) || !std::isfinite(options.point_tolerance)
            || !(options.parameter_tolerance>0.0) || !std::isfinite(options.parameter_tolerance)
            || !(options.normal_tolerance>0.0) || !std::isfinite(options.normal_tolerance)
            || (options.initial_precision_bits!=53 && options.initial_precision_bits!=128 && options.initial_precision_bits!=256 && options.initial_precision_bits!=512)
            || options.initial_precision_bits>budget.max_precision_bits
            || (certified_bounds && certified_bounds->size()!=exact.size()))
            throw Stop{PathStatus3D::InvalidInput,"InvalidNativeEndpointQuery"};
        if(options.endpoint_extension_ratio!=0.0)
            throw Stop{PathStatus3D::UnsupportedEndpoint,"EndpointExtensionNotImplemented"};
        const auto& endpoint_patch=model.patch(endpoint.patch_index);
        if(endpoint.u<endpoint_patch.domain_start_u() || endpoint.u>endpoint_patch.domain_end_u()
            || endpoint.v<endpoint_patch.domain_start_v() || endpoint.v>endpoint_patch.domain_end_v())
            throw Stop{PathStatus3D::InvalidInput,"EndpointParameterOutsidePatch"};
        if(!(endpoint.u>endpoint_patch.domain_start_u() && endpoint.u<endpoint_patch.domain_end_u()
            && endpoint.v>endpoint_patch.domain_start_v() && endpoint.v<endpoint_patch.domain_end_v()))
            throw Stop{PathStatus3D::UnsupportedEndpoint,"EndpointOnPatchBoundary"};
        std::vector<std::size_t> endpoint_elements;
        for(std::size_t i=0;i<exact.size();++i) if(exact[i].patch_index==endpoint.patch_index && contains(box_of(exact[i]),endpoint.u,endpoint.v)) endpoint_elements.push_back(i);
        if(endpoint_elements.empty()) throw Stop{PathStatus3D::InvalidInput,"EndpointAbsentFromExactGeometry"};
        Line line;
        line.q=evaluate_native_bezier_homogeneous(exact[endpoint_elements.front()],endpoint.u,endpoint.v);
        if(!(line.q[3]>Q(0))) throw Stop{PathStatus3D::InvalidInput,"NonPositiveEndpointWeight"};
        // Exact common positive rescaling keeps the endpoint homogeneous
        // identity, while protecting interval arithmetic from weight scale.
        const Q qw=line.q[3]; for(Q& x:line.q) x/=qw;
        for(int k=0;k<3;++k) { line.p[k]=Q(start[k]); line.D[k]=line.q[k]-line.p[k]; }
        const auto magnitude=[](const Q& x){return x<Q(0)?-x:x;};
        for(int k=1;k<3;++k) if(magnitude(line.D[k])>magnitude(line.D[line.dominant])) line.dominant=k;
        if(line.D[line.dominant]==Q(0)) throw Stop{PathStatus3D::DegenerateSegment,"CoincidentNativeEndpoints"};
        int next=0;for(int k=0;k<3;++k) if(k!=line.dominant) line.transverse[next++]=k;
        std::optional<std::array<Q,3>> first_normal;
        for(std::size_t id:endpoint_elements) {
            const auto q=evaluate_native_bezier_homogeneous(exact[id],endpoint.u,endpoint.v);
            for(int k=0;k<3;++k) if(q[k]!=line.q[k]*q[3]) throw Stop{PathStatus3D::UnsupportedEndpoint,"DiscontinuousNativeEndpoint"};
            const auto normal=exact_normal_at(exact[id],endpoint.u,endpoint.v);
            Q norm(0),dot(0);for(int k=0;k<3;++k){norm+=normal[k]*normal[k];dot+=normal[k]*line.D[k];}
            if(norm==Q(0)) throw Stop{PathStatus3D::UnsupportedEndpoint,"SingularNativeEndpoint"};
            if(dot==Q(0)) throw Stop{PathStatus3D::UnsupportedEndpoint,"TangentNativeEndpoint"};
            if(first_normal) {
                Q agreement(0);for(int k=0;k<3;++k) agreement+=normal[k]*(*first_normal)[k];
                if(!(agreement>Q(0)) || normal[0]*(*first_normal)[1]!=normal[1]*(*first_normal)[0]
                    || normal[0]*(*first_normal)[2]!=normal[2]*(*first_normal)[0]
                    || normal[1]*(*first_normal)[2]!=normal[2]*(*first_normal)[1])
                    throw Stop{PathStatus3D::UnsupportedEndpoint,"EndpointOnNonSmoothNativeKnot"};
            } else first_normal=normal;
        }
        interval_precision=53;
        NurbsAabb3D query;
        for(int k=0;k<3;++k) { const I q(line.q[k]);query.lower[k]=std::min(start[k],q.lo);query.upper[k]=std::max(start[k],q.hi); }
        std::vector<std::size_t> candidates;
        for(std::size_t i=0;i<exact.size();++i) {
            const auto bounds=certified_bounds?(*certified_bounds)[i]:exact[i].bounds();
            bool overlap=true;for(int k=0;k<3;++k) overlap=overlap && bounds.lower[k]<=query.upper[k] && query.lower[k]<=bounds.upper[k];
            if(overlap) candidates.push_back(i);
        }
        if(candidates.size()>budget.max_candidate_regions) throw Stop{PathStatus3D::BudgetExceeded,"CandidateBudgetExceeded"};
        Work work{budget,result.diagnostics};
        unsigned precision=options.initial_precision_bits;
        for(;;) {
#ifndef KFBIM_NATIVE_ENDPOINT_HAS_MPFR
            if(precision>53) throw Stop{PathStatus3D::Unresolved,"PrecisionBackendUnavailable"};
#endif
            interval_precision=precision;
            result.diagnostics.highest_precision_bits=precision;
            try {run_pass(model,exact,candidates,endpoint,line,options,work,result);break;}
            catch(const NeedPrecision&) {
                if(!options.enable_precision_escalation) throw Stop{PathStatus3D::Unresolved,"PrecisionEscalationDisabled"};
                const unsigned higher=precision<128?128:2*precision;
                if(higher>budget.max_precision_bits || result.diagnostics.precision_escalations>=budget.max_precision_escalations)
                    throw Stop{PathStatus3D::BudgetExceeded,"PrecisionBudgetExceeded"};
                ++result.diagnostics.precision_escalations; precision=higher;
            }
        }
    } catch(const Stop& error) {
        result.status=error.status;result.diagnostics.reason=error.reason;result.message=error.reason;
        ++result.coverage.unresolved_regions;
    } catch(const std::exception& error) {
        result.status=PathStatus3D::Unresolved;result.diagnostics.reason="ExactGeometryOrArithmeticFailure";result.message=error.what();
        ++result.coverage.unresolved_regions;
    }
    if(result.status!=PathStatus3D::Certified) result.coverage.complete=false;
    result.dump=diagnostic_dump(model,start,endpoint,options,budget,result);
    interval_precision=saved_precision;
    return result;
}

} // namespace kfbim::geometry3d
