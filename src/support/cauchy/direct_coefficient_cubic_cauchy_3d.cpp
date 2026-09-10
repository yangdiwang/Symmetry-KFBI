#include "src/support/cauchy/direct_coefficient_cubic_cauchy_3d.hpp"
#include "src/geometry/nurbs_basis_derivatives.hpp"

#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfbim::app3d {
namespace {
constexpr std::array<int, 10> du{{0,1,0,2,1,0,3,2,1,0}};
constexpr std::array<int, 10> dv{{0,0,1,0,1,2,0,1,2,3}};
constexpr int factorial(int n) { return n < 2 ? 1 : n == 2 ? 2 : 6; }
constexpr int choose(int n, int k) { return factorial(n) / factorial(k) / factorial(n-k); }
int surface_index(int a, int b)
{
    for (int q = 0; q < 10; ++q) if (du[q] == a && dv[q] == b) return q;
    throw std::invalid_argument("Invalid surface jet exponent");
}
int spatial_index(int a, int b, int c)
{
    const auto& powers = cubic_cauchy_powers_3d();
    for (int q = 0; q < 20; ++q)
        if (powers[q].s == a && powers[q].t == b && powers[q].r == c) return q;
    throw std::invalid_argument("Invalid cubic spatial exponent");
}
Eigen::Vector3d geometry_second(const NativeSurfaceParameterJet3D& g, int a, int b)
{
    return a+b == 0 ? g.x_uu : a+b == 1 ? g.x_uv : g.x_vv;
}
Eigen::Vector3d geometry_third(const NativeSurfaceCubicParameterJet3D& g, int a, int b, int c)
{
    const int count = a+b+c;
    return count == 0 ? g.x_uuu : count == 1 ? g.x_uuv : count == 2 ? g.x_uvv : g.x_vvv;
}
void validate_graph(const TangentGraphThirdJet3D& graph)
{
    if (!graph.hessian.all_finite() || !std::isfinite(graph.sss)
        || !std::isfinite(graph.sst) || !std::isfinite(graph.stt) || !std::isfinite(graph.ttt))
        throw std::invalid_argument("Non-finite cubic tangent graph");
}
TangentGraphThirdJet3D graph_from_map(const NativeSurfaceCubicParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame,const Eigen::Matrix<double,10,10>& map)
{
    CubicValueJet3D parameter;
    parameter<<0,frame.normal.dot(geometry.lower.x_u),frame.normal.dot(geometry.lower.x_v),
        frame.normal.dot(geometry.lower.x_uu),frame.normal.dot(geometry.lower.x_uv),frame.normal.dot(geometry.lower.x_vv),
        frame.normal.dot(geometry.x_uuu),frame.normal.dot(geometry.x_uuv),frame.normal.dot(geometry.x_uvv),frame.normal.dot(geometry.x_vvv);
    const CubicValueJet3D jet=map*parameter;
    TangentGraphThirdJet3D result;
    result.hessian={jet[3],jet[4],jet[5]};result.sss=jet[6];result.sst=jet[7];result.stt=jet[8];result.ttt=jet[9];
    return result;
}
Eigen::Matrix<double, 1, 20> monomials(
    const Eigen::Vector3d& displacement, const LocalOrthonormalFrame3D& frame, int degree)
{
    if (degree != 2 && degree != 3) throw std::invalid_argument("Cauchy degree must be 2 or 3");
    if (!displacement.allFinite()) throw std::invalid_argument("Non-finite Cauchy displacement");
    const double s = frame.tangent1.dot(displacement), t = frame.tangent2.dot(displacement);
    const double r = frame.normal.dot(displacement);
    const std::array<double, 4> sp{{1,s,s*s,s*s*s}}, tp{{1,t,t*t,t*t*t}}, rp{{1,r,r*r,r*r*r}};
    Eigen::Matrix<double, 1, 20> result;
    const auto& powers = cubic_cauchy_powers_3d();
    for (int q = 0; q < 20; ++q) {
        const auto p = powers[q];
        result[q] = p.degree() <= degree ? sp[p.s]*tp[p.t]*rp[p.r] : 0.0;
    }
    return result;
}
template<std::size_t N>
NativeDensityC0Stencil3D combine(const std::array<NativeDensityC0Stencil3D, N>& rows,
    const Eigen::Matrix<double, 1, static_cast<int>(N)>& weights)
{
    NativeDensityC0Stencil3D result;
    if (!weights.allFinite()) throw std::invalid_argument("Non-finite Cauchy row weights");
    for (std::size_t row=0; row<N; ++row) {
        if (weights[static_cast<int>(row)] == 0.0) continue;
        for (int q=0; q<rows[row].count; ++q) {
            const int index=rows[row].indices[q];
            int slot=0;
            while (slot<result.count && result.indices[slot]!=index) ++slot;
            if (slot==result.count) {
                if (slot>=16) throw std::runtime_error("Cubic Cauchy row exceeds 16 coefficients");
                result.indices[slot]=index;
                ++result.count;
            }
            result.weights[slot]+=weights[static_cast<int>(row)]*rows[row].weights[q];
        }
    }
    return result;
}

// Explicit differential closure.  Variables are physical derivatives, then
// factorial-normalized only once at the end.  All ten cubic derivatives are
// determined by the four value third derivatives, three normal second
// derivatives, and the three differentiated Laplace equations.
Eigen::Matrix<double,20,1> close_jet(const TangentGraphThirdJet3D& f,
    const CubicValueJet3D& g, const CubicNormalJet3D& j)
{
    Eigen::Matrix<double,20,1> d=Eigen::Matrix<double,20,1>::Zero();
    auto at=[&](int a,int b,int c)->double& { return d[spatial_index(a,b,c)]; };
    const double A=f.hessian.h11, B=f.hessian.h12, C=f.hessian.h22;
    at(0,0,0)=g[0]; at(1,0,0)=g[1]; at(0,1,0)=g[2]; at(0,0,1)=j[0];
    const double Hss=g[3]-A*j[0], Hst=g[4]-B*j[0], Htt=g[5]-C*j[0];
    const double Hsr=j[1]+A*g[1]+B*g[2], Htr=j[2]+B*g[1]+C*g[2];
    const double Hrr=-Hss-Htt;
    at(2,0,0)=Hss; at(1,1,0)=Hst; at(0,2,0)=Htt;
    at(1,0,1)=Hsr; at(0,1,1)=Htr; at(0,0,2)=Hrr;
    at(3,0,0)=g[6]-3*A*Hsr-f.sss*j[0];
    at(2,1,0)=g[7]-A*Htr-2*B*Hsr-f.sst*j[0];
    at(1,2,0)=g[8]-C*Hsr-2*B*Htr-f.stt*j[0];
    at(0,3,0)=g[9]-3*C*Htr-f.ttt*j[0];
    at(2,0,1)=j[3]-A*Hrr+2*A*Hss+2*B*Hst+f.sss*g[1]+f.sst*g[2]+(A*A+B*B)*j[0];
    at(1,1,1)=j[4]-B*Hrr+(A+C)*Hst+B*(Hss+Htt)+f.sst*g[1]+f.stt*g[2]+B*(A+C)*j[0];
    at(0,2,1)=j[5]-C*Hrr+2*B*Hst+2*C*Htt+f.stt*g[1]+f.ttt*g[2]+(B*B+C*C)*j[0];
    at(1,0,2)=-at(3,0,0)-at(1,2,0);
    at(0,1,2)=-at(2,1,0)-at(0,3,0);
    at(0,0,3)=-at(2,0,1)-at(0,2,1);
    const auto& powers=cubic_cauchy_powers_3d();
    for(int q=0;q<20;++q) d[q]/=factorial(powers[q].s)*factorial(powers[q].t)*factorial(powers[q].r);
    return d;
}
struct LocalAmbient {
    Eigen::Vector3d gradient;
    Eigen::Matrix3d hessian;
    double third[3][3][3]{};
};
LocalAmbient local_ambient(const KnownAmbientThird3D& data, const LocalOrthonormalFrame3D& frame)
{
    validate_local_orthonormal_frame_3d(frame);
    if (!std::isfinite(data.value) || !data.gradient.allFinite() || !data.hessian.allFinite())
        throw std::invalid_argument("Non-finite ambient jet");
    for (const auto& slice:data.third) if(!slice.allFinite()) throw std::invalid_argument("Non-finite ambient third derivative");
    Eigen::Matrix3d rotation;
    rotation.col(0)=frame.tangent1; rotation.col(1)=frame.tangent2; rotation.col(2)=frame.normal;
    LocalAmbient local;
    local.gradient=rotation.transpose()*data.gradient;
    local.hessian=rotation.transpose()*data.hessian*rotation;
    for(int a=0;a<3;++a) for(int b=0;b<3;++b) for(int c=0;c<3;++c)
        for(int k=0;k<3;++k)
            local.third[a][b][c]+=rotation(k,c)*rotation.col(a).dot(data.third[k]*rotation.col(b));
    return local;
}
} // namespace

const std::array<CauchyMonomial3D,20>& cubic_cauchy_powers_3d()
{
    static const auto powers=[] {
        std::array<CauchyMonomial3D,20> result{}; int q=0;
        for(int s=0;s<=3;++s) for(int t=0;t<=3-s;++t) for(int r=0;r<=3-s-t;++r)
            result[q++]={s,t,r};
        return result;
    }();
    return powers;
}
const std::array<CauchyMonomial3D,10>& quadratic_cauchy_powers_3d()
{
    static const auto powers=[] {
        std::array<CauchyMonomial3D,10> result{}; int q=0;
        for(int s=0;s<=2;++s) for(int t=0;t<=2-s;++t) for(int r=0;r<=2-s-t;++r)
            result[q++]={s,t,r};
        return result;
    }();
    return powers;
}
Eigen::Matrix<double,10,20> cubic_to_quadratic_selection_3d()
{
    Eigen::Matrix<double,10,20> result=Eigen::Matrix<double,10,20>::Zero();
    const auto& powers=quadratic_cauchy_powers_3d();
    for(int row=0;row<10;++row) result(row,spatial_index(powers[row].s,powers[row].t,powers[row].r))=1.0;
    return result;
}
double CubicCauchyPolynomial3D::evaluate(const Eigen::Vector3d& target,int degree) const
{
    validate_local_orthonormal_frame_3d(frame);
    if(!center.allFinite() || !coefficients.allFinite()) throw std::invalid_argument("Non-finite cubic polynomial");
    return (monomials(target-center,frame,degree)*coefficients)[0];
}
Eigen::Matrix<double,10,1> CubicCauchyPolynomial3D::quadratic_coefficients() const
{
    return cubic_to_quadratic_selection_3d()*coefficients;
}

NativeSurfaceCubicParameterJet3D native_surface_cubic_parameter_jet_3d(
    const NativeNurbsDensitySpace3D& density,int patch_index,double u,double v)
{
    if(patch_index<0 || patch_index>=density.patch_count()) throw std::out_of_range("Cubic geometry patch out of range");
    return native_surface_cubic_parameter_jet_3d(
        density.surface().patches[static_cast<std::size_t>(patch_index)],u,v);
}
NativeSurfaceCubicParameterJet3D native_surface_cubic_parameter_jet_3d(
    const geometry3d::NurbsSurfacePatch3D& patch,double u,double v)
{
    if(!std::isfinite(u) || !std::isfinite(v) || u < -1e-12 || u > 1+1e-12 || v < -1e-12 || v > 1+1e-12)
        throw std::out_of_range("Cubic geometry parameters outside [0,1]");
    u=std::clamp(u,0.0,1.0); v=std::clamp(v,0.0,1.0);
    const double su=patch.domain_end_u()-patch.domain_start_u(), sv=patch.domain_end_v()-patch.domain_start_v();
    if(!(su>0 && sv>0)) throw std::runtime_error("Degenerate cubic geometry domain");
    const double nu=patch.domain_start_u()+su*u, nv=patch.domain_start_v()+sv*v;
    const auto& ub=patch.basis_u(); const auto& vb=patch.basis_v();
    const auto iu=ub.active_basis_indices(nu), iv=vb.active_basis_indices(nv);
    const std::array<std::vector<double>,4> bu{{ub.evaluate_nonzero(nu),ub.evaluate_nonzero_first_derivatives(nu),
        ub.evaluate_nonzero_second_derivatives(nu),geometry::nonzero_basis_third_derivatives(ub,nu)}};
    const std::array<std::vector<double>,4> bv{{vb.evaluate_nonzero(nv),vb.evaluate_nonzero_first_derivatives(nv),
        vb.evaluate_nonzero_second_derivatives(nv),geometry::nonzero_basis_third_derivatives(vb,nv)}};
    Eigen::Vector3d numerator[4][4], derivatives[4][4]; double denominator[4][4]{};
    for(int a=0;a<=3;++a) for(int b=0;b<=3-a;++b) {
        numerator[a][b]=Eigen::Vector3d::Zero();
        for(std::size_t i=0;i<iu.size();++i) for(std::size_t j=0;j<iv.size();++j) {
            const double w=bu[a][i]*bv[b][j]*patch.weights()[iu[i]][iv[j]];
            numerator[a][b]+=w*patch.control_net()[iu[i]][iv[j]]; denominator[a][b]+=w;
        }
    }
    if(!std::isfinite(denominator[0][0]) || std::abs(denominator[0][0])<=64*std::numeric_limits<double>::epsilon())
        throw std::runtime_error("Cubic NURBS homogeneous weight is singular");
    for(int order=0;order<=3;++order) for(int a=0;a<=order;++a) {
        const int b=order-a; Eigen::Vector3d residual=numerator[a][b];
        for(int i=0;i<=a;++i) for(int j=0;j<=b;++j) if(i+j>0)
            residual-=choose(a,i)*choose(b,j)*denominator[i][j]*derivatives[a-i][b-j];
        derivatives[a][b]=residual/denominator[0][0];
        if(!derivatives[a][b].allFinite()) throw std::runtime_error("Non-finite cubic NURBS derivative");
    }
    NativeSurfaceCubicParameterJet3D result; auto& low=result.lower;
    low.point=derivatives[0][0]; low.x_u=su*derivatives[1][0]; low.x_v=sv*derivatives[0][1];
    low.x_uu=su*su*derivatives[2][0]; low.x_uv=su*sv*derivatives[1][1]; low.x_vv=sv*sv*derivatives[0][2];
    // Same oriented Jacobian convention as density.geometry(), reusing the
    // already evaluated parameter derivatives instead of querying it again.
    const Eigen::Vector3d cross=low.x_u.cross(low.x_v);
    const double area=cross.norm();
    if(!(area>0.0) || !std::isfinite(area)) throw std::runtime_error("Degenerate cubic surface Jacobian");
    low.normal=cross/area;
    result.x_uuu=su*su*su*derivatives[3][0]; result.x_uuv=su*su*sv*derivatives[2][1];
    result.x_uvv=su*sv*sv*derivatives[1][2]; result.x_vvv=sv*sv*sv*derivatives[0][3];
    return result;
}

Eigen::Matrix<double,10,10> parameter_to_cubic_cauchy_jet_matrix_3d(
    const NativeSurfaceCubicParameterJet3D& g,const LocalOrthonormalFrame3D& frame)
{
    const auto low=parameter_to_cauchy_value_jet_matrix_3d(g.lower,frame);
    Eigen::Matrix2d Y; Y<<frame.tangent1.dot(g.lower.x_u),frame.tangent1.dot(g.lower.x_v),
        frame.tangent2.dot(g.lower.x_u),frame.tangent2.dot(g.lower.x_v);
    const Eigen::Matrix2d inverse=Y.inverse();
    if(std::abs(frame.normal.dot(g.lower.x_u))+std::abs(frame.normal.dot(g.lower.x_v))
       >1e-8*(g.lower.x_u.norm()+g.lower.x_v.norm())) throw std::invalid_argument("Cubic Cauchy frame not tangent");
    const std::array<Eigen::Vector3d,2> tangent{{frame.tangent1,frame.tangent2}};
    Eigen::Matrix<double,10,10> map=Eigen::Matrix<double,10,10>::Zero();
    map.topLeftCorner<6,6>()=low;
    for(int col=0;col<10;++col) {
        double residual[2][2][2]{};
        for(int i=0;i<2;++i) for(int j=0;j<2;++j) for(int k=0;k<2;++k) {
            double value=col==surface_index(3-i-j-k,i+j+k)?1.0:0.0;
            for(int a=0;a<2;++a) {
                value-=map(1+a,col)*tangent[a].dot(geometry_third(g,i,j,k));
                for(int b=0;b<2;++b) {
                    const double h=map(surface_index(2-a-b,a+b),col);
                    value-=h*(tangent[a].dot(geometry_second(g.lower,i,j))*Y(b,k)
                        +tangent[a].dot(geometry_second(g.lower,i,k))*Y(b,j)
                        +Y(a,i)*tangent[b].dot(geometry_second(g.lower,j,k)));
                }
            }
            residual[i][j][k]=value;
        }
        for(int row=6;row<10;++row) {
            const int count=dv[row]; const int a=count==3?1:0, b=count>=2?1:0, c=count>=1?1:0;
            double value=0;
            for(int i=0;i<2;++i) for(int j=0;j<2;++j) for(int k=0;k<2;++k)
                value+=residual[i][j][k]*inverse(i,a)*inverse(j,b)*inverse(k,c);
            map(row,col)=value;
        }
    }
    if(!map.allFinite()) throw std::runtime_error("Non-finite cubic coordinate map");
    return map;
}
TangentGraphThirdJet3D tangent_graph_third_jet_3d(
    const NativeSurfaceCubicParameterJet3D& geometry,const LocalOrthonormalFrame3D& frame)
{
    return graph_from_map(geometry,frame,parameter_to_cubic_cauchy_jet_matrix_3d(geometry,frame));
}
CubicCauchyClosure3D build_cubic_cauchy_closure_3d(const TangentGraphThirdJet3D& graph)
{
    validate_graph(graph); CubicCauchyClosure3D result;
    for(int q=0;q<10;++q) result.value_map.col(q)=close_jet(graph,CubicValueJet3D::Unit(q),CubicNormalJet3D::Zero());
    for(int q=0;q<6;++q) result.normal_map.col(q)=close_jet(graph,CubicValueJet3D::Zero(),CubicNormalJet3D::Unit(q));
    return result;
}
CubicCauchyPolynomial3D CubicCauchyClosure3D::polynomial(const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,const CubicValueJet3D& value,const CubicNormalJet3D& normal) const
{
    if(!value.allFinite() || !normal.allFinite() || !center.allFinite()) throw std::invalid_argument("Non-finite cubic Cauchy data");
    validate_local_orthonormal_frame_3d(frame);
    CubicCauchyPolynomial3D result;result.center=center;result.frame=frame;
    result.coefficients=value_map*value+normal_map*normal; return result;
}
DirectCoefficientCubicCauchyPlan3D build_direct_coefficient_cubic_cauchy_plan_3d(
    const NativeNurbsDensitySpace3D& density,int patch,double u,double v,const LocalOrthonormalFrame3D& frame)
{
    return build_direct_coefficient_cubic_cauchy_plan_3d(density,patch,u,v,
        native_surface_cubic_parameter_jet_3d(density,patch,u,v),frame);
}
DirectCoefficientCubicCauchyPlan3D build_direct_coefficient_cubic_cauchy_plan_3d(
    const NativeNurbsDensitySpace3D& density,int patch,double u,double v,
    const NativeSurfaceCubicParameterJet3D& geometry,const LocalOrthonormalFrame3D& frame)
{
    const auto map=parameter_to_cubic_cauchy_jet_matrix_3d(geometry,frame);
    const auto rows=density.c0_parameter_cubic_jet_stencils(patch,u,v);
    DirectCoefficientCubicCauchyPlan3D result;result.patch=patch;result.u=u;result.v=v;result.center=geometry.lower.point;result.frame=frame;
    result.graph=graph_from_map(geometry,frame,map);result.closure=build_cubic_cauchy_closure_3d(result.graph);
    Eigen::Matrix2d jacobian;
    jacobian<<frame.tangent1.dot(geometry.lower.x_u),frame.tangent1.dot(geometry.lower.x_v),
        frame.tangent2.dot(geometry.lower.x_u),frame.tangent2.dot(geometry.lower.x_v);
    const Eigen::JacobiSVD<Eigen::Matrix2d> svd(jacobian);
    result.diagnostics.parameter_to_tangent_determinant=jacobian.determinant();
    result.diagnostics.parameter_to_tangent_condition=svd.singularValues()[0]/svd.singularValues()[1];
    result.diagnostics.tangent_plane_residual=std::max(std::abs(frame.normal.dot(geometry.lower.x_u)),
        std::abs(frame.normal.dot(geometry.lower.x_v)));
    for(int row=0;row<10;++row) {
        const Eigen::Matrix<double,1,10> weights=map.row(row);result.value_rows[row]=combine(rows,weights);
        if(row<6) result.normal_rows[row]=result.value_rows[row];
    }
    return result;
}
DirectCoefficientValueJetPlan3D DirectCoefficientCubicCauchyPlan3D::lower_value_plan() const
{
    DirectCoefficientValueJetPlan3D result;
    result.patch=patch;result.u=u;result.v=v;result.frame=frame;result.graph_hessian=graph.hessian;
    result.diagnostics=diagnostics;
    // Explicit surface derivative labels, not ambient P3 coefficient offsets.
    constexpr std::array<std::array<int,2>,6> powers{{{{0,0}},{{1,0}},{{0,1}},{{2,0}},{{1,1}},{{0,2}}}};
    for(int q=0;q<6;++q) result.cauchy_rows[q]=value_rows[surface_index(powers[q][0],powers[q][1])];
    return result;
}
DirectCoefficientNormalJetPlan3D DirectCoefficientCubicCauchyPlan3D::lower_normal_plan() const
{
    DirectCoefficientNormalJetPlan3D result;
    result.patch=patch;result.u=u;result.v=v;result.frame=frame;result.graph_hessian=graph.hessian;
    result.diagnostics=diagnostics;
    constexpr std::array<std::array<int,2>,3> powers{{{{0,0}},{{1,0}},{{0,1}}}};
    for(int q=0;q<3;++q) result.cauchy_rows[q]=normal_rows[surface_index(powers[q][0],powers[q][1])];
    return result;
}
CubicValueJet3D DirectCoefficientCubicCauchyPlan3D::evaluate_value_jet(const Eigen::Ref<const Eigen::VectorXd>& c) const
{
    CubicValueJet3D result;for(int q=0;q<10;++q) result[q]=value_rows[q].dot(c);return result;
}
CubicNormalJet3D DirectCoefficientCubicCauchyPlan3D::evaluate_normal_jet(const Eigen::Ref<const Eigen::VectorXd>& c) const
{
    CubicNormalJet3D result;for(int q=0;q<6;++q) result[q]=normal_rows[q].dot(c);return result;
}
Eigen::Matrix<double,1,10> DirectCoefficientCubicCauchyPlan3D::value_weights(const Eigen::Vector3d& d,int degree) const
{ return monomials(d,frame,degree)*closure.value_map; }
Eigen::Matrix<double,1,6> DirectCoefficientCubicCauchyPlan3D::normal_weights(const Eigen::Vector3d& d,int degree) const
{ return monomials(d,frame,degree)*closure.normal_map; }
NativeDensityC0Stencil3D DirectCoefficientCubicCauchyPlan3D::compose_value_row(const Eigen::Vector3d& d,int degree) const
{ return combine(value_rows,value_weights(d,degree)); }
NativeDensityC0Stencil3D DirectCoefficientCubicCauchyPlan3D::compose_normal_row(const Eigen::Vector3d& d,int degree) const
{ return combine(normal_rows,normal_weights(d,degree)); }

CubicValueJet3D known_cubic_dirichlet_jet_3d(const KnownAmbientThird3D& data,
    const TangentGraphThirdJet3D& f,const LocalOrthonormalFrame3D& frame)
{
    validate_graph(f);const auto a=local_ambient(data,frame);const auto& g=a.gradient;const auto& H=a.hessian;
    const double A=f.hessian.h11,B=f.hessian.h12,C=f.hessian.h22;
    CubicValueJet3D result;
    result<<data.value,g[0],g[1],H(0,0)+A*g[2],H(0,1)+B*g[2],H(1,1)+C*g[2],
        a.third[0][0][0]+3*A*H(0,2)+f.sss*g[2],
        a.third[0][0][1]+A*H(1,2)+2*B*H(0,2)+f.sst*g[2],
        a.third[0][1][1]+C*H(0,2)+2*B*H(1,2)+f.stt*g[2],
        a.third[1][1][1]+3*C*H(1,2)+f.ttt*g[2];
    return result;
}
CubicNormalJet3D known_cubic_neumann_jet_3d(const KnownAmbientThird3D& data,
    const TangentGraphThirdJet3D& f,const LocalOrthonormalFrame3D& frame)
{
    validate_graph(f);const auto a=local_ambient(data,frame);const auto& g=a.gradient;const auto& H=a.hessian;
    const double A=f.hessian.h11,B=f.hessian.h12,C=f.hessian.h22;
    CubicNormalJet3D result;
    result<<g[2],H(0,2)-A*g[0]-B*g[1],H(1,2)-B*g[0]-C*g[1],
        a.third[0][0][2]+A*H(2,2)-2*A*H(0,0)-2*B*H(0,1)-f.sss*g[0]-f.sst*g[1]-(A*A+B*B)*g[2],
        a.third[0][1][2]+B*H(2,2)-(A+C)*H(0,1)-B*(H(0,0)+H(1,1))-f.sst*g[0]-f.stt*g[1]-B*(A+C)*g[2],
        a.third[1][1][2]+C*H(2,2)-2*B*H(0,1)-2*C*H(1,1)-f.stt*g[0]-f.ttt*g[1]-(B*B+C*C)*g[2];
    return result;
}
} // namespace kfbim::app3d
