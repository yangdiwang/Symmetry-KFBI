#include "src/support/cauchy/direct_coefficient_cubic_cauchy_3d.hpp"
#include "src/geometry/nurbs_basis_derivatives.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace kfbim::app3d;
void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
void near(double a,double b,double tolerance,const char* message)
{ require(std::isfinite(a)&&std::abs(a-b)<tolerance,message); }

kfbim::geometry3d::NurbsSurfacePatch3D rational_patch()
{
    // X(u,v)=(u/(1+w*u),v,0); nonaffine rational coordinates on a plane.
    // Native knot domains are deliberately not [0,1].
    constexpr double w=0.4;
    kfbim::geometry::NurbsBasis1D ub(3,{2.,5.},{4,4});
    kfbim::geometry::NurbsBasis1D vb(3,{-4.,1.},{4,4});
    std::vector<std::vector<Eigen::Vector3d>> control(4,std::vector<Eigen::Vector3d>(4));
    std::vector<std::vector<double>> weights(4,std::vector<double>(4));
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) {
        weights[i][j]=1+w*i/3.;control[i][j]={i/3./weights[i][j],j/3.,0};
    }
    return kfbim::geometry3d::NurbsSurfacePatch3D(ub,vb,control,weights);
}

NativeNurbsDensitySpace3D closed_density()
{
    NativeNurbsDensityOptions3D options;options.coefficients_per_direction=4;
    options.reduction_backend=NativeDensityReductionBackend3D::BaseOnly;
    return NativeNurbsDensitySpace3D(make_native_nurbs_surface_3d(GeometryKind3D::Torus),options);
}

void test_basis_and_rational_geometry()
{
    kfbim::geometry::NurbsBasis1D basis(3,{0.,1.},{4,4});
    for(double u:{0.,0.37,1.}) {
        const auto d=kfbim::geometry::nonzero_basis_third_derivatives(basis,u);
        const double expected[]{-6,18,-18,6};
        for(int q=0;q<4;++q) near(d[q],expected[q],1e-13,"Bernstein third derivative");
    }
    const auto patch=rational_patch();
    for(double u:{0.,0.37,1.}) {
        const auto g=native_surface_cubic_parameter_jet_3d(patch,u,0.41);
        const double den=1+0.4*u;
        near(g.lower.point.x(),u/den,2e-14,"Rational point");
        near(g.lower.x_u.x(),1/(den*den),2e-13,"Rational first derivative normalized domain");
        near(g.lower.x_uu.x(),-0.8/std::pow(den,3),2e-12,"Rational second derivative");
        near(g.x_uuu.x(),0.96/std::pow(den,4),2e-11,"Rational third derivative quotient recurrence");
        require(g.x_uuv.norm()<2e-11 && g.x_uvv.norm()<2e-11 && g.x_vvv.norm()<2e-11,"Rational mixed derivative");
    }
}

void test_coefficient_jet_and_inverse_coordinates()
{
    auto density=closed_density();
    // Bernstein coefficients for mu(u,v)=u^3+2u^2 v+3u v^2+4v^3.
    Eigen::VectorXd c=Eigen::VectorXd::Zero(density.c0_coefficient_count());
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) {
        const double u1=i/3., v1=j/3.,u2=i*(i-1)/6.,v2=j*(j-1)/6.;
        c[density.local_to_c0()[4*i+j]]=(i==3?1.:0.)+2*u2*v1+3*u1*v2+(j==3?4.:0.);
    }
    for(double u:{0.,0.37,1.}) {
        const double v=0.41;
        const auto rows=density.c0_parameter_cubic_jet_stencils(0,u,v);
        const double expected[]{u*u*u+2*u*u*v+3*u*v*v+4*v*v*v,
            3*u*u+4*u*v+3*v*v,2*u*u+6*u*v+12*v*v,6*u+4*v,4*u+6*v,6*u+24*v,6,4,6,24};
        constexpr int du[]{0,1,0,2,1,0,3,2,1,0},dv[]{0,0,1,0,1,2,0,1,2,3};
        const auto low=density.c0_parameter_jet_stencils(0,u,v);
        for(int q=0;q<10;++q) {
            near(rows[q].dot(c),expected[q],4e-12,"Direct cubic density analytic derivative");
            near(rows[q].dot(c),density.c0_parameter_derivative_stencil(0,u,v,du[q],dv[q]).dot(c),1e-12,"Batched cubic derivative equivalence");
            if(q<6) near(rows[q].dot(c),low[q].dot(c),1e-14,"Batched P2 lower rows unchanged");
        }
    }
    const double u=0.37,v=0.41,den=1+0.4*u;
    const auto rational=native_surface_cubic_parameter_jet_3d(rational_patch(),u,v);
    const auto rational_frame=make_local_orthonormal_frame_3d(rational.lower.normal,Eigen::Vector3d::UnitX());
    CubicValueJet3D parameter_jet;
    const auto parameter_rows=density.c0_parameter_cubic_jet_stencils(0,u,v);
    for(int q=0;q<10;++q) parameter_jet[q]=parameter_rows[q].dot(c);
    // Inverse u(x)=x/(1-w*x): u_x=den^2,u_xx=2w den^3,u_xxx=6w^2 den^4.
    const CubicValueJet3D jet=parameter_to_cubic_cauchy_jet_matrix_3d(rational,rational_frame)*parameter_jet;
    const double fu=3*u*u+4*u*v+3*v*v, fuu=6*u+4*v;
    near(jet[6],6*std::pow(den,6)+3*fuu*den*den*(0.8*std::pow(den,3))
        +fu*0.96*std::pow(den,4),2e-10,"Third inverse-coordinate chain rule");
    near(jet[7],4*std::pow(den,4)+(4*u+6*v)*0.8*std::pow(den,3),2e-10,"Mixed third inverse-coordinate chain rule");
    near(jet[8],6*den*den,2e-10,"Second mixed third inverse-coordinate chain rule");
    // The assembled density plan itself is tested only against its actual
    // closed torus geometry; the planar chart above tests the coordinate map.
    const auto g=native_surface_cubic_parameter_jet_3d(density,0,u,v);
    const auto frame=make_local_orthonormal_frame_3d(g.lower.normal,g.lower.x_u);
    const auto plan=build_direct_coefficient_cubic_cauchy_plan_3d(density,0,u,v,g,frame);
    const auto old_geometry=native_surface_parameter_jet_3d(density,0,u,v);
    require((g.lower.x_uu-old_geometry.x_uu).norm()<2e-12,"Geometry P3 lower reuse equivalence");
    const auto old=build_direct_coefficient_value_jet_plan_3d(density,0,u,v,g.lower,frame);
    require((plan.lower_value_plan().evaluate(c)-old.evaluate(c)).norm()<2e-12,"P3 sparse lower value plan equals P2");
    const auto normal=build_direct_coefficient_normal_jet_plan_3d(density,0,u,v,g.lower,frame);
    require((plan.lower_normal_plan().evaluate(c)-normal.evaluate(c)).norm()<2e-12,"P3 sparse lower normal plan equals P2");
    const Eigen::Vector3d delta(0.1,-0.07,0.03);
    near(plan.compose_value_row(delta).dot(c),(plan.value_weights(delta)*plan.evaluate_value_jet(c))[0],3e-12,"Sparse cubic value row");
    near(plan.compose_normal_row(delta).dot(c),(plan.normal_weights(delta)*plan.evaluate_normal_jet(c))[0],3e-12,"Sparse cubic normal row");
}

void test_curved_cauchy_closure()
{
    TangentGraphThirdJet3D f;f.hessian={0.7,-0.23,0.41};f.sss=0.3;f.sst=-0.9;f.stt=0.6;f.ttt=-0.2;
    const double A=f.hessian.h11,B=f.hessian.h12,C=f.hessian.h22;
    const auto closure=build_cubic_cauchy_closure_3d(f);
    LocalOrthonormalFrame3D frame;
    // Independently prescribed surface Taylor derivatives for U=r, U=s,
    // U=r^2-s^2, U=s^3-3sr^2.  These are not generated by the inverse helper.
    for(int mode=0;mode<4;++mode) {
        CubicValueJet3D g=CubicValueJet3D::Zero();CubicNormalJet3D j=CubicNormalJet3D::Zero();
        if(mode==0) {g<<0,0,0,A,B,C,f.sss,f.sst,f.stt,f.ttt;j<<1,0,0,-A*A-B*B,-B*(A+C),-B*B-C*C;}
        if(mode==1) {g[1]=1;j<<0,-A,-B,-f.sss,-f.sst,-f.stt;}
        if(mode==2) {g[3]=-2;j<<0,0,0,6*A,4*B,2*C;}
        if(mode==3) {g[6]=6;}
        const auto p=closure.polynomial(Eigen::Vector3d::Zero(),frame,g,j);
        for(Eigen::Vector3d x:{Eigen::Vector3d(.17,-.11,.09),Eigen::Vector3d(-.23,.19,-.07)}) {
            const double expected=mode==0?x.z():mode==1?x.x():mode==2?x.z()*x.z()-x.x()*x.x():x.x()*x.x()*x.x()-3*x.x()*x.z()*x.z();
            near(p.evaluate(x),expected,4e-15,"Curved Cauchy cubic harmonic reproduction");
        }
    }
    DirectCoefficientCubicCauchyPlan3D p;p.frame=frame;p.graph=f;p.closure=closure;
    for(Eigen::Vector3d x:{Eigen::Vector3d(.17,-.11,.09),Eigen::Vector3d(-.23,.19,-.07)}) {
        const auto old=pweights_general_3d(x,frame,f.hessian);
        const auto w0=p.value_weights(x,2);const auto w1=p.normal_weights(x,2);
        require((w0.head<6>()-old.w0).norm()<2e-15 && w0.tail<4>().norm()==0,"P3 degree-2 value weights equal P2");
        require((w1.head<3>()-old.w1).norm()<2e-15 && w1.tail<3>().norm()==0,"P3 degree-2 normal weights equal P2");
    }
    CubicCauchyPolynomial3D polynomial;
    for(int q=0;q<20;++q) polynomial.coefficients[q]=q+0.5;
    const auto lower=polynomial.quadratic_coefficients();
    require((lower-polynomial.coefficients.head<10>()).norm()>1,"P3 to P2 must not slice first ten spatial coefficients");
    const Eigen::Vector3d x(.2,-.3,.1);double expected=0;
    for(int q=0;q<10;++q) {const auto p=quadratic_cauchy_powers_3d()[q];expected+=lower[q]*std::pow(x.x(),p.s)*std::pow(x.y(),p.t)*std::pow(x.z(),p.r);}
    near(polynomial.evaluate(x,2),expected,1e-14,"Multi-index quadratic selection evaluation");
}

void test_nonlinear_curved_chart()
{
    // A non-diagonal, nonlinear parameter chart over a prescribed cubic
    // tangent graph.  The forward chain rule is prescribed independently;
    // the production inverse must recover the original graph derivatives.
    Eigen::Matrix2d Y;Y<<2,.4,-.3,1.7;
    Eigen::Matrix2d chartH[2];chartH[0]<<.6,-.2,-.2,.1;chartH[1]<<-.1,.3,.3,.5;
    const double chartT[2][4]{{.2,-.4,.7,.1},{-.3,.5,.2,-.8}};
    Eigen::Matrix2d H;H<<.7,-.23,-.23,.41;
    const double T[4]{.3,-.9,.6,-.2};
    NativeSurfaceCubicParameterJet3D geometry;
    geometry.lower.x_u={Y(0,0),Y(1,0),0};geometry.lower.x_v={Y(0,1),Y(1,1),0};
    geometry.lower.normal=Eigen::Vector3d::UnitZ();
    const Eigen::Matrix2d parameterH=Y.transpose()*H*Y;
    geometry.lower.x_uu={chartH[0](0,0),chartH[1](0,0),parameterH(0,0)};
    geometry.lower.x_uv={chartH[0](0,1),chartH[1](0,1),parameterH(0,1)};
    geometry.lower.x_vv={chartH[0](1,1),chartH[1](1,1),parameterH(1,1)};
    for(int count=0;count<4;++count) {
        const int i=count==3?1:0,j=count>=2?1:0,k=count>=1?1:0;
        double third=0;
        for(int a=0;a<2;++a)for(int b=0;b<2;++b) {
            third+=H(a,b)*(chartH[a](i,j)*Y(b,k)+chartH[a](i,k)*Y(b,j)+Y(a,i)*chartH[b](j,k));
            for(int c=0;c<2;++c) third+=T[a+b+c]*Y(a,i)*Y(b,j)*Y(c,k);
        }
        Eigen::Vector3d derivative(chartT[0][count],chartT[1][count],third);
        if(count==0) geometry.x_uuu=derivative;if(count==1) geometry.x_uuv=derivative;
        if(count==2) geometry.x_uvv=derivative;if(count==3) geometry.x_vvv=derivative;
    }
    const auto graph=tangent_graph_third_jet_3d(geometry,LocalOrthonormalFrame3D{});
    require((graph.hessian.matrix()-H).norm()<3e-14,"Nonlinear curved chart Hessian");
    near(graph.sss,T[0],3e-14,"Nonlinear curved chart sss");near(graph.sst,T[1],3e-14,"Nonlinear curved chart sst");
    near(graph.stt,T[2],3e-14,"Nonlinear curved chart stt");near(graph.ttt,T[3],3e-14,"Nonlinear curved chart ttt");
}

void test_ambient_jets_and_rotation()
{
    TangentGraphThirdJet3D f;f.hessian={0.7,-0.23,0.41};f.sss=.3;f.sst=-.9;f.stt=.6;f.ttt=-.2;
    const Eigen::Matrix3d R=Eigen::AngleAxisd(.61,Eigen::Vector3d(.2,.3,.7).normalized()).toRotationMatrix();
    LocalOrthonormalFrame3D frame;frame.tangent1=R.col(0);frame.tangent2=R.col(1);frame.normal=R.col(2);
    const Eigen::Vector3d center(.11,-.23,.31);
    // U=0.4+2s-3t+0.8r+(r^2-s^2)+s^3-3sr^2.
    KnownAmbientThird3D data;data.value=.4;data.gradient=R*Eigen::Vector3d(2,-3,.8);
    Eigen::Matrix3d H=Eigen::Matrix3d::Zero();H(0,0)=-2;H(2,2)=2;data.hessian=R*H*R.transpose();
    double T[3][3][3]{};T[0][0][0]=6;T[0][2][2]=T[2][0][2]=T[2][2][0]=-6;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)
        for(int a=0;a<3;++a)for(int b=0;b<3;++b)for(int c=0;c<3;++c)
            data.third[k](i,j)+=R(i,a)*R(j,b)*R(k,c)*T[a][b][c];
    const auto g=known_cubic_dirichlet_jet_3d(data,f,frame);
    const auto j=known_cubic_neumann_jet_3d(data,f,frame);
    const auto polynomial=build_cubic_cauchy_closure_3d(f).polynomial(center,frame,g,j);
    const Eigen::Vector3d x(.17,-.11,.09);
    near(polynomial.evaluate(center+R*x),.4+2*x.x()-3*x.y()+.8*x.z()+x.z()*x.z()-x.x()*x.x()
        +x.x()*x.x()*x.x()-3*x.x()*x.z()*x.z(),4e-14,"Ambient third jet rotation and translation reproduction");
}
} // namespace
int main()
{
    try {
        test_basis_and_rational_geometry();test_coefficient_jet_and_inverse_coordinates();
        test_curved_cauchy_closure();test_nonlinear_curved_chart();test_ambient_jets_and_rotation();
        std::cout<<"direct_coefficient_cubic_cauchy_3d_test: PASS\n";return 0;
    } catch(const std::exception& e) {std::cerr<<"direct_coefficient_cubic_cauchy_3d_test: FAIL: "<<e.what()<<'\n';return 1;}
}
