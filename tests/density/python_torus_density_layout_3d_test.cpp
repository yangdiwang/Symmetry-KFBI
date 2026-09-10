#include "src/support/density/python_torus_density_layout_3d.hpp"
#include "src/support/geometry/trace_first_case_3d.hpp"
#include "src/geometry/nurbs_basis_derivatives.hpp"

#include <Eigen/QR>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
using namespace kfbim::app3d;
void require(bool valid,const char* message) { if(!valid) throw std::runtime_error(message); }
void near(double actual,double expected,double tolerance,const char* message)
{ require(std::isfinite(actual)&&std::abs(actual-expected)<tolerance,message); }

kfbim::geometry::NurbsBasis1D basis(int spans)
{
    std::vector<double> knots;std::vector<int> multiplicity;
    for(int i=0;i<=spans;++i) {knots.push_back(double(i)/spans);multiplicity.push_back(i==0||i==spans?4:1);}
    return {3,knots,multiplicity};
}
double evaluate(const kfbim::geometry::NurbsBasis1D& b,const Eigen::VectorXd& c,double x,int derivative)
{
    const auto indices=b.active_basis_indices(x);
    const auto values=derivative==0?b.evaluate_nonzero(x):derivative==1?
        b.evaluate_nonzero_first_derivatives(x):derivative==2?
        b.evaluate_nonzero_second_derivatives(x):kfbim::geometry::nonzero_basis_third_derivatives(b,x);
    double value=0;for(std::size_t i=0;i<indices.size();++i) value+=values[i]*c[indices[i]];return value;
}
void test_exact_midpoint_insertion()
{
    std::mt19937 random(20260909);std::uniform_real_distribution<double> uniform(-1,1);
    for(int spans:{1,2,4,8}) {
        const auto P=cubic_midpoint_insertion_matrix_3d(spans);
        require(P.rows()==2*spans+3&&P.cols()==spans+3,"Midpoint insertion dimensions");
        require(P.minCoeff()>=-1e-15,"Boehm prolongation must have nonnegative entries");
        require((P.rowwise().sum().array()-1.0).abs().maxCoeff()<2e-14,"Boehm row sums preserve constants");
        const auto coarse=basis(spans),fine=basis(2*spans);
        for(int trial=0;trial<3;++trial) {
            Eigen::VectorXd c(spans+3);for(int i=0;i<c.size();++i) c[i]=uniform(random);
            const Eigen::VectorXd refined=P*c;
            // None of the interior points lie on a coarse or inserted knot;
            // the one-sided third derivative therefore has no ambiguity.
            for(double x:{0.0,.031,.173,.369,.581,.827,.979,1.0})
                for(int derivative=0;derivative<=3;++derivative) {
                    const double a=evaluate(coarse,c,x,derivative),b=evaluate(fine,refined,x,derivative);
                    near(a,b,2e-10*(1+std::abs(a)),"Knot insertion changes a spline derivative");
                }
        }
    }
    bool failed=false;try {(void)cubic_midpoint_insertion_matrix_3d(0);} catch(const std::invalid_argument&) {failed=true;}
    require(failed,"Invalid midpoint span count must be rejected");
}
void check_seams(const NativeNurbsDensitySpace3D& carrier,const PythonTorusDensityLayout3D& layout)
{
    Eigen::VectorXd y(layout.Z.cols());
    for(int i=0;i<y.size();++i) y[i]=std::sin(.73*(i+1))+.3*std::cos(.19*i);
    const Eigen::VectorXd c=layout.particular+layout.Z*y;
    for(int major=0;major<4;++major) for(int minor=0;minor<4;++minor) {
        const int p=4*major+minor;
        for(bool major_seam:{false,true}) {
            const int next=major_seam?4*((major+1)%4)+minor:4*major+(minor+1)%4;
            for(double s:{.13,.37,.81}) {
                const double ua=major_seam?1:s,va=major_seam?s:1;
                const double ub=major_seam?0:s,vb=major_seam?s:0;
                const auto ga=carrier.geometry(p,ua,va),gb=carrier.geometry(next,ub,vb);
                require((ga.point-gb.point).norm()<3e-12,"Reference torus geometric seam mismatch");
                require((ga.tangents-gb.tangents).norm()<3e-11,"Reference torus parameter C1 premise is false");
                for(const auto order:{std::array<int,2>{0,0},std::array<int,2>{1,0},std::array<int,2>{0,1}}) {
                    const double a=carrier.c0_parameter_derivative_stencil(p,ua,va,order[0],order[1]).dot(c);
                    const double b=carrier.c0_parameter_derivative_stencil(next,ub,vb,order[0],order[1]).dot(c);
                    near(a,b,4e-11,"Anisotropic embedded C0/C1 seam constraint");
                }
            }
        }
    }
}
void test_torus_layout()
{
    const auto reference=make_trace_first_python_torus_case_3d("rotate");
    require(reference.density_spans(32)==std::array<int,2>{4,2},"Reference N32 density allocation");
    require(reference.density_spans(64)==std::array<int,2>{8,4},"Reference dyadic N64 allocation");
    NativeNurbsDensityOptions3D options;options.coefficients_per_direction=7;
    options.field=NativeDensityField3D::ValueTrace;
    options.reduction_backend=NativeDensityReductionBackend3D::BaseOnly;
    options.trace_gauss_order=4;
    NativeNurbsDensitySpace3D carrier(reference.surface,options);
    const auto dirichlet=build_python_torus_density_layout_3d(carrier,4,2,false);
    const auto neumann=build_python_torus_density_layout_3d(carrier,4,2,true);
    require(dirichlet.reference_raw_dofs==560&&neumann.reference_raw_dofs==560,"Reference anisotropic raw DOFs");
    require(dirichlet.Z.cols()==240&&neumann.Z.cols()==239,"Reference constrained N/D DOFs");
    require(dirichlet.Z.rows()==carrier.c0_coefficient_count(),"Layout must map into native C0 carrier");
    require(dirichlet.traces.size()==2048&&neumann.traces.size()==2048,"Four by four Gauss per anisotropic cell");
    require(dirichlet.weights.minCoeff()>0&&neumann.weights.minCoeff()>0,"Positive trace weights");
    require(dirichlet.c0_embedding_error<2e-12&&neumann.c0_embedding_error<2e-12,"Exact C0 embedding");
    require(neumann.mean_nullspace_error<2e-12,"Mean-free Neumann tensor space");
    require((neumann.mean_row*neumann.Z).norm()<1e-11,"Independent mean-nullspace check");
    require(dirichlet.particular.norm()==0&&neumann.particular.norm()==0,"Smooth torus affine particular is zero");
    near(dirichlet.mean_row.sum(),reference.surface.expected_area,2e-10,"Gauss-8 area/mean integration");
    near(dirichlet.weights.sum(),reference.surface.expected_area,2e-5,"Trace quadrature area");
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(dirichlet.Z);qr.setThreshold(1e-11);
    require(qr.rank()==240,"Prolongation/collapse preserves full trial-space rank");
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_n(neumann.Z);qr_n.setThreshold(1e-11);
    require(qr_n.rank()==239,"Mean removal reduces exactly one dimension");
    // Constants belong to D but are excluded by the N gauge.
    const Eigen::VectorXd ones=Eigen::VectorXd::Ones(dirichlet.Z.rows());
    require((dirichlet.Z*qr.solve(ones)-ones).norm()<2e-11,"Dirichlet density space reproduces constants");
    check_seams(carrier,dirichlet);check_seams(carrier,neumann);
    for(const auto& q:dirichlet.traces) {
        require(q.element_u>=0&&q.element_u<4&&q.element_v>=0&&q.element_v<2,"Anisotropic cell indices");
        require(q.u>q.element_u/4.&&q.u<(q.element_u+1)/4.&&q.v>q.element_v/2.&&q.v<(q.element_v+1)/2.,"Gauss node lies strictly inside its cell");
    }
    bool failed=false;
    try {(void)build_python_torus_density_layout_3d(carrier,4,4,false);} catch(const std::invalid_argument&) {failed=true;}
    require(failed,"Incorrect anisotropic ratio must not silently run");
}
} // namespace
int main()
{
    try {test_exact_midpoint_insertion();test_torus_layout();std::cout<<"Python torus density layout tests passed\n";return 0;}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
