#include "src/support/density/trace93_density_layout_3d.hpp"
#include "src/support/geometry/trace93_case_3d.hpp"
#include <Eigen/Cholesky>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace kfbim::app3d;
namespace {
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
Eigen::VectorXd greville(int ne) {
    std::vector<double> knots(4,0.);for(int i=1;i<ne;++i)knots.push_back(double(i)/ne);
    knots.insert(knots.end(),4,1.);Eigen::VectorXd g(ne+3);
    for(int i=0;i<g.size();++i)g[i]=(knots[i+1]+knots[i+2]+knots[i+3])/3.;return g;
}
void jets_and_trace(const Trace93DensityLayout3D& s) {
    Eigen::VectorXd c(s.reference_raw_dofs);
    for(int p=0;p<int(s.patch_spans.size());++p) {
        const auto u=greville(s.patch_spans[p][0]),v=greville(s.patch_spans[p][1]);
        for(int j=0;j<v.size();++j)for(int i=0;i<u.size();++i)
            c[s.patch_offsets[p]+j*u.size()+i]=u[i]+2*v[j]+u[i]*v[j];
    }
    for(int p=0;p<int(s.patch_spans.size());++p) {
        const double u=.371,v=.613;const auto rows=s.parameter_cubic_jet_stencils(p,u,v);
        const std::array<double,10> exact{{u+2*v+u*v,1+v,2+u,0,1,0,0,0,0,0}};
        for(int k=0;k<10;++k)require(std::abs(rows[k].dot(c)-exact[k])<5e-10,"analytic cubic parameter jet mismatch");
        const auto plan=s.cauchy_plan(p,u,v);
        const auto g=s.geometry_jet(p,u,v);
        const auto j=plan.evaluate_value_jet(c);
        Eigen::Matrix<double,3,2> T;T.col(0)=g.lower.x_u;T.col(1)=g.lower.x_v;
        const Eigen::Vector3d grad=T*(T.transpose()*T).ldlt().solve(Eigen::Vector2d(1+v,2+u));
        require(std::abs(j[0]-exact[0])<2e-12,"physical graph value jet mismatch");
        require(std::abs(j[1]-grad.dot(plan.frame.tangent1))<2e-11,"physical graph first tangent mismatch");
        require(std::abs(j[2]-grad.dot(plan.frame.tangent2))<2e-11,"physical graph second tangent mismatch");
        require(plan.frame.normal.dot(g.lower.normal)>1-1e-12,"physical graph normal orientation mismatch");
    }
    Eigen::VectorXd sample=s.trace_basis*c;
    for(int i=0;i<int(s.traces.size());i+=17) {
        const auto& q=s.traces[i];require(std::abs(sample[i]-(q.u+2*q.v+q.u*q.v))<3e-12,"trace basis polynomial mismatch");
    }
    if(s.edge_weights.size())require((s.edge_trace*sample-s.edge_basis*c).cwiseAbs().maxCoeff()<2e-10,"edge extrapolation is not exact on cubic density");
    if(s.vertex_weights.size())require((s.vertex_trace*sample-s.vertex_basis*c).cwiseAbs().maxCoeff()<2e-10,"vertex extrapolation is not exact on cubic density");
}
struct Expected {
    const char* geometry;
    int N,raw,dred,nred,traces,polar_rank,polar_kept;
    double dirichlet_constraint_residual,neumann_constraint_residual;
};
} // namespace

int main() {
    try {
        // September-9 SourceAndSummary archive, fresh_run_manifest.json,
        // rotate cases. Planar Dirichlet's independently interpolated edge
        // data are not exactly compatible with all smooth C1 constraints.
        // The source deliberately uses a least-squares affine lifting; test
        // the recorded defect, not an invented exact-feasibility assumption.
        // No PDE is run here.
        const Expected expected[]{
            {"l",32,350,122,67,896,132,132,1.388067457558373e-6,4.748263414400267e-16},
            {"u",32,550,190,109,1408,192,192,4.8586834837660575e-6,2.474149357611921e-16},
            {"cylinder",32,350,126,101,896,64,48,3.774758283725532e-15,2.894819800536297e-16},
            {"l",64,478,202,127,1792,172,172,4.967766834840237e-7,2.7755575615628914e-16},
            {"u",64,718,294,189,2560,240,240,1.7897361606067363e-6,2.914335439641036e-16},
            {"cylinder",64,574,270,229,2560,96,80,2.7978650550308176e-15,1.457167719820518e-16}};
        for(const auto& e:expected) {
            const auto problem=make_trace93_case_3d(e.geometry,"rotate");
            for(bool neumann:{false,true}) {
                const auto s=build_trace93_density_layout_3d(problem,2./e.N,neumann);
                std::cout<<e.geometry<<" N="<<e.N<<" bvp="<<(neumann?'N':'D')
                    <<" raw="<<s.reference_raw_dofs<<" reduced="<<s.Z.cols()<<" traces="<<s.traces.size()
                    <<" constraint_residual="<<s.constraint_residual<<" nullspace_residual="<<s.nullspace_residual<<std::endl;
                require(s.reference_raw_dofs==e.raw,"reference raw coefficient count mismatch");
                require(s.Z.cols()==(neumann?e.nred:e.dred),"reference constrained DOF count mismatch");
                require(int(s.traces.size())==e.traces,"reference exterior trace count mismatch");
                const double reference_residual=neumann?e.neumann_constraint_residual:e.dirichlet_constraint_residual;
                require(std::abs(s.constraint_residual-reference_residual)<2e-11+1e-7*reference_residual,
                    "affine constraint defect differs from archived source");
                const Eigen::VectorXd defect=s.C*s.particular-s.d;
                const double stationarity=(s.C.transpose()*defect).cwiseAbs().maxCoeff()
                    /std::max(1.,s.C.norm()*s.d.norm());
                require(stationarity<1e-11,"particular lifting is not a constraint least-squares solution");
                require((s.Z.transpose()*s.particular).norm()<2e-10*std::max(1.,s.particular.norm()),
                    "particular lifting is not the minimum-norm constraint solution");
                Eigen::VectorXd y(s.Z.cols());for(int i=0;i<y.size();++i)y[i]=std::sin(.31*(i+1));
                require((s.C*(s.particular+s.Z*y)-s.d-defect).cwiseAbs().maxCoeff()<2e-10,
                    "reduced coordinates changed the fixed affine constraint defect");
                require(s.nullspace_residual<2e-10,"constraint nullspace residual too large");
                require(s.Z.rows()==s.reference_raw_dofs,"raw coefficient space unexpectedly collapsed");
                require((s.Z.transpose()*s.Z-Eigen::MatrixXd::Identity(s.Z.cols(),s.Z.cols())).norm()<5e-10,"nullspace lost orthonormality");
                require(s.weights.minCoeff()>0,"nonpositive trace quadrature weight");
                if(neumann) {
                    require(s.polar.polar_rank==e.polar_rank&&s.polar.kept==e.polar_kept,"Polar-Star spectral selection differs from source");
                    require((s.mean_row*s.Z).norm()<2e-10,"Neumann mean gauge missing");
                    require(s.edge_weights.size()>0,"Neumann edge-star projector missing");
                } else {
                    require(s.edge_weights.size()==0&&s.vertex_weights.size()==0,"Dirichlet unexpectedly gained extra projector constraints");
                }
                if(e.N==32)jets_and_trace(s);
            }
        }
        std::cout<<"trace93 analysis density/Polar-Star/affine projector tests passed"<<std::endl;
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
