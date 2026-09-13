#include "src/support/trace/shared_field_transfer_3d.hpp"
#include "src/support/geometry/trace93_case_3d.hpp"
#include "src/bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace kfbim;
using namespace kfbim::app3d;
namespace {
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
double polynomial(const Eigen::Vector3d& x) {
    return 1.0+0.3*x.x()-0.2*x.z()+x.x()*x.x()-x.y()*x.y()+0.7*x.x()*x.z();
}
void check_event_telescoping(const Trace93Case3D& problem,const CartesianGrid3D& grid,
    const Eigen::VectorXi& labels,const SharedFieldSpace3D& space,const SharedFieldTransfer3D& transfer) {
    // Independent finite-face intersections, including short enter/exit paths
    // whose endpoints have the same label. The production transfer sees no events.
    struct Edge {int a,b;std::vector<std::pair<int,int>> jumps;};
    std::vector<Edge> edges;
    std::vector<int> query_ids;
    std::vector<Eigen::Vector3d> local(grid.num_dofs());
    for(int id=0;id<grid.num_dofs();++id) {
        const auto x=grid.coord(id);local[id]=space.geometry().to_reference({x[0],x[1],x[2]});
    }
    const auto dims=grid.dof_dims();
    int double_crossings=0;double independent_face_defect=0.;
    for(int k=1;k<dims[2]-1;++k) for(int j=1;j<dims[1]-1;++j) for(int i=1;i<dims[0]-1;++i)
        for(int axis=0;axis<3;++axis) {
            std::array<int,3> next{{i,j,k}};++next[axis];
            if(next[axis]>=dims[axis]-1) continue;
            const int a=grid.index(i,j,k),b=grid.index(next[0],next[1],next[2]);
            const Eigen::Vector3d delta=local[b]-local[a];
            std::vector<std::pair<double,int>> hits;
            for(const auto& face:space.geometry().rectangles) {
                if(std::abs(delta[face.axis])<1e-14) continue;
                const double t=(face.value-local[a][face.axis])/delta[face.axis];
                if(t<=1e-10||t>=1.-1e-10) continue;
                const Eigen::Vector3d x=local[a]+t*delta;
                bool on_face=true;
                for(int d=0;d<2;++d) on_face=on_face&&x[face.axes[d]]>face.limits[d][0]+1e-12
                    &&x[face.axes[d]]<face.limits[d][1]-1e-12;
                if(on_face) hits.emplace_back(t,face.patch);
            }
            if(hits.empty()) continue;
            std::sort(hits.begin(),hits.end());
            Edge edge{a,b,{}};int previous=labels[a];
            for(std::size_t q=0;q<hits.size();++q) {
                const double midpoint=.5*(hits[q].first+(q+1<hits.size()?hits[q+1].first:1.));
                const auto world=space.geometry().to_world(local[a]+midpoint*delta);
                const int after=problem.surface.exact_inside(world)?1:0;
                if(after!=previous) edge.jumps.emplace_back(after-previous,hits[q].second);
                previous=after;
            }
            require(previous==labels[b],"independent face crossings must recover endpoint label");
            if(edge.jumps.empty()) continue;
            if(labels[a]==labels[b]) {
                require(edge.jumps.size()>=2,"same-label crossing must have an enter/exit pair");
                ++double_crossings;int net=0;double wrong_face_net=0.;
                for(const auto& jump:edge.jumps) {net+=jump.first;wrong_face_net+=jump.first*.125*(jump.second+1);}
                require(net==0,"shared field must telescope on a same-label edge");
                independent_face_defect=std::max(independent_face_defect,std::abs(wrong_face_net));
            }
            edges.push_back(edge);query_ids.push_back(a);query_ids.push_back(b);
        }
    require(double_crossings>0,"fixture must contain actual same-label multiple crossings");
    require(independent_face_defect>.01,"independent per-face fields must expose a cancellation defect");
    std::sort(query_ids.begin(),query_ids.end());query_ids.erase(std::unique(query_ids.begin(),query_ids.end()),query_ids.end());
    std::vector<Eigen::Vector3d> query_points;
    for(int id:query_ids) query_points.push_back(local[id]);
    Eigen::VectorXd coefficients(space.coefficient_count());
    for(int i=0;i<coefficients.size();++i) coefficients[i]=std::sin(.17*i)+.3*std::cos(.37*i);
    const Eigen::VectorXd values=space.evaluation_matrix(query_points)*coefficients;
    Eigen::VectorXd expected=Eigen::VectorXd::Zero(grid.num_dofs());
    const double h=grid.spacing()[0];
    for(const auto& edge:edges) {
        const int ia=int(std::lower_bound(query_ids.begin(),query_ids.end(),edge.a)-query_ids.begin());
        const int ib=int(std::lower_bound(query_ids.begin(),query_ids.end(),edge.b)-query_ids.begin());
        for(const auto& jump:edge.jumps) {
            expected[edge.a]+=jump.first*values[ib]/(h*h);
            expected[edge.b]-=jump.first*values[ia]/(h*h);
        }
    }
    const auto actual=transfer.evaluate(coefficients).rhs_for_delta;
    const double error=(actual-expected).lpNorm<Eigen::Infinity>()/
        std::max({1.,actual.lpNorm<Eigen::Infinity>(),expected.lpNorm<Eigen::Infinity>()});
    require(error<2e-13,"shared spread must equal independently telescoped face-event corrections");
    std::cout<<"same-label multiple crossings="<<double_crossings<<" event RHS error="<<error<<"\n";
}
void run(const std::string& pose) {
    const int N=32;const double h=2.0/N;
    auto problem=make_trace93_case_3d("u",pose);
    CartesianGrid3D grid({-1.,-1.,-1.},{h,h,h},{N,N,N},DofLayout3D::Node);
    Eigen::VectorXi labels(grid.num_dofs());
    Eigen::VectorXd expected(grid.num_dofs());
    for(int i=0;i<grid.num_dofs();++i) {
        const auto p=grid.coord(i);const Eigen::Vector3d x(p[0],p[1],p[2]);
        labels[i]=problem.surface.exact_inside(x)?1:0;
        expected[i]=labels[i]*polynomial(problem.transform.inverse_point(x));
    }
    SharedFieldSpaceOptions3D options;options.h=h;
    SharedFieldSpace3D space(make_trace93_shared_field_geometry_3d(problem),options);
    std::vector<Eigen::Vector3d> points,normals;
    const std::vector<Eigen::Vector3d> local={{.55,.07,.12},{.22,.2,.1},{.3,.2,.5},{-.55,.1,-.2},{.24,-.08,.5}};
    const std::vector<Eigen::Vector3d> local_n={{1,0,0},{-1,0,0},{0,0,1},{-1,0,0},{0,0,1}};
    for(std::size_t i=0;i<local.size();++i) {
        points.push_back(problem.transform.forward_point(local[i]));
        normals.push_back(problem.transform.forward_vector(local_n[i]));
    }
    Eigen::VectorXd coef(space.coefficient_count());
    const auto& lattice=space.coefficient_lattice();
    // Harmonic P2 has zero Laplacian, so its cubic cardinal coefficients
    // are exactly the polynomial evaluated at the coefficient lattice.
    for(int i=0;i<coef.size();++i) coef[i]=polynomial(space.spacing()*lattice[i].cast<double>());
    for(bool neumann:{true,false}) for(auto mode:{SharedFieldRestrictMode3D::Staged,SharedFieldRestrictMode3D::Direct}) {
        SharedFieldTransfer3D transfer(grid,labels,points,normals,space,neumann,mode);
        if(neumann&&mode==SharedFieldRestrictMode3D::Staged) check_event_telescoping(problem,grid,labels,space,transfer);
        const auto result=transfer.evaluate(coef);
        LaplaceFftBulkSolverZfft3D poisson(grid,ZfftBcType::Dirichlet,0.,2);
        Eigen::VectorXd U;poisson.solve(result.rhs_for_delta,U);
        const auto trace=transfer.observe_grid(U);
        const double grid_error=(U-expected).lpNorm<Eigen::Infinity>();
        const double value_error=(trace.value+result.raw_correction.value).lpNorm<Eigen::Infinity>();
        const double normal_error=(trace.normal+result.raw_correction.normal).lpNorm<Eigen::Infinity>();
        std::cout<<pose<<" "<<neumann<<" "<<int(mode)<<" grid="<<grid_error<<" value="<<value_error<<" normal="<<normal_error<<"\n";
        require(grid_error<2e-12,"shared spread Delta sign/P2 grid identity");
        require(value_error<2e-12,"shared restrict P2 value identity");
        require(normal_error<2e-11,"shared restrict P2 normal identity");
        for(int k=0;k<=N;++k) for(int j=0;j<=N;++j) for(int i=0;i<=N;++i)
            if(i==0||i==N||j==0||j==N||k==0||k==N)
                require(result.rhs_for_delta[grid.index(i,j,k)]==0.,"box boundary rhs must remain zero");
        require(transfer.evaluate(Eigen::VectorXd::Zero(coef.size())).rhs_for_delta.norm()==0.,"zero field linearity");
        const auto before=transfer.observe_grid(U);transfer.evaluate(0.3*coef);
        require((transfer.observe_grid(U).value-before.value).norm()==0.,"Rg must not depend on last field");
    }
    bool failed=false;
    try { auto bad=points;bad[0]=Eigen::Vector3d(2,0,0);SharedFieldTransfer3D t(grid,labels,bad,normals,space,true); }
    catch(const std::exception&) {failed=true;}
    require(failed,"outside box query must fail");
    failed=false;
    try { auto bad=points;bad[0]=problem.transform.forward_point({0,0,0}); SharedFieldSpaceOptions3D small;small.h=h;small.width=.01;small.ratio=1.;SharedFieldSpace3D thin(make_trace93_shared_field_geometry_3d(problem),small);SharedFieldTransfer3D t(grid,labels,bad,normals,thin,true); }
    catch(const std::exception&) {failed=true;}
    require(failed,"insufficient band coverage must fail");
}
}
int main() {try {run("rotate");run("rotate_translate");std::cout<<"shared field transfer passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
