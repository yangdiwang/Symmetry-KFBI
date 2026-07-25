#include "neumann_edge_augmented_cauchy_3d.hpp"
#include "native_nurbs_surface_transform_3d.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace kfbim::app3d;
using Interval = kfbim::geometry3d::NurbsPatchEdgeInterval3D;

void require(bool ok, const std::string& message)
{
    if (!ok) throw std::runtime_error(message);
}

template<class Exception, class Function>
void require_throws(Function fn, const std::string& text)
{
    try { fn(); }
    catch (const Exception& e) {
        if (text.empty() || std::string(e.what()).find(text) != std::string::npos) return;
        throw std::runtime_error("exception omitted context: " + std::string(e.what()));
    }
    throw std::runtime_error("expected exception was not thrown");
}

std::pair<double,double> edge_uv(const kfbim::geometry3d::NurbsSurfacePatch3D& p,
                                 PatchEdge3D edge, double parameter)
{
    switch (edge) {
    case PatchEdge3D::UMin: return {p.domain_start_u(), parameter};
    case PatchEdge3D::UMax: return {p.domain_end_u(), parameter};
    case PatchEdge3D::VMin: return {parameter, p.domain_start_v()};
    case PatchEdge3D::VMax: return {parameter, p.domain_end_v()};
    }
    throw std::runtime_error("invalid edge in test");
}

Eigen::Vector3d edge_point(const NativeNurbsSurface3D& surface,
                           const Interval& interval, double parameter)
{
    const auto& patch = surface.patches.at(static_cast<std::size_t>(interval.patch));
    const auto uv = edge_uv(patch, interval.edge, parameter);
    return patch.evaluate(uv.first, uv.second);
}

double edge_length(const NativeNurbsSurface3D& surface, const Interval& interval)
{
    constexpr std::array<double,8> x{{-0.9602898564975363,-0.7966664774136267,
        -0.5255324099163290,-0.1834346424956498,0.1834346424956498,
        0.5255324099163290,0.7966664774136267,0.9602898564975363}};
    constexpr std::array<double,8> w{{0.1012285362903763,0.2223810344533745,
        0.3137066458778873,0.3626837833783620,0.3626837833783620,
        0.3137066458778873,0.2223810344533745,0.1012285362903763}};
    const auto& patch = surface.patches.at(static_cast<std::size_t>(interval.patch));
    const double half = 0.5*(interval.end-interval.begin);
    const double middle = 0.5*(interval.begin+interval.end);
    double result = 0.0;
    for (std::size_t q=0;q<x.size();++q) {
        const auto uv=edge_uv(patch,interval.edge,middle+half*x[q]);
        const auto d=patch.evaluate_with_derivatives(uv.first,uv.second);
        result += w[q]*((interval.edge==PatchEdge3D::UMin
            || interval.edge==PatchEdge3D::UMax)?d.dv.norm():d.du.norm());
    }
    return std::abs(half)*result;
}

std::set<int> reachable(const NativeNurbsSurface3D& surface,int start,int forbidden)
{
    std::set<int> result{start}; std::queue<int> pending; pending.push(start);
    while(!pending.empty()) {
        const int patch=pending.front(); pending.pop();
        for(const auto& slot:surface.smooth_neighbors.at(static_cast<std::size_t>(patch)))
            if(slot && slot->patch!=forbidden && result.insert(slot->patch).second)
                pending.push(slot->patch);
    }
    return result;
}

void require_side(const SurfaceDofCloud3D& cloud,const Eigen::Vector3d& point,
                  const std::vector<int>& ids,int count,int opposite,
                  const std::set<int>& allowed)
{
    require(static_cast<int>(ids.size())==count,"edge side has wrong stencil size");
    for(int k=0;k<count;++k) {
        const int id=ids[static_cast<std::size_t>(k)];
        require(id>=0 && id<static_cast<int>(cloud.dofs.size()),"invalid side DOF");
        const auto& dof=cloud.dofs[static_cast<std::size_t>(id)];
        require(dof.patch_id!=opposite && allowed.count(dof.patch_id)==1,
                "side selection crossed a non-G1 edge");
        if(k>0) {
            const int previous=ids[static_cast<std::size_t>(k-1)];
            const double d0=(cloud.dofs[static_cast<std::size_t>(previous)].point-point).squaredNorm();
            const double d1=(dof.point-point).squaredNorm();
            require(d0<d1 || (d0==d1 && previous<id),"side DOFs are not distance sorted");
        }
    }
}

void local_harmonic_data(const SurfaceDofCloud3D& cloud,
                         const NeumannEdgeAuxiliarySample3D& sample,double h,int column,
                         Eigen::VectorXd& values,Eigen::VectorXd& normals)
{
    const HarmonicPolynomialSpace3D space(3);
    values.resize(static_cast<Eigen::Index>(cloud.dofs.size())); normals.resize(values.size());
    for(int i=0;i<static_cast<int>(cloud.dofs.size());++i) {
        const auto& dof=cloud.dofs[static_cast<std::size_t>(i)];
        const Eigen::Vector3d xi=sample.frame.transpose()*(dof.point-sample.point)/h;
        values[i]=space.basis(xi.x(),xi.y(),xi.z())[column];
        const Eigen::Vector3d nc=sample.frame.transpose()*dof.normal;
        normals[i]=nc.dot(space.gradient(xi.x(),xi.y(),xi.z()).col(column))/h;
    }
}

double direct_prediction(const SurfaceDofCloud3D& cloud,
                         const NeumannEdgeAuxiliarySample3D& sample,double h,
                         const Eigen::VectorXd& values,const Eigen::VectorXd& normals)
{
    const HarmonicPolynomialSpace3D space(3); Eigen::MatrixXd design(76,16);
    Eigen::VectorXd rhs(76),sqrt_weights(76);
    auto add=[&](const std::vector<int>& ids,int offset,bool normal) {
        for(int k=0;k<static_cast<int>(ids.size());++k) {
            const int id=ids[static_cast<std::size_t>(k)]; const auto& dof=cloud.dofs[static_cast<std::size_t>(id)];
            const Eigen::Vector3d xi=sample.frame.transpose()*(dof.point-sample.point)/h;
            if(normal) {
                const Eigen::Vector3d nc=sample.frame.transpose()*dof.normal;
                design.row(offset+k)=nc.transpose()*space.gradient(xi.x(),xi.y(),xi.z());
                sqrt_weights[offset+k]=std::sqrt(0.85)/(0.35+xi.norm()); rhs[offset+k]=h*normals[id];
            } else {
                design.row(offset+k)=space.basis(xi.x(),xi.y(),xi.z()).transpose();
                sqrt_weights[offset+k]=1.0/(0.35+xi.norm()); rhs[offset+k]=values[id];
            }
        }
    };
    add(sample.first_value_dofs,0,false); add(sample.second_value_dofs,24,false);
    add(sample.first_normal_dofs,48,true); add(sample.second_normal_dofs,62,true);
    const Eigen::MatrixXd weighted=sqrt_weights.asDiagonal()*design;
    const Eigen::VectorXd coefficients=svd_pseudoinverse_3d(weighted,3.0e-12)
        *sqrt_weights.asDiagonal()*rhs;
    return space.basis(0,0,0).dot(coefficients);
}

void test_l_prism_geometry_topology_reproduction_and_direct_map()
{
    const auto surface=make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const double h=3.0/32.0; const auto cloud=make_native_surface_dofs_3d(surface,h);
    const auto map=build_neumann_edge_auxiliary_value_map_3d(surface,cloud,h);
    require(map.surface_size==static_cast<int>(cloud.dofs.size()),"edge map has wrong surface size");
    require(map.value_map.rows()==static_cast<int>(map.samples.size())
        && map.normal_map.rows()==map.value_map.rows()
        && map.value_map.cols()==map.surface_size && map.normal_map.cols()==map.surface_size,
        "edge map dimensions are inconsistent");
    require(map.diagnostics.expected_non_g1_connections==22
        && map.diagnostics.covered_non_g1_connections==22,"L-prism non-G1 coverage is wrong");
    require(map.diagnostics.pass,"edge diagnostics did not pass");
    const double gap_limit=1.0e-11*surface.geometry_model().control_bounds().diameter();
    std::map<int,std::vector<int>> occurrences; std::set<std::pair<int,int>> keys;
    Eigen::VectorXd values(map.surface_size),normals(map.surface_size);
    for(int i=0;i<map.surface_size;++i) { values[i]=std::sin(0.17*(i+1)); normals[i]=std::cos(0.11*(i+2)); }
    const Eigen::VectorXd sparse_values=evaluate_neumann_edge_values_3d(map,values,normals);
    const HarmonicPolynomialSpace3D space(3); const Eigen::VectorXd origin=space.basis(0,0,0);
    for(int row=0;row<static_cast<int>(map.samples.size());++row) {
        const auto& sample=map.samples[static_cast<std::size_t>(row)];
        const auto& connection=surface.geometric_connections.at(static_cast<std::size_t>(sample.connection_index));
        require(!connection.g1 && keys.insert({sample.connection_index,sample.sample_index}).second,
                "duplicate or G1 edge sample row");
        occurrences[sample.connection_index].push_back(sample.sample_index);
        const double l1=edge_length(surface,connection.first),l2=edge_length(surface,connection.second);
        require(std::abs(l1-l2)<=1.0e-11*std::max(l1,l2),"incident edge lengths disagree");
        const int expected=std::max(4,static_cast<int>(std::ceil(0.5*(l1+l2)/h)));
        require(sample.sample_count==expected && sample.sample_index>=0 && sample.sample_index<expected,
                "edge sample count/index is wrong");
        const double s=(sample.sample_index+0.5)/static_cast<double>(expected);
        const double p1=connection.first.begin+s*(connection.first.end-connection.first.begin);
        const double mapped_s=connection.reversed?1.0-s:s;
        const double p2=connection.second.begin+mapped_s*(connection.second.end-connection.second.begin);
        require(std::abs(sample.normalized_parameter-s)<=1.0e-15
            && std::abs(sample.first_parameter-p1)<=1.0e-15
            && std::abs(sample.second_parameter-p2)<=1.0e-15,"partial edge parameters are wrong");
        require((edge_point(surface,connection.first,p1)-edge_point(surface,connection.second,p2)).norm()<=gap_limit
            && sample.mapped_point_gap<=gap_limit,"mapped NURBS points disagree");
        require((sample.frame.transpose()*sample.frame-Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff()<=1.0e-12
            && sample.frame.determinant()>0.0,"edge frame is not right-handed orthonormal");
        const auto first=reachable(surface,sample.first_patch,sample.second_patch);
        const auto second=reachable(surface,sample.second_patch,sample.first_patch);
        require_side(cloud,sample.point,sample.first_value_dofs,24,sample.second_patch,first);
        require_side(cloud,sample.point,sample.first_normal_dofs,14,sample.second_patch,first);
        require_side(cloud,sample.point,sample.second_value_dofs,24,sample.first_patch,second);
        require_side(cloud,sample.point,sample.second_normal_dofs,14,sample.first_patch,second);
        require(sample.first_owner_dof==sample.first_value_dofs.front()
            && sample.second_owner_dof==sample.second_value_dofs.front(),"owner is not nearest value DOF");
        require(std::isfinite(sample.condition)&&sample.condition>0.0,"invalid fit condition");
        for(int column=0;column<16;++column) {
            Eigen::VectorXd v,n; local_harmonic_data(cloud,sample,h,column,v,n);
            const Eigen::VectorXd edge=evaluate_neumann_edge_values_3d(map,v,n);
            require(std::abs(edge[row]-origin[column])<=1.0e-11,
                    "edge map does not reproduce a harmonic cubic");
        }
        require(std::abs(sparse_values[row]-direct_prediction(cloud,sample,h,values,normals))<=1.0e-12,
                "sparse edge value differs from direct weighted fit");
    }
    for(const auto& item:occurrences) {
        auto indices=item.second; std::sort(indices.begin(),indices.end());
        for(int q=0;q<static_cast<int>(indices.size());++q) require(indices[static_cast<std::size_t>(q)]==q,
            "literal edge sample indices do not occur once");
    }
    Eigen::VectorXd v1(map.surface_size),n1(map.surface_size),v2(map.surface_size),n2(map.surface_size);
    for(int i=0;i<map.surface_size;++i) { v1[i]=std::sin(i+0.1); n1[i]=std::cos(i+0.2); v2[i]=0.01*(i+1); n2[i]=-0.02*(i+3); }
    const Eigen::VectorXd lhs=evaluate_neumann_edge_values_3d(map,v1+0.37*v2,n1+0.37*n2);
    const Eigen::VectorXd rhs=evaluate_neumann_edge_values_3d(map,v1,n1)+0.37*evaluate_neumann_edge_values_3d(map,v2,n2);
    require((lhs-rhs).lpNorm<Eigen::Infinity>()<=2.0e-13,"edge value map is not linear");
}
void world_harmonic_data(const SurfaceDofCloud3D& cloud,const RigidTransform3D& transform,
                         Eigen::VectorXd& values,Eigen::VectorXd& normals)
{
    const HarmonicPolynomialSpace3D space(3); Eigen::VectorXd coefficients(16);
    for(int i=0;i<16;++i) coefficients[i]=0.13*std::sin(0.37*(i+1));
    values.resize(static_cast<Eigen::Index>(cloud.dofs.size())); normals.resize(values.size());
    for(int i=0;i<static_cast<int>(cloud.dofs.size());++i) {
        const auto& dof=cloud.dofs[static_cast<std::size_t>(i)];
        const Eigen::Vector3d x=transform.inverse_point(dof.point);
        const Eigen::Vector3d base_normal=transform.rotation().transpose()*dof.normal;
        values[i]=space.basis(x.x(),x.y(),x.z()).dot(coefficients);
        normals[i]=base_normal.dot(space.gradient(x.x(),x.y(),x.z())*coefficients);
    }
}

double compare_rigid_prediction(const NativeNurbsSurface3D& source,double h,
                                const NeumannEdgeAuxiliaryValueMap3D& baseline,
                                const Eigen::VectorXd& baseline_prediction,
                                const RigidTransform3D& transform)
{
    const auto moved=transform_native_nurbs_surface_3d(source,transform);
    const auto cloud=make_native_surface_dofs_3d(moved,h);
    const auto map=build_neumann_edge_auxiliary_value_map_3d(moved,cloud,h);
    require(map.samples.size()==baseline.samples.size(),"rigid transform changed sample count");
    for(std::size_t i=0;i<map.samples.size();++i) {
        const auto& a=baseline.samples[i]; const auto& b=map.samples[i];
        require(a.connection_index==b.connection_index && a.sample_index==b.sample_index
            && a.sample_count==b.sample_count,"rigid transform changed connection/sample key");
    }
    Eigen::VectorXd values,normals; world_harmonic_data(cloud,transform,values,normals);
    return (evaluate_neumann_edge_values_3d(map,values,normals)-baseline_prediction)
        .lpNorm<Eigen::Infinity>();
}

void test_rigid_covariance()
{
    const auto surface=make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const double h=3.0/32.0; const auto cloud=make_native_surface_dofs_3d(surface,h);
    const auto map=build_neumann_edge_auxiliary_value_map_3d(surface,cloud,h);
    const RigidTransform3D identity; Eigen::VectorXd values,normals;
    world_harmonic_data(cloud,identity,values,normals);
    const Eigen::VectorXd reference=evaluate_neumann_edge_values_3d(map,values,normals);
    const RigidTransform3D translated(Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero(),
                                      Eigen::Vector3d(0.23,-0.17,0.31));
    const RigidTransform3D rotated=RigidTransform3D::from_axis_angle(
        Eigen::Vector3d(1.0,-2.0,0.7),0.61,Eigen::Vector3d(0.04,-0.03,0.08),
        Eigen::Vector3d(-0.12,0.09,0.15));
    const double defect=std::max(compare_rigid_prediction(surface,h,map,reference,translated),
                                 compare_rigid_prediction(surface,h,map,reference,rotated));
    require(defect<=1.0e-11,"rigid covariance defect is too large");
    std::cout<<"rigid covariance defect "<<defect<<'\n';
}

SurfaceDofCloud3D restricted_cloud(const SurfaceDofCloud3D& dense)
{
    SurfaceDofCloud3D result; result.expected_area=dense.expected_area;
    for(int patch=0;patch<static_cast<int>(dense.patches.size());++patch) {
        SurfaceDofPatch3D tensor=dense.patches[static_cast<std::size_t>(patch)];
        tensor.first_dof=static_cast<int>(result.dofs.size());
        const int wanted=patch<6?8:24; tensor.nu=wanted; tensor.nv=1;
        int copied=0;
        for(const auto& dof:dense.dofs) if(dof.patch_id==patch && copied<wanted) {
            result.dofs.push_back(dof); ++copied;
        }
        require(copied==wanted,"restricted fixture lacked source DOFs");
        result.patches.push_back(std::move(tensor));
    }
    return result;
}

NativeNurbsSurface3D degenerate_normal_surface()
{
    using Patch=kfbim::geometry3d::NurbsSurfacePatch3D;
    NativeNurbsSurface3D surface; surface.name="degenerate_normals"; surface.description="opposite normals";
    surface.patches.push_back(Patch::make_bilinear_plane({0,0,0},{1,0,0},{0,1,0},{1,1,0}));
    surface.patches.push_back(Patch::make_bilinear_plane({0,0,0},{0,1,0},{1,0,0},{1,1,0}));
    surface.patch_names={"positive","negative"}; surface.smooth_neighbors.resize(2);
    surface.topological_patch_neighbors={{1},{0}}; surface.patch_components={0,0}; surface.expected_area=2.0;
    surface.geometric_connections.push_back({{0,PatchEdge3D::VMin,0,1},
                                               {1,PatchEdge3D::UMin,0,1},false,false});
    return surface;
}

void test_rejections_and_shared_values()
{
    const auto surface=make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const double h=3.0/32.0; const auto cloud=make_native_surface_dofs_3d(surface,h);
    require(std::string(neumann_edge_cauchy_mode_name_3d(NeumannEdgeCauchyMode3D::None))=="none"
        && std::string(neumann_edge_cauchy_mode_name_3d(NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues))
            =="non_g1_auxiliary_values","mode names are wrong");
    auto invalid_surface=surface; invalid_surface.geometric_connections.front().first.patch=-1;
    require_throws<std::invalid_argument>([&]{(void)build_neumann_edge_auxiliary_value_map_3d(invalid_surface,cloud,h);},"connection 0");
    const auto coarse=make_native_surface_dofs_3d(surface,10.0);
    require_throws<std::runtime_error>([&]{(void)build_neumann_edge_auxiliary_value_map_3d(surface,coarse,10.0);},"insufficient");
    require_throws<std::invalid_argument>([&]{(void)build_neumann_edge_auxiliary_value_map_3d(surface,cloud,0.0);},"positive h");
    for(int which=0;which<5;++which) {
        NeumannEdgeAuxiliaryOptions3D options;
        if(which==0) options.degree=2; if(which==1) options.value_samples_per_side=23;
        if(which==2) options.normal_samples_per_side=13; if(which==3) options.minimum_edge_samples=5;
        if(which==4) options.rank_relative_cutoff=4.0e-12;
        require_throws<std::invalid_argument>([&]{(void)build_neumann_edge_auxiliary_value_map_3d(surface,cloud,h,options);},"");
    }
    auto restricted_surface=surface; std::size_t chosen=0;
    for(;chosen<restricted_surface.geometric_connections.size();++chosen) {
        const auto& c=restricted_surface.geometric_connections[chosen];
        if(!c.g1 && c.first.patch==6 && c.second.patch==0) break;
    }
    require(chosen<restricted_surface.geometric_connections.size(),"restricted connection missing");
    const auto connection=restricted_surface.geometric_connections[chosen];
    restricted_surface.geometric_connections={connection};
    const auto small=restricted_cloud(make_native_surface_dofs_3d(surface,0.08));
    (void)build_neumann_edge_auxiliary_value_map_3d(restricted_surface,small,h);
    bool erased=false;
    for(auto& slot:restricted_surface.smooth_neighbors[0]) if(slot && slot->patch==1) { slot.reset(); erased=true; break; }
    require(erased,"restricted fixture smooth relation missing");
    require_throws<std::runtime_error>([&]{(void)build_neumann_edge_auxiliary_value_map_3d(restricted_surface,small,h);},"insufficient");
    const auto degenerate=degenerate_normal_surface(); const auto degenerate_cloud=make_native_surface_dofs_3d(degenerate,0.1);
    require_throws<std::runtime_error>([&]{(void)build_neumann_edge_auxiliary_value_map_3d(degenerate,degenerate_cloud,0.1);},"connection 0 sample 0");
    const auto map=build_neumann_edge_auxiliary_value_map_3d(surface,cloud,h);
    require_throws<std::invalid_argument>([&]{(void)evaluate_neumann_edge_values_3d(map,Eigen::VectorXd::Zero(map.surface_size-1),Eigen::VectorXd::Zero(map.surface_size));},"jump data");
}

} // namespace

int main()
{
    try {
        test_l_prism_geometry_topology_reproduction_and_direct_map();
        test_rigid_covariance();
        test_rejections_and_shared_values();
        std::cout<<"3D Neumann auxiliary edge-value tests passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"3D Neumann auxiliary edge-value test failure: "<<error.what()<<'\n';
        return 1;
    }
}