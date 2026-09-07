#include "src/support/cauchy/neumann_edge_augmented_cauchy_3d.hpp"
#include "src/support/cauchy/neumann_edge_augmented_cauchy_3d_detail.hpp"
#include "src/support/geometry/native_nurbs_surface_transform_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <new>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#if defined(_DEBUG)
#include <crtdbg.h>
#endif
#endif

namespace {
bool track_runtime_allocations=false;
std::size_t runtime_allocation_count=0;

void record_runtime_allocation() noexcept
{
    if(track_runtime_allocations) ++runtime_allocation_count;
}

void* allocate_runtime_memory(std::size_t size) noexcept
{
    return std::malloc(size==0?1:size);
}

void* allocate_aligned_runtime_memory(
    std::size_t size,std::size_t alignment) noexcept
{
#if defined(_MSC_VER)
    return _aligned_malloc(size==0?1:size,alignment);
#else
    const std::size_t requested=size==0?1:size;
    const std::size_t remainder=requested%alignment;
    if(remainder!=0
        && requested>std::numeric_limits<std::size_t>::max()
            -(alignment-remainder)) return nullptr;
    const std::size_t rounded=remainder==0
        ?requested:requested+(alignment-remainder);
    return std::aligned_alloc(alignment,rounded);
#endif
}

void release_aligned_runtime_memory(void* memory) noexcept
{
#if defined(_MSC_VER)
    _aligned_free(memory);
#else
    std::free(memory);
#endif
}

#if defined(_MSC_VER) && defined(_DEBUG)
_CRT_ALLOC_HOOK previous_crt_allocation_hook=nullptr;

int __cdecl runtime_crt_allocation_hook(
    int allocation_type,
    void* user_data,
    std::size_t size,
    int block_type,
    long request_number,
    const unsigned char* filename,
    int line_number)
{
    if(track_runtime_allocations
        && (allocation_type==_HOOK_ALLOC
            || allocation_type==_HOOK_REALLOC)) {
        ++runtime_allocation_count;
    }
    if(previous_crt_allocation_hook) {
        return previous_crt_allocation_hook(
            allocation_type,user_data,size,block_type,request_number,
            filename,line_number);
    }
    return 1;
}
#endif

class ScopedRuntimeAllocationCounter {
public:
    ScopedRuntimeAllocationCounter()
    {
#if defined(_MSC_VER) && defined(_DEBUG)
        previous_crt_allocation_hook=
            _CrtSetAllocHook(runtime_crt_allocation_hook);
#endif
        runtime_allocation_count=0;
        track_runtime_allocations=true;
    }

    ~ScopedRuntimeAllocationCounter()
    {
        track_runtime_allocations=false;
#if defined(_MSC_VER) && defined(_DEBUG)
        _CrtSetAllocHook(previous_crt_allocation_hook);
        previous_crt_allocation_hook=nullptr;
#endif
    }

    ScopedRuntimeAllocationCounter(
        const ScopedRuntimeAllocationCounter&)=delete;
    ScopedRuntimeAllocationCounter& operator=(
        const ScopedRuntimeAllocationCounter&)=delete;

    std::size_t count() const noexcept
    {
        return runtime_allocation_count;
    }
};
}

void* operator new(std::size_t size)
{
    record_runtime_allocation();
    if(void* memory=allocate_runtime_memory(size)) return memory;
    throw std::bad_alloc();
}

void* operator new[](std::size_t size)
{
    record_runtime_allocation();
    if(void* memory=allocate_runtime_memory(size)) return memory;
    throw std::bad_alloc();
}

void* operator new(std::size_t size,const std::nothrow_t&) noexcept
{
    record_runtime_allocation();
    return allocate_runtime_memory(size);
}

void* operator new[](std::size_t size,const std::nothrow_t&) noexcept
{
    record_runtime_allocation();
    return allocate_runtime_memory(size);
}

void* operator new(std::size_t size,std::align_val_t alignment)
{
    record_runtime_allocation();
    if(void* memory=allocate_aligned_runtime_memory(
            size,static_cast<std::size_t>(alignment))) return memory;
    throw std::bad_alloc();
}

void* operator new[](std::size_t size,std::align_val_t alignment)
{
    record_runtime_allocation();
    if(void* memory=allocate_aligned_runtime_memory(
            size,static_cast<std::size_t>(alignment))) return memory;
    throw std::bad_alloc();
}

void* operator new(
    std::size_t size,std::align_val_t alignment,
    const std::nothrow_t&) noexcept
{
    record_runtime_allocation();
    return allocate_aligned_runtime_memory(
        size,static_cast<std::size_t>(alignment));
}

void* operator new[](
    std::size_t size,std::align_val_t alignment,
    const std::nothrow_t&) noexcept
{
    record_runtime_allocation();
    return allocate_aligned_runtime_memory(
        size,static_cast<std::size_t>(alignment));
}

void operator delete(void* memory) noexcept
{
    std::free(memory);
}

void operator delete[](void* memory) noexcept
{
    std::free(memory);
}

void operator delete(void* memory,std::size_t) noexcept
{
    std::free(memory);
}

void operator delete[](void* memory,std::size_t) noexcept
{
    std::free(memory);
}

void operator delete(void* memory,const std::nothrow_t&) noexcept
{
    std::free(memory);
}

void operator delete[](void* memory,const std::nothrow_t&) noexcept
{
    std::free(memory);
}

void operator delete(void* memory,std::align_val_t) noexcept
{
    release_aligned_runtime_memory(memory);
}

void operator delete[](void* memory,std::align_val_t) noexcept
{
    release_aligned_runtime_memory(memory);
}

void operator delete(
    void* memory,std::size_t,std::align_val_t) noexcept
{
    release_aligned_runtime_memory(memory);
}

void operator delete[](
    void* memory,std::size_t,std::align_val_t) noexcept
{
    release_aligned_runtime_memory(memory);
}

void operator delete(
    void* memory,std::align_val_t,const std::nothrow_t&) noexcept
{
    release_aligned_runtime_memory(memory);
}

void operator delete[](
    void* memory,std::align_val_t,const std::nothrow_t&) noexcept
{
    release_aligned_runtime_memory(memory);
}

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

void test_exact_distance_selection_boundaries()
{
    using Candidate=kfbim::app3d::detail::NeumannEdgeDistanceCandidate3D;
    using kfbim::app3d::detail::select_neumann_edge_distance_candidates_3d;
    const double epsilon=std::numeric_limits<double>::epsilon();

    const std::vector<Candidate> radius_candidates{
        {1.0,4,104,-1},
        {1.0+32.0*epsilon,0,100,-1},
        {2.0,1,101,-1},
        {3.0,2,102,-1}};
    require(select_neumann_edge_distance_candidates_3d(
                radius_candidates,1.0+16.0*epsilon,1)
            ==std::vector<int>({104}),
        "sub-tolerance lower-index sample narrowed exact radius eligibility");

    const std::vector<Candidate> cutoff_candidates{
        {0.1,1,101,-1},
        {0.2,2,102,-1},
        {0.3,3,103,-1},
        {1.0,4,104,-1},
        {1.0+32.0*epsilon,0,100,-1}};
    require(select_neumann_edge_distance_candidates_3d(
                cutoff_candidates,2.0,4)
            ==std::vector<int>({101,102,103,104}),
        "sub-tolerance fifth sample replaced a strictly nearer fourth sample");

    const std::vector<Candidate> certified_tie{
        {0.1,1,101,-1},
        {0.2,2,102,-1},
        {0.3,3,103,-1},
        {1.0,4,104,0},
        {1.0+32.0*epsilon,0,100,4}};
    require(select_neumann_edge_distance_candidates_3d(
                certified_tie,2.0,4)
            ==std::vector<int>({101,102,103,100}),
        "certified parameter-symmetric tie did not use sample index");
}

void test_geometric_symmetry_certificate()
{
    using Connection=kfbim::geometry3d::NurbsPatchEdgeConnection3D;
    using kfbim::app3d::detail::certified_l_prism_symmetric_partner_3d;
    const double h=3.0/32.0;
    const auto source=make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    std::vector<NativeNurbsSurface3D> rigid_poses;
    rigid_poses.push_back(source);
    rigid_poses.push_back(transform_native_nurbs_surface_3d(
        source,RigidTransform3D::from_axis_angle(
            Eigen::Vector3d(1.0,-2.0,0.7),0.61,
            Eigen::Vector3d(0.04,-0.03,0.08),
            Eigen::Vector3d(0.23,-0.17,0.31))));

    const Connection full{
        {0,PatchEdge3D::UMin,0.0,1.0},
        {6,PatchEdge3D::VMin,0.0,1.0},false,false};
    const Connection partial{
        {0,PatchEdge3D::UMin,0.0,0.5},
        {6,PatchEdge3D::VMin,0.0,1.0},false,false};
    const Connection reversed{
        {6,PatchEdge3D::VMin,0.0,1.0},
        {0,PatchEdge3D::UMin,0.0,1.0},true,false};
    const Connection partial_reversed{
        {6,PatchEdge3D::VMin,0.0,1.0},
        {0,PatchEdge3D::UMin,0.0,0.5},true,false};

    auto require_pair=[&](const NativeNurbsSurface3D& surface,
                          const SurfaceDofCloud3D& cloud,
                          const Connection& connection,
                          int first,int second,const std::string& context) {
        const auto& tensor=cloud.patches.at(0);
        const auto& center=cloud.dofs.at(static_cast<std::size_t>(
            tensor.dof_index(0,2)));
        NeumannEdgeAuxiliarySample3D sample;
        sample.sample_count=7;
        sample.sample_index=first;
        require(certified_l_prism_symmetric_partner_3d(
                    surface,cloud,center,connection,sample)==second,
            context+" did not certify its first partner");
        sample.sample_index=second;
        require(certified_l_prism_symmetric_partner_3d(
                    surface,cloud,center,connection,sample)==first,
            context+" did not certify its reciprocal partner");
    };
    for(const auto& surface:rigid_poses) {
        const auto cloud=make_native_surface_dofs_3d(surface,h);
        require_pair(surface,cloud,full,0,4,"full interval");
        require_pair(surface,cloud,partial,3,6,"partial interval");
        require_pair(surface,cloud,reversed,2,6,"reversed interval");
        require_pair(surface,cloud,partial_reversed,0,3,
            "partial reversed interval");
    }

    constexpr int huge_count=2100000000;
    auto huge_cloud=make_native_surface_dofs_3d(source,h);
    huge_cloud.patches[0].nv=huge_count;
    auto huge_center=huge_cloud.dofs.front();
    huge_center.j=huge_count-2;
    NeumannEdgeAuxiliarySample3D huge_sample;
    huge_sample.sample_count=huge_count;
    huge_sample.sample_index=huge_count-1;
    require(certified_l_prism_symmetric_partner_3d(
                source,huge_cloud,huge_center,full,huge_sample)
            ==huge_count-3,
        "large exact symmetry overflowed its integer product");

    auto replace_patch_zero=[](NativeNurbsSurface3D& surface,
                               std::vector<std::vector<Eigen::Vector3d>> net) {
        const auto& patch=surface.patches.at(0);
        surface.patches[0]=kfbim::geometry3d::NurbsSurfacePatch3D(
            patch.basis_u(),patch.basis_v(),std::move(net),patch.weights());
    };
    auto rejected_partner=[&](const NativeNurbsSurface3D& surface) {
        const auto cloud=make_native_surface_dofs_3d(surface,h);
        const auto& tensor=cloud.patches.at(0);
        const auto& center=cloud.dofs.at(static_cast<std::size_t>(
            tensor.dof_index(0,2)));
        NeumannEdgeAuxiliarySample3D sample;
        sample.sample_count=7;
        sample.sample_index=0;
        return certified_l_prism_symmetric_partner_3d(
            surface,cloud,center,full,sample);
    };

    auto sheared=source;
    auto sheared_net=sheared.patches.at(0).control_net();
    const Eigen::Vector3d along=
        sheared_net[0][1]-sheared_net[0][0];
    sheared_net[1][0]+=0.25*along;
    sheared_net[1][1]+=0.25*along;
    replace_patch_zero(sheared,std::move(sheared_net));
    require(sheared.name==source.name && sheared.description==source.description,
        "sheared fixture did not retain native metadata");
    require(rejected_partner(sheared)==-1,
        "metadata-preserving sheared control net was certified");

    auto bilinear=source;
    auto bilinear_net=bilinear.patches.at(0).control_net();
    const Eigen::Vector3d transverse=
        bilinear_net[1][0]-bilinear_net[0][0];
    const Eigen::Vector3d normal=along.cross(transverse).normalized();
    bilinear_net[1][1]+=0.1*normal;
    replace_patch_zero(bilinear,std::move(bilinear_net));
    require(bilinear.name==source.name && bilinear.description==source.description,
        "bilinear fixture did not retain native metadata");
    require(rejected_partner(bilinear)==-1,
        "metadata-preserving bilinear control net was certified");
}

void test_negative_reduced_symmetry_numerator_is_rejected()
{
    using kfbim::app3d::detail::certified_l_prism_symmetric_partner_3d;
    const auto surface=make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const auto cloud=make_native_surface_dofs_3d(surface,3.0/64.0);
    const auto& tensor=cloud.patches.at(6);
    const auto& center=cloud.dofs.at(static_cast<std::size_t>(
        tensor.dof_index(6,0)));
    constexpr int connection_index=11;
    const auto& connection=surface.geometric_connections.at(connection_index);
    NeumannEdgeAuxiliarySample3D sample;
    sample.connection_index=connection_index;
    sample.sample_index=0;
    sample.sample_count=13;
    require(center.patch_id==6 && center.i==6 && center.j==0,
        "negative reduced-numerator fixture selected the wrong center");
    require(certified_l_prism_symmetric_partner_3d(
                surface,cloud,center,connection,sample)==-1,
        "negative reduced symmetry numerator was not rejected");
}

void test_sample_count_integer_boundaries()
{
    using kfbim::app3d::detail::plan_neumann_edge_sample_count_3d;
    constexpr int maximum_factorization_units =
        std::numeric_limits<int>::max() / 2;
    constexpr int maximum_sparse_samples =
        std::numeric_limits<int>::max() / 48;

    require_throws<std::overflow_error>([]{
        (void)plan_neumann_edge_sample_count_3d(
            static_cast<double>(std::numeric_limits<int>::max()),
            4,0,0,17);
    },"connection 17");

    const auto exact_sparse_limit =
        plan_neumann_edge_sample_count_3d(
            4.0,4,maximum_sparse_samples-4,0,18);
    require(exact_sparse_limit.connection_sample_count==4
            && exact_sparse_limit.cumulative_sample_count
                ==maximum_sparse_samples,
        "sample count planner rejected the last sparse-storage-safe count");

    require_throws<std::overflow_error>([=]{
        (void)plan_neumann_edge_sample_count_3d(
            4.0,4,maximum_sparse_samples-3,0,19);
    },"sparse storage");

    const auto exact_factorization_reserve =
        plan_neumann_edge_sample_count_3d(
            4.0,4,0,maximum_factorization_units-4,20);
    require(exact_factorization_reserve.cumulative_sample_count==4,
        "sample count planner did not reserve downstream factorizations");

    require_throws<std::overflow_error>([=]{
        (void)plan_neumann_edge_sample_count_3d(
            4.0,4,0,maximum_factorization_units-3,21);
    },"factorization");
}

std::vector<NeumannEdgeFaceStencil3D> exact_face_stencils(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud)
{
    std::vector<NeumannEdgeFaceStencil3D> result;
    result.reserve(cloud.dofs.size());
    for(int center=0;center<static_cast<int>(cloud.dofs.size());++center) {
        result.push_back({
            nearest_g1_cauchy_dofs(surface,cloud,center,48),
            nearest_g1_cauchy_dofs(surface,cloud,center,28)});
    }
    return result;
}

bool independently_parameter_symmetric(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int center_id,
    const NeumannEdgeAuxiliarySample3D& first,
    const NeumannEdgeAuxiliarySample3D& second)
{
    if(first.connection_index!=second.connection_index
        || first.sample_count!=second.sample_count
        || first.sample_count<=0) return false;
    const auto& center=cloud.dofs.at(static_cast<std::size_t>(center_id));
    const auto& connection=surface.geometric_connections.at(
        static_cast<std::size_t>(first.connection_index));
    const Interval* interval=nullptr;
    bool reversed=false;
    if(center.patch_id==connection.first.patch) {
        interval=&connection.first;
    } else if(center.patch_id==connection.second.patch) {
        interval=&connection.second;
        reversed=connection.reversed;
    } else {
        return false;
    }
    const auto& tensor=cloud.patches.at(static_cast<std::size_t>(center.patch_id));
    const bool u_edge=interval->edge==PatchEdge3D::UMin
        || interval->edge==PatchEdge3D::UMax;
    const long long index=u_edge?center.j:center.i;
    const long long count=u_edge?tensor.nv:tensor.nu;
    const double doubled_begin=2.0*interval->begin;
    const double doubled_end=2.0*interval->end;
    const long long begin=static_cast<long long>(std::llround(doubled_begin));
    const long long end=static_cast<long long>(std::llround(doubled_end));
    if(doubled_begin!=static_cast<double>(begin)
        || doubled_end!=static_cast<double>(end)
        || count<=0 || index<0 || index>=count || begin==end) return false;
    long long numerator=2LL*index+1LL-begin*count;
    long long denominator=count*(end-begin);
    if(denominator<0) {numerator=-numerator;denominator=-denominator;}
    if(reversed) numerator=denominator-numerator;
    return static_cast<long long>(first.sample_index+second.sample_index+1)
            *denominator
        ==2LL*numerator*first.sample_count;
}

std::map<int,std::vector<int>> expected_edge_groups(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const std::vector<NeumannEdgeFaceStencil3D>& stencils,
    const NeumannEdgeAuxiliaryValueMap3D& edge_map,
    int center,
    double h)
{
    const auto& center_dof=cloud.dofs.at(static_cast<std::size_t>(center));
    double radius_squared=0.0;
    for(const int id:stencils.at(static_cast<std::size_t>(center)).value_dofs) {
        radius_squared=std::max(radius_squared,
            (cloud.dofs.at(static_cast<std::size_t>(id)).point-center_dof.point)
                .squaredNorm());
    }
    const auto component_vector=smooth_patch_component(surface,center_dof.patch_id);
    const std::set<int> component(component_vector.begin(),component_vector.end());
    std::map<int,std::vector<std::pair<std::pair<double,int>,int>>> candidates;
    for(int index=0;index<static_cast<int>(edge_map.samples.size());++index) {
        const auto& sample=edge_map.samples[static_cast<std::size_t>(index)];
        candidates[sample.connection_index].push_back({
            {(sample.point-center_dof.point).squaredNorm(),sample.sample_index},
            index});
    }
    std::map<int,std::vector<int>> result;
    for(auto& item:candidates) {
        const auto& connection=surface.geometric_connections.at(
            static_cast<std::size_t>(item.first));
        require(!connection.g1,"auxiliary map unexpectedly contains a G1 sample");
        const bool first=component.count(connection.first.patch)==1;
        const bool second=component.count(connection.second.patch)==1;
        auto& sorted=item.second;
        std::sort(sorted.begin(),sorted.end(),
            [](const auto& a,const auto& b){return a.first<b.first;});
        if(first==second || sorted.front().first.first>radius_squared)
            continue;
        require(sorted.size()>=4,"eligible connection has fewer than four samples");
        std::vector<int> selected;
        for(int q=0;q<4;++q) selected.push_back(sorted[static_cast<std::size_t>(q)].second);
        std::sort(selected.begin(),selected.end(),[&](int a,int b) {
            const auto& x=edge_map.samples[static_cast<std::size_t>(a)];
            const auto& y=edge_map.samples[static_cast<std::size_t>(b)];
            return std::tie(x.connection_index,x.sample_index)
                <std::tie(y.connection_index,y.sample_index);
        });
        result.emplace(item.first,std::move(selected));
    }
    return result;
}

const NeumannEdgeLocalMap3D* find_local_map(
    const NeumannEdgeAugmentedCauchy3D& augmented,
    int center)
{
    for(const auto& map:augmented.local_maps())
        if(map.center_dof==center) return &map;
    return nullptr;
}

bool bitwise_equal(double first,double second)
{
    return std::memcmp(&first,&second,sizeof(double))==0;
}

double independently_recomputed_local_defect(
    const NeumannEdgeAugmentedCauchy3D& augmented,
    const SurfaceDofCloud3D& cloud,
    double h)
{
    const HarmonicPolynomialSpace3D space(3);
    double defect=0.0;
    const auto& edge_map=augmented.edge_value_map();
    const Eigen::VectorXd origin_basis=space.basis(0.0,0.0,0.0);
    for(int row=0;row<static_cast<int>(edge_map.samples.size());++row) {
        const auto& sample=edge_map.samples[static_cast<std::size_t>(row)];
        for(int column=0;column<16;++column) {
            double predicted=0.0;
            for(Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator
                    it(edge_map.value_map,row);it;++it) {
                const auto& dof=cloud.dofs[static_cast<std::size_t>(it.col())];
                const Eigen::Vector3d xi=sample.frame.transpose()
                    *(dof.point-sample.point)/h;
                predicted+=it.value()
                    *space.basis(xi.x(),xi.y(),xi.z())[column];
            }
            for(Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator
                    it(edge_map.normal_map,row);it;++it) {
                const auto& dof=cloud.dofs[static_cast<std::size_t>(it.col())];
                const Eigen::Vector3d xi=sample.frame.transpose()
                    *(dof.point-sample.point)/h;
                const Eigen::Vector3d nc=sample.frame.transpose()*dof.normal;
                predicted+=it.value()*nc.dot(
                    space.gradient(xi.x(),xi.y(),xi.z()).col(column))/h;
            }
            defect=std::max(defect,
                std::abs(predicted-origin_basis[column]));
        }
    }
    for(const auto& local:augmented.local_maps()) {
        const auto& center=cloud.dofs[static_cast<std::size_t>(local.center_dof)];
        for(int column=0;column<16;++column) {
            Eigen::VectorXd coefficients=Eigen::VectorXd::Zero(16);
            for(int k=0;k<48;++k) {
                const auto& sample=cloud.dofs[static_cast<std::size_t>(
                    local.value_dofs[static_cast<std::size_t>(k)])];
                const Eigen::Vector3d d=(sample.point-center.point)/h;
                const Eigen::Vector3d xi(
                    d.dot(center.tangent1),d.dot(center.tangent2),d.dot(center.normal));
                coefficients+=local.value_map.col(k)
                    *space.basis(xi.x(),xi.y(),xi.z())[column];
            }
            for(int k=0;k<28;++k) {
                const auto& sample=cloud.dofs[static_cast<std::size_t>(
                    local.normal_dofs[static_cast<std::size_t>(k)])];
                const Eigen::Vector3d d=(sample.point-center.point)/h;
                const Eigen::Vector3d xi(
                    d.dot(center.tangent1),d.dot(center.tangent2),d.dot(center.normal));
                const Eigen::Vector3d nc(
                    sample.normal.dot(center.tangent1),
                    sample.normal.dot(center.tangent2),
                    sample.normal.dot(center.normal));
                const double physical_normal=nc.dot(
                    space.gradient(xi.x(),xi.y(),xi.z()).col(column))/h;
                coefficients+=local.normal_map.col(k)*physical_normal;
            }
            for(int k=0;k<static_cast<int>(local.edge_sample_indices.size());++k) {
                const auto& sample=augmented.edge_value_map().samples[
                    static_cast<std::size_t>(local.edge_sample_indices[
                        static_cast<std::size_t>(k)])];
                const Eigen::Vector3d d=(sample.point-center.point)/h;
                const Eigen::Vector3d xi(
                    d.dot(center.tangent1),d.dot(center.tangent2),d.dot(center.normal));
                coefficients+=local.edge_map.col(k)
                    *space.basis(xi.x(),xi.y(),xi.z())[column];
            }
            Eigen::VectorXd exact=Eigen::VectorXd::Zero(16);
            exact[column]=1.0;
            defect=std::max(defect,(coefficients-exact).lpNorm<Eigen::Infinity>());
            const Eigen::VectorXd center_basis=space.basis(0.0,0.0,0.0);
            defect=std::max(defect,
                std::abs(center_basis.dot(coefficients)-center_basis[column]));
            for(const int edge_index:local.edge_sample_indices) {
                const auto& sample=augmented.edge_value_map().samples[
                    static_cast<std::size_t>(edge_index)];
                const Eigen::Vector3d d=(sample.point-center.point)/h;
                const Eigen::Vector3d xi(
                    d.dot(center.tangent1),d.dot(center.tangent2),d.dot(center.normal));
                const Eigen::VectorXd basis=space.basis(xi.x(),xi.y(),xi.z());
                defect=std::max(defect,
                    std::abs(basis.dot(coefficients)-basis[column]));
            }
        }
    }
    return defect;
}

void test_local_attachment_groups_and_overwrite()
{
    const auto surface=make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const double h=3.0/32.0;
    const auto cloud=make_native_surface_dofs_3d(surface,h);
    const auto stencils=exact_face_stencils(surface,cloud);
    const auto augmented=build_neumann_edge_augmented_cauchy_3d(
        surface,cloud,h,stencils);
    require(augmented.surface_size()==static_cast<int>(cloud.dofs.size())
        && augmented.edge_sample_count()
            ==static_cast<int>(augmented.edge_value_map().samples.size()),
        "augmented Cauchy dimensions are inconsistent");
    std::set<int> seen_centers;
    int expected_affected=0,expected_corners=0,far_centers=0;
    std::vector<bool> affected(cloud.dofs.size(),false);
    for(int center=0;center<static_cast<int>(cloud.dofs.size());++center) {
        const auto expected=expected_edge_groups(
            surface,cloud,stencils,augmented.edge_value_map(),center,h);
        const auto* local=find_local_map(augmented,center);
        if(expected.empty()) {
            require(local==nullptr,"far center unexpectedly has a local edge map");
            ++far_centers;
            continue;
        }
        ++expected_affected;
        if(expected.size()>=2) ++expected_corners;
        require(local!=nullptr,"eligible center is missing a local edge map");
        require(seen_centers.insert(center).second,
            "center ID occurs more than once in local maps");
        affected[static_cast<std::size_t>(center)]=true;
        require(local->center_dof==center
            && local->value_dofs==stencils[static_cast<std::size_t>(center)].value_dofs
            && local->normal_dofs==stencils[static_cast<std::size_t>(center)].normal_dofs,
            "local map did not preserve the exact legacy stencil IDs");
        require(local->value_map.rows()==16 && local->value_map.cols()==48
            && local->normal_map.rows()==16 && local->normal_map.cols()==28
            && local->edge_map.rows()==16
            && local->edge_map.cols()
                ==static_cast<Eigen::Index>(local->edge_sample_indices.size())
            && local->value_map.allFinite() && local->normal_map.allFinite()
            && local->edge_map.allFinite(),
            "affected local map dimensions or entries are invalid");
        std::map<int,std::vector<int>> actual;
        std::pair<int,int> previous{-1,-1};
        for(const int index:local->edge_sample_indices) {
            require(index>=0 && index<augmented.edge_sample_count(),
                "local map contains an invalid edge sample index");
            const auto& sample=augmented.edge_value_map().samples[
                static_cast<std::size_t>(index)];
            const std::pair<int,int> key{sample.connection_index,sample.sample_index};
            require(previous<key,"local edge samples are not grouped deterministically");
            previous=key;
            actual[sample.connection_index].push_back(index);
        }
        require(actual.size()==expected.size(),
            "local edge groups do not match exact eligible connections");
        for(const auto& expected_group:expected) {
            const auto found=actual.find(expected_group.first);
            require(found!=actual.end(),"eligible exact-distance group is absent");
            if(found->second==expected_group.second) continue;
            std::vector<int> actual_only,expected_only;
            for(const int id:found->second)
                if(std::find(expected_group.second.begin(),expected_group.second.end(),id)
                    ==expected_group.second.end()) actual_only.push_back(id);
            for(const int id:expected_group.second)
                if(std::find(found->second.begin(),found->second.end(),id)
                    ==found->second.end()) expected_only.push_back(id);
            require(actual_only.size()==expected_only.size() && !actual_only.empty(),
                "nearest-four group differs by more than tied samples");
            std::vector<bool> matched(expected_only.size(),false);
            for(const int actual_id:actual_only) {
                const auto& actual_sample=augmented.edge_value_map().samples[
                    static_cast<std::size_t>(actual_id)];
                bool found_partner=false;
                for(std::size_t k=0;k<expected_only.size();++k) {
                    if(matched[k]) continue;
                    const auto& exact_sample=augmented.edge_value_map().samples[
                        static_cast<std::size_t>(expected_only[k])];
                    if(actual_sample.sample_index<exact_sample.sample_index
                        && independently_parameter_symmetric(
                            surface,cloud,center,actual_sample,exact_sample)) {
                        matched[k]=true;
                        found_partner=true;
                        break;
                    }
                }
                require(found_partner,
                    "nearest-four group replaced a strictly nearer uncertified sample");
            }
        }
        for(const auto& group:actual)
            require(group.second.size()==4,"local edge group is partial");
    }
    require(static_cast<int>(augmented.local_maps().size())==expected_affected
        && augmented.diagnostics().affected_center_count==expected_affected,
        "affected-center count is wrong");
    require(expected_corners>0
        && augmented.diagnostics().corner_center_count==expected_corners,
        "no corner center received multiple non-G1 edge groups");
    require(far_centers>0,"fixture has no unaffected far center");
    require(augmented.diagnostics().unrelated_attachment_count==0,
        "an unrelated non-G1 connection was attached");
    require(augmented.diagnostics().factorization_count
            ==2*(augmented.edge_sample_count()+expected_affected),
        "setup factorization count is incomplete");
    require(augmented.diagnostics().pass,
        "augmented Cauchy setup diagnostics did not pass");
    const double independently_recomputed=independently_recomputed_local_defect(
        augmented,cloud,h);

    require(std::abs(augmented.diagnostics().harmonic_cubic_reproduction_defect_max
            -independently_recomputed)<=2.0e-15,
        "combined reproduction diagnostic omits a coefficient or evaluation defect");
    for(int index=0;index<augmented.edge_sample_count();++index) {
        const auto& sample=augmented.edge_value_map().samples[
            static_cast<std::size_t>(index)];
        for(const int owner:{sample.first_owner_dof,sample.second_owner_dof}) {
            const auto* local=find_local_map(augmented,owner);
            require(local!=nullptr
                && std::find(local->edge_sample_indices.begin(),
                             local->edge_sample_indices.end(),index)
                    !=local->edge_sample_indices.end(),
                "edge sample owner lacks its incident local attachment");
        }
    }

    Eigen::VectorXd values(augmented.surface_size());
    Eigen::VectorXd normals(augmented.surface_size());
    for(int i=0;i<augmented.surface_size();++i) {
        values[i]=std::sin(0.031*(i+1));
        normals[i]=std::cos(0.047*(i+2));
    }
    const Eigen::VectorXd edge_values=augmented.edge_values(values,normals);
    Eigen::MatrixXd coefficients(augmented.surface_size(),16);
    for(int row=0;row<coefficients.rows();++row)
        for(int column=0;column<coefficients.cols();++column)
            coefficients(row,column)=0.125+0.003*row-0.007*column;
    const Eigen::MatrixXd before=coefficients;
    const int factorizations=augmented.diagnostics().factorization_count;
    augmented.overwrite_affected_coefficients(
        values,normals,edge_values,coefficients);
    require(augmented.diagnostics().factorization_count==factorizations,
        "evaluation changed the setup factorization count");
    for(int center=0;center<coefficients.rows();++center) {
        if(affected[static_cast<std::size_t>(center)]) continue;
        for(int column=0;column<coefficients.cols();++column)
            require(bitwise_equal(coefficients(center,column),before(center,column)),
                "unaffected coefficient row changed bitwise");
    }
}

void test_runtime_overwrite_has_no_per_center_heap_path()
{
    const auto surface=make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const double h=3.0/32.0;
    const auto cloud=make_native_surface_dofs_3d(surface,h);
    const auto stencils=exact_face_stencils(surface,cloud);
    const auto augmented=build_neumann_edge_augmented_cauchy_3d(
        surface,cloud,h,stencils);
    Eigen::VectorXd values=Eigen::VectorXd::LinSpaced(
        augmented.surface_size(),-0.25,0.75);
    Eigen::VectorXd normals=Eigen::VectorXd::LinSpaced(
        augmented.surface_size(),0.5,-0.5);
    const Eigen::VectorXd edges=augmented.edge_values(values,normals);
    Eigen::MatrixXd coefficients=Eigen::MatrixXd::Constant(
        augmented.surface_size(),16,0.2718281828459045);
    const Eigen::MatrixXd before=coefficients;
    std::vector<bool> affected(static_cast<std::size_t>(augmented.surface_size()),false);
    for(const auto& local:augmented.local_maps())
        affected[static_cast<std::size_t>(local.center_dof)]=true;
    const int factorizations=augmented.diagnostics().factorization_count;

    std::size_t observed_allocations=0;
    {
        ScopedRuntimeAllocationCounter scope;
        augmented.overwrite_affected_coefficients(
            values,normals,edges,coefficients);
        observed_allocations=scope.count();
    }
    require(observed_allocations==0,
        "runtime overwrite used a validation/workspace heap allocation path");
    require(augmented.diagnostics().factorization_count==factorizations,
        "runtime overwrite performed setup work or factorization");
    for(int row=0;row<coefficients.rows();++row) {
        if(affected[static_cast<std::size_t>(row)]) continue;
        for(int column=0;column<coefficients.cols();++column)
            require(bitwise_equal(coefficients(row,column),before(row,column)),
                "no-allocation runtime changed an unaffected row");
    }
}

struct GlobalHarmonicFields {
    Eigen::MatrixXd values;
    Eigen::MatrixXd normals;
};

GlobalHarmonicFields global_harmonic_fields(
    const SurfaceDofCloud3D& cloud,
    const RigidTransform3D& transform,
    const Eigen::Vector3d& origin,
    double length)
{
    const HarmonicPolynomialSpace3D space(3);
    GlobalHarmonicFields result;
    result.values.resize(static_cast<Eigen::Index>(cloud.dofs.size()),16);
    result.normals.resize(static_cast<Eigen::Index>(cloud.dofs.size()),16);
    for(int id=0;id<static_cast<int>(cloud.dofs.size());++id) {
        const auto& dof=cloud.dofs[static_cast<std::size_t>(id)];
        const Eigen::Vector3d point=transform.inverse_point(dof.point);
        const Eigen::Vector3d normal=transform.rotation().transpose()*dof.normal;
        const Eigen::Vector3d xi=(point-origin)/length;
        result.values.row(id)=space.basis(xi.x(),xi.y(),xi.z()).transpose();
        result.normals.row(id)=normal.transpose()
            *space.gradient(xi.x(),xi.y(),xi.z())/length;
    }
    return result;
}

double exact_global_harmonic_value(
    const Eigen::Vector3d& moved_point,
    const RigidTransform3D& transform,
    const Eigen::Vector3d& origin,
    double length,
    int column)
{
    const Eigen::Vector3d point=transform.inverse_point(moved_point);
    const Eigen::Vector3d xi=(point-origin)/length;
    return HarmonicPolynomialSpace3D(3).basis(xi.x(),xi.y(),xi.z())[column];
}

double require_local_harmonic_reproduction(
    const SurfaceDofCloud3D& cloud,
    double h,
    const NeumannEdgeAugmentedCauchy3D& augmented,
    const RigidTransform3D& transform,
    const Eigen::Vector3d& origin,
    double length)
{
    const HarmonicPolynomialSpace3D space(3);
    const auto fields=global_harmonic_fields(cloud,transform,origin,length);
    std::vector<Eigen::MatrixXd> expected_coefficients;
    expected_coefficients.reserve(augmented.local_maps().size());
    for(const auto& local:augmented.local_maps()) {
        const auto& center=cloud.dofs[static_cast<std::size_t>(local.center_dof)];
        Eigen::MatrixXd design(76,16),rhs(76,16);
        for(int k=0;k<48;++k) {
            const int id=local.value_dofs[static_cast<std::size_t>(k)];
            const auto& sample=cloud.dofs[static_cast<std::size_t>(id)];
            const Eigen::Vector3d d=(sample.point-center.point)/h;
            const Eigen::Vector3d xi(
                d.dot(center.tangent1),d.dot(center.tangent2),d.dot(center.normal));
            design.row(k)=space.basis(xi.x(),xi.y(),xi.z()).transpose();
            rhs.row(k)=fields.values.row(id);
        }
        for(int k=0;k<28;++k) {
            const int id=local.normal_dofs[static_cast<std::size_t>(k)];
            const auto& sample=cloud.dofs[static_cast<std::size_t>(id)];
            const Eigen::Vector3d d=(sample.point-center.point)/h;
            const Eigen::Vector3d xi(
                d.dot(center.tangent1),d.dot(center.tangent2),d.dot(center.normal));
            const Eigen::Vector3d normal_components(
                sample.normal.dot(center.tangent1),
                sample.normal.dot(center.tangent2),
                sample.normal.dot(center.normal));
            design.row(48+k)=normal_components.transpose()
                *space.gradient(xi.x(),xi.y(),xi.z());
            rhs.row(48+k)=h*fields.normals.row(id);
        }
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
        require(qr.rank()==16,"independent expected-coefficient solve is rank deficient");
        expected_coefficients.push_back(qr.solve(rhs));
    }

    double defect=0.0;
    const int factorization_count=augmented.diagnostics().factorization_count;
    for(int column=0;column<16;++column) {
        const Eigen::VectorXd values=fields.values.col(column);
        const Eigen::VectorXd normals=fields.normals.col(column);
        const Eigen::VectorXd edge_values=augmented.edge_values(values,normals);
        Eigen::MatrixXd coefficients=Eigen::MatrixXd::Constant(
            augmented.surface_size(),16,0.3141592653589793);
        augmented.overwrite_affected_coefficients(
            values,normals,edge_values,coefficients);
        for(std::size_t map_index=0;map_index<augmented.local_maps().size();++map_index) {
            const auto& local=augmented.local_maps()[map_index];
            const auto& center=cloud.dofs[static_cast<std::size_t>(local.center_dof)];
            defect=std::max(defect,
                (coefficients.row(local.center_dof).transpose()
                    -expected_coefficients[map_index].col(column))
                    .lpNorm<Eigen::Infinity>());
            const double center_value=space.basis(0.0,0.0,0.0).dot(
                coefficients.row(local.center_dof).transpose());
            defect=std::max(defect,std::abs(center_value
                -fields.values(local.center_dof,column)));
            for(const int edge_index:local.edge_sample_indices) {
                const auto& sample=augmented.edge_value_map().samples[
                    static_cast<std::size_t>(edge_index)];
                const Eigen::Vector3d d=(sample.point-center.point)/h;
                const Eigen::Vector3d xi(
                    d.dot(center.tangent1),d.dot(center.tangent2),
                    d.dot(center.normal));
                const double predicted=space.basis(xi.x(),xi.y(),xi.z()).dot(
                    coefficients.row(local.center_dof).transpose());
                const double exact=exact_global_harmonic_value(
                    sample.point,transform,origin,length,column);
                defect=std::max(defect,std::abs(predicted-exact));
            }
        }
    }
    require(augmented.diagnostics().factorization_count==factorization_count,
        "harmonic evaluation performed a new factorization");
    require(defect<=1.0e-11,
        "affected local maps do not reproduce all harmonic cubics");
    require(augmented.diagnostics().harmonic_cubic_reproduction_defect_max<=1.0e-11,
        "combined setup reproduction diagnostic is too large");
    return defect;
}

void test_local_harmonic_reproduction_and_rigid_covariance()
{
    const auto source=make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const double h=3.0/32.0;
    const auto bounds=source.geometry_model().control_bounds();
    const Eigen::Vector3d origin=0.5*(bounds.lower+bounds.upper);
    const double length=bounds.diameter();
    const std::array<RigidTransform3D,3> transforms{{
        RigidTransform3D{},
        RigidTransform3D(Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero(),
                         Eigen::Vector3d(0.23,-0.17,0.31)),
        RigidTransform3D::from_axis_angle(
            Eigen::Vector3d(1.0,-2.0,0.7),0.61,Eigen::Vector3d(0.04,-0.03,0.08),
            Eigen::Vector3d(-0.12,0.09,0.15))}};
    std::vector<std::tuple<int,std::vector<std::pair<int,int>>>> baseline_keys;
    double defect=0.0;
    for(std::size_t pose=0;pose<transforms.size();++pose) {
        const auto surface=transform_native_nurbs_surface_3d(source,transforms[pose]);
        const auto cloud=make_native_surface_dofs_3d(surface,h);
        const auto stencils=exact_face_stencils(surface,cloud);
        const auto augmented=build_neumann_edge_augmented_cauchy_3d(
            surface,cloud,h,stencils);
        std::vector<std::tuple<int,std::vector<std::pair<int,int>>>> keys;
        for(const auto& local:augmented.local_maps()) {
            std::vector<std::pair<int,int>> samples;
            for(const int index:local.edge_sample_indices) {
                const auto& sample=augmented.edge_value_map().samples[
                    static_cast<std::size_t>(index)];
                samples.push_back({sample.connection_index,sample.sample_index});
            }
            keys.push_back({local.center_dof,std::move(samples)});
        }
        if(pose<=1) {
            std::vector<int> tied_endpoint_samples;
            for(const auto& local_key:keys) {
                if(std::get<0>(local_key)!=9) continue;
                for(const auto& sample_key:std::get<1>(local_key))
                    if(sample_key.first==10)
                        tied_endpoint_samples.push_back(sample_key.second);
            }
            require(tied_endpoint_samples==std::vector<int>({0,1,2,3}),
                "baseline/translated tied endpoint samples are not canonical");
        }
        if(pose==0) baseline_keys=keys;
        else if(keys!=baseline_keys) {
            std::cout<<"attachment mismatch pose "<<pose
                <<" baseline_maps "<<baseline_keys.size()
                <<" moved_maps "<<keys.size()<<'\n';
            const std::size_t common=std::min(keys.size(),baseline_keys.size());
            for(std::size_t i=0;i<common;++i) if(keys[i]!=baseline_keys[i]) {
                std::cout<<"first attachment mismatch index "<<i
                    <<" baseline_center "<<std::get<0>(baseline_keys[i])
                    <<" moved_center "<<std::get<0>(keys[i])<<'\n';
                std::cout<<"baseline samples";
                for(const auto& key:std::get<1>(baseline_keys[i]))
                    std::cout<<" ("<<key.first<<','<<key.second<<')';
                std::cout<<"\nmoved samples";
                for(const auto& key:std::get<1>(keys[i]))
                    std::cout<<" ("<<key.first<<','<<key.second<<')';
                std::cout<<'\n';
                break;
            }
            require(false,
                "rigid transform changed center/connection/sample attachment keys");
        }
        defect=std::max(defect,require_local_harmonic_reproduction(
            cloud,h,augmented,transforms[pose],origin,length));
    }
    std::cout<<"local rigid covariance defect "<<defect<<'\n';
}
void test_local_map_rejections()
{
    const auto surface=make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const double h=3.0/32.0;
    const auto cloud=make_native_surface_dofs_3d(surface,h);
    const auto stencils=exact_face_stencils(surface,cloud);
    auto wrong_count=stencils;
    wrong_count.pop_back();
    require_throws<std::invalid_argument>([&]{
        (void)build_neumann_edge_augmented_cauchy_3d(surface,cloud,h,wrong_count);
    },"stencil count");
    auto wrong_value=stencils;
    wrong_value.front().value_dofs.pop_back();
    require_throws<std::invalid_argument>([&]{
        (void)build_neumann_edge_augmented_cauchy_3d(surface,cloud,h,wrong_value);
    },"48");
    auto wrong_normal=stencils;
    wrong_normal.front().normal_dofs.pop_back();
    require_throws<std::invalid_argument>([&]{
        (void)build_neumann_edge_augmented_cauchy_3d(surface,cloud,h,wrong_normal);
    },"28");
    for(int which=0;which<4;++which) {
        NeumannEdgeAugmentedCauchyOptions3D options;
        if(which==0) options.degree=2;
        if(which==1) options.edge_samples_per_connection=3;
        if(which==2) options.edge_weight_scale=0.5;
        if(which==3) options.rank_relative_cutoff=4.0e-12;
        require_throws<std::invalid_argument>([&]{
            (void)build_neumann_edge_augmented_cauchy_3d(
                surface,cloud,h,stencils,{},options);
        },"");
    }
    const auto baseline=build_neumann_edge_augmented_cauchy_3d(
        surface,cloud,h,stencils);
    require(!baseline.local_maps().empty(),"rank-deficiency fixture has no affected center");
    const int component_center=baseline.local_maps().front().center_dof;
    const auto component_patches=smooth_patch_component(
        surface,cloud.dofs[static_cast<std::size_t>(component_center)].patch_id);
    const std::set<int> center_component(
        component_patches.begin(),component_patches.end());
    int outside_dof=-1;
    for(int id=0;id<static_cast<int>(cloud.dofs.size());++id) {
        if(center_component.count(cloud.dofs[static_cast<std::size_t>(id)].patch_id)==0) {
            outside_dof=id;
            break;
        }
    }
    require(outside_dof>=0,"wrong-side fixture lacks an outside-component DOF");
    const int outside_patch=cloud.dofs[static_cast<std::size_t>(outside_dof)].patch_id;
    auto wrong_side_value=stencils;
    wrong_side_value[static_cast<std::size_t>(component_center)].value_dofs.back()
        =outside_dof;
    require_throws<std::invalid_argument>([&]{
        (void)build_neumann_edge_augmented_cauchy_3d(
            surface,cloud,h,wrong_side_value);
    },"value DOF "+std::to_string(outside_dof)
        +" patch "+std::to_string(outside_patch));
    auto wrong_side_normal=stencils;
    wrong_side_normal[static_cast<std::size_t>(component_center)].normal_dofs.back()
        =outside_dof;
    require_throws<std::invalid_argument>([&]{
        (void)build_neumann_edge_augmented_cauchy_3d(
            surface,cloud,h,wrong_side_normal);
    },"normal DOF "+std::to_string(outside_dof)
        +" patch "+std::to_string(outside_patch));

    auto repeated=stencils;
    const int repeated_center=baseline.local_maps().front().center_dof;
    const int repeated_value=
        repeated[static_cast<std::size_t>(repeated_center)].value_dofs.back();
    const int repeated_normal=
        repeated[static_cast<std::size_t>(repeated_center)].normal_dofs.back();
    repeated[static_cast<std::size_t>(repeated_center)].value_dofs.assign(
        48,repeated_value);
    repeated[static_cast<std::size_t>(repeated_center)].normal_dofs.assign(
        28,repeated_normal);
    require_throws<std::runtime_error>([&]{
        (void)build_neumann_edge_augmented_cauchy_3d(surface,cloud,h,repeated);
    },"rank-deficient local");

    auto expanded=stencils;
    const int center=baseline.local_maps().front().center_dof;
    int farthest=center;
    for(int id=0;id<static_cast<int>(cloud.dofs.size());++id) {
        if(center_component.count(cloud.dofs[static_cast<std::size_t>(id)].patch_id)==0)
            continue;
        if((cloud.dofs[static_cast<std::size_t>(id)].point
            -cloud.dofs[static_cast<std::size_t>(center)].point).squaredNorm()
           >(cloud.dofs[static_cast<std::size_t>(farthest)].point
            -cloud.dofs[static_cast<std::size_t>(center)].point).squaredNorm())
            farthest=id;
    }
    expanded[static_cast<std::size_t>(center)].value_dofs.back()=farthest;
    const auto expanded_map=build_neumann_edge_augmented_cauchy_3d(
        surface,cloud,h,expanded);
    const auto* expanded_local=find_local_map(expanded_map,center);
    require(expanded_local!=nullptr,"expanded-radius center lost its local map");

    for(const int index:expanded_local->edge_sample_indices) {
        const auto& sample=expanded_map.edge_value_map().samples[
            static_cast<std::size_t>(index)];
        const auto& connection=surface.geometric_connections[
            static_cast<std::size_t>(sample.connection_index)];
        require((center_component.count(connection.first.patch)==1)
                !=(center_component.count(connection.second.patch)==1),
            "expanded radius attached an unrelated connection");
    }

    Eigen::VectorXd values=Eigen::VectorXd::Ones(baseline.surface_size());
    Eigen::VectorXd normals=Eigen::VectorXd::Ones(baseline.surface_size());
    const Eigen::VectorXd edge_values=baseline.edge_values(values,normals);
    Eigen::MatrixXd coefficients=Eigen::MatrixXd::Zero(baseline.surface_size(),16);
    require_throws<std::invalid_argument>([&]{
        (void)baseline.edge_values(values.head(values.size()-1),normals);
    },"value");
    require_throws<std::invalid_argument>([&]{
        (void)baseline.edge_values(values,normals.head(normals.size()-1));
    },"normal");
    require_throws<std::invalid_argument>([&]{
        baseline.overwrite_affected_coefficients(
            values.head(values.size()-1),normals,edge_values,coefficients);
    },"value");
    require_throws<std::invalid_argument>([&]{
        baseline.overwrite_affected_coefficients(
            values,normals.head(normals.size()-1),edge_values,coefficients);
    },"normal");
    require_throws<std::invalid_argument>([&]{
        baseline.overwrite_affected_coefficients(
            values,normals,edge_values.head(edge_values.size()-1),coefficients);
    },"edge");
    Eigen::MatrixXd wrong_rows=Eigen::MatrixXd::Zero(baseline.surface_size()-1,16);
    require_throws<std::invalid_argument>([&]{
        baseline.overwrite_affected_coefficients(
            values,normals,edge_values,wrong_rows);
    },"coefficient");
    Eigen::MatrixXd wrong_columns=Eigen::MatrixXd::Zero(baseline.surface_size(),15);
    require_throws<std::invalid_argument>([&]{
        baseline.overwrite_affected_coefficients(
            values,normals,edge_values,wrong_columns);
    },"coefficient");

    auto duplicate=baseline;
    auto& duplicate_maps=const_cast<std::vector<NeumannEdgeLocalMap3D>&>(
        duplicate.local_maps());
    require(duplicate_maps.front().edge_sample_indices.size()>=2,
        "duplicate-edge mutation lacks samples");
    duplicate_maps.front().edge_sample_indices[1]
        =duplicate_maps.front().edge_sample_indices[0];
    require_throws<std::invalid_argument>([&]{
        kfbim::app3d::detail::validate_neumann_edge_local_map_3d(
            duplicate_maps.front(),duplicate.surface_size(),
            duplicate.edge_sample_count());
    },"duplicate edge sample");
}

SurfaceDofCloud3D restricted_cloud(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& dense)
{
    SurfaceDofCloud3D result; result.expected_area=dense.expected_area;
    for(int patch_id=0;patch_id<static_cast<int>(dense.patches.size());++patch_id) {
        SurfaceDofPatch3D tensor=dense.patches[static_cast<std::size_t>(patch_id)];
        tensor.first_dof=static_cast<int>(result.dofs.size());
        const int wanted=patch_id<6?8:24;
        tensor.nu=wanted;
        tensor.nv=1;
        const auto& patch=surface.patches[static_cast<std::size_t>(patch_id)];
        const double du=(patch.domain_end_u()-patch.domain_start_u())/wanted;
        const double dv=patch.domain_end_v()-patch.domain_start_v();
        for(int i=0;i<wanted;++i) {
            const double u=patch.domain_start_u()+(static_cast<double>(i)+0.5)*du;
            const double v=patch.domain_start_v()+0.5*dv;
            const auto geometry=patch.evaluate_with_derivatives(u,v);
            Eigen::Vector3d normal=geometry.du.cross(geometry.dv);
            const double jacobian=normal.norm();
            normal/=jacobian;
            Eigen::Vector3d tangent1=geometry.du
                -geometry.du.dot(normal)*normal;
            tangent1.normalize();
            const Eigen::Vector3d tangent2=normal.cross(tangent1).normalized();
            result.dofs.push_back({geometry.point,normal,tangent1,tangent2,
                jacobian*du*dv,u,v,patch_id,i,0});
        }
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

template<class Exception, class Function>
void record_required_rejection(std::vector<std::string>& failures,
                               const std::string& label,
                               Function function,
                               const std::string& required_text)
{
    try {
        function();
        failures.push_back(label + ": malformed input was accepted");
    } catch(const Exception& error) {
        if(std::string(error.what()).find(required_text)==std::string::npos)
            failures.push_back(label + ": missing context in '" + error.what() + "'");
    } catch(const std::exception& error) {
        failures.push_back(label + ": wrong exception type: " + error.what());
    }
}

void remove_reciprocal_g1_relation(NativeNurbsSurface3D& surface,
                                   int first_patch,
                                   int second_patch)
{
    int removed=0;
    for(auto& slot:surface.smooth_neighbors[static_cast<std::size_t>(first_patch)]) {
        if(slot && slot->patch==second_patch) { slot.reset(); ++removed; break; }
    }
    for(auto& slot:surface.smooth_neighbors[static_cast<std::size_t>(second_patch)]) {
        if(slot && slot->patch==first_patch) { slot.reset(); ++removed; break; }
    }
    require(removed==2,"reciprocal G1 relation fixture missing");
    auto erase_neighbor=[](std::vector<int>& neighbors,int patch) {
        neighbors.erase(std::remove(neighbors.begin(),neighbors.end(),patch),neighbors.end());
    };
    erase_neighbor(surface.topological_patch_neighbors[static_cast<std::size_t>(first_patch)],
                   second_patch);
    erase_neighbor(surface.topological_patch_neighbors[static_cast<std::size_t>(second_patch)],
                   first_patch);
}
void test_review_rejection_paths()
{
    const auto surface=make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const double h=3.0/32.0;
    const auto cloud=make_native_surface_dofs_3d(surface,h);
    std::vector<std::string> failures;

    auto smooth_asymmetric=surface;
    bool removed_smooth=false;
    for(auto& slot:smooth_asymmetric.smooth_neighbors[1]) {
        if(slot && slot->patch==0) {
            slot.reset();
            removed_smooth=true;
            break;
        }
    }
    require(removed_smooth,"smooth-asymmetry fixture relation missing");
    record_required_rejection<std::invalid_argument>(failures,"asymmetric smooth topology",[&]{
        (void)build_neumann_edge_auxiliary_value_map_3d(smooth_asymmetric,cloud,h);
    },"patch 0");

    auto topological_asymmetric=surface;
    auto& patch_zero_neighbors=topological_asymmetric.topological_patch_neighbors[0];
    patch_zero_neighbors.erase(std::remove(patch_zero_neighbors.begin(),patch_zero_neighbors.end(),1),
                               patch_zero_neighbors.end());
    record_required_rejection<std::invalid_argument>(failures,"asymmetric topological adjacency",[&]{
        (void)build_neumann_edge_auxiliary_value_map_3d(topological_asymmetric,cloud,h);
    },"patch 0");

    auto unrelated_connection=surface;
    const int connection_index=4;
    const auto& unrelated=unrelated_connection.geometric_connections[connection_index];
    require(unrelated.first.patch==6 && unrelated.second.patch==7 && !unrelated.g1,
            "unrelated-connection fixture changed");
    auto erase_neighbor=[](std::vector<int>& neighbors,int patch) {
        neighbors.erase(std::remove(neighbors.begin(),neighbors.end(),patch),neighbors.end());
    };
    erase_neighbor(unrelated_connection.topological_patch_neighbors[6],7);
    erase_neighbor(unrelated_connection.topological_patch_neighbors[7],6);
    record_required_rejection<std::invalid_argument>(failures,"unrelated geometric connection",[&]{
        (void)build_neumann_edge_auxiliary_value_map_3d(unrelated_connection,cloud,h);
    },"connection 4");

    auto non_g1_marked_smooth=surface;
    const auto& non_g1=non_g1_marked_smooth.geometric_connections[connection_index];
    non_g1_marked_smooth.smooth_neighbors[static_cast<std::size_t>(non_g1.first.patch)]
        [static_cast<std::size_t>(non_g1.first.edge)] =
        SmoothPatchNeighbor3D{non_g1.second.patch,non_g1.second.edge,non_g1.reversed};
    non_g1_marked_smooth.smooth_neighbors[static_cast<std::size_t>(non_g1.second.patch)]
        [static_cast<std::size_t>(non_g1.second.edge)] =
        SmoothPatchNeighbor3D{non_g1.first.patch,non_g1.first.edge,non_g1.reversed};
    record_required_rejection<std::invalid_argument>(failures,"non-G1 connection marked smooth",[&]{
        (void)build_neumann_edge_auxiliary_value_map_3d(non_g1_marked_smooth,cloud,h);
    },"connection 4");

    const RigidTransform3D translation(Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero(),
                                       Eigen::Vector3d(0.19,-0.13,0.07));
    const auto translated_surface=transform_native_nurbs_surface_3d(surface,translation);
    const auto translated_cloud=make_native_surface_dofs_3d(translated_surface,h);
    record_required_rejection<std::invalid_argument>(failures,"foreign translated cloud",[&]{
        (void)build_neumann_edge_auxiliary_value_map_3d(surface,translated_cloud,h);
    },"DOF 0 patch 0");

    auto shifted_tensor_parameter=cloud;
    auto& shifted=shifted_tensor_parameter.dofs[0];
    const auto& shifted_patch=surface.patches[0];
    const double cell_width=(shifted_patch.domain_end_u()
        -shifted_patch.domain_start_u())/shifted_tensor_parameter.patches[0].nu;
    shifted.u+=0.1*cell_width;
    const auto shifted_geometry=
        shifted_patch.evaluate_with_derivatives(shifted.u,shifted.v);
    shifted.point=shifted_geometry.point;
    shifted.normal=shifted_geometry.du.cross(shifted_geometry.dv).normalized();
    shifted.tangent1=(shifted_geometry.du
        -shifted_geometry.du.dot(shifted.normal)*shifted.normal).normalized();
    shifted.tangent2=shifted.normal.cross(shifted.tangent1).normalized();
    record_required_rejection<std::invalid_argument>(failures,
        "shifted tensor-center parameters",[&]{
        (void)build_neumann_edge_auxiliary_value_map_3d(
            surface,shifted_tensor_parameter,h);
    },"tensor-center parameters");

    auto wrong_tensor_index=cloud;
    wrong_tensor_index.dofs[0].i=1;
    record_required_rejection<std::invalid_argument>(failures,"wrong tensor index",[&]{
        (void)build_neumann_edge_auxiliary_value_map_3d(surface,wrong_tensor_index,h);
    },"DOF 0 patch 0");

    auto wrong_tensor_name=cloud;
    wrong_tensor_name.patches[0].name="wrong_patch_name";
    record_required_rejection<std::invalid_argument>(failures,"wrong tensor name",[&]{
        (void)build_neumann_edge_auxiliary_value_map_3d(surface,wrong_tensor_name,h);
    },"patch 0");

    auto wrong_normal=cloud;
    wrong_normal.dofs[0].normal=-wrong_normal.dofs[0].normal;
    record_required_rejection<std::invalid_argument>(failures,"wrong physical normal",[&]{
        (void)build_neumann_edge_auxiliary_value_map_3d(surface,wrong_normal,h);
    },"wrong physical normal");

    auto restricted_surface=surface;
    std::size_t chosen=0;
    for(;chosen<restricted_surface.geometric_connections.size();++chosen) {
        const auto& connection=restricted_surface.geometric_connections[chosen];
        if(!connection.g1 && connection.first.patch==6 && connection.second.patch==0) break;
    }
    require(chosen<restricted_surface.geometric_connections.size(),"side-label fixture connection missing");
    restricted_surface.geometric_connections={restricted_surface.geometric_connections[chosen]};
    remove_reciprocal_g1_relation(restricted_surface,0,1);
    const auto small=restricted_cloud(surface,make_native_surface_dofs_3d(surface,0.08));
    record_required_rejection<std::runtime_error>(failures,"insufficient side label",[&]{
        (void)build_neumann_edge_auxiliary_value_map_3d(restricted_surface,small,h);
    },"second-value side");

    if(!failures.empty()) {
        std::string message="review rejection failures";
        for(const auto& failure:failures) message += "\n - " + failure;
        throw std::runtime_error(message);
    }
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
    require_throws<std::overflow_error>([&]{
        (void)build_neumann_edge_auxiliary_value_map_3d(surface,cloud,1.0e-12);
    },"sample count");
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
    remove_reciprocal_g1_relation(restricted_surface,0,1);
    const auto small=restricted_cloud(surface,make_native_surface_dofs_3d(surface,0.08));
    require_throws<std::runtime_error>([&]{(void)build_neumann_edge_auxiliary_value_map_3d(restricted_surface,small,h);},"second-value side");
    const auto degenerate=degenerate_normal_surface(); const auto degenerate_cloud=make_native_surface_dofs_3d(degenerate,0.1);
    require_throws<std::runtime_error>([&]{(void)build_neumann_edge_auxiliary_value_map_3d(degenerate,degenerate_cloud,0.1);},"connection 0 sample 0");
    const auto map=build_neumann_edge_auxiliary_value_map_3d(surface,cloud,h);
    require_throws<std::invalid_argument>([&]{(void)evaluate_neumann_edge_values_3d(map,Eigen::VectorXd::Zero(map.surface_size-1),Eigen::VectorXd::Zero(map.surface_size));},"jump data");
}

} // namespace

int main(int argc,char** argv)
{
    try {
        if(argc==2 && std::string(argv[1])=="--runtime-no-alloc") {
            test_runtime_overwrite_has_no_per_center_heap_path();
            std::cout<<"3D Neumann runtime no-allocation test passed\n";
            return 0;
        }
        if(argc==2 && std::string(argv[1])=="--sample-count-boundaries") {
            test_sample_count_integer_boundaries();
            std::cout<<"3D Neumann sample-count boundary tests passed\n";
            return 0;
        }
        if(argc==2
            && std::string(argv[1])=="--negative-symmetry-guard") {
            test_negative_reduced_symmetry_numerator_is_rejected();
            std::cout<<"3D Neumann negative symmetry guard test passed\n";
            return 0;
        }
        test_l_prism_geometry_topology_reproduction_and_direct_map();
        test_rigid_covariance();
        test_runtime_overwrite_has_no_per_center_heap_path();
        test_exact_distance_selection_boundaries();
        test_geometric_symmetry_certificate();
        test_negative_reduced_symmetry_numerator_is_rejected();
        test_sample_count_integer_boundaries();
        test_local_attachment_groups_and_overwrite();
        test_local_harmonic_reproduction_and_rigid_covariance();
        test_local_map_rejections();
        test_review_rejection_paths();
        test_rejections_and_shared_values();
        std::cout<<"3D Neumann edge-augmented Cauchy tests passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"3D Neumann auxiliary edge-value test failure: "<<error.what()<<'\n';
        return 1;
    }
}