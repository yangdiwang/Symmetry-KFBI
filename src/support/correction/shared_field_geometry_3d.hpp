#pragma once
#include <Eigen/Dense>
#include <array>
#include <vector>
namespace kfbim::app3d {
struct Trace93Case3D;
struct SharedFieldRectangle3D {
 int axis=0,patch=-1;
 double value=0;
 std::array<int,2> axes{{1,2}};
 std::array<Eigen::Vector2d,2> limits;
 Eigen::Vector3d normal=Eigen::Vector3d::Zero();
 Eigen::Vector3d analysis_origin=Eigen::Vector3d::Zero(),analysis_du=Eigen::Vector3d::Zero(),analysis_dv=Eigen::Vector3d::Zero();
};
struct SharedFieldGeometry3D {
 Eigen::Matrix3d rotation=Eigen::Matrix3d::Identity();
 Eigen::Vector3d translation=Eigen::Vector3d::Zero();
 Eigen::Vector3d reference_half_extent=Eigen::Vector3d::Zero();
 std::vector<SharedFieldRectangle3D> rectangles;
 Eigen::Vector3d to_reference(const Eigen::Vector3d& x)const{return rotation.transpose()*(x-translation);}
 Eigen::Vector3d to_world(const Eigen::Vector3d& x)const{return rotation*x+translation;}
 Eigen::Vector3d normal_to_reference(const Eigen::Vector3d& n)const{return rotation.transpose()*n;}
 Eigen::Vector3d normal_to_world(const Eigen::Vector3d& n)const{return rotation*n;}
 double distance(const Eigen::Vector3d& reference_point)const;
};
struct SharedFieldSurfaceSamples3D {
 std::vector<Eigen::Vector3d> reference_points,world_points,reference_normals,world_normals;
 Eigen::VectorXd weights;
 std::vector<int> patches;
 std::vector<Eigen::Vector2d> analysis_uv;
};
struct SharedFieldVolumeSamples3D {
 std::vector<Eigen::Vector3d> reference_points;
 Eigen::VectorXd weights;
};
SharedFieldGeometry3D make_trace93_shared_field_geometry_3d(const Trace93Case3D& problem);
SharedFieldSurfaceSamples3D sample_shared_field_surface_3d(const SharedFieldGeometry3D& geometry,double spacing);
}
