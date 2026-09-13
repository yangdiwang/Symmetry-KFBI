#pragma once
#include "src/support/correction/shared_field_geometry_3d.hpp"
#include <Eigen/SparseCore>
#include <map>
#include <set>
namespace kfbim::app3d {
using SharedFieldSparseMatrix3D=Eigen::SparseMatrix<double,Eigen::RowMajor>;
enum class SharedFieldDerivative3D {Value,Dx,Dy,Dz,Dxx,Dyy,Dzz,Dxy,Dxz,Dyz,Laplacian};
struct SharedFieldSpaceOptions3D {double h=0.,ratio=4.,width=4.;int volume_quadrature_order=3;};
class SharedFieldSpace3D {
public:
 SharedFieldSpace3D(SharedFieldGeometry3D geometry,SharedFieldSpaceOptions3D options);
 int coefficient_count()const{return static_cast<int>(lattice_.size());}
 double spacing()const{return options_.ratio*options_.h;}
 const SharedFieldGeometry3D& geometry()const{return geometry_;}
 const SharedFieldSpaceOptions3D& options()const{return options_;}
 const std::vector<Eigen::Vector3i>& coefficient_lattice()const{return lattice_;}
 const std::vector<Eigen::Vector3i>& active_cells()const{return cells_;}
 const SharedFieldSurfaceSamples3D& surface_samples()const{return surface_;}
 const SharedFieldVolumeSamples3D& volume_samples()const{return volume_;}
 bool contains(const Eigen::Vector3d& point)const;
 std::vector<bool> contains(const std::vector<Eigen::Vector3d>& points)const;
 SharedFieldSparseMatrix3D evaluation_matrix(const std::vector<Eigen::Vector3d>& reference_points,SharedFieldDerivative3D derivative=SharedFieldDerivative3D::Value)const;
private:
 SharedFieldGeometry3D geometry_;
 SharedFieldSpaceOptions3D options_;
 std::vector<Eigen::Vector3i> cells_,lattice_;
 std::set<std::array<int,3>> cell_set_;
 std::map<std::array<int,3>,int> indices_;
 SharedFieldSurfaceSamples3D surface_;
 SharedFieldVolumeSamples3D volume_;
};
Eigen::MatrixXd shared_field_polynomial_3d(const std::vector<Eigen::Vector3d>& points,const Eigen::Vector3i& derivative=Eigen::Vector3i::Zero());
}
