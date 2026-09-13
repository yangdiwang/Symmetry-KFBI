#pragma once
#include "src/support/density/trace93_density_layout_3d.hpp"
#include <Eigen/Cholesky>
#include <string>
namespace kfbim::app3d {
class IAffineTraceCoordinates3D {
public:
    virtual ~IAffineTraceCoordinates3D()=default;
    virtual int raw_coefficient_count()const=0;
    virtual int trace_count()const=0;
    virtual int reduced_size()const=0;
    virtual const std::string& density_layout_id()const=0;
    virtual const std::string& trace_layout_id()const=0;
    virtual const Eigen::VectorXd& particular()const=0;
    virtual Eigen::VectorXd lift_homogeneous(const Eigen::VectorXd&)const=0;
    virtual Eigen::VectorXd project(const Eigen::VectorXd&)const=0;
};
// Stable fingerprints include numerical coordinate maps and ordered analysis
// trace positions, normals and parameters, never object addresses.
std::string trace93_density_layout_id_3d(const Trace93DensityLayout3D&);
std::string trace93_trace_layout_id_3d(const Trace93DensityLayout3D&);
class Trace93AffineTraceCoordinates3D final:public IAffineTraceCoordinates3D {
public:
    explicit Trace93AffineTraceCoordinates3D(const Trace93DensityLayout3D&,
        std::string density_layout_id={},std::string trace_layout_id={});
    int raw_coefficient_count()const override{return static_cast<int>(Z_.rows());}
    int trace_count()const override{return static_cast<int>(B_.rows());}
    int reduced_size()const override{return static_cast<int>(Z_.cols());}
    const std::string& density_layout_id()const override{return density_id_;}
    const std::string& trace_layout_id()const override{return trace_id_;}
    const Eigen::VectorXd& particular()const override{return particular_;}
    Eigen::VectorXd lift_homogeneous(const Eigen::VectorXd&)const override;
    Eigen::VectorXd project(const Eigen::VectorXd&)const override;
    double gram_regularization()const{return regularization_;}
    double pivot_ratio()const{return pivot_ratio_;}
    double setup_seconds()const{return setup_seconds_;}
private:
    Trace93DensityLayout3D::Sparse B_,edge_trace_,vertex_trace_;
    Eigen::MatrixXd Z_,Ber_,Bvr_;
    Eigen::VectorXd particular_,weights_,edge_weights_,vertex_weights_;
    Eigen::LLT<Eigen::MatrixXd> chol_;
    std::string density_id_,trace_id_;
    double regularization_=0,pivot_ratio_=0,setup_seconds_=0;
};
}
