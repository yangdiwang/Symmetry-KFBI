#include "nurbs_surface_closest_point_3d.hpp"
#include "native_nurbs_exact_geometry_3d.hpp"
#include "nurbs_bezier_extraction_3d.hpp"
#include "nurbs_patch_polar_evaluator_3d.hpp"

#include <CGAL/number_utils.h>
#include <Eigen/Cholesky>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace kfbim::geometry3d {
namespace {
constexpr double infinity = std::numeric_limits<double>::infinity();
double down(double x) { return std::nextafter(x, -infinity); }
double up(double x) { return std::nextafter(x, infinity); }
struct Interval { double lo = 0, hi = 0; };
Interval exact(double x) { return {x,x}; }
Interval enclose(const CGAL::Gmpq& x) { const auto b=CGAL::to_interval(x); return {b.first,b.second}; }
Interval add(Interval a, Interval b) {
    const double lo=a.lo+b.lo,hi=a.hi+b.hi;
    if(std::isnan(lo)||std::isnan(hi))return {-infinity,infinity};
    return {down(lo),up(hi)};
}
Interval sub(Interval a, Interval b) {
    const double lo=a.lo-b.hi,hi=a.hi-b.lo;
    if(std::isnan(lo)||std::isnan(hi))return {-infinity,infinity};
    return {down(lo),up(hi)};
}
Interval multiply(Interval a, Interval b) {
    const std::array<double,4> p{{a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi}};
    for(double value:p)if(std::isnan(value))return {-infinity,infinity};
    return {down(*std::min_element(p.begin(),p.end())),up(*std::max_element(p.begin(),p.end()))};
}
Interval divide_positive(Interval a, Interval b) {
    if (!(b.lo>0)) return {-infinity,infinity};
    return multiply(a,{down(1/b.hi),up(1/b.lo)});
}
Interval square(Interval a) {
    const double low = a.lo<=0 && a.hi>=0 ? 0 : std::min(a.lo*a.lo,a.hi*a.hi);
    return {std::max(0.0,down(low)),up(std::max(a.lo*a.lo,a.hi*a.hi))};
}
double norm_upper(const Eigen::Vector3d& vector) {
    Interval sum{};
    for(int k=0;k<3;++k) sum=add(sum,square(exact(vector[k])));
    return up(std::sqrt(std::max(0.0,sum.hi)));
}
using Control = std::array<Interval,4>;
using Net = std::vector<Control>;
NurbsAabb3D net_bounds(const Net& net) {
    NurbsAabb3D b; b.lower.setConstant(infinity); b.upper.setConstant(-infinity);
    for(const auto& h:net) for(int k=0;k<3;++k) {
        const auto c=divide_positive(h[k],h[3]);
        b.lower[k]=std::min(b.lower[k],c.lo); b.upper[k]=std::max(b.upper[k],c.hi);
    }
    return b;
}
double box_distance_lower(const NurbsAabb3D& b,const Eigen::Vector3d& x) {
    Interval sum{};
    for(int k=0;k<3;++k) {
        Interval delta{};
        if(x[k]<b.lower[k]) delta=sub(exact(b.lower[k]),exact(x[k]));
        else if(x[k]>b.upper[k]) delta=sub(exact(x[k]),exact(b.upper[k]));
        sum=add(sum,square(delta));
    }
    return std::max(0.0,down(std::sqrt(std::max(0.0,sum.lo))));
}
double box_radius_upper(const NurbsAabb3D& b,const Eigen::Vector3d& x) {
    Interval sum{};
    for(int k=0;k<3;++k) sum=add(sum,square(sub({b.lower[k],b.upper[k]},exact(x[k]))));
    return up(std::sqrt(std::max(0.0,sum.hi)));
}
// Any direction with norm <=1 gives a valid support-plane distance lower bound.
// Checking its rounded squared norm avoids assuming floating normalization is exact.
Eigen::Vector3d separating_direction(const Eigen::Vector3d& x,const Eigen::Vector3d& witness) {
    Eigen::Vector3d d=x-witness;
    const double n=norm_upper(d);
    if(!(n>0) || !std::isfinite(n)) return Eigen::Vector3d::Zero();
    d *= (1-16*std::numeric_limits<double>::epsilon())/n;
    if(norm_upper(d)>1) d*=0.5;
    return d;
}
double support_distance_lower(const Net& net,const Eigen::Vector3d& x,const Eigen::Vector3d& direction) {
    double bound=infinity;
    for(const auto& h:net) {
        Interval value{};
        for(int k=0;k<3;++k)
            value=add(value,multiply(exact(direction[k]),sub(exact(x[k]),divide_positive(h[k],h[3]))));
        bound=std::min(bound,value.lo);
    }
    return std::max(0.0,bound);
}
struct Cell {
    std::size_t base=0;
    int depth=0;
    double u0=0,u1=0,v0=0,v1=0;
    Net controls;
    Net derivative_u,derivative_v;
    NurbsAabb3D bounds;
};
struct Frontier { int bvh=-1; std::size_t cell=0; double lower=0; };
bool farther(const Frontier& a,const Frontier& b) { return a.lower>b.lower; }
struct BvhNode { NurbsAabb3D bounds; int left=-1,right=-1,base=-1; };
struct Base {
    ExactNativeBezierElement3D exact;
    Net controls,derivative_u,derivative_v;
    NurbsAabb3D bounds;
};
// A global minimum in the interior of a base span must have zero first
// derivatives. Bound their signs using homogeneous derivative nets extracted
// exactly at the base, then restricted alongside the position net. Subtracting
// nearly equal interval controls of a deep child instead causes a sqrt(eps)
// localization floor: the interval width does not shrink like that difference.
// Positive denominators do not affect signs.
// Artificial subdivision boundaries are not constraints of the original
// minimization problem; original base boundaries are conservatively retained.
bool excludes_stationary_minimum(const Cell& cell,const ExactNativeBezierElement3D& base,
                                const Eigen::Vector3d& point) {
    Control h;
    for(auto& interval:h)interval={infinity,-infinity};
    for(const auto& control:cell.controls)for(int k=0;k<4;++k) {
        h[k].lo=std::min(h[k].lo,control[k].lo);h[k].hi=std::max(h[k].hi,control[k].hi);
    }
    for(int axis=0;axis<2;++axis) {
        const int degree=axis==0?base.degree_u:base.degree_v;
        if(degree==0)continue;
        Control derivative;
        for(auto& interval:derivative)interval={infinity,-infinity};
        const auto& derivative_net=axis==0?cell.derivative_u:cell.derivative_v;
        for(const auto& control:derivative_net)for(int k=0;k<4;++k) {
            derivative[k].lo=std::min(derivative[k].lo,control[k].lo);
            derivative[k].hi=std::max(derivative[k].hi,control[k].hi);
        }
        Interval gradient{};
        for(int k=0;k<3;++k) {
            const auto residual=sub(h[k],multiply(exact(point[k]),h[3]));
            const auto tangent=sub(multiply(derivative[k],h[3]),multiply(h[k],derivative[3]));
            gradient=add(gradient,multiply(residual,tangent));
        }
        const bool touches_lower=axis==0?cell.u0==base.parameter_u0:cell.v0==base.parameter_v0;
        const bool touches_upper=axis==0?cell.u1==base.parameter_u1:cell.v1==base.parameter_v1;
        if((gradient.lo>0&&!touches_lower)||(gradient.hi<0&&!touches_upper))return true;
    }
    return false;
}
void split_net(const Net& controls,int p,int q,bool along_u,Net& first,Net& second,std::vector<Control>& work) {
    const int degree=along_u?p:q, lines=along_u?q+1:p+1;
    first.resize(controls.size()); second.resize(controls.size());
    work.resize(static_cast<std::size_t>(degree)+1);
    auto offset=[q,along_u](int line,int index) { return along_u?index*(q+1)+line:line*(q+1)+index; };
    for(int line=0;line<lines;++line) {
        for(int i=0;i<=degree;++i) work[i]=controls[offset(line,i)];
        first[offset(line,0)]=work[0]; second[offset(line,degree)]=work[degree];
        for(int level=1;level<=degree;++level) {
            for(int i=0;i<=degree-level;++i) for(int k=0;k<4;++k)
                work[i][k]=add(multiply(work[i][k],exact(0.5)),
                               multiply(work[i+1][k],exact(0.5)));
            first[offset(line,level)]=work[0];
            second[offset(line,degree-level)]=work[degree-level];
        }
    }
}
}

struct NurbsSurfaceClosestPointWorkspace3D::Impl {
    NurbsPolarEvaluationWorkspace3D polar;
    std::vector<Cell> cells;
    std::size_t used=0;
    std::vector<Frontier> queue,terminal;
    std::vector<Control> split_scratch;
    std::vector<std::pair<double,std::size_t>> seeds;
};
NurbsSurfaceClosestPointWorkspace3D::NurbsSurfaceClosestPointWorkspace3D():impl_(std::make_unique<Impl>()){}
NurbsSurfaceClosestPointWorkspace3D::~NurbsSurfaceClosestPointWorkspace3D()=default;
NurbsSurfaceClosestPointWorkspace3D::NurbsSurfaceClosestPointWorkspace3D(NurbsSurfaceClosestPointWorkspace3D&&) noexcept=default;
NurbsSurfaceClosestPointWorkspace3D& NurbsSurfaceClosestPointWorkspace3D::operator=(NurbsSurfaceClosestPointWorkspace3D&&) noexcept=default;

struct NurbsSurfaceClosestPointIndex3D::Impl {
    NurbsSurfaceModel3D model;
    std::vector<Base> bases;
    std::vector<NurbsPatchPolarEvaluator3D> polar;
    std::vector<BvhNode> bvh;
    double scale=1;
    bool supported=false;
    explicit Impl(const NurbsSurfaceModel3D& input):model(input) {
        std::vector<ExactNativeBezierElement3D> extracted;
        try { extracted=extract_exact_native_bezier_elements_3d(model); }
        catch(const std::invalid_argument&) { return; }
        auto approximate=extract_rational_bezier_elements_3d(model);
        std::vector<std::vector<RationalBezierElement3D>> grouped(model.num_patches());
        for(auto& e:approximate) grouped[e.patch_index].push_back(std::move(e));
        for(int p=0;p<model.num_patches();++p) polar.emplace_back(model.patch(p),std::move(grouped[p]));
        for(auto& e:extracted) {
            Base base; base.exact=std::move(e);
            for(const auto& h:base.exact.controls) {
                Control c; for(int k=0;k<4;++k)c[k]=enclose(h[k]);
                if(!(c[3].lo>0)) return;
                base.controls.push_back(c);
            }
            for(int axis=0;axis<2;++axis) {
                const int degree=axis==0?base.exact.degree_u:base.exact.degree_v;
                if(degree==0)continue;
                const CGAL::Gmpq span=axis==0
                    ?CGAL::Gmpq(base.exact.parameter_u1)-CGAL::Gmpq(base.exact.parameter_u0)
                    :CGAL::Gmpq(base.exact.parameter_v1)-CGAL::Gmpq(base.exact.parameter_v0);
                const CGAL::Gmpq scale=CGAL::Gmpq(degree)/span;
                auto& derivative_net=axis==0?base.derivative_u:base.derivative_v;
                const int nu=base.exact.degree_u+1-(axis==0?1:0);
                const int nv=base.exact.degree_v+1-(axis==1?1:0);
                for(int i=0;i<nu;++i)for(int j=0;j<nv;++j) {
                    const int offset=i*(base.exact.degree_v+1)+j;
                    const int adjacent=offset+(axis==0?base.exact.degree_v+1:1);
                    Control derivative;
                    for(int k=0;k<4;++k)derivative[k]=enclose(
                        scale*(base.exact.controls[adjacent][k]-base.exact.controls[offset][k]));
                    derivative_net.push_back(derivative);
                }
            }
            base.bounds=net_bounds(base.controls);
            if(!base.bounds.lower.allFinite() || !base.bounds.upper.allFinite())return;
            bases.push_back(std::move(base));
        }
        if(bases.empty())return;
        std::vector<int> order(bases.size()); std::iota(order.begin(),order.end(),0);
        bvh.reserve(2*bases.size()); build(order,0,static_cast<int>(order.size()));
        scale=std::max(1.0,bvh.front().bounds.diameter()); supported=std::isfinite(scale);
    }
    int build(std::vector<int>& order,int begin,int end) {
        BvhNode node; node.bounds=bases[order[begin]].bounds;
        for(int i=begin+1;i<end;++i) {
            node.bounds.lower=node.bounds.lower.cwiseMin(bases[order[i]].bounds.lower);
            node.bounds.upper=node.bounds.upper.cwiseMax(bases[order[i]].bounds.upper);
        }
        const int id=static_cast<int>(bvh.size()); bvh.push_back(node);
        if(end-begin==1){bvh[id].base=order[begin];return id;}
        Eigen::Index axis=0;(node.bounds.upper-node.bounds.lower).maxCoeff(&axis);
        const int middle=begin+(end-begin)/2;
        std::nth_element(order.begin()+begin,order.begin()+middle,order.begin()+end,[&](int a,int b){
            return bases[a].bounds.lower[axis]+bases[a].bounds.upper[axis]
                 < bases[b].bounds.lower[axis]+bases[b].bounds.upper[axis]; });
        const int left=build(order,begin,middle),right=build(order,middle,end);
        bvh[id].left=left;bvh[id].right=right;return id;
    }
};
NurbsSurfaceClosestPointIndex3D::NurbsSurfaceClosestPointIndex3D(const NurbsSurfaceModel3D& model)
    :impl_(std::make_shared<Impl>(model)){}

NurbsSurfaceClosestPointResult3D NurbsSurfaceClosestPointIndex3D::query(
    const Eigen::Vector3d& point,const NurbsSurfaceClosestPointOptions3D& options,
    NurbsSurfaceClosestPointWorkspace3D& workspace) const {
    if(!point.allFinite())throw std::invalid_argument("Closest-point input must be finite");
    for(double value:{options.distance_tolerance,options.localization_tolerance,
                      options.tie_distance_tolerance,options.separation_tolerance})
        if(!std::isfinite(value)||value<0)throw std::invalid_argument("Closest-point tolerances must be finite and nonnegative");
    if(!options.max_boxes || options.max_boxes>1000000 || options.max_subdivision_depth<0
       || options.max_subdivision_depth>128 || options.max_local_iterations<1 || options.max_local_iterations>1000)
        throw std::invalid_argument("Invalid closest-point work budget");
    NurbsSurfaceClosestPointResult3D result;
    if(!impl_->supported)return result;
    if(!workspace.impl_)workspace.impl_=std::make_unique<NurbsSurfaceClosestPointWorkspace3D::Impl>();
    auto& work=*workspace.impl_; const auto& data=*impl_;
    work.used=0;work.queue.clear();work.terminal.clear();
    const std::size_t capacity=2*options.max_boxes+data.bases.size()+1;
    if(work.cells.size()<capacity)work.cells.resize(capacity);
    const double gap=options.distance_tolerance>0?options.distance_tolerance:1e-9*data.scale;
    const double localization=options.localization_tolerance>0?options.localization_tolerance:1e-7*data.scale;
    const double tie=options.tie_distance_tolerance>0?options.tie_distance_tolerance:gap;
    const double separation=options.separation_tolerance>0?options.separation_tolerance:1e-6*data.scale;
    struct Witness { NurbsSurfaceClosestPointCandidate3D candidate; double rounding_radius=0; };
    std::vector<Witness> witnesses;
    double best_rounding_radius=0;
    auto record=[&](std::size_t base_index,double u,double v,bool record_owner) {
        const auto& base=data.bases[base_index].exact;
        const auto h=evaluate_native_bezier_homogeneous(base,u,v);
        CGAL::Gmpq squared_distance(0);
        Interval rounding_squared{};
        NurbsSurfaceClosestPointCandidate3D candidate;
        candidate.patch_index=base.patch_index;candidate.component=base.component;candidate.u=u;candidate.v=v;
        for(int k=0;k<3;++k) {
            const auto coordinate=h[k]/h[3]; candidate.point[k]=CGAL::to_double(coordinate);
            rounding_squared=add(rounding_squared,square(sub(enclose(coordinate),exact(candidate.point[k]))));
            const auto delta=coordinate-CGAL::Gmpq(point[k]); squared_distance+=delta*delta;
        }
        candidate.distance_upper=up(std::sqrt(std::max(0.0,enclose(squared_distance).hi)));
        ++result.stats.point_evaluations;
        const auto jet=data.polar[base.patch_index].evaluate_with_derivatives(data.model.patch(base.patch_index),u,v,work.polar);
        const Eigen::Vector3d cross=jet.du.cross(jet.dv); const double normal_size=cross.norm();
        candidate.normal_valid=cross.allFinite() && std::isfinite(normal_size) && normal_size>0;
        if(candidate.normal_valid)candidate.normal=cross/normal_size;
        const double rounding_radius=up(std::sqrt(std::max(0.0,rounding_squared.hi)));
        if(candidate.distance_upper<result.distance_upper) {
            static_cast<NurbsSurfaceClosestPointCandidate3D&>(result)=candidate;
            best_rounding_radius=rounding_radius;
        }
        // Midpoint witnesses and minimizers constrained to artificial child
        // boxes are upper bounds, not evidence of additional nearest owners.
        if(record_owner && candidate.distance_upper<=up(result.distance_upper+tie)) {
            bool repeated=false;
            for(auto& old:witnesses)if(old.candidate.patch_index==candidate.patch_index && (old.candidate.point-candidate.point).norm()<=separation) {
                if(candidate.distance_upper<old.candidate.distance_upper)old={candidate,rounding_radius};
                repeated=true;break;
            }
            if(!repeated)witnesses.push_back({candidate,rounding_radius});
        }
    };
    auto local_solve=[&](std::size_t base_index,double u0,double u1,double v0,double v1,bool record_owner) {
        const auto& base=data.bases[base_index].exact;
        const auto& patch=data.model.patch(base.patch_index);
        Eigen::Vector2d parameter(0.5,0.5);
        auto eval=[&](const Eigen::Vector2d& t) {
            ++result.stats.point_evaluations;
            return data.polar[base.patch_index].evaluate_with_derivatives(
                patch,std::clamp(u0+t.x()*(u1-u0),u0,u1),std::clamp(v0+t.y()*(v1-v0),v0,v1),work.polar);
        };
        auto current=eval(parameter);
        for(int iteration=0;iteration<options.max_local_iterations;++iteration) {
            ++result.stats.local_iterations;
            Eigen::Matrix<double,3,2> j;j.col(0)=(u1-u0)*current.du;j.col(1)=(v1-v0)*current.dv;
            const Eigen::Vector3d residual=current.point-point;
            Eigen::Vector2d gradient=j.transpose()*residual;
            Eigen::Matrix2d normal=j.transpose()*j;
            normal.diagonal().array()+=64*std::numeric_limits<double>::epsilon()*std::max(1.0,normal.trace());
            for(int k=0;k<2;++k)if((parameter[k]<=0 && gradient[k]>=0)||(parameter[k]>=1 && gradient[k]<=0)) {
                gradient[k]=0;normal.row(k).setZero();normal.col(k).setZero();normal(k,k)=1;
            }
            Eigen::Vector2d step=-normal.ldlt().solve(gradient);
            if(!step.allFinite())break;
            if(step.norm()<=1e-14)break;
            bool accepted=false;
            for(int search=0;search<20;++search) {
                const Eigen::Vector2d trial=(parameter+std::ldexp(1.0,-search)*step).cwiseMax(0.0).cwiseMin(1.0);
                const auto jet=eval(trial);
                if((jet.point-point).squaredNorm()<residual.squaredNorm()) {
                    parameter=trial;current=jet;accepted=true;break;
                }
            }
            if(!accepted)break;
        }
        record(base_index,std::clamp(u0+parameter.x()*(u1-u0),u0,u1),std::clamp(v0+parameter.y()*(v1-v0),v0,v1),record_owner);
    };
    // Local iterations provide feasible witnesses only, never a global status.
    work.seeds.resize(data.bases.size());
    for(std::size_t i=0;i<data.bases.size();++i)
        work.seeds[i]={box_distance_lower(data.bases[i].bounds,point),i};
    std::sort(work.seeds.begin(),work.seeds.end());
    for(const auto& seed:work.seeds) {
        // A conservative geometric lower bound, not a local solver conclusion,
        // proves this base cannot improve the incumbent or share its minimum.
        if(seed.first>result.distance_upper)continue;
        const auto& b=data.bases[seed.second].exact;
        local_solve(seed.second,b.parameter_u0,b.parameter_u1,b.parameter_v0,b.parameter_v1,true);
    }
    auto push=[&](Frontier entry) {
        if(entry.lower>result.distance_upper)return;
        work.queue.push_back(entry);std::push_heap(work.queue.begin(),work.queue.end(),farther);
    };
    auto bounds_of=[&](const Frontier& f)->const NurbsAabb3D& {
        return f.bvh>=0?data.bvh[f.bvh].bounds:work.cells[f.cell].bounds;
    };
    auto update_enclosure=[&]() {
        double lower=result.distance_upper,radius=0;
        for(const auto* entries:{&work.queue,&work.terminal})for(const auto& entry:*entries) {
            if(entry.lower>result.distance_upper)continue;
            lower=std::min(lower,entry.lower);
            radius=std::max(radius,box_radius_upper(bounds_of(entry),result.point));
        }
        result.distance_lower=std::max(0.0,std::min(lower,result.distance_upper));
        result.localization_radius_upper=radius;
        result.localized=std::isfinite(radius)&&radius<=localization;
    };
    push({0,0,box_distance_lower(data.bvh.front().bounds,point)});
    while(!work.queue.empty() && result.stats.boxes_visited<options.max_boxes) {
        update_enclosure();
        if(result.distance_upper-result.distance_lower<=gap && result.localized)break;
        Frontier entry;
        if(result.distance_upper-result.distance_lower<=gap) {
            // Once distance is bounded, repeatedly refining the smallest lower
            // bound can starve distant boxes that control the location bound.
            // Eliminate the widest remaining location ambiguity first.
            const auto selected=std::max_element(work.queue.begin(),work.queue.end(),
                [&](const Frontier& a,const Frontier& b) {
                    const double ar=a.lower<=result.distance_upper?box_radius_upper(bounds_of(a),result.point):-1;
                    const double br=b.lower<=result.distance_upper?box_radius_upper(bounds_of(b),result.point):-1;
                    return ar<br;
                });
            entry=*selected;*selected=work.queue.back();work.queue.pop_back();
            std::make_heap(work.queue.begin(),work.queue.end(),farther);
        } else {
            std::pop_heap(work.queue.begin(),work.queue.end(),farther);
            entry=work.queue.back();work.queue.pop_back();
        }
        if(entry.lower>result.distance_upper)continue;
        ++result.stats.boxes_visited;
        if(entry.bvh>=0) {
            const auto& node=data.bvh[entry.bvh];
            if(node.base<0) {
                for(int child:{node.left,node.right})push({child,0,box_distance_lower(data.bvh[child].bounds,point)});
            } else {
                const auto& base=data.bases[node.base];const std::size_t id=work.used++;
                auto& cell=work.cells[id];cell.base=node.base;cell.depth=0;
                cell.u0=base.exact.parameter_u0;cell.u1=base.exact.parameter_u1;
                cell.v0=base.exact.parameter_v0;cell.v1=base.exact.parameter_v1;
                cell.controls=base.controls;cell.bounds=base.bounds;
                cell.derivative_u=base.derivative_u;cell.derivative_v=base.derivative_v;
                const double lower=std::max(entry.lower,support_distance_lower(cell.controls,point,separating_direction(point,result.point)));
                push({-1,id,lower});
            }
            continue;
        }
        auto& parent=work.cells[entry.cell];const auto& base=data.bases[parent.base].exact;
        if(excludes_stationary_minimum(parent,base,point))continue;
        const double u_mid=0.5*parent.u0+0.5*parent.u1,v_mid=0.5*parent.v0+0.5*parent.v1;
        if(parent.depth>=options.max_subdivision_depth
           || !(u_mid>parent.u0&&u_mid<parent.u1&&v_mid>parent.v0&&v_mid<parent.v1)) {
            work.terminal.push_back(entry);continue;
        }
        // Refine a witness occasionally; exact midpoint evaluation is also a
        // feasible upper bound even if local Gauss-Newton missed another basin.
        if(parent.depth<5)local_solve(parent.base,parent.u0,parent.u1,parent.v0,parent.v1,false);
        else record(parent.base,u_mid,v_mid,false);
        if(entry.lower>result.distance_upper)continue;
        const std::size_t first_id=work.used++,second_id=work.used++;
        auto& first=work.cells[first_id];auto& second=work.cells[second_id];
        first.base=second.base=parent.base;first.depth=second.depth=parent.depth+1;
        first.u0=second.u0=parent.u0;first.u1=second.u1=parent.u1;
        first.v0=second.v0=parent.v0;first.v1=second.v1=parent.v1;
        const bool along_u=(parent.u1-parent.u0)/(base.parameter_u1-base.parameter_u0)
                         >=(parent.v1-parent.v0)/(base.parameter_v1-base.parameter_v0);
        if(along_u){first.u1=u_mid;second.u0=u_mid;}else{first.v1=v_mid;second.v0=v_mid;}
        split_net(parent.controls,base.degree_u,base.degree_v,along_u,
                  first.controls,second.controls,work.split_scratch);
        if(base.degree_u>0)split_net(parent.derivative_u,base.degree_u-1,base.degree_v,along_u,
                                    first.derivative_u,second.derivative_u,work.split_scratch);
        else {first.derivative_u.clear();second.derivative_u.clear();}
        if(base.degree_v>0)split_net(parent.derivative_v,base.degree_u,base.degree_v-1,along_u,
                                    first.derivative_v,second.derivative_v,work.split_scratch);
        else {first.derivative_v.clear();second.derivative_v.clear();}
        ++result.stats.subdivisions;
        const auto direction=separating_direction(point,result.point);
        for(std::size_t child:{first_id,second_id}) {
            auto& cell=work.cells[child];cell.bounds=net_bounds(cell.controls);
            const double lower=std::max({entry.lower,box_distance_lower(cell.bounds,point),
                                        support_distance_lower(cell.controls,point,direction)});
            push({-1,child,lower});
        }
    }
    update_enclosure();
    result.status=result.distance_upper-result.distance_lower<=gap
        ?NurbsSurfaceClosestPointStatus3D::Bounded:NurbsSurfaceClosestPointStatus3D::BudgetExceeded;
    // The incumbent remains a reported feasible candidate, even when first
    // discovered through a subdivided cell rather than a whole-base solve.
    witnesses.push_back({static_cast<const NurbsSurfaceClosestPointCandidate3D&>(result),best_rounding_radius});
    for(const auto& witness:witnesses) {
        const auto& candidate=witness.candidate;
        if(candidate.distance_upper>up(result.distance_upper+tie))continue;
        if(result.localized) {
            NurbsAabb3D location;location.lower=location.upper=candidate.point;
            if(box_distance_lower(location,result.point)
               >up(result.localization_radius_upper+witness.rounding_radius))continue;
        }
        const bool repeated=std::any_of(result.candidates.begin(),result.candidates.end(),[&](const auto& old) {
            return old.patch_index==candidate.patch_index && (old.point-candidate.point).norm()<=separation;
        });
        if(!repeated)result.candidates.push_back(candidate);
    }
    for(std::size_t i=0;i<result.candidates.size();++i)for(std::size_t j=0;j<i;++j)
        if((result.candidates[i].point-result.candidates[j].point).norm()>separation)
            result.separated_near_ties=true;
    return result;
}
} // namespace kfbim::geometry3d
