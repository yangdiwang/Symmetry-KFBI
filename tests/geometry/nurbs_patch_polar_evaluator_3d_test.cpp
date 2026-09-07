#include "src/geometry/nurbs_patch_polar_evaluator_3d.hpp"
#include "src/geometry/nurbs_bezier_extraction_3d.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace {
thread_local bool count_allocations = false;
thread_local std::size_t allocations = 0;
void note_allocation() { if (count_allocations) ++allocations; }
}
void* operator new(std::size_t n) {
    note_allocation();
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#if defined(__cpp_aligned_new)
void* operator new(std::size_t n, std::align_val_t alignment) {
    note_allocation();
    void* p = nullptr;
#ifdef _MSC_VER
    p = _aligned_malloc(n ? n : 1, static_cast<std::size_t>(alignment));
#else
    if (posix_memalign(&p, static_cast<std::size_t>(alignment), n ? n : 1)) p = nullptr;
#endif
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* p, std::align_val_t) noexcept {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    std::free(p);
#endif
}
void operator delete[](void* p, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete(void* p, std::size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete[](void* p, std::size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }
#endif

namespace {
using namespace kfbim::geometry3d;
using kfbim::geometry::NurbsBasis1D;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
NurbsSurfacePatch3D make_patch(int p, int q, int multiplicity = 0,
                             double tolerance = 1e-12, bool tiny = false) {
    std::vector<double> u(p + 1, -2.0), v(q + 1, 3.0);
    if (multiplicity) {
        u.insert(u.end(), multiplicity, 0.125);
        v.insert(v.end(), std::min(multiplicity, q), 4.0);
    }
    if (tiny) { u.push_back(0.5); u.push_back(0.5 + tolerance * 0.5); }
    u.insert(u.end(), p + 1, 2.0); v.insert(v.end(), q + 1, 6.0);
    NurbsBasis1D bu(p, u, tolerance), bv(q, v, tolerance);
    std::vector<std::vector<Eigen::Vector3d>> points(bu.num_basis_functions(),
        std::vector<Eigen::Vector3d>(bv.num_basis_functions()));
    std::vector<std::vector<double>> weights(points.size(), std::vector<double>(points.front().size()));
    for (std::size_t i = 0; i < points.size(); ++i) for (std::size_t j = 0; j < points[i].size(); ++j) {
        const double x = static_cast<double>(i), y = static_cast<double>(j);
        points[i][j] = Eigen::Vector3d(std::sin(x + y), x - 0.3 * y, 0.17*x*x + 0.13*y*y);
        weights[i][j] = 0.7 + 0.11 * ((i + 3*j) % 7);
    }
    return NurbsSurfacePatch3D(std::move(bu), std::move(bv), std::move(points), std::move(weights));
}
std::vector<RationalBezierElement3D> extract(const NurbsSurfacePatch3D& patch) {
    return extract_rational_bezier_elements_3d(NurbsSurfaceModel3D({patch}, {0}, {}));
}
void compare(const NurbsSurfacePatch3D& patch, const NurbsPatchPolarEvaluator3D& evaluator,
             NurbsPolarEvaluationWorkspace3D& workspace, double u, double v) {
    const auto expected = patch.evaluate_with_derivatives(u, v);
    const auto actual = evaluator.evaluate_with_derivatives(patch, u, v, workspace);
    require((expected.point - actual.point).norm() <= 4e-12 * (1 + expected.point.norm()), "polar point mismatch");
    require((expected.du - actual.du).norm() <= 2e-10 * (1 + expected.du.norm()), "polar du mismatch");
    require((expected.dv - actual.dv).norm() <= 2e-10 * (1 + expected.dv.norm()), "polar dv mismatch");
}
void test_patch(int p, int q, int multiplicity = 0) {
    const auto patch = make_patch(p, q, multiplicity);
    const NurbsPatchPolarEvaluator3D evaluator(patch, extract(patch));
    NurbsPolarEvaluationWorkspace3D workspace;
    std::vector<double> us = {-100, -2, -2 + 0.5e-12, -2 + 2e-12, -1.41, -0.7, 0.125, 0.3, 1.5, 2 - 0.5e-12, 2, 100};
    std::vector<double> vs = {-100, 3, 3 + 0.5e-12, 3.25, 4, 4.6, 6 - 0.5e-12, 6, 100};
    for (double x : {-2., 0.125, 2.}) {
        us.push_back(std::nextafter(x, -std::numeric_limits<double>::infinity()));
        us.push_back(std::nextafter(x, std::numeric_limits<double>::infinity()));
    }
    for (double y : {3., 4., 6.}) {
        vs.push_back(std::nextafter(y, -std::numeric_limits<double>::infinity()));
        vs.push_back(std::nextafter(y, std::numeric_limits<double>::infinity()));
    }
    for (double u : us) for (double v : vs) compare(patch, evaluator, workspace, u, v);
    // A fresh common-degree workspace must not allocate; high degree is warmed once.
    NurbsPolarEvaluationWorkspace3D scratch;
    if (p > NurbsPolarEvaluationWorkspace3D::fixed_degree || q > NurbsPolarEvaluationWorkspace3D::fixed_degree)
        (void)evaluator.evaluate_with_derivatives(patch, -0.5, 4.3, scratch);
    allocations = 0; count_allocations = true;
    double sum = 0;
    for (int i = 0; i < 40; ++i)
        sum += evaluator.evaluate_with_derivatives(patch, -1.0 + i * 0.05, 4.3, scratch).point.x();
    count_allocations = false;
    require(std::isfinite(sum), "nonfinite allocation-test sum");
    require(allocations == 0, "polar evaluation allocated heap storage");
}
void test_endpoint_absorption() {
    const auto patch = make_patch(3, 2, 0, 1e-4);
    NurbsPatchPolarEvaluator3D evaluator(patch, extract(patch));
    NurbsPolarEvaluationWorkspace3D workspace;
    for (double u : {-2.0 + 0.5e-4, 2.0 - 0.5e-4})
        for (double v : {3.0 + 0.5e-4, 6.0 - 0.5e-4}) compare(patch, evaluator, workspace, u, v);
}
void test_fallbacks() {
    const auto patch = make_patch(2, 1, 0, 1e-8, true);
    NurbsPatchPolarEvaluator3D evaluator(patch, extract(patch));
    NurbsPolarEvaluationWorkspace3D workspace;
    // The reference may reject a tiny span because its basis divisions zero out.
    const double u = 0.5 + 0.25e-8;
    bool reference_threw = false, polar_threw = false;
    try { (void)patch.evaluate_with_derivatives(u, 4.5); } catch (const std::exception&) { reference_threw = true; }
    try { (void)evaluator.evaluate_with_derivatives(patch, u, 4.5, workspace); } catch (const std::exception&) { polar_threw = true; }
    require(reference_threw == polar_threw, "tiny-span fallback exception mismatch");
    if (!reference_threw) compare(patch, evaluator, workspace, u, 4.5);
    NurbsPatchPolarEvaluator3D empty(patch, {});
    compare(patch, empty, workspace, -1.0, 4.5);
    bool rejected = false;
    try { (void)evaluator.evaluate_with_derivatives(patch, std::numeric_limits<double>::infinity(), 4.5, workspace); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "nonfinite input must be rejected");

    const auto ordinary = make_patch(2, 3);
    auto damaged = extract(ordinary);
    damaged.front().homogeneous_controls.front().x() = std::numeric_limits<double>::infinity();
    NurbsPatchPolarEvaluator3D unsafe(ordinary, std::move(damaged));
    compare(ordinary, unsafe, workspace, 0.3, 4.5);

    auto duplicated = extract(ordinary);
    duplicated.push_back(duplicated.front());
    NurbsPatchPolarEvaluator3D duplicate(ordinary, std::move(duplicated));
    compare(ordinary, duplicate, workspace, 0.3, 4.5);

    // A p+1 internal multiplicity is deliberately unsupported by the original
    // basis; the largest valid repeated internal knot is the C0 case above.
    rejected = false;
    try { (void)NurbsBasis1D(2, std::vector<double>{0,0,0,1,1,1,2,2,2}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "full discontinuous interior knot must retain original rejection");
}
void test_owned_nets() {
    const auto patch = make_patch(3, 2, 1);
    auto base = extract(patch);
    NurbsPatchPolarEvaluator3D evaluator(patch, base);
    for (auto& element : base) for (auto& control : element.homogeneous_controls) control.setZero();
    auto moved = std::move(evaluator);
    NurbsPolarEvaluationWorkspace3D workspace;
    compare(patch, moved, workspace, 0.2, 4.2);
}
}
int main() {
    try {
        test_patch(0, 0); test_patch(0, 2); test_patch(2, 0);
        for (int p : {1, 2, 3, NurbsPolarEvaluationWorkspace3D::fixed_degree + 3}) {
            test_patch(p, p); test_patch(p, p, 1); test_patch(p, p, p);
        }
        test_endpoint_absorption(); test_fallbacks(); test_owned_nets();
        std::cout << "NURBS polar evaluation tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        count_allocations = false;
        std::cerr << e.what() << '\n'; return 1;
    }
}
