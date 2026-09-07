#pragma once

#include "restrict_crossing_selector_3d.hpp"
#include "src/geometry/native_endpoint_path_3d.hpp"

namespace kfbim::app3d {

// Geometry is shared by the two trace branches; only the conditional endpoint
// term changes. Never send an incomplete native result through the legacy
// floating-point endpoint selector to "repair" it.
struct NativeEndpointEventSequence3D {
    geometry3d::NurbsSurfaceIntersectionResult3D intersection;
    SegmentPhysicalEventSequence3D sequence;
};

NativeEndpointEventSequence3D select_native_endpoint_path_events_3d(
    const geometry3d::NativeEndpointPathResult3D& path,
    bool start_inside, bool desired_inside);

} // namespace kfbim::app3d
