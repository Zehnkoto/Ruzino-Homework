#include "stage/ecs_components.hpp"

#include <pxr/usd/usdGeom/mesh.h>

#include "GCore/usd_extension.h"

RUZINO_NAMESPACE_OPEN_SCOPE
namespace ecs {

// ============================================================================
// UsdPrimComponent 同步方法实现
// ============================================================================

bool UsdPrimComponent::sync_to_geometry(
    Geometry& geometry,
    const pxr::UsdTimeCode& time) const
{
    if (!prim) {
        return false;
    }
    return read_geometry_from_usd(geometry, prim, time);
}

bool UsdPrimComponent::sync_to_geometry(Geometry& geometry) const
{
    return sync_to_geometry(geometry, current_time);
}

bool UsdPrimComponent::sync_from_geometry(
    const Geometry& geometry,
    const pxr::UsdTimeCode& time)
{
    if (!prim || !prim.GetStage()) {
        return false;
    }
    return write_geometry_to_usd(
        geometry, prim.GetStage(), prim.GetPath(), time);
}

bool UsdPrimComponent::sync_from_geometry(const Geometry& geometry)
{
    return sync_from_geometry(geometry, current_time);
}

}  // namespace ecs
RUZINO_NAMESPACE_CLOSE_SCOPE
