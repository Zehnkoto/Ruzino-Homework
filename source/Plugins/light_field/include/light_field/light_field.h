#pragma once

#include "api.h"
#include <pxr/base/vt/array.h>
#include <pxr/base/gf/vec3f.h>

USTC_CG_NAMESPACE_OPEN_SCOPE

LIGHT_FIELD_API pxr::VtArray<pxr::GfVec3f> get_light_field_lens_locations();
LIGHT_FIELD_API pxr::VtArray<pxr::GfVec3f> set_light_field_lens_locations(
	const pxr::VtArray<pxr::GfVec3f>& lens_locations);

USTC_CG_NAMESPACE_CLOSE_SCOPE