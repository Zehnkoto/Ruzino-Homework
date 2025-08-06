#include "light_field/light_field.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
pxr::VtArray<pxr::GfVec3f> g_light_field_lens_locations;

pxr::VtArray<pxr::GfVec3f> get_light_field_lens_locations()
{
	return g_light_field_lens_locations;
}

pxr::VtArray<pxr::GfVec3f> set_light_field_lens_locations(
	const pxr::VtArray<pxr::GfVec3f>& lens_locations)
{
	g_light_field_lens_locations = lens_locations;
	return g_light_field_lens_locations;
}


USTC_CG_NAMESPACE_CLOSE_SCOPE