
#include "nodes/core/def/node_def.hpp"'
#include <GCore/Components/PointsComponent.h>

#include "GCore/Components/MeshComponent.h"
#include "light_field/light_field.h"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(visualize_light_field_lenses)
{
	// Function content omitted
	b.add_input<pxr::VtArray<pxr::GfVec3f>>("Set Locations").optional(true);
	b.add_input<float>("Width").min(0.1).max(1).default_val(0.1);
	b.add_output<Geometry>("Lenses points");

}

NODE_EXECUTION_FUNCTION(visualize_light_field_lenses)
{
	bool has_input_locations = params.has_input("Set Locations");

	if (has_input_locations)
	{
		auto input_locations = params.get_input<pxr::VtArray<pxr::GfVec3f>>("Set Locations");
		set_light_field_lens_locations(input_locations);
	}

	auto locations = get_light_field_lens_locations();

	auto points = Geometry::CreatePoints();
	auto points_component = points.get_component<PointsComponent>();
	points_component->set_vertices(locations);
	auto width = params.get_input<float>("Width");
	points_component->set_width(pxr::VtArray<float>(locations.size(), width));
	params.set_output("Lenses points", points);

	// Function content omitted
	return true;
}

NODE_DECLARATION_UI(visualize_light_field_lenses);
NODE_DEF_CLOSE_SCOPE
