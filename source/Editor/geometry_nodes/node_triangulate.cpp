
#include "GCore/Components/MeshComponent.h"
#include "GCore/util_openmesh_bind.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(triangulate)
{
    // Function content omitted
    b.add_input<Geometry>("Input");
    b.add_output<Geometry>("Ouput");
}

NODE_EXECUTION_FUNCTION(triangulate)
{
    // Function content omitted
    auto input_geom = params.get_input<Geometry>("Input");

    input_geom.apply_transform();

    auto mesh = input_geom.get_component<MeshComponent>();
    auto omesh = operand_to_openmesh(&input_geom);
    omesh->triangulate();
    auto triangulated = openmesh_to_operand(omesh.get());

    // Iterate through components of the original geometry
    for (const auto& component : input_geom.get_components()) {
        // Check if the component is NOT a MeshComponent
        // If it's not a mesh component, attach it to the new geometry.
        // The triangulated geometry already has the new MeshComponent from
        // openmesh_to_operand.
        if (!std::dynamic_pointer_cast<MeshComponent>(component)) {
            triangulated->attach_component(component);
        }
    }

    params.set_output<Geometry>("Ouput", std::move(*triangulated));

    return true;
}

NODE_DECLARATION_UI(triangulate);
NODE_DEF_CLOSE_SCOPE
