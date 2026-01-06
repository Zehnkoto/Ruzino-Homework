#include <glm/glm.hpp>
#include <memory>

#include "GCore/Components/MeshComponent.h"
#include "GCore/geom_payload.hpp"
#include "nodes/core/def/node_def.hpp"
#include "nodes/core/io/json.hpp"
#include "rzsim/reduced_order_basis.h"

using namespace Ruzino;

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(reduced_ordered_mass_spring)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<std::shared_ptr<ReducedOrderedBasis>>("Reduced Basis");

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(reduced_ordered_mass_spring)
{
    // Get inputs
    auto input_geom = params.get_input<Geometry>("Geometry");
    auto reduced_basis =
        params.get_input<std::shared_ptr<ReducedOrderedBasis>>("Reduced Basis");

    return true;
}

NODE_DECLARATION_UI(reduced_ordered_mass_spring);

NODE_DEF_CLOSE_SCOPE
