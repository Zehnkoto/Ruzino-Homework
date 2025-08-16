#include "fem_bem.hpp"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(fem_solver)
{
    // Function content omitted
    b.add_input<Geometry>("Geometry");
    b.add_input<int>("Problem Dimension").default_val(2).min(2).max(3);
    
}

NODE_EXECUTION_FUNCTION(fem_solver)
{
    // Function content omitted
    return true;
}

NODE_DECLARATION_UI(fem_solver);
NODE_DEF_CLOSE_SCOPE
