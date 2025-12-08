#include "basic_node_base.h"
NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(int)
{
    b.add_input<int>("in_value").min(0).max(10).default_val(1);
    b.add_output<int>("out_value");
}

NODE_EXECUTION_FUNCTION(int)
{
    auto val = params.get_input<int>("in_value");
    params.set_output("out_value", val);
    return true;
}

NODE_DECLARATION_FUNCTION(float)
{
    b.add_input<float>("in_value").min(0).max(10).default_val(1);
    b.add_output<float>("out_value");
}

NODE_EXECUTION_FUNCTION(float)
{
    auto val = params.get_input<float>("in_value");
    params.set_output("out_value", val);
    return true;
}

NODE_DECLARATION_FUNCTION(bool)
{
    b.add_input<bool>("in_value").default_val(true);
    b.add_output<bool>("out_value");
}

NODE_EXECUTION_FUNCTION(bool)
{
    auto val = params.get_input<bool>("in_value");
    params.set_output("out_value", val);
    return true;
}

NODE_DECLARATION_UI(value);
NODE_DEF_CLOSE_SCOPE
