#include <gtest/gtest.h>

#include "MCore/Graph.h"
#include "MCore/MaterialXNodeTree.hpp"

using namespace USTC_CG;

int main()
{
    MaterialXNodeTreeDescriptor descriptor;
    MaterialXNodeTree tree(
        "resources/Materials/Examples/StandardSurface/"
        "standard_surface_marble_solid.mtlx",
        std::make_shared<MaterialXNodeTreeDescriptor>());

    std::cout << tree.nodes.size() << std::endl;
}