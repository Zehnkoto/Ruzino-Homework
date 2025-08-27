#include <gtest/gtest.h>

#include "GUI/window.h"
#include "rzpython/rzpython.hpp"

using namespace USTC_CG;

TEST(RZPythonRuntimeTest, RHI_package)
{
    python::initialize();

    python::import("RHI_py");
    int result = python::call("RHI_py.initialize()");
    EXPECT_EQ(result, 0);

    nvrhi::IDevice* = python::call("RHI_py.get_device()");
    nvrhi::GraphicsAPI backend = python::call("RHI_py.get_backend()");

    result = python::call("RHI_py.shutdown()");

    python::finalize();
}

TEST(RZPythonRuntimeTest, GUI_package)
{
    python::initialize();

    python::import("GUI_py");

    Window window;
    window.run();
    python::reference("w", &window);  // Or some other kind of reference
    python::call("print(w.get_elapsed_time())");

    float time = python::call("w.get_elapsed_time()");
    EXPECT_GT(time, 0.0f);

    python::finalize();
}
