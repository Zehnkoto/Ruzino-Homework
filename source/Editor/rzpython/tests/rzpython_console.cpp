#include <GUI/widget.h>
#include <GUI/window.h>
#include <gtest/gtest.h>
#include <rzconsole/ConsoleInterpreter.h>
#include <rzconsole/ConsoleObjects.h>
#include <rzconsole/imgui_console.h>
#include <rzconsole/spdlog_console_sink.h>
#include <rzpython/interpreter.hpp>
#include <rzpython/rzpython.hpp>
#include <spdlog/spdlog.h>

#include <memory>

using namespace USTC_CG;

class PythonConsoleWidgetFactory : public IWidgetFactory {
   public:
    std::unique_ptr<IWidget> Create(
        const std::vector<std::unique_ptr<IWidget>>& others) override
    {
        // Create Python interpreter
        auto interpreter = python::CreatePythonInterpreter();

        // Register some test commands that work with Python
        console::CommandDesc test_cmd = {
            "test",
            "A test command that demonstrates Python integration",
            [](console::Command::Args const& args) -> console::Command::Result {
                try {
                    // Use Python to do some computation
                    python::send("test_value", 42);
                    python::call<void>("test_result = test_value * 2");
                    int result = python::call<int>("test_result");
                    
                    return { true, "Test result from Python: " + std::to_string(result) + "\n" };
                } catch (const std::exception& e) {
                    return { false, "Python test failed: " + std::string(e.what()) + "\n" };
                }
            }
        };
        console::RegisterCommand(test_cmd);

        // Create console with capture_log enabled
        ImGui_Console::Options opts;
        opts.show_info = true;
        opts.show_warnings = true;
        opts.show_errors = true;
        opts.capture_log = true;

        auto console = std::make_unique<ImGui_Console>(interpreter, opts);

        // Add some initial messages
        console->Print("Python Console initialized successfully!");
        console->Print("You can now enter Python code directly (e.g., 'x = 5')");
        console->Print("Or use console commands like 'help', 'test'");
        console->Print("Try: python print('Hello from Python!')");

        return std::move(console);
    }
};

int main()
{
    // Create Python console test
    auto interpreter = python::CreatePythonInterpreter();

    // Test Python functionality
    console::CommandDesc math_cmd = {
        "math_test",
        "Test Python math operations",
        [](console::Command::Args const& args) -> console::Command::Result {
            try {
                python::call<void>("import math");
                python::send("radius", 5.0f);
                python::call<void>("area = math.pi * radius ** 2");
                float area = python::call<float>("area");
                
                return { true, "Circle area (r=5): " + std::to_string(area) + "\n" };
            } catch (const std::exception& e) {
                return { false, "Math test failed: " + std::string(e.what()) + "\n" };
            }
        }
    };
    console::RegisterCommand(math_cmd);

    ImGui_Console::Options opts;
    opts.capture_log = true;
    auto console = std::make_unique<ImGui_Console>(interpreter, opts);

    console->Print("Python Interactive Console Test");
    console->Print("Python commands are handled directly:");
    console->Print("  x = 10        # Assignment");
    console->Print("  x             # Variable lookup");  
    console->Print("  print('hi')   # Function calls");
    console->Print("  2 + 3         # Expressions");
    console->Print("");
    console->Print("Console commands:");
    console->Print("  help          # Show help");
    console->Print("  math_test     # Math operations test");
    console->Print("  python <code> # Explicit Python execution");

    Window window;
    setup_console_logging(console.get());
    window.register_widget(std::move(console));
    window.run();

    return 0;
}
