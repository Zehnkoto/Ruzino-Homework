#include <GUI/widget.h>
#include <GUI/window.h>
#include <gtest/gtest.h>
#include <rzconsole/ConsoleInterpreter.h>
#include <rzconsole/ConsoleObjects.h>
#include <rzconsole/imgui_console.h>
#include <rzconsole/spdlog_console_sink.h>
#include <spdlog/spdlog.h>

#include <memory>

using namespace USTC_CG;

class ConsoleWidgetFactory : public IWidgetFactory {
   public:
    std::unique_ptr<IWidget> Create(
        const std::vector<std::unique_ptr<IWidget>>& others) override
    {
        auto interpreter = std::make_shared<console::Interpreter>();

        // Register some test commands
        console::CommandDesc test_cmd = {
            "test",
            "A test command that prints hello world",
            [](console::Command::Args const& args) -> console::Command::Result {
                return { true, "Hello from test command!\n" };
            }
        };
        console::RegisterCommand(test_cmd);

        console::CommandDesc log_test_cmd = {
            "log_test",
            "Test spdlog integration",
            [](console::Command::Args const& args) -> console::Command::Result {
                spdlog::info("This is an info message");
                spdlog::warn("This is a warning message");
                spdlog::error("This is an error message");
                return { true, "Log test messages sent\n" };
            }
        };
        console::RegisterCommand(log_test_cmd);

        // Create console with interpreter
        ImGui_Console::Options opts;
        opts.show_info = true;
        opts.show_warnings = true;
        opts.show_errors = true;

        auto console = std::make_unique<ImGui_Console>(interpreter, opts);

        // Setup spdlog integration AFTER creating console
        setup_console_logging(console.get());

        // Add some initial messages
        console->Print("Console initialized successfully!");
        console->Print("Type 'help' for available commands");
        console->Print("Try 'test' or 'log_test' commands");
        console->Print("Console is ready for use!");

        return std::move(console);
    }
};

int main()
{
    // Create console directly
    auto interpreter = std::make_shared<console::Interpreter>();

    // Register a simple test command
    console::CommandDesc echo_cmd = {
        "echo",
        "Echo text to console",
        [](console::Command::Args const& args) -> console::Command::Result {
            std::string output;
            for (size_t i = 1; i < args.size(); ++i) {
                output += args[i] + " ";
            }
            return { true, output + "\n" };
        }
    };
    console::RegisterCommand(echo_cmd);

    ImGui_Console::Options opts;
    auto console = std::make_unique<ImGui_Console>(interpreter, opts);

    console->Print("Direct widget console test");

    Window window;
    setup_console_logging(console.get());

    window.register_widget(std::move(console));
    window.run();
}
