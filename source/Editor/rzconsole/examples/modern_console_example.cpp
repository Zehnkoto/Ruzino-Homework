#include <GUI/widget.h>
#include <GUI/window.h>
#include <rzconsole/console_core.h>
#include <rzconsole/modern_console.h>
#include <memory>

using namespace USTC_CG;

// 使用新设计的示例
class ModernConsoleExample {
public:
    static std::unique_ptr<IWidget> CreateConsoleWidget() {
        // 1. 创建核心组件 - 依赖注入，可测试
        auto registry = std::make_shared<console::ConsoleRegistry>();
        auto log_manager = std::make_shared<console::LogManager>();
        
        // 2. 注册一些示例命令
        RegisterExampleCommands(*registry, *log_manager);
        
        // 3. 注册一些示例变量
        RegisterExampleVariables(*registry);
        
        // 4. 创建UI组件
        ModernConsoleWidget::Options opts;
        opts.show_timestamps = true;
        opts.max_log_entries = 500;
        
        auto console = std::make_unique<ModernConsoleWidget>(
            registry, log_manager, opts);
        
        // 5. 设置spdlog集成
        SetupModernConsoleLogging(log_manager);
        
        // 6. 添加欢迎消息
        log_manager->AddMessage(console::LogMessage::Level::Info, 
                               "Modern Console System Initialized!");
        log_manager->AddMessage(console::LogMessage::Level::Info, 
                               "Type 'help' for available commands");
        
        return std::move(console);
    }

private:
    static void RegisterExampleCommands(
        console::ConsoleRegistry& registry,
        console::LogManager& log_manager) {
        
        // Echo command
        console::SimpleCommandDesc echo_cmd;
        echo_cmd.name = "echo";
        echo_cmd.description = "Echo text to console";
        echo_cmd.execute = [&log_manager](const std::vector<std::string>& args) {
            std::string output;
            for (size_t i = 1; i < args.size(); ++i) {
                output += args[i] + " ";
            }
            log_manager.AddMessage(console::LogMessage::Level::Info, output);
            return console::CommandResult(true, "Echoed: " + output);
        };
        registry.RegisterCommand(echo_cmd);
        
        // Clear command
        console::SimpleCommandDesc clear_cmd;
        clear_cmd.name = "clear";
        clear_cmd.description = "Clear console log";
        clear_cmd.execute = [&log_manager](const std::vector<std::string>&) {
            log_manager.Clear();
            return console::CommandResult(true, "Console cleared");
        };
        registry.RegisterCommand(clear_cmd);
        
        // List commands
        console::SimpleCommandDesc list_cmd;
        list_cmd.name = "list";
        list_cmd.description = "List all commands and variables";
        list_cmd.execute = [&registry](const std::vector<std::string>&) {
            std::string output = "Commands:\n";
            for (const auto& cmd : registry.ListCommands()) {
                output += "  " + cmd + "\n";
            }
            output += "Variables:\n";
            for (const auto& var : registry.ListVariables()) {
                output += "  " + var + "\n";
            }
            return console::CommandResult(true, output);
        };
        registry.RegisterCommand(list_cmd);
    }
    
    static void RegisterExampleVariables(console::ConsoleRegistry& registry) {
        // 使用新的简化API
        console::VariableDesc debug_var;
        debug_var.name = "debug";
        debug_var.description = "Enable debug mode";
        debug_var.default_value = false;
        debug_var.on_change = [](const std::string& name, const console::VariableValue& value) {
            bool debug_enabled = std::get<bool>(value);
            // 在这里处理变量变化
            spdlog::info("Debug mode changed to: {}", debug_enabled);
        };
        registry.RegisterVariable(debug_var);
        
        console::VariableDesc speed_var;
        speed_var.name = "speed";
        speed_var.description = "Movement speed";
        speed_var.default_value = 5.0f;
        registry.RegisterVariable(speed_var);
        
        console::VariableDesc position_var;
        position_var.name = "position";
        position_var.description = "Player position";
        position_var.default_value = glm::vec3(0.0f, 0.0f, 0.0f);
        registry.RegisterVariable(position_var);
    }
};

class ModernConsoleFactory : public IWidgetFactory {
public:
    std::unique_ptr<IWidget> Create(
        const std::vector<std::unique_ptr<IWidget>>& others) override {
        return ModernConsoleExample::CreateConsoleWidget();
    }
};

int main() {
    Window window;
    
    // 简单直接的注册方式
    window.register_openable_widget(
        std::make_unique<ModernConsoleFactory>(), 
        { "Tools", "Modern Console" }
    );
    
    window.run();
    return 0;
}