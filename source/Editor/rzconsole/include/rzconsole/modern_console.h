#pragma once

#include <GUI/widget.h>
#include <memory>
#include <vector>
#include <string>
#include "console_core.h"
#include "api.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

// 现代化的控制台UI - 职责单一，依赖注入
class RZCONSOLE_API ModernConsoleWidget : public IWidget {
public:
    struct Options {
        bool auto_scroll = true;
        bool show_timestamps = false;
        bool show_info = true;
        bool show_warnings = true; 
        bool show_errors = true;
        size_t max_log_entries = 1000;
    };

    ModernConsoleWidget(
        std::shared_ptr<console::ConsoleRegistry> registry,
        std::shared_ptr<console::LogManager> log_manager,
        const Options& options = {});

    ~ModernConsoleWidget();

    // IWidget interface
    bool BuildUI() override;

    // 直接添加日志消息
    void AddLogMessage(console::LogMessage::Level level, const std::string& text);

protected:
    const char* GetWindowName() override { return "Modern Console"; }
    bool HasMenuBar() const override { return true; }

private:
    void DrawMenuBar();
    void DrawLogArea();
    void DrawCommandInput();
    void DrawFilters();
    
    void ExecuteCommand();
    void UpdateAutoComplete();
    
    // Input callbacks
    int TextEditCallback(ImGuiInputTextCallbackData* data);
    int HistoryCallback(ImGuiInputTextCallbackData* data);
    int AutoCompleteCallback(ImGuiInputTextCallbackData* data);

    std::shared_ptr<console::ConsoleRegistry> registry_;
    std::shared_ptr<console::LogManager> log_manager_;
    Options options_;
    
    // UI state
    std::array<char, 512> input_buffer_{};
    std::vector<std::string> command_history_;
    int history_pos_ = -1;
    std::vector<std::string> current_suggestions_;
    bool scroll_to_bottom_ = false;
};

// 工厂函数 - 便于创建预配置的控制台
RZCONSOLE_API std::unique_ptr<ModernConsoleWidget> CreateDefaultConsole();

// spdlog集成
RZCONSOLE_API void SetupModernConsoleLogging(
    std::shared_ptr<console::LogManager> log_manager);

USTC_CG_NAMESPACE_CLOSE_SCOPE