#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>
#include <optional>
#include <chrono>
#include <glm/glm.hpp>
#include "api.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

namespace console {

// 简化的变量值类型
using VariableValue = std::variant<
    bool, int, float, std::string,
    glm::vec2, glm::vec3, glm::vec4,
    glm::ivec2, glm::ivec3
>;

// 命令结果
struct CommandResult {
    bool success = false;
    std::string output;
    
    CommandResult(bool s, std::string o = "") : success(s), output(std::move(o)) {}
};

// 命令函数类型
using CommandFunction = std::function<CommandResult(const std::vector<std::string>&)>;
using SuggestionFunction = std::function<std::vector<std::string>(std::string_view, size_t)>;

// 变量改变回调
using VariableChangeCallback = std::function<void(const std::string&, const VariableValue&)>;

// 前向声明
class ConsoleRegistry;

// 命令描述符 - 简化版本
struct SimpleCommandDesc {
    std::string name;
    std::string description;
    CommandFunction execute;
    SuggestionFunction suggest = nullptr;
};

// 变量描述符 - 简化版本
struct VariableDesc {
    std::string name;
    std::string description;
    VariableValue default_value;
    bool read_only = false;
    bool is_cheat = false;
    VariableChangeCallback on_change = nullptr;
};

// 核心控制台注册表 - 非单例，可测试
class RZCONSOLE_API ConsoleRegistry {
public:
    ConsoleRegistry() = default;
    ~ConsoleRegistry() = default;

    // 注册命令
    bool RegisterCommand(const SimpleCommandDesc& desc);
    
    // 注册变量
    bool RegisterVariable(const VariableDesc& desc);
    
    // 执行命令
    CommandResult ExecuteCommand(std::string_view command_line);
    
    // 获取建议
    std::vector<std::string> GetSuggestions(std::string_view command_line, size_t cursor_pos);
    
    // 变量操作
    bool SetVariable(const std::string& name, const VariableValue& value);
    std::optional<VariableValue> GetVariable(const std::string& name);
    
    // 列出所有对象
    std::vector<std::string> ListCommands() const;
    std::vector<std::string> ListVariables() const;
    
    // 清除所有注册的对象
    void Clear();

private:
    std::unordered_map<std::string, SimpleCommandDesc> commands_;
    std::unordered_map<std::string, VariableDesc> variables_;
    std::unordered_map<std::string, VariableValue> variable_values_;
    
    // 解析命令行
    std::vector<std::string> ParseCommandLine(std::string_view command_line);
};

// 日志消息
struct LogMessage {
    enum class Level { Info, Warning, Error };
    
    Level level = Level::Info;
    std::string text;
    std::chrono::system_clock::time_point timestamp;
    
    LogMessage(Level l, std::string t) 
        : level(l), text(std::move(t)), timestamp(std::chrono::system_clock::now()) {}
};

// 日志管理器 - 分离关注点
class RZCONSOLE_API LogManager {
public:
    LogManager() = default;
    ~LogManager() = default;

    void AddMessage(LogMessage::Level level, const std::string& text);
    void AddMessage(const LogMessage& msg);
    
    const std::vector<LogMessage>& GetMessages() const { return messages_; }
    void Clear() { messages_.clear(); }
    
    // 设置过滤器
    void SetLevelFilter(LogMessage::Level min_level) { min_level_ = min_level; }
    
    // 获取过滤后的消息
    std::vector<const LogMessage*> GetFilteredMessages() const;

private:
    std::vector<LogMessage> messages_;
    LogMessage::Level min_level_ = LogMessage::Level::Info;
};

// 便利类型别名
template<typename T>
class TypedVariable {
public:
    TypedVariable(ConsoleRegistry& registry, const std::string& name, 
                  const std::string& desc, const T& default_val, bool read_only = false)
        : registry_(registry), name_(name) {
        VariableDesc desc_obj;
        desc_obj.name = name;
        desc_obj.description = desc;
        desc_obj.default_value = default_val;
        desc_obj.read_only = read_only;
        registry_.RegisterVariable(desc_obj);
    }
    
    T Get() const {
        auto val = registry_.GetVariable(name_);
        if (val && std::holds_alternative<T>(*val)) {
            return std::get<T>(*val);
        }
        return T{};
    }
    
    bool Set(const T& value) {
        return registry_.SetVariable(name_, value);
    }
    
    operator T() const { return Get(); }
    TypedVariable& operator=(const T& value) { Set(value); return *this; }

private:
    ConsoleRegistry& registry_;
    std::string name_;
};

// 便利类型定义
using BoolVar = TypedVariable<bool>;
using IntVar = TypedVariable<int>;
using FloatVar = TypedVariable<float>;
using StringVar = TypedVariable<std::string>;
using Vec2Var = TypedVariable<glm::vec2>;
using Vec3Var = TypedVariable<glm::vec3>;
using Vec4Var = TypedVariable<glm::vec4>;

} // namespace console

USTC_CG_NAMESPACE_CLOSE_SCOPE