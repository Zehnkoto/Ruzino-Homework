# RZ Console 重构设计文档

## 当前设计问题

### 1. 复杂的类型系统
- 使用大量宏定义和模板特化
- `VariableImpl`类层次复杂，难以理解和维护
- 类型转换逻辑分散在多个地方

### 2. 全局状态问题
- `ObjectDictionary`是全局单例
- 难以进行单元测试和并发使用
- 全局注册可能导致初始化顺序问题

### 3. 职责混合
- `ImGui_Console`既负责UI渲染又管理日志
- spdlog集成复杂且容易出错
- 命令解释和UI紧耦合

### 4. 接口复杂
- 需要继承多个类和实现复杂的虚函数
- 用户需要了解内部实现细节
- 代码重复度高

## 重构后的设计

### 1. 简化的类型系统 (`console_core.h`)

```cpp
// 使用std::variant替代复杂的类继承
using VariableValue = std::variant<
    bool, int, float, std::string,
    glm::vec2, glm::vec3, glm::vec4,
    glm::ivec2, glm::ivec3
>;

// 简化的变量API
template<typename T>
class TypedVariable {
    T Get() const;
    bool Set(const T& value);
    operator T() const;
    TypedVariable& operator=(const T& value);
};
```

**优势：**
- 类型安全且简单
- 编译时类型检查
- 无需宏定义
- 易于扩展新类型

### 2. 依赖注入设计

```cpp
// 非单例，可测试的注册表
class ConsoleRegistry {
    bool RegisterCommand(const SimpleCommandDesc& desc);
    bool RegisterVariable(const VariableDesc& desc);
    CommandResult ExecuteCommand(std::string_view command_line);
};

// UI组件接受依赖注入
class ModernConsoleWidget : public IWidget {
public:
    ModernConsoleWidget(
        std::shared_ptr<ConsoleRegistry> registry,
        std::shared_ptr<LogManager> log_manager);
};
```

**优势：**
- 可测试性：可以注入mock对象
- 线程安全：每个实例独立
- 灵活性：可以有多个控制台实例
- 清晰的依赖关系

### 3. 关注点分离

```cpp
// 专职日志管理
class LogManager {
    void AddMessage(LogMessage::Level level, const std::string& text);
    std::vector<const LogMessage*> GetFilteredMessages() const;
};

// 专职命令/变量管理
class ConsoleRegistry {
    CommandResult ExecuteCommand(std::string_view command_line);
    std::vector<std::string> GetSuggestions(...);
};

// 专职UI渲染
class ModernConsoleWidget : public IWidget {
    bool BuildUI() override;
};
```

**优势：**
- 单一职责原则
- 易于测试和维护
- 可以独立替换组件
- 减少耦合

### 4. 简化的用户接口

```cpp
// 之前：复杂的注册过程
console::CommandDesc cmd = {
    "name", "desc", 
    [](console::Command::Args const& args) -> console::Command::Result {
        return { true, "output" };
    }
};
console::RegisterCommand(cmd);  // 全局状态

// 现在：简洁的API
console::SimpleCommandDesc cmd;
cmd.name = "name";
cmd.description = "desc";
cmd.execute = [](const std::vector<std::string>& args) {
    return console::CommandResult(true, "output");
};
registry.RegisterCommand(cmd);  // 实例方法
```

**优势：**
- API更直观
- 减少样板代码
- 类型更简单
- 错误更少

## 迁移指南

### 1. 替换旧的变量定义

```cpp
// 旧代码
cvarFloat speed("speed", "Movement speed", 5.0f);

// 新代码
console::FloatVar speed(registry, "speed", "Movement speed", 5.0f);
// 或者
console::VariableDesc speed_desc;
speed_desc.name = "speed";
speed_desc.description = "Movement speed";
speed_desc.default_value = 5.0f;
registry.RegisterVariable(speed_desc);
```

### 2. 替换命令注册

```cpp
// 旧代码
console::CommandDesc echo_cmd = {
    "echo", "Echo text",
    [](console::Command::Args const& args) -> console::Command::Result {
        return { true, "output" };
    }
};
console::RegisterCommand(echo_cmd);

// 新代码
console::SimpleCommandDesc echo_cmd;
echo_cmd.name = "echo";
echo_cmd.description = "Echo text";
echo_cmd.execute = [](const std::vector<std::string>& args) {
    return console::CommandResult(true, "output");
};
registry.RegisterCommand(echo_cmd);
```

### 3. 替换控制台创建

```cpp
// 旧代码
auto interpreter = std::make_shared<console::Interpreter>();
auto console = std::make_unique<ImGui_Console>(interpreter, opts);

// 新代码
auto registry = std::make_shared<console::ConsoleRegistry>();
auto log_manager = std::make_shared<console::LogManager>();
auto console = std::make_unique<ModernConsoleWidget>(registry, log_manager, opts);
```

## 性能改进

1. **减少虚函数调用**：使用`std::variant`替代虚函数
2. **减少字符串拷贝**：使用`string_view`和移动语义
3. **更好的内存管理**：智能指针和RAII
4. **减少模板实例化**：简化的类型系统

## 测试改进

```cpp
// 现在可以轻松进行单元测试
TEST(ConsoleRegistry, BasicCommands) {
    console::ConsoleRegistry registry;
    
    console::SimpleCommandDesc test_cmd;
    test_cmd.name = "test";
    test_cmd.execute = [](const auto&) { 
        return console::CommandResult(true, "success"); 
    };
    
    ASSERT_TRUE(registry.RegisterCommand(test_cmd));
    
    auto result = registry.ExecuteCommand("test");
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.output, "success");
}
```

## 总结

重构后的设计：
- **更简单**：减少了类层次和模板复杂性
- **更灵活**：依赖注入支持多种配置
- **更可测试**：非全局状态，清晰的接口
- **更易维护**：关注点分离，代码更清晰
- **更好的性能**：减少虚函数和字符串拷贝

这个重构保持了原有功能的同时，显著改善了代码质量和可维护性。