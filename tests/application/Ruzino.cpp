#include <MaterialXCore/Document.h>
#include <MaterialXFormat/XmlIo.h>
#include <rzconsole/ConsoleInterpreter.h>
#include <rzconsole/ConsoleObjects.h>
#include <rzconsole/imgui_console.h>
#include <rzconsole/spdlog_console_sink.h>
#include <spdlog/spdlog.h>

#include <any>
#include <filesystem>
#include <rzpython/interpreter.hpp>
#include <rzpython/rzpython.hpp>

#include "GCore/GOP.h"
#include "GCore/algorithms/intersection.h"
#include "GCore/geom_payload.hpp"
#include "GUI/window.h"
#include "MCore/MaterialXDocumentViewer.hpp"
#include "MCore/MaterialXNodeTree.hpp"
#include "MCore/MaterialXNodeTreeWidget.h"
#include "cmdparser.hpp"
#include "nodes/system/node_system.hpp"
#include "nodes/ui/imgui.hpp"
#include "pxr/base/tf/setenv.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/sphere.h"
#include "pxr/usd/usdShade/material.h"
#include "pxr/usd/usdShade/shader.h"
#include "stage/stage.hpp"
#include "usd_nodejson.hpp"
#include "widgets/usdtree/usd_fileviewer.h"
#include "widgets/usdview/usdview_widget.hpp"

using namespace USTC_CG;
namespace mx = MaterialX;

class MaterialXNodeSystem : public NodeSystem {
   public:
    MaterialXNodeSystem()
    {
        descriptor = std::make_shared<MaterialXNodeTreeDescriptor>();
    }

    // Factory method to create a MaterialX system with a default material
    // document
    static std::shared_ptr<MaterialXNodeSystem> create_with_default_material(
        const std::string& material_name)
    {
        auto system = std::make_shared<MaterialXNodeSystem>();

        // Create a minimal MaterialX document in memory using MaterialX API
        mx::DocumentPtr doc = mx::createDocument();

        // Create only a surface material node
        mx::NodePtr materialNode =
            doc->addNode("surfacematerial", material_name, "material");

        // Create MaterialXNodeTree with the in-memory document
        std::unique_ptr<NodeTree> tree = std::make_unique<MaterialXNodeTree>(
            system->node_tree_descriptor(), doc);

        system->init(std::move(tree));
        system->set_node_tree_executor(create_node_tree_executor({}));

        return system;
    }

    void set_node_tree_executor(
        std::unique_ptr<NodeTreeExecutor> executor) override
    {
    }

    bool load_configuration(const std::string& config) override
    {
        return true;
    }

    ~MaterialXNodeSystem() override
    {
    }

    void execute(bool is_ui_execution, Node* required_node) const override
    {
    }

    std::shared_ptr<NodeTreeDescriptor> node_tree_descriptor() override
    {
        return descriptor;
    }

   private:
    std::shared_ptr<MaterialXNodeTreeDescriptor> descriptor;
};

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

                    return { true,
                             "Test result from Python: " +
                                 std::to_string(result) + "\n" };
                }
                catch (const std::exception& e) {
                    return { false,
                             "Python test failed: " + std::string(e.what()) +
                                 "\n" };
                }
            }
        };
        console::RegisterCommand(test_cmd);

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

                    return { true,
                             "Circle area (r=5): " + std::to_string(area) +
                                 "\n" };
                }
                catch (const std::exception& e) {
                    return { false,
                             "Math test failed: " + std::string(e.what()) +
                                 "\n" };
                }
            }
        };
        console::RegisterCommand(math_cmd);

        // Add simple debug command to test if commands work at all
        console::CommandDesc simple_test_cmd = {
            "simple_test",
            "Simple test command",
            [](console::Command::Args const& args) -> console::Command::Result {
                return { true, "Simple test command works!\n" };
            }
        };
        console::RegisterCommand(simple_test_cmd);

        // Add debug command to test Python interpreter directly
        console::CommandDesc debug_cmd = {
            "debug_python",
            "Debug Python interpreter state",
            [](console::Command::Args const& args) -> console::Command::Result {
                try {
                    std::string debug_info = "Python initialized: ";
                    debug_info += python::initialized ? "Yes" : "No";
                    debug_info += "\nTesting basic operation...\n";

                    python::call<void>("test_var = 42");
                    int result = python::call<int>("test_var");
                    debug_info +=
                        "Basic test result: " + std::to_string(result) + "\n";

                    return { true, debug_info };
                }
                catch (const std::exception& e) {
                    return { false,
                             "Debug failed: " + std::string(e.what()) + "\n" };
                }
            }
        };
        console::RegisterCommand(debug_cmd);

        // Create console with capture_log enabled
        ImGui_Console::Options opts;
        opts.show_info = true;
        opts.show_warnings = true;
        opts.show_errors = true;
        opts.capture_log = false;

        auto console = std::make_unique<ImGui_Console>(interpreter, opts);

        // Add some initial messages
        console->Print("=== Python Interactive Console ===");
        console->Print("Python Console initialized successfully!");
        console->Print("FIRST: Test basic console commands:");
        console->Print("  simple_test      # Test if console commands work");
        console->Print("  debug_python     # Test Python state");
        console->Print("  math_test        # Test Python math");
        console->Print("");
        console->Print(
            "THEN: Test Python commands directly (these should work):");
        console->Print("  x = 10           # Python assignment");
        console->Print("  x                # Python variable lookup");
        console->Print("  print('hello')   # Python function call");
        console->Print("  2 + 3            # Python expression");
        console->Print("");
        console->Print(
            "You can now enter Python code directly or use console commands");
        console->Print("Type 'help' for available commands");

        return std::move(console);
    }
};

int main(int argc, char* argv[])
{
#ifdef _DEBUG
    spdlog::set_level(spdlog::level::debug);
#endif
    spdlog::set_pattern("%^[%T] %n: %v%$");
    auto window = std::make_unique<Window>();

    // Set MaterialX standard library path using USD's TfSetenv (preferred
    // method)
    std::string mtlx_stdlib = "libraries";
    if (std::filesystem::exists(mtlx_stdlib)) {
        pxr::TfSetenv("PXR_MTLX_STDLIB_SEARCH_PATHS", mtlx_stdlib.c_str());
        spdlog::info("Set PXR_MTLX_STDLIB_SEARCH_PATHS={}", mtlx_stdlib);
    }
    else {
        spdlog::warn("MaterialX stdlib not found at {}", mtlx_stdlib);
    }

    python::initialize();
    // Check for command line arguments to specify USD file
    std::unique_ptr<Stage> stage;
    if (argc > 1) {
        // Use custom stage path from command line
        std::string stage_path = argv[1];
        stage = create_custom_global_stage(stage_path);
    }
    else {
        // Use default stage
        stage = create_global_stage();
    }

#ifdef REAL_TIME
    window->register_function_before_frame(
        [&stage](Window* window) { stage->tick(window->get_elapsed_time()); });
#else

    window->register_function_before_frame(
        [&stage](Window* window) { stage->tick(1.0f / 30.f); });
#endif
    // Add a sphere

    auto usd_file_viewer = std::make_unique<UsdFileViewer>(stage.get());
    auto render = std::make_unique<UsdviewEngine>(stage.get());

    auto render_bare = render.get();

    render->SetCallBack([](Window* window, IWidget* render_widget) {
        auto node_system = static_cast<const std::shared_ptr<NodeSystem>*>(
            dynamic_cast<UsdviewEngine*>(render_widget)
                ->emit_create_renderer_ui_control());
        if (node_system) {
            FileBasedNodeWidgetSettings desc;
            desc.system = *node_system;
            desc.json_path = "../../Assets/render_nodes_save.json";

            std::unique_ptr<IWidget> node_widget =
                std::move(create_node_imgui_widget(desc));

            window->register_widget(std::move(node_widget));
        }
    });

    window->register_widget(std::move(render));
    window->register_widget(std::move(usd_file_viewer));

    // Register Python Console widget in menu
    auto python_console_factory =
        std::make_unique<PythonConsoleWidgetFactory>();
    window->register_widget(python_console_factory->Create({}));
    window->register_openable_widget(
        std::move(python_console_factory), { "Tools", "Python Console" });

    // Add Python reference to window for console access
    python::reference("window", window.get());

    // Subscribe to material editor events
    window->events().subscribe(
        "material_editor_requested",
        [&stage, &window](const std::string& material_path_str) {
            spdlog::info(
                "Material editor requested for: {}", material_path_str);

            pxr::SdfPath material_path(material_path_str);
            auto material_prim =
                stage->get_usd_stage()->GetPrimAtPath(material_path);

            if (!material_prim) {
                spdlog::error("Material prim not found: {}", material_path_str);
                return;
            }

            // Step 1: Create MaterialX file next to the stage file
            std::string stage_path = stage->GetStagePath();
            std::filesystem::path stage_file(stage_path);
            std::filesystem::path stage_dir = stage_file.parent_path();

            // Get material name from the prim
            std::string material_name = material_prim.GetName();
            std::string mtlx_filename = material_name + ".mtlx";
            std::filesystem::path mtlx_path = stage_dir / mtlx_filename;

            spdlog::info("Creating MaterialX file at: {}", mtlx_path.string());

            // Create MaterialX system with default material
            auto mtlx_system =
                MaterialXNodeSystem::create_with_default_material(
                    material_name);

            // Save the MaterialX document to file
            auto* mtlx_tree =
                static_cast<MaterialXNodeTree*>(mtlx_system->get_node_tree());
            mtlx_tree->saveDocument(mx::FilePath(mtlx_path.string()));

            // Launch MaterialX editor widget
            FileBasedNodeWidgetSettings widget_desc;
            widget_desc.system = mtlx_system;
            widget_desc.json_path =
                (stage_dir / (material_name + "_layout.json")).string();

            std::unique_ptr<IWidget> node_widget =
                std::make_unique<MaterialXNodeTreeWidget>(widget_desc);

            // Setup callback to save MaterialX file and update USD reference
            // when editor closes
            auto mtlx_path_copy = mtlx_path.string();
            auto material_path_copy = material_path;
            auto stage_ptr = stage.get();

            node_widget->SetCallBack([mtlx_path_copy,
                                      mtlx_system,
                                      material_path_copy,
                                      stage_ptr,
                                      mtlx_tree](Window*, IWidget*) {
                // Save MaterialX document when editing is done
                spdlog::info(
                    "Saving MaterialX document to: {}", mtlx_path_copy);
                mtlx_tree->saveDocument(mx::FilePath(mtlx_path_copy));

                // Step 2: Setup USD material reference to MaterialX file
                auto material_prim = stage_ptr->get_usd_stage()->GetPrimAtPath(
                    material_path_copy);
                if (!material_prim) {
                    spdlog::error(
                        "Material prim not found when saving: {}",
                        material_path_copy.GetString());
                    return;
                }

                // Load the MaterialX file to find the material and shader nodes
                mx::DocumentPtr mtlx_doc = mx::createDocument();
                try {
                    mx::readFromXmlFile(mtlx_doc, mx::FilePath(mtlx_path_copy));
                }
                catch (const std::exception& e) {
                    spdlog::error(
                        "Failed to read MaterialX file: {}", e.what());
                    return;
                }

                // Find the surface material in the MaterialX document
                std::string mtlx_material_path_str;
                std::string surface_shader_path_str;

                for (auto material_node : mtlx_doc->getMaterialNodes()) {
                    mtlx_material_path_str =
                        "/MaterialX/Materials/" + material_node->getName();

                    // Find the surface shader output
                    auto surfaceshader_input =
                        material_node->getInput("surfaceshader");
                    if (surfaceshader_input) {
                        auto connected_node =
                            surfaceshader_input->getConnectedNode();
                        if (connected_node) {
                            surface_shader_path_str =
                                material_path_copy.GetString() + "/" +
                                connected_node->getName();
                        }
                    }
                    break;  // Use the first material found
                }

                if (mtlx_material_path_str.empty()) {
                    spdlog::warn(
                        "No material node found in MaterialX file: {}",
                        mtlx_path_copy);
                    return;
                }

                // Add reference to the MaterialX file
                auto references = material_prim.GetReferences();
                references.ClearReferences();
                references.AddReference(
                    pxr::SdfReference(
                        mtlx_path_copy, pxr::SdfPath(mtlx_material_path_str)));

                spdlog::info(
                    "Added reference: {} -> {}",
                    mtlx_path_copy,
                    mtlx_material_path_str);

                // Connect surface output if the shader exists
                if (!surface_shader_path_str.empty()) {
                    pxr::UsdShadeMaterial usd_material(material_prim);

                    // Find the shader prim
                    for (auto child : material_prim.GetChildren()) {
                        if (child.IsA<pxr::UsdShadeShader>()) {
                            pxr::UsdShadeShader shader(child);
                            auto surface_output =
                                shader.GetOutput(pxr::TfToken("surface"));

                            if (surface_output) {
                                // Create or get the material's surface output
                                auto material_surface =
                                    usd_material.CreateSurfaceOutput();
                                material_surface.ConnectToSource(
                                    surface_output);

                                spdlog::info(
                                    "Connected surface output: {}",
                                    surface_shader_path_str);
                            }
                            break;
                        }
                    }
                }

                // Save the USD stage
                stage_ptr->get_usd_stage()->Save();
                spdlog::info("USD stage saved with MaterialX reference");
            });

            window->register_widget(std::move(node_widget));
        });

    // Subscribe to document viewer events
    window->events().subscribe(
        "material_doc_viewer_requested",
        [&stage, &window](const std::string& material_path_str) {
            spdlog::info(
                "Material document viewer requested for: {}",
                material_path_str);

            // TODO: Implement document viewer creation
            // This would create a MaterialXDocumentViewer widget
        });

    window->register_function_after_frame([&stage,
                                           render_bare](Window* window) {
        pxr::SdfPath json_path;
        if (stage->consume_editor_creation(json_path)) {
            auto system = create_dynamic_loading_system();
            /* Load the node system */
            auto loaded = system->load_configuration("geometry_nodes.json");
            loaded = system->load_configuration("basic_nodes.json");

            // iterate over path Plugin (not recursively), get all the json and
            // load them

            auto plugin_path = std::filesystem::path("./Plugins");

            if (std::filesystem::exists(plugin_path))
                for (auto& p :
                     std::filesystem::directory_iterator(plugin_path)) {
                    if (p.path().extension() == ".json") {
                        system->load_configuration(p.path().string());
                    }
                }

            system->init();
            system->set_node_tree_executor(create_node_tree_executor({}));
            /* Done! */
            UsdBasedNodeWidgetSettings desc;

            desc.json_path = json_path;
            desc.system = system;
            desc.stage = stage.get();

            std::unique_ptr<IWidget> node_widget =
                std::move(create_node_imgui_widget(desc));
            node_widget->SetCallBack(
                [&stage, json_path, system, render_bare](Window*, IWidget*) {
                    GeomPayload geom_global_params;
#ifdef GEOM_USD_EXTENSION
                    geom_global_params.stage = stage->get_usd_stage();
                    geom_global_params.prim_path = json_path;
#endif

                    geom_global_params.has_simulation = false;

                    // Pass pick event from UI to geometry payload
                    geom_global_params.pick = render_bare->consume_pick_event();

                    system->set_global_params(geom_global_params);
                    if (geom_global_params.pick) {
                        system->execute();
                    }
                });

            window->register_widget(std::move(node_widget));
        }
    });

    window->register_function_after_frame(
        [render_bare](Window* window) { render_bare->finish_render(); });

    window->register_function_after_frame(
        [&stage](Window* window) { stage->finish_tick(); });
    window->SetMaximized(true);
    window->run();

    unregister_cpp_type();

#ifdef GPU_GEOM_ALGORITHM
    deinit_gpu_geometry_algorithms();
#endif

    window.reset();
    stage.reset();
}
