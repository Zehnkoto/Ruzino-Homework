#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "nodes/core/node.hpp"  // Required for Node* type in execute
#include "nodes/core/node_tree.hpp"  // Required for NodeTree* and unique_ptr<NodeTree>
#include "nodes/system/node_system.hpp"
#include "nodes/system/node_system_dl.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace USTC_CG;

// Trampoline class for NodeSystem if Python inheritance is needed later.
// For now, we only bind the C++ implementation NodeDynamicLoadingSystem.
/*
class PyNodeSystem : public NodeSystem {
public:
    using NodeSystem::NodeSystem; // Inherit constructors

    // Override pure virtual functions
    bool load_configuration(const std::filesystem::path& config) override {
        NB_OVERRIDE_PURE(
            bool,              // Return type
            NodeSystem,        // Parent class
            "load_configuration", // Python method name
            config             // Arguments
        );
    }

    std::shared_ptr<NodeTreeDescriptor> node_tree_descriptor() override {
         NB_OVERRIDE_PURE(
            std::shared_ptr<NodeTreeDescriptor>, // Return type
            NodeSystem,                          // Parent class
            "node_tree_descriptor"               // Python method name
                                                 // No arguments
        );
    }

    // Override virtual functions if needed
    void set_node_tree_executor(std::unique_ptr<NodeTreeExecutor> executor)
override { NB_OVERRIDE( void,               // Return type NodeSystem, // Parent
class "set_node_tree_executor", // Python method name std::move(executor) //
Arguments
        );
    }

     void execute(bool is_ui_execution = false, Node* required_node = nullptr)
const override { NB_OVERRIDE( void,               // Return type NodeSystem, //
Parent class "execute",          // Python method name is_ui_execution,    //
Arguments required_node
        );
    }
};
*/

NB_MODULE(nodes_system_py, m)
{
    // Ensure Node, NodeTree, NodeTreeExecutor types are known.
    // Assuming they are bound in other modules and nanobind handles type
    // resolution. nb::class_<Node>(m, "Node"); // Placeholder if not bound
    // elsewhere nb::class_<NodeTree>(m, "NodeTree"); // Placeholder if not
    // bound elsewhere nb::class_<NodeTreeExecutor>(m, "NodeTreeExecutor"); //
    // Placeholder if not bound elsewhere

    // Bind the base class NodeSystem
    // We don't instantiate NodeSystem directly as it's abstract.
    // If Python inheritance was needed, we'd use PyNodeSystem trampoline.
    nb::class_<NodeSystem /*, PyNodeSystem */>(m, "NodeSystem")
        // .def(nb::init<>()) // Abstract class, no public constructor
        .def(
            "init",
            static_cast<void (NodeSystem::*)()>(&NodeSystem::init),
            "Initialize the node system with a default empty tree.")
        .def(
            "init",
            [](NodeSystem& self, std::unique_ptr<NodeTree> tree) {
                // Need to manage lifetime if tree is passed from Python
                // For now, assume tree ownership is transferred fully.
                self.init(std::move(tree));
            },
            "tree"_a.none(),  // Allow passing None to potentially clear/reset
            "Initialize the node system with a specific node tree.")
        // set_node_tree_executor might be complex due to unique_ptr ownership
        // .def("set_node_tree_executor", &NodeSystem::set_node_tree_executor)
        .def(
            "load_configuration",
            &NodeSystem::load_configuration,
            "config"_a,
            "Load node tree configuration from a file (pure virtual).")
        .def(
            "execute",
            &NodeSystem::execute,
            "is_ui_execution"_a = false,
            "required_node"_a.none() =
                nullptr,  // Use .none() for optional pointer
            nb::call_guard<nb::gil_scoped_release>(),
            "Execute the node tree.")
        .def(
            "get_node_tree",
            &NodeSystem::get_node_tree,
            nb::rv_policy::reference_internal,
            "Get the node tree associated with this system.")
        .def(
            "get_node_tree_executor",
            &NodeSystem::get_node_tree_executor,
            nb::rv_policy::reference_internal,
            "Get the node tree executor.")
        .def_rw(
            "allow_ui_execution",
            &NodeSystem::allow_ui_execution,
            "Flag to allow execution triggered by UI interactions.")
        .def("finalize", &NodeSystem::finalize, "Finalize the node system.")
        // node_tree_descriptor is protected/pure virtual in base, implemented
        // in derived set_global_params is a template, difficult to bind
        // directly. Expose specific types if needed, e.g.:
        // .def("set_global_params_int", [](NodeSystem& self, int params) {
        //     self.set_global_params(params);
        // })
        ;

    // Bind NodeDynamicLoadingSystem inheriting from NodeSystem
    nb::class_<NodeDynamicLoadingSystem, NodeSystem>(
        m, "NodeDynamicLoadingSystem")
        // Don't bind constructor, use factory function
        // create_dynamic_loading_system .def(nb::init<>())
        .def(
            "load_configuration",
            &NodeDynamicLoadingSystem::load_configuration,
            "config"_a,
            "Load node tree configuration and dynamic libraries from a file.");
    // Other methods are inherited from NodeSystem binding

    // Bind the factory function
    m.def(
        "create_dynamic_loading_system",
        &create_dynamic_loading_system,
        nb::rv_policy::take_ownership,  // The function returns a shared_ptr
        "Create a NodeSystem instance that supports dynamic loading of nodes.");
}
