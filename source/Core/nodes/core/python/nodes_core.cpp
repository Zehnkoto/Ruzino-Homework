#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>

#include "nodes/core/node_link.hpp"
#include "nodes/core/node_tree.hpp"

namespace nb = nanobind;

using namespace USTC_CG;

NB_MODULE(nodes_core_py, m)
{
    // Enum bindings
    nb::enum_<PinKind>(m, "PinKind")
        .value("Input", PinKind::Input)
        .value("Output", PinKind::Output);

    // NodeTypeInfo bindings
    nb::class_<NodeTypeInfo>(m, "NodeTypeInfo")
        .def(nb::init<const char*>())
        .def_rw("id_name", &NodeTypeInfo::id_name)
        .def_rw("ui_name", &NodeTypeInfo::ui_name)
        .def_prop_rw(
            "color",
            [](NodeTypeInfo& n) {
                return std::make_tuple(
                    n.color[0], n.color[1], n.color[2], n.color[3]);
            },
            [](NodeTypeInfo& n,
               const std::tuple<float, float, float, float>& t) {
                n.color[0] = std::get<0>(t);
                n.color[1] = std::get<1>(t);
                n.color[2] = std::get<2>(t);
                n.color[3] = std::get<3>(t);
            })
        .def_rw("ALWAYS_REQUIRED", &NodeTypeInfo::ALWAYS_REQUIRED)
        .def_rw("INVISIBLE", &NodeTypeInfo::INVISIBLE)
        .def(
            "set_ui_name", &NodeTypeInfo::set_ui_name, nb::rv_policy::reference)
        .def(
            "set_declare_function",
            &NodeTypeInfo::set_declare_function,
            nb::rv_policy::reference)
        .def(
            "set_execution_function",
            &NodeTypeInfo::set_execution_function,
            nb::rv_policy::reference)
        .def(
            "set_always_required",
            &NodeTypeInfo::set_always_required,
            nb::rv_policy::reference);

    // Node bindings
    nb::class_<Node>(m, "Node")
        .def("getName", &Node::getName)
        .def_rw("ID", &Node::ID)
        .def_rw("ui_name", &Node::ui_name)
        // Use def_prop_rw for arrays
        .def_prop_rw(
            "Color",
            [](const Node& n) {
                return std::make_tuple(
                    n.Color[0], n.Color[1], n.Color[2], n.Color[3]);
            },
            [](Node& n, const std::tuple<float, float, float, float>& t) {
                n.Color[0] = std::get<0>(t);
                n.Color[1] = std::get<1>(t);
                n.Color[2] = std::get<2>(t);
                n.Color[3] = std::get<3>(t);
            })
        .def_prop_rw(
            "Size",
            [](const Node& n) { return std::make_tuple(n.Size[0], n.Size[1]); },
            [](Node& n, const std::tuple<unsigned, unsigned>& t) {
                n.Size[0] = std::get<0>(t);
                n.Size[1] = std::get<1>(t);
            })
        .def_rw("REQUIRED", &Node::REQUIRED)
        .def_rw("MISSING_INPUT", &Node::MISSING_INPUT)
        .def_rw("execution_failed", &Node::execution_failed)
        .def_rw("error_message", &Node::error_message)
        .def(
            "find_socket_group",
            &Node::find_socket_group,
            nb::rv_policy::reference)
        .def("set_error", &Node::set_error)
        .def("is_node_group", &Node::is_node_group)
        .def(
            "get_output_socket",
            &Node::get_output_socket,
            nb::rv_policy::reference)
        .def(
            "get_input_socket",
            &Node::get_input_socket,
            nb::rv_policy::reference)
        .def("find_socket", &Node::find_socket, nb::rv_policy::reference)
        .def("find_socket_id", &Node::find_socket_id)
        .def("find_socket_group_ids", &Node::find_socket_group_ids)
        .def("get_inputs", &Node::get_inputs, nb::rv_policy::reference)
        .def("get_outputs", &Node::get_outputs, nb::rv_policy::reference)
        .def("getInputConnections", &Node::getInputConnections)
        .def("getOutputConnections", &Node::getOutputConnections)
        .def("valid", &Node::valid)
        .def("refresh_node", &Node::refresh_node)
        .def("add_socket", &Node::add_socket, nb::rv_policy::reference)
        .def(
            "group_add_socket",
            &Node::group_add_socket,
            nb::rv_policy::reference)
        .def("group_remove_socket", &Node::group_remove_socket)
        .def_rw("paired_node", &Node::paired_node);

    // NodeGroup bindings
    nb::class_<NodeGroup, Node>(m, "NodeGroup")
        .def("is_node_group", &NodeGroup::is_node_group)
        .def(
            "group_add_socket",
            &NodeGroup::group_add_socket,
            nb::rv_policy::reference)
        .def("group_remove_socket", &NodeGroup::group_remove_socket)
        .def(
            "node_group_add_input_socket",
            &NodeGroup::node_group_add_input_socket)
        .def(
            "node_group_add_output_socket",
            &NodeGroup::node_group_add_output_socket);

    // NodeSocket bindings
    nb::class_<NodeSocket>(m, "NodeSocket")
        // Use def_prop_rw for char arrays to handle string conversion and
        // assignment safely
        .def_prop_rw(
            "identifier",
            [](const NodeSocket& s) { return std::string(s.identifier); },
            [](NodeSocket& s, const std::string& value) {
                strncpy(s.identifier, value.c_str(), sizeof(s.identifier) - 1);
                s.identifier[sizeof(s.identifier) - 1] =
                    '\0';  // Ensure null termination
            })
        .def_prop_rw(
            "ui_name",
            [](const NodeSocket& s) { return std::string(s.ui_name); },
            [](NodeSocket& s, const std::string& value) {
                strncpy(s.ui_name, value.c_str(), sizeof(s.ui_name) - 1);
                s.ui_name[sizeof(s.ui_name) - 1] =
                    '\0';  // Ensure null termination
            })
        // Read-only access for internal vectors and pointers
        .def_ro(
            "directly_linked_links",
            &NodeSocket::directly_linked_links,
            nb::rv_policy::reference_internal)
        .def_ro(
            "directly_linked_sockets",
            &NodeSocket::directly_linked_sockets,
            nb::rv_policy::reference_internal)
        .def_rw("ID", &NodeSocket::ID)  // Assuming SocketID is assignable
        .def_ro(
            "node",
            &NodeSocket::node,
            nb::rv_policy::reference)  // Return non-owning pointer
        .def_ro(
            "socket_group",
            &NodeSocket::socket_group,
            nb::rv_policy::reference)  // Return non-owning pointer
        // Read-write access for standard types
        .def_rw(
            "socket_group_identifier",
            &NodeSocket::socket_group_identifier)  // std::string works with
                                                   // def_rw
        .def_rw("in_out", &NodeSocket::in_out)
        .def_rw("optional", &NodeSocket::optional)
        // Methods
        .def("is_placeholder", &NodeSocket::is_placeholder);
    // Note: type_info, dataField, and storage are not exposed directly here.

    // SocketGroup bindings
    nb::class_<SocketGroup>(m, "SocketGroup")
        .def_ro("node", &SocketGroup::node)
        .def_rw("runtime_dynamic", &SocketGroup::runtime_dynamic)
        .def_rw("kind", &SocketGroup::kind)
        .def_rw("identifier", &SocketGroup::identifier)
        .def(
            "add_socket",
            &SocketGroup::add_socket,
            nb::arg("type_name"),
            nb::arg("identifier"),
            nb::arg("name"),
            nb::arg("need_to_propagate_sync") = true,
            nb::rv_policy::reference)
        .def("find_socket", &SocketGroup::find_socket, nb::rv_policy::reference)
        .def("add_sync_group", &SocketGroup::add_sync_group)
        .def(
            "remove_socket",
            static_cast<void (SocketGroup::*)(const char*, bool)>(
                &SocketGroup::remove_socket),
            nb::arg("identifier"),
            nb::arg("need_to_propagate_sync") = true)
        .def(
            "remove_socket",
            static_cast<void (SocketGroup::*)(NodeSocket*, bool)>(
                &SocketGroup::remove_socket),
            nb::arg("socket"),
            nb::arg("need_to_propagate_sync") = true);

    // NodeLink bindings
    nb::class_<NodeLink>(m, "NodeLink")
        .def_rw("ID", &NodeLink::ID)
        .def_ro("from_node", &NodeLink::from_node)
        .def_ro("to_node", &NodeLink::to_node)
        .def_ro("from_sock", &NodeLink::from_sock)
        .def_ro("to_sock", &NodeLink::to_sock)
        .def_rw("StartPinID", &NodeLink::StartPinID)
        .def_rw("EndPinID", &NodeLink::EndPinID)
        .def_ro("fromLink", &NodeLink::fromLink)
        .def_ro("nextLink", &NodeLink::nextLink)
        .def(
            "get_logical_from_socket",
            &NodeLink::get_logical_from_socket,
            nb::rv_policy::reference)
        .def(
            "get_logical_to_socket",
            &NodeLink::get_logical_to_socket,
            nb::rv_policy::reference)
        .def(
            "get_conversion_node",
            &NodeLink::get_conversion_node,
            nb::rv_policy::reference);

    // NodeTreeDescriptor bindings
    // Binds the NodeTreeDescriptor class. Assumes default lifetime management
    // unless specified otherwise.
    nb::class_<NodeTreeDescriptor>(m, "NodeTreeDescriptor")
        .def(nb::init<>())
        .def(
            "register_node",
            &NodeTreeDescriptor::register_node,
            nb::rv_policy::reference)
        .def(
            "register_conversion_name",
            &NodeTreeDescriptor::register_conversion_name,
            nb::rv_policy::reference)
        .def(
            "get_node_type",
            &NodeTreeDescriptor::get_node_type,
            nb::rv_policy::reference)
        .def_static(
            "conversion_node_name", &NodeTreeDescriptor::conversion_node_name)
        .def("can_convert", &NodeTreeDescriptor::can_convert)
        .def(
            "add_socket_group_syncronization",
            &NodeTreeDescriptor::add_socket_group_syncronization,
            nb::rv_policy::reference)
        .def(
            "require_syncronization",
            &NodeTreeDescriptor::require_syncronization);

    // NodeTree bindings
    // Binds the NodeTree class. Note: Removed std::shared_ptr holder type
    // specification. Ensure lifetime management aligns with C++ usage.
    nb::class_<NodeTree>(m, "NodeTree")
        // Constructors
        // Note: Constructor taking shared_ptr might need adjustment if NodeTree
        // isn't managed by shared_ptr on Python side.
        .def(
            nb::init<std::shared_ptr<NodeTreeDescriptor>>(),
            nb::arg("descriptor"))
        .def(nb::init<const NodeTree&>(), nb::arg("other"))  // Copy constructor

        // Methods returning potentially complex types or references/pointers
        .def(
            "get_inverse_tree",
            &NodeTree::get_inverse_tree)  // Assuming returns new tree or
                                          // shared_ptr
        .def_ro(  // Read-only access to internal socket list
            "input_sockets",
            &NodeTree::input_sockets,
            nb::rv_policy::reference_internal)  // Policy: reference internal
                                                // data
        .def_ro(  // Read-only access to internal socket list
            "output_sockets",
            &NodeTree::output_sockets,
            nb::rv_policy::reference_internal)  // Policy: reference internal
                                                // data
        .def(  // Returns reference to internal data structure
            "get_toposort_right_to_left",
            &NodeTree::get_toposort_right_to_left,
            nb::rv_policy::reference_internal)  // Policy: reference internal
                                                // data
        .def(  // Returns reference to internal data structure
            "get_toposort_left_to_right",
            &NodeTree::get_toposort_left_to_right,
            nb::rv_policy::reference_internal)  // Policy: reference internal
                                                // data

        // Methods returning raw pointers (Python does not own the result)
        .def(
            "find_node",
            static_cast<Node* (NodeTree::*)(NodeId) const>(
                &NodeTree::find_node),
            nb::arg("node_id"),
            nb::rv_policy::reference)  // Policy: return raw pointer, non-owning
        .def(
            "find_node",
            static_cast<Node* (NodeTree::*)(const char*) const>(
                &NodeTree::find_node),
            nb::arg("name"),
            nb::rv_policy::reference)  // Policy: return raw pointer, non-owning
        .def(
            "find_pin",
            &NodeTree::find_pin,
            nb::arg("pin_id"),
            nb::rv_policy::reference)  // Policy: return raw pointer, non-owning
        .def(
            "find_link",
            &NodeTree::find_link,
            nb::arg("link_id"),
            nb::rv_policy::reference)  // Policy: return raw pointer, non-owning
        .def(
            "add_node",
            &NodeTree::add_node,
            nb::arg("type_name"),
            nb::rv_policy::reference)  // Policy: return raw pointer, non-owning
        .def(                          // group_up overload returning NodeGroup*
            "group_up",
            static_cast<NodeGroup* (NodeTree::*)(std::vector<Node*>)>(
                &NodeTree::group_up),
            nb::arg("nodes"),
            nb::rv_policy::reference)  // Policy: return raw pointer, non-owning
        .def(                          // group_up overload returning NodeGroup*
            "group_up",
            static_cast<NodeGroup* (NodeTree::*)(std::vector<NodeId>)>(
                &NodeTree::group_up),
            nb::arg("node_ids"),
            nb::rv_policy::reference)  // Policy: return raw pointer, non-owning
        .def(                          // add_link overload returning NodeLink*
            "add_link",
            static_cast<NodeLink* (
                NodeTree::*)(Node*, Node*, const char*, const char*, bool)>(
                &NodeTree::add_link),
            nb::arg("fromnode"),
            nb::arg("tonode"),
            nb::arg("from_identifier"),
            nb::arg("to_identifier"),
            nb::arg("refresh_topology") = true,
            nb::rv_policy::reference)  // Policy: return raw pointer, non-owning
        .def(                          // add_link overload returning NodeLink*
            "add_link",
            static_cast<NodeLink* (
                NodeTree::*)(NodeSocket*, NodeSocket*, bool, bool)>(
                &NodeTree::add_link),
            nb::arg("fromsock"),
            nb::arg("tosock"),
            nb::arg("allow_relink_to_output") = false,
            nb::arg("refresh_topology") = true,
            nb::rv_policy::reference)  // Policy: return raw pointer, non-owning
        .def(                          // add_link overload returning NodeLink*
            "add_link",
            static_cast<NodeLink* (NodeTree::*)(SocketID, SocketID, bool)>(
                &NodeTree::add_link),
            nb::arg("startPinId"),
            nb::arg("endPinId"),
            nb::arg("refresh_topology") = true,
            nb::rv_policy::reference)  // Policy: return raw pointer, non-owning

        // Methods returning references (usually to self)
        .def(  // merge returns reference to self
            "merge",
            static_cast<NodeTree& (NodeTree::*)(const NodeTree&)>(
                &NodeTree::merge),
            nb::arg("other"),
            nb::rv_policy::reference)  // Policy: return reference to self

        // Read/write members (use with caution if pointer types)
        .def_rw("has_available_link_cycle", &NodeTree::has_available_link_cycle)
        .def_rw(
            "parent_node", &NodeTree::parent_node)  // Assuming Node*, careful
                                                    // assignment from Python

        // Other methods (void or simple return types)
        .def("input_socket_id", &NodeTree::input_socket_id)
        .def("output_socket_id", &NodeTree::output_socket_id)
        .def("clear", &NodeTree::clear)
        .def("is_pin_linked", &NodeTree::is_pin_linked, nb::arg("pin_id"))
        .def("add_base_id", &NodeTree::add_base_id, nb::arg("base_id"))
        .def("ungroup", &NodeTree::ungroup, nb::arg("group_node"))
        .def("UniqueID", &NodeTree::UniqueID)
        .def(
            "update_socket_vectors_and_owner_node",
            &NodeTree::update_socket_vectors_and_owner_node)
        .def("update_toposort", &NodeTree::update_toposort)
        .def("ensure_topology_cache", &NodeTree::ensure_topology_cache)
        .def(  // delete_link overload
            "delete_link",
            static_cast<void (NodeTree::*)(LinkId, bool, bool)>(
                &NodeTree::delete_link),
            nb::arg("linkId"),
            nb::arg("refresh_topology") = true,
            nb::arg("remove_from_group") = true)
        .def(  // delete_node overload
            "delete_node",
            static_cast<void (NodeTree::*)(Node*, bool)>(
                &NodeTree::delete_node),
            nb::arg("node"),
            nb::arg("allow_repeat_delete") = false)
        .def(  // delete_node overload
            "delete_node",
            static_cast<void (NodeTree::*)(NodeId, bool)>(
                &NodeTree::delete_node),
            nb::arg("nodeId"),
            nb::arg("allow_repeat_delete") = false)
        .def(
            "can_create_link",
            &NodeTree::can_create_link,
            nb::arg("from_sock"),
            nb::arg("to_sock"))
        .def_static(
            "can_create_direct_link",
            &NodeTree::can_create_direct_link,
            nb::arg("from_sock"),
            nb::arg("to_sock"))
        .def(
            "can_create_convert_link",
            &NodeTree::can_create_convert_link,
            nb::arg("from_sock"),
            nb::arg("to_sock"))
        .def("socket_count", &NodeTree::socket_count)
        .def(
            "get_descriptor",
            &NodeTree::get_descriptor)  // Returns shared_ptr, default policy
                                        // might need review
        .def("set_ui_settings", &NodeTree::set_ui_settings, nb::arg("settings"))
        .def("serialize", &NodeTree::serialize)
        .def("deserialize", &NodeTree::deserialize, nb::arg("data"))
        .def("SetDirty", &NodeTree::SetDirty, nb::arg("dirty") = true)
        .def("GetDirty", &NodeTree::GetDirty);

    // Expose related constants to Python module
    m.attr("NODE_GROUP_IDENTIFIER") = NODE_GROUP_IDENTIFIER;
    m.attr("NODE_GROUP_IN_IDENTIFIER") = NODE_GROUP_IN_IDENTIFIER;
    m.attr("NODE_GROUP_OUT_IDENTIFIER") = NODE_GROUP_OUT_IDENTIFIER;
    m.attr("OutsideInputsPH") = OutsideInputsPH;
    m.attr("OutsideOutputsPH") = OutsideOutputsPH;
    m.attr("InsideInputsPH") = InsideInputsPH;
    m.attr("InsideOutputsPH") = InsideOutputsPH;
}
