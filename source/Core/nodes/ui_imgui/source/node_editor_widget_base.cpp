#include "nodes/ui/node_editor_widget_base.hpp"

#include "entt/core/type_info.hpp"
#include "entt/meta/meta.hpp"
#include "nodes/core/node_exec_eager.hpp"
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>

#include <fstream>
#include <string>

#include "RHI/rhi.hpp"
#include "imgui.h"
#include "imgui/blueprint-utilities/builders.h"
#include "imgui/blueprint-utilities/images.inl"
#include "imgui/blueprint-utilities/widgets.h"
#include "imgui/imgui-node-editor/imgui_node_editor.h"
#include "imgui_internal.h"
#include "nodes/core/node_link.hpp"
#include "nodes/core/node_tree.hpp"
#include "nodes/core/socket.hpp"
#include "nodes/system/node_system.hpp"
#include "nodes/ui/imgui.hpp"
#include "stb_image.h"
#include "ui_imgui.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE
void NodeEditorWidgetBase::connectLinks()
{
    for (std::unique_ptr<NodeLink>& link : tree_->links) {
        auto type = link->from_sock->type_info;
        if (!type)
            type = link->to_sock->type_info;

        ImColor color = GetIconColor(type);

        auto linkId = link->ID;
        auto startPin = link->StartPinID;
        auto endPin = link->EndPinID;

        // If there is an invisible node after the link, use the first as
        // the id for the ui link
        if (link->nextLink) {
            endPin = link->nextLink->to_sock->ID;
        }

        ed::Link(linkId, startPin, endPin, color, 2.0f);
    }
}

ImColor NodeEditorWidgetBase::GetIconColor(SocketType type)
{
    auto hashColorComponent = [](const std::string& prefix,
                                 const std::string& typeName) {
        return static_cast<int>(
            entt::hashed_string{ (prefix + typeName).c_str() }.value());
    };

    const std::string typeName = get_type_name(type);
    auto hashValue_r = hashColorComponent("r", typeName);
    auto hashValue_g = hashColorComponent("g", typeName);
    auto hashValue_b = hashColorComponent("b", typeName);

    return ImColor(
        hashValue_r % 192 + 63, hashValue_g % 192 + 63, hashValue_b % 192 + 63);
}

void NodeEditorWidgetBase::DrawPinIcon(
    const NodeSocket& pin,
    bool connected,
    int alpha)
{
    IconType iconType;

    ImColor color = GetIconColor(pin.type_info);

    if (!pin.type_info) {
        if (pin.directly_linked_sockets.size() > 0) {
            color = GetIconColor(pin.directly_linked_sockets[0]->type_info);
        }
    }

    color.Value.w = alpha / 255.0f;
    iconType = IconType::Circle;

    Widgets::Icon(
        ImVec2(
            static_cast<float>(m_PinIconSize),
            static_cast<float>(m_PinIconSize)),
        iconType,
        connected,
        color,
        ImColor(32, 32, 32, alpha));
}

USTC_CG_NAMESPACE_CLOSE_SCOPE