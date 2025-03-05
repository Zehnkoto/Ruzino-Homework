#pragma once
#include <string>

#include "GUI/widget.h"
#include "RHI/rhi.hpp"
#include "api.h"
#include "imgui.h"
#include "imgui/blueprint-utilities/builders.h"
#include "imgui/blueprint-utilities/images.inl"
#include "imgui/blueprint-utilities/widgets.h"
#include "imgui/imgui-node-editor/imgui_node_editor.h"
#include "nodes/core/node_link.hpp"
#include "nodes/core/node_tree.hpp"
#include "nodes/core/socket.hpp"
#include "nodes/system/node_system.hpp"
#include "nodes/ui/imgui.hpp"
#include "nodes/ui/node_editor_widget_base.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE
class NodeEditorWidgetBase : public IWidget {
   protected:
    void connectLinks();
    static ImColor GetIconColor(SocketType type);
    void DrawPinIcon(const NodeSocket& pin, bool connected, int alpha);
    NodeTree* tree_;

    static const int m_PinIconSize = 20;
};

USTC_CG_NAMESPACE_CLOSE_SCOPE