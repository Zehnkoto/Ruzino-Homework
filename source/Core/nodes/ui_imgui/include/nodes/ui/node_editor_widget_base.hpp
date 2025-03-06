#pragma once
#define IMGUI_DEFINE_MATH_OPERATORS

#include "GUI/widget.h"
#include "blueprints/builders.h"
#include "blueprints/images.inl"
#include "blueprints/imgui_node_editor.h"
#include "blueprints/widgets.h"
#include "imgui.h"
#include "nodes/core/node_link.hpp"
#include "nodes/core/node_tree.hpp"
#include "nodes/core/socket.hpp"
#include "nodes/system/node_system.hpp"
#include "nodes/ui/imgui.hpp"
#include "nodes/ui/node_editor_widget_base.hpp"
#include "nvrhi/nvrhi.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
namespace ed = ax::NodeEditor;
namespace util = ax::NodeEditor::Utilities;
using namespace ax;
using ax::Widgets::IconType;

class NODES_UI_IMGUI_API NodeEditorWidgetBase : public IWidget {
   public:
    NodeEditorWidgetBase(const NodeWidgetSettings& desc);

    bool BuildUI() override;

   protected:
    virtual void initialize()
    {
    }
    virtual bool draw_socket_controllers(NodeSocket* input)
    {
        return true;
    }

    virtual void execute_tree(Node* node)
    {
        // Do nothing if base class
    }

    virtual void create_new_node(ImVec2 openPopupPosition)
    {  // Do nothing if base class
    }

    void connectLinks();
    static ImColor GetIconColor(SocketType type);
    void DrawPinIcon(const NodeSocket& pin, bool connected, int alpha);
    NodeTree* tree_;

    static const int m_PinIconSize = 20;

    bool first_draw = true;
    ed::EditorContext* m_Editor = nullptr;

    unsigned _level = 0;
    NodeSocket* newNodeLinkPin = nullptr;
    NodeSocket* newLinkPin = nullptr;

    NodeId contextNodeId = 0;
    LinkId contextLinkId = 0;
    SocketID contextPinId = 0;
    bool createNewNode = false;
    bool create_new_node_search_cursor;
    std::unique_ptr<NodeSystemStorage> storage_;

   private:
    static nvrhi::TextureHandle LoadTexture(
        const unsigned char* data,
        size_t buffer_size);
    nvrhi::TextureHandle m_HeaderBackground = nullptr;
};

USTC_CG_NAMESPACE_CLOSE_SCOPE