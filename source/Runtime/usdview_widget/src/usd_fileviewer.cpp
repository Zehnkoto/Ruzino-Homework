#define IMGUI_DEFINE_MATH_OPERATORS

#include "widgets/usdtree/usd_fileviewer.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <future>
#include <map>
#include <vector>

#include "GUI/ImGuiFileDialog.h"
#include "imgui.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/base/gf/rotation.h"
#include "pxr/base/vt/typeHeaders.h"
#include "pxr/base/vt/visitValue.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/property.h"
#include "pxr/usd/usdGeom/xformOp.h"
#include "stage/stage.hpp"
USTC_CG_NAMESPACE_OPEN_SCOPE
void UsdFileViewer::ShowFileTree()
{
    auto root = stage->get_usd_stage()->GetPseudoRoot();
    ImGuiTableFlags flags = ImGuiTableFlags_SizingFixedFit |
                            ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                            ImGuiTableFlags_Resizable;
    if (ImGui::BeginTable("stage_table", 2, flags)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch);
        DrawChild(root, true);

        ImGui::EndTable();
    }
}

void UsdFileViewer::ShowPrimInfo()
{
    using namespace pxr;
    ImGuiTableFlags flags = ImGuiTableFlags_SizingFixedFit |
                            ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                            ImGuiTableFlags_Resizable;
    if (ImGui::BeginTable("table", 3, flags)) {
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn(
            "Property Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableHeadersRow();
        UsdPrim prim = stage->get_usd_stage()->GetPrimAtPath(selected);
        if (prim) {
            auto attributes = prim.GetAttributes();
            std::vector<std::future<std::string>> futures;

            for (auto&& attr : attributes) {
                futures.push_back(std::async(std::launch::async, [&attr]() {
                    VtValue v;
                    attr.Get(&v);
                    if (v.IsArrayValued()) {
                        std::string displayString;
                        auto formatArray = [&](auto array) {
                            size_t arraySize = array.size();
                            size_t displayCount = 3;
                            for (size_t i = 0;
                                 i < std::min(displayCount, arraySize);
                                 ++i) {
                                displayString += TfStringify(array[i]) + ", \n";
                            }
                            if (arraySize > 2 * displayCount) {
                                displayString += "... \n";
                            }
                            for (size_t i = std::max(
                                     displayCount, arraySize - displayCount);
                                 i < arraySize;
                                 ++i) {
                                displayString += TfStringify(array[i]) + ", \n";
                            }
                            if (!displayString.empty()) {
                                displayString.pop_back();
                                displayString.pop_back();
                                displayString.pop_back();
                            }
                        };
                        if (v.IsHolding<VtArray<double>>()) {
                            formatArray(v.Get<VtArray<double>>());
                        }
                        else if (v.IsHolding<VtArray<float>>()) {
                            formatArray(v.Get<VtArray<float>>());
                        }
                        else if (v.IsHolding<VtArray<int>>()) {
                            formatArray(v.Get<VtArray<int>>());
                        }
                        else if (v.IsHolding<VtArray<unsigned int>>()) {
                            formatArray(v.Get<VtArray<unsigned int>>());
                        }
                        else if (v.IsHolding<VtArray<int64_t>>()) {
                            formatArray(v.Get<VtArray<int64_t>>());
                        }
                        else if (v.IsHolding<VtArray<uint64_t>>()) {
                            formatArray(v.Get<VtArray<uint64_t>>());
                        }
                        else if (v.IsHolding<VtArray<GfMatrix4d>>()) {
                            formatArray(v.Get<VtArray<GfMatrix4d>>());
                        }
                        else if (v.IsHolding<VtArray<GfMatrix4f>>()) {
                            formatArray(v.Get<VtArray<GfMatrix4f>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec2d>>()) {
                            formatArray(v.Get<VtArray<GfVec2d>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec2f>>()) {
                            formatArray(v.Get<VtArray<GfVec2f>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec2i>>()) {
                            formatArray(v.Get<VtArray<GfVec2i>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec3d>>()) {
                            formatArray(v.Get<VtArray<GfVec3d>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec3f>>()) {
                            formatArray(v.Get<VtArray<GfVec3f>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec3i>>()) {
                            formatArray(v.Get<VtArray<GfVec3i>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec4d>>()) {
                            formatArray(v.Get<VtArray<GfVec4d>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec4f>>()) {
                            formatArray(v.Get<VtArray<GfVec4f>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec4i>>()) {
                            formatArray(v.Get<VtArray<GfVec4i>>());
                        }
                        else {
                            displayString = "Unsupported array type";
                        }
                        return displayString;
                    }
                    else {
                        return VtVisitValue(
                            v, [](auto&& v) { return TfStringify(v); });
                    }
                }));
            }

            auto relations = prim.GetRelationships();
            std::vector<std::future<std::string>> relation_futures;
            for (auto&& relation : relations) {
                relation_futures.push_back(
                    std::async(std::launch::async, [&relation]() {
                        std::string displayString;
                        SdfPathVector relation_targets;
                        relation.GetTargets(&relation_targets);
                        for (auto&& target : relation_targets) {
                            displayString += target.GetString() + ",\n";
                        }
                        if (!displayString.empty()) {
                            displayString.pop_back();
                            displayString.pop_back();
                        }
                        return displayString;
                    }));
            }
            auto displayRow = [](const char* type,
                                 const std::string& name,
                                 const std::string& value) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(type);
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(name.c_str());
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted(value.c_str());
            };

            for (size_t i = 0; i < attributes.size(); ++i) {
                displayRow(
                    "A", attributes[i].GetName().GetString(), futures[i].get());
            }

            for (size_t i = 0; i < relations.size(); ++i) {
                displayRow(
                    "R",
                    relations[i].GetName().GetString(),
                    relation_futures[i].get());
            }
        }
        ImGui::EndTable();
    }
}

void UsdFileViewer::EditValue()
{
    using namespace pxr;
    UsdPrim prim = stage->get_usd_stage()->GetPrimAtPath(selected);
    if (!prim) {
        ImGui::TextDisabled("No prim selected");
        return;
    }

    // Prim Info Section
    if (ImGui::CollapsingHeader("Prim Info", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Path: %s", prim.GetPath().GetString().c_str());
        ImGui::Text("Type: %s", prim.GetTypeName().GetText());
        ImGui::Text("Active: %s", prim.IsActive() ? "Yes" : "No");
        ImGui::Separator();
    }

    // Transform controls in a collapsible section
    auto xformable = UsdGeomXformable::Get(stage->get_usd_stage(), selected);
    if (xformable) {
        if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
            bool rst_stack;
            auto xform_op = xformable.GetOrderedXformOps(&rst_stack);
            
            // Initialize transform if not exists
            if (xform_op.size() == 0) {
                GfMatrix4d mat = GfMatrix4d(1);
                auto trans = xformable.AddTransformOp();
                trans.Set(mat);
                xformable.SetXformOpOrder({ trans });
                xform_op = xformable.GetOrderedXformOps(&rst_stack);
            }
            
            if (xform_op.size() == 1 &&
                xform_op[0].GetOpType() == UsdGeomXformOp::TypeTransform) {
                auto trans = xform_op[0];
                GfMatrix4d mat;
                trans.Get(&mat);
                
                // Decompose matrix into translation, rotation, scale
                GfVec3d translation = mat.ExtractTranslation();
                
                // Extract scale by getting the length of each basis vector
                GfVec3d scaleVec(
                    GfVec3d(mat[0][0], mat[0][1], mat[0][2]).GetLength(),
                    GfVec3d(mat[1][0], mat[1][1], mat[1][2]).GetLength(),
                    GfVec3d(mat[2][0], mat[2][1], mat[2][2]).GetLength()
                );
                
                // Check if we need to recompute Euler angles (new prim selected or first time)
                GfVec3d eulerXYZ;
                if (!has_cached_transform || cached_transform_path != selected) {
                    // Extract rotation using GfRotation (more stable than Euler angles)
                    GfMatrix3d rotMat3 = mat.ExtractRotationMatrix();
                    
                    // Calculate Euler angles with proper handling of edge cases
                    double sy = -rotMat3[2][0];
                    
                    // Clamp to avoid numerical issues
                    sy = GfClamp(sy, -1.0, 1.0);
                    
                    if (std::abs(sy) < 0.99999) {
                        // Normal case
                        eulerXYZ[0] = atan2(rotMat3[2][1], rotMat3[2][2]) * 180.0 / M_PI;  // X
                        eulerXYZ[1] = asin(sy) * 180.0 / M_PI;  // Y
                        eulerXYZ[2] = atan2(rotMat3[1][0], rotMat3[0][0]) * 180.0 / M_PI;  // Z
                    } else {
                        // Gimbal lock case
                        eulerXYZ[0] = atan2(-rotMat3[0][1], rotMat3[1][1]) * 180.0 / M_PI;
                        eulerXYZ[1] = sy > 0 ? 89.9 : -89.9;  // Clamp to safe range
                        eulerXYZ[2] = 0.0;
                    }
                    
                    // Cache the result
                    cached_euler_angles = eulerXYZ;
                    cached_transform_path = selected;
                    has_cached_transform = true;
                } else {
                    // Use cached values to avoid jitter
                    eulerXYZ = cached_euler_angles;
                }
                
                bool modified = false;
                
                ImGui::PushItemWidth(-1);
                
                // Translation
                ImGui::Text("Translation");
                float trans_tmp[3] = { 
                    static_cast<float>(translation[0]),
                    static_cast<float>(translation[1]),
                    static_cast<float>(translation[2])
                };
                if (ImGui::DragFloat("X##trans", &trans_tmp[0], 0.1f, -1000.f, 1000.f, "%.3f")) {
                    translation[0] = trans_tmp[0];
                    modified = true;
                }
                if (ImGui::DragFloat("Y##trans", &trans_tmp[1], 0.1f, -1000.f, 1000.f, "%.3f")) {
                    translation[1] = trans_tmp[1];
                    modified = true;
                }
                if (ImGui::DragFloat("Z##trans", &trans_tmp[2], 0.1f, -1000.f, 1000.f, "%.3f")) {
                    translation[2] = trans_tmp[2];
                    modified = true;
                }
                
                ImGui::Spacing();
                
                // Rotation (Euler angles in degrees)
                ImGui::Text("Rotation (degrees)");
                float rot_tmp[3] = {
                    static_cast<float>(eulerXYZ[0]),
                    static_cast<float>(eulerXYZ[1]),
                    static_cast<float>(eulerXYZ[2])
                };
                
                bool rotModified = false;
                if (ImGui::DragFloat("X##rot", &rot_tmp[0], 1.0f, -180.f, 180.f, "%.1f°")) {
                    eulerXYZ[0] = rot_tmp[0];
                    rotModified = true;
                }
                if (ImGui::DragFloat("Y##rot", &rot_tmp[1], 1.0f, -89.9f, 89.9f, "%.1f°")) {
                    eulerXYZ[1] = rot_tmp[1];
                    rotModified = true;
                }
                if (ImGui::DragFloat("Z##rot", &rot_tmp[2], 1.0f, -180.f, 180.f, "%.1f°")) {
                    eulerXYZ[2] = rot_tmp[2];
                    rotModified = true;
                }
                
                if (rotModified) {
                    modified = true;
                    // Update cache with new values
                    cached_euler_angles = eulerXYZ;
                }
                
                ImGui::Spacing();
                
                // Scale
                ImGui::Text("Scale");
                float scale_tmp[3] = {
                    static_cast<float>(scaleVec[0]),
                    static_cast<float>(scaleVec[1]),
                    static_cast<float>(scaleVec[2])
                };
                if (ImGui::DragFloat("X##scale", &scale_tmp[0], 0.01f, 0.001f, 100.f, "%.3f")) {
                    scaleVec[0] = scale_tmp[0];
                    modified = true;
                }
                if (ImGui::DragFloat("Y##scale", &scale_tmp[1], 0.01f, 0.001f, 100.f, "%.3f")) {
                    scaleVec[1] = scale_tmp[1];
                    modified = true;
                }
                if (ImGui::DragFloat("Z##scale", &scale_tmp[2], 0.01f, 0.001f, 100.f, "%.3f")) {
                    scaleVec[2] = scale_tmp[2];
                    modified = true;
                }
                
                // Uniform scale button
                if (ImGui::Button("Uniform Scale")) {
                    float uniformScale = (scale_tmp[0] + scale_tmp[1] + scale_tmp[2]) / 3.0f;
                    scaleVec[0] = scaleVec[1] = scaleVec[2] = uniformScale;
                    modified = true;
                }
                
                if (modified) {
                    // Reconstruct matrix from components
                    // First create scale matrix
                    GfMatrix4d scaleMat(1);
                    scaleMat[0][0] = scaleVec[0];
                    scaleMat[1][1] = scaleVec[1];
                    scaleMat[2][2] = scaleVec[2];
                    
                    // Create rotation matrix from Euler angles using GfRotation
                    // This is more stable and avoids gimbal lock issues
                    GfRotation rotX_rotation(GfVec3d(1, 0, 0), eulerXYZ[0]);
                    GfRotation rotY_rotation(GfVec3d(0, 1, 0), eulerXYZ[1]);
                    GfRotation rotZ_rotation(GfVec3d(0, 0, 1), eulerXYZ[2]);
                    
                    // Combine rotations: Z * Y * X (standard XYZ Euler order)
                    GfRotation combinedRotation = rotZ_rotation * rotY_rotation * rotX_rotation;
                    
                    // Convert to 3x3 matrix then to 4x4
                    GfMatrix3d rotMat3(combinedRotation.GetQuat());
                    GfMatrix4d rotMat4(1);
                    rotMat4.SetRotate(rotMat3);
                    
                    // Combine: Scale * Rotation
                    GfMatrix4d newMat = rotMat4 * scaleMat;
                    
                    // Set translation
                    newMat.SetTranslateOnly(translation);
                    
                    // Verify matrix is valid before setting
                    if (!std::isnan(newMat[0][0]) && !std::isnan(newMat[3][3])) {
                        try {
                            trans.Set(newMat);
                        } catch (...) {
                            spdlog::warn("Failed to set transform matrix");
                        }
                    } else {
                        spdlog::warn("Invalid transform matrix detected, skipping update");
                    }
                }
                
                ImGui::PopItemWidth();
            }
            else if (xform_op.size() > 1) {
                ImGui::TextWrapped("Complex transform stack detected. Only simple transform editing is supported.");
                ImGui::TextDisabled("Transform stack has %zu operations", xform_op.size());
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
        }
    }

    // Attributes in a collapsible section
    if (prim) {
        if (ImGui::CollapsingHeader("Attributes", ImGuiTreeNodeFlags_DefaultOpen)) {
            auto attributes = prim.GetAttributes();
            
            // Group attributes by category
            std::map<std::string, std::vector<UsdAttribute>> attributeGroups;
            for (auto&& attr : attributes) {
                std::string attrName = attr.GetName().GetString();
                
                // Skip xformOp attributes as they're handled in Transform section
                if (attrName.find("xformOp") != std::string::npos) {
                    continue;
                }
                
                // Categorize by namespace
                std::string category = "General";
                size_t colonPos = attrName.find(':');
                if (colonPos != std::string::npos) {
                    category = attrName.substr(0, colonPos);
                }
                
                attributeGroups[category].push_back(attr);
            }
            
            // Display attributes by category
            for (auto& [category, attrs] : attributeGroups) {
                if (attrs.empty()) continue;
                
                if (ImGui::TreeNodeEx(category.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                    for (auto&& attr : attrs) {
                        VtValue v;
                        attr.Get(&v);
                        std::string attrName = attr.GetName().GetString();
                        std::string label = attrName + "##" + attrName;

                        ImGui::PushID(label.c_str());
                        ImGui::AlignTextToFramePadding();
                        ImGui::Text("%s", attrName.c_str());
                        ImGui::SameLine(200);
                        ImGui::SetNextItemWidth(-1);

                        if (v.IsHolding<double>()) {
                            double value = v.Get<double>();
                            if (ImGui::DragScalar("##value", ImGuiDataType_Double, &value, 0.01, nullptr, nullptr, "%.3f")) {
                                attr.Set(value);
                            }
                        }
                        else if (v.IsHolding<float>()) {
                            float value = v.Get<float>();
                            if (ImGui::DragFloat("##value", &value, 0.01f, 0.0f, 0.0f, "%.3f")) {
                                attr.Set(value);
                            }
                        }
                        else if (v.IsHolding<int>()) {
                            int value = v.Get<int>();
                            if (ImGui::DragInt("##value", &value, 1.0f)) {
                                attr.Set(value);
                            }
                        }
                        else if (v.IsHolding<unsigned int>()) {
                            unsigned int value = v.Get<unsigned int>();
                            if (ImGui::DragScalar("##value", ImGuiDataType_U32, &value, 1.0f)) {
                                attr.Set(value);
                            }
                        }
                        else if (v.IsHolding<int64_t>()) {
                            int64_t value = v.Get<int64_t>();
                            if (ImGui::DragScalar("##value", ImGuiDataType_S64, &value, 1.0f)) {
                                attr.Set(value);
                            }
                        }
                        else if (v.IsHolding<GfVec2f>()) {
                            GfVec2f value = v.Get<GfVec2f>();
                            if (ImGui::DragFloat2("##value", value.data(), 0.01f)) {
                                attr.Set(value);
                            }
                        }
                        else if (v.IsHolding<GfVec3f>()) {
                            GfVec3f value = v.Get<GfVec3f>();
                            
                            // Special handling for color attributes
                            if (attrName.find("color") != std::string::npos || 
                                attrName.find("Color") != std::string::npos) {
                                if (ImGui::ColorEdit3("##value", value.data())) {
                                    attr.Set(value);
                                }
                            } else {
                                if (ImGui::DragFloat3("##value", value.data(), 0.01f)) {
                                    attr.Set(value);
                                }
                            }
                        }
                        else if (v.IsHolding<GfVec4f>()) {
                            GfVec4f value = v.Get<GfVec4f>();
                            
                            // Special handling for color attributes
                            if (attrName.find("color") != std::string::npos || 
                                attrName.find("Color") != std::string::npos) {
                                if (ImGui::ColorEdit4("##value", value.data())) {
                                    attr.Set(value);
                                }
                            } else {
                                if (ImGui::DragFloat4("##value", value.data(), 0.01f)) {
                                    attr.Set(value);
                                }
                            }
                        }
                        else if (v.IsHolding<GfVec2i>()) {
                            GfVec2i value = v.Get<GfVec2i>();
                            if (ImGui::DragInt2("##value", value.data())) {
                                attr.Set(value);
                            }
                        }
                        else if (v.IsHolding<GfVec3i>()) {
                            GfVec3i value = v.Get<GfVec3i>();
                            if (ImGui::DragInt3("##value", value.data())) {
                                attr.Set(value);
                            }
                        }
                        else if (v.IsHolding<GfVec4i>()) {
                            GfVec4i value = v.Get<GfVec4i>();
                            if (ImGui::DragInt4("##value", value.data())) {
                                attr.Set(value);
                            }
                        }
                        else if (v.IsHolding<GfVec2d>()) {
                            GfVec2d value = v.Get<GfVec2d>();
                            float tmp[2] = { static_cast<float>(value[0]),
                                           static_cast<float>(value[1]) };
                            if (ImGui::DragFloat2("##value", tmp, 0.01f)) {
                                value[0] = static_cast<double>(tmp[0]);
                                value[1] = static_cast<double>(tmp[1]);
                                attr.Set(value);
                            }
                        }
                        else if (v.IsHolding<GfVec3d>()) {
                            GfVec3d value = v.Get<GfVec3d>();
                            float tmp[3] = { static_cast<float>(value[0]),
                                           static_cast<float>(value[1]),
                                           static_cast<float>(value[2]) };

                            // Special handling for color attributes
                            bool isColor = attrName.find("color") != std::string::npos || 
                                          attrName.find("Color") != std::string::npos;
                            
                            if (isColor) {
                                if (ImGui::ColorEdit3("##value", tmp)) {
                                    value[0] = static_cast<double>(tmp[0]);
                                    value[1] = static_cast<double>(tmp[1]);
                                    value[2] = static_cast<double>(tmp[2]);
                                    attr.Set(value);
                                }
                            } else {
                                if (ImGui::DragFloat3("##value", tmp, 0.01f)) {
                                    value[0] = static_cast<double>(tmp[0]);
                                    value[1] = static_cast<double>(tmp[1]);
                                    value[2] = static_cast<double>(tmp[2]);
                                    attr.Set(value);
                                }
                            }
                        }
                        else if (v.IsHolding<GfVec4d>()) {
                            GfVec4d value = v.Get<GfVec4d>();
                            float tmp[4] = { static_cast<float>(value[0]),
                                           static_cast<float>(value[1]),
                                           static_cast<float>(value[2]),
                                           static_cast<float>(value[3]) };

                            // Special handling for color attributes
                            bool isColor = attrName.find("color") != std::string::npos || 
                                          attrName.find("Color") != std::string::npos;
                            
                            if (isColor) {
                                if (ImGui::ColorEdit4("##value", tmp)) {
                                    value[0] = static_cast<double>(tmp[0]);
                                    value[1] = static_cast<double>(tmp[1]);
                                    value[2] = static_cast<double>(tmp[2]);
                                    value[3] = static_cast<double>(tmp[3]);
                                    attr.Set(value);
                                }
                            } else {
                                if (ImGui::DragFloat4("##value", tmp, 0.01f)) {
                                    value[0] = static_cast<double>(tmp[0]);
                                    value[1] = static_cast<double>(tmp[1]);
                                    value[2] = static_cast<double>(tmp[2]);
                                    value[3] = static_cast<double>(tmp[3]);
                                    attr.Set(value);
                                }
                            }
                        }
                        else {
                            // Read-only display for unsupported types
                            if (v.IsArrayValued()) {
                                // For arrays, show type and size
                                size_t arraySize = v.GetArraySize();
                                ImGui::TextDisabled("%s [%zu elements]", v.GetTypeName().c_str(), arraySize);
                                
                                // Show preview of elements (first 3 and last 3)
                                size_t previewCount = std::min<size_t>(3, arraySize);
                                bool hasMore = arraySize > 6;
                                
                                ImGui::Indent();
                                
                                if (v.IsHolding<VtArray<float>>()) {
                                    auto arr = v.Get<VtArray<float>>();
                                    for (size_t i = 0; i < previewCount; ++i) {
                                        ImGui::Text("[%zu]: %.3f", i, arr[i]);
                                    }
                                    if (hasMore) {
                                        ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6);
                                        for (size_t i = arraySize - previewCount; i < arraySize; ++i) {
                                            ImGui::Text("[%zu]: %.3f", i, arr[i]);
                                        }
                                    }
                                }
                                else if (v.IsHolding<VtArray<double>>()) {
                                    auto arr = v.Get<VtArray<double>>();
                                    for (size_t i = 0; i < previewCount; ++i) {
                                        ImGui::Text("[%zu]: %.3f", i, arr[i]);
                                    }
                                    if (hasMore) {
                                        ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6);
                                        for (size_t i = arraySize - previewCount; i < arraySize; ++i) {
                                            ImGui::Text("[%zu]: %.3f", i, arr[i]);
                                        }
                                    }
                                }
                                else if (v.IsHolding<VtArray<int>>()) {
                                    auto arr = v.Get<VtArray<int>>();
                                    for (size_t i = 0; i < previewCount; ++i) {
                                        ImGui::Text("[%zu]: %d", i, arr[i]);
                                    }
                                    if (hasMore) {
                                        ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6);
                                        for (size_t i = arraySize - previewCount; i < arraySize; ++i) {
                                            ImGui::Text("[%zu]: %d", i, arr[i]);
                                        }
                                    }
                                }
                                else if (v.IsHolding<VtArray<GfVec2f>>()) {
                                    auto arr = v.Get<VtArray<GfVec2f>>();
                                    for (size_t i = 0; i < previewCount; ++i) {
                                        ImGui::Text("[%zu]: (%.3f, %.3f)", i, arr[i][0], arr[i][1]);
                                    }
                                    if (hasMore) {
                                        ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6);
                                        for (size_t i = arraySize - previewCount; i < arraySize; ++i) {
                                            ImGui::Text("[%zu]: (%.3f, %.3f)", i, arr[i][0], arr[i][1]);
                                        }
                                    }
                                }
                                else if (v.IsHolding<VtArray<GfVec2d>>()) {
                                    auto arr = v.Get<VtArray<GfVec2d>>();
                                    for (size_t i = 0; i < previewCount; ++i) {
                                        ImGui::Text("[%zu]: (%.3f, %.3f)", i, arr[i][0], arr[i][1]);
                                    }
                                    if (hasMore) {
                                        ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6);
                                        for (size_t i = arraySize - previewCount; i < arraySize; ++i) {
                                            ImGui::Text("[%zu]: (%.3f, %.3f)", i, arr[i][0], arr[i][1]);
                                        }
                                    }
                                }
                                else if (v.IsHolding<VtArray<GfVec3f>>()) {
                                    auto arr = v.Get<VtArray<GfVec3f>>();
                                    for (size_t i = 0; i < previewCount; ++i) {
                                        ImGui::Text("[%zu]: (%.3f, %.3f, %.3f)", i, arr[i][0], arr[i][1], arr[i][2]);
                                    }
                                    if (hasMore) {
                                        ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6);
                                        for (size_t i = arraySize - previewCount; i < arraySize; ++i) {
                                            ImGui::Text("[%zu]: (%.3f, %.3f, %.3f)", i, arr[i][0], arr[i][1], arr[i][2]);
                                        }
                                    }
                                }
                                else if (v.IsHolding<VtArray<GfVec3d>>()) {
                                    auto arr = v.Get<VtArray<GfVec3d>>();
                                    for (size_t i = 0; i < previewCount; ++i) {
                                        ImGui::Text("[%zu]: (%.3f, %.3f, %.3f)", i, arr[i][0], arr[i][1], arr[i][2]);
                                    }
                                    if (hasMore) {
                                        ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6);
                                        for (size_t i = arraySize - previewCount; i < arraySize; ++i) {
                                            ImGui::Text("[%zu]: (%.3f, %.3f, %.3f)", i, arr[i][0], arr[i][1], arr[i][2]);
                                        }
                                    }
                                }
                                else if (v.IsHolding<VtArray<GfVec4f>>()) {
                                    auto arr = v.Get<VtArray<GfVec4f>>();
                                    for (size_t i = 0; i < previewCount; ++i) {
                                        ImGui::Text("[%zu]: (%.3f, %.3f, %.3f, %.3f)", i, arr[i][0], arr[i][1], arr[i][2], arr[i][3]);
                                    }
                                    if (hasMore) {
                                        ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6);
                                        for (size_t i = arraySize - previewCount; i < arraySize; ++i) {
                                            ImGui::Text("[%zu]: (%.3f, %.3f, %.3f, %.3f)", i, arr[i][0], arr[i][1], arr[i][2], arr[i][3]);
                                        }
                                    }
                                }
                                else if (v.IsHolding<VtArray<GfVec4d>>()) {
                                    auto arr = v.Get<VtArray<GfVec4d>>();
                                    for (size_t i = 0; i < previewCount; ++i) {
                                        ImGui::Text("[%zu]: (%.3f, %.3f, %.3f, %.3f)", i, arr[i][0], arr[i][1], arr[i][2], arr[i][3]);
                                    }
                                    if (hasMore) {
                                        ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6);
                                        for (size_t i = arraySize - previewCount; i < arraySize; ++i) {
                                            ImGui::Text("[%zu]: (%.3f, %.3f, %.3f, %.3f)", i, arr[i][0], arr[i][1], arr[i][2], arr[i][3]);
                                        }
                                    }
                                }
                                else {
                                    // Unknown array type
                                    ImGui::TextDisabled("(preview not available)");
                                }
                                
                                ImGui::Unindent();
                            }
                            else {
                                // For scalar unsupported types, show type name
                                ImGui::TextDisabled("[%s]", v.GetTypeName().c_str());
                            }
                        }

                        ImGui::PopID();
                    }
                    ImGui::TreePop();
                }
            }
        }
    }
}

void UsdFileViewer::select_file()
{
    auto instance = IGFD::FileDialog::Instance();
    if (instance->Display("SelectFile")) {
        auto selected = instance->GetFilePathName();
        spdlog::info(selected.c_str());

        is_selecting_file = false;

        stage->import_usd_as_payload(selected, selecting_file_base);
    }
}

int UsdFileViewer::delete_pass_id = 0;

void UsdFileViewer::remove_prim_logic()
{
    if (delete_pass_id == 3) {
        stage->remove_prim(to_delete);
    }

    if (delete_pass_id == 2) {
        stage->add_prim(to_delete);
    }

    if (delete_pass_id == 1) {
        stage->remove_prim(to_delete);
    }

    if (delete_pass_id > 0) {
        delete_pass_id--;
    }
}

void UsdFileViewer::show_right_click_menu()
{
    if (ImGui::BeginPopupContextWindow("Prim Operation")) {
        if (ImGui::BeginMenu("Create Geometry")) {
            if (ImGui::MenuItem("Mesh")) {
                stage->create_mesh(selected);
            }
            if (ImGui::MenuItem("Cylinder")) {
                stage->create_cylinder(selected);
            }
            if (ImGui::MenuItem("Sphere")) {
                stage->create_sphere(selected);
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Create Light")) {
            if (ImGui::MenuItem("Dome Light")) {
                stage->create_dome_light(selected);
            }
            if (ImGui::MenuItem("Disk Light")) {
                stage->create_disk_light(selected);
            }
            if (ImGui::MenuItem("Distant Light")) {
                stage->create_distant_light(selected);
            }
            if (ImGui::MenuItem("Rect Light")) {
                stage->create_rect_light(selected);
            }
            if (ImGui::MenuItem("Sphere Light")) {
                stage->create_sphere_light(selected);
            }
            ImGui::EndMenu();
        }

        // create_material
        if (ImGui::BeginMenu("Create Material")) {
            if (ImGui::MenuItem("Material")) {
                stage->create_material(selected);
            }
            if (ImGui::MenuItem("Scratch Material")) {
                auto material = stage->create_material(selected);
            }

            ImGui::EndMenu();
        }

        if (selected != pxr::SdfPath("/")) {
            if (ImGui::MenuItem("Import...")) {
                is_selecting_file = true;
                selecting_file_base = selected;
            }
            if (ImGui::MenuItem("Edit")) {
                stage->create_editor_at_path(selected);
            }

            if (ImGui::MenuItem("Delete")) {
                to_delete = selected;
                delete_pass_id = 3;
            }
        }

        ImGui::EndPopup();
    }
}

void UsdFileViewer::DrawChild(const pxr::UsdPrim& prim, bool is_root)
{
    auto flags =
        ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow;
    if (is_root) {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }

    bool is_leaf = prim.GetChildren().empty();
    if (is_leaf) {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet |
                 ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    if (prim.GetPath() == selected) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    ImGui::TableNextRow();
    ImGui::TableNextColumn();

    bool open = ImGui::TreeNodeEx(prim.GetName().GetText(), flags);

    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        selected = prim.GetPath();
    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
        selected = prim.GetPath();
        ImGui::OpenPopup("Prim Operation");
    }

    ImGui::TableNextColumn();
    ImGui::TextUnformatted(prim.GetTypeName().GetText());

    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        selected = prim.GetPath();
    }

    if (!is_leaf) {
        if (open) {
            for (const pxr::UsdPrim& child : prim.GetChildren()) {
                DrawChild(child);
            }

            ImGui::TreePop();
        }
    }

    if (prim.GetPath() == selected) {
        show_right_click_menu();
    }
    if (is_selecting_file) {
        select_file();
    }
}

bool UsdFileViewer::BuildUI()
{
    ImGui::Begin("Stage Viewer", nullptr, ImGuiWindowFlags_None);
    ShowFileTree();
    ImGui::End();

    ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_None);
    EditValue();
    ImGui::End();
    
    remove_prim_logic();

    return true;
}

UsdFileViewer::UsdFileViewer(Stage* stage) : stage(stage)
{
}

UsdFileViewer::~UsdFileViewer()
{
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
