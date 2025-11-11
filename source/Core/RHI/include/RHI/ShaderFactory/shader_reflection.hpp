#pragma once

#include "RHI/ShaderFactory/vector_map.hpp"
#include "RHI/api.h"
#include "RHI/internal/nvrhi_patch.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE
class RHI_API ShaderReflectionInfo {
   public:
    [[nodiscard]] const nvrhi::BindingLayoutDescVector&
    get_binding_layout_descs() const;
    
    // Main interface for nested path resolution
    unsigned get_binding_space(const std::string& path);
    unsigned get_binding_location(const std::string& path);
    nvrhi::ResourceType get_binding_type(const std::string& path);
    
    // Get the array size for a binding (returns 1 for non-array bindings)
    unsigned get_binding_array_size(const std::string& path);
    
    // Parse array index from path like "samplers[3]", returns -1 if not an array access
    int parse_array_index(const std::string& path) const;
    
    // Get base name without array indices (e.g., "samplers[3]" -> "samplers")
    std::string get_base_name(const std::string& path) const;
    
    // Check if a binding exists
    bool has_binding(const std::string& path) const;

    ShaderReflectionInfo operator+(const ShaderReflectionInfo& other) const;
    ShaderReflectionInfo& operator+=(const ShaderReflectionInfo& other);

   private:
    nvrhi::BindingLayoutDescVector binding_spaces;
    VectorMap<std::string, std::tuple<unsigned, unsigned>> binding_locations;
    
    // Helper methods for path parsing
    std::string resolve_base_name(const std::string& path) const;
    std::string extract_array_indices(const std::string& path) const;
    bool is_array_access(const std::string& path) const;
    std::string normalize_path(const std::string& path) const;

    friend class ShaderFactory;
    friend RHI_API std::ostream& operator<<(
        std::ostream& os,
        const ShaderReflectionInfo& info);
};

RHI_API std::ostream& operator<<(
    std::ostream& os,
    const ShaderReflectionInfo& info);
USTC_CG_NAMESPACE_CLOSE_SCOPE
