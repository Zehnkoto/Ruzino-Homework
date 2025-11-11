#include "GPUContext/program_vars.hpp"

#include <nvrhi/nvrhi.h>

#include "RHI/ResourceManager/resource_allocator.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE

// ProgramVarsProxy implementation
ProgramVarsProxy::ProgramVarsProxy(ProgramVars* parent, const std::string& path)
    : parent_(parent),
      path_(path)
{
}

ProgramVarsProxy ProgramVarsProxy::operator[](const std::string& name)
{
    return ProgramVarsProxy(parent_, build_path(name));
}

ProgramVarsProxy ProgramVarsProxy::operator[](int index)
{
    return ProgramVarsProxy(parent_, build_path(index));
}

ProgramVarsProxy& ProgramVarsProxy::operator=(nvrhi::IResource* resource)
{
    parent_->get_resource_direct(path_) = resource;
    return *this;
}

ProgramVarsProxy::operator nvrhi::IResource*&()
{
    return parent_->get_resource_direct(path_);
}

std::string ProgramVarsProxy::build_path(const std::string& name) const
{
    if (path_.empty()) {
        return name;
    }
    // Handle both member access and nested structures
    return path_ + "." + name;
}

std::string ProgramVarsProxy::build_path(int index) const
{
    if (path_.empty()) {
        return "[" + std::to_string(index) + "]";
    }
    // Support chained array access like array[0][1] or member.array[0]
    return path_ + "[" + std::to_string(index) + "]";
}

// ProgramVars implementation
ProgramVars::ProgramVars(ResourceAllocator& r) : resource_allocator_(r)
{
}

ProgramVars::~ProgramVars()
{
    for (int i = 0; i < binding_sets_solid.size(); ++i) {
        resource_allocator_.destroy(binding_sets_solid[i]);
    }
    for (int i = 0; i < binding_layouts.size(); ++i) {
        resource_allocator_.destroy(binding_layouts[i]);
    }
}

void ProgramVars::finish_setting_vars()
{
    for (int i = 0; i < binding_sets_solid.size(); ++i) {
        resource_allocator_.destroy(binding_sets_solid[i]);
    }
    binding_sets_solid.resize(0);
    binding_sets_solid.resize(binding_spaces.size());

    for (int i = 0; i < binding_spaces.size(); ++i) {
        if (!descriptor_tables[i]) {
            BindingSetDesc desc{};
            desc.bindings = binding_spaces[i];

            for (int j = 0; j < desc.bindings.size(); ++j) {
                if (dynamic_cast<nvrhi::IBuffer*>(
                        desc.bindings[j].resourceHandle)) {
                    desc.bindings[j].range = nvrhi::EntireBuffer;
                }
                else if (dynamic_cast<nvrhi::ITexture*>(
                             desc.bindings[j].resourceHandle)) {
                    desc.bindings[j].subresources = nvrhi::AllSubresources;
                }
            }
            binding_sets_solid[i] =
                resource_allocator_.create(desc, binding_layouts[i].Get());
        }
    }
}

// This is based on reflection
unsigned ProgramVars::get_binding_space(const std::string& name)
{
    return final_reflection_info.get_binding_space(name);
}

// This is based on reflection  
unsigned ProgramVars::get_binding_id(const std::string& name)
{
    auto binding_space = get_binding_space(name);
    if (binding_space == -1) {
        return -1;
    }
    
    auto binding_location = final_reflection_info.get_binding_location(name);
    if (binding_location == -1) {
        return -1;
    }

    auto slot = final_reflection_info.get_binding_layout_descs()[binding_space]
                    .bindings[binding_location]
                    .slot;

    return slot;
}

// This is based on reflection
nvrhi::ResourceType ProgramVars::get_binding_type(const std::string& name)
{
    return final_reflection_info.get_binding_type(name);
}

// This is where it is within the binding set
std::tuple<unsigned, unsigned> ProgramVars::get_binding_location(
    const std::string& name)
{
    // Check if we've already created a binding location for this exact path (including array index)
    auto path_it = path_to_binding_location.find(name);
    if (path_it != path_to_binding_location.end()) {
        return path_it->second;
    }
    
    // Get the base name without array indices
    std::string base_name = final_reflection_info.get_base_name(name);
    int array_index = final_reflection_info.parse_array_index(name);
    
    unsigned binding_space_id = get_binding_space(base_name);

    if (binding_space_id == -1) {
        return std::make_tuple(-1, -1);
    }

    if (binding_spaces.size() <= binding_space_id) {
        binding_spaces.resize(binding_space_id + 1);
    }
    if (descriptor_tables.size() <= binding_space_id) {
        descriptor_tables.resize(binding_space_id + 1);
    }

    auto& binding_space = binding_spaces[binding_space_id];

    auto& binding_layout = get_binding_layout()[binding_space_id];
    auto& layout_items = binding_layout->getDesc()->bindings;

    // Find the layout item for the base binding
    auto pos = std::find_if(
        layout_items.begin(),
        layout_items.end(),
        [&base_name, this](const nvrhi::BindingLayoutItem& binding) {
            return binding.slot == get_binding_id(base_name) &&
                   binding.type == get_binding_type(base_name);
        });

    assert(pos != layout_items.end());

    // Get array size to validate array index
    unsigned array_size = final_reflection_info.get_binding_array_size(base_name);
    if (array_index >= 0 && static_cast<unsigned>(array_index) >= array_size) {
        assert(false && "Array index out of bounds");
        return std::make_tuple(-1, -1);
    }

    // Create a new BindingSetItem for this specific array element (or single binding)
    unsigned binding_set_location = binding_space.size();
    binding_space.resize(binding_set_location + 1);

    nvrhi::BindingSetItem& item = binding_space[binding_set_location];

    // Initialize all fields properly (default constructor doesn't initialize for performance)
    item.resourceHandle = nullptr;
    item.slot = get_binding_id(base_name);
    item.type = get_binding_type(base_name);
    item.format = nvrhi::Format::UNKNOWN;
    item.dimension = nvrhi::TextureDimension::Unknown;
    item.unused = 0;
    item.unused2 = 0;
    item.subresources = nvrhi::AllSubresources;
    
    // Set the array element if this is an array access
    if (array_index >= 0) {
        item.arrayElement = static_cast<uint32_t>(array_index);
    } else {
        item.arrayElement = 0;
    }

    // Cache this path -> binding location mapping
    auto result = std::make_tuple(binding_space_id, binding_set_location);
    path_to_binding_location[name] = result;

    return result;
}

static nvrhi::IResource* placeholder;

ProgramVarsProxy ProgramVars::operator[](const std::string& name)
{
    return ProgramVarsProxy(this, name);
}

nvrhi::IResource*& ProgramVars::get_resource_direct(const std::string& name)
{
    auto [binding_space_id, binding_set_location] = get_binding_location(name);

    if (binding_space_id == -1) {
        return placeholder;
    }

    return binding_spaces[binding_space_id][binding_set_location]
        .resourceHandle;
}

void ProgramVars::set_descriptor_table(
    const std::string& name,
    nvrhi::IDescriptorTable* table,
    BindingLayoutHandle layout_handle)
{
    auto [binding_space_id, binding_set_location] = get_binding_location(name);
    if (binding_space_id == -1) {
        return;
    }
    descriptor_tables[binding_space_id] = table;

    if (binding_layouts[binding_space_id]) {
        resource_allocator_.destroy(binding_layouts[binding_space_id]);
        binding_layouts[binding_space_id] = layout_handle;
    }
}

void ProgramVars::set_binding(
    const std::string& name,
    nvrhi::ITexture* resource,
    const nvrhi::TextureSubresourceSet& subset)
{
    auto [binding_space_id, binding_set_location] = get_binding_location(name);
    if (binding_space_id == -1) {
        return;
    }
    auto& binding_set = binding_spaces[binding_space_id][binding_set_location];

    binding_set.resourceHandle = resource;
    binding_set.subresources = subset;
    if (subset.baseArraySlice != 0 || subset.numArraySlices != 1) {
        binding_set.dimension = nvrhi::TextureDimension::Texture2DArray;
    }
}

nvrhi::BindingSetVector ProgramVars::get_binding_sets() const
{
    nvrhi::BindingSetVector result;

    result.resize(binding_sets_solid.size());
    for (int i = 0; i < binding_sets_solid.size(); ++i) {
        if (binding_sets_solid[i]) {
            result[i] = binding_sets_solid[i].Get();
        }
        if (descriptor_tables[i]) {
            result[i] = descriptor_tables[i];
        }
    }
    return result;
}

nvrhi::BindingLayoutVector& ProgramVars::get_binding_layout()
{
    if (binding_layouts.empty()) {
        auto binding_layout_descs =
            final_reflection_info.get_binding_layout_descs();
        for (int i = 0; i < binding_layout_descs.size(); ++i) {
            auto binding_layout =
                resource_allocator_.create(binding_layout_descs[i]);
            binding_layouts.push_back(binding_layout);
        }
    }
    return binding_layouts;
}

std::vector<IProgram*> ProgramVars::get_programs() const
{
    return programs;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
