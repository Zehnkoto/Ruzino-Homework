#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Framework includes
#include "GCore/GOP.h"
#include "GCore/algorithms/intersection.h"
#include "Logger/Logger.h"
#include "RHI/rhi.hpp"
#include "nodes/system/node_system.hpp"
#include "stage/stage.hpp"
#include "usd_nodejson.hpp"

// USD includes
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/camera.h"

// Hydra includes
#include "pxr/base/gf/camera.h"
#include "pxr/base/gf/frustum.h"
#include "pxr/imaging/hd/driver.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hdx/tokens.h"
#include "pxr/imaging/hgi/blitCmdsOps.h"
#include "pxr/imaging/hgi/tokens.h"
#include "pxr/usdImaging/usdImagingGL/engine.h"

// NVRHI includes
#include "nvrhi/nvrhi.h"

// Image saving
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "pxr/imaging/garch/glApi.h"
#include "stb_image_write.h"

#ifdef _WIN32
#include <gl/GL.h>
#include <windows.h>

#endif

using namespace USTC_CG;
using namespace pxr;

// Configuration
struct RenderSettings {
    std::string usd_file;
    std::string json_script;
    std::string output_image;
    int width = 1920;
    int height = 1080;
};

// Command line parsing
bool ParseCommandLine(int argc, char* argv[], RenderSettings& settings)
{
    if (argc < 4) {
        std::cout
            << "Usage: " << argv[0]
            << " <usd_file> <json_script> <output_image> [width] [height]\n"
            << "  usd_file: Path to USD file to render\n"
            << "  json_script: Path to JSON rendering script\n"
            << "  output_image: Output image filename (PNG format)\n"
            << "  width: Image width (default: 1920)\n"
            << "  height: Image height (default: 1080)\n";
        return false;
    }

    settings.usd_file = argv[1];
    settings.json_script = argv[2];
    settings.output_image = argv[3];

    if (argc > 4)
        settings.width = std::atoi(argv[4]);
    if (argc > 5)
        settings.height = std::atoi(argv[5]);

    // Validate input files
    if (!std::filesystem::exists(settings.usd_file)) {
        std::cerr << "Error: USD file not found: " << settings.usd_file
                  << std::endl;
        return false;
    }
    if (!std::filesystem::exists(settings.json_script)) {
        std::cerr << "Error: JSON script not found: " << settings.json_script
                  << std::endl;
        return false;
    }

    return true;
}

// USD utilities
UsdGeomCamera FindFirstCamera(const UsdStageRefPtr& stage)
{
    for (const UsdPrim& prim : stage->Traverse()) {
        if (prim.IsA<UsdGeomCamera>()) {
            log::info("Found camera: %s", prim.GetPath().GetString().c_str());
            return UsdGeomCamera(prim);
        }
    }
    return UsdGeomCamera();
}

// Graphics context initialization
void CreateGLContext()
{
#ifdef _WIN32
    HDC hdc = GetDC(GetConsoleWindow());
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;

    int pixelFormat = ChoosePixelFormat(hdc, &pfd);
    SetPixelFormat(hdc, pixelFormat, &pfd);

    HGLRC hglrc = wglCreateContext(hdc);
    wglMakeCurrent(hdc, hglrc);
#endif
}

// Image utilities
bool SaveImageToFile(
    const std::string& filename,
    int width,
    int height,
    const std::vector<uint8_t>& data)
{
    std::vector<uint8_t> rgba_data(width * height * 4);
    const float* float_data = reinterpret_cast<const float*>(data.data());

    for (int i = 0; i < width * height * 4; ++i) {
        rgba_data[i] = static_cast<uint8_t>(
            std::clamp(float_data[i] * 255.0f, 0.0f, 255.0f));
    }

    return stbi_write_png(
               filename.c_str(),
               width,
               height,
               4,
               rgba_data.data(),
               width * 4) != 0;
}

// Texture reading methods
bool ReadTextureDirectly(
    UsdImagingGLEngine* renderer,
    int width,
    int height,
    std::vector<uint8_t>& texture_data)
{
    auto hacked_handle =
        renderer->GetRendererSetting(pxr::TfToken("VulkanColorAov"));
    if (!hacked_handle.IsHolding<const void*>()) {
        return false;
    }

    log::info("Using direct texture copy method...");

    auto bare_pointer = hacked_handle.Get<const void*>();
    auto texture =
        *static_cast<nvrhi::ITexture**>(const_cast<void*>(bare_pointer));

    texture_data.resize(width * height * 4 * sizeof(float));

    auto command_list = RHI::get_device()->createCommandList();
    if (!command_list) {
        command_list = RHI::get_device()->createCommandList();
    }

    // Create staging texture
    nvrhi::TextureDesc staging_desc;
    staging_desc.debugName = "headless_staging";
    staging_desc.width = width;
    staging_desc.height = height;
    staging_desc.format = texture->getDesc().format;
    staging_desc.initialState = nvrhi::ResourceStates::CopyDest;

    auto staging_texture = RHI::get_device()->createStagingTexture(
        staging_desc, nvrhi::CpuAccessMode::Read);

    // Copy and read back
    command_list->open();
    command_list->copyTexture(staging_texture, {}, texture, {});
    command_list->close();
    RHI::get_device()->executeCommandList(command_list.Get());
    RHI::get_device()->waitForIdle();

    size_t pitch;
    auto mapped = RHI::get_device()->mapStagingTexture(
        staging_texture, {}, nvrhi::CpuAccessMode::Read, &pitch);

    for (int i = 0; i < height; ++i) {
        memcpy(
            texture_data.data() + i * width * 4 * sizeof(float),
            static_cast<uint8_t*>(mapped) + i * pitch,
            width * 4 * sizeof(float));
    }

    RHI::get_device()->unmapStagingTexture(staging_texture);
    log::info("Direct texture copy completed successfully");
    return true;
}

bool ReadTextureCPU(
    UsdImagingGLEngine* renderer,
    HgiUniquePtr& hgi,
    int width,
    int height,
    std::vector<uint8_t>& texture_data)
{
    log::info("Using CPU readback method...");

    auto hgi_texture = renderer->GetAovTexture(HdAovTokens->color);
    if (!hgi_texture) {
        std::cerr << "Error: Failed to get rendered texture" << std::endl;
        return false;
    }

    size_t buffer_size = width * height * 4 * sizeof(float);
    texture_data.resize(buffer_size);

    auto blit_cmds = hgi->CreateBlitCmds();
    HgiTextureGpuToCpuOp copy_op;
    copy_op.gpuSourceTexture = hgi_texture;
    copy_op.cpuDestinationBuffer = texture_data.data();
    copy_op.destinationBufferByteSize = texture_data.size();
    blit_cmds->CopyTextureGpuToCpu(copy_op);

    hgi->SubmitCmds(blit_cmds.get(), HgiSubmitWaitTypeWaitUntilCompleted);
    return true;
}

std::string LoadJSONScript(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error(
            "Could not open JSON script file: " + filename);
    }

    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }
    return content;
}

int main(int argc, char* argv[])
{
    // Parse command line
    RenderSettings settings;
    if (!ParseCommandLine(argc, argv, settings)) {
        return 1;
    }

    // Initialize logging
    log::SetMinSeverity(Severity::Info);
    log::EnableOutputToConsole(true);

    log::info("Starting headless render...");
    log::info("USD file: %s", settings.usd_file.c_str());
    log::info("JSON script: %s", settings.json_script.c_str());
    log::info("Output image: %s", settings.output_image.c_str());
    log::info("Resolution: %dx%d", settings.width, settings.height);

    try {
        // Initialize OpenGL context
        CreateGLContext();
        GarchGLApiLoad();

        // Create USD stage
        auto stage = create_custom_global_stage(settings.usd_file);
        if (!stage) {
            throw std::runtime_error(
                "Failed to load USD stage from " + settings.usd_file);
        }
        init(stage.get());

        // Find camera
        auto camera = FindFirstCamera(stage->get_usd_stage());
        if (!camera) {
            throw std::runtime_error("No camera found in USD file");
        }

        // Setup rendering engine
        auto hgi = Hgi::CreateNamedHgi(HgiTokens->OpenGL);
        HdDriver hd_driver;
        hd_driver.name = HgiTokens->renderDriver;
        hd_driver.driver = VtValue(hgi.get());

        UsdImagingGLEngine::Parameters params;
        params.allowAsynchronousSceneProcessing = false;
        params.driver = hd_driver;

        auto renderer = std::make_unique<UsdImagingGLEngine>(params);
        renderer->SetRendererPlugin(TfToken("Hd_USTC_CG_RendererPlugin"));
        renderer->SetEnablePresentation(false);

        // Configure render settings
        GfVec2i render_size(settings.width, settings.height);
        renderer->SetRenderBufferSize(render_size);
        renderer->SetRenderViewport(
            GfVec4d(0.0, 0.0, settings.width, settings.height));

        // Setup camera
        auto gf_camera = camera.GetCamera(UsdTimeCode::Default());
        auto frustum = gf_camera.GetFrustum();
        renderer->SetCameraState(
            frustum.ComputeViewMatrix(), frustum.ComputeProjectionMatrix());

        // Configure render parameters
        UsdImagingGLRenderParams render_params;
        render_params.enableLighting = true;
        render_params.enableSceneMaterials = true;
        render_params.showRender = true;
        render_params.frame = UsdTimeCode(0);
        render_params.drawMode = UsdImagingGLDrawMode::DRAW_SHADED_SMOOTH;
        render_params.colorCorrectionMode = HdxColorCorrectionTokens->disabled;
        render_params.clearColor = GfVec4f(0.2f, 0.2f, 0.2f, 1.0f);
        renderer->SetRendererAov(HdAovTokens->color);

        // Load and apply JSON script
        auto node_system = static_cast<const std::shared_ptr<NodeSystem>*>(
            renderer->GetRendererSetting(pxr::TfToken("RenderNodeSystem"))
                .Get<const void*>());

        std::string nodes_json = LoadJSONScript(settings.json_script);
        (*node_system)->get_node_tree()->deserialize(nodes_json);

        // Render the scene
        UsdPrim root = stage->get_usd_stage()->GetPseudoRoot();
        log::info("Starting render...");
        renderer->Render(root, render_params);
        renderer->StopRenderer();
        log::info("Render complete.");

        // Read back texture data
        std::vector<uint8_t> texture_data;
        bool success = ReadTextureDirectly(
            renderer.get(), settings.width, settings.height, texture_data);

        if (!success) {
            success = ReadTextureCPU(
                renderer.get(),
                hgi,
                settings.width,
                settings.height,
                texture_data);
        }

        if (!success) {
            throw std::runtime_error("Failed to read back rendered texture");
        }

        // Save the image
        log::info("Saving image to: %s", settings.output_image.c_str());
        if (!SaveImageToFile(
                settings.output_image,
                settings.width,
                settings.height,
                texture_data)) {
            throw std::runtime_error(
                "Failed to save image to " + settings.output_image);
        }

        log::info("Headless render completed successfully!");

        // Cleanup
        renderer.reset();
        hgi.reset();
        stage.reset();
        unregister_cpp_type();
        deinit_gpu_geometry_algorithms();

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
