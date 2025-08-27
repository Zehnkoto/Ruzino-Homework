#include <GUI/window.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <rzpython/rzpython.hpp>
#include <stdexcept>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <string>
#include <vector>
#endif

namespace nb = nanobind;

USTC_CG_NAMESPACE_OPEN_SCOPE

namespace python {

// Global variables - accessible from template implementations
PyObject* main_module = nullptr;
PyObject* main_dict = nullptr;
bool initialized = false;
std::unordered_map<std::string, nb::object> bound_objects;

#ifdef _WIN32
// Helper function to forcefully unload DLLs
void force_unload_dlls() {
    // Force unload CUDA and PyTorch related DLLs
    std::vector<std::string> dlls_to_unload = {
        "torch_cpu.dll",
        "torch_cuda.dll", 
        "c10.dll",
        "c10_cuda.dll",
        "caffe2_nvrtc.dll",
        "cublas64_11.dll",
        "cublas64_12.dll",
        "cublasLt64_11.dll", 
        "cublasLt64_12.dll",
        "cudart64_110.dll",
        "cudart64_120.dll",
        "cudnn64_8.dll",
        "cufft64_10.dll",
        "cufft64_11.dll",
        "curand64_10.dll",
        "curand64_11.dll",
        "cusolver64_11.dll",
        "cusparse64_11.dll",
        "cusparse64_12.dll",
        "nvrtc64_112_0.dll",
        "nvrtc64_120_0.dll",
        "nvToolsExt64_1.dll",
        "_multiarray_umath.cp311-win_amd64.pyd",
        "_multiarray_umath.cp312-win_amd64.pyd"
    };
    
    for (const auto& dll_name : dlls_to_unload) {
        HMODULE hModule = GetModuleHandleA(dll_name.c_str());
        if (hModule) {
            // Try to free the library multiple times as some DLLs might have multiple references
            for (int i = 0; i < 20; ++i) {  // Increased attempts
                if (!FreeLibrary(hModule)) {
                    break;
                }
                Sleep(1); // Small delay between attempts
            }
        }
    }
    
    // Also try to unload any modules containing specific patterns
    HANDLE hProcess = GetCurrentProcess();
    HMODULE hMods[1024];
    DWORD cbNeeded;
    
    if (EnumProcessModules(hProcess, hMods, sizeof(hMods), &cbNeeded)) {
        for (unsigned int i = 0; i < (cbNeeded / sizeof(HMODULE)); i++) {
            char szModName[MAX_PATH];
            if (GetModuleFileNameExA(hProcess, hMods[i], szModName, sizeof(szModName) / sizeof(char))) {
                std::string moduleName(szModName);
                
                // Check if this module contains problematic patterns
                if (moduleName.find("torch") != std::string::npos ||
                    moduleName.find("cuda") != std::string::npos ||
                    moduleName.find("cublas") != std::string::npos ||
                    moduleName.find("cudnn") != std::string::npos ||
                    moduleName.find("multiarray") != std::string::npos ||
                    moduleName.find("numpy") != std::string::npos) {
                    
                    // Try to free this module
                    for (int j = 0; j < 10; ++j) {
                        if (!FreeLibrary(hMods[i])) {
                            break;
                        }
                        Sleep(1);
                    }
                }
            }
        }
    }
    
    // Give the system time to fully unload
    Sleep(100);
}
#endif

void initialize()
{
    if (initialized) {
        return;
    }

    // Add path to ensure Python finds our modules
    Py_Initialize();
    if (!Py_IsInitialized()) {
        throw std::runtime_error("Failed to initialize Python interpreter");
    }

    // Set up import hooks to prevent problematic modules from being loaded multiple times
    try {
        PyRun_SimpleString(
            "import sys\n"
            "if not hasattr(sys, '_rzpython_initialized'):\n"
            "    sys._rzpython_initialized = True\n"
            "    # Set up some module loading protections\n"
            "    import importlib.util\n"
            "    original_find_spec = importlib.util.find_spec\n"
            "    def protected_find_spec(name, package=None, path=None):\n"
            "        # Add any module loading protections here if needed\n"
            "        return original_find_spec(name, package, path)\n"
            "    importlib.util.find_spec = protected_find_spec\n"
        );
    } catch (...) {
        // Ignore setup errors
    }

    main_module = PyImport_AddModule("__main__");
    if (!main_module) {
        throw std::runtime_error("Failed to get __main__ module");
    }

    main_dict = PyModule_GetDict(main_module);
    if (!main_dict) {
        throw std::runtime_error("Failed to get __main__ dictionary");
    }

    initialized = true;
}

void finalize()
{
    if (!initialized) {
        return;
    }

    // Clear all bound objects first
    bound_objects.clear();
    
    // Step 1: Try to explicitly unload PyTorch CUDA context if it exists
    try {
        PyRun_SimpleString(
            "try:\n"
            "    import torch\n"
            "    if torch.cuda.is_available():\n"
            "        torch.cuda.empty_cache()\n"
            "        torch.cuda.synchronize()\n"
            "        # Clear CUDA context\n"
            "        import gc\n"
            "        gc.collect()\n"
            "except: pass\n"
        );
    } catch (...) {
        // Ignore errors in cleanup
    }
    
    // Step 2: Force garbage collection before finalizing
    PyObject* gc_module = PyImport_ImportModule("gc");
    if (gc_module) {
        PyObject* collect_func = PyObject_GetAttrString(gc_module, "collect");
        if (collect_func && PyCallable_Check(collect_func)) {
            PyObject_CallObject(collect_func, nullptr);
            Py_DECREF(collect_func);
        }
        Py_DECREF(gc_module);
    }
    
    // Clear sys.modules for problematic modules that might hold DLL references
    PyObject* sys_modules = PySys_GetObject("modules");
    if (sys_modules && PyDict_Check(sys_modules)) {
        // List of modules that commonly cause DLL loading issues
        const char* problematic_modules[] = {
            "torch", "torch.cuda", "torch._C", "torch._C._cuda", 
            "torch.backends", "torch.backends.cuda", "torch.backends.cudnn",
            "torchvision", "torchaudio",
            "numpy", "numpy.core", "numpy.core._multiarray_umath",
            "cupy", "cupyx", "cupy.cuda", "cupy._core",
            "tensorflow", "tensorflow.python", "tensorflow.python.framework",
            "jax", "jaxlib", "jaxlib.xla_extension",
            "cv2", "PIL", "matplotlib"
        };
        
        for (const char* module_name : problematic_modules) {
            PyObject* key = PyUnicode_FromString(module_name);
            if (key) {
                if (PyDict_Contains(sys_modules, key)) {
                    PyDict_DelItem(sys_modules, key);
                }
                Py_DECREF(key);
            }
        }
        
        // Also remove any modules that contain these prefixes
        PyObject* keys = PyDict_Keys(sys_modules);
        if (keys) {
            Py_ssize_t size = PyList_Size(keys);
            for (Py_ssize_t i = 0; i < size; i++) {
                PyObject* key = PyList_GetItem(keys, i);
                if (PyUnicode_Check(key)) {
                    const char* module_name = PyUnicode_AsUTF8(key);
                    if (module_name) {
                        // Remove modules starting with problematic prefixes
                        if (strncmp(module_name, "torch", 5) == 0 ||
                            strncmp(module_name, "numpy", 5) == 0 ||
                            strncmp(module_name, "cupy", 4) == 0 ||
                            strncmp(module_name, "tensorflow", 10) == 0 ||
                            strncmp(module_name, "jax", 3) == 0 ||
                            strncmp(module_name, "cv2", 3) == 0) {
                            PyDict_DelItem(sys_modules, key);
                        }
                    }
                }
            }
            Py_DECREF(keys);
        }
    }
    
    // Force another garbage collection after module cleanup
    if (gc_module) {
        PyObject* collect_func = PyObject_GetAttrString(gc_module, "collect");
        if (collect_func && PyCallable_Check(collect_func)) {
            PyObject_CallObject(collect_func, nullptr);
            Py_DECREF(collect_func);
        }
    }
    
    // Clear Python caches
    PyObject* importlib = PyImport_ImportModule("importlib");
    if (importlib) {
        PyObject* invalidate_caches = PyObject_GetAttrString(importlib, "invalidate_caches");
        if (invalidate_caches && PyCallable_Check(invalidate_caches)) {
            PyObject_CallObject(invalidate_caches, nullptr);
            Py_DECREF(invalidate_caches);
        }
        Py_DECREF(importlib);
    }
    
    // Step 4: Windows-specific DLL cleanup for problematic modules - performed after finalization
    
    // Reset main module references
    main_module = nullptr;
    main_dict = nullptr;
    
    // Finalize Python interpreter
    Py_Finalize();
    
    // After Python finalization, force unload problematic DLLs
#ifdef _WIN32
    force_unload_dlls();
#endif
    
    initialized = false;
}

void import(const std::string& module_name)
{
    if (!initialized) {
        throw std::runtime_error("Python interpreter not initialized");
    }

    PyObject* module = PyImport_ImportModule(module_name.c_str());
    if (!module) {
        PyErr_Print();
        throw std::runtime_error("Failed to import module: " + module_name);
    }

    // Add module to main dict so it can be accessed
    PyDict_SetItemString(main_dict, module_name.c_str(), module);
    Py_DECREF(module);
}

// Template specializations for call function
template<>
int call<int>(const std::string& code)
{
    if (!initialized) {
        throw std::runtime_error("Python interpreter not initialized");
    }

    PyObject* result =
        PyRun_String(code.c_str(), Py_eval_input, main_dict, main_dict);
    if (!result) {
        PyErr_Print();
        throw std::runtime_error("Failed to execute Python code: " + code);
    }

    if (!PyLong_Check(result)) {
        Py_DECREF(result);
        throw std::runtime_error("Expected int return type");
    }

    int value = PyLong_AsLong(result);
    Py_DECREF(result);
    return value;
}

template<>
float call<float>(const std::string& code)
{
    if (!initialized) {
        throw std::runtime_error("Python interpreter not initialized");
    }

    PyObject* result =
        PyRun_String(code.c_str(), Py_eval_input, main_dict, main_dict);
    if (!result) {
        PyErr_Print();
        throw std::runtime_error("Failed to execute Python code: " + code);
    }

    if (!PyFloat_Check(result) && !PyLong_Check(result)) {
        Py_DECREF(result);
        throw std::runtime_error("Expected float return type");
    }

    float value = PyFloat_AsDouble(result);
    Py_DECREF(result);
    return value;
}

template<>
void call<void>(const std::string& code)
{
    if (!initialized) {
        throw std::runtime_error("Python interpreter not initialized");
    }

    PyObject* result =
        PyRun_String(code.c_str(), Py_file_input, main_dict, main_dict);
    if (!result) {
        PyErr_Print();
        throw std::runtime_error("Failed to execute Python code: " + code);
    }

    Py_DECREF(result);
}

// Internal helper for raw Python object return
PyObject* call_raw(const std::string& code)
{
    if (!initialized) {
        throw std::runtime_error("Python interpreter not initialized");
    }

    PyObject* result =
        PyRun_String(code.c_str(), Py_eval_input, main_dict, main_dict);
    if (!result) {
        PyErr_Print();
        throw std::runtime_error("Failed to execute Python code: " + code);
    }

    return result;  // Caller is responsible for DECREF
}

// Template specializations for std::vector types
template<>
std::vector<int> call<std::vector<int>>(const std::string& code)
{
    PyObject* py_result = call_raw(code);
    if (!py_result) {
        throw std::runtime_error(
            "Failed to get result from Python code: " + code);
    }

    try {
        // Use nanobind to convert the Python object to std::vector<int>
        nb::object nb_result = nb::steal(py_result);  // Takes ownership
        std::vector<int> result = nb::cast<std::vector<int>>(nb_result);
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to convert Python result to std::vector<int>: " +
            std::string(e.what()));
    }
}

template<>
std::vector<float> call<std::vector<float>>(const std::string& code)
{
    PyObject* py_result = call_raw(code);
    if (!py_result) {
        throw std::runtime_error(
            "Failed to get result from Python code: " + code);
    }

    try {
        // Use nanobind to convert the Python object to std::vector<float>
        nb::object nb_result = nb::steal(py_result);  // Takes ownership
        std::vector<float> result = nb::cast<std::vector<float>>(nb_result);
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to convert Python result to std::vector<float>: " +
            std::string(e.what()));
    }
}

template<>
std::vector<std::string> call<std::vector<std::string>>(const std::string& code)
{
    PyObject* py_result = call_raw(code);
    if (!py_result) {
        throw std::runtime_error(
            "Failed to get result from Python code: " + code);
    }

    try {
        // Use nanobind to convert the Python object to std::vector<std::string>
        nb::object nb_result = nb::steal(py_result);  // Takes ownership
        std::vector<std::string> result =
            nb::cast<std::vector<std::string>>(nb_result);
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to convert Python result to std::vector<std::string>: " +
            std::string(e.what()));
    }
}

// Template specializations for ndarray types
template<>
numpy_array_f32 call<numpy_array_f32>(const std::string& code)
{
    PyObject* py_result = call_raw(code);
    if (!py_result) {
        throw std::runtime_error(
            "Failed to get result from Python code: " + code);
    }

    try {
        nb::object nb_result = nb::steal(py_result);
        numpy_array_f32 result = nb::cast<numpy_array_f32>(nb_result);
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to convert Python result to numpy_array_f32: " +
            std::string(e.what()));
    }
}

template<>
numpy_array_f64 call<numpy_array_f64>(const std::string& code)
{
    PyObject* py_result = call_raw(code);
    if (!py_result) {
        throw std::runtime_error(
            "Failed to get result from Python code: " + code);
    }

    try {
        nb::object nb_result = nb::steal(py_result);
        numpy_array_f64 result = nb::cast<numpy_array_f64>(nb_result);
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to convert Python result to numpy_array_f64: " +
            std::string(e.what()));
    }
}

template<>
torch_tensor_f32 call<torch_tensor_f32>(const std::string& code)
{
    PyObject* py_result = call_raw(code);
    if (!py_result) {
        throw std::runtime_error(
            "Failed to get result from Python code: " + code);
    }

    try {
        nb::object nb_result = nb::steal(py_result);
        torch_tensor_f32 result = nb::cast<torch_tensor_f32>(nb_result);
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to convert Python result to torch_tensor_f32: " +
            std::string(e.what()));
    }
}

template<>
torch_tensor_f64 call<torch_tensor_f64>(const std::string& code)
{
    PyObject* py_result = call_raw(code);
    if (!py_result) {
        throw std::runtime_error(
            "Failed to get result from Python code: " + code);
    }

    try {
        nb::object nb_result = nb::steal(py_result);
        torch_tensor_f64 result = nb::cast<torch_tensor_f64>(nb_result);
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to convert Python result to torch_tensor_f64: " +
            std::string(e.what()));
    }
}

template<>
cuda_array_f32 call<cuda_array_f32>(const std::string& code)
{
    PyObject* py_result = call_raw(code);
    if (!py_result) {
        throw std::runtime_error(
            "Failed to get result from Python code: " + code);
    }

    try {
        nb::object nb_result = nb::steal(py_result);
        cuda_array_f32 result = nb::cast<cuda_array_f32>(nb_result);
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to convert Python result to cuda_array_f32: " +
            std::string(e.what()));
    }
}

template<>
cuda_array_f64 call<cuda_array_f64>(const std::string& code)
{
    PyObject* py_result = call_raw(code);
    if (!py_result) {
        throw std::runtime_error(
            "Failed to get result from Python code: " + code);
    }

    try {
        nb::object nb_result = nb::steal(py_result);
        cuda_array_f64 result = nb::cast<cuda_array_f64>(nb_result);
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to convert Python result to cuda_array_f64: " +
            std::string(e.what()));
    }
}

}  // namespace python

USTC_CG_NAMESPACE_CLOSE_SCOPE
