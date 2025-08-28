#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <type_traits>
#include <unordered_map>

#include "api.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

namespace python {

namespace nb = nanobind;

// Forward declarations for external variables
RZPYTHON_EXTERN RZPYTHON_API PyObject* main_dict;
RZPYTHON_EXTERN RZPYTHON_API bool initialized;
RZPYTHON_EXTERN RZPYTHON_API std::unordered_map<std::string, nb::object>
    bound_objects;
RZPYTHON_API PyObject* call_raw(const std::string& code);

// Helper to check if a Python module uses Boost.Python
RZPYTHON_API bool is_boost_python_module(const std::string& module_name);

// Safe import function that handles Boost.Python conflicts
RZPYTHON_API void safe_import(const std::string& module_name);

// Helper to determine if we should use eval or file input mode
template<typename T>
constexpr bool is_void_type()
{
    return std::is_void_v<T>;
}

// Generic template implementation using nanobind casting
template<typename T>
T call(const std::string& code)
{
    static_assert(!std::is_void_v<T>, "Use call<void>() for void return type");

    PyObject* py_result = call_raw(code);
    if (!py_result) {
        throw std::runtime_error(
            "Failed to get result from Python code: " + code);
    }

    try {
        // Use nanobind to convert the Python object to the desired C++ type
        nb::object nb_result = nb::steal(py_result);  // Takes ownership

        // Let nanobind handle all the type conversions automatically
        // This works for primitives, STL containers, ndarrays, etc.
        T result = nb::cast<T>(nb_result);
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to convert Python result to C++ type: " +
            std::string(e.what()));
    }
}

// Specialized version for handling potentially mixed binding types
template<typename T>
T call_safe(const std::string& code)
{
    try {
        return call<T>(code);
    }
    catch (const std::exception& e) {
        // If regular nanobind conversion fails, we can't do much more
        // in the core library since we don't link to specific external
        // libraries
        throw std::runtime_error(
            "Mixed binding conversion failed: " + std::string(e.what()) +
            ". Consider using specialized conversion functions for external "
            "libraries.");
    }
}

template<typename T>
void reference(const std::string& name, T* obj)
{
    if (!initialized) {
        throw std::runtime_error("Python interpreter not initialized");
    }

    try {
        // Create nanobind object wrapper
        nb::object py_obj = nb::cast(obj, nb::rv_policy::reference);

        // Store in our map to keep it alive
        bound_objects[name] = py_obj;

        // Add to Python's main dict
        PyDict_SetItemString(main_dict, name.c_str(), py_obj.ptr());
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to bind object '" + name + "': " + e.what());
    }
}

template<typename T>
void bind_object(const std::string& name, T* obj)
{
    reference(name, obj);
}

template<typename T>
void send(const std::string& name, const T& value)
{
    if (!initialized) {
        throw std::runtime_error("Python interpreter not initialized");
    }

    try {
        // Create nanobind object from the C++ value
        // This automatically handles conversion of STL containers, ndarrays,
        // etc.
        nb::object py_obj = nb::cast(value);

        // Store in our map to keep it alive
        bound_objects[name] = py_obj;

        // Add to Python's main dict
        PyDict_SetItemString(main_dict, name.c_str(), py_obj.ptr());
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to send value to Python variable '" + name +
            "': " + e.what());
    }
}

}  // namespace python

USTC_CG_NAMESPACE_CLOSE_SCOPE
