#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>

#include <stdexcept>
#include <unordered_map>
#include <vector>

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

// Generic template implementation using nanobind casting
template<typename T>
T call(const std::string& code)
{
    PyObject* py_result = call_raw(code);
    if (!py_result) {
        throw std::runtime_error(
            "Failed to get result from Python code: " + code);
    }

    try {
        // Use nanobind to convert the Python object to the desired C++ type
        nb::object nb_result = nb::steal(py_result);  // Takes ownership
        
        // For pointer types, nanobind usually binds objects as pointers
        // So we can directly cast to the pointer type
        T result = nb::cast<T>(nb_result);
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to convert Python result to C++ type: " +
            std::string(e.what()));
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
        nb::object py_obj = nb::cast(value);

        // Store in our map to keep it alive
        bound_objects[name] = py_obj;

        // Add to Python's main dict
        PyDict_SetItemString(main_dict, name.c_str(), py_obj.ptr());
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to send value to Python variable '" + name + "': " + e.what());
    }
}

}  // namespace python

USTC_CG_NAMESPACE_CLOSE_SCOPE
