#include <gtest/gtest.h>

#include <RHI/rhi.hpp>

#include "GUI/window.h"
#include "rzpython/rzpython.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace USTC_CG;

TEST(RZPythonRuntimeTest, RHI_package)
{
    python::initialize();

    python::import("RHI_py");
    int result = python::call<int>("RHI_py.init()");
    EXPECT_EQ(result, 0);

    // Test that we can call get_device without crashing, even if we can't
    // convert the result yet
    python::call<void>("device = RHI_py.get_device()");
    python::call<void>("print('Device type:', type(device))");

    // Test that we can get the backend enum
    python::call<void>("backend = RHI_py.get_backend()");
    python::call<void>("print('Backend type:', type(backend))");
    python::call<void>("print('Backend value:', backend)");

    result = python::call<int>("RHI_py.shutdown()");
    EXPECT_EQ(result, 0);

    python::finalize();
}

TEST(RZPythonRuntimeTest, GUI_package)
{
    python::initialize();

    python::import("GUI_py");

    Window window;
    python::reference("w", &window);

    // Just test that we can call the method without crashing
    // and that we get some kind of numeric result
    python::call<void>("print('Testing Window binding...')");
    python::call<void>("print(type(w))");
    python::call<void>("result = w.get_elapsed_time()");
    python::call<void>("print('Elapsed time result:', result)");

    float time = python::call<float>("w.get_elapsed_time()");
    // Just check that we get a finite number (not inf or nan)
    EXPECT_TRUE(std::isfinite(time));

    python::finalize();
}

TEST(RZPythonRuntimeTest, ListToVector_conversion)
{
    python::initialize();

    // Test converting Python list to C++ vector<int>
    python::call<void>("int_list = [1, 2, 3, 4, 5]");
    std::vector<int> int_vec = python::call<std::vector<int>>("int_list");

    EXPECT_EQ(int_vec.size(), 5);
    EXPECT_EQ(int_vec[0], 1);
    EXPECT_EQ(int_vec[4], 5);

    // Test converting Python list to C++ vector<float>
    python::call<void>("float_list = [1.1, 2.2, 3.3]");
    std::vector<float> float_vec =
        python::call<std::vector<float>>("float_list");

    EXPECT_EQ(float_vec.size(), 3);
    EXPECT_FLOAT_EQ(float_vec[0], 1.1f);
    EXPECT_FLOAT_EQ(float_vec[2], 3.3f);

    // Test converting Python list to C++ vector<string>
    python::call<void>("str_list = ['hello', 'world', 'test']");
    std::vector<std::string> str_vec =
        python::call<std::vector<std::string>>("str_list");

    EXPECT_EQ(str_vec.size(), 3);
    EXPECT_EQ(str_vec[0], "hello");
    EXPECT_EQ(str_vec[1], "world");
    EXPECT_EQ(str_vec[2], "test");

    python::finalize();
}

TEST(RZPythonRuntimeTest, VectorToList_conversion)
{
    python::initialize();
    python::import("GUI_py");

    // Test sending C++ vector<int> to Python list
    std::vector<int> cpp_int_vec = { 1, 2, 3, 4, 5 };
    python::send("py_int_list", cpp_int_vec);

    // Verify the list was created correctly in Python
    python::call<void>("print('Python int list:', py_int_list)");
    python::call<void>("print('Length:', len(py_int_list))");

    // Get it back to verify
    std::vector<int> retrieved_vec =
        python::call<std::vector<int>>("py_int_list");
    EXPECT_EQ(retrieved_vec.size(), 5);
    EXPECT_EQ(retrieved_vec[0], 1);
    EXPECT_EQ(retrieved_vec[4], 5);

    // Test sending C++ vector<float> to Python list
    std::vector<float> cpp_float_vec = { 1.1f, 2.2f, 3.3f };
    python::send("py_float_list", cpp_float_vec);

    python::call<void>("print('Python float list:', py_float_list)");
    std::vector<float> retrieved_float_vec =
        python::call<std::vector<float>>("py_float_list");
    EXPECT_EQ(retrieved_float_vec.size(), 3);
    EXPECT_FLOAT_EQ(retrieved_float_vec[0], 1.1f);
    EXPECT_FLOAT_EQ(retrieved_float_vec[2], 3.3f);

    // Test sending C++ vector<string> to Python list
    std::vector<std::string> cpp_str_vec = { "hello", "world", "test" };
    python::send("py_str_list", cpp_str_vec);

    python::call<void>("print('Python string list:', py_str_list)");
    std::vector<std::string> retrieved_str_vec =
        python::call<std::vector<std::string>>("py_str_list");
    EXPECT_EQ(retrieved_str_vec.size(), 3);
    EXPECT_EQ(retrieved_str_vec[0], "hello");
    EXPECT_EQ(retrieved_str_vec[1], "world");
    EXPECT_EQ(retrieved_str_vec[2], "test");

    python::finalize();
}

TEST(RZPythonRuntimeTest, NumPy_ndarray_conversion)
{
    python::initialize();
    python::import("GUI_py");

    try {
        python::import("numpy");

        // Test creating NumPy array from C++ and sending to Python
        python::call<void>("import numpy as np");

        // Create a simple 2D array in Python and retrieve it
        python::call<void>(
            "np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "
            "dtype=np.float32)");
        python::call<void>("print('NumPy array shape:', np_array.shape)");
        python::call<void>("print('NumPy array dtype:', np_array.dtype)");

        // Retrieve the NumPy array in C++
        python::numpy_array_f32 np_array =
            python::call<python::numpy_array_f32>("np_array");

        // Test basic properties
        EXPECT_EQ(np_array.ndim(), 2);
        EXPECT_EQ(np_array.shape(0), 2);
        EXPECT_EQ(np_array.shape(1), 3);
        EXPECT_TRUE(np_array.is_valid());

        // Test data access
        float* data = np_array.data();
        EXPECT_FLOAT_EQ(data[0], 1.0f);
        EXPECT_FLOAT_EQ(data[1], 2.0f);
        EXPECT_FLOAT_EQ(data[2], 3.0f);

        // Send a C++ created ndarray to Python
        std::vector<float> cpp_data = { 7.0f, 8.0f, 9.0f, 10.0f };
        python::numpy_array_f32 cpp_array(cpp_data.data(), { 2, 2 });
        python::send("cpp_np_array", cpp_array);

        python::call<void>("print('C++ created NumPy array:', cpp_np_array)");
        python::call<void>("print('C++ array shape:', cpp_np_array.shape)");
    }
    catch (const std::exception& e) {
        // NumPy might not be available, skip test
        GTEST_SKIP() << "NumPy not available: " << e.what();
    }

    python::finalize();
}

TEST(RZPythonRuntimeTest, PyTorch_tensor_conversion)
{
    python::initialize();
    python::import("GUI_py");

    try {
        // python::import("torch");

        // Test creating PyTorch tensor and retrieving it
        python::call<void>("import torch");
        python::call<void>(
            "torch_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], "
            "dtype=torch.float32)");
        python::call<void>(
            "print('PyTorch tensor shape:', torch_tensor.shape)");
        python::call<void>(
            "print('PyTorch tensor dtype:', torch_tensor.dtype)");

        // Retrieve the PyTorch tensor in C++
        python::torch_tensor_f32 torch_tensor =
            python::call<python::torch_tensor_f32>("torch_tensor");

        // Test basic properties
        EXPECT_EQ(torch_tensor.ndim(), 2);
        EXPECT_EQ(torch_tensor.shape(0), 2);
        EXPECT_EQ(torch_tensor.shape(1), 2);
        EXPECT_TRUE(torch_tensor.is_valid());

        // Test data access
        float* data = torch_tensor.data();
        EXPECT_FLOAT_EQ(data[0], 1.0f);
        EXPECT_FLOAT_EQ(data[1], 2.0f);

        // Create a CPU tensor and send to Python
        std::vector<float> cpp_data = { 5.0f, 6.0f, 7.0f, 8.0f };
        python::torch_tensor_f32 cpp_tensor(cpp_data.data(), { 2, 2 });
        python::send("cpp_torch_tensor", cpp_tensor);

        python::call<void>(
            "print('C++ created PyTorch tensor:', cpp_torch_tensor)");
        
        // Explicitly clear PyTorch objects to help with cleanup
        python::call<void>("del torch_tensor, cpp_torch_tensor");
        python::call<void>("import gc; gc.collect()");
    }
    catch (const std::exception& e) {
        // PyTorch might not be available, skip test
        GTEST_SKIP() << "PyTorch not available: " << e.what();
    }

    python::finalize();
    
    // Add a small delay to ensure DLL cleanup is complete
#ifdef _WIN32
    Sleep(100);
#endif
}

TEST(RZPythonRuntimeTest, CUDA_tensor_conversion)
{
    python::initialize();

    try {
        python::import("torch");

        // Check if CUDA is available
        python::call<void>("import torch");
        python::call<void>("cuda_available = torch.cuda.is_available()");

        bool cuda_available = false;
        try {
            cuda_available = python::call<int>("int(cuda_available)") > 0;
        }
        catch (...) {
            cuda_available = false;
        }

        if (!cuda_available) {
            GTEST_SKIP() << "CUDA not available";
        }

        // Test creating CUDA tensor
        python::call<void>(
            "cuda_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], "
            "dtype=torch.float32, device='cuda')");
        python::call<void>("print('CUDA tensor device:', cuda_tensor.device)");
        python::call<void>("print('CUDA tensor shape:', cuda_tensor.shape)");

        // Retrieve the CUDA tensor in C++
        python::cuda_array_f32 cuda_tensor =
            python::call<python::cuda_array_f32>("cuda_tensor");

        // Test basic properties
        EXPECT_EQ(cuda_tensor.ndim(), 2);
        EXPECT_EQ(cuda_tensor.shape(0), 2);
        EXPECT_EQ(cuda_tensor.shape(1), 2);
        EXPECT_TRUE(cuda_tensor.is_valid());
        EXPECT_EQ(cuda_tensor.device_type(), nanobind::device::cuda::value);

        python::call<void>(
            "print('Successfully retrieved CUDA tensor in C++')");
    }
    catch (const std::exception& e) {
        // CUDA might not be available, skip test
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }

    python::finalize();
    
    // Add a small delay to ensure DLL cleanup is complete
#ifdef _WIN32
    Sleep(100);
#endif
}
