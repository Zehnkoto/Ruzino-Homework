#include <gtest/gtest.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include <array>
#include <chrono>
#include <iomanip>
#include <random>
#include <vector>

#include "rzpython/rzpython.hpp"
#include "rzpython/usd_extensions.hpp"

using namespace USTC_CG;
using namespace std::chrono;

class RZPythonPerformanceTest : public ::testing::Test {
   protected:
    static void SetUpTestSuite()
    {
        python::initialize();

        // Import required modules
        try {
            python::import("numpy");
            python::import("RHI_py");
            python::call<void>("import numpy as np");
        }
        catch (...) {
            // NumPy not available
        }

        try {
            python::call<void>("import torch");
        }
        catch (...) {
            // PyTorch not available
        }

        try {
            python::call<void>("from pxr import Vt, Gf");
        }
        catch (...) {
            // USD not available
        }
    }

    static void TearDownTestSuite()
    {
        python::finalize();
    }

    void SetUp() override
    {
        // Clean up between tests
        python::call<void>("import gc; gc.collect()");
    }

    // Generate test data
    std::vector<float> generate_float_data(size_t size)
    {
        std::vector<float> data;
        data.reserve(size);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 100.0f);

        for (size_t i = 0; i < size; ++i) {
            data.push_back(dis(gen));
        }
        return data;
    }

    // Bandwidth measurement helper
    template<typename Func>
    void measure_bandwidth(
        const std::string& operation_name,
        size_t data_size_floats,
        Func&& operation,
        int iterations = 50)
    {
        // Warm up
        for (int i = 0; i < 3; ++i) {
            operation();
        }

        auto start = high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            operation();
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);

        double avg_time_ms = duration.count() / 1000.0 / iterations;
        size_t memory_bytes = data_size_floats * sizeof(float);
        double bandwidth_mbps =
            (memory_bytes / 1024.0 / 1024.0) / (avg_time_ms / 1000.0);

        std::cout << std::fixed << std::setprecision(2) << std::setw(25)
                  << operation_name << ": " << std::setw(8) << data_size_floats
                  << " floats, " << std::setw(8) << avg_time_ms << " ms, "
                  << std::setw(10) << bandwidth_mbps << " MB/s" << std::endl;
    }

    void test_scale(const std::string& scale_name, size_t data_size)
    {
        std::cout << "\n=== " << scale_name << " (" << data_size
                  << " floats) ===" << std::endl;
        auto test_data = generate_float_data(data_size);

        // Test 1: C++ vector -> Python list (send operation)
        measure_bandwidth("Vector->List (Send)", data_size, [&]() {
            python::send("test_vector", test_data);
        });

        // Test 2: Python list -> C++ vector (call operation)
        python::send("python_list", test_data);  // Setup data
        measure_bandwidth("List->Vector (Call)", data_size, [&]() {
            auto retrieved = python::call<std::vector<float>>("python_list");
            (void)retrieved;
        });

        // Test 3: C++ -> NumPy ndarray (send operation)
        measure_bandwidth("Vector->NumpyArray", data_size, [&]() {
            python::numpy_array_f32 np_array(test_data.data(), { data_size });
            python::send("test_np_array", np_array);
        });

        // Test 4: NumPy ndarray -> C++ (call operation)
        python::numpy_array_f32 setup_array(test_data.data(), { data_size });
        python::send("numpy_array", setup_array);  // Setup data
        measure_bandwidth("NumpyArray->Vector", data_size, [&]() {
            auto retrieved_np =
                python::call<python::numpy_array_f32>("numpy_array");
            std::vector<float> retrieved(
                retrieved_np.data(), retrieved_np.data() + retrieved_np.size());
            (void)retrieved;
        });

        // Test 5: PyTorch tensor operations (if available)
        try {
            // C++ -> PyTorch tensor
            measure_bandwidth("Vector->TorchTensor", data_size, [&]() {
                python::torch_tensor_f32 torch_tensor(
                    test_data.data(), { data_size });
                python::send("test_torch_tensor", torch_tensor);
            });

            // Setup PyTorch tensor for retrieval test
            python::torch_tensor_f32 setup_tensor(
                test_data.data(), { data_size });
            python::send("torch_tensor", setup_tensor);

            // PyTorch tensor -> C++
            measure_bandwidth("TorchTensor->Vector", data_size, [&]() {
                auto retrieved_tensor =
                    python::call<python::torch_tensor_f32>("torch_tensor");
                std::vector<float> retrieved(
                    retrieved_tensor.data(),
                    retrieved_tensor.data() + retrieved_tensor.size());
                (void)retrieved;
            });
        }
        catch (const std::exception& e) {
            std::cout << std::setw(25) << "PyTorch operations" << ": Skipped ("
                      << e.what() << ")" << std::endl;
        }

        // Test 6: USD VtArray operations (if available)
        try {
            pxr::VtArray<float> usd_data;
            usd_data.assign(test_data.begin(), test_data.end());

            // C++ VtArray -> Python
            measure_bandwidth(
                "VtArray->Python",
                data_size,
                [&]() { python::usd::send_usd("test_vt_array", usd_data); },
                20);  // Fewer iterations for USD

            // Setup USD data for retrieval test
            python::usd::send_usd("vt_array", usd_data);

            // Python VtArray -> C++
            measure_bandwidth(
                "Python->VtArray",
                data_size,
                [&]() {
                    auto retrieved_vt =
                        python::usd::call_usd<pxr::VtArray<float>>("vt_array");
                    (void)retrieved_vt;
                },
                20);
        }
        catch (const std::exception& e) {
            std::cout << std::setw(25) << "USD VtArray operations"
                      << ": Skipped (" << e.what() << ")" << std::endl;
        }
    }
};

TEST_F(RZPythonPerformanceTest, Small_Scale_Operations)
{
    test_scale("Small Scale", 100);  // 100 floats = 400 bytes
}

TEST_F(RZPythonPerformanceTest, Medium_Scale_Operations)
{
    test_scale("Medium Scale", 10000);  // 10K floats = 40 KB
}

TEST_F(RZPythonPerformanceTest, Large_Scale_Operations)
{
    test_scale("Large Scale", 1000000);  // 1M floats = 4 MB
}

TEST_F(RZPythonPerformanceTest, Memory_Copy_Overhead_Analysis)
{
    std::cout << "\n=== Memory Copy Overhead Analysis ===" << std::endl;

    const std::vector<size_t> test_sizes = { 1000, 10000, 100000, 1000000 };

    for (size_t size : test_sizes) {
        auto test_data = generate_float_data(size);

        std::cout << "\nData size: " << size << " floats ("
                  << (size * sizeof(float) / 1024.0) << " KB)" << std::endl;

        // Direct vector send (nanobind handles conversion)
        measure_bandwidth(
            "Direct Vector Send",
            size,
            [&]() { python::send("direct_vector", test_data); },
            30);

        // Via numpy conversion (copy overhead)
        measure_bandwidth(
            "Via Numpy Convert",
            size,
            [&]() {
                python::send("temp_list", test_data);
                python::call<void>(
                    "temp_np = np.array(temp_list, dtype=np.float32)");
                python::call<void>("del temp_list, temp_np");
            },
            20);

        // Direct numpy array (zero-copy when possible)
        measure_bandwidth(
            "Direct Numpy Array",
            size,
            [&]() {
                python::numpy_array_f32 np_array(test_data.data(), { size });
                python::send("direct_numpy", np_array);
            },
            30);
    }
}

TEST_F(RZPythonPerformanceTest, Round_Trip_Performance)
{
    std::cout << "\n=== Round Trip Performance ===" << std::endl;

    const std::vector<size_t> test_sizes = { 1000, 10000, 100000 };

    for (size_t size : test_sizes) {
        auto test_data = generate_float_data(size);

        std::cout << "\nRound trip for " << size << " floats:" << std::endl;

        // Vector round trip
        measure_bandwidth(
            "Vector Round Trip",
            size,
            [&]() {
                python::send("round_trip_vector", test_data);
                auto retrieved =
                    python::call<std::vector<float>>("round_trip_vector");
                (void)retrieved;
            },
            20);

        // NumPy array round trip
        measure_bandwidth(
            "NumPy Round Trip",
            size,
            [&]() {
                python::numpy_array_f32 np_array(test_data.data(), { size });
                python::send("round_trip_numpy", np_array);
                auto retrieved_np =
                    python::call<python::numpy_array_f32>("round_trip_numpy");
                (void)retrieved_np;
            },
            20);

        // PyTorch tensor round trip (if available)
        try {
            measure_bandwidth(
                "Torch Round Trip",
                size,
                [&]() {
                    python::torch_tensor_f32 torch_tensor(
                        test_data.data(), { size });
                    python::send("round_trip_torch", torch_tensor);
                    auto retrieved_tensor =
                        python::call<python::torch_tensor_f32>(
                            "round_trip_torch");
                    (void)retrieved_tensor;
                },
                15);
        }
        catch (const std::exception& e) {
            std::cout << std::setw(25) << "Torch Round Trip" << ": Skipped"
                      << std::endl;
        }
    }
}

TEST_F(RZPythonPerformanceTest, Data_Type_Comparison)
{
    std::cout << "\n=== Data Type Performance Comparison ===" << std::endl;

    const size_t test_size = 100000;  // 100K elements for comparison

    // Float data
    auto float_data = generate_float_data(test_size);
    std::cout << "\nFloat data (" << test_size << " elements):" << std::endl;
    measure_bandwidth("Float Vector Send", test_size, [&]() {
        python::send("float_test", float_data);
    });

    // Int data
    std::vector<int> int_data;
    int_data.reserve(test_size);
    for (size_t i = 0; i < test_size; ++i) {
        int_data.push_back(static_cast<int>(float_data[i]));
    }
    std::cout << "\nInt data (" << test_size << " elements):" << std::endl;
    measure_bandwidth("Int Vector Send", test_size, [&]() {
        python::send("int_test", int_data);
    });

    // String data (much larger per element)
    std::vector<std::string> string_data;
    string_data.reserve(1000);  // Much smaller count for strings
    for (size_t i = 0; i < 1000; ++i) {
        string_data.push_back("test_string_" + std::to_string(i));
    }
    std::cout << "\nString data (1000 elements):" << std::endl;
    measure_bandwidth("String Vector Send", 1000, [&]() {
        python::send("string_test", string_data);
    });
}
