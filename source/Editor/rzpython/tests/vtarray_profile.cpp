#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/vt/array.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "rzpython/rzpython.hpp"
#include "rzpython/usd_extensions.hpp"

using namespace USTC_CG;
using namespace std::chrono;

double time_ms(auto&& operation, int iterations = 10)
{
    // Warm up
    for (int i = 0; i < 2; ++i) {
        operation();
    }

    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        operation();
    }
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start);
    return duration.count() / 1000.0 / iterations;  // Return average time in ms
}

int main()
{
    try {
        std::cout << "=== VtArray Performance Profiler (100K elements) ==="
                  << std::endl;

        // Initialize Python
        python::initialize();
        python::import("RHI_py");
        python::call<void>("import numpy as np");
        python::call<void>("from pxr import Vt, Gf");

        const size_t SIZE = 100000;

        // Generate test data
        std::cout << "\nGenerating test data..." << std::endl;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 100.0f);

        std::vector<float> float_data;
        float_data.reserve(SIZE);
        for (size_t i = 0; i < SIZE; ++i) {
            float_data.push_back(dis(gen));
        }

        pxr::VtArray<float> vt_data;
        vt_data.assign(float_data.begin(), float_data.end());

        std::cout << "Test data ready: " << SIZE << " floats ("
                  << (SIZE * sizeof(float) / 1024.0) << " KB)" << std::endl;

        // =================================================================
        // BASELINE PERFORMANCE
        // =================================================================
        std::cout << "\n=== Baseline Performance ===" << std::endl;

        // 1. Direct vector send (nanobind)
        double vector_time =
            time_ms([&]() { python::send("test_vector", float_data); });
        std::cout << "Vector Send:       " << std::fixed << std::setprecision(3)
                  << vector_time << " ms" << std::endl;

        // 2. NumPy array send
        double numpy_time = time_ms([&]() {
            python::numpy_array_f32 np_array(float_data.data(), { SIZE });
            python::send("test_numpy", np_array);
        });
        std::cout << "NumPy Send:        " << std::fixed << std::setprecision(3)
                  << numpy_time << " ms" << std::endl;

        // =================================================================
        // VTARRAY PERFORMANCE
        // =================================================================
        std::cout << "\n=== VtArray<float> Performance ===" << std::endl;

        // 3. VtArray C++ -> Python
        double vtarray_send_time = time_ms(
            [&]() { python::usd::send_usd("test_vtarray", vt_data); },
            400);  // Fewer iterations for slow operations
        std::cout << "VtArray Send:      " << std::fixed << std::setprecision(3)
                  << vtarray_send_time << " ms" << std::endl;

        // 4. VtArray Python -> C++ (setup first)
        python::usd::send_usd("vtarray_for_recv", vt_data);
        double vtarray_recv_time = time_ms(
            [&]() {
                auto result = python::usd::call_usd<pxr::VtArray<float>>(
                    "vtarray_for_recv");
                (void)result;
            },
            0);
        std::cout << "VtArray Recv:      " << std::fixed << std::setprecision(3)
                  << vtarray_recv_time << " ms" << std::endl;

        // 5. VtArray round trip
        double vtarray_roundtrip_time = time_ms(
            [&]() {
                python::usd::send_usd("roundtrip_vtarray", vt_data);
                auto result = python::usd::call_usd<pxr::VtArray<float>>(
                    "roundtrip_vtarray");
                (void)result;
            },
            3);
        std::cout << "VtArray Round Trip:" << std::fixed << std::setprecision(3)
                  << vtarray_roundtrip_time << " ms" << std::endl;

        // =================================================================
        // DETAILED BREAKDOWN
        // =================================================================
        std::cout << "\n=== Detailed VtArray Breakdown ===" << std::endl;

        // Step 1: Data preparation
        double prep_time = time_ms([&]() {
            std::vector<float> temp_data(vt_data.begin(), vt_data.end());
            (void)temp_data;
        });
        std::cout << "1. Data Prep:      " << std::fixed << std::setprecision(3)
                  << prep_time << " ms" << std::endl;

        // Step 2: Vector send to Python
        std::vector<float> temp_data(vt_data.begin(), vt_data.end());
        double send_step_time =
            time_ms([&]() { python::send("breakdown_vector", temp_data); });
        std::cout << "2. Vector Send:    " << std::fixed << std::setprecision(3)
                  << send_step_time << " ms" << std::endl;

        // Step 3: USD conversion in Python
        python::send("breakdown_data", temp_data);
        double usd_convert_time = time_ms([&]() {
            python::call<void>(
                "breakdown_result = Vt.FloatArray(breakdown_data)");
        });
        std::cout << "3. USD Convert:    " << std::fixed << std::setprecision(3)
                  << usd_convert_time << " ms" << std::endl;

        double estimated_total = prep_time + send_step_time + usd_convert_time;
        std::cout << "   Estimated Total:" << std::fixed << std::setprecision(3)
                  << estimated_total << " ms" << std::endl;
        std::cout << "   Actual Total:   " << std::fixed << std::setprecision(3)
                  << vtarray_send_time << " ms" << std::endl;

        double overhead = vtarray_send_time - estimated_total;
        std::cout << "   Overhead:       " << std::fixed << std::setprecision(3)
                  << overhead << " ms (" << std::fixed << std::setprecision(1)
                  << (overhead / vtarray_send_time * 100.0) << "%)"
                  << std::endl;

        // =================================================================
        // BANDWIDTH ANALYSIS
        // =================================================================
        std::cout << "\n=== Bandwidth Analysis ===" << std::endl;

        size_t bytes = SIZE * sizeof(float);
        double mb_size = bytes / 1024.0 / 1024.0;

        std::cout << "Data size: " << mb_size << " MB" << std::endl;
        std::cout << "Vector Bandwidth:  " << std::fixed << std::setprecision(2)
                  << mb_size / (vector_time / 1000.0) << " MB/s" << std::endl;
        std::cout << "NumPy Bandwidth:   " << std::fixed << std::setprecision(2)
                  << mb_size / (numpy_time / 1000.0) << " MB/s" << std::endl;
        std::cout << "VtArray Send BW:   " << std::fixed << std::setprecision(2)
                  << mb_size / (vtarray_send_time / 1000.0) << " MB/s"
                  << std::endl;
        std::cout << "VtArray Recv BW:   " << std::fixed << std::setprecision(2)
                  << mb_size / (vtarray_recv_time / 1000.0) << " MB/s"
                  << std::endl;

        // =================================================================
        // PERFORMANCE COMPARISON
        // =================================================================
        std::cout << "\n=== Performance Ratios ===" << std::endl;
        std::cout << "VtArray vs Vector: " << std::fixed << std::setprecision(1)
                  << (vtarray_send_time / vector_time) << "x slower"
                  << std::endl;
        std::cout << "VtArray vs NumPy:  " << std::fixed << std::setprecision(1)
                  << (vtarray_send_time / numpy_time) << "x slower"
                  << std::endl;
        std::cout << "Vector vs NumPy:   " << std::fixed << std::setprecision(1)
                  << (vector_time / numpy_time) << "x slower" << std::endl;

        python::finalize();

        std::cout << "\nProfiling complete!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
