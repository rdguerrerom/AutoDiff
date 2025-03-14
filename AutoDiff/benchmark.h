// benchmark.h
#pragma once
#include <chrono>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace ad {
namespace benchmark {

struct BenchmarkResult {
    double time_ms;      // Average time per iteration
    double memory_kb;    // Average memory difference per iteration
};

namespace detail {
    long get_memory_usage_kb() {
        std::ifstream status("/proc/self/status");
        std::string line;
        
        while (std::getline(status, line)) {
            if (line.compare(0, 6, "VmRSS:") == 0) {
                // Extract numeric value from line like "VmRSS:    1234 kB"
                std::istringstream iss(line.substr(6));
                long value;
                std::string unit;
                iss >> value >> unit;
                return value;  // Returns KB (as reported by VmRSS)
            }
        }
        return 0;  // Fallback if not found
    }
} // namespace detail

template <typename Func>
BenchmarkResult measure_performance(Func&& func, int iterations = 1000) {
    // Warm-up run to initialize any static memory
    func();
    
    // Memory measurement
    const long base_mem = detail::get_memory_usage_kb();
    long start_mem = detail::get_memory_usage_kb();
    long peak_mem = start_mem;
    
    // Time measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        // Update peak memory before each iteration
        peak_mem = std::max(peak_mem, detail::get_memory_usage_kb());
        func();
        // Update peak memory after each iteration
        peak_mem = std::max(peak_mem, detail::get_memory_usage_kb());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    //const long end_mem = detail::get_memory_usage_kb();
    
    // Calculate results
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    
    return {
        duration.count() / iterations,
        static_cast<double>(peak_mem - base_mem)  // Peak memory delta from baseline
    };
}

} // namespace benchmark
} // namespace ad
