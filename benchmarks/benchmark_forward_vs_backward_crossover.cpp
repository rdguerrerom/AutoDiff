#include <benchmark/benchmark.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

// Fix for visibility warnings
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(default)
#include <locale>
#pragma GCC visibility pop
#endif

#include "../AutoDiff/reverse_mode.h"
#include "../AutoDiff/forward_mode.h"
#include "../AutoDiff/computational_graph.h"

using namespace ad;

// Neural network layer computation benchmark
// Computes y = W·x + b where:
// - x is an input vector of size input_dim
// - W is a weight matrix of size output_dim × input_dim
// - b is a bias vector of size output_dim
// - y is the output vector of size output_dim

// Structure to hold benchmark results for analysis
struct BenchmarkResult {
    int input_dim;
    int output_dim;
    double forward_time;
    double reverse_time;
    double ratio;  // reverse/forward
};

/**
 * Forward mode benchmark for neural network layer
 * Computing gradients of all outputs with respect to all inputs
 */
template <typename T>
static void BM_NeuralLayer_Forward(benchmark::State& state) {
    const int input_dim = state.range(0);
    const int output_dim = state.range(1);
    
    // Initialize input values
    std::vector<T> x_values(input_dim);
    for (int i = 0; i < input_dim; i++) {
        x_values[i] = 0.1 * (i + 1);
    }
    
    // Initialize weights and biases
    std::vector<std::vector<T>> W_values(output_dim, std::vector<T>(input_dim));
    std::vector<T> b_values(output_dim);
    
    for (int i = 0; i < output_dim; i++) {
        b_values[i] = 0.01 * i;
        for (int j = 0; j < input_dim; j++) {
            W_values[i][j] = 0.01 * (i + 1) * (j + 1);
        }
    }
    
    // Preallocate vectors to avoid allocation during benchmarking
    std::vector<forward::ForwardVar<T>> x_forward(input_dim);
    std::vector<std::vector<forward::ForwardVar<T>>> W_forward(output_dim, std::vector<forward::ForwardVar<T>>(input_dim));
    std::vector<forward::ForwardVar<T>> b_forward(output_dim);
    std::vector<forward::ForwardVar<T>> y_forward(output_dim);
    std::vector<std::vector<T>> gradients(input_dim, std::vector<T>(output_dim));
    
    // Benchmark loop
    for (auto _ : state) {
        state.PauseTiming();
        // Reset all derivatives
        for (int i = 0; i < input_dim; i++) {
            x_forward[i] = forward::ForwardVar<T>(x_values[i], T(0));
        }
        
        for (int i = 0; i < output_dim; i++) {
            b_forward[i] = forward::ForwardVar<T>(b_values[i], T(0));
            for (int j = 0; j < input_dim; j++) {
                W_forward[i][j] = forward::ForwardVar<T>(W_values[i][j], T(0));
            }
        }
        state.ResumeTiming();
        
        // Compute gradients for each input
        for (int k = 0; k < input_dim; k++) {
            // Set derivative for current input to 1.0
            x_forward[k].deriv = T(1);
            
            // Compute layer outputs
            for (int i = 0; i < output_dim; i++) {
                // Start with bias
                y_forward[i] = b_forward[i];
                
                // Add weighted inputs
                for (int j = 0; j < input_dim; j++) {
                    y_forward[i] = y_forward[i] + W_forward[i][j] * x_forward[j];
                }
                
                // Store gradient of output i with respect to input k
                gradients[k][i] = y_forward[i].deriv;
            }
            
            // Reset derivative for current input
            x_forward[k].deriv = T(0);
        }
        
        benchmark::DoNotOptimize(gradients);
    }
    
    // Report complexity metrics
    state.SetComplexityN(input_dim * output_dim);
}

/**
 * Reverse mode benchmark for neural network layer
 * Computing gradients of all outputs with respect to all inputs
 */
template <typename T>
static void BM_NeuralLayer_Reverse(benchmark::State& state) {
    const int input_dim = state.range(0);
    const int output_dim = state.range(1);
    
    // Initialize variables outside the benchmark loop
    std::vector<T> x_values(input_dim);
    std::vector<std::vector<T>> W_values(output_dim, std::vector<T>(input_dim));
    std::vector<T> b_values(output_dim);
    
    for (int i = 0; i < input_dim; i++) {
        x_values[i] = 0.1 * (i + 1);
    }
    
    for (int i = 0; i < output_dim; i++) {
        b_values[i] = 0.01 * i;
        for (int j = 0; j < input_dim; j++) {
            W_values[i][j] = 0.01 * (i + 1) * (j + 1);
        }
    }
    
    std::vector<std::vector<T>> gradients(input_dim, std::vector<T>(output_dim));
    
    // Benchmark loop
    for (auto _ : state) {
        state.PauseTiming();
        // Create fresh computational graph for each iteration
        reverse::ReverseMode<T> rm;
        
        // Add variables to the graph
        std::vector<std::shared_ptr<graph::VariableNode<T>>> x_nodes;
        std::vector<std::vector<std::shared_ptr<graph::VariableNode<T>>>> W_nodes;
        std::vector<std::shared_ptr<graph::VariableNode<T>>> b_nodes;
        
        for (int i = 0; i < input_dim; i++) {
            x_nodes.push_back(rm.add_variable("x" + std::to_string(i), x_values[i]));
        }
        
        for (int i = 0; i < output_dim; i++) {
            std::vector<std::shared_ptr<graph::VariableNode<T>>> row;
            for (int j = 0; j < input_dim; j++) {
                row.push_back(rm.add_variable("W" + std::to_string(i) + "_" + std::to_string(j), W_values[i][j]));
            }
            W_nodes.push_back(row);
            b_nodes.push_back(rm.add_variable("b" + std::to_string(i), b_values[i]));
        }
        
                    // Build the computational graph for all outputs
        std::vector<std::shared_ptr<graph::GraphNode<T>>> y_nodes;
        
        for (int i = 0; i < output_dim; i++) {
            // Start with the bias node
            std::shared_ptr<graph::GraphNode<T>> output = b_nodes[i];
            
            // Add weighted inputs
            for (int j = 0; j < input_dim; j++) {
                auto term = graph::multiply<T>(W_nodes[i][j], x_nodes[j]);
                output = graph::add<T>(output, term);
            }
            y_nodes.push_back(output);
        }
        state.ResumeTiming();
        
        // Compute gradients for each output
        for (int i = 0; i < output_dim; i++) {
            auto grads = rm.compute_gradients(y_nodes[i]);
            
            // Extract gradients for all inputs
            for (int j = 0; j < input_dim; j++) {
                gradients[j][i] = grads["x" + std::to_string(j)];
            }
        }
        
        benchmark::DoNotOptimize(gradients);
    }
    
    // Report complexity metrics
    state.SetComplexityN(input_dim * output_dim);
}

/**
 * Custom benchmark that directly compares forward and reverse modes
 * and saves results for further analysis
 */
template <typename T>
static void BM_CompareForwardReverse(benchmark::State& state) {
    const int input_dim = state.range(0);
    const int output_dim = state.range(1);
    
    if (state.thread_index() == 0) {
        // Only output this information for the first thread
        std::cout << "\nBenchmarking: input_dim=" << input_dim 
                  << ", output_dim=" << output_dim << std::endl;
    }
    
    // Initialize vectors
    std::vector<T> x_values(input_dim);
    std::vector<std::vector<T>> W_values(output_dim, std::vector<T>(input_dim));
    std::vector<T> b_values(output_dim);
    
    for (int i = 0; i < input_dim; i++) {
        x_values[i] = 0.1 * (i + 1);
    }
    
    for (int i = 0; i < output_dim; i++) {
        b_values[i] = 0.01 * i;
        for (int j = 0; j < input_dim; j++) {
            W_values[i][j] = 0.01 * (i + 1) * (j + 1);
        }
    }
    
    // Timing variables
    double forward_time = 0;
    double reverse_time = 0;
    int iterations = 0;
    
    for (auto _ : state) {
        auto forward_start = std::chrono::high_resolution_clock::now();
        
        // Forward mode implementation
        // Preallocate vectors
        std::vector<forward::ForwardVar<T>> x_forward(input_dim);
        std::vector<std::vector<forward::ForwardVar<T>>> W_forward(output_dim, std::vector<forward::ForwardVar<T>>(input_dim));
        std::vector<forward::ForwardVar<T>> b_forward(output_dim);
        std::vector<forward::ForwardVar<T>> y_forward(output_dim);
        std::vector<std::vector<T>> gradients_forward(input_dim, std::vector<T>(output_dim));
        
        // Initialize values
        for (int i = 0; i < input_dim; i++) {
            x_forward[i] = forward::ForwardVar<T>(x_values[i], T(0));
        }
        
        for (int i = 0; i < output_dim; i++) {
            b_forward[i] = forward::ForwardVar<T>(b_values[i], T(0));
            for (int j = 0; j < input_dim; j++) {
                W_forward[i][j] = forward::ForwardVar<T>(W_values[i][j], T(0));
            }
        }
        
        // Compute gradients for each input
        for (int k = 0; k < input_dim; k++) {
            // Set derivative for current input to 1.0
            x_forward[k].deriv = T(1);
            
            // Compute layer outputs
            for (int i = 0; i < output_dim; i++) {
                // Start with bias
                y_forward[i] = b_forward[i];
                
                // Add weighted inputs
                for (int j = 0; j < input_dim; j++) {
                    y_forward[i] = y_forward[i] + W_forward[i][j] * x_forward[j];
                }
                
                // Store gradient
                gradients_forward[k][i] = y_forward[i].deriv;
            }
            
            // Reset derivative for current input
            x_forward[k].deriv = T(0);
        }
        
        benchmark::DoNotOptimize(gradients_forward);
        auto forward_end = std::chrono::high_resolution_clock::now();
        
        auto reverse_start = std::chrono::high_resolution_clock::now();
        
        // Reverse mode implementation
        reverse::ReverseMode<T> rm;
        
        // Add variables to the graph
        std::vector<std::shared_ptr<graph::VariableNode<T>>> x_nodes;
        std::vector<std::vector<std::shared_ptr<graph::VariableNode<T>>>> W_nodes;
        std::vector<std::shared_ptr<graph::VariableNode<T>>> b_nodes;
        std::vector<std::vector<T>> gradients_reverse(input_dim, std::vector<T>(output_dim));
        
        for (int i = 0; i < input_dim; i++) {
            x_nodes.push_back(rm.add_variable("x" + std::to_string(i), x_values[i]));
        }
        
        for (int i = 0; i < output_dim; i++) {
            std::vector<std::shared_ptr<graph::VariableNode<T>>> row;
            for (int j = 0; j < input_dim; j++) {
                row.push_back(rm.add_variable("W" + std::to_string(i) + "_" + std::to_string(j), W_values[i][j]));
            }
            W_nodes.push_back(row);
            b_nodes.push_back(rm.add_variable("b" + std::to_string(i), b_values[i]));
        }
        
        // Build the computational graph for all outputs
        std::vector<std::shared_ptr<graph::GraphNode<T>>> y_nodes;
        
        for (int i = 0; i < output_dim; i++) {
            // Start with the bias node - explicit cast to GraphNode<T>::Ptr
            std::shared_ptr<graph::GraphNode<T>> output = std::static_pointer_cast<graph::GraphNode<T>>(b_nodes[i]);
            
            // Add weighted inputs
            for (int j = 0; j < input_dim; j++) {
                auto term = graph::multiply<T>(W_nodes[i][j], x_nodes[j]);
                output = graph::add<T>(output, term);
            }
            y_nodes.push_back(output);
        }
        
        // Compute gradients for each output
        for (int i = 0; i < output_dim; i++) {
            auto grads = rm.compute_gradients(y_nodes[i]);
            
            // Extract gradients for all inputs
            for (int j = 0; j < input_dim; j++) {
                gradients_reverse[j][i] = grads["x" + std::to_string(j)];
            }
        }
        
        benchmark::DoNotOptimize(gradients_reverse);
        auto reverse_end = std::chrono::high_resolution_clock::now();
        
        // Calculate timings
        std::chrono::duration<double, std::milli> forward_duration = forward_end - forward_start;
        std::chrono::duration<double, std::milli> reverse_duration = reverse_end - reverse_start;
        
        forward_time += forward_duration.count();
        reverse_time += reverse_duration.count();
        iterations++;
        
        // Verify correctness by comparing gradients
        if (state.thread_index() == 0 && iterations == 1) {
            bool correct = true;
            for (int i = 0; i < input_dim && correct; i++) {
                for (int j = 0; j < output_dim && correct; j++) {
                    if (std::abs(gradients_forward[i][j] - gradients_reverse[i][j]) > 1e-10) {
                        std::cout << "Gradient mismatch at [" << i << "][" << j << "]: "
                                  << gradients_forward[i][j] << " vs " << gradients_reverse[i][j] << std::endl;
                        correct = false;
                    }
                }
            }
            if (correct) {
                std::cout << "Gradients match between forward and reverse mode" << std::endl;
            }
        }
    }
    
    // Compute average times
    forward_time /= iterations;
    reverse_time /= iterations;
    double ratio = reverse_time / forward_time;
    
    if (state.thread_index() == 0) {
        std::cout << "Forward mode: " << forward_time << " ms, "
                  << "Reverse mode: " << reverse_time << " ms, "
                  << "Ratio (reverse/forward): " << ratio << std::endl;
        
        // Save results to file
        std::ofstream result_file("ad_crossover_results.csv", std::ios_base::app);
        if (result_file.is_open()) {
            result_file << input_dim << "," << output_dim << ","
                       << forward_time << "," << reverse_time << ","
                       << ratio << std::endl;
            result_file.close();
        }
    }
    
    // Set custom counter metrics
    state.counters["forward_ms"] = forward_time;
    state.counters["reverse_ms"] = reverse_time;
    state.counters["ratio"] = ratio;
    state.counters["input_dim"] = input_dim;
    state.counters["output_dim"] = output_dim;
    state.counters["parameters"] = input_dim * output_dim + output_dim;
    state.counters["io_ratio"] = static_cast<double>(input_dim) / output_dim;
}

// Range configurations for input and output dimensions
static void CrossoverArgumentGenerator(benchmark::internal::Benchmark* b) {
    // Create header for results file
    std::ofstream result_file("ad_crossover_results.csv", std::ios_base::trunc);
    if (result_file.is_open()) {
        result_file << "input_dim,output_dim,forward_time,reverse_time,ratio" << std::endl;
        result_file.close();
    }
    
    // Output cases: Fixed small output (1) with varying inputs
    for (int i = 1; i <= 100; i += (i < 10 ? 1 : (i < 50 ? 5 : 10))) {
        b->Args({i, 1});
    }
    
    // Input cases: Fixed small input (1) with varying outputs
    for (int i = 1; i <= 100; i += (i < 10 ? 1 : (i < 50 ? 5 : 10))) {
        b->Args({1, i});
    }
    
    // Square cases: Equal inputs and outputs
    for (int i = 1; i <= 50; i += (i < 10 ? 1 : 5)) {
        b->Args({i, i});
    }
    
    // Additional interesting cases near expected crossover point
    // Based on your benchmark data showing crossover beyond 12 variables
    for (int in_dim = 10; in_dim <= 30; in_dim += 2) {
        for (int out_dim = 1; out_dim <= 10; out_dim += 1) {
            b->Args({in_dim, out_dim});
        }
    }
    
    // Some extreme cases
    b->Args({100, 1});   // Many inputs, one output (favors reverse)
    b->Args({1, 100});   // One input, many outputs (favors forward)
    b->Args({50, 10});   // Many inputs, fewer outputs (potential crossover)
    b->Args({10, 50});   // Fewer inputs, many outputs (favors forward)
}

// Register benchmarks
BENCHMARK_TEMPLATE(BM_CompareForwardReverse, double)->Apply(CrossoverArgumentGenerator);

// Optional: Individual forward/reverse benchmarks for more detailed analysis
BENCHMARK_TEMPLATE(BM_NeuralLayer_Forward, double)
    ->Args({1, 1})
    ->Args({10, 1})
    ->Args({1, 10})
    ->Args({10, 10})
    ->Args({50, 1})
    ->Args({1, 50})
    ->Args({20, 20})
    ->Args({100, 1})
    ->Args({1, 100})
    ->Complexity();

BENCHMARK_TEMPLATE(BM_NeuralLayer_Reverse, double)
    ->Args({1, 1})
    ->Args({10, 1})
    ->Args({1, 10})
    ->Args({10, 10})
    ->Args({50, 1})
    ->Args({1, 50})
    ->Args({20, 20})
    ->Args({100, 1})
    ->Args({1, 100})
    ->Complexity();

// Optional: Analysis Helper Function 
// This Python script will automatically generate a plot from the CSV results
BENCHMARK_MAIN();

/*
// Create a Python script for visualization (not executed directly)
const char* python_script = R"(
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the results
df = pd.read_csv('ad_crossover_results.csv')

# Calculate the crossover point (where ratio ≈ 1.0)
df['abs_log_ratio'] = abs(np.log(df['ratio']))
crossover_idx = df['abs_log_ratio'].idxmin()
crossover_point = df.iloc[crossover_idx]

# Create a scatterplot of input_dim vs output_dim, colored by ratio
plt.figure(figsize=(12, 10))

# Use a diverging colormap centered at ratio=1.0 (log ratio = 0)
# Create a custom colormap
cmap = sns.diverging_palette(240, 10, as_cmap=True)

plt.figure(figsize=(14, 12))

# Create a scatter plot where point size correlates with the total parameters
scatter = plt.scatter(df['input_dim'], df['output_dim'], 
                      c=np.log2(df['ratio']), cmap=cmap, 
                      s=np.sqrt(df['input_dim'] * df['output_dim'] + df['output_dim'])*3,
                      alpha=0.7, edgecolors='black', linewidth=0.5)

# Add labels for the crossover point
plt.annotate(f"Crossover Point\nInput Dim: {crossover_point['input_dim']}\nOutput Dim: {crossover_point['output_dim']}\nRatio: {crossover_point['ratio']:.2f}",
             xy=(crossover_point['input_dim'], crossover_point['output_dim']),
             xytext=(crossover_point['input_dim']+5, crossover_point['output_dim']+5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('log₂(Reverse/Forward Ratio)', fontsize=12)
# Add custom ticks to make interpretation easier
cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
cbar.set_ticklabels(['1/8 (Forward 8× faster)', '1/4 (Forward 4× faster)', '1/2 (Forward 2× faster)', 
                     '1 (Equal)', 
                     '2 (Reverse 2× faster)', '4 (Reverse 4× faster)', '8 (Reverse 8× faster)'])

# Add diagonal line where input_dim = output_dim
max_dim = max(df['input_dim'].max(), df['output_dim'].max())
plt.plot([0, max_dim], [0, max_dim], 'k--', alpha=0.3, label='input_dim = output_dim')

# Add a horizontal line at y=1 (representing single output cases)
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.3, label='Single Output')

# Add a vertical line at x=1 (representing single input cases)
plt.axvline(x=1, color='gray', linestyle='--', alpha=0.3, label='Single Input')

# Emphasize the region where the ratio is close to 1
# Create a circle around the crossover point
circle = plt.Circle((crossover_point['input_dim'], crossover_point['output_dim']), 
                   radius=max(crossover_point['input_dim'], crossover_point['output_dim'])/10, 
                   fill=False, color='black', linewidth=2)
ax = plt.gca()
ax.add_patch(circle)

plt.xscale('log')
plt.yscale('log')
plt.xlim(0.9, df['input_dim'].max() * 1.1)
plt.ylim(0.9, df['output_dim'].max() * 1.1)
plt.xlabel('Input Dimension', fontsize=14)
plt.ylabel('Output Dimension', fontsize=14)
plt.title('Automatic Differentiation Mode Crossover Analysis\nColored by log₂(Reverse/Forward Ratio)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Add some annotations to guide interpretation
plt.annotate("Forward Mode Advantage", 
             xy=(5, 60), 
             fontsize=14, 
             color='blue',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

plt.annotate("Reverse Mode Advantage", 
             xy=(60, 5), 
             fontsize=14, 
             color='red',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

plt.savefig('ad_crossover_analysis.png', dpi=300, bbox_inches='tight')

# Create another plot focusing on the input/output ratio
plt.figure(figsize=(14, 8))
df['io_ratio'] = df['input_dim'] / df['output_dim']
df = df.sort_values('io_ratio')

plt.plot(df['io_ratio'], df['ratio'], 'o-', markersize=8)
plt.axhline(y=1, color='r', linestyle='--', label='Equal Performance')

# Find approximate crossover point
crossover_idx = (df['ratio'] - 1).abs().idxmin()
crossover_ratio = df.iloc[crossover_idx]['io_ratio']
crossover_value = df.iloc[crossover_idx]['ratio']

plt.annotate(f"Crossover at I/O ratio ≈ {crossover_ratio:.2f}",
             xy=(crossover_ratio, crossover_value),
             xytext=(crossover_ratio, crossover_value*1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.xlabel('Input/Output Dimension Ratio', fontsize=14)
plt.ylabel('Performance Ratio (Reverse/Forward)', fontsize=14)
plt.title('Performance Ratio as Function of Input/Output Dimension Ratio', fontsize=16)
plt.legend()
plt.savefig('ad_ratio_analysis.png', dpi=300, bbox_inches='tight')

print(f"Crossover point analysis complete. See generated images.")
)";
*/
