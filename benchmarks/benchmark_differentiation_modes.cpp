#include <benchmark/benchmark.h>
#include "../AutoDiff/reverse_mode.h"
#include "../AutoDiff/forward_mode.h"

using namespace ad;

// ==================== MODIFIED ELEMENTARY FUNCTION BENCHMARK ====================
template <typename T>
static void BM_ForwardMode_Elementary(benchmark::State& state) {
    forward::ForwardMode<T> fm;
    auto x = fm.variable(1.5);
    
    for (auto _ : state) {
        // Replaced lgamma with asinh
        auto z = forward::erf(x) / forward::asinh(x);
        benchmark::DoNotOptimize(fm.get_derivative(z));
    }
}
BENCHMARK_TEMPLATE(BM_ForwardMode_Elementary, double);

template <typename T>
static void BM_ReverseMode_Elementary(benchmark::State& state) {
    reverse::ReverseMode<T> rm;
    auto x = rm.add_variable("x", 1.5);
    
    for (auto _ : state) {
        // Use same function as forward mode for fair comparison
        auto erf_x = graph::erf<T>(x);
        auto asinh_x = graph::asinh<T>(x);
        auto z = graph::divide<T>(erf_x, asinh_x);
        auto gradients = rm.compute_gradients(z);
        benchmark::DoNotOptimize(gradients);
    }
}
BENCHMARK_TEMPLATE(BM_ReverseMode_Elementary, double);

// ==================== 12-VARIABLE COMPARATIVE BENCHMARKS ====================
template <typename T>
static void BM_12Vars_Forward_Eval(benchmark::State& state) {
    forward::ForwardMode<T> fm;
    std::vector<forward::ForwardVar<T>> vars;
    for(int i = 0; i < 12; ++i) {
        vars.push_back(fm.variable(static_cast<T>(i) + 1.0));
    }
    
    // Complex function using all 12 variables
    auto expr = forward::sin(vars[0]) * vars[1] + 
               forward::exp(vars[2]) / vars[3] +
               forward::tanh(vars[4]) * forward::cos(vars[5]) +
               vars[6] * vars[7] + 
               forward::sqrt(vars[8]) + forward::asinh(vars[9]) +
               vars[10] * forward::atanh(vars[11]);

    for(auto _ : state) {
        benchmark::DoNotOptimize(expr.value);
    }
}
BENCHMARK_TEMPLATE(BM_12Vars_Forward_Eval, double);

template <typename T>
static void BM_12Vars_Reverse_Eval(benchmark::State& state) {
    reverse::ReverseMode<T> rm;
    std::vector<std::shared_ptr<graph::VariableNode<T>>> vars;
    
    for(int i = 0; i < 12; ++i) {
        vars.push_back(rm.add_variable("var" + std::to_string(i), static_cast<T>(i) + 1.0));
    }
    
    // Same complex function as forward mode
    auto sin_0 = graph::sin<T>(vars[0]);
    auto term1 = graph::multiply<T>(sin_0, vars[1]);
    
    auto exp_2 = graph::exp<T>(vars[2]);
    auto term2 = graph::divide<T>(exp_2, vars[3]);
    
    auto tanh_4 = graph::tanh<T>(vars[4]);
    auto cos_5 = graph::cos<T>(vars[5]);
    auto term3 = graph::multiply<T>(tanh_4, cos_5);
    
    auto term4 = graph::multiply<T>(vars[6], vars[7]);
    
    auto sqrt_8 = graph::sqrt<T>(vars[8]);
    auto asinh_9 = graph::asinh<T>(vars[9]);
    
    auto atanh_11 = graph::atanh<T>(vars[11]);
    auto term5 = graph::multiply<T>(vars[10], atanh_11);
    
    // Chain additions
    auto sum1 = graph::add<T>(term1, term2);
    auto sum2 = graph::add<T>(sum1, term3);
    auto sum3 = graph::add<T>(sum2, term4);
    auto sum4 = graph::add<T>(sum3, sqrt_8);
    auto sum5 = graph::add<T>(sum4, asinh_9);
    auto expr = graph::add<T>(sum5, term5);
    
    for(auto _ : state) {
        // Just force evaluation
        expr->forward();
        benchmark::DoNotOptimize(expr->get_value());
    }
}
BENCHMARK_TEMPLATE(BM_12Vars_Reverse_Eval, double);

template <typename T>
static void BM_12Vars_Forward_Grad(benchmark::State& state) {
    const int numVars = 12;
    std::vector<forward::ForwardVar<T>> vars;
    
    // Explicitly initialize the variables
    for(int i = 0; i < numVars; ++i) {
        vars.push_back(forward::ForwardVar<T>(static_cast<T>(i) + 1.0, T(0)));
    }
    
    for(auto _ : state) {
        std::vector<T> gradients(numVars);
        
        for(int i = 0; i < numVars; ++i) {
            // Reset all derivatives
            for(int j = 0; j < numVars; ++j) {
                vars[j].deriv = T(0);
            }
            
            // Set the derivative of the current variable to 1
            vars[i].deriv = T(1);
            
            // Compute the expression
            auto expr = forward::sin(vars[0]) * vars[1] + 
                       forward::exp(vars[2]) / vars[3] +
                       forward::tanh(vars[4]) * forward::cos(vars[5]) +
                       vars[6] * vars[7] + 
                       forward::sqrt(vars[8]) + forward::asinh(vars[9]) +
                       vars[10] * forward::atanh(vars[11]);
            
            // Store the derivative with respect to variable i
            gradients[i] = expr.deriv;
        }
        
        benchmark::DoNotOptimize(gradients);
    }
}
BENCHMARK_TEMPLATE(BM_12Vars_Forward_Grad, double);

template <typename T>
static void BM_12Vars_Reverse_Grad(benchmark::State& state) {
    reverse::ReverseMode<T> rm;
    std::vector<std::shared_ptr<graph::VariableNode<T>>> vars;
    
    for(int i = 0; i < 12; ++i) {
        vars.push_back(rm.add_variable("var" + std::to_string(i), static_cast<T>(i) + 1.0));
    }
    
    // Same complex function as forward mode
    auto sin_0 = graph::sin<T>(vars[0]);
    auto term1 = graph::multiply<T>(sin_0, vars[1]);
    
    auto exp_2 = graph::exp<T>(vars[2]);
    auto term2 = graph::divide<T>(exp_2, vars[3]);
    
    auto tanh_4 = graph::tanh<T>(vars[4]);
    auto cos_5 = graph::cos<T>(vars[5]);
    auto term3 = graph::multiply<T>(tanh_4, cos_5);
    
    auto term4 = graph::multiply<T>(vars[6], vars[7]);
    
    auto sqrt_8 = graph::sqrt<T>(vars[8]);
    auto asinh_9 = graph::asinh<T>(vars[9]);
    
    auto atanh_11 = graph::atanh<T>(vars[11]);
    auto term5 = graph::multiply<T>(vars[10], atanh_11);
    
    // Chain additions
    auto sum1 = graph::add<T>(term1, term2);
    auto sum2 = graph::add<T>(sum1, term3);
    auto sum3 = graph::add<T>(sum2, term4);
    auto sum4 = graph::add<T>(sum3, sqrt_8);
    auto sum5 = graph::add<T>(sum4, asinh_9);
    auto expr = graph::add<T>(sum5, term5);
    
    for(auto _ : state) {
        auto gradients = rm.compute_gradients(expr);
        benchmark::DoNotOptimize(gradients);
    }
}
BENCHMARK_TEMPLATE(BM_12Vars_Reverse_Grad, double);

BENCHMARK_MAIN();
