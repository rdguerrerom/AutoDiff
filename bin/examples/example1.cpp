// example1.cpp
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"
#include "../AutoDiff/validation.h"
#include "../AutoDiff/benchmark.h"
#include <iostream>

void run_examples() {
    using namespace ad::expr;

    
    auto x = std::make_unique<Variable<double>>("x", 2.0);
    auto sin_expr = std::make_unique<Sin<double>>(
        std::make_unique<Pow<double>>(
            x->clone(),
            std::make_unique<Constant<double>>(2)
        )
    );

    // Validate derivative of sin(x^2) at x=0.5
    bool valid = ad::test::validate_derivative(
        *sin_expr,  // The expression to differentiate
        *x,         // The variable to differentiate with respect to
        0.5         // Test point
    );
    std::cout << "Derivative validation: " << (valid ? "Passed" : "Failed") << std::endl;
    
    // Benchmarking
    auto bench = ad::benchmark::measure_performance([&]() {
        sin_expr->evaluate();
    });
    std::cout << "Benchmark - Average time per iteration: " << bench.time_ms << " ms\n";
    std::cout << "Benchmark - Memory delta: " << bench.memory_kb << " KB\n";
}

int main() {
    run_examples();
    return 0;
}
