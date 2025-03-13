// examples.cpp
#include "expression.h"
#include "elementary_functions.h"
#include "validation.h"
#include <iostream>

void run_examples() {
    using namespace ad::expr;
    
    // Example 1: Polynomial
    auto x = make_variable<double>("x", 2.0);
    auto poly = make_expression<double, Addition<double>>(
        make_expression<double, Multiplication<double>>(
            x->clone(),
            make_expression<double, Addition<double>>(
                x->clone(),
                make_constant<double>(3.0)
            )
        ),
        make_constant<double>(2.0)
    );
    
    auto deriv = poly->differentiate("x");
    std::cout << "Polynomial derivative at x=2: " << deriv->evaluate() << std::endl;

    // Example 2: Trigonometric function
    auto sin_expr = make_expression<double, Sin<double>>(
        make_expression<double, Multiplication<double>>(
            x->clone(),
            x->clone()
        )
    );
    
    // Validation
    bool valid = ad::test::validate_derivative(*sin_expr, *x, 0.5);
    std::cout << "Derivative validation: " << (valid ? "PASSED" : "FAILED") << std::endl;

    // Benchmark
    auto bench = ad::benchmark::measure_performance([&]() {
        sin_expr->evaluate();
    });
    std::cout << "Evaluation time: " << bench.time_ms << "ms\n";
}

int main() {
    run_examples();
    return 0;
}
