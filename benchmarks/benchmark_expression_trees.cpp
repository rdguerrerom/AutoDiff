#include <benchmark/benchmark.h>
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"

// In benchmark_expression_trees.cpp
template <typename T>
static void BM_Expression_Construction(benchmark::State& state) {
    for (auto _ : state) {
        auto x = std::make_unique<ad::expr::Variable<T>>("x", 2.0);
        auto expr = std::make_unique<ad::expr::Exp<T>>(  // Use make_unique
            std::make_unique<ad::expr::Multiplication<T>>(  // Create unique_ptr
                x->clone(),
                std::make_unique<ad::expr::Constant<T>>(1.5)
            )
        );
        benchmark::DoNotOptimize(expr->clone());
    }
}

/*template <typename T>
static void BM_Deep_Nesting(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 0.5);
    ad::expr::ExprPtr<T> expr = x->clone();
    
    for (int i = 0; i < state.range(0); ++i) {
        expr = std::make_unique<ad::expr::Sin<T>>(
            std::make_unique<ad::expr::Addition<T>>(
                std::move(expr),
                x->clone()
            )
        );
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Deep_Nesting, double)->Arg(5)->Arg(10)->Arg(20);*/

BENCHMARK_MAIN();
