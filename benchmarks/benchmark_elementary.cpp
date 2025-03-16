#include <benchmark/benchmark.h>
#include "../AutoDiff/elementary_functions.h"

template <typename T>
static void BM_Sin(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Sin<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Sin, double);

BENCHMARK_MAIN();
