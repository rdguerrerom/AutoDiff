#include <benchmark/benchmark.h>
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"  // Add missing include

template <typename T>
static void BM_Memory_Overhead(benchmark::State& state) {
    for (auto _ : state) {
        auto expr = std::make_unique<ad::expr::Sin<T>>(
            std::make_unique<ad::expr::Exp<T>>(
                std::make_unique<ad::expr::Variable<T>>("x", 1.0)
            )
        );
        benchmark::DoNotOptimize(expr->evaluate());
    }
}
BENCHMARK_TEMPLATE(BM_Memory_Overhead, double);  // Use TEMPLATE version
BENCHMARK_MAIN();  // Add main function
