#include <benchmark/benchmark.h>
#include "../AutoDiff/dual_number.h"

template <typename T>
static void BM_DualNumber_Arithmetic(benchmark::State& state) {
    ad::core::DualNumber<T> a(2.0, 3.0);
    ad::core::DualNumber<T> b(1.5, 2.5);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(a + b);
        benchmark::DoNotOptimize(a * b);
        benchmark::DoNotOptimize(a / b);
        benchmark::DoNotOptimize(a - b);
    }
}
BENCHMARK_TEMPLATE(BM_DualNumber_Arithmetic, double);

static void BM_Dual_vs_Raw(benchmark::State& state) {
    double a = 2.0, b = 1.5;
    for (auto _ : state) {
        benchmark::DoNotOptimize(a + b);
        benchmark::DoNotOptimize(a * b);
        benchmark::DoNotOptimize(a / b);
        benchmark::DoNotOptimize(a - b);
    }
}
BENCHMARK(BM_Dual_vs_Raw);
BENCHMARK_MAIN();
