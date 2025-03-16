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

template <typename T>
static void BM_Cos(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Cos<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Cos, double);

template <typename T>
static void BM_Tan(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Tan<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Tan, double);

template <typename T>
static void BM_Exp(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Exp<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Exp, double);

template <typename T>
static void BM_Log(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Log<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Log, double);

template <typename T>
static void BM_Sqrt(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Sqrt<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Sqrt, double);

template <typename T>
static void BM_Reciprocal(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Reciprocal<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Reciprocal, double);

template <typename T>
static void BM_Erf(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Erf<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Erf, double);

template <typename T>
static void BM_Erfc(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Erfc<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Erfc, double);

template <typename T>
static void BM_Tgamma(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Tgamma<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Tgamma, double);

template <typename T>
static void BM_Lgamma(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Lgamma<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Lgamma, double);

template <typename T>
static void BM_Sinh(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Sinh<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Sinh, double);

template <typename T>
static void BM_Cosh(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Cosh<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Cosh, double);

template <typename T>
static void BM_Tanh(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Tanh<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Tanh, double);

template <typename T>
static void BM_Asinh(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Asinh<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Asinh, double);

template <typename T>
static void BM_Acosh(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto expr = std::make_unique<ad::expr::Acosh<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Acosh, double);

template <typename T>
static void BM_Atanh(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 0.5); // Use 0.5 to stay within domain (-1 < x < 1)
    auto expr = std::make_unique<ad::expr::Atanh<T>>(x->clone());
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
        benchmark::DoNotOptimize(expr->differentiate("x"));
    }
}
BENCHMARK_TEMPLATE(BM_Atanh, double);

BENCHMARK_MAIN();
