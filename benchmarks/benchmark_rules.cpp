#include <benchmark/benchmark.h>
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"
#include "../AutoDiff/differentiation_rules.h"

namespace ad = ad;

// Chain Rule Benchmarks
template <typename T>
static void BM_ChainRule_Construction(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto outer = std::make_unique<ad::expr::Exp<T>>(x->clone());
    auto inner = std::make_unique<ad::expr::Sin<T>>(x->clone());
    
    for (auto _ : state) {
        auto outer_deriv = outer->clone();  // f'(g(x)) = exp(x)
        auto inner_deriv = inner->differentiate("x"); // g'(x) = cos(x)
        auto result = ad::rules::chain_rule(std::move(outer_deriv), std::move(inner_deriv));
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK_TEMPLATE(BM_ChainRule_Construction, double);

template <typename T>
static void BM_ChainRule_Evaluation(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto outer = std::make_unique<ad::expr::Exp<T>>(x->clone());
    auto inner = std::make_unique<ad::expr::Sin<T>>(x->clone());
    auto expr = ad::rules::chain_rule(outer->clone(), inner->differentiate("x"));
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
    }
}
BENCHMARK_TEMPLATE(BM_ChainRule_Evaluation, double);

// Product Rule Benchmarks
template <typename T>
static void BM_ProductRule_Construction(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto f = std::make_unique<ad::expr::Multiplication<T>>(x->clone(), x->clone());
    auto g = std::make_unique<ad::expr::Sin<T>>(x->clone());
    
    for (auto _ : state) {
        auto df = f->differentiate("x");
        auto dg = g->differentiate("x");
        auto result = ad::rules::product_rule(
            f->clone(), std::move(df),
            g->clone(), std::move(dg)
        );
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK_TEMPLATE(BM_ProductRule_Construction, double);

template <typename T>
static void BM_ProductRule_Evaluation(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto f = std::make_unique<ad::expr::Multiplication<T>>(x->clone(), x->clone());
    auto g = std::make_unique<ad::expr::Sin<T>>(x->clone());
    auto expr = ad::rules::product_rule(
        f->clone(), f->differentiate("x"),
        g->clone(), g->differentiate("x")
    );
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
    }
}
BENCHMARK_TEMPLATE(BM_ProductRule_Evaluation, double);

// Quotient Rule Benchmarks
template <typename T>
static void BM_QuotientRule_Construction(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto f = std::make_unique<ad::expr::Multiplication<T>>(x->clone(), x->clone());
    auto g = std::make_unique<ad::expr::Log<T>>(x->clone());
    
    for (auto _ : state) {
        auto df = f->differentiate("x");
        auto dg = g->differentiate("x");
        auto result = ad::rules::quotient_rule(
            f->clone(), std::move(df),
            g->clone(), std::move(dg)
        );
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK_TEMPLATE(BM_QuotientRule_Construction, double);

template <typename T>
static void BM_QuotientRule_Evaluation(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto f = std::make_unique<ad::expr::Multiplication<T>>(x->clone(), x->clone());
    auto g = std::make_unique<ad::expr::Log<T>>(x->clone());
    auto expr = ad::rules::quotient_rule(
        f->clone(), f->differentiate("x"),
        g->clone(), g->differentiate("x")
    );
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
    }
}
BENCHMARK_TEMPLATE(BM_QuotientRule_Evaluation, double);

// Combined Rules Benchmark
template <typename T>
static void BM_CombinedRules(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.5);
    auto f = std::make_unique<ad::expr::Sin<T>>(x->clone());
    auto g = std::make_unique<ad::expr::Exp<T>>(x->clone());
    
    // Build: d/dx [sin(x)*exp(x)/x^2]
    auto numerator = ad::rules::product_rule(
        f->clone(), f->differentiate("x"),
        g->clone(), g->differentiate("x")
    );
    auto denominator = std::make_unique<ad::expr::Multiplication<T>>(x->clone(), x->clone());
    auto expr = ad::rules::quotient_rule(
        std::move(numerator), numerator->differentiate("x"),
        denominator->clone(), denominator->differentiate("x")
    );
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(expr->evaluate());
    }
}
BENCHMARK_TEMPLATE(BM_CombinedRules, double);

BENCHMARK_MAIN();
