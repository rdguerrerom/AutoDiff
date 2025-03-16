#include <benchmark/benchmark.h>
#include "../AutoDiff/elementary_functions.h"
#include "../AutoDiff/differentiation_rules.h"

template <typename T>
static void BM_ChainRule(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 2.0);
    auto inner = std::make_unique<ad::expr::Sin<T>>(x->clone());
    auto outer = std::make_unique<ad::expr::Exp<T>>(inner->clone());

    for (auto _ : state) {
        auto deriv = ad::rules::chain_rule<T>(
            outer->differentiate("x"),
            inner->differentiate("x")
        );
        benchmark::DoNotOptimize(deriv->evaluate());
    }
}
BENCHMARK_TEMPLATE(BM_ChainRule, double);

template <typename T>
static void BM_ProductRule(benchmark::State& state) {
    auto x = std::make_unique<ad::expr::Variable<T>>("x", 1.0);
    auto f = std::make_unique<ad::expr::Sin<T>>(x->clone());
    auto g = std::make_unique<ad::expr::Exp<T>>(x->clone());

    for (auto _ : state) {
        auto deriv = ad::rules::product_rule<T>(
            f->clone(), f->differentiate("x"),
            g->clone(), g->differentiate("x")
        );
        benchmark::DoNotOptimize(deriv->evaluate());
    }
}
BENCHMARK_TEMPLATE(BM_ProductRule, double);
BENCHMARK_MAIN();
