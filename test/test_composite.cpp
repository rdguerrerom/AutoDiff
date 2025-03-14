#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"

using namespace ad::expr;
using Catch::Matchers::WithinRel;

TEST_CASE("Composite Functions", "[composite]") {
    auto x = std::make_unique<Variable<double>>("x", 0.5);
    
    SECTION("Nested Exponential") {
        auto expr = std::make_unique<Exp<double>>(
            std::make_unique<Sin<double>>(
                std::make_unique<Multiplication<double>>(
                    std::make_unique<Constant<double>>(2.0),
                    x->clone()
                )
            )
        );
        
        const double expected = std::exp(std::sin(1.0));
        x->set_value(0.5);
        REQUIRE_THAT(expr->evaluate(), WithinRel(expected, 1e-6));
    }
    
    SECTION("Deeply Nested Derivative") {
        auto expr = std::make_unique<Log<double>>(
            std::make_unique<Addition<double>>(
                std::make_unique<Constant<double>>(1.0),
                std::make_unique<Tanh<double>>(x->clone())
            )
        );
        
        x->set_value(0.5);
        auto deriv = expr->differentiate("x");
        const double expected = (1 - std::pow(std::tanh(0.5), 2)) / (1 + std::tanh(0.5));
        REQUIRE_THAT(deriv->evaluate(), WithinRel(expected, 1e-6));
    }
}
