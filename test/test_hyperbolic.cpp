#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"

using namespace ad::expr;
using Catch::Matchers::WithinRel;

TEST_CASE("Hyperbolic Functions", "[hyper]") {
    auto x = std::make_unique<Variable<double>>("x", 1.0);
    
    SECTION("Sinh Evaluation") {
        auto expr = std::make_unique<Sinh<double>>(x->clone());
        REQUIRE_THAT(expr->evaluate(), WithinRel(std::sinh(1.0), 1e-6));
    }
    
    SECTION("Cosh Derivative") {
        auto expr = std::make_unique<Cosh<double>>(x->clone());
        auto deriv = expr->differentiate("x");
        x->set_value(0.5);
        REQUIRE_THAT(deriv->evaluate(), WithinRel(std::sinh(0.5), 1e-6));
    }
}
