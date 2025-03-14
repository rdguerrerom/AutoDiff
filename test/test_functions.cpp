#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../AutoDiff/elementary_functions.h"
#include "../AutoDiff/elementary_functions.h"  // Ensure this is included

using namespace ad::expr;
using Catch::Matchers::WithinRel;

TEST_CASE("Exponential Function", "[functions][exp]") {
    auto x = std::make_unique<Variable<double>>("x", 1.0);
    auto expr = std::make_unique<Exp<double>>(x->clone());
    
    x->set_value(1.0);
    REQUIRE_THAT(expr->evaluate(), WithinRel(std::exp(1.0), 1e-6));
    
    auto deriv = expr->differentiate("x");
    REQUIRE_THAT(deriv->evaluate(), WithinRel(std::exp(1.0), 1e-6));
}

TEST_CASE("Composite Trigonometric Function", "[functions][trig]") {
    auto x = std::make_unique<Variable<double>>("x", M_PI/4);
    auto expr = std::make_unique<Sin<double>>(
        std::make_unique<Multiplication<double>>(
            std::make_unique<Constant<double>>(2.0),
            x->clone()
        )
    );
    
    x->set_value(M_PI/4);
    auto deriv = expr->differentiate("x");
    const double expected = 2.0 * std::cos(2.0 * M_PI/4); // 2*cos(π/2) = 0
    
    REQUIRE_THAT(deriv->evaluate(), WithinRel(expected, 1e-6));
}
