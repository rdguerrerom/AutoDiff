#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"  // Add this line

using namespace ad::expr;
using Catch::Matchers::WithinRel;

TEST_CASE("Addition Rule", "[operations][addition]") {
    auto x = std::make_unique<Variable<double>>("x", 2.0);
    auto y = std::make_unique<Variable<double>>("y", 3.0);
    auto expr = std::make_unique<Addition<double>>(x->clone(), y->clone());
    
    REQUIRE(expr->evaluate() == 5.0);
    
    auto dx = expr->differentiate("x");
    auto dy = expr->differentiate("y");
    
    REQUIRE(dx->evaluate() == 1.0);
    REQUIRE(dy->evaluate() == 1.0);
}
TEST_CASE("Product Rule", "[operations][multiplication]") {
    auto x = std::make_unique<Variable<double>>("x", 3.0);
    auto y = std::make_unique<Variable<double>>("y", 2.0); // Corrected name
    auto expr = std::make_unique<Multiplication<double>>(
        std::make_unique<Sin<double>>(x->clone()),
        std::make_unique<Exp<double>>(y->clone())
    );
    
    x->set_value(3.0);
    y->set_value(2.0);
    
    const double expected = std::cos(3.0) * std::exp(2.0); // cos(x)*exp(y)
    auto deriv = expr->differentiate("x");
    
    REQUIRE_THAT(deriv->evaluate(), WithinRel(expected, 1e-6));
}
