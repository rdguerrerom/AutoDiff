#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"
#include "../AutoDiff/validation.h"

using namespace ad::expr;
using Catch::Matchers::WithinRel;

TEST_CASE("Numerical Validation", "[validation]") {
    auto x = std::make_unique<Variable<double>>("x", 0.0);
    
    SECTION("Simple Polynomial") {
        auto poly = std::make_unique<Addition<double>>(
            std::make_unique<Pow<double>>(x->clone(), std::make_unique<Constant<double>>(3)),
            std::make_unique<Multiplication<double>>(
                std::make_unique<Constant<double>>(2.0),
                x->clone()
            )
        );
        
        x->set_value(1.5);
        auto deriv = poly->differentiate("x");
        REQUIRE_THAT(deriv->evaluate(), WithinRel(3*pow(1.5, 2) + 2, 1e-6));
    }
}
