#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"

using namespace ad::expr;
using Catch::Matchers::WithinRel;

TEST_CASE("Multi-variable Expressions", "[multi]") {
    auto x = std::make_unique<Variable<double>>("x", 2.0);
    auto y = std::make_unique<Variable<double>>("y", 3.0);
    
    SECTION("Partial Derivatives") {
        auto expr = std::make_unique<Multiplication<double>>(
            std::make_unique<Pow<double>>(x->clone(), std::make_unique<Constant<double>>(2)),
            y->clone()
        );
        
        x->set_value(2.0);  // Explicitly set values
        y->set_value(3.0);
        
        auto dx = expr->differentiate("x");
        auto dy = expr->differentiate("y");
        
        REQUIRE_THAT(dx->evaluate(), WithinRel(2*2*3, 1e-6));  // 12.0
        REQUIRE_THAT(dy->evaluate(), WithinRel(2*2, 1e-6));     // 4.0
    }
}
