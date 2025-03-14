#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../AutoDiff/expression.h"

using namespace ad::expr;
using Catch::Matchers::WithinRel;

TEST_CASE("Basic Variable Operations", "[basic][variable]") {
    auto x = std::make_unique<Variable<double>>("x", 2.0);
    
    SECTION("Evaluation") {
        REQUIRE(x->evaluate() == 2.0);
        x->set_value(3.0);
        REQUIRE(x->evaluate() == 3.0);
    }
    
    SECTION("Differentiation") {
        auto dx = x->differentiate("x");
        auto dy = x->differentiate("y");
        
        REQUIRE(dx->evaluate() == 1.0);
        REQUIRE(dy->evaluate() == 0.0);
    }
}

TEST_CASE("Constant Expressions", "[basic][constant]") {
    auto c = std::make_unique<Constant<double>>(5.0);
    
    SECTION("Evaluation") {
        REQUIRE(c->evaluate() == 5.0);
    }
    
    SECTION("Differentiation") {
        auto dc = c->differentiate("x");
        REQUIRE(dc->evaluate() == 0.0);
    }
}
