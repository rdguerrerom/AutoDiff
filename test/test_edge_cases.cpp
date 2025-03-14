#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"

using namespace ad::expr;
using Catch::Matchers::WithinRel;

TEST_CASE("Edge Case Handling", "[edge]") {
    auto x = std::make_unique<Variable<double>>("x", 0.0);
    
    SECTION("Division Near Zero") {
        auto expr = std::make_unique<Division<double>>(
            std::make_unique<Constant<double>>(1.0),
            std::make_unique<Sin<double>>(x->clone())
        );
        
        x->set_value(1e-10);
        REQUIRE_THAT(expr->evaluate(), WithinRel(1.0/1e-10, 1e-6));
    }
    
    SECTION("Exponential at Zero") {
        auto expr = std::make_unique<Exp<double>>(
            std::make_unique<Constant<double>>(0.0)
        );
        REQUIRE_THAT(expr->evaluate(), WithinRel(1.0, 1e-12));
    }
}
