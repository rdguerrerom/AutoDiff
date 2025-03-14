#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"
#include "../AutoDiff/validation.h"

using namespace ad::expr;
using Catch::Matchers::WithinRel;
using namespace ad::test;

TEST_CASE("Core Differentiation Rules", "[rules]") {
    auto x = std::make_unique<Variable<double>>("x", 1.5);
    auto y = std::make_unique<Variable<double>>("y", 2.0);

    SECTION("Chain Rule") {
        // Test sin(3x)
        auto expr = std::make_unique<Sin<double>>(
            std::make_unique<Multiplication<double>>(
                std::make_unique<Constant<double>>(3.0),
                x->clone()
            )
        );

        x->set_value(1.5);
        auto deriv = expr->differentiate("x");
        const double expected = 3.0 * std::cos(3 * 1.5);
        
        REQUIRE_THAT(deriv->evaluate(), WithinRel(expected, 1e-6));
        REQUIRE(validate_derivative(*expr, *x, 1.5));
    }

    SECTION("Product Rule") {
        // Test x² * e^x
        auto expr = std::make_unique<Multiplication<double>>(
            std::make_unique<Pow<double>>(x->clone(), std::make_unique<Constant<double>>(2)),
            std::make_unique<Exp<double>>(x->clone())
        );

        x->set_value(1.5);
        auto deriv = expr->differentiate("x");
        const double x_val = 1.5;
        const double expected = (2 * x_val * std::exp(x_val)) + (x_val*x_val * std::exp(x_val));
        
        REQUIRE_THAT(deriv->evaluate(), WithinRel(expected, 1e-6));
        REQUIRE(validate_derivative(*expr, *x, 1.5));
    }

    SECTION("Quotient Rule") {
        // Test (x² + 1)/(3x - 2)
        auto expr = std::make_unique<Division<double>>(
            std::make_unique<Addition<double>>(
                std::make_unique<Pow<double>>(x->clone(), std::make_unique<Constant<double>>(2)),
                std::make_unique<Constant<double>>(1)
            ),
            std::make_unique<Subtraction<double>>(
                std::make_unique<Multiplication<double>>(
                    std::make_unique<Constant<double>>(3),
                    x->clone()
                ),
                std::make_unique<Constant<double>>(2)
            )
        );

        x->set_value(2.0);
        auto deriv = expr->differentiate("x");
        
        const double x_val = 2.0;
        const double num = (2*x_val)*(3*x_val - 2) - (x_val*x_val + 1)*3;
        const double den = std::pow(3*x_val - 2, 2);
        const double expected = num / den;
        
        REQUIRE_THAT(deriv->evaluate(), WithinRel(expected, 1e-6));
        REQUIRE(validate_derivative(*expr, *x, 2.0));
    }

    SECTION("Combined Rules") {
        // Test ln(cos(x²)) * e^(sin(x))
        auto expr = std::make_unique<Multiplication<double>>(
            std::make_unique<Log<double>>(  // Fixed: Single operand for Log
                std::make_unique<Cos<double>>(
                    std::make_unique<Pow<double>>(
                        x->clone(), 
                        std::make_unique<Constant<double>>(2)
                    )
                )
            ),  // Properly closed Log constructor
            std::make_unique<Exp<double>>(  // Second multiplication operand
                std::make_unique<Sin<double>>(x->clone())
            )
        );

        REQUIRE(validate_derivative(*expr, *x, 1.0));
        REQUIRE(validate_derivative(*expr, *x, 0.5));
    }

    SECTION("Multi-variable Product") {
        // Test x³ * y²
        auto expr = std::make_unique<Multiplication<double>>(
            std::make_unique<Pow<double>>(x->clone(), std::make_unique<Constant<double>>(3)),
            std::make_unique<Pow<double>>(y->clone(), std::make_unique<Constant<double>>(2))
        );

        SECTION("Partial derivative w.r.t x") {
            auto dx = expr->differentiate("x");
            const double expected = 3 * std::pow(1.5, 2) * std::pow(2.0, 2);
            REQUIRE_THAT(dx->evaluate(), WithinRel(expected, 1e-6));
        }

        SECTION("Partial derivative w.r.t y") {
            auto dy = expr->differentiate("y");
            const double expected = 2 * std::pow(1.5, 3) * 2.0;
            REQUIRE_THAT(dy->evaluate(), WithinRel(expected, 1e-6));
        }
    }

    SECTION("Edge Cases") {
        SECTION("Division by constant") {
            auto expr = std::make_unique<Division<double>>(
                x->clone(),
                std::make_unique<Constant<double>>(5.0)
            );
            
            auto deriv = expr->differentiate("x");
            REQUIRE_THAT(deriv->evaluate(), WithinRel(0.2, 1e-6));
        }

        SECTION("Triple product") {
            auto expr = std::make_unique<Multiplication<double>>(
                x->clone(),
                std::make_unique<Multiplication<double>>(
                    y->clone(),
                    std::make_unique<Sin<double>>(x->clone())
                )
            );
            
            REQUIRE(validate_derivative(*expr, *x, 1.0));
            REQUIRE(validate_derivative(*expr, *y, 2.0));
        }
    }
}
