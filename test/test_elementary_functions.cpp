#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"
#include "../AutoDiff/validation.h"

using namespace ad::expr;
using Catch::Matchers::WithinRel;
using namespace ad::test;

TEST_CASE("Elementary Function Coverage", "[functions]") {
    auto x = std::make_unique<Variable<double>>("x", 0.5);
    const double pi = 3.14159265358979323846;

    SECTION("Trigonometric Functions") {
        x->set_value(pi/4);
        
        SECTION("Sin") {
            auto expr = std::make_unique<Sin<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(std::sin(pi/4), 1e-6));
            REQUIRE(validate_derivative(*expr, *x, pi/4));
        }

        SECTION("Cos") {
            auto expr = std::make_unique<Cos<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(std::cos(pi/4), 1e-6));
            REQUIRE(validate_derivative(*expr, *x, pi/4));
        }

        SECTION("Tan") {
            auto expr = std::make_unique<Tan<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(1.0, 1e-6));
            REQUIRE(validate_derivative(*expr, *x, pi/4));
        }
    }

    SECTION("Exponential/Logarithmic") {
        x->set_value(1.0);
        
        SECTION("Exp") {
            auto expr = std::make_unique<Exp<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(std::exp(1.0), 1e-6));
            REQUIRE(validate_derivative(*expr, *x, 1.0));
        }

        SECTION("Log") {
            auto expr = std::make_unique<Log<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(0.0, 1e-6));
            REQUIRE(validate_derivative(*expr, *x, 1.0));
        }
    }

    SECTION("Square Root/Reciprocal") {
        x->set_value(4.0);
        
        SECTION("Sqrt") {
            auto expr = std::make_unique<Sqrt<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(2.0, 1e-6));
            REQUIRE(validate_derivative(*expr, *x, 4.0));
        }

        SECTION("Reciprocal") {
            auto expr = std::make_unique<Reciprocal<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(0.25, 1e-6));
            REQUIRE(validate_derivative(*expr, *x, 4.0));
        }
    }

    SECTION("Error Functions") {
        x->set_value(0.5);
        
        SECTION("Erf") {
            auto expr = std::make_unique<Erf<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(std::erf(0.5), 1e-6));
            REQUIRE(validate_derivative(*expr, *x, 0.5));
        }

        SECTION("Erfc") {
            auto expr = std::make_unique<Erfc<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(std::erfc(0.5), 1e-6));
            REQUIRE(validate_derivative(*expr, *x, 0.5));
        }
    }

    SECTION("Gamma Functions") {
        x->set_value(5.0);
        
        SECTION("Tgamma") {
            auto expr = std::make_unique<Tgamma<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(24.0, 1e-6));
        }

        SECTION("Lgamma") {
            auto expr = std::make_unique<Lgamma<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(std::lgamma(5.0), 1e-6));
        }
    }

    SECTION("Hyperbolic Functions") {
        x->set_value(0.5);
        
        SECTION("Sinh") {
            auto expr = std::make_unique<Sinh<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(std::sinh(0.5), 1e-6));
            REQUIRE(validate_derivative(*expr, *x, 0.5));
        }

        SECTION("Cosh") {
            auto expr = std::make_unique<Cosh<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(std::cosh(0.5), 1e-6));
            REQUIRE(validate_derivative(*expr, *x, 0.5));
        }

        SECTION("Tanh") {
            auto expr = std::make_unique<Tanh<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(std::tanh(0.5), 1e-6));
            REQUIRE(validate_derivative(*expr, *x, 0.5));
        }
    }

    SECTION("Inverse Hyperbolic Functions") {
        SECTION("Asinh") {
            x->set_value(0.0);
            auto expr = std::make_unique<Asinh<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(0.0, 1e-6));
            REQUIRE(validate_derivative(*expr, *x, 0.0));
        }

        SECTION("Acosh") {
            x->set_value(1.5);
            auto expr = std::make_unique<Acosh<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(std::acosh(1.5), 1e-6));
            REQUIRE(validate_derivative(*expr, *x, 1.5));
        }

        SECTION("Atanh") {
            x->set_value(0.5);
            auto expr = std::make_unique<Atanh<double>>(x->clone());
            REQUIRE_THAT(expr->evaluate(), WithinRel(std::atanh(0.5), 1e-6));
            REQUIRE(validate_derivative(*expr, *x, 0.5));
        }
    }

    SECTION("Composite Function Validation") {
        x->set_value(0.25);
        
        SECTION("Nested Functions") {
            auto expr = std::make_unique<Tanh<double>>(
                std::make_unique<Exp<double>>(
                    std::make_unique<Sin<double>>(x->clone())
                )
            );
            REQUIRE(validate_derivative(*expr, *x, 0.25));
        }

        SECTION("Deep Composition") {
            auto expr = std::make_unique<Erf<double>>(
                std::make_unique<Log<double>>(
                    std::make_unique<Cosh<double>>(x->clone())
                )
            );
            REQUIRE(validate_derivative(*expr, *x, 0.5));
        }
    }
}
