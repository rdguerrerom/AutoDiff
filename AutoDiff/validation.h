/**
 * @file validation.h
 * @brief Derivative validation utilities
 */

#pragma once
#include "expression.h"
#include <cmath>
#include <iostream>
#include <functional>

namespace ad {
/**
 * @namespace test
 * @brief Testing and validation utilities
 */
namespace test {

/**
 * @brief Computes central finite difference approximation
 * @tparam T Numeric type
 * @param f Function to differentiate
 * @param x Evaluation point
 * @param h Step size (default 1e-6)
 * @return Numerical derivative approximation
 *
 * @note Uses symmetric difference formula: [f(x+h) - f(x-h)] / (2h)
 */
template <typename T>
T finite_difference(
    std::function<T(T)> f,
    T x,
    T h = static_cast<T>(1e-6)
) {
    return (f(x + h) - f(x - h)) / (2 * h);
}

/**
 * @brief Validates analytical derivative against numerical approximation
 * @tparam T Numeric type
 * @param expr Expression to validate
 * @param var Reference variable for differentiation
 * @param point Evaluation point
 * @param tolerance Acceptable error threshold (default 1e-5)
 * @return true if error < tolerance, false otherwise
 *
 * @note Prints detailed comparison results to stdout
 */
template <typename T>
bool validate_derivative(
    expr::Expression<T>& expr,
    expr::Variable<T>& var,  // Changed to take variable reference
    T point,
    T tolerance = static_cast<T>(1e-5)
) {
    var.set_value(point);
    const T analytical = expr.differentiate(var.name())->evaluate();
    
    auto numerical_fn = [&](T x) {
        var.set_value(x);
        return expr.evaluate();
    };
    
    const T numerical = finite_difference<T>(numerical_fn, point);
    const T error = std::abs(analytical - numerical);

    std::cout << "At x = " << point
              << "\nAnalytical: " << analytical
              << "\nNumerical:  " << numerical
              << "\nError:      " << error << std::endl;

    return error < tolerance;
}
} // namespace test
} // namespace ad
