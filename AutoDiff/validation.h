// validation.h
#pragma once
#include "expression.h"
#include <cmath>
#include <iostream>
#include <functional>

namespace ad {
namespace test {

template <typename T>
T finite_difference(
    std::function<T(T)> f,
    T x,
    T h = static_cast<T>(1e-6)
) {
    return (f(x + h) - f(x - h)) / (2 * h);
}

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
