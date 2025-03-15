/**
 * @file forward_mode.h
 * @brief Forward mode automatic differentiation implementation
 */

#pragma once
#include <cmath>
#include <stdexcept>
#include <memory>

namespace ad {
namespace forward {

// ==================== FORWARD DECLARATIONS ====================
template <typename T> class ForwardVar;

// Arithmetic operations
template <typename T> ForwardVar<T> operator+(const ForwardVar<T>& a, const ForwardVar<T>& b);
template <typename T> ForwardVar<T> operator-(const ForwardVar<T>& a, const ForwardVar<T>& b);
template <typename T> ForwardVar<T> operator*(const ForwardVar<T>& a, const ForwardVar<T>& b);
template <typename T> ForwardVar<T> operator/(const ForwardVar<T>& a, const ForwardVar<T>& b);

// Scalar operations
template <typename T> ForwardVar<T> operator+(T a, const ForwardVar<T>& b);
template <typename T> ForwardVar<T> operator-(T a, const ForwardVar<T>& b);
template <typename T> ForwardVar<T> operator*(T a, const ForwardVar<T>& b);
template <typename T> ForwardVar<T> operator/(T a, const ForwardVar<T>& b);
template <typename T> ForwardVar<T> operator+(const ForwardVar<T>& a, T b);
template <typename T> ForwardVar<T> operator-(const ForwardVar<T>& a, T b);
template <typename T> ForwardVar<T> operator*(const ForwardVar<T>& a, T b);
template <typename T> ForwardVar<T> operator/(const ForwardVar<T>& a, T b);

// Elementary functions
template <typename T> ForwardVar<T> sin(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> cos(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> tan(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> exp(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> log(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> sqrt(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> sinh(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> cosh(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> tanh(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> asinh(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> acosh(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> atanh(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> erf(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> tgamma(const ForwardVar<T>& x);
template <typename T> ForwardVar<T> lgamma(const ForwardVar<T>& x);


template <typename T>
class ForwardVar {
public:
    T value;
    T deriv;

    // Constructors
    ForwardVar(T val = T(0), T drv = T(0)) : value(val), deriv(drv) {}
    ForwardVar(const ForwardVar<T>&) = default;
    ForwardVar<T>& operator=(const ForwardVar<T>&) = default;

    // Assignment from scalar
    ForwardVar<T>& operator=(T scalar) {
        value = scalar;
        deriv = T(0);
        return *this;
    }
};

// ==================== OPERATOR OVERLOADS ====================
template <typename T>
ForwardVar<T> operator+(const ForwardVar<T>& a, const ForwardVar<T>& b) {
    return {a.value + b.value, a.deriv + b.deriv};
}

template <typename T>
ForwardVar<T> operator-(const ForwardVar<T>& a, const ForwardVar<T>& b) {
    return {a.value - b.value, a.deriv - b.deriv};
}

template <typename T>
ForwardVar<T> operator*(const ForwardVar<T>& a, const ForwardVar<T>& b) {
    return {
        a.value * b.value,
        a.deriv * b.value + a.value * b.deriv
    };
}

template <typename T>
ForwardVar<T> operator/(const ForwardVar<T>& a, const ForwardVar<T>& b) {
    if (b.value == 0) throw std::runtime_error("Division by zero");
    return {
        a.value / b.value,
        (a.deriv * b.value - a.value * b.deriv) / (b.value * b.value)
    };
}

// Scalar operations
template <typename T>
ForwardVar<T> operator+(T a, const ForwardVar<T>& b) {
    return ForwardVar<T>(a) + b;
}

template <typename T>
ForwardVar<T> operator-(T a, const ForwardVar<T>& b) {
    return ForwardVar<T>(a) - b;
}

template <typename T>
ForwardVar<T> operator*(T a, const ForwardVar<T>& b) {
    return ForwardVar<T>(a) * b;
}

template <typename T>
ForwardVar<T> operator/(T a, const ForwardVar<T>& b) {
    return ForwardVar<T>(a) / b;
}

template <typename T>
ForwardVar<T> operator+(const ForwardVar<T>& a, T b) { return a + ForwardVar<T>(b); }

template <typename T>
ForwardVar<T> operator-(const ForwardVar<T>& a, T b) { return a - ForwardVar<T>(b); }

template <typename T>
ForwardVar<T> operator*(const ForwardVar<T>& a, T b) { return a * ForwardVar<T>(b); }

template <typename T>
ForwardVar<T> operator/(const ForwardVar<T>& a, T b) { return a / ForwardVar<T>(b); }

// ==================== ELEMENTARY FUNCTIONS ====================
template <typename T>
ForwardVar<T> sin(const ForwardVar<T>& x) {
    return {std::sin(x.value), std::cos(x.value) * x.deriv};
}

template <typename T>
ForwardVar<T> cos(const ForwardVar<T>& x) {
    return {std::cos(x.value), -std::sin(x.value) * x.deriv};
}

template <typename T>
ForwardVar<T> tan(const ForwardVar<T>& x) {
    T tan_x = std::tan(x.value);
    return {tan_x, (T(1) + tan_x * tan_x) * x.deriv};
}

template <typename T>
ForwardVar<T> exp(const ForwardVar<T>& x) {
    T ex = std::exp(x.value);
    return {ex, ex * x.deriv};
}

template <typename T>
ForwardVar<T> log(const ForwardVar<T>& x) {
    return {std::log(x.value), x.deriv / x.value};
}

template <typename T>
ForwardVar<T> sqrt(const ForwardVar<T>& x) {
    T sqrt_x = std::sqrt(x.value);
    return {sqrt_x, x.deriv / (T(2) * sqrt_x)};
}

template <typename T>
ForwardVar<T> sinh(const ForwardVar<T>& x) {
    return {std::sinh(x.value), std::cosh(x.value) * x.deriv};
}

template <typename T>
ForwardVar<T> cosh(const ForwardVar<T>& x) {
    return {std::cosh(x.value), std::sinh(x.value) * x.deriv};
}

template <typename T>
ForwardVar<T> tanh(const ForwardVar<T>& x) {
    T tanh_x = std::tanh(x.value);
    return {tanh_x, (T(1) - tanh_x * tanh_x) * x.deriv};
}

template <typename T>
ForwardVar<T> asinh(const ForwardVar<T>& x) {
    return {std::asinh(x.value), x.deriv / std::sqrt(x.value * x.value + T(1))};
}

template <typename T>
ForwardVar<T> acosh(const ForwardVar<T>& x) {
    return {std::acosh(x.value), x.deriv / std::sqrt(x.value * x.value - T(1))};
}

template <typename T>
ForwardVar<T> atanh(const ForwardVar<T>& x) {
    return {std::atanh(x.value), x.deriv / (T(1) - x.value * x.value)};
}

template <typename T>
ForwardVar<T> erf(const ForwardVar<T>& x) {
    const T pi = std::acos(-T(1));
    return {
        std::erf(x.value),
        (T(2) / std::sqrt(pi)) * std::exp(-x.value * x.value) * x.deriv
    };
}


// ==================== FORWARD MODE CONTROLLER ====================
template <typename T>
class ForwardMode {
public:
    // Create independent variable (seed derivative = 1)
    ForwardVar<T> variable(T value) {
        return ForwardVar<T>(value, T(1));
    }

    // Create constant (derivative = 0)
    ForwardVar<T> constant(T value) {
        return ForwardVar<T>(value, T(0));
    }

    // Get derivative from ForwardVar
    T get_derivative(const ForwardVar<T>& var) const {
        return var.deriv;
    }
};

} // namespace forward
} // namespace ad
