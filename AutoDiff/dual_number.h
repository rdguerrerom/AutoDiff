/**
 * @file dual_number.h
 * @brief Dual number implementation for forward-mode automatic differentiation
 */

#pragma once
#include <iostream>
#include <type_traits>

namespace ad {
/**
 * @namespace core
 * @brief Core numerical implementations for automatic differentiation
 */
namespace core {

/**
 * @class DualNumber
 * @brief Represents a dual number for automatic differentiation
 * @tparam T Arithmetic type for value and derivative components (double, float, etc.)
 *
 * Implements dual numbers in the form: value + ε·derivative where ε² = 0
 * Provides basic arithmetic operations with proper derivative propagation
 */
template <typename T>
class DualNumber {
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type");

public:
  /**
   * @brief Constructs a dual number with given value and derivative
   * @param value Primary value component
   * @param derivative Infinitesimal derivative component (default 0)
   */
    DualNumber(T value, T derivative = T(0))
        : value_(std::move(value)), derivative_(std::move(derivative)) {}

      /// @return Primary value component
    T value() const { return value_; }
    
    /// @return Derivative component
    T derivative() const { return derivative_; }

  /**
   * @brief Dual number addition operator
   * @param other Right-hand operand
   * @return New DualNumber with summed components
   */
    DualNumber operator+(const DualNumber& other) const {
        return {value_ + other.value_, derivative_ + other.derivative_};
    }

  /**
   * @brief Dual number substraction operator
   * @param other Right-hand operand
   * @return New DualNumber with the substraction of the components
   */
    DualNumber operator-(const DualNumber& other) const {
        return {value_ - other.value_, derivative_ - other.derivative_};
    }

  /**
   * @brief Dual number multiplication operator
   * @param other Right-hand operand
   * @return New DualNumber with the multiplication of the components
   */
    DualNumber operator*(const DualNumber& other) const {
        return {value_ * other.value_,
                derivative_ * other.value_ + value_ * other.derivative_};
    }

  /**
   * @brief Dual number division operator
   * @param other Right-hand operand
   * @return New DualNumber with the quotient of the components
   */
    DualNumber operator/(const DualNumber& other) const {
        T denominator = other.value_;
        return {value_ / denominator,
                (derivative_ * denominator - value_ * other.derivative_) /
                    (denominator * denominator)};
    }

    DualNumber operator-() const { return {-value_, -derivative_}; }

    DualNumber& operator+=(const DualNumber& other) {
        value_ += other.value_;
        derivative_ += other.derivative_;
        return *this;
    }

    DualNumber& operator-=(const DualNumber& other) {
        value_ -= other.value_;
        derivative_ -= other.derivative_;
        return *this;
    }

    DualNumber& operator*=(const DualNumber& other) {
        *this = *this * other;
        return *this;
    }

    DualNumber& operator/=(const DualNumber& other) {
        *this = *this / other;
        return *this;
    }

    bool operator==(const DualNumber& other) const {
        return value_ == other.value_ && derivative_ == other.derivative_;
    }

    bool operator!=(const DualNumber& other) const { return !(*this == other); }

    /// Stream insertion operator for debugging
    friend std::ostream& operator<<(std::ostream& os, const DualNumber& dn) {
        os << "DualNumber(" << dn.value_ << ", " << dn.derivative_ << ")";
        return os;
    }

private:
    T value_;
    T derivative_;
};

template <typename T>
DualNumber<T> operator+(T scalar, const DualNumber<T>& dn) {
    return DualNumber<T>(scalar) + dn;
}

template <typename T>
DualNumber<T> operator-(T scalar, const DualNumber<T>& dn) {
    return DualNumber<T>(scalar) - dn;
}

template <typename T>
DualNumber<T> operator*(T scalar, const DualNumber<T>& dn) {
    return DualNumber<T>(scalar) * dn;
}

template <typename T>
DualNumber<T> operator/(T scalar, const DualNumber<T>& dn) {
    return DualNumber<T>(scalar) / dn;
}

} // namespace core
} // namespace ad
