// dual_number.h
#pragma once
#include <iostream>
#include <type_traits>

namespace ad {
namespace core {

template <typename T>
class DualNumber {
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type");

public:
    DualNumber(T value, T derivative = T(0))
        : value_(std::move(value)), derivative_(std::move(derivative)) {}

    T value() const { return value_; }
    T derivative() const { return derivative_; }

    DualNumber operator+(const DualNumber& other) const {
        return {value_ + other.value_, derivative_ + other.derivative_};
    }

    DualNumber operator-(const DualNumber& other) const {
        return {value_ - other.value_, derivative_ - other.derivative_};
    }

    DualNumber operator*(const DualNumber& other) const {
        return {value_ * other.value_,
                derivative_ * other.value_ + value_ * other.derivative_};
    }

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
