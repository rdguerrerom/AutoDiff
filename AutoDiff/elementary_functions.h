// elementary_functions.h
#pragma once
#include "expression.h"
#include <cmath>
#include <limits>

namespace ad {
namespace expr {

// ==================== ERROR FUNCTIONS ====================
template <typename T>
class Erf : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::erf(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        const T sqrt_pi = std::sqrt(std::acos(-T(1)));  // Compute sqrt(π)
        const T factor = T(2) / sqrt_pi;

        auto exponent = make_expression<T, Multiplication<T>>(
            make_constant<T>(-1),
            make_expression<T, Pow<T>>(
                this->operand_->clone(),
                make_constant<T>(2)
            )
        );

        return make_expression<T, Multiplication<T>>(
            make_constant<T>(factor),
            make_expression<T, Multiplication<T>>(
                make_expression<T, Exp<T>>(exponent),
                this->operand_->differentiate(variable)
            )
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Erf<T>>(this->operand_->clone());
    }
};

template <typename T>
class Erfc : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::erfc(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return make_expression<T, Multiplication<T>>(
            make_constant<T>(-1),
            make_expression<T, Erf<T>>(this->operand_->clone())
                ->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Erfc<T>>(this->operand_->clone());
    }
};

// ==================== GAMMA FUNCTIONS ====================
template <typename T>
class Tgamma : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::tgamma(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        // Derivative: tgamma(x) * digamma(x) * dx
        // Note: digamma not in standard library - symbolic representation
        return make_expression<T, Multiplication<T>>(
            this->clone(),
            make_expression<T, Multiplication<T>>(
                make_expression<T, Digamma<T>>(this->operand_->clone()),
                this->operand_->differentiate(variable)
            )
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Tgamma<T>>(this->operand_->clone());
    }
};

template <typename T>
class Lgamma : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::lgamma(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        // Derivative: digamma(x) * dx
        return make_expression<T, Multiplication<T>>(
            make_expression<T, Digamma<T>>(this->operand_->clone()),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Lgamma<T>>(this->operand_->clone());
    }
};

// ==================== ABSOLUTE VALUE ====================
template <typename T>
class Abs : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::abs(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return make_expression<T, Multiplication<T>>(
            make_expression<T, Sign<T>>(this->operand_->clone()),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Abs<T>>(this->operand_->clone());
    }
};

template <typename T>
class Sign : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        T val = this->operand_->evaluate();
        return (T(0) < val) - (val < T(0));
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return make_constant<T>(0);  // Derivative of sign function is 0 almost everywhere
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Sign<T>>(this->operand_->clone());
    }
};

// ==================== HYPERBOLIC FUNCTIONS ====================
template <typename T>
class Sinh : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::sinh(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return make_expression<T, Multiplication<T>>(
            make_expression<T, Cosh<T>>(this->operand_->clone()),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Sinh<T>>(this->operand_->clone());
    }
};

template <typename T>
class Cosh : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::cosh(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return make_expression<T, Multiplication<T>>(
            make_expression<T, Sinh<T>>(this->operand_->clone()),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Cosh<T>>(this->operand_->clone());
    }
};

template <typename T>
class Tanh : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::tanh(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        auto sech_sq = make_expression<T, Subtraction<T>>(
            make_constant<T>(1),
            make_expression<T, Pow<T>>(
                this->clone(),
                make_constant<T>(2)
        );
        return make_expression<T, Multiplication<T>>(
            sech_sq,
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Tanh<T>>(this->operand_->clone());
    }
};

// ==================== INVERSE HYPERBOLIC FUNCTIONS ====================
template <typename T>
class Asinh : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::asinh(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        auto denominator = make_expression<T, Sqrt<T>>(
            make_expression<T, Addition<T>>(
                make_expression<T, Pow<T>>(
                    this->operand_->clone(),
                    make_constant<T>(2)
                ),
                make_constant<T>(1)
            )
        );
        return make_expression<T, Multiplication<T>>(
            make_expression<T, Reciprocal<T>>(denominator),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Asinh<T>>(this->operand_->clone());
    }
};

template <typename T>
class Acosh : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::acosh(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        auto denominator = make_expression<T, Sqrt<T>>(
            make_expression<T, Subtraction<T>>(
                make_expression<T, Pow<T>>(
                    this->operand_->clone(),
                    make_constant<T>(2)
                ),
                make_constant<T>(1)
            )
        );
        return make_expression<T, Multiplication<T>>(
            make_expression<T, Reciprocal<T>>(denominator),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Acosh<T>>(this->operand_->clone());
    }
};

template <typename T>
class Atanh : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::atanh(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        auto denominator = make_expression<T, Subtraction<T>>(
            make_constant<T>(1),
            make_expression<T, Pow<T>>(
                this->operand_->clone(),
                make_constant<T>(2)
            )
        );
        return make_expression<T, Multiplication<T>>(
            make_expression<T, Reciprocal<T>>(denominator),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Atanh<T>>(this->operand_->clone());
    }
};

// ==================== DIGAMMA PLACEHOLDER ====================
template <typename T>
class Digamma : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        // Placeholder implementation - would require external library
        throw std::runtime_error("Digamma implementation not available");
    }

    ExprPtr<T> differentiate(const std::string&) const override {
        throw std::runtime_error("Digamma differentiation not implemented");
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Digamma<T>>(this->operand_->clone());
    }
};

} // namespace expr
} // namespace ad
