/**
 * @file elementary_functions.h
 * @brief Elementary function implementations and their derivatives
 */

#pragma once
#include "expression.h"
#include <cmath>
#include <stdexcept>

namespace ad {
namespace expr {

// ==================== BASIC MATH FUNCTIONS ====================
template <typename T>
class Exp : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::exp(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return std::make_unique<Multiplication<T>>(
            this->clone(),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Exp<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Exp<T>>(std::move(new_operand));
    }
};

template <typename T>
class Log : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::log(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Reciprocal<T>>(this->operand_->clone()),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Log<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Log<T>>(std::move(new_operand));
    }
};

template <typename T>
class Sqrt : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::sqrt(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Constant<T>>(0.5),
            std::make_unique<Multiplication<T>>(
                std::make_unique<Reciprocal<T>>(this->clone()),
                this->operand_->differentiate(variable)
            )
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Sqrt<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Sqrt<T>>(std::move(new_operand));
    }
};

template <typename T>
class Reciprocal : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return T(1) / this->operand_->evaluate();
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Constant<T>>(-1),
            std::make_unique<Multiplication<T>>(
                std::make_unique<Pow<T>>(this->clone(), std::make_unique<Constant<T>>(2)),
                this->operand_->differentiate(variable)
            )
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Reciprocal<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Reciprocal<T>>(std::move(new_operand));
    }
};

// ==================== TRIGONOMETRIC FUNCTIONS ====================
template <typename T>
class Sin : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::sin(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Cos<T>>(this->operand_->clone()),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Sin<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Sin<T>>(std::move(new_operand));
    }
};

template <typename T>
class Cos : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::cos(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Constant<T>>(-1),
            std::make_unique<Multiplication<T>>(
                std::make_unique<Sin<T>>(this->operand_->clone()),
                this->operand_->differentiate(variable)
            )
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Cos<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Cos<T>>(std::move(new_operand));
    }
};

template <typename T>
class Tan : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::tan(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        auto sec_sq = std::make_unique<Addition<T>>(
            std::make_unique<Constant<T>>(1),
            std::make_unique<Pow<T>>(this->clone(), std::make_unique<Constant<T>>(2))
        );
        return std::make_unique<Multiplication<T>>(
            std::move(sec_sq),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Tan<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Tan<T>>(std::move(new_operand));
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
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Cosh<T>>(this->operand_->clone()),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Sinh<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Sinh<T>>(std::move(new_operand));
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
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Sinh<T>>(this->operand_->clone()),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Cosh<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Cosh<T>>(std::move(new_operand));
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
        auto sech_sq = std::make_unique<Subtraction<T>>(
            std::make_unique<Constant<T>>(1),
            std::make_unique<Pow<T>>(this->clone(), std::make_unique<Constant<T>>(2))
        );
        return std::make_unique<Multiplication<T>>(
            std::move(sech_sq),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Tanh<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Tanh<T>>(std::move(new_operand));
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
        auto denominator = std::make_unique<Sqrt<T>>(
            std::make_unique<Addition<T>>(
                std::make_unique<Pow<T>>(this->operand_->clone(), std::make_unique<Constant<T>>(2)),
                std::make_unique<Constant<T>>(1)
            )
        );
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Reciprocal<T>>(std::move(denominator)),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Asinh<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Asinh<T>>(std::move(new_operand));
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
        auto denominator = std::make_unique<Sqrt<T>>(
            std::make_unique<Subtraction<T>>(
                std::make_unique<Pow<T>>(this->operand_->clone(), std::make_unique<Constant<T>>(2)),
                std::make_unique<Constant<T>>(1)
            )
        );
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Reciprocal<T>>(std::move(denominator)),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Acosh<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Acosh<T>>(std::move(new_operand));
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
        auto denominator = std::make_unique<Subtraction<T>>(
            std::make_unique<Constant<T>>(1),
            std::make_unique<Pow<T>>(this->operand_->clone(), std::make_unique<Constant<T>>(2))
        );
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Reciprocal<T>>(std::move(denominator)),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Atanh<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Atanh<T>>(std::move(new_operand));
    }
};

// ==================== SPECIAL FUNCTIONS ====================
template <typename T>
class Erf : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::erf(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        const T pi = std::acos(-T(1));
        const T factor = T(2) / std::sqrt(pi);
        
        auto inner_mult = std::make_unique<Multiplication<T>>(
            std::make_unique<Constant<T>>(-1),
            std::make_unique<Pow<T>>(
                this->operand_->clone(),
                std::make_unique<Constant<T>>(2)
            )
        );
        auto exp_term = std::make_unique<Exp<T>>(std::move(inner_mult));
        auto outer_mult = std::make_unique<Multiplication<T>>(
            std::move(exp_term),
            this->operand_->differentiate(variable)
        );
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Constant<T>>(factor),
            std::move(outer_mult)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Erf<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Erf<T>>(std::move(new_operand));
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
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Constant<T>>(-1),
            std::make_unique<Erf<T>>(this->operand_->clone())->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Erfc<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Erfc<T>>(std::move(new_operand));
    }
};

// ==================== GAMMA FUNCTIONS ====================
template <typename T>
class Digamma : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        throw std::runtime_error("Digamma implementation not available");
    }

    ExprPtr<T> differentiate(const std::string&) const override {
        throw std::runtime_error("Digamma differentiation not implemented");
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Digamma<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Digamma<T>>(std::move(new_operand));
    }
};

template <typename T>
class Tgamma : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        return std::tgamma(this->operand_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return std::make_unique<Multiplication<T>>(
            this->clone(),
            std::make_unique<Multiplication<T>>(
                std::make_unique<Digamma<T>>(this->operand_->clone()),
                this->operand_->differentiate(variable)
            )
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Tgamma<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Tgamma<T>>(std::move(new_operand));
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
        return std::make_unique<Multiplication<T>>(
            std::make_unique<Digamma<T>>(this->operand_->clone()),
            this->operand_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Lgamma<T>>(this->operand_->clone());
    }

    ExprPtr<T> clone_with(ExprPtr<T> new_operand) const override {
        return std::make_unique<Lgamma<T>>(std::move(new_operand));
    }
};

} // namespace expr
} // namespace ad
