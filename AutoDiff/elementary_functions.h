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

// Forward declarations
template <typename T> class Expression;
template <typename T> class Constant;
template <typename T> class Variable;
template <typename T> class Addition;
template <typename T> class Subtraction;
template <typename T> class Multiplication;
template <typename T> class Division;
template <typename T> class Pow;
template <typename T> class UnaryOperation;
template <typename T> class Sign;

// Elementary function forward declarations
template <typename T> class Sin;
template <typename T> class Cos;
template <typename T> class Tan;
template <typename T> class Exp;
template <typename T> class Log;
template <typename T> class Sqrt;
template <typename T> class Reciprocal;
template <typename T> class Erf;
template <typename T> class Erfc;
template <typename T> class Tgamma;
template <typename T> class Lgamma;
template <typename T> class Sinh;
template <typename T> class Cosh;
template <typename T> class Tanh;
template <typename T> class Asinh;
template <typename T> class Acosh;
template <typename T> class Atanh;


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

    std::string func_name() const override { return "exp"; }
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

    std::string func_name() const override { return "log"; }
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

    std::string func_name() const override { return "sqrt"; }
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

    std::string to_string() const override {
        return "1/(" + this->operand_->to_string() + ")";
    }
    // Override func_name if needed, though to_string is directly implemented
    std::string func_name() const override { return ""; } // Not used

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

  std::string func_name() const override { return "sin"; }

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

  std::string func_name() const override { return "cos"; }

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

    std::string func_name() const override { return "tan"; }
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

    std::string func_name() const override { return "sinh"; }
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

    std::string func_name() const override { return "cosh"; }
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

    std::string func_name() const override { return "tanh"; }
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

    std::string func_name() const override { return "asinh"; }
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

    std::string func_name() const override { return "acosh"; }
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

    std::string func_name() const override { return "atanh"; }
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

    std::string func_name() const override { return "erf"; }
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

    std::string func_name() const override { return "erfc"; }

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

    std::string func_name() const override { return "digamma"; }
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

    std::string func_name() const override { return "tgamma"; }
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

    std::string func_name() const override { return "lgamma"; }
};

// ==================== ELEMENTARY FUNCTION FACTORIES ====================
template <typename T>
ExprPtr<T> sin(ExprPtr<T> operand) {
    return std::make_unique<Sin<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> cos(ExprPtr<T> operand) {
    return std::make_unique<Cos<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> tan(ExprPtr<T> operand) {
    return std::make_unique<Tan<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> exp(ExprPtr<T> operand) {
    return std::make_unique<Exp<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> log(ExprPtr<T> operand) {
    return std::make_unique<Log<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> sqrt(ExprPtr<T> operand) {
    return std::make_unique<Sqrt<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> reciprocal(ExprPtr<T> operand) {
    return std::make_unique<Reciprocal<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> erf(ExprPtr<T> operand) {
    return std::make_unique<Erf<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> erfc(ExprPtr<T> operand) {
    return std::make_unique<Erfc<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> tgamma(ExprPtr<T> operand) {
    return std::make_unique<Tgamma<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> lgamma(ExprPtr<T> operand) {
    return std::make_unique<Lgamma<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> sinh(ExprPtr<T> operand) {
    return std::make_unique<Sinh<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> cosh(ExprPtr<T> operand) {
    return std::make_unique<Cosh<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> tanh(ExprPtr<T> operand) {
    return std::make_unique<Tanh<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> asinh(ExprPtr<T> operand) {
    return std::make_unique<Asinh<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> acosh(ExprPtr<T> operand) {
    return std::make_unique<Acosh<T>>(std::move(operand));
}

template <typename T>
ExprPtr<T> atanh(ExprPtr<T> operand) {
    return std::make_unique<Atanh<T>>(std::move(operand));
}

} // namespace expr
} // namespace ad
