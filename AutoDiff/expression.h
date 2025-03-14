// expression.h
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <cmath>
#include <stdexcept>
#include <unordered_map>  // Added missing header

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

// Base expression class
template <typename T>
class Expression {
public:
    virtual ~Expression() = default;
    virtual T evaluate() const = 0;
    virtual std::unique_ptr<Expression<T>> differentiate(const std::string& variable) const = 0;
    virtual std::unique_ptr<Expression<T>> clone() const = 0;
};

template <typename T>
using ExprPtr = std::unique_ptr<Expression<T>>;

// Constant expression
template <typename T>
class Constant : public Expression<T> {
public:
    explicit Constant(T value) : value_(value) {}

    T evaluate() const override { return value_; }

    ExprPtr<T> differentiate(const std::string&) const override {
        return std::make_unique<Constant<T>>(T(0));
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Constant<T>>(value_);
    }

private:
    T value_;
};

// Variable expression
template <typename T>
class Variable : public Expression<T> {
public:
    // Public constructor with mandatory initial value
    explicit Variable(std::string name, T initial_value)
        : name_(std::move(name)) {
        set_value(initial_value);
    }

    const std::string& name() const { return name_; }

    void set_value(T value) {
        Variable<T>::value_map()[name_] = value;
    }

    T evaluate() const override {
        auto& vm = Variable<T>::value_map();
        auto it = vm.find(name_);
        return it != vm.end() ? it->second : T(0);
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return std::make_unique<Constant<T>>(variable == name_ ? T(1) : T(0));
    }

    ExprPtr<T> clone() const override {
        // Use name() accessor instead of copy constructor
        return std::make_unique<Variable<T>>(name(), evaluate());
    }

private:
    std::string name_;
    
    static std::unordered_map<std::string, T>& value_map() {
        static std::unordered_map<std::string, T> map;
        return map;
    }
};

// Binary operation base class
template <typename T>
class BinaryOperation : public Expression<T> {
public:
    BinaryOperation(ExprPtr<T> left, ExprPtr<T> right)
        : left_(std::move(left)), right_(std::move(right)) {}

protected:
    ExprPtr<T> left_;
    ExprPtr<T> right_;
};

// Addition operation
template <typename T>
class Addition : public BinaryOperation<T> {
public:
    using BinaryOperation<T>::BinaryOperation;

    T evaluate() const override {
        return this->left_->evaluate() + this->right_->evaluate();
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return std::make_unique<Addition<T>>(
            this->left_->differentiate(variable),
            this->right_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Addition<T>>(
            this->left_->clone(),
            this->right_->clone()
        );
    }
};

// Subtraction operation
template <typename T>
class Subtraction : public BinaryOperation<T> {
public:
    using BinaryOperation<T>::BinaryOperation;

    T evaluate() const override {
        return this->left_->evaluate() - this->right_->evaluate();
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return std::make_unique<Subtraction<T>>(
            this->left_->differentiate(variable),
            this->right_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Subtraction<T>>(
            this->left_->clone(),
            this->right_->clone()
        );
    }
};

// Multiplication operation
template <typename T>
class Multiplication : public BinaryOperation<T> {
public:
    using BinaryOperation<T>::BinaryOperation;

    T evaluate() const override {
        return this->left_->evaluate() * this->right_->evaluate();
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        auto dleft = this->left_->differentiate(variable);
        auto dright = this->right_->differentiate(variable);

        auto term1 = std::make_unique<Multiplication<T>>(
            std::move(dleft),
            this->right_->clone()
        );
        auto term2 = std::make_unique<Multiplication<T>>(
            this->left_->clone(),
            std::move(dright)
        );

        return std::make_unique<Addition<T>>(std::move(term1), std::move(term2));
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Multiplication<T>>(
            this->left_->clone(),
            this->right_->clone()
        );
    }
};

// Division operation
template <typename T>
class Division : public BinaryOperation<T> {
public:
    using BinaryOperation<T>::BinaryOperation;

    T evaluate() const override {
        return this->left_->evaluate() / this->right_->evaluate();
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        auto f = this->left_->clone();
        auto g = this->right_->clone();
        auto df = this->left_->differentiate(variable);
        auto dg = this->right_->differentiate(variable);

        auto numerator = std::make_unique<Subtraction<T>>(
            std::make_unique<Multiplication<T>>(std::move(df), g->clone()),
            std::make_unique<Multiplication<T>>(f->clone(), std::move(dg))
        );

        auto denominator = std::make_unique<Pow<T>>(
            g->clone(),
            std::make_unique<Constant<T>>(2)
        );

        return std::make_unique<Division<T>>(
            std::move(numerator),
            std::move(denominator)
        );
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Division<T>>(
            this->left_->clone(),
            this->right_->clone()
        );
    }
};

// Power operation
template <typename T>
class Pow : public BinaryOperation<T> {
public:
    using BinaryOperation<T>::BinaryOperation;

    T evaluate() const override {
        return std::pow(this->left_->evaluate(), this->right_->evaluate());
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        auto base = this->left_->clone();
        auto exponent = this->right_->clone();
        
        auto dBase = this->left_->differentiate(variable);
        auto dExponent = this->right_->differentiate(variable);

        // Term1: exponent * base^(exponent - 1) * dBase
        auto term1 = std::make_unique<Multiplication<T>>(
            exponent->clone(),
            std::make_unique<Multiplication<T>>(
                std::make_unique<Pow<T>>(
                    base->clone(),
                    std::make_unique<Subtraction<T>>(  // Fixed parenthesis here
                        exponent->clone(), 
                        std::make_unique<Constant<T>>(1)
                    )  // Added closing ) for Subtraction
                ),
                std::move(dBase)
            )
        );

        // Term2: base^exponent * ln(base) * dExponent
        auto term2 = std::make_unique<Multiplication<T>>(
            std::make_unique<Multiplication<T>>(
                this->clone(),
                std::make_unique<Log<T>>(base->clone())
            ),
            std::move(dExponent)
        );

        return std::make_unique<Addition<T>>(std::move(term1), std::move(term2));
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Pow<T>>(
            this->left_->clone(),
            this->right_->clone()
        );
    }
};

// Unary operation base class
template <typename T>
class UnaryOperation : public Expression<T> {
public:
    explicit UnaryOperation(ExprPtr<T> operand)
        : operand_(std::move(operand)) {}

protected:
    ExprPtr<T> operand_;
};

// Sign operation
template <typename T>
class Sign : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        T val = this->operand_->evaluate();
        return (T(0) < val) - (val < T(0));
    }

    ExprPtr<T> differentiate(const std::string&) const override {
        return std::make_unique<Constant<T>>(0);
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Sign<T>>(this->operand_->clone());
    }
};

// Operator overloads
template <typename T>
ExprPtr<T> operator-(ExprPtr<T> a, ExprPtr<T> b) {
    return std::make_unique<Subtraction<T>>(std::move(a), std::move(b));
}

template <typename T>
ExprPtr<T> operator/(ExprPtr<T> a, ExprPtr<T> b) {
    return std::make_unique<Division<T>>(std::move(a), std::move(b));
}

} // namespace expr
} // namespace ad
