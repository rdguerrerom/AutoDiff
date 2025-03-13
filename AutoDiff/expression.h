// expression.h
#pragma once
#include <memory>
#include <string>
#include <utility>

namespace ad {
namespace expr {

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

template <typename T>
class Variable : public Expression<T> {
public:
    explicit Variable(std::string name, T value = T(0))
        : name_(std::move(name)), value_(value) {}

    T evaluate() const override { return value_; }

    void set_value(T value) { value_ = value; }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return std::make_unique<Constant<T>>(variable == name_ ? T(1) : T(0));
    }

    ExprPtr<T> clone() const override {
        return std::make_unique<Variable>(name_, value_);
    }

private:
    std::string name_;
    T value_;
};

template <typename T>
class Constant : public Expression<T> {
public:
    explicit Constant(T value) : value_(value) {}

    T evaluate() const override { return value_; }

    ExprPtr<T> differentiate(const std::string&) const override {
        return std::make_unique<Constant<T>>(T(0));
    }

    ExprPtr<T> clone() const override { return std::make_unique<Constant>(value_); }

private:
    T value_;
};

// ==================== BINARY OPERATIONS ====================
template <typename T>
class BinaryOperation : public Expression<T> {
public:
    BinaryOperation(ExprPtr<T> left, ExprPtr<T> right)
        : left_(std::move(left)), right_(std::move(right)) {}

protected:
    ExprPtr<T> left_;
    ExprPtr<T> right_;
};

template <typename T>
class Addition : public BinaryOperation<T> {
public:
    using BinaryOperation<T>::BinaryOperation;
    // Existing implementation
};

template <typename T>
class Subtraction : public BinaryOperation<T> {
public:
    using BinaryOperation<T>::BinaryOperation;

    T evaluate() const override {
        return this->left_->evaluate() - this->right_->evaluate();
    }

    ExprPtr<T> differentiate(const std::string& variable) const override {
        return make_expression<T, Subtraction<T>>(
            this->left_->differentiate(variable),
            this->right_->differentiate(variable)
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Subtraction<T>>(
            this->left_->clone(),
            this->right_->clone()
        );
    }
};

template <typename T>
class Multiplication : public BinaryOperation<T> { /* Existing */ };

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

        auto numerator = make_expression<T, Subtraction<T>>(
            make_expression<T, Multiplication<T>>(df, g->clone()),
            make_expression<T, Multiplication<T>>(f->clone(), dg)
        );

        auto denominator = make_expression<T, Pow<T>>(
            g->clone(),
            make_constant<T>(2)
        );

        return make_expression<T, Division<T>>(
            std::move(numerator),
            std::move(denominator)
        );
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Division<T>>(
            this->left_->clone(),
            this->right_->clone()
        );
    }
};

// ==================== UNARY OPERATIONS ====================
template <typename T>
class UnaryOperation : public Expression<T> {
public:
    explicit UnaryOperation(ExprPtr<T> operand)
        : operand_(std::move(operand)) {}

protected:
    ExprPtr<T> operand_;
};

template <typename T>
class Sign : public UnaryOperation<T> {
public:
    using UnaryOperation<T>::UnaryOperation;

    T evaluate() const override {
        T val = this->operand_->evaluate();
        return (T(0) < val) - (val < T(0));  // Returns -1, 0, or 1
    }

    ExprPtr<T> differentiate(const std::string&) const override {
        return make_constant<T>(0);  // Derivative is 0 almost everywhere
    }

    ExprPtr<T> clone() const override {
        return make_expression<T, Sign<T>>(this->operand_->clone());
    }
};

// ==================== FACTORY FUNCTIONS ====================
template <typename T, typename ExprType, typename... Args>
ExprPtr<T> make_expression(Args&&... args) {
    return std::make_unique<ExprType>(std::forward<Args>(args)...);
}

template <typename T>
ExprPtr<T> operator-(ExprPtr<T> a, ExprPtr<T> b) {
    return make_expression<T, Subtraction<T>>(std::move(a), std::move(b));
}

template <typename T>
ExprPtr<T> operator/(ExprPtr<T> a, ExprPtr<T> b) {
    return make_expression<T, Division<T>>(std::move(a), std::move(b));
}


} // namespace expr
} // namespace ad
