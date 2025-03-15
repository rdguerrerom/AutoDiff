/**
 * @file optimizer.h
 * @brief Expression optimization techniques including constant propagation
 */

#pragma once
#include "expression.h"
#include <unordered_map>
#include <memory>
#include <string>

namespace ad {
namespace optimizer {

template <typename T>
class ExpressionOptimizer {
public:
    explicit ExpressionOptimizer(std::unordered_map<std::string, T> constants = {})
        : constants_(std::move(constants)) {}

    expr::ExprPtr<T> optimize(expr::ExprPtr<T> expr) {
        expr = propagate_constants(std::move(expr));
        expr = fold_constants(std::move(expr));
        expr = eliminate_common_subexpressions(std::move(expr));
        return expr;
    }

    void set_constants(const std::unordered_map<std::string, T>& constants) {
        constants_ = constants;
    }

private:
    std::unordered_map<std::string, T> constants_;

    expr::ExprPtr<T> propagate_constants(expr::ExprPtr<T> expr) {
        if (auto* var = dynamic_cast<expr::Variable<T>*>(expr.get())) {
            auto it = constants_.find(var->name());
            return it != constants_.end() 
                ? std::make_unique<expr::Constant<T>>(it->second)
                : expr->clone();
        }
        
        if (auto* bin_op = dynamic_cast<expr::BinaryOperation<T>*>(expr.get())) {
            auto left = propagate_constants(bin_op->left()->clone());
            auto right = propagate_constants(bin_op->right()->clone());
            return bin_op->clone_with(std::move(left), std::move(right));
        }
        
        if (auto* unary_op = dynamic_cast<expr::UnaryOperation<T>*>(expr.get())) {
            auto operand = propagate_constants(unary_op->operand()->clone());
            return unary_op->clone_with(std::move(operand));
        }
        
        return expr->clone();
    }

    expr::ExprPtr<T> fold_constants(expr::ExprPtr<T> expr) {
        if (auto* bin_op = dynamic_cast<expr::BinaryOperation<T>*>(expr.get())) {
            auto left = fold_constants(bin_op->left()->clone());
            auto right = fold_constants(bin_op->right()->clone());
            
            if (auto* lc = dynamic_cast<expr::Constant<T>*>(left.get())) {
                if (auto* rc = dynamic_cast<expr::Constant<T>*>(right.get())) {
                    return std::make_unique<expr::Constant<T>>(
                        evaluate_binary(bin_op, lc->evaluate(), rc->evaluate()));
                }
            }
            return bin_op->clone_with(std::move(left), std::move(right));
        }
        return expr->clone();
    }

    expr::ExprPtr<T> eliminate_common_subexpressions(expr::ExprPtr<T> expr) {
        std::unordered_map<std::string, expr::ExprPtr<T>> expr_map;
        return cse(std::move(expr), expr_map);
    }

    T evaluate_binary(expr::BinaryOperation<T>* op, T lhs, T rhs) {
        if (dynamic_cast<expr::Addition<T>*>(op)) return lhs + rhs;
        if (dynamic_cast<expr::Subtraction<T>*>(op)) return lhs - rhs;
        if (dynamic_cast<expr::Multiplication<T>*>(op)) return lhs * rhs;
        if (dynamic_cast<expr::Division<T>*>(op)) return lhs / rhs;
        return T{};
    }

    expr::ExprPtr<T> cse(expr::ExprPtr<T> expr, 
                        std::unordered_map<std::string, expr::ExprPtr<T>>& expr_map) {
        std::string hash = hash_expression(expr.get());
        if (expr_map.count(hash)) return expr_map[hash]->clone();
        expr_map[hash] = expr->clone();
        return expr;
    }

    std::string hash_expression(const expr::Expression<T>* expr) {
        if (auto* c = dynamic_cast<const expr::Constant<T>*>(expr)) {
            return "const:" + std::to_string(c->evaluate());
        }
        if (auto* var = dynamic_cast<const expr::Variable<T>*>(expr)) {
            return "var:" + var->name();
        }
        if (auto* bin_op = dynamic_cast<const expr::BinaryOperation<T>*>(expr)) {
            // Use .get() to convert ExprPtr to raw pointers
            return "binop:" + hash_expression(bin_op->left().get()) + 
                   "," + hash_expression(bin_op->right().get());
        }
        if (auto* unary_op = dynamic_cast<const expr::UnaryOperation<T>*>(expr)) {
            // Use .get() to convert ExprPtr to raw pointer
            return "unop:" + hash_expression(unary_op->operand().get());
        }
        return "unknown";
    }
};

} // namespace optimizer
} // namespace ad
