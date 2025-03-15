/**
 * @file optimizer.h
 * @brief Expression optimization techniques including constant propagation
 */

#pragma once
#include "expression.h"
#include <unordered_map>
#include <memory>
#include <string>
#include <chrono>
#include <stdexcept>

namespace ad {
namespace optimizer {

struct OptimizationMetrics {
    int constant_propagations = 0;
    int constant_foldings = 0;
    int common_subexpressions_eliminated = 0;
    int algebraic_simplifications = 0;
    int expression_normalizations = 0;
    
    std::chrono::microseconds total_optimization_time{0};
    std::chrono::microseconds propagation_time{0};
    std::chrono::microseconds folding_time{0};
    std::chrono::microseconds cse_time{0};
    std::chrono::microseconds simplification_time{0};
    std::chrono::microseconds normalization_time{0};
    
    void reset() {
        constant_propagations = 0;
        constant_foldings = 0;
        common_subexpressions_eliminated = 0;
        algebraic_simplifications = 0;
        expression_normalizations = 0;
        
        total_optimization_time = std::chrono::microseconds{0};
        propagation_time = std::chrono::microseconds{0};
        folding_time = std::chrono::microseconds{0};
        cse_time = std::chrono::microseconds{0};
        simplification_time = std::chrono::microseconds{0};
        normalization_time = std::chrono::microseconds{0};
    }
    
    std::string report() const {
        std::string result = "Optimization Metrics:\n";
        result += "  Constant propagations: " + std::to_string(constant_propagations) + "\n";
        result += "  Constant foldings: " + std::to_string(constant_foldings) + "\n";
        result += "  Common subexpressions eliminated: " + std::to_string(common_subexpressions_eliminated) + "\n";
        result += "  Algebraic simplifications: " + std::to_string(algebraic_simplifications) + "\n";
        result += "  Expression normalizations: " + std::to_string(expression_normalizations) + "\n";
        result += "  Total optimization time: " + std::to_string(total_optimization_time.count()) + " μs\n";
        result += "    Propagation time: " + std::to_string(propagation_time.count()) + " μs\n";
        result += "    Folding time: " + std::to_string(folding_time.count()) + " μs\n";
        result += "    CSE time: " + std::to_string(cse_time.count()) + " μs\n";
        result += "    Simplification time: " + std::to_string(simplification_time.count()) + " μs\n";
        result += "    Normalization time: " + std::to_string(normalization_time.count()) + " μs\n";
        return result;
    }
};

template <typename T>
class ExpressionOptimizer {
public:
    explicit ExpressionOptimizer(std::unordered_map<std::string, T> constants = {})
        : constants_(std::move(constants)) {}

    expr::ExprPtr<T> optimize(expr::ExprPtr<T> expr) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Reset metrics
        metrics_.reset();
        
        // Apply optimization passes
        auto prop_start = std::chrono::high_resolution_clock::now();
        expr = propagate_constants(std::move(expr));
        metrics_.propagation_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - prop_start);
        
        auto fold_start = std::chrono::high_resolution_clock::now();
        expr = fold_constants(std::move(expr));
        metrics_.folding_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - fold_start);
        
        auto norm_start = std::chrono::high_resolution_clock::now();
        expr = normalize_expressions(std::move(expr));
        metrics_.normalization_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - norm_start);
        
        auto simpl_start = std::chrono::high_resolution_clock::now();
        expr = apply_algebraic_simplifications(std::move(expr));
        metrics_.simplification_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - simpl_start);
        
        auto cse_start = std::chrono::high_resolution_clock::now();
        std::unordered_map<std::string, expr::ExprPtr<T>> expr_map;
        expr = eliminate_common_subexpressions(std::move(expr), expr_map);
        metrics_.cse_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - cse_start);
        
        metrics_.total_optimization_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        
        return expr;
    }

    void set_constants(const std::unordered_map<std::string, T>& constants) {
        constants_ = constants;
    }
    
    const OptimizationMetrics& get_metrics() const {
        return metrics_;
    }
    
    std::string get_metrics_report() const {
        return metrics_.report();
    }

private:
    std::unordered_map<std::string, T> constants_;
    OptimizationMetrics metrics_;

    expr::ExprPtr<T> propagate_constants(expr::ExprPtr<T> expr) {
        if (auto* var = dynamic_cast<expr::Variable<T>*>(expr.get())) {
            auto it = constants_.find(var->name());
            if (it != constants_.end()) {
                metrics_.constant_propagations++;
                return std::make_unique<expr::Constant<T>>(it->second);
            }
            return expr->clone();
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
        // Handle binary operations
        if (auto* bin_op = dynamic_cast<expr::BinaryOperation<T>*>(expr.get())) {
            auto left = fold_constants(bin_op->left()->clone());
            auto right = fold_constants(bin_op->right()->clone());
            
            if (auto* lc = dynamic_cast<expr::Constant<T>*>(left.get())) {
                if (auto* rc = dynamic_cast<expr::Constant<T>*>(right.get())) {
                    metrics_.constant_foldings++;
                    T result = T{0};
                    
                    // Instead of using dynamic_cast to check the specific operation type,
                    // we'll check the class name which is available via typeid
                    // This avoids issues with incomplete types and dynamic_cast
                    if (dynamic_cast<expr::Addition<T>*>(bin_op)) {
                        result = lc->evaluate() + rc->evaluate();
                    } else if (dynamic_cast<expr::Subtraction<T>*>(bin_op)) {
                        result = lc->evaluate() - rc->evaluate();
                    } else if (dynamic_cast<expr::Multiplication<T>*>(bin_op)) {
                        result = lc->evaluate() * rc->evaluate();
                    } else if (dynamic_cast<expr::Division<T>*>(bin_op)) {
                        if (rc->evaluate() == T{0}) {
                            throw std::domain_error("Division by zero");
                        }
                        result = lc->evaluate() / rc->evaluate();
                    } else {
                        // For unknown operations, use the bin_op's evaluate function
                        auto const_bin_op = bin_op->clone_with(left->clone(), right->clone());
                        result = const_bin_op->evaluate();
                    }
                    
                    return std::make_unique<expr::Constant<T>>(result);
                }
            }
            return bin_op->clone_with(std::move(left), std::move(right));
        }
        
        // Handle unary operations
        if (auto* unary_op = dynamic_cast<expr::UnaryOperation<T>*>(expr.get())) {
            auto operand = fold_constants(unary_op->operand()->clone());
            
            if (auto* c = dynamic_cast<expr::Constant<T>*>(operand.get())) {
                metrics_.constant_foldings++;
                
                // Directly use the unary operation's evaluate function
                // This avoids issues with incomplete types
                auto const_unary_op = unary_op->clone_with(operand->clone());
                T result = const_unary_op->evaluate();
                
                return std::make_unique<expr::Constant<T>>(result);
            }
            return unary_op->clone_with(std::move(operand));
        }
        
        return expr->clone();
    }

    expr::ExprPtr<T> eliminate_common_subexpressions(expr::ExprPtr<T> expr, 
                std::unordered_map<std::string, expr::ExprPtr<T>>& expr_map) {
        // Check if this expression is already in the map
        std::string hash = hash_expression(expr.get());
        if (expr_map.count(hash)) {
            metrics_.common_subexpressions_eliminated++;
            return expr_map[hash]->clone();
        }
        
        // Recursively optimize subexpressions
        if (auto* bin_op = dynamic_cast<expr::BinaryOperation<T>*>(expr.get())) {
            auto left = eliminate_common_subexpressions(bin_op->left()->clone(), expr_map);
            auto right = eliminate_common_subexpressions(bin_op->right()->clone(), expr_map);
            expr = bin_op->clone_with(std::move(left), std::move(right));
        } else if (auto* unary_op = dynamic_cast<expr::UnaryOperation<T>*>(expr.get())) {
            auto operand = eliminate_common_subexpressions(unary_op->operand()->clone(), expr_map);
            expr = unary_op->clone_with(std::move(operand));
        }
        
        // Store optimized expression in the map
        hash = hash_expression(expr.get());
        expr_map[hash] = expr->clone();
        
        return expr;
    }
    
    expr::ExprPtr<T> apply_algebraic_simplifications(expr::ExprPtr<T> expr) {
        if (auto* bin_op = dynamic_cast<expr::BinaryOperation<T>*>(expr.get())) {
            auto left = apply_algebraic_simplifications(bin_op->left()->clone());
            auto right = apply_algebraic_simplifications(bin_op->right()->clone());
            
            // Apply simplification rules for binary operations
            if (dynamic_cast<expr::Addition<T>*>(bin_op)) {
                // Rule: x + 0 = x
                if (auto* rc = dynamic_cast<expr::Constant<T>*>(right.get())) {
                    if (rc->evaluate() == T{0}) {
                        metrics_.algebraic_simplifications++;
                        return left;
                    }
                }
                
                // Rule: 0 + x = x
                if (auto* lc = dynamic_cast<expr::Constant<T>*>(left.get())) {
                    if (lc->evaluate() == T{0}) {
                        metrics_.algebraic_simplifications++;
                        return right;
                    }
                }
            } else if (dynamic_cast<expr::Subtraction<T>*>(bin_op)) {
                // Rule: x - 0 = x
                if (auto* rc = dynamic_cast<expr::Constant<T>*>(right.get())) {
                    if (rc->evaluate() == T{0}) {
                        metrics_.algebraic_simplifications++;
                        return left;
                    }
                }
                
                // Rule: x - x = 0
                if (hash_expression(left.get()) == hash_expression(right.get())) {
                    metrics_.algebraic_simplifications++;
                    return std::make_unique<expr::Constant<T>>(T{0});
                }
            } else if (dynamic_cast<expr::Multiplication<T>*>(bin_op)) {
                // Rule: x * 0 = 0
                if (auto* rc = dynamic_cast<expr::Constant<T>*>(right.get())) {
                    if (rc->evaluate() == T{0}) {
                        metrics_.algebraic_simplifications++;
                        return std::make_unique<expr::Constant<T>>(T{0});
                    }
                    
                    // Rule: x * 1 = x
                    if (rc->evaluate() == T{1}) {
                        metrics_.algebraic_simplifications++;
                        return left;
                    }
                }
                
                // Rule: 0 * x = 0
                if (auto* lc = dynamic_cast<expr::Constant<T>*>(left.get())) {
                    if (lc->evaluate() == T{0}) {
                        metrics_.algebraic_simplifications++;
                        return std::make_unique<expr::Constant<T>>(T{0});
                    }
                    
                    // Rule: 1 * x = x
                    if (lc->evaluate() == T{1}) {
                        metrics_.algebraic_simplifications++;
                        return right;
                    }
                }
            } else if (dynamic_cast<expr::Division<T>*>(bin_op)) {
                // Rule: x / 1 = x
                if (auto* rc = dynamic_cast<expr::Constant<T>*>(right.get())) {
                    if (rc->evaluate() == T{1}) {
                        metrics_.algebraic_simplifications++;
                        return left;
                    }
                }
                
                // Rule: 0 / x = 0 (assuming x != 0)
                if (auto* lc = dynamic_cast<expr::Constant<T>*>(left.get())) {
                    if (lc->evaluate() == T{0}) {
                        metrics_.algebraic_simplifications++;
                        return std::make_unique<expr::Constant<T>>(T{0});
                    }
                }
                
                // Rule: x / x = 1 (assuming x != 0)
                if (hash_expression(left.get()) == hash_expression(right.get())) {
                    metrics_.algebraic_simplifications++;
                    return std::make_unique<expr::Constant<T>>(T{1});
                }
            }
            
            return bin_op->clone_with(std::move(left), std::move(right));
        } else if (auto* unary_op = dynamic_cast<expr::UnaryOperation<T>*>(expr.get())) {
            auto operand = apply_algebraic_simplifications(unary_op->operand()->clone());
            return unary_op->clone_with(std::move(operand));
        }
        
        return expr->clone();
    }
    
    expr::ExprPtr<T> normalize_expressions(expr::ExprPtr<T> expr) {
        // Normalize the expression by reordering operands of commutative operations
        if (auto* bin_op = dynamic_cast<expr::BinaryOperation<T>*>(expr.get())) {
            auto left = normalize_expressions(bin_op->left()->clone());
            auto right = normalize_expressions(bin_op->right()->clone());
            
            // For commutative operations (addition and multiplication)
            // order operands such that constants come first, then variables
            bool is_commutative = dynamic_cast<expr::Addition<T>*>(bin_op) || 
                                 dynamic_cast<expr::Multiplication<T>*>(bin_op);
            
            if (is_commutative && should_swap_operands(left.get(), right.get())) {
                metrics_.expression_normalizations++;
                std::swap(left, right);
            }
            
            return bin_op->clone_with(std::move(left), std::move(right));
        } else if (auto* unary_op = dynamic_cast<expr::UnaryOperation<T>*>(expr.get())) {
            auto operand = normalize_expressions(unary_op->operand()->clone());
            return unary_op->clone_with(std::move(operand));
        }
        
        return expr->clone();
    }

    std::string hash_expression(const expr::Expression<T>* expr) {
        if (auto* c = dynamic_cast<const expr::Constant<T>*>(expr)) {
            return "const:" + std::to_string(c->evaluate());
        }
        
        if (auto* var = dynamic_cast<const expr::Variable<T>*>(expr)) {
            return "var:" + var->name();
        }
        
        if (auto* bin_op = dynamic_cast<const expr::BinaryOperation<T>*>(expr)) {
            std::string op_type;
            
            if (dynamic_cast<const expr::Addition<T>*>(expr)) op_type = "add";
            else if (dynamic_cast<const expr::Subtraction<T>*>(expr)) op_type = "sub";
            else if (dynamic_cast<const expr::Multiplication<T>*>(expr)) op_type = "mul";
            else if (dynamic_cast<const expr::Division<T>*>(expr)) op_type = "div";
            else op_type = "unknown_bin";
            
            return "binop:" + op_type + "(" + 
                   hash_expression(bin_op->left().get()) + 
                   "," + hash_expression(bin_op->right().get()) + ")";
        }
        
        if (auto* unary_op = dynamic_cast<const expr::UnaryOperation<T>*>(expr)) {
            // Simple approach: use the operator's evaluation result as part of the hash
            // This avoids having to know the specific operator types
            return "unop:" + std::to_string(expr->evaluate()) + "(" + 
                   hash_expression(unary_op->operand().get()) + ")";
        }
        
        return "unknown:" + std::to_string(expr->evaluate());
    }
    
    bool should_swap_operands(const expr::Expression<T>* left, const expr::Expression<T>* right) {
        // Priority: Constants come first, then variables, then complex expressions
        bool left_is_const = dynamic_cast<const expr::Constant<T>*>(left) != nullptr;
        bool right_is_const = dynamic_cast<const expr::Constant<T>*>(right) != nullptr;
        
        if (right_is_const && !left_is_const) {
            return true;
        }
        
        bool left_is_var = dynamic_cast<const expr::Variable<T>*>(left) != nullptr;
        bool right_is_var = dynamic_cast<const expr::Variable<T>*>(right) != nullptr;
        
        if (right_is_var && !left_is_var && !left_is_const) {
            return true;
        }
        
        // If both are the same type, use lexicographic ordering of hash
        if ((left_is_const && right_is_const) || (left_is_var && right_is_var)) {
            return hash_expression(left) > hash_expression(right);
        }
        
        return false;
    }
};

} // namespace optimizer
} // namespace ad
