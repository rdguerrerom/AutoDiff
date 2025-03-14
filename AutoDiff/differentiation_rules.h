/**
 * @file differentiation_rules.h
 * @brief Fundamental differentiation rules as expression operations
 */

#pragma once
#include "expression.h"

namespace ad {
/**
 * @namespace rules
 * @brief Contains implementations of fundamental differentiation rules
 */
namespace rules {

/**
 * @brief Applies the chain rule to derivative expressions
 * @tparam T Numeric type
 * @param outer_deriv Derivative of outer function
 * @param inner_deriv Derivative of inner function
 * @return Combined derivative expression
 *
 * @note Implements: d/dx[f(g(x))] = f'(g(x)) * g'(x)
 */
template <typename T>
expr::ExprPtr<T> chain_rule(
    expr::ExprPtr<T> outer_deriv,
    expr::ExprPtr<T> inner_deriv
) {
    return expr::make_expression<T, expr::Multiplication<T>>(
        std::move(outer_deriv),
        std::move(inner_deriv)
    );
}

/**
 * @brief Applies the product rule to derivative expressions
 * @tparam T Numeric type
 * @param f Left operand function
 * @param df Derivative of left operand
 * @param g Right operand function
 * @param dg Derivative of right operand
 * @return Combined derivative expression
 *
 * @note Implements: d/dx[f·g] = f'·g + f·g'
 */
template <typename T>
expr::ExprPtr<T> product_rule(
    expr::ExprPtr<T> f, expr::ExprPtr<T> df,
    expr::ExprPtr<T> g, expr::ExprPtr<T> dg
) {
    return expr::make_expression<T, expr::Addition<T>>(
        expr::make_expression<T, expr::Multiplication<T>>(std::move(df), std::move(g)),
        expr::make_expression<T, expr::Multiplication<T>>(std::move(f), std::move(dg))
    );
}

/**
 * @brief Applies the quotient rule to derivative expressions
 * @tparam T Numeric type
 * @param f Numerator function
 * @param df Derivative of numerator
 * @param g Denominator function
 * @param dg Derivative of denominator
 * @return Combined derivative expression
 *
 * @note Implements: d/dx[f/g] = (f'·g - f·g') / g²
 */
template <typename T>
expr::ExprPtr<T> quotient_rule(
    expr::ExprPtr<T> f, expr::ExprPtr<T> df,
    expr::ExprPtr<T> g, expr::ExprPtr<T> dg
) {
    auto numerator = expr::make_expression<T, expr::Subtraction<T>>(
        expr::make_expression<T, expr::Multiplication<T>>(std::move(df), g->clone()),
        expr::make_expression<T, expr::Multiplication<T>>(f->clone(), std::move(dg))
    );

    auto denominator = expr::make_expression<T, expr::Pow<T>>(
        std::move(g),
        expr::make_expression<T, expr::Constant<T>>(2)
    );

    return expr::make_expression<T, expr::Division<T>>(
        std::move(numerator),
        std::move(denominator)
    );
}

} // namespace rules
} // namespace ad
