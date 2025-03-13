// differentiation_rules.h
#pragma once
#include "expression.h"

namespace ad {
namespace rules {

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

} // namespace rules
} // namespace ad
