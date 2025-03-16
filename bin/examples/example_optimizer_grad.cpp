#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"
#include "../AutoDiff/optimizer.h"
#include <iostream>
#include <cmath>

using namespace ad;

int main() {
    // Create variables with initial values
    auto x = std::make_unique<expr::Variable<double>>("x", 1.5);
    auto y = std::make_unique<expr::Variable<double>>("y", 0.5);
    auto z = std::make_unique<expr::Variable<double>>("z", 2.0);

    // Build complex multivariate expression
    auto expression = 
        (sin(x->clone() + y->clone()) * 
        exp(z->clone())) /
        (x->clone()*x->clone() + y->clone()*y->clone() + z->clone()) -
        (x->clone() * y->clone()) / z->clone();

    // Set up optimizer with z as constant
    optimizer::ExpressionOptimizer<double> optimizer({{"z", 2.0}});

    // Compute partial derivatives before moving
    auto df_dx = expression->differentiate("x");
    auto df_dy = expression->differentiate("y");

    std::cout << "===== Original Function =====\n";
    std::cout << "Expression: " << expression->to_string() << "\n";
    std::cout << "Value: " << expression->evaluate() << "\n\n";
    
    // Print original derivatives first
    std::cout << "Partial Derivative ∂f/∂x:\n";
    std::cout << "Original: " << df_dx->to_string() << "\n";
    
    std::cout << "Partial Derivative ∂f/∂y:\n";
    std::cout << "Original: " << df_dy->to_string() << "\n\n";

    // Optimize after printing originals
    auto optimized_expr = optimizer.optimize(std::move(expression));
    auto optimized_dfdx = optimizer.optimize(std::move(df_dx));
    auto optimized_dfdy = optimizer.optimize(std::move(df_dy));

    std::cout << "\n===== Optimized Results =====\n";
    std::cout << "Optimized Function: " << optimized_expr->to_string() << "\n";
    std::cout << "Optimized Value: " << optimized_expr->evaluate() << "\n\n";
    
    std::cout << "Optimized ∂f/∂x:\n";
    std::cout << optimized_dfdx->to_string() << "\n";
    std::cout << "Value: " << optimized_dfdx->evaluate() << "\n\n";
    
    std::cout << "Optimized ∂f/∂y:\n";
    std::cout << optimized_dfdy->to_string() << "\n";
    std::cout << "Value: " << optimized_dfdy->evaluate() << "\n\n";

    std::cout << "===== Optimization Metrics =====\n";
    std::cout << optimizer.get_metrics_report();

    return 0;
}
