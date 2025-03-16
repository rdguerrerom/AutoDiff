#include "../AutoDiff/expression.h"
#include "../AutoDiff/optimizer.h"
#include <iostream>

using namespace ad;

// Helper function to print expression structure
void print_expression(const expr::ExprPtr<double>& expr, const std::string& prefix = "") {
    if (auto* bin = dynamic_cast<expr::BinaryOperation<double>*>(expr.get())) {
        std::cout << prefix << "BinaryOperation(" << typeid(*bin).name() << ")\n";
        print_expression(bin->left(), prefix + "  L:");
        print_expression(bin->right(), prefix + "  R:");
    }
    else if (auto* var = dynamic_cast<expr::Variable<double>*>(expr.get())) {
        std::cout << prefix << "Variable: " << var->name() << "\n";
    }
    else if (auto* c = dynamic_cast<expr::Constant<double>*>(expr.get())) {
        std::cout << prefix << "Constant: " << c->evaluate() << "\n";
    }
}

int main() {
    // Create variables (x will be constant, a/b remain variables)
    auto x = std::make_unique<expr::Variable<double>>("x", 2.0);
    auto a = std::make_unique<expr::Variable<double>>("a", 3.0);  // Initial value for demonstration
    auto b = std::make_unique<expr::Variable<double>>("b", 4.0);

    // Create expression with multiple identical subexpressions
    auto subexpr = x->clone() + a->clone();
    auto expression = 
        (subexpr->clone() * subexpr->clone()) +  // (x+a)^2
        (subexpr->clone() * b->clone()) +        // (x+a)*b
        subexpr->clone();                        // (x+a)

    std::cout << "Original Expression Structure:\n";
    print_expression(expression);
    std::cout << "\nOriginal expression value: " << expression->evaluate() << "\n\n";

    // Set up optimizer with x=2 as known constant
    optimizer::ExpressionOptimizer<double> optimizer({{"x", 2.0}});

    // Optimize the expression
    auto optimized = optimizer.optimize(std::move(expression));

    std::cout << "\nOptimized Expression Structure:\n";
    print_expression(optimized);
    std::cout << "\nOptimized expression value: " << optimized->evaluate() << "\n\n";

    // Show optimization details
    std::cout << "Optimization Metrics:\n";
    std::cout << optimizer.get_metrics_report();

    return 0;
}
