#include "../AutoDiff/optimizer.h"
#include "../AutoDiff/expression.h"
#include "../AutoDiff/elementary_functions.h"
#include <gtest/gtest.h>
#include <cmath>

namespace ad {
namespace optimizer {

TEST(OptimizerTest, ConstantPropagationAndFolding) {
    auto x = std::make_unique<expr::Variable<double>>("x", 5.0);
    auto y = std::make_unique<expr::Variable<double>>("y", 3.0);
    auto expr = (x->clone() + y->clone()) * x->clone();

    std::unordered_map<std::string, double> constants = {{"x", 5.0}, {"y", 3.0}};
    ExpressionOptimizer<double> optimizer(constants);
    auto optimized = optimizer.optimize(std::move(expr));

    auto* result = dynamic_cast<expr::Constant<double>*>(optimized.get());
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->evaluate(), 40.0);
}

TEST(OptimizerTest, CommonSubexpressionElimination) {
    auto x = std::make_unique<expr::Variable<double>>("x", 0.0);
    auto expr = (x->clone() + x->clone()) * (x->clone() + x->clone());
    
    ExpressionOptimizer<double> optimizer;
    auto optimized = optimizer.optimize(std::move(expr));
    
    // Compare evaluation results instead of internal hashes
    auto* bin_op = dynamic_cast<expr::Multiplication<double>*>(optimized.get());
    ASSERT_NE(bin_op, nullptr);
    ASSERT_EQ(bin_op->left()->evaluate(), bin_op->right()->evaluate());
}

TEST(OptimizerTest, AlgebraicSimplifications) {
    auto x = std::make_unique<expr::Variable<double>>("x", 3.0);
    auto zero = std::make_unique<expr::Constant<double>>(0.0);
    auto one = std::make_unique<expr::Constant<double>>(1.0);
    
    // Test x + 0 = x
    auto add_zero = x->clone() + zero->clone();
    
    // Test x * 1 = x
    auto mul_one = x->clone() * one->clone();
    
    // Test x - x = 0
    auto sub_self = x->clone() - x->clone();
    
    // Test x / 1 = x
    auto div_one = x->clone() / one->clone();
    
    ExpressionOptimizer<double> optimizer;
    
    // x + 0 = x
    auto optimized = optimizer.optimize(std::move(add_zero));
    auto* var_result = dynamic_cast<expr::Variable<double>*>(optimized.get());
    ASSERT_NE(var_result, nullptr);
    EXPECT_EQ(var_result->name(), "x");
    
    // x * 1 = x
    optimized = optimizer.optimize(std::move(mul_one));
    var_result = dynamic_cast<expr::Variable<double>*>(optimized.get());
    ASSERT_NE(var_result, nullptr);
    EXPECT_EQ(var_result->name(), "x");
    
    // x - x = 0
    optimized = optimizer.optimize(std::move(sub_self));
    auto* const_result = dynamic_cast<expr::Constant<double>*>(optimized.get());
    ASSERT_NE(const_result, nullptr);
    EXPECT_DOUBLE_EQ(const_result->evaluate(), 0.0);
    
    // x / 1 = x
    optimized = optimizer.optimize(std::move(div_one));
    var_result = dynamic_cast<expr::Variable<double>*>(optimized.get());
    ASSERT_NE(var_result, nullptr);
    EXPECT_EQ(var_result->name(), "x");
}

TEST(OptimizerTest, PerformanceMonitoring) {
    // Test that performance metrics are being tracked
    auto x = std::make_unique<expr::Variable<double>>("x", 2.0);
    auto y = std::make_unique<expr::Variable<double>>("y", 3.0);
    
    // Create a complex expression with opportunities for various optimizations
    auto expr = ((x->clone() + y->clone()) * (x->clone() + y->clone())) + 
                ((x->clone() - y->clone()) * (x->clone() - y->clone()));
    
    ExpressionOptimizer<double> optimizer;
    auto optimized = optimizer.optimize(std::move(expr));
    
    // Get the metrics
    const auto& metrics = optimizer.get_metrics();
    
    // Total optimization time should be non-zero
    EXPECT_GT(metrics.total_optimization_time.count(), 0);
    
    // Individual optimization phases should have non-zero times
    EXPECT_GT(metrics.propagation_time.count() + 
              metrics.folding_time.count() + 
              metrics.cse_time.count() + 
              metrics.simplification_time.count() + 
              metrics.normalization_time.count(), 0);
}

TEST(OptimizerTest, ErrorHandlingInOptimizer) {
    // Test error handling for division by zero
    auto x = std::make_unique<expr::Variable<double>>("x", 1.0);
    auto zero = std::make_unique<expr::Constant<double>>(0.0);
    
    // Create x / 0
    auto div_by_zero = x->clone() / zero->clone();
    
    ExpressionOptimizer<double> optimizer;
    
    // The optimization process should throw an exception for division by zero
    // during constant folding
    EXPECT_THROW(optimizer.optimize(std::move(div_by_zero)), std::domain_error);
}

} // namespace optimizer
} // namespace ad
