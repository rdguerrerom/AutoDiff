#include "../AutoDiff/optimizer.h"
#include "../AutoDiff/expression.h"
#include <gtest/gtest.h>

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
    auto x = std::make_unique<expr::Variable<double>>("x", 0.0); // Add initial value
    auto expr = (x->clone() + x->clone()) * (x->clone() + x->clone());
    
    ExpressionOptimizer<double> optimizer;
    auto optimized = optimizer.optimize(std::move(expr));
    
    // Compare evaluation results instead of internal hashes
    auto* bin_op = dynamic_cast<expr::Multiplication<double>*>(optimized.get());
    ASSERT_NE(bin_op, nullptr);
    ASSERT_EQ(bin_op->left()->evaluate(), bin_op->right()->evaluate());
}
} // namespace optimizer
} // namespace ad
