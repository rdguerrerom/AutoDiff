#include "../AutoDiff/control_flow.h"
#include "../AutoDiff/computational_graph.h"
#include <gtest/gtest.h>

namespace ad {
namespace graph {

TEST(ControlFlowTest, SimpleLoopDifferentiation) {
    auto counter = std::make_shared<VariableNode<double>>("counter", 0.0);
    
    auto loop = std::make_shared<LoopNode<double>>(
        counter,
        [](const double& val) { return val < 3.0; },
        [](auto node) { 
            return add<double>(node, std::make_shared<ConstantNode<double>>(1.0)); 
        }
    );

    loop->forward();
    loop->backward(1.0);
    EXPECT_DOUBLE_EQ(counter->get_value(), 3.0);
    EXPECT_DOUBLE_EQ(counter->get_gradient(), 3.0);
}

TEST(ControlFlowTest, ConditionalBranchGradient) {
    auto x = std::make_shared<VariableNode<double>>("x", 1.0);
    auto true_branch = multiply<double>(x, x);
    auto false_branch = add<double>(x, x);
    auto cond = std::make_shared<ConditionalNode<double>>(
        x, true_branch, false_branch, 0.1
    );

    cond->forward();
    cond->backward(1.0);
    double blend = 1.0 / (1.0 + std::exp(-1.0 / 0.1));
    EXPECT_DOUBLE_EQ(x->get_gradient(), 2.0 * blend * 1.0 + 2.0 * (1 - blend));
}

} // namespace graph
} // namespace ad
