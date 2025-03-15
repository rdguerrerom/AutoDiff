#include "../AutoDiff/control_flow.h"
#include "../AutoDiff/computational_graph.h"
#include <gtest/gtest.h>

namespace ad {
namespace graph {

TEST(ControlFlowTest, SimpleLoopDifferentiation) {
    auto counter = std::make_shared<VariableNode<double>>("counter", 0.0);
    
    auto loop = make_loop<double>(
        counter,
        [](const double& val) { return val < 3.0; },
        [](auto node) { 
            return add<double>(node, std::make_shared<ConstantNode<double>>(1.0)); 
        }
    );

    loop->forward();
    loop->backward(1.0);
    
    // Updated assertions
    EXPECT_DOUBLE_EQ(loop->get_value(), 3.0);
    EXPECT_DOUBLE_EQ(counter->get_gradient(), 3.0);
}

TEST(ControlFlowTest, ConditionalBranchGradient) {
    auto x = std::make_shared<VariableNode<double>>("x", 1.0);
    auto true_branch = multiply<double>(x, x);  // x² when x=1.0 → 1.0
    auto false_branch = add<double>(x, x);      // x+x when x=1.0 → 2.0
    auto cond = make_conditional<double>(x, true_branch, false_branch, 0.1);

    cond->forward();
    cond->backward(1.0);
    
    double cond_val = x->get_value();  // 1.0
    double smoothing = 0.1;
    double blend = 1.0 / (1.0 + std::exp(-cond_val / smoothing));
    double dblend = blend * (1 - blend) / smoothing;
    
    // Correct gradient calculation
    double expected_grad = 
        (2.0 * 1.0 * blend) +        // true_branch contribution
        (2.0 * (1 - blend)) +        // false_branch contribution
        (1.0 - 2.0) * dblend;       // condition contribution
    
    EXPECT_NEAR(x->get_gradient(), expected_grad, 1e-9);
}

} // namespace graph
} // namespace ad
