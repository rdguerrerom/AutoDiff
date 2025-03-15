#include "../AutoDiff/custom_function.h"
#include "../AutoDiff/computational_graph.h"
#include <gtest/gtest.h>

namespace ad {
namespace graph {

TEST(CustomFunctionTest, UserDefinedForwardBackward) {
    auto x = std::make_shared<VariableNode<double>>("x", 2.0);
    auto y = std::make_shared<VariableNode<double>>("y", 3.0);

    auto custom = make_custom_function<double>(
        {x, y},
        [](const std::vector<double>& inputs) { 
            return inputs[0]*inputs[0] + inputs[1]*inputs[1]*inputs[1];
        },
        [](const std::vector<double>& inputs, double grad) { 
            return std::vector<double>{2*inputs[0]*grad, 3*inputs[1]*inputs[1]*grad};
        }
    );

    custom->forward();
    custom->backward(1.0);
    EXPECT_DOUBLE_EQ(x->get_gradient(), 4.0);
    EXPECT_DOUBLE_EQ(y->get_gradient(), 27.0);
}

} // namespace graph
} // namespace ad
