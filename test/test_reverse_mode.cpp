// test_reverse_mode.cpp
#include "../AutoDiff/reverse_mode.h"
#include "../AutoDiff/computational_graph.h"
#include <gtest/gtest.h>

namespace ad {
namespace reverse {

TEST(ReverseModeTest, SingleVariableGradient) {
    ReverseMode<double> rm;
    auto x = rm.add_variable("x", 4.0);
    auto y = graph::multiply<double>(
        graph::exp<double>(x), 
        x
    );
    
    auto gradients = rm.compute_gradients(y);
    EXPECT_DOUBLE_EQ(gradients["x"], (4.0 + 1.0) * std::exp(4.0));
}

} // namespace reverse
} // namespace ad
