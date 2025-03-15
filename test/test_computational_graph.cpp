#include "../AutoDiff/computational_graph.h"
#include <gtest/gtest.h>

namespace ad {
namespace graph {

TEST(ComputationalGraphTest, BasicForwardPass) {
    auto var = std::make_shared<VariableNode<double>>("x", 2.0);
    auto op = add<double>(multiply<double>(var, var), var);
    EXPECT_DOUBLE_EQ(op->forward(), 6.0);
}

TEST(ComputationalGraphTest, BackwardGradientAccumulation) {
    auto x = std::make_shared<VariableNode<double>>("x", 3.0);
    auto y = multiply<double>(x, x);
    y->forward();
    y->backward(1.0);
    EXPECT_DOUBLE_EQ(x->get_gradient(), 6.0);
}

TEST(ComputationalGraphTest, MultiDependentNodes) {
    auto a = std::make_shared<VariableNode<double>>("a", 2.0);
    auto b = std::make_shared<VariableNode<double>>("b", 3.0);
    auto c = add<double>(a, b);
    auto d = multiply<double>(c, a);
    d->forward();
    d->backward(1.0);
    EXPECT_DOUBLE_EQ(a->get_gradient(), 7.0);
    EXPECT_DOUBLE_EQ(b->get_gradient(), 2.0);
}

TEST(ComputationalGraphTest, SubtractionGradient) {
    auto a = std::make_shared<VariableNode<double>>("a", 5.0);
    auto b = std::make_shared<VariableNode<double>>("b", 3.0);
    auto sub = subtract<double>(a, b);
    sub->forward();
    sub->backward(1.0);
    EXPECT_DOUBLE_EQ(a->get_gradient(), 1.0);
    EXPECT_DOUBLE_EQ(b->get_gradient(), -1.0);
}

TEST(ComputationalGraphTest, DivisionGradient) {
    auto x = std::make_shared<VariableNode<double>>("x", 6.0);
    auto y = std::make_shared<VariableNode<double>>("y", 2.0);
    auto div = divide<double>(x, y);
    div->forward();
    div->backward(1.0);
    EXPECT_DOUBLE_EQ(x->get_gradient(), 0.5);
    EXPECT_DOUBLE_EQ(y->get_gradient(), -1.5);
}

} // namespace graph
} // namespace ad
