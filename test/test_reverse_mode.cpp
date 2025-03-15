// test_reverse_mode.cpp
#include "../AutoDiff/reverse_mode.h"
#include "../AutoDiff/computational_graph.h"
#include <gtest/gtest.h>
#include <cmath>

namespace ad {
namespace reverse {

TEST(ReverseModeTest, SingleVariableGradient) {
    ReverseMode<double> rm;
    auto x = rm.add_variable("x", 4.0);
    auto y = graph::multiply<double>(
        graph::exp<double>(x), 
        x
    );
    
    // Ensure forward pass is complete before computing gradients
    y->forward();
    
    auto gradients = rm.compute_gradients(y);
    
    // Correct derivative: d/dx (x*e^x) = e^x + x*e^x
    double expected = (4.0 + 1.0) * std::exp(4.0);
    EXPECT_NEAR(gradients["x"], expected, 1e-9);
}

TEST(ReverseModeTest, BasicOperations) {
    ReverseMode<double> rm;
    
    // Create variables
    auto x = rm.add_variable("x", 2.0);
    auto y = rm.add_variable("y", 3.0);
    
    // Test addition
    auto add = graph::add<double>(x, y);
    add->forward();
    auto add_grads = rm.compute_gradients(add);
    EXPECT_DOUBLE_EQ(add->get_value(), 5.0);
    EXPECT_DOUBLE_EQ(add_grads["x"], 1.0);
    EXPECT_DOUBLE_EQ(add_grads["y"], 1.0);
    
    // Test subtraction
    auto sub = graph::subtract<double>(x, y);
    sub->forward();
    auto sub_grads = rm.compute_gradients(sub);
    EXPECT_DOUBLE_EQ(sub->get_value(), -1.0);
    EXPECT_DOUBLE_EQ(sub_grads["x"], 1.0);
    EXPECT_DOUBLE_EQ(sub_grads["y"], -1.0);
    
    // Test multiplication
    auto mul = graph::multiply<double>(x, y);
    mul->forward();
    auto mul_grads = rm.compute_gradients(mul);
    EXPECT_DOUBLE_EQ(mul->get_value(), 6.0);
    EXPECT_DOUBLE_EQ(mul_grads["x"], 3.0);
    EXPECT_DOUBLE_EQ(mul_grads["y"], 2.0);
    
    // Test division
    auto div = graph::divide<double>(x, y);
    div->forward();
    auto div_grads = rm.compute_gradients(div);
    EXPECT_DOUBLE_EQ(div->get_value(), 2.0/3.0);
    EXPECT_DOUBLE_EQ(div_grads["x"], 1.0/3.0);
    EXPECT_DOUBLE_EQ(div_grads["y"], -2.0/(3.0*3.0));
}

TEST(ReverseModeTest, ElementaryFunctions) {
    ReverseMode<double> rm;
    auto x = rm.add_variable("x", 1.0);
    
    // Test sin
    auto sin_x = graph::sin<double>(x);
    sin_x->forward();
    auto sin_grads = rm.compute_gradients(sin_x);
    EXPECT_DOUBLE_EQ(sin_x->get_value(), std::sin(1.0));
    EXPECT_DOUBLE_EQ(sin_grads["x"], std::cos(1.0));
    
    // Test cos
    auto cos_x = graph::cos<double>(x);
    cos_x->forward();
    auto cos_grads = rm.compute_gradients(cos_x);
    EXPECT_DOUBLE_EQ(cos_x->get_value(), std::cos(1.0));
    EXPECT_DOUBLE_EQ(cos_grads["x"], -std::sin(1.0));
    
    // Test exp
    auto exp_x = graph::exp<double>(x);
    exp_x->forward();
    auto exp_grads = rm.compute_gradients(exp_x);
    EXPECT_DOUBLE_EQ(exp_x->get_value(), std::exp(1.0));
    EXPECT_DOUBLE_EQ(exp_grads["x"], std::exp(1.0));
    
    // Test log
    auto log_x = graph::log<double>(x);
    log_x->forward();
    auto log_grads = rm.compute_gradients(log_x);
    EXPECT_DOUBLE_EQ(log_x->get_value(), 0.0);
    EXPECT_DOUBLE_EQ(log_grads["x"], 1.0);
    
    // Test tanh
    auto tanh_x = graph::tanh<double>(x);
    tanh_x->forward();
    auto tanh_grads = rm.compute_gradients(tanh_x);
    EXPECT_DOUBLE_EQ(tanh_x->get_value(), std::tanh(1.0));
    EXPECT_DOUBLE_EQ(tanh_grads["x"], 1.0 - std::tanh(1.0) * std::tanh(1.0));
}

TEST(ReverseModeTest, CompositeFunction) {
    ReverseMode<double> rm;
    auto x = rm.add_variable("x", 2.0);
    
    // Test y = sin(x^2 + exp(x))
    auto x_squared = graph::multiply<double>(x, x);
    auto exp_x = graph::exp<double>(x);
    auto inner = graph::add<double>(x_squared, exp_x);
    auto y = graph::sin<double>(inner);
    
    y->forward();
    auto gradients = rm.compute_gradients(y);
    
    // Compute expected derivative manually
    // d/dx sin(x^2 + exp(x)) = cos(x^2 + exp(x)) * (2x + exp(x))
    double inner_val = 2.0 * 2.0 + std::exp(2.0);
    double expected = std::cos(inner_val) * (2.0 * 2.0 + std::exp(2.0));
    
    EXPECT_NEAR(y->get_value(), std::sin(inner_val), 1e-9);
    EXPECT_NEAR(gradients["x"], expected, 1e-9);
}

TEST(ReverseModeTest, MultipleVariables) {
    ReverseMode<double> rm;
    
    // Test f(x,y) = x*y + sin(x*y)
    auto x = rm.add_variable("x", 0.5);
    auto y = rm.add_variable("y", 1.5);
    
    auto xy = graph::multiply<double>(x, y);
    auto sin_xy = graph::sin<double>(xy);
    auto f = graph::add<double>(xy, sin_xy);
    
    f->forward();
    auto gradients = rm.compute_gradients(f);
    
    // Compute expected partial derivatives manually
    // ∂f/∂x = y + cos(x*y) * y = y(1 + cos(x*y))
    // ∂f/∂y = x + cos(x*y) * x = x(1 + cos(x*y))
    double expected_dx = 1.5 * (1.0 + std::cos(0.5 * 1.5));
    double expected_dy = 0.5 * (1.0 + std::cos(0.5 * 1.5));
    
    EXPECT_NEAR(f->get_value(), 0.5 * 1.5 + std::sin(0.5 * 1.5), 1e-9);
    EXPECT_NEAR(gradients["x"], expected_dx, 1e-9);
    EXPECT_NEAR(gradients["y"], expected_dy, 1e-9);
}

TEST(ReverseModeTest, GradientAccumulation) {
    ReverseMode<double> rm;
    auto x = rm.add_variable("x", 3.0);
    
    // Create a computational graph where x is used multiple times
    // y = x^2 + sin(x)
    auto x_squared = graph::multiply<double>(x, x);
    auto sin_x = graph::sin<double>(x);
    auto y = graph::add<double>(x_squared, sin_x);
    
    y->forward();
    auto gradients = rm.compute_gradients(y);
    
    // Expected gradient: d/dx (x^2 + sin(x)) = 2x + cos(x)
    double expected = 2.0 * 3.0 + std::cos(3.0);
    
    EXPECT_NEAR(gradients["x"], expected, 1e-9);
}

TEST(ReverseModeTest, SetVariable) {
    ReverseMode<double> rm;
    auto x = rm.add_variable("x", 1.0);
    auto y = graph::exp<double>(x);
    
    y->forward();
    auto gradients1 = rm.compute_gradients(y);
    EXPECT_NEAR(gradients1["x"], std::exp(1.0), 1e-9);
    
    // Change variable value and recompute
    rm.set_variable("x", 2.0);
    y->forward();
    auto gradients2 = rm.compute_gradients(y);
    EXPECT_NEAR(gradients2["x"], std::exp(2.0), 1e-9);
}

TEST(ReverseModeTest, ChainedOperations) {
    ReverseMode<double> rm;
    auto x = rm.add_variable("x", 2.0);
    
    // Test a complex chained expression: y = sin(x) * exp(x^2) / (1 + x)
    auto sin_x = graph::sin<double>(x);
    auto x_squared = graph::multiply<double>(x, x);
    auto exp_x_squared = graph::exp<double>(x_squared);
    auto const_one = std::make_shared<graph::ConstantNode<double>>(1.0);
    auto one_plus_x = graph::add<double>(const_one, x);
    auto numerator = graph::multiply<double>(sin_x, exp_x_squared);
    auto y = graph::divide<double>(numerator, one_plus_x);
    
    y->forward();
    auto gradients = rm.compute_gradients(y);
    
    // Manual computation of the expected value and derivative
    double val_x = 2.0;
    double val_sin_x = std::sin(val_x);
    double val_x_squared = val_x * val_x;
    double val_exp_x_squared = std::exp(val_x_squared);
    double val_one_plus_x = 1.0 + val_x;
    double val_y = val_sin_x * val_exp_x_squared / val_one_plus_x;
    
    // d/dx [sin(x)] = cos(x)
    // d/dx [x^2] = 2x
    // d/dx [exp(x^2)] = exp(x^2) * 2x
    // d/dx [1 + x] = 1
    // Use product and quotient rules for the full derivative
    double deriv_sin_x = std::cos(val_x);
    double deriv_exp_x_squared = val_exp_x_squared * 2.0 * val_x;
    double deriv_one_plus_x = 1.0;
    
    double expected_deriv = 
        (deriv_sin_x * val_exp_x_squared * val_one_plus_x + 
         val_sin_x * deriv_exp_x_squared * val_one_plus_x - 
         val_sin_x * val_exp_x_squared * deriv_one_plus_x) / 
        (val_one_plus_x * val_one_plus_x);
    
    EXPECT_NEAR(y->get_value(), val_y, 1e-9);
    EXPECT_NEAR(gradients["x"], expected_deriv, 1e-9);
}

} // namespace reverse
} // namespace ad
