// test_forward_mode.cpp
#include "../AutoDiff/forward_mode.h"
#include <gtest/gtest.h>
#include <cmath>

namespace ad {
namespace forward {

TEST(ForwardModeTest, SingleVariableGradient) {
    ForwardMode<double> fm;
    
    // Test y = x * exp(x) at x = 4.0
    auto x = fm.variable(4.0);
    auto y = x * exp(x);
    
    // Correct derivative: d/dx (x*e^x) = e^x + x*e^x = e^x(1 + x)
    double expected = (4.0 + 1.0) * std::exp(4.0);
    EXPECT_NEAR(fm.get_derivative(y), expected, 1e-9);
}

TEST(ForwardModeTest, BasicOperations) {
    ForwardMode<double> fm;
    
    // Create variables
    auto x = fm.variable(2.0);
    auto y = fm.variable(3.0);
    
    // Test addition
    auto add = x + y;
    EXPECT_DOUBLE_EQ(add.value, 5.0);
    EXPECT_DOUBLE_EQ(fm.get_derivative(add), 2.0); // dx/dx + dy/dx = 1 + 1 = 2
    
    // Test subtraction
    auto sub = x - y;
    EXPECT_DOUBLE_EQ(sub.value, -1.0);
    EXPECT_DOUBLE_EQ(fm.get_derivative(sub), 0.0); // dx/dx - dy/dx = 1 - 1 = 0
    
    // Test multiplication
    auto mul = x * y;
    EXPECT_DOUBLE_EQ(mul.value, 6.0);
    EXPECT_DOUBLE_EQ(fm.get_derivative(mul), 5.0); // y*dx/dx + x*dy/dx = 3*1 + 2*1 = 5
    
    // Test division
    auto div = x / y;
    EXPECT_DOUBLE_EQ(div.value, 2.0/3.0);
    EXPECT_DOUBLE_EQ(fm.get_derivative(div), 1.0/9.0); // (y*dx/dx - x*dy/dx)/y^2 = (3*1 - 2*1)/9 = 1/9
}

TEST(ForwardModeTest, ElementaryFunctions) {
    ForwardMode<double> fm;
    auto x = fm.variable(1.0);
    
    // Test sin
    auto sin_x = sin(x);
    EXPECT_DOUBLE_EQ(sin_x.value, std::sin(1.0));
    EXPECT_DOUBLE_EQ(fm.get_derivative(sin_x), std::cos(1.0));
    
    // Test cos
    auto cos_x = cos(x);
    EXPECT_DOUBLE_EQ(cos_x.value, std::cos(1.0));
    EXPECT_DOUBLE_EQ(fm.get_derivative(cos_x), -std::sin(1.0));
    
    // Test exp
    auto exp_x = exp(x);
    EXPECT_DOUBLE_EQ(exp_x.value, std::exp(1.0));
    EXPECT_DOUBLE_EQ(fm.get_derivative(exp_x), std::exp(1.0));
    
    // Test log
    auto log_x = log(x);
    EXPECT_DOUBLE_EQ(log_x.value, 0.0);
    EXPECT_DOUBLE_EQ(fm.get_derivative(log_x), 1.0);
    
    // Test sqrt
    auto sqrt_x = sqrt(x);
    EXPECT_DOUBLE_EQ(sqrt_x.value, 1.0);
    EXPECT_DOUBLE_EQ(fm.get_derivative(sqrt_x), 0.5);
    
    // Test tanh
    auto tanh_x = tanh(x);
    EXPECT_DOUBLE_EQ(tanh_x.value, std::tanh(1.0));
    EXPECT_DOUBLE_EQ(fm.get_derivative(tanh_x), 1.0 - std::tanh(1.0) * std::tanh(1.0));
}

TEST(ForwardModeTest, CompositeFunction) {
    ForwardMode<double> fm;
    auto x = fm.variable(2.0);
    
    // Test y = sin(x^2 + exp(x))
    auto inner = x * x + exp(x);
    auto y = sin(inner);
    
    // Compute expected derivative manually
    // d/dx sin(x^2 + exp(x)) = cos(x^2 + exp(x)) * (2x + exp(x))
    double inner_val = 2.0 * 2.0 + std::exp(2.0);
    double expected = std::cos(inner_val) * (2.0 * 2.0 + std::exp(2.0));
    
    EXPECT_NEAR(y.value, std::sin(inner_val), 1e-9);
    EXPECT_NEAR(fm.get_derivative(y), expected, 1e-9);
}

TEST(ForwardModeTest, MultipleVariables) {
    ForwardMode<double> fm;
    
    // Test f(x,y) = x*y + sin(x*y)
    // For partial derivative with respect to x, we set dx/dx = 1, dy/dx = 0
    auto x = fm.variable(0.5);  // With derivative seed 1.0
    auto y = fm.constant(1.5);  // With derivative 0.0 (as dy/dx = 0)
    
    auto f = x * y + sin(x * y);
    
    // Compute expected partial derivative manually
    // ∂f/∂x = y + cos(x*y) * y = y(1 + cos(x*y))
    double expected = 1.5 * (1.0 + std::cos(0.5 * 1.5));
    
    EXPECT_NEAR(f.value, 0.5 * 1.5 + std::sin(0.5 * 1.5), 1e-9);
    EXPECT_NEAR(fm.get_derivative(f), expected, 1e-9);
    
    // Now test partial derivative with respect to y
    x = fm.constant(0.5);  // With derivative 0.0 (as dx/dy = 0)
    y = fm.variable(1.5);  // With derivative seed 1.0
    
    f = x * y + sin(x * y);
    
    // ∂f/∂y = x + cos(x*y) * x = x(1 + cos(x*y))
    expected = 0.5 * (1.0 + std::cos(0.5 * 1.5));
    
    EXPECT_NEAR(fm.get_derivative(f), expected, 1e-9);
}

TEST(ForwardModeTest, ScalarOperations) {
    ForwardMode<double> fm;
    auto x = fm.variable(3.0);
    
    // Test scalar addition
    auto add1 = x + 2.0;
    auto add2 = 2.0 + x;
    EXPECT_DOUBLE_EQ(add1.value, 5.0);
    EXPECT_DOUBLE_EQ(add2.value, 5.0);
    EXPECT_DOUBLE_EQ(fm.get_derivative(add1), 1.0);
    EXPECT_DOUBLE_EQ(fm.get_derivative(add2), 1.0);
    
    // Test scalar multiplication
    auto mul1 = x * 2.0;
    auto mul2 = 2.0 * x;
    EXPECT_DOUBLE_EQ(mul1.value, 6.0);
    EXPECT_DOUBLE_EQ(mul2.value, 6.0);
    EXPECT_DOUBLE_EQ(fm.get_derivative(mul1), 2.0);
    EXPECT_DOUBLE_EQ(fm.get_derivative(mul2), 2.0);
    
    // Test scalar division
    auto div1 = x / 2.0;
    auto div2 = 6.0 / x;
    EXPECT_DOUBLE_EQ(div1.value, 1.5);
    EXPECT_DOUBLE_EQ(div2.value, 2.0);
    EXPECT_DOUBLE_EQ(fm.get_derivative(div1), 0.5);
    EXPECT_DOUBLE_EQ(fm.get_derivative(div2), -6.0 / (3.0 * 3.0)); // d/dx (6/x) = -6/x^2
}

TEST(ForwardModeTest, ChainedOperations) {
    ForwardMode<double> fm;
    auto x = fm.variable(2.0);
    
    // Test a complex chained expression: y = sin(x) * exp(x^2) / (1 + x)
    auto sin_x = sin(x);
    auto x_squared = x * x;
    auto exp_x_squared = exp(x_squared);
    auto one_plus_x = 1.0 + x;
    auto y = sin_x * exp_x_squared / one_plus_x;
    
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
    
    EXPECT_NEAR(y.value, val_y, 1e-9);
    EXPECT_NEAR(fm.get_derivative(y), expected_deriv, 1e-9);
}

} // namespace forward
} // namespace ad
