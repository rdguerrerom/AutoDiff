#include "../AutoDiff/custom_function.h"
#include "../AutoDiff/computational_graph.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace ad {
namespace graph {

TEST(CustomFunctionTest, BasicFunction) {
    // Test a simple function: f(x,y) = x^2 + y^3
    auto x = std::make_shared<VariableNode<double>>("x", 2.0);
    auto y = std::make_shared<VariableNode<double>>("y", 3.0);

    auto custom = make_custom_function<double>(
        {x, y},
        // Forward function
        [](const std::vector<double>& inputs) -> double {
            double result = 0.0;
            if (!inputs.empty()) {
                result += inputs[0] * inputs[0];
            }
            if (inputs.size() > 1) {
                result += inputs[1] * inputs[1] * inputs[1];
            }
            return result;
        },
        // Backward function
        [](const std::vector<double>& inputs, double grad) -> std::vector<double> {
            std::vector<double> gradients;
            if (!inputs.empty()) {
                gradients.push_back(2.0 * inputs[0] * grad);
            }
            if (inputs.size() > 1) {
                gradients.push_back(3.0 * inputs[1] * inputs[1] * grad);
            }
            return gradients;
        }
    );

    custom->forward();
    custom->backward(1.0);
    
    EXPECT_DOUBLE_EQ(x->get_gradient(), 4.0);  // 2 * 2.0 = 4.0
    EXPECT_DOUBLE_EQ(y->get_gradient(), 27.0); // 3 * 3.0^2 = 27.0
}

TEST(CustomFunctionTest, SingleInput) {
    // Test a function with a single input: f(x) = sin(x)
    auto x = std::make_shared<VariableNode<double>>("x", M_PI/4);

    auto custom = make_custom_function<double>(
        {x},
        // Forward function
        [](const std::vector<double>& inputs) -> double {
            if (!inputs.empty()) {
                return std::sin(inputs[0]);
            }
            return 0.0;
        },
        // Backward function
        [](const std::vector<double>& inputs, double grad) -> std::vector<double> {
            std::vector<double> gradients;
            if (!inputs.empty()) {
                gradients.push_back(std::cos(inputs[0]) * grad);
            }
            return gradients;
        }
    );

    custom->forward();
    double val = custom->get_value();
    custom->backward(1.0);
    
    EXPECT_NEAR(val, std::sin(M_PI/4), 1e-10);
    EXPECT_NEAR(x->get_gradient(), std::cos(M_PI/4), 1e-10);
}

TEST(CustomFunctionTest, MultipleInputs) {
    // Test a function with multiple inputs: f(x,y,z) = x*y + y*z + z*x
    auto x = std::make_shared<VariableNode<double>>("x", 2.0);
    auto y = std::make_shared<VariableNode<double>>("y", 3.0);
    auto z = std::make_shared<VariableNode<double>>("z", 4.0);

    auto custom = make_custom_function<double>(
        {x, y, z},
        // Forward function
        [](const std::vector<double>& inputs) -> double {
            double result = 0.0;
            if (inputs.size() >= 3) {
                result = inputs[0] * inputs[1] + inputs[1] * inputs[2] + inputs[2] * inputs[0];
            }
            return result;
        },
        // Backward function
        [](const std::vector<double>& inputs, double grad) -> std::vector<double> {
            std::vector<double> gradients(3, 0.0);
            if (inputs.size() >= 3) {
                // df/dx = y + z
                gradients[0] = (inputs[1] + inputs[2]) * grad;
                // df/dy = x + z
                gradients[1] = (inputs[0] + inputs[2]) * grad;
                // df/dz = x + y
                gradients[2] = (inputs[0] + inputs[1]) * grad;
            }
            return gradients;
        }
    );

    custom->forward();
    double val = custom->get_value();
    custom->backward(1.0);
    
    double expected_val = 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 2.0; // 6 + 12 + 8 = 26
    EXPECT_DOUBLE_EQ(val, expected_val);
    
    EXPECT_DOUBLE_EQ(x->get_gradient(), 3.0 + 4.0); // y + z = 7
    EXPECT_DOUBLE_EQ(y->get_gradient(), 2.0 + 4.0); // x + z = 6
    EXPECT_DOUBLE_EQ(z->get_gradient(), 2.0 + 3.0); // x + y = 5
}

TEST(CustomFunctionTest, Composition) {
    // Test composition with other graph nodes
    auto x = std::make_shared<VariableNode<double>>("x", 2.0);
    
    // Create a custom node for f(x) = x^3
    auto custom = make_custom_function<double>(
        {x},
        // Forward function
        [](const std::vector<double>& inputs) -> double {
            if (!inputs.empty()) {
                return inputs[0] * inputs[0] * inputs[0];
            }
            return 0.0;
        },
        // Backward function
        [](const std::vector<double>& inputs, double grad) -> std::vector<double> {
            std::vector<double> gradients;
            if (!inputs.empty()) {
                gradients.push_back(3.0 * inputs[0] * inputs[0] * grad);
            }
            return gradients;
        }
    );
    
    // Compose with sin(x^3)
    auto sin_custom = graph::sin<double>(custom);
    
    sin_custom->forward();
    double val = sin_custom->get_value();
    sin_custom->backward(1.0);
    
    double x_cubed = 2.0 * 2.0 * 2.0; // 8
    double expected_val = std::sin(x_cubed);
    double expected_grad = std::cos(x_cubed) * 3.0 * 2.0 * 2.0; // cos(8) * 3 * 2^2
    
    EXPECT_NEAR(val, expected_val, 1e-10);
    EXPECT_NEAR(x->get_gradient(), expected_grad, 1e-10);
}

TEST(CustomFunctionTest, GradientAccumulation) {
    // Test gradient accumulation when a node is used multiple times
    auto x = std::make_shared<VariableNode<double>>("x", 3.0);
    
    // Create a custom node for f(x) = x^2
    auto custom1 = make_custom_function<double>(
        {x},
        // Forward function
        [](const std::vector<double>& inputs) -> double {
            if (!inputs.empty()) {
                return inputs[0] * inputs[0];
            }
            return 0.0;
        },
        // Backward function
        [](const std::vector<double>& inputs, double grad) -> std::vector<double> {
            std::vector<double> gradients;
            if (!inputs.empty()) {
                gradients.push_back(2.0 * inputs[0] * grad);
            }
            return gradients;
        }
    );
    
    // Create another custom node for g(x) = sin(x)
    auto custom2 = make_custom_function<double>(
        {x},
        // Forward function
        [](const std::vector<double>& inputs) -> double {
            if (!inputs.empty()) {
                return std::sin(inputs[0]);
            }
            return 0.0;
        },
        // Backward function
        [](const std::vector<double>& inputs, double grad) -> std::vector<double> {
            std::vector<double> gradients;
            if (!inputs.empty()) {
                gradients.push_back(std::cos(inputs[0]) * grad);
            }
            return gradients;
        }
    );
    
    // Combine as h(x) = f(x) + g(x) = x^2 + sin(x)
    auto sum = graph::add<double>(custom1, custom2);
    
    sum->forward();
    double val = sum->get_value();
    sum->backward(1.0);
    
    double expected_val = 3.0 * 3.0 + std::sin(3.0);
    double expected_grad = 2.0 * 3.0 + std::cos(3.0);
    
    EXPECT_NEAR(val, expected_val, 1e-10);
    EXPECT_NEAR(x->get_gradient(), expected_grad, 1e-10);
}

TEST(CustomFunctionTest, NumericalGradientCheck) {
    // Test gradient correctness using numerical differentiation
    
    auto x = std::make_shared<VariableNode<double>>("x", 1.5);
    auto y = std::make_shared<VariableNode<double>>("y", 2.5);
    
    // Complex function: f(x,y) = exp(x*y) / (x + sin(y))
    auto custom = make_custom_function<double>(
        {x, y},
        // Forward function
        [](const std::vector<double>& inputs) -> double {
            if (inputs.size() >= 2) {
                double numerator = std::exp(inputs[0] * inputs[1]);
                double denominator = inputs[0] + std::sin(inputs[1]);
                if (std::abs(denominator) < 1e-10) {
                    return 0.0; // Avoid division by zero
                }
                return numerator / denominator;
            }
            return 0.0;
        },
        // Backward function
        [](const std::vector<double>& inputs, double grad) -> std::vector<double> {
            std::vector<double> gradients(2, 0.0);
            if (inputs.size() >= 2) {
                double x = inputs[0];
                double y = inputs[1];
                double numerator = std::exp(x * y);
                double denominator = x + std::sin(y);
                
                if (std::abs(denominator) < 1e-10) {
                    return gradients; // Avoid division by zero
                }
                
                double f = numerator / denominator;
                
                // df/dx = (exp(xy) * y) / (x + sin(y)) - exp(xy) / (x + sin(y))^2
                gradients[0] = (numerator * y / denominator - f / denominator) * grad;
                
                // df/dy = (exp(xy) * x) / (x + sin(y)) - exp(xy) * cos(y) / (x + sin(y))^2
                gradients[1] = (numerator * x / denominator - f * std::cos(y) / denominator) * grad;
            }
            return gradients;
        }
    );
    
    // Compute analytical gradients
    custom->forward();
    custom->backward(1.0);
    double analytical_dx = x->get_gradient();
    double analytical_dy = y->get_gradient();
    
    // Reset gradients
    x->reset_gradient();
    y->reset_gradient();
    
    // Compute numerical gradients using finite differences
    const double epsilon = 1e-6;
    
    // Compute df/dx numerically
    double original_x = x->get_value();
    x->set_value(original_x + epsilon);
    custom->forward();
    double f_plus_dx = custom->get_value();
    
    x->set_value(original_x - epsilon);
    custom->forward();
    double f_minus_dx = custom->get_value();
    
    double numerical_dx = (f_plus_dx - f_minus_dx) / (2 * epsilon);
    
    // Reset x
    x->set_value(original_x);
    
    // Compute df/dy numerically
    double original_y = y->get_value();
    y->set_value(original_y + epsilon);
    custom->forward();
    double f_plus_dy = custom->get_value();
    
    y->set_value(original_y - epsilon);
    custom->forward();
    double f_minus_dy = custom->get_value();
    
    double numerical_dy = (f_plus_dy - f_minus_dy) / (2 * epsilon);
    
    // Compare analytical vs numerical gradients (they should be close)
    EXPECT_NEAR(analytical_dx, numerical_dx, 1e-5);
    EXPECT_NEAR(analytical_dy, numerical_dy, 1e-5);
}

TEST(CustomFunctionTest, VariableReuse) {
    // Test a function that uses the same variable multiple times: f(x) = x^4
    auto x = std::make_shared<VariableNode<double>>("x", 2.0);

    auto custom = make_custom_function<double>(
        {x, x, x, x}, // Use x four times
        // Forward function
        [](const std::vector<double>& inputs) -> double {
            double result = 1.0;
            for (const auto& input : inputs) {
                result *= input;
            }
            return result;
        },
        // Backward function for f(x,x,x,x) = x^4
        [](const std::vector<double>& inputs, double grad) -> std::vector<double> {
            std::vector<double> gradients;
            if (inputs.size() >= 4) {
                // Each x contributes x^3 to the gradient
                double x_cubed = inputs[0] * inputs[0] * inputs[0];
                for (size_t i = 0; i < inputs.size(); ++i) {
                    gradients.push_back(x_cubed * grad);
                }
            }
            return gradients;
        }
    );

    custom->forward();
    double val = custom->get_value();
    custom->backward(1.0);
    
    double expected_val = std::pow(2.0, 4); // 16
    
    // The gradient accumulates all partial derivatives
    // df/dx = 4x^3 = 4 * 2^3 = 32
    double expected_grad = 4 * std::pow(2.0, 3); 
    
    EXPECT_DOUBLE_EQ(val, expected_val);
    EXPECT_DOUBLE_EQ(x->get_gradient(), expected_grad);
}

TEST(CustomFunctionTest, EmptyInputs) {
    // Test behavior with empty inputs
    auto custom = make_custom_function<double>(
        {},
        // Forward function
        [](const std::vector<double>& /*inputs*/) -> double {
            return 42.0; // Constant function
        },
        // Backward function
        [](const std::vector<double>& /*inputs*/, double /*grad*/) -> std::vector<double> {
            return {}; // No gradients
        }
    );

    custom->forward();
    double val = custom->get_value();
    custom->backward(1.0);
    
    EXPECT_DOUBLE_EQ(val, 42.0);
    // No gradients to check since there are no inputs
}

TEST(CustomFunctionTest, MismatchedGradients) {
    // Test handling of mismatched gradient counts
    auto x = std::make_shared<VariableNode<double>>("x", 2.0);
    auto y = std::make_shared<VariableNode<double>>("y", 3.0);

    auto custom = make_custom_function<double>(
        {x, y},
        // Forward function
        [](const std::vector<double>& inputs) -> double {
            double result = 0.0;
            if (!inputs.empty()) {
                result += inputs[0];
            }
            if (inputs.size() > 1) {
                result += inputs[1];
            }
            return result;
        },
        // Backward function that returns insufficient gradients
        [](const std::vector<double>& inputs, double grad) -> std::vector<double> {
            std::vector<double> gradients;
            if (!inputs.empty()) {
                gradients.push_back(grad); // Only one gradient
            }
            return gradients;
        }
    );

    custom->forward();
    custom->backward(1.0);
    
    // Only the first variable should receive a gradient
    EXPECT_DOUBLE_EQ(x->get_gradient(), 1.0);
    EXPECT_DOUBLE_EQ(y->get_gradient(), 0.0); // No gradient propagated
}

} // namespace graph
} // namespace ad
