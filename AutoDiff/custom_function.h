/**
 * @file custom_function.h
 * @brief User-defined function nodes for computational graphs
 */

#pragma once
#include "computational_graph.h"
#include <vector>
#include <functional>
#include <stdexcept>

namespace ad {
namespace graph {

template <typename T>
class CustomFunctionNode : public GraphNode<T> {
public:
    using NodePtr = typename GraphNode<T>::Ptr;
    
    // Function type that takes an array of values and returns a single value
    using ForwardFuncType = std::function<T(const std::vector<T>&)>;
    
    // Function type that takes an array of values and a gradient, returns gradients
    using BackwardFuncType = std::function<std::vector<T>(const std::vector<T>&, T)>;

    CustomFunctionNode(ForwardFuncType forward_func, BackwardFuncType backward_func)
        : forward_func_(std::move(forward_func)),
          backward_func_(std::move(backward_func)) {}

    T forward() override {
        // Collect input values
        std::vector<T> input_values;
        for (const auto& input : this->inputs_) {
            if (input) {
                input_values.push_back(input->forward());
            }
        }
        
        // Apply forward function
        this->value_ = forward_func_(input_values);
        return this->value_;
    }

    void backward(const T& gradient) override {
        // Collect input values
        std::vector<T> input_values;
        for (const auto& input : this->inputs_) {
            if (input) {
                input_values.push_back(input->get_value());
            }
        }
        
        // Compute gradients
        std::vector<T> input_grads = backward_func_(input_values, gradient);
        
        // Apply gradients to inputs, with safety checks
        size_t num_inputs = this->inputs_.size();
        size_t num_grads = input_grads.size();
        
        for (size_t i = 0; i < num_inputs && i < num_grads; ++i) {
            if (this->inputs_[i]) {
                this->inputs_[i]->backward(input_grads[i]);
            }
        }
    }

private:
    ForwardFuncType forward_func_;
    BackwardFuncType backward_func_;
};

// Helper function to create a custom function node
template <typename T>
auto make_custom_function(
    std::vector<typename GraphNode<T>::Ptr> inputs,
    typename CustomFunctionNode<T>::ForwardFuncType forward_func,
    typename CustomFunctionNode<T>::BackwardFuncType backward_func) {
    
    auto node = std::make_shared<CustomFunctionNode<T>>(
        std::move(forward_func), std::move(backward_func));
    
    for (const auto& input : inputs) {
        if (input) {
            node->add_input(input);
        }
    }
    
    return node;
}

} // namespace graph
} // namespace ad
