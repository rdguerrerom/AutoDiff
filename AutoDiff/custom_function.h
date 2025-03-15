/**
 * @file custom_function.h
 * @brief User-defined function nodes for computational graphs
 */

#pragma once
#include "computational_graph.h"
#include <vector>

namespace ad {
namespace graph {

template <typename T, typename ForwardFunc, typename BackwardFunc>
class CustomFunctionNode : public GraphNode<T> {
public:
    using NodePtr = typename GraphNode<T>::Ptr;

    CustomFunctionNode(std::vector<NodePtr> inputs, ForwardFunc forward, BackwardFunc backward)
        : inputs_(std::move(inputs)), forward_func_(forward), backward_func_(backward) {
        for (const auto& input : inputs_) {
            this->add_input(input);
        }
    }

    T forward() override {
        std::vector<T> input_values;
        for (const auto& input : inputs_) {
            input_values.push_back(input->forward());
        }
        this->value_ = forward_func_(input_values);
        return this->value_;
    }

    void backward(const T& gradient) override {
        std::vector<T> input_values;
        for (const auto& input : inputs_) {
            input_values.push_back(input->get_value());
        }
        std::vector<T> input_grads = backward_func_(input_values, gradient);
        for (size_t i = 0; i < inputs_.size(); ++i) {
            inputs_[i]->backward(input_grads[i]);
        }
    }

private:
    std::vector<NodePtr> inputs_;
    ForwardFunc forward_func_;
    BackwardFunc backward_func_;
};

template <typename T, typename ForwardFunc, typename BackwardFunc>
auto make_custom_function(std::vector<typename GraphNode<T>::Ptr> inputs,
                          ForwardFunc&& forward_func,
                          BackwardFunc&& backward_func) {
    return std::make_shared<CustomFunctionNode<T, ForwardFunc, BackwardFunc>>(
        std::move(inputs),
        std::forward<ForwardFunc>(forward_func),
        std::forward<BackwardFunc>(backward_func));
}

} // namespace graph
} // namespace ad
