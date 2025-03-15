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

    CustomFunctionNode(ForwardFunc forward, BackwardFunc backward)
        : forward_func_(std::move(forward)),
          backward_func_(std::move(backward)) {}

    T forward() override {
        std::vector<T> input_values;
        for (const auto& input : this->inputs_) {
            input_values.push_back(input->forward());
        }
        this->value_ = forward_func_(input_values);
        return this->value_;
    }

    void backward(const T& gradient) override {
        std::vector<T> input_values;
        for (const auto& input : this->inputs_) {
            input_values.push_back(input->get_value());
        }
        std::vector<T> input_grads = backward_func_(input_values, gradient);
        for (size_t i = 0; i < this->inputs_.size(); ++i) {
            this->inputs_[i]->backward(input_grads[i]);
        }
    }

private:
    ForwardFunc forward_func_;
    BackwardFunc backward_func_;
};

template <typename T, typename ForwardFunc, typename BackwardFunc>
auto make_custom_function(std::vector<typename GraphNode<T>::Ptr> inputs,
                          ForwardFunc&& forward_func,
                          BackwardFunc&& backward_func) {
    auto node = std::make_shared<CustomFunctionNode<T, ForwardFunc, BackwardFunc>>(
        std::forward<ForwardFunc>(forward_func),
        std::forward<BackwardFunc>(backward_func));
    for (const auto& input : inputs) {
        node->add_input(input);
    }
    return node;
}

} // namespace graph
} // namespace ad
