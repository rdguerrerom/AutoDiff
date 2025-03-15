/**
 * @file reverse_mode.h
 * @brief Reverse mode automatic differentiation implementation
 */

#pragma once
#include "computational_graph.h"
#include <unordered_map>
#include <memory>
#include <cmath>

namespace ad {
namespace reverse {

template <typename T>
class ReverseMode {
public:
    std::shared_ptr<graph::VariableNode<T>> add_variable(const std::string& name, T initial_value = T(0)) {
        auto node = std::make_shared<graph::VariableNode<T>>(name, initial_value);
        variables_[name] = node;
        return node;
    }

    void set_variable(const std::string& name, T value) {
        if (auto it = variables_.find(name); it != variables_.end()) {
            it->second->set_value(value);
        }
    }

    std::unordered_map<std::string, T> compute_gradients(
        std::shared_ptr<graph::GraphNode<T>> output_node
    ) {
        // Reset gradients before backward pass
        for (auto& [name, node] : variables_) node->reset_gradient();
        output_node->backward(T(1));
        
        // Collect results
        std::unordered_map<std::string, T> gradients;
        for (auto& [name, node] : variables_) {
            gradients[name] = node->get_gradient();
        }
        return gradients;
    }

private:
    std::unordered_map<std::string, std::shared_ptr<graph::VariableNode<T>>> variables_;
};

} // namespace reverse
} // namespace ad
