/**
 * @file control_flow.h
 * @brief Control flow nodes for computational graphs
 */

#pragma once
#include "computational_graph.h"
#include <functional>

namespace ad {
namespace graph {

template <typename T>
class LoopNode : public GraphNode<T> {
public:
    using NodePtr = typename GraphNode<T>::Ptr;
    using ConditionFunc = std::function<bool(const T&)>;
    using BodyFunc = std::function<NodePtr(NodePtr)>;

    LoopNode(NodePtr initial_state, ConditionFunc condition, BodyFunc body)
        : initial_state_(initial_state), condition_(condition), body_(body) {}

    T forward() override {
        current_state_ = initial_state_;
        iterations_.clear();
        
        while (condition_(current_state_->forward())) {
            auto new_state = body_(current_state_);
            iterations_.push_back(new_state);
            current_state_ = new_state;
        }
        this->value_ = current_state_->get_value();
        return this->value_;
    }

    void backward(const T& gradient) override {
        T current_grad = gradient;
        for (auto it = iterations_.rbegin(); it != iterations_.rend(); ++it) {
            (*it)->backward(current_grad);
        }
        initial_state_->backward(current_grad);
    }

private:
    NodePtr initial_state_;
    NodePtr current_state_;
    ConditionFunc condition_;
    BodyFunc body_;
    std::vector<NodePtr> iterations_;
};

template <typename T>
class ConditionalNode : public GraphNode<T> {
public:
    using NodePtr = typename GraphNode<T>::Ptr;

    ConditionalNode(NodePtr condition, NodePtr true_branch, NodePtr false_branch, T smoothing = T(0.01))
        : condition_(condition), true_branch_(true_branch), false_branch_(false_branch),
          smoothing_(smoothing) {}

    T forward() override {
        T cond_val = condition_->forward();
        T blend = sigmoid(cond_val / smoothing_);
        
        T true_val = true_branch_->forward();
        T false_val = false_branch_->forward();
        
        this->value_ = blend * true_val + (T(1) - blend) * false_val;
        return this->value_;
    }

    void backward(const T& gradient) override {
        T cond_val = condition_->get_value();
        T blend = sigmoid(cond_val / smoothing_);
        T dblend = blend * (T(1) - blend) / smoothing_;

        true_branch_->backward(gradient * blend);
        false_branch_->backward(gradient * (T(1) - blend));
        condition_->backward(gradient * (true_branch_->get_value() - false_branch_->get_value()) * dblend);
    }

private:
    NodePtr condition_;
    NodePtr true_branch_;
    NodePtr false_branch_;
    T smoothing_;

    T sigmoid(T x) const { return T(1) / (T(1) + std::exp(-x)); }
};

} // namespace graph
} // namespace ad
