/**
 * @file computational_graph.h
 * @brief Computational graph implementation for automatic differentiation
 */

#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <string>
#include <stdexcept>
#include <cmath>

namespace ad {
namespace graph {

template <typename T>
class GraphNode : public std::enable_shared_from_this<GraphNode<T>> {
public:
    using Ptr = std::shared_ptr<GraphNode<T>>;
    using WeakPtr = std::weak_ptr<GraphNode<T>>;

    virtual ~GraphNode() = default;

    virtual T forward() = 0;
    virtual void backward(const T& gradient) = 0;

    void add_dependent(Ptr node) {
        dependents_.push_back(node);
    }

    void add_input(Ptr node) {
        inputs_.push_back(node);
        node->add_dependent(this->shared_from_this());
    }

    const std::vector<Ptr>& get_inputs() const { return inputs_; }
    const std::vector<WeakPtr>& get_dependents() const { return dependents_; }

    T get_value() const { return value_; }
    T get_gradient() const { return gradient_; }
    void reset_gradient() { gradient_ = T(0); }

protected:
    T value_{};
    T gradient_{};
    std::vector<Ptr> inputs_;
    std::vector<WeakPtr> dependents_;
};

template <typename T>
class VariableNode : public GraphNode<T> {
public:
    explicit VariableNode(std::string name, T initial_value = T(0))
        : name_(std::move(name)) {
        this->value_ = initial_value;
    }

    void set_value(T value) { this->value_ = value; }
    const std::string& name() const { return name_; }

    T forward() override { return this->value_; }
    void backward(const T& gradient) override {
        this->gradient_ += gradient;
    }

private:
    std::string name_;
};

template <typename T>
class ConstantNode : public GraphNode<T> {
public:
    explicit ConstantNode(T value) : value_(value) {}

    T forward() override { return value_; }
    void backward(const T&) override {} // No-op for constants

private:
    T value_;
};

template <typename T>
class BinaryOperationNode : public GraphNode<T> {
public:
    using ForwardFunc = std::function<T(T, T)>;
    using BackwardFunc = std::function<std::pair<T, T>(T, T, T)>;

    BinaryOperationNode(typename GraphNode<T>::Ptr lhs,
                       typename GraphNode<T>::Ptr rhs,
                       ForwardFunc forward,
                       BackwardFunc backward)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)),
          forward_func_(std::move(forward)),
          backward_func_(std::move(backward)) {
        this->add_input(lhs_);
        this->add_input(rhs_);
    }

    T forward() override {
        T lhs_val = lhs_->forward();
        T rhs_val = rhs_->forward();
        this->value_ = forward_func_(lhs_val, rhs_val);
        return this->value_;
    }

    void backward(const T& gradient) override {
        auto [dlhs, drhs] = backward_func_(lhs_->get_value(), rhs_->get_value(), gradient);
        lhs_->backward(dlhs);
        rhs_->backward(drhs);
    }

private:
    typename GraphNode<T>::Ptr lhs_;
    typename GraphNode<T>::Ptr rhs_;
    ForwardFunc forward_func_;
    BackwardFunc backward_func_;
};

template <typename T>
class UnaryOperationNode : public GraphNode<T> {
public:
    using ForwardFunc = std::function<T(T)>;
    using BackwardFunc = std::function<T(T, T)>;

    UnaryOperationNode(typename GraphNode<T>::Ptr input,
                      ForwardFunc forward,
                      BackwardFunc backward)
        : input_(std::move(input)), 
          forward_func_(std::move(forward)),
          backward_func_(std::move(backward)) {
        this->add_input(input_);
    }

    T forward() override {
        T val = input_->forward();
        this->value_ = forward_func_(val);
        return this->value_;
    }

    void backward(const T& gradient) override {
        T input_val = input_->get_value();
        T dinput = backward_func_(input_val, gradient);
        input_->backward(dinput);
    }

private:
    typename GraphNode<T>::Ptr input_;
    ForwardFunc forward_func_;
    BackwardFunc backward_func_;
};

// ==================== OPERATION HELPERS ====================
template <typename T>
typename GraphNode<T>::Ptr add(typename GraphNode<T>::Ptr lhs, 
                              typename GraphNode<T>::Ptr rhs) {
    return std::make_shared<BinaryOperationNode<T>>(
        lhs, rhs,
        [](T a, T b) { return a + b; },
        [](T, T, T grad) { return std::make_pair(grad, grad); }
    );
}

template <typename T>
typename GraphNode<T>::Ptr multiply(typename GraphNode<T>::Ptr lhs, 
                                   typename GraphNode<T>::Ptr rhs) {
    return std::make_shared<BinaryOperationNode<T>>(
        lhs, rhs,
        [](T a, T b) { return a * b; },
        [](T a, T b, T grad) { return std::make_pair(grad * b, grad * a); }
    );
}

template <typename T>
typename GraphNode<T>::Ptr subtract(typename GraphNode<T>::Ptr lhs, 
                                   typename GraphNode<T>::Ptr rhs) {
    return std::make_shared<BinaryOperationNode<T>>(
        lhs, rhs,
        [](T a, T b) { return a - b; },
        [](T, T, T grad) { return std::make_pair(grad, -grad); }
    );
}

template <typename T>
typename GraphNode<T>::Ptr divide(typename GraphNode<T>::Ptr lhs,
                                 typename GraphNode<T>::Ptr rhs) {
    return std::make_shared<BinaryOperationNode<T>>(
        lhs, rhs,
        [](T a, T b) {
            if (b == 0) throw std::runtime_error("Division by zero");
            return a / b;
        },
        [](T a, T b, T grad) {
            T inv_b = T(1) / b;
            return std::make_pair(
                grad * inv_b,
                grad * (-a * inv_b * inv_b)
            );
        }
    );
}

// ==================== ELEMENTARY FUNCTIONS ====================
template <typename T>
typename GraphNode<T>::Ptr exp(typename GraphNode<T>::Ptr input) {
    return std::make_shared<UnaryOperationNode<T>>(
        input,
        [](T x) { return std::exp(x); },
        [](T x, T grad) { return grad * std::exp(x); }
    );
}

template <typename T>
typename GraphNode<T>::Ptr log(typename GraphNode<T>::Ptr input) {
    return std::make_shared<UnaryOperationNode<T>>(
        input,
        [](T x) { return std::log(x); },
        [](T x, T grad) { return grad / x; }
    );
}

template <typename T>
typename GraphNode<T>::Ptr sqrt(typename GraphNode<T>::Ptr input) {
    return std::make_shared<UnaryOperationNode<T>>(
        input,
        [](T x) { return std::sqrt(x); },
        [](T x, T grad) { return grad * T(0.5) / std::sqrt(x); }
    );
}

template <typename T>
typename GraphNode<T>::Ptr sin(typename GraphNode<T>::Ptr input) {
    return std::make_shared<UnaryOperationNode<T>>(
        input,
        [](T x) { return std::sin(x); },
        [](T x, T grad) { return grad * std::cos(x); }
    );
}

template <typename T>
typename GraphNode<T>::Ptr cos(typename GraphNode<T>::Ptr input) {
    return std::make_shared<UnaryOperationNode<T>>(
        input,
        [](T x) { return std::cos(x); },
        [](T x, T grad) { return -grad * std::sin(x); }
    );
}

template <typename T>
typename GraphNode<T>::Ptr tanh(typename GraphNode<T>::Ptr input) {
    return std::make_shared<UnaryOperationNode<T>>(
        input,
        [](T x) { return std::tanh(x); },
        [](T x, T grad) { 
            T t = std::tanh(x);
            return grad * (T(1) - t * t); 
        }
    );
}

} // namespace graph
} // namespace ad
