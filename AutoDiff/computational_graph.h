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

// Forward declarations
template <typename T> class GraphNode;
template <typename T> class VariableNode;
template <typename T> class ConstantNode;
template <typename T> class BinaryOperationNode;
template <typename T> class UnaryOperationNode;
template <typename T> class AdditionNode;
template <typename T> class MultiplicationNode;
template <typename T> class SubtractionNode;
template <typename T> class DivisionNode;
template <typename T> class ExpNode;
template <typename T> class LogNode;
template <typename T> class SinNode;
template <typename T> class CosNode;
template <typename T> class TanNode;
template <typename T> class SinhNode;
template <typename T> class CoshNode;
template <typename T> class TanhNode;
template <typename T> class AsinhNode;
template <typename T> class AcoshNode;
template <typename T> class AtanhNode;
template <typename T> class SqrtNode;
template <typename T> class ReciprocalNode;
template <typename T> class ErfNode;
template <typename T> class ErfcNode;
template <typename T> class TgammaNode;
template <typename T> class LgammaNode;

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
        this->gradient_ += gradient; // Fixed: Accumulate gradients
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

    BinaryOperationNode(ForwardFunc forward, BackwardFunc backward)
        : forward_func_(std::move(forward)),
          backward_func_(std::move(backward)) {}

    T forward() override {
        T lhs_val = this->inputs_[0]->forward();
        T rhs_val = this->inputs_[1]->forward();
        this->value_ = forward_func_(lhs_val, rhs_val);
        return this->value_;
    }

    void backward(const T& gradient) override {
        T lhs_val = this->inputs_[0]->get_value();
        T rhs_val = this->inputs_[1]->get_value();
        auto [dlhs, drhs] = backward_func_(lhs_val, rhs_val, gradient);
        this->inputs_[0]->backward(dlhs);
        this->inputs_[1]->backward(drhs);
    }

private:
    ForwardFunc forward_func_;
    BackwardFunc backward_func_;
};

template <typename T>
class UnaryOperationNode : public GraphNode<T> {
public:
    using ForwardFunc = std::function<T(T)>;
    using BackwardFunc = std::function<T(T, T)>;

    UnaryOperationNode(ForwardFunc forward, BackwardFunc backward)
        : forward_func_(std::move(forward)),
          backward_func_(std::move(backward)) {}

    T forward() override {
        T val = this->inputs_[0]->forward();
        this->value_ = forward_func_(val);
        return this->value_;
    }

    void backward(const T& gradient) override {
        T input_val = this->inputs_[0]->get_value();
        T dinput = backward_func_(input_val, gradient);
        this->inputs_[0]->backward(dinput);
    }

private:
    ForwardFunc forward_func_;
    BackwardFunc backward_func_;
};

// ==================== OPERATION HELPERS ====================
template <typename T>
typename GraphNode<T>::Ptr add(typename GraphNode<T>::Ptr lhs, 
                              typename GraphNode<T>::Ptr rhs) {
    auto node = std::make_shared<BinaryOperationNode<T>>(
        [](T a, T b) { return a + b; },
        [](T, T, T grad) { return std::make_pair(grad, grad); }
    );
    node->add_input(lhs);
    node->add_input(rhs);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr multiply(typename GraphNode<T>::Ptr lhs, 
                                   typename GraphNode<T>::Ptr rhs) {
    auto node = std::make_shared<BinaryOperationNode<T>>(
        [](T a, T b) { return a * b; },
        [](T a, T b, T grad) { return std::make_pair(grad * b, grad * a); }
    );
    node->add_input(lhs);
    node->add_input(rhs);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr subtract(typename GraphNode<T>::Ptr lhs, 
                                   typename GraphNode<T>::Ptr rhs) {
    auto node = std::make_shared<BinaryOperationNode<T>>(
        [](T a, T b) { return a - b; },
        [](T, T, T grad) { return std::make_pair(grad, -grad); }
    );
    node->add_input(lhs);
    node->add_input(rhs);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr divide(typename GraphNode<T>::Ptr lhs,
                                 typename GraphNode<T>::Ptr rhs) {
    auto node = std::make_shared<BinaryOperationNode<T>>(
        [](T a, T b) {
            if (b == 0) throw std::runtime_error("Division by zero");
            return a / b;
        },
        [](T a, T b, T grad) {
            T inv_b = T(1) / b;
            return std::make_pair(grad * inv_b, grad * (-a * inv_b * inv_b));
        }
    );
    node->add_input(lhs);
    node->add_input(rhs);
    return node;
}

// ==================== ELEMENTARY FUNCTIONS ====================
template <typename T>
typename GraphNode<T>::Ptr exp(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::exp(x); },
        [](T x, T grad) { return grad * std::exp(x); }
    );
    node->add_input(input);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr log(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::log(x); },
        [](T x, T grad) { return grad / x; }
    );
    node->add_input(input);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr sqrt(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::sqrt(x); },
        [](T x, T grad) { return grad * T(0.5) / std::sqrt(x); }
    );
    node->add_input(input);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr sin(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::sin(x); },
        [](T x, T grad) { return grad * std::cos(x); }
    );
    node->add_input(input);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr cos(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::cos(x); },
        [](T x, T grad) { return -grad * std::sin(x); }
    );
    node->add_input(input);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr tanh(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::tanh(x); },
        [](T x, T grad) { 
            T t = std::tanh(x);
            return grad * (T(1) - t * t); 
        }
    );
    node->add_input(input);
    return node;
}

// ==================== ELEMENTARY FUNCTION NODES ====================
template <typename T>
typename GraphNode<T>::Ptr tan(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::tan(x); },
        [](T x, T grad) { return grad * (T(1) + std::tan(x)*std::tan(x)); }
    );
    node->add_input(input);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr sinh(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::sinh(x); },
        [](T x, T grad) { return grad * std::cosh(x); }
    );
    node->add_input(input);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr cosh(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::cosh(x); },
        [](T x, T grad) { return grad * std::sinh(x); }
    );
    node->add_input(input);
    return node;
}


template <typename T>
typename GraphNode<T>::Ptr asinh(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::asinh(x); },
        [](T x, T grad) { return grad / std::sqrt(x*x + T(1)); }
    );
    node->add_input(input);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr acosh(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::acosh(x); },
        [](T x, T grad) { return grad / std::sqrt(x*x - T(1)); }
    );
    node->add_input(input);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr atanh(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::atanh(x); },
        [](T x, T grad) { return grad / (T(1) - x*x); }
    );
    node->add_input(input);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr erf(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::erf(x); },
        [](T x, T grad) { 
            const T pi = std::acos(-T(1));
            return grad * T(2)/std::sqrt(pi) * std::exp(-x*x); 
        }
    );
    node->add_input(input);
    return node;
}

template <typename T>
typename GraphNode<T>::Ptr erfc(typename GraphNode<T>::Ptr input) {
    auto node = std::make_shared<UnaryOperationNode<T>>(
        [](T x) { return std::erfc(x); },
        [](T x, T grad) { 
            const T pi = std::acos(-T(1));
            return -grad * T(2)/std::sqrt(pi) * std::exp(-x*x); 
        }
    );
    node->add_input(input);
    return node;
}




} // namespace graph
} // namespace ad
