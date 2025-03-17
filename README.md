# AutoDiff

A comprehensive C++ framework for automatic differentiation supporting forward mode, reverse mode, expression optimization, and control flow differentiation.

## Introduction to Automatic Differentiation

### Understanding Continuous Variable Calculus Through Computation

Automatic Differentiation (AD) is a computational technique that allows us to calculate exact derivatives of functions, without resorting to numerical approximations or manual derivation. Unlike symbolic differentiation (which can lead to expression explosion) or finite differences (which suffer from numerical precision issues), AD provides a way to efficiently compute derivatives with machine precision.

The key insight of AD is that any function, no matter how complex, is ultimately composed of elementary operations (addition, multiplication, sin, exp, etc.) whose derivatives are known. By applying the chain rule through these operations, we can systematically compute derivatives of arbitrary functions.

For example, consider calculating the derivative of f(x) = sin(x²). A manual approach would require applying the chain rule:
- f'(x) = cos(x²) · d/dx(x²)
- f'(x) = cos(x²) · 2x

With automatic differentiation, this process happens transparently through computation:
1. We track both the value and derivative information as we compute
2. Each operation updates both components according to calculus rules
3. The result includes both the function value and its derivative

Our framework implements this approach through dual numbers, computational graphs, and rule-based differentiation, providing a powerful tool for scientific computing, optimization, and machine learning.

## Design Principles

This AD framework is built upon solid software engineering principles to ensure maintainability, extensibility, and performance. The core design follows the SOLID principles:

### Single Responsibility Principle
Each component in the framework has a clear, focused responsibility:
- `Expression` classes represent mathematical expressions
- `DualNumber` handles forward-mode differentiation
- `GraphNode` manages computational graph operations
- `Optimizer` focuses exclusively on expression optimization

### Open-Closed Principle
The framework is designed to be extended without modifying existing code:
- New mathematical operations can be added by deriving from `BinaryOperation` or `UnaryOperation`
- New optimizations can be implemented by extending the `ExpressionOptimizer`
- Custom functions can be integrated through the `CustomFunctionNode` interface

### Liskov Substitution Principle
Derived types maintain the contract of their base classes:
- All expressions follow the `Expression` interface contract
- All operations provide consistent differentiation behavior
- Graph nodes follow a consistent interface for forward/backward passes

### Interface Segregation Principle
Interfaces are kept focused and minimal:
- `Expression` provides only essential methods for evaluation and differentiation
- `GraphNode` separates forward and backward passes
- Separate mechanisms for expressing mathematical relationships vs. computing gradients

### Dependency Inversion Principle
Higher-level modules depend on abstractions:
- Algorithms operate on the `Expression` interface, not concrete types
- Optimization strategies depend on abstract expression interfaces
- Differentiation relies on abstract rule definitions

## Architecture Overview

```mermaid
flowchart TD
    %% Core Components
    subgraph "Core Components"
        A("fa:fa-project-diagram Expression System")
        B("fa:fa-arrow-right Forward Mode")
        C("fa:fa-arrow-left Reverse Mode")
        D("fa:fa-calculator Elementary Functions")
        E("fa:fa-code-branch Control Flow Constructs")
        F("fa:fa-rocket Optimization System")
        
        A --> B
        A --> C
        B --> D
        C --> D
        B --> E
        C --> E
        D --> F
        E --> F
    end
    
    %% Optimization System
    subgraph "Optimization System"
        G("fa:fa-compress-arrows-alt Constant Propagation")
        H("fa:fa-clone Common Subexpression Elimination")
        I("fa:fa-magic Algebraic Simplification")
        
        F --> G
        F --> H
        F --> I
    end
    
    %% Styling
    style A fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    style B fill:#d0ffe0,stroke:#30ff80,stroke-width:2px
    style C fill:#ffe0d0,stroke:#ff8030,stroke-width:2px
    style D fill:#e0d0ff,stroke:#8030ff,stroke-width:2px
    style E fill:#ffffd0,stroke:#ffff30,stroke-width:2px
    style F fill:#ffd0e0,stroke:#ff30ff,stroke-width:2px
    style G fill:#f0f0f0,stroke:#555555,stroke-width:1px
    style H fill:#f0f0f0,stroke:#555555,stroke-width:1px
    style I fill:#f0f0f0,stroke:#555555,stroke-width:1px
```

### Key Components

1. **Expression System**: Represents mathematical expressions as a tree of operations
2. **Differentiation Modes**:
   - Forward Mode: Efficient for functions with few inputs and many outputs
   - Reverse Mode: Efficient for functions with many inputs and few outputs
3. **Elementary Functions**: Comprehensive library of mathematical operations with derivatives
4. **Control Flow**: Handles loops and conditionals in a differentiable manner
5. **Optimization System**: Improves performance through various optimization techniques

## Implementation Highlights

### Expression System

```mermaid
graph TD
    %% Nodes
    A("fa:fa-project-diagram Expression Tree")
    B("fa:fa-plus +")
    C("fa:fa-times *")
    D("fa:fa-hashtag Constant: 2")
    E("fa:fa-superscript Variable: x")
    
    %% Connections
    A --> B
    B --> C
    B --> D
    C --> E
    
    %% Styling
    style A fill:#f5f5f5,stroke:#333,stroke-width:2px
    style B fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    style C fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    style D fill:#ffe0d0,stroke:#ff8030,stroke-width:2px
    style E fill:#d0ffe0,stroke:#30ff80,stroke-width:2px
```

This allows us to:
- Represent complex expressions through composition
- Perform symbolic differentiation by traversing the tree
- Apply optimizations through tree transformations

### Dual Numbers

Forward-mode AD is implemented using dual numbers, which represent both value and derivative:

```cpp
template <typename T>
class DualNumber {
public:
    T value;     // Function value
    T deriv;     // Derivative information
    
    // Operations update both components according to calculus rules
    DualNumber operator*(const DualNumber& other) const {
        return {
            value * other.value,                    // Value part
            deriv * other.value + value * other.deriv  // Derivative part (product rule)
        };
    }
};
```

### Computational Graph

```mermaid
graph TD
    %% Nodes
    A("fa:fa-chart-line Output Node")
    B("fa:fa-times Multiplication")
    C("fa:fa-wave-square Sin")
    D("fa:fa-superscript Variable: x")
    E("fa:fa-superscript Variable: x")
    
    %% Connections
    A --> B
    B --> C
    B --> D
    C --> E
    
    %% Styling
    style A fill:#ffe0d0,stroke:#ff8030,stroke-width:2px
    style B fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    style C fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    style D fill:#d0ffe0,stroke:#30ff80,stroke-width:2px
    style E fill:#d0ffe0,stroke:#30ff80,stroke-width:2px
```

During backward pass:
1. Start with gradient = 1 at output
2. Propagate gradients backward through the graph
3. Accumulate gradients at input nodes

### Common Subexpression Elimination

```mermaid
graph TD
    %% Before Optimization
    subgraph "Before Optimization"
        A1("fa:fa-plus +")
        B1("fa:fa-times *")
        B2("fa:fa-times *")
        C1("fa:fa-code x+a")
        C2("fa:fa-code x+a")
        
        A1 --> B1
        A1 --> B2
        B1 --> C1
        B2 --> C2
    end
    
    %% After Optimization
    subgraph "After Optimization"
        A2("fa:fa-plus +")
        B3("fa:fa-times *")
        C3("fa:fa-code x+a")
        
        A2 --> B3
        A2 --> C3
        B3 --> C3
    end
    
    %% Styling
    style A1 fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    style A2 fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    style B1 fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    style B2 fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    style B3 fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    style C1 fill:#ffe0d0,stroke:#ff8030,stroke-width:2px
    style C2 fill:#ffe0d0,stroke:#ff8030,stroke-width:2px
    style C3 fill:#ffe0d0,stroke:#ff8030,stroke-width:2px
```

### Control Flow Differentiation

The framework supports differentiating through loops and conditionals:

For loops:
- Unroll loop during forward pass
- Record all iterations
- Apply chain rule through iterations during backward pass

For conditionals:
- Use smooth approximations for differentiability
- Evaluate both branches and blend results
- Weight gradients based on the condition value

## Understanding the Development Stages

The framework is implemented in three progressive stages, with each stage building upon the foundation of the previous one. This progression allows for incremental development and testing.

### Stage 1 and 2 Relationship

The following diagram illustrates the relationship between key components from Stage 1 and Stage 2:

```mermaid
flowchart LR
    %% Stage 1 Components
    subgraph "S1: Expression System"
        A1("Expression<T>")
        A2("evaluate()")
        A3("differentiate()")
        A4("Terminal Expressions")
        A5("Variable<T>")
        A6("Constant<T>")
        A7("Operations")
        A8("BinaryOp<T>")
        A9("UnaryOp<T>")
        
        A1 --> A2
        A1 --> A3
        A1 --> A4
        A4 --> A5
        A4 --> A6
        A1 --> A7
        A7 --> A8
        A7 --> A9
    end
    
    %% Stage 2 Components
    subgraph "S2: Computational Graph"
        B1("GraphNode<T>")
        B2("forward()")
        B3("backward()")
        B4("Terminal Nodes")
        B5("VariableNode<T>")
        B6("ConstantNode<T>")
        B7("Operation Nodes")
        B8("BinaryOpNode<T>")
        B9("UnaryOpNode<T>")
        
        B1 --> B2
        B1 --> B3
        B1 --> B4
        B4 --> B5
        B4 --> B6
        B1 --> B7
        B7 --> B8
        B7 --> B9
    end
    
    %% Connections between stages
    A1 -- "Evolves Into" --> B1
    A5 -- "Maps To" --> B5
    A6 -- "Maps To" --> B6
    A8 -- "Maps To" --> B8
    A9 -- "Maps To" --> B9
    
    %% Bridge component
    subgraph "Bridge"
        C1("DualNumber<T>")
    end
    
    A1 -- "Connects via" --> C1
    B1 -- "Connects via" --> C1
    
    %% Styling
    style A1 fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    style B1 fill:#ffe0d0,stroke:#ff8030,stroke-width:2px
    style C1 fill:#d0ffe0,stroke:#30ff80,stroke-width:2px
```

### Stage 1 to Stage 2 Transition

The transition from Stage 1 to Stage 2 represents a shift in paradigm:

1. **Symbolic to Computational Graph**: While Stage 1 focuses on symbolic representation and manipulation of expressions (similar to mathematical formalism), Stage 2 introduces a computational graph approach that is optimized for efficient evaluation and gradient propagation.

2. **Value vs. Value+Gradient**: Stage 1's Expression system primarily computes function values with differentiation as a separate operation, while Stage 2's GraphNode system incorporates both forward value computation and backward gradient computation.

3. **Unidirectional vs. Bidirectional**: Stage 1 expressions flow in a single direction (evaluation), while Stage 2 nodes support bidirectional flow (forward for values, backward for gradients).

### Key Mappings Between Stages

The components of Stage 1 have direct analogs in Stage 2:

| Stage 1 Component    | Stage 2 Equivalent    | Evolution                                       |
|----------------------|-----------------------|------------------------------------------------|
| `Expression<T>`      | `GraphNode<T>`        | Adds gradient tracking and backward propagation |
| `Variable<T>`        | `VariableNode<T>`     | Adds gradient accumulation                      |
| `Constant<T>`        | `ConstantNode<T>`     | Zero gradient behavior                          |
| `BinaryOperation<T>` | `BinaryOperationNode<T>` | Encapsulates both forward and backward rules    |
| `UnaryOperation<T>`  | `UnaryOperationNode<T>`  | Encapsulates both forward and backward rules    |

### DualNumber as a Bridge

The `DualNumber<T>` class serves as a conceptual bridge between stages:

- It encapsulates the core idea of simultaneously tracking values and derivatives
- It provides the foundation for forward-mode differentiation
- Its principles inform the design of the computational graph in Stage 2

This progression from symbolic expressions to computational graphs enables the framework to support both forward-mode and reverse-mode differentiation efficiently, while maintaining a consistent interface for users.

## Template Metaprogramming and Static Analysis

The framework makes extensive use of modern C++ template metaprogramming and static analysis techniques to achieve compile-time optimizations, type safety, and improved runtime performance.

### Template Metaprogramming in Stage 1

In the Expression System stage, template metaprogramming is employed in several key ways:

1. **Expression Templates**

   Expression templates enable the framework to represent complex mathematical expressions as nested template instantiations. This approach allows the compiler to optimize expression evaluation by:
   
   ```cpp
   // Example of expression templates in action
   template <typename T>
   class Addition : public BinaryOperation<T> {
   public:
       // Constructor inherits from base class
       using BinaryOperation<T>::BinaryOperation;
       
       // Evaluation performed at compile time when possible
       T evaluate() const override {
           return this->left_->evaluate() + this->right_->evaluate();
       }
       
       // Type information preserved throughout template instantiation
       ExprPtr<T> differentiate(const std::string& variable) const override {
           return std::make_unique<Addition<T>>(
               this->left_->differentiate(variable),
               this->right_->differentiate(variable)
           );
       }
   };
   ```
   
   This allows for expressions to be composed at compile time while preserving the mathematical structure for differentiation.

2. **CRTP (Curiously Recurring Template Pattern)**

   The framework uses CRTP to achieve static polymorphism for operation classes, reducing the overhead of dynamic dispatch:
   
   ```cpp
   template <typename T, typename Derived>
   class BasicOperation {
   public:
       // Implementation that uses the derived class's specific behavior
       T compute() {
           return static_cast<Derived*>(this)->perform_operation();
       }
   };
   
   template <typename T>
   class Multiplication : public BasicOperation<T, Multiplication<T>> {
   public:
       // Specific implementation used by the base class
       T perform_operation() {
           return left_operand_ * right_operand_;
       }
   };
   ```

3. **Type Traits and Concept-Like Constraints**

   Although full C++20 concepts are not used, the framework employs type traits and static assertions to ensure proper types are used with the templates:
   
   ```cpp
   template <typename T>
   class DualNumber {
       static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type");
       // Implementation...
   };
   ```

**Advantages in Stage 1:**

- **Compile-Time Expression Evaluation**: Constant expressions can be evaluated at compile time, reducing runtime overhead
- **Type Safety**: Template parameters enforce type correctness throughout the expression tree
- **Zero-Cost Abstractions**: Template specializations allow for optimized implementations based on type characteristics
- **Generic Programming**: The same code can work with different numeric types (double, float, etc.)

### Template Metaprogramming in Stage 2

The Computational Graph stage extends the template metaprogramming approach with:

1. **Policy-Based Design**

   The framework uses policy classes to customize the behavior of graph nodes:
   
   ```cpp
   template <typename T, typename ForwardPolicy, typename BackwardPolicy>
   class CustomizableNode : public GraphNode<T> {
   public:
       T forward() override {
           return forward_policy_.compute(inputs_);
       }
       
       void backward(const T& gradient) override {
           backward_policy_.propagate(inputs_, gradient);
       }
       
   private:
       ForwardPolicy forward_policy_;
       BackwardPolicy backward_policy_;
   };
   ```
   
   This allows different differentiation strategies to be implemented and composed flexibly.

2. **Template Parameter Deduction**

   Stage 2 leverages template argument deduction to simplify API usage:
   
   ```cpp
   // Helper functions with automatic type deduction
   template <typename T>
   auto add(auto lhs, auto rhs) {
       return std::make_shared<BinaryOperationNode<T>>(
           std::move(lhs), std::move(rhs),
           [](T a, T b) { return a + b; },
           [](T, T, T grad) { return std::make_pair(grad, grad); }
       );
   }
   ```

3. **Variadic Templates**

   For handling operations with varying numbers of parameters:
   
   ```cpp
   template <typename T, typename... Args>
   auto make_custom_function(Args&&... args) {
       return std::make_shared<CustomFunctionNode<T>>(
           std::forward<Args>(args)...
       );
   }
   ```

**Advantages in Stage 2:**

- **Runtime Performance**: Template specializations for different operations optimize graph traversal
- **Memory Efficiency**: Template-based static memory allocation avoids dynamic allocation overhead
- **Compile-Time Function Composition**: Complex mathematical operations can be composed at compile time
- **Type-Safe Callbacks**: Lambda functions used in graph nodes have their types verified at compile time

### Static Analysis

The framework leverages static analysis both internally and through external tools:

1. **Internal Static Analysis**

   The expression optimizer performs static analysis on the expression tree to identify:
   
   - Constant subexpressions that can be evaluated once
   - Common subexpressions that can be reused
   - Algebraic simplifications that reduce computation
   
   ```cpp
   template <typename T>
   ExprPtr<T> ExpressionOptimizer<T>::fold_constants(ExprPtr<T> expr) {
       // Analyze the expression structure
       if (auto* binary_op = dynamic_cast<BinaryOperation<T>*>(expr.get())) {
           // Recursively optimize operands
           auto left = fold_constants(binary_op->left_clone());
           auto right = fold_constants(binary_op->right_clone());
           
           // If both operands are constants, compute the result statically
           if (dynamic_cast<Constant<T>*>(left.get()) && 
               dynamic_cast<Constant<T>*>(right.get())) {
               // Evaluate at compile-time or initialization time
               T result = evaluate_binary_op(binary_op, left, right);
               return std::make_unique<Constant<T>>(result);
           }
           // Otherwise, reconstruct with optimized operands
           return binary_op->clone_with(std::move(left), std::move(right));
       }
       // Handle other expression types...
       return expr->clone();
   }
   ```

2. **Compile-Time Expression Validation**

   The framework uses static assertions and SFINAE to validate expressions at compile-time:
   
   ```cpp
   // Ensure expressions are compatible for operations
   template <typename T>
   auto operator+(ExprPtr<T> a, ExprPtr<T> b) -> 
       typename std::enable_if<
           std::is_same<decltype(std::declval<T>() + std::declval<T>()), T>::value,
           ExprPtr<T>
       >::type {
       return std::make_unique<Addition<T>>(std::move(a), std::move(b));
   }
   ```

**Advantages of Static Analysis:**

- **Early Error Detection**: Type-related errors are caught at compile time rather than runtime
- **Performance Optimization**: Constant folding and algebraic simplifications reduce computational load
- **Memory Usage Reduction**: Common subexpression elimination reduces redundant computations and memory usage
- **Dead Code Elimination**: Unreachable branches in the expression tree are eliminated

### Synergy Between Stages

The template metaprogramming and static analysis techniques work synergistically across stages:

1. Stage 1 establishes the type-safe expression framework that Stage 2 builds upon
2. Template patterns in Stage 1 inform the more specialized templates in Stage 2
3. Static analysis in Stage 1 simplifies expressions before they enter the computational graph in Stage 2
4. The DualNumber implementation bridges both stages with consistent template patterns

This integration of modern C++ template techniques with mathematical concepts creates a framework that is both mathematically rigorous and computationally efficient, providing users with a powerful tool for automatic differentiation while maintaining excellent performance characteristics.

## Performance Considerations

The framework is optimized for both speed and memory efficiency:

- **Expression Optimization**: Constant folding, CSE, and algebraic simplifications
- **Memory Management**: Custom allocators and pooling for node creation
- **Computational Efficiency**: Memoization and lazy evaluation strategies
- **Checkpointing**: Trade computation for memory in large graphs

## Extension Points

The framework is designed for extensibility:

- **Custom Functions**: Register user-defined functions with custom derivatives
- **Domain-Specific Extensions**: Special-purpose functionality for ML, physics, finance
- **Hardware Acceleration**: SIMD and parallelization support
- **Integration APIs**: Connect with external libraries and frameworks

## Test Results

### Stage 1 Tests

The framework also includes extensive Catch2 unit tests, covering all aspects of automatic differentiation:

```
(base) 192:test ruben$ ./AutoDiffTests --success
Randomness seeded to: 457263149

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AutoDiffTests is a Catch2 v3.4.0 host application.
Run with -? for options

-------------------------------------------------------------------------------
Basic Variable Operations
  Evaluation
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_basic.cpp:11
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_basic.cpp:12: PASSED:
  REQUIRE( x->evaluate() == 2.0 )
with expansion:
  2.0 == 2.0

/Users/ruben/Research/AutoDiff/test/test_basic.cpp:14: PASSED:
  REQUIRE( x->evaluate() == 3.0 )
with expansion:
  3.0 == 3.0

-------------------------------------------------------------------------------
Basic Variable Operations
  Differentiation
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_basic.cpp:17
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_basic.cpp:21: PASSED:
  REQUIRE( dx->evaluate() == 1.0 )
with expansion:
  1.0 == 1.0

/Users/ruben/Research/AutoDiff/test/test_basic.cpp:22: PASSED:
  REQUIRE( dy->evaluate() == 0.0 )
with expansion:
  0.0 == 0.0

-------------------------------------------------------------------------------
Constant Expressions
  Evaluation
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_basic.cpp:29
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_basic.cpp:30: PASSED:
  REQUIRE( c->evaluate() == 5.0 )
with expansion:
  5.0 == 5.0

-------------------------------------------------------------------------------
Constant Expressions
  Differentiation
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_basic.cpp:33
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_basic.cpp:35: PASSED:
  REQUIRE( dc->evaluate() == 0.0 )
with expansion:
  0.0 == 0.0

-------------------------------------------------------------------------------
Addition Rule
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_operations.cpp:9
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_operations.cpp:14: PASSED:
  REQUIRE( expr->evaluate() == 5.0 )
with expansion:
  5.0 == 5.0

/Users/ruben/Research/AutoDiff/test/test_operations.cpp:19: PASSED:
  REQUIRE( dx->evaluate() == 1.0 )
with expansion:
  1.0 == 1.0

/Users/ruben/Research/AutoDiff/test/test_operations.cpp:20: PASSED:
  REQUIRE( dy->evaluate() == 1.0 )
with expansion:
  1.0 == 1.0

-------------------------------------------------------------------------------
Product Rule
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_operations.cpp:22
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_operations.cpp:36: PASSED:
  REQUIRE_THAT( deriv->evaluate(), WithinRel(expected, 1e-6) )
with expansion:
  -7.3151100949 and -7.31511 are within 0.0001% of each other

-------------------------------------------------------------------------------
Exponential Function
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_functions.cpp:9
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_functions.cpp:14: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::exp(1.0), 1e-6) )
with expansion:
  2.7182818285 and 2.71828 are within 0.0001% of each other

/Users/ruben/Research/AutoDiff/test/test_functions.cpp:17: PASSED:
  REQUIRE_THAT( deriv->evaluate(), WithinRel(std::exp(1.0), 1e-6) )
with expansion:
  2.7182818285 and 2.71828 are within 0.0001% of each other

-------------------------------------------------------------------------------
Composite Trigonometric Function
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_functions.cpp:20
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_functions.cpp:33: PASSED:
  REQUIRE_THAT( deriv->evaluate(), WithinRel(expected, 1e-6) )
with expansion:
  0.0 and 1.22465e-16 are within 0.0001% of each other

-------------------------------------------------------------------------------
Numerical Validation
  Simple Polynomial
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_validation.cpp:13
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_validation.cpp:24: PASSED:
  REQUIRE_THAT( deriv->evaluate(), WithinRel(3*pow(1.5, 2) + 2, 1e-6) )
with expansion:
  8.75 and 8.75 are within 0.0001% of each other

-------------------------------------------------------------------------------
Edge Case Handling
  Division Near Zero
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_edge_cases.cpp:12
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_edge_cases.cpp:19: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(1.0/1e-10, 1e-6) )
with expansion:
  10000000000.0 and 1e+10 are within 0.0001% of each other

-------------------------------------------------------------------------------
Edge Case Handling
  Exponential at Zero
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_edge_cases.cpp:22
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_edge_cases.cpp:26: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(1.0, 1e-12) )
with expansion:
  1.0 and 1 are within 1e-10% of each other

-------------------------------------------------------------------------------
Hyperbolic Functions
  Sinh Evaluation
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_hyperbolic.cpp:12
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_hyperbolic.cpp:14: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::sinh(1.0), 1e-6) )
with expansion:
  1.1752011936 and 1.1752 are within 0.0001% of each other

-------------------------------------------------------------------------------
Hyperbolic Functions
  Cosh Derivative
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_hyperbolic.cpp:17
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_hyperbolic.cpp:21: PASSED:
  REQUIRE_THAT( deriv->evaluate(), WithinRel(std::sinh(0.5), 1e-6) )
with expansion:
  0.5210953055 and 0.521095 are within 0.0001% of each other

-------------------------------------------------------------------------------
Composite Functions
  Nested Exponential
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_composite.cpp:12
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_composite.cpp:24: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(expected, 1e-6) )
with expansion:
  2.3197768247 and 2.31978 are within 0.0001% of each other

-------------------------------------------------------------------------------
Composite Functions
  Deeply Nested Derivative
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_composite.cpp:27
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_composite.cpp:38: PASSED:
  REQUIRE_THAT( deriv->evaluate(), WithinRel(expected, 1e-6) )
with expansion:
  0.5378828427 and 0.537883 are within 0.0001% of each other

-------------------------------------------------------------------------------
Multi-variable Expressions
  Partial Derivatives
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_multivariable.cpp:13
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_multivariable.cpp:25: PASSED:
  REQUIRE_THAT( dx->evaluate(), WithinRel(2*2*3, 1e-6) )
with expansion:
  12.0 and 12 are within 0.0001% of each other

/Users/ruben/Research/AutoDiff/test/test_multivariable.cpp:26: PASSED:
  REQUIRE_THAT( dy->evaluate(), WithinRel(2*2, 1e-6) )
with expansion:
  4.0 and 4 are within 0.0001% of each other

-------------------------------------------------------------------------------
Core Differentiation Rules
  Chain Rule
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:15
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_rules.cpp:28: PASSED:
  REQUIRE_THAT( deriv->evaluate(), WithinRel(expected, 1e-6) )
with expansion:
  -0.6323873983 and -0.632387 are within 0.0001% of each other

At x = 1.5
Analytical: -0.632387
Numerical:  -0.632387
Error:      1.2283e-10
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:29: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 1.5) )
with expansion:
  true

-------------------------------------------------------------------------------
Core Differentiation Rules
  Product Rule
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:32
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_rules.cpp:44: PASSED:
  REQUIRE_THAT( deriv->evaluate(), WithinRel(expected, 1e-6) )
with expansion:
  23.5288676193 and 23.5289 are within 0.0001% of each other

At x = 1.5
Analytical: 23.5289
Numerical:  23.5289
Error:      1.46903e-09
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:45: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 1.5) )
with expansion:
  true

-------------------------------------------------------------------------------
Core Differentiation Rules
  Quotient Rule
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:48
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_rules.cpp:72: PASSED:
  REQUIRE_THAT( deriv->evaluate(), WithinRel(expected, 1e-6) )
with expansion:
  0.0625 and 0.0625 are within 0.0001% of each other

At x = 2
Analytical: 0.0625
Numerical:  0.0625
Error:      6.42473e-11
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:73: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 2.0) )
with expansion:
  true

At x = 1
Analytical: -7.99729
Numerical:  -7.99729
Error:      1.87419e-10
-------------------------------------------------------------------------------
Core Differentiation Rules
  Combined Rules
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:76
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_rules.cpp:92: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 1.0) )
with expansion:
  true

At x = 0.5
Analytical: -0.457178
Numerical:  -0.457178
Error:      2.23407e-11
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:93: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 0.5) )
with expansion:
  true

-------------------------------------------------------------------------------
Core Differentiation Rules
  Multi-variable Product
  Partial derivative w.r.t x
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:103
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_rules.cpp:106: PASSED:
  REQUIRE_THAT( dx->evaluate(), WithinRel(expected, 1e-6) )
with expansion:
  27.0 and 27 are within 0.0001% of each other

-------------------------------------------------------------------------------
Core Differentiation Rules
  Multi-variable Product
  Partial derivative w.r.t y
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:109
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_rules.cpp:112: PASSED:
  REQUIRE_THAT( dy->evaluate(), WithinRel(expected, 1e-6) )
with expansion:
  13.5 and 13.5 are within 0.0001% of each other

-------------------------------------------------------------------------------
Core Differentiation Rules
  Edge Cases
  Division by constant
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:117
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_rules.cpp:124: PASSED:
  REQUIRE_THAT( deriv->evaluate(), WithinRel(0.2, 1e-6) )
with expansion:
  0.2 and 0.2 are within 0.0001% of each other

At x = 1
Analytical: 2.76355
Numerical:  2.76355
Error:      3.99236e-13
-------------------------------------------------------------------------------
Core Differentiation Rules
  Edge Cases
  Triple product
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:127
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_rules.cpp:136: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 1.0) )
with expansion:
  true

At x = 2
Analytical: 0.84147
Numerical:  0.84147
Error:      7.30997e-11
/Users/ruben/Research/AutoDiff/test/test_rules.cpp:137: PASSED:
  REQUIRE( validate_derivative(*expr, *y, 2.0) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Trigonometric Functions
  Sin
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:18
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:20: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::sin(pi/4), 1e-6) )
with expansion:
  0.7071067812 and 0.707107 are within 0.0001% of each other

At x = 0.785398
Analytical: 0.707107
Numerical:  0.707107
Error:      2.86138e-12
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:21: PASSED:
  REQUIRE( validate_derivative(*expr, *x, pi/4) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Trigonometric Functions
  Cos
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:24
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:26: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::cos(pi/4), 1e-6) )
with expansion:
  0.7071067812 and 0.707107 are within 0.0001% of each other

At x = 0.785398
Analytical: -0.707107
Numerical:  -0.707107
Error:      5.83724e-11
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:27: PASSED:
  REQUIRE( validate_derivative(*expr, *x, pi/4) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Trigonometric Functions
  Tan
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:30
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:32: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(1.0, 1e-6) )
with expansion:
  1.0 and 1 are within 0.0001% of each other

At x = 0.785398
Analytical: 2
Numerical:  2
Error:      1.09022e-10
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:33: PASSED:
  REQUIRE( validate_derivative(*expr, *x, pi/4) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Exponential/Logarithmic
  Exp
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:40
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:42: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::exp(1.0), 1e-6) )
with expansion:
  2.7182818285 and 2.71828 are within 0.0001% of each other

At x = 1
Analytical: 2.71828
Numerical:  2.71828
Error:      1.63457e-10
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:43: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 1.0) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Exponential/Logarithmic
  Log
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:46
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:48: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(0.0, 1e-6) )
with expansion:
  0.0 and 0 are within 0.0001% of each other

At x = 1
Analytical: 1
Numerical:  1
Error:      2.64221e-11
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:49: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 1.0) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Square Root/Reciprocal
  Sqrt
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:56
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:58: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(2.0, 1e-6) )
with expansion:
  2.0 and 2 are within 0.0001% of each other

At x = 4
Analytical: 0.25
Numerical:  0.25
Error:      7.60778e-11
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:59: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 4.0) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Square Root/Reciprocal
  Reciprocal
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:62
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:64: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(0.25, 1e-6) )
with expansion:
  0.25 and 0.25 are within 0.0001% of each other

At x = 4
Analytical: -0.0625
Numerical:  -0.0625
Error:      5.14166e-12
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:65: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 4.0) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Error Functions
  Erf
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:72
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:74: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::erf(0.5), 1e-6) )
with expansion:
  0.5204998778 and 0.5205 are within 0.0001% of each other

At x = 0.5
Analytical: 0.878783
Numerical:  0.878783
Error:      4.81054e-11
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:75: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 0.5) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Error Functions
  Erfc
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:78
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:80: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::erfc(0.5), 1e-6) )
with expansion:
  0.4795001222 and 0.4795 are within 0.0001% of each other

At x = 0.5
Analytical: -0.878783
Numerical:  -0.878783
Error:      2.03498e-11
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:81: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 0.5) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Gamma Functions
  Tgamma
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:88
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:90: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(24.0, 1e-6) )
with expansion:
  24.0 and 24 are within 0.0001% of each other

-------------------------------------------------------------------------------
Elementary Function Coverage
  Gamma Functions
  Lgamma
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:93
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:95: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::lgamma(5.0), 1e-6) )
with expansion:
  3.1780538303 and 3.17805 are within 0.0001% of each other

-------------------------------------------------------------------------------
Elementary Function Coverage
  Hyperbolic Functions
  Sinh
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:102
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:104: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::sinh(0.5), 1e-6) )
with expansion:
  0.5210953055 and 0.521095 are within 0.0001% of each other

At x = 0.5
Analytical: 1.12763
Numerical:  1.12763
Error:      7.38254e-12
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:105: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 0.5) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Hyperbolic Functions
  Cosh
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:108
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:110: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::cosh(0.5), 1e-6) )
with expansion:
  1.1276259652 and 1.12763 are within 0.0001% of each other

At x = 0.5
Analytical: 0.521095
Numerical:  0.521095
Error:      3.2453e-11
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:111: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 0.5) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Hyperbolic Functions
  Tanh
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:114
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:116: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::tanh(0.5), 1e-6) )
with expansion:
  0.4621171573 and 0.462117 are within 0.0001% of each other

At x = 0.5
Analytical: 0.786448
Numerical:  0.786448
Error:      1.49258e-12
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:117: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 0.5) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Inverse Hyperbolic Functions
  Asinh
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:122
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:125: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(0.0, 1e-6) )
with expansion:
  0.0 and 0 are within 0.0001% of each other

At x = 0
Analytical: 1
Numerical:  1
Error:      1.66644e-13
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:126: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 0.0) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Inverse Hyperbolic Functions
  Acosh
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:129
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:132: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::acosh(1.5), 1e-6) )
with expansion:
  0.9624236501 and 0.962424 are within 0.0001% of each other

At x = 1.5
Analytical: 0.894427
Numerical:  0.894427
Error:      1.32618e-10
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:133: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 1.5) )
with expansion:
  true

-------------------------------------------------------------------------------
Elementary Function Coverage
  Inverse Hyperbolic Functions
  Atanh
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:136
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:139: PASSED:
  REQUIRE_THAT( expr->evaluate(), WithinRel(std::atanh(0.5), 1e-6) )
with expansion:
  0.5493061443 and 0.549306 are within 0.0001% of each other

At x = 0.5
Analytical: 1.33333
Numerical:  1.33333
Error:      1.98372e-11
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:140: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 0.5) )
with expansion:
  true

At x = 0.25
Analytical: 0.330219
Numerical:  0.330219
Error:      1.34213e-11
-------------------------------------------------------------------------------
Elementary Function Coverage
  Composite Function Validation
  Nested Functions
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:147
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:153: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 0.25) )
with expansion:
  true

At x = 0.5
Analytical: 0.513974
Numerical:  0.513974
Error:      2.03545e-11
-------------------------------------------------------------------------------
Elementary Function Coverage
  Composite Function Validation
  Deep Composition
-------------------------------------------------------------------------------
/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:156
...............................................................................

/Users/ruben/Research/AutoDiff/test/test_elementary_functions.cpp:162: PASSED:
  REQUIRE( validate_derivative(*expr, *x, 0.5) )
with expansion:
  true

===============================================================================
All tests passed (69 assertions in 13 test cases)
```
### Stage 2 Tests

The framework has been thoroughly tested to ensure proper functionality. Below are the Stage 2 test results:

```
(base) 192:test ruben$ ./stage2_tests
Running main() from /Users/ruben/Research/AutoDiff/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 36 tests from 6 test suites.
[----------] Global test environment set-up.
[----------] 5 tests from ComputationalGraphTest
[ RUN      ] ComputationalGraphTest.BasicForwardPass
[       OK ] ComputationalGraphTest.BasicForwardPass (0 ms)
[ RUN      ] ComputationalGraphTest.BackwardGradientAccumulation
[       OK ] ComputationalGraphTest.BackwardGradientAccumulation (0 ms)
[ RUN      ] ComputationalGraphTest.MultiDependentNodes
[       OK ] ComputationalGraphTest.MultiDependentNodes (0 ms)
[ RUN      ] ComputationalGraphTest.SubtractionGradient
[       OK ] ComputationalGraphTest.SubtractionGradient (0 ms)
[ RUN      ] ComputationalGraphTest.DivisionGradient
[       OK ] ComputationalGraphTest.DivisionGradient (0 ms)
[----------] 5 tests from ComputationalGraphTest (0 ms total)

[----------] 8 tests from ReverseModeTest
[ RUN      ] ReverseModeTest.SingleVariableGradient
[       OK ] ReverseModeTest.SingleVariableGradient (0 ms)
[ RUN      ] ReverseModeTest.BasicOperations
[       OK ] ReverseModeTest.BasicOperations (0 ms)
[ RUN      ] ReverseModeTest.ElementaryFunctions
[       OK ] ReverseModeTest.ElementaryFunctions (0 ms)
[ RUN      ] ReverseModeTest.CompositeFunction
[       OK ] ReverseModeTest.CompositeFunction (0 ms)
[ RUN      ] ReverseModeTest.MultipleVariables
[       OK ] ReverseModeTest.MultipleVariables (0 ms)
[ RUN      ] ReverseModeTest.GradientAccumulation
[       OK ] ReverseModeTest.GradientAccumulation (0 ms)
[ RUN      ] ReverseModeTest.SetVariable
[       OK ] ReverseModeTest.SetVariable (0 ms)
[ RUN      ] ReverseModeTest.ChainedOperations
[       OK ] ReverseModeTest.ChainedOperations (0 ms)
[----------] 8 tests from ReverseModeTest (0 ms total)

[----------] 2 tests from ControlFlowTest
[ RUN      ] ControlFlowTest.SimpleLoopDifferentiation
[       OK ] ControlFlowTest.SimpleLoopDifferentiation (0 ms)
[ RUN      ] ControlFlowTest.ConditionalBranchGradient
[       OK ] ControlFlowTest.ConditionalBranchGradient (0 ms)
[----------] 2 tests from ControlFlowTest (0 ms total)

[----------] 9 tests from CustomFunctionTest
[ RUN      ] CustomFunctionTest.BasicFunction
[       OK ] CustomFunctionTest.BasicFunction (0 ms)
[ RUN      ] CustomFunctionTest.SingleInput
[       OK ] CustomFunctionTest.SingleInput (0 ms)
[ RUN      ] CustomFunctionTest.MultipleInputs
[       OK ] CustomFunctionTest.MultipleInputs (0 ms)
[ RUN      ] CustomFunctionTest.Composition
[       OK ] CustomFunctionTest.Composition (0 ms)
[ RUN      ] CustomFunctionTest.GradientAccumulation
[       OK ] CustomFunctionTest.GradientAccumulation (0 ms)
[ RUN      ] CustomFunctionTest.NumericalGradientCheck
[       OK ] CustomFunctionTest.NumericalGradientCheck (1 ms)
[ RUN      ] CustomFunctionTest.VariableReuse
[       OK ] CustomFunctionTest.VariableReuse (0 ms)
[ RUN      ] CustomFunctionTest.EmptyInputs
[       OK ] CustomFunctionTest.EmptyInputs (0 ms)
[ RUN      ] CustomFunctionTest.MismatchedGradients
[       OK ] CustomFunctionTest.MismatchedGradients (0 ms)
[----------] 9 tests from CustomFunctionTest (1 ms total)

[----------] 5 tests from OptimizerTest
[ RUN      ] OptimizerTest.ConstantPropagationAndFolding
[       OK ] OptimizerTest.ConstantPropagationAndFolding (0 ms)
[ RUN      ] OptimizerTest.CommonSubexpressionElimination
[       OK ] OptimizerTest.CommonSubexpressionElimination (0 ms)
[ RUN      ] OptimizerTest.AlgebraicSimplifications
[       OK ] OptimizerTest.AlgebraicSimplifications (0 ms)
[ RUN      ] OptimizerTest.PerformanceMonitoring
[       OK ] OptimizerTest.PerformanceMonitoring (0 ms)
[ RUN      ] OptimizerTest.ErrorHandlingInOptimizer
[       OK ] OptimizerTest.ErrorHandlingInOptimizer (1 ms)
[----------] 5 tests from OptimizerTest (1 ms total)

[----------] 7 tests from ForwardModeTest
[ RUN      ] ForwardModeTest.SingleVariableGradient
[       OK ] ForwardModeTest.SingleVariableGradient (0 ms)
[ RUN      ] ForwardModeTest.BasicOperations
[       OK ] ForwardModeTest.BasicOperations (0 ms)
[ RUN      ] ForwardModeTest.ElementaryFunctions
[       OK ] ForwardModeTest.ElementaryFunctions (0 ms)
[ RUN      ] ForwardModeTest.CompositeFunction
[       OK ] ForwardModeTest.CompositeFunction (0 ms)
[ RUN      ] ForwardModeTest.MultipleVariables
[       OK ] ForwardModeTest.MultipleVariables (0 ms)
[ RUN      ] ForwardModeTest.ScalarOperations
[       OK ] ForwardModeTest.ScalarOperations (0 ms)
[ RUN      ] ForwardModeTest.ChainedOperations
[       OK ] ForwardModeTest.ChainedOperations (0 ms)
[----------] 7 tests from ForwardModeTest (0 ms total)

[----------] Global test environment tear-down
[==========] 36 tests from 6 test suites ran. (4 ms total)
[  PASSED  ] 36 tests.
```

