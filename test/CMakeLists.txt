cmake_minimum_required(VERSION 3.14)
project(AutoDiffTests)

# Ensure GTest is found
find_package(GTest REQUIRED)  # Add this line

# Catch2 Tests (for Catch2-based tests)
add_executable(AutoDiffTests
    test_basic.cpp
    test_operations.cpp
    test_functions.cpp
    test_validation.cpp
    test_edge_cases.cpp
    test_hyperbolic.cpp
    test_composite.cpp
    test_multivariable.cpp
    test_rules.cpp
    test_elementary_functions.cpp  # Removed test_optimizer.cpp
)
target_link_libraries(AutoDiffTests PRIVATE 
    AutoDiff 
    Catch2::Catch2WithMain
)

# GoogleTest Tests (for GTest-based tests)
add_executable(stage2_tests
    test_computational_graph.cpp
    test_reverse_mode.cpp
    test_control_flow.cpp
    test_custom_functions.cpp
    test_optimizer.cpp  # Moved here
    test_forward_mode.cpp
)
target_link_libraries(stage2_tests PRIVATE 
    AutoDiff 
    GTest::GTest 
    GTest::Main 
    Threads::Threads  # Add pthread support
)

# Register tests
add_test(NAME AutoDiffTests COMMAND AutoDiffTests)
add_test(NAME stage2_tests COMMAND stage2_tests)
