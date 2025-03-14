cmake_minimum_required(VERSION 3.12)

project(AutoDiffTests)

# Create test executable
add_executable(AutoDiffTests
    test_basic.cpp
    test_operations.cpp
    test_functions.cpp
    test_validation.cpp
    test_edge_cases.cpp  # New
    test_hyperbolic.cpp  # New
    test_composite.cpp   # New
    test_multivariable.cpp  # New
    test_rules.cpp
    test_elementary_functions.cpp
)
# Link dependencies
target_link_libraries(AutoDiffTests PRIVATE
    AutoDiff
    Catch2::Catch2WithMain
)

# Add to CTest
add_test(NAME AutoDiffTests COMMAND AutoDiffTests)
