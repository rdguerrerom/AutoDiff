cmake_minimum_required(VERSION 3.12)

project(AutoDiffTests)

# Create test executable
add_executable(AutoDiffTests
    test_basic.cpp
    test_operations.cpp
    test_functions.cpp
    test_validation.cpp
)

# Link dependencies
target_link_libraries(AutoDiffTests PRIVATE
    AutoDiff
    Catch2::Catch2WithMain
)

# Add to CTest
add_test(NAME AutoDiffTests COMMAND AutoDiffTests)
