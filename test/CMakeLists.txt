# Add test sources
add_executable(autodiff_test test_main.cpp test_dual.cpp test_expressions.cpp)
target_link_libraries(autodiff_test PRIVATE AutoDiff)

# Google Test configuration (example)
find_package(GTest REQUIRED)
target_link_libraries(autodiff_test PRIVATE GTest::GTest GTest::Main)

# Add test cases
add_test(NAME DualNumberTest COMMAND autodiff_test --gtest_filter=DualNumber.*)
add_test(NAME ExpressionTest COMMAND autodiff_test --gtest_filter=Expression.*)
