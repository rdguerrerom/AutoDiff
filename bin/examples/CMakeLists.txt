add_executable(example1 example1.cpp)
add_executable(optimizer_example example_optimizer.cpp)
add_executable(optimizer_grad_example example_optimizer_grad.cpp)

target_link_libraries(example1 PRIVATE AutoDiff)
target_link_libraries(optimizer_example PRIVATE AutoDiff)
target_link_libraries(optimizer_grad_example PRIVATE AutoDiff)

# Installation for examples (optional)
install(TARGETS 
    example1 
    optimizer_example
    optimizer_grad_example
    DESTINATION bin/examples
)
