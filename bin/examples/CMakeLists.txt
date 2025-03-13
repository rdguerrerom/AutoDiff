add_executable(example1 example1.cpp)
target_link_libraries(example1 PRIVATE AutoDiff)

# Installation for examples (optional)
install(TARGETS example1 DESTINATION bin/examples)
