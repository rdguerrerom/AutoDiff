cmake_minimum_required(VERSION 3.14)
project(AutoDiff LANGUAGES C CXX)

# Modern C++ Standard Configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build Type Configuration
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug or Release" FORCE)
endif()

# Compiler Flags (Split into individual flags)
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wextra>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Werror>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wno-analyzer-malloc-leak>"    # Suppress malloc-leak warnings
    "$<$<COMPILE_LANGUAGE:CXX>:-Wno-analyzer-use-of-uninitialized-value>"  # Suppress uninitialized warnings
    "$<$<COMPILE_LANGUAGE:C>:-Wall>"
    "$<$<COMPILE_LANGUAGE:C>:-Wextra>"
    "$<$<COMPILE_LANGUAGE:C>:-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:C>:-Werror>"
)

# Configuration-Specific Flags
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Core Library
add_subdirectory(AutoDiff)

# Benchmark Configuration
option(BUILD_BENCHMARKS "Build benchmark suite" ON)
if(BUILD_BENCHMARKS)
    include(FetchContent)
    
    # Fetch and configure Google Benchmark
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.0
    )
    
    # Set these options BEFORE making benchmark available
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests")
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable GTest dependency")
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "Download missing deps")
    
    FetchContent_MakeAvailable(benchmark)
    
    # Add this to suppress pthread warnings on macOS
    if(APPLE)
        set_target_properties(benchmark PROPERTIES 
            INTERFACE_COMPILE_OPTIONS "-Wno-deprecated-declarations"
        )
    endif()
    
    add_subdirectory(benchmarks)
endif()

# Testing Configuration
option(BUILD_TESTING "Build test suite" ON)
if(BUILD_TESTING)
    include(FetchContent)

    # Fetch and configure Catch2 (v3.4.0)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    # Fetch and configure GoogleTest (latest main branch)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_subdirectory(test)
endif()

# Examples
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(bin/examples)
endif()

# Installation
install(DIRECTORY AutoDiff/ DESTINATION include/AutoDiff)
