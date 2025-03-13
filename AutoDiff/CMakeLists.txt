# Header-only library configuration
add_library(AutoDiff INTERFACE)
target_include_directories(AutoDiff INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/AutoDiff>
)

target_compile_features(AutoDiff INTERFACE cxx_std_17)

# Static analysis configuration
target_compile_options(AutoDiff INTERFACE
    -fanalyzer
    -Wduplicated-cond
    -Wlogical-op
    -Wuseless-cast
)
