file(GLOB BENCH_SOURCES *.cpp)
foreach(source ${BENCH_SOURCES})
    get_filename_component(exec_name ${source} NAME_WE)
    add_executable(${exec_name} ${source})
    target_link_libraries(${exec_name} PRIVATE 
        AutoDiff 
        benchmark::benchmark
    )
endforeach()
