include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(googlebenchmark)

add_executable(brutal_mlp_benchmarks brutal_mlp_benchmarks.cpp)
target_link_libraries(brutal_mlp_benchmarks
    PRIVATE
        brutal_mlp::brutal_mlp
        benchmark::benchmark
)
