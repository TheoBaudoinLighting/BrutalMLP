include(FetchContent)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(googletest)

add_executable(brutal_mlp_tests brutal_mlp_tests.cpp)
target_link_libraries(brutal_mlp_tests
    PRIVATE
        brutal_mlp::brutal_mlp
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(brutal_mlp_tests)
