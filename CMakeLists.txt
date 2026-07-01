cmake_minimum_required(VERSION 3.20)

project(brutal_mlp
    VERSION 0.1.0
    DESCRIPTION "Production-ready multilayer perceptron library"
    LANGUAGES CXX
)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(BRUTAL_MLP_TOP_LEVEL_DEFAULT ON)
else()
    set(BRUTAL_MLP_TOP_LEVEL_DEFAULT OFF)
endif()

option(BRUTAL_MLP_BUILD_TESTS "Build brutal_mlp unit tests" ${BRUTAL_MLP_TOP_LEVEL_DEFAULT})
option(BRUTAL_MLP_BUILD_BENCHMARKS "Build brutal_mlp benchmarks" OFF)
option(BRUTAL_MLP_BUILD_EXAMPLES "Build brutal_mlp examples" ${BRUTAL_MLP_TOP_LEVEL_DEFAULT})
option(BRUTAL_MLP_USE_DOUBLE "Use double precision Scalar for debug and numerical investigation builds" OFF)

add_library(brutal_mlp src/brutal_mlp.cpp)
add_library(brutal_mlp::brutal_mlp ALIAS brutal_mlp)

target_compile_features(brutal_mlp PUBLIC cxx_std_17)

target_compile_definitions(brutal_mlp
    PUBLIC
        BRUTAL_MLP_VERSION_STRING="${PROJECT_VERSION}"
        $<$<BOOL:${BRUTAL_MLP_USE_DOUBLE}>:BRUTAL_MLP_USE_DOUBLE=1>
)

target_include_directories(brutal_mlp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(brutal_mlp PRIVATE /W4 /permissive-)
else()
    target_compile_options(brutal_mlp PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS brutal_mlp
    EXPORT brutal_mlpTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/brutal_mlpConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/brutal_mlpConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/brutal_mlpConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brutal_mlp
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/brutal_mlpConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/brutal_mlpConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brutal_mlp
)

install(EXPORT brutal_mlpTargets
    NAMESPACE brutal_mlp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brutal_mlp
)

if(BRUTAL_MLP_BUILD_EXAMPLES)
    add_executable(brutal_mlp_xor examples/xor.cpp)
    target_link_libraries(brutal_mlp_xor PRIVATE brutal_mlp::brutal_mlp)
endif()

if(BRUTAL_MLP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BRUTAL_MLP_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
