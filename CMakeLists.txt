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
option(BRUTAL_MLP_ENABLE_ISPC "Build the optional ISPC SIMD backend (inference and training kernels) when the ispc compiler is available" ON)
option(BRUTAL_MLP_REQUIRE_ISPC "Fail CMake configuration if the ISPC SIMD backend cannot be enabled" OFF)
set(BRUTAL_MLP_ISPC_TARGET "sse4.2-i32x4,avx2-i32x8,avx512skx-x16" CACHE STRING
    "ISPC targets of the optional SIMD backend; several comma-separated targets are dispatched at run time")

if(BRUTAL_MLP_REQUIRE_ISPC AND NOT BRUTAL_MLP_ENABLE_ISPC)
    message(FATAL_ERROR "BRUTAL_MLP_REQUIRE_ISPC=ON requires BRUTAL_MLP_ENABLE_ISPC=ON")
endif()

add_library(brutal_mlp
    src/config/config_factories.cpp
    src/config/config_validation.cpp
    src/core/checks.cpp
    src/core/dense_matrix.cpp
    src/core/enum_strings.cpp
    src/core/library_info.cpp
    src/core/parallel_workers.cpp
    src/core/random_utils.cpp
    src/core/scalar_type.cpp
    src/core/time_utils.cpp
    src/data/data_validation.cpp
    src/data/datasets.cpp
    src/data/mini_batch.cpp
    src/data/streaming_datasets.cpp
    src/evaluation/metrics.cpp
    src/inference/inference_model.cpp
    src/inference/inference_model_io.cpp
    src/inference/inference_workspace.cpp
    src/inference/layer_validation.cpp
    src/math/activations.cpp
    src/math/dense_batch.cpp
    src/math/dense_layer.cpp
    src/normalization/normalization.cpp
    src/serialization/binary_io.cpp
    src/serialization/binary_model_format.cpp
    src/serialization/enum_codes.cpp
    src/serialization/serialization_limits.cpp
    src/serialization/text_format.cpp
    src/training/layer_training_state.cpp
    src/training/learning_rate_schedule.cpp
    src/training/training_diagnostics.cpp
    src/training/training_fit.cpp
    src/training/training_fit_streaming.cpp
    src/training/training_model.cpp
    src/training/training_model_impl.cpp
    src/training/training_model_io.cpp
    src/training/training_monitor.cpp
    src/training/training_run_config.cpp
    src/training/weight_initialization.cpp
)
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
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(MSVC)
    target_compile_options(brutal_mlp PRIVATE /W4 /permissive-)
else()
    target_compile_options(brutal_mlp PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BRUTAL_MLP_ENABLE_ISPC AND NOT BRUTAL_MLP_USE_DOUBLE)
    find_program(BRUTAL_MLP_ISPC_EXECUTABLE NAMES ispc)
    if(BRUTAL_MLP_ISPC_EXECUTABLE)
        set(BRUTAL_MLP_ISPC_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/ispc")
        set(BRUTAL_MLP_ISPC_COMMON_FLAGS -O2)
        if(NOT WIN32)
            # Required when the object ends up in a shared library or a PIE executable.
            list(APPEND BRUTAL_MLP_ISPC_COMMON_FLAGS --pic)
        endif()

        # With several comma-separated targets ISPC writes one object per instruction set plus
        # a dispatch object that selects the best one supported by the CPU at run time.
        string(REPLACE "," ";" BRUTAL_MLP_ISPC_TARGET_LIST "${BRUTAL_MLP_ISPC_TARGET}")
        list(LENGTH BRUTAL_MLP_ISPC_TARGET_LIST BRUTAL_MLP_ISPC_TARGET_COUNT)
        set(BRUTAL_MLP_ISPC_TARGET_SUFFIXES "")
        if(BRUTAL_MLP_ISPC_TARGET_COUNT GREATER 1)
            foreach(ispc_target IN LISTS BRUTAL_MLP_ISPC_TARGET_LIST)
                string(REGEX REPLACE "-.*$" "" ispc_isa "${ispc_target}")
                string(REGEX REPLACE "\\.[0-9]+$" "" ispc_isa "${ispc_isa}")
                list(APPEND BRUTAL_MLP_ISPC_TARGET_SUFFIXES "_${ispc_isa}")
            endforeach()
        endif()

        # Compiles one .ispc file into <name>_ispc.h and its object file(s).
        function(brutal_mlp_add_ispc_kernel name source)
            set(header "${BRUTAL_MLP_ISPC_GENERATED_DIR}/${name}_ispc.h")
            set(object "${BRUTAL_MLP_ISPC_GENERATED_DIR}/${name}_ispc${CMAKE_CXX_OUTPUT_EXTENSION}")
            set(outputs "${header}" "${object}")
            foreach(suffix IN LISTS BRUTAL_MLP_ISPC_TARGET_SUFFIXES)
                list(APPEND outputs
                    "${BRUTAL_MLP_ISPC_GENERATED_DIR}/${name}_ispc${suffix}.h"
                    "${BRUTAL_MLP_ISPC_GENERATED_DIR}/${name}_ispc${suffix}${CMAKE_CXX_OUTPUT_EXTENSION}")
            endforeach()
            add_custom_command(
                OUTPUT ${outputs}
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${BRUTAL_MLP_ISPC_GENERATED_DIR}"
                COMMAND "${BRUTAL_MLP_ISPC_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
                        --target=${BRUTAL_MLP_ISPC_TARGET}
                        ${BRUTAL_MLP_ISPC_COMMON_FLAGS}
                        ${ARGN}
                        -h "${header}"
                        -o "${object}"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
                VERBATIM
            )
            add_custom_target(brutal_mlp_${name}_ispc DEPENDS ${outputs})
            add_dependencies(brutal_mlp brutal_mlp_${name}_ispc)
            set(objects "")
            foreach(output IN LISTS outputs)
                if(output MATCHES "${CMAKE_CXX_OUTPUT_EXTENSION}$")
                    list(APPEND objects "${output}")
                endif()
            endforeach()
            set_source_files_properties(${objects} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
            target_sources(brutal_mlp PRIVATE ${objects})
        endfunction()

        # Inference kernel: fused multiply-add allowed (results differ from the scalar path only
        # by rounding).
        brutal_mlp_add_ispc_kernel(dense_layer_kernel src/math/dense_layer_kernel.ispc)
        # Training kernels: no fused multiply-add, so every value is rounded exactly like the
        # C++ kernels and training stays bit-identical with and without ISPC.
        brutal_mlp_add_ispc_kernel(dense_batch_kernels src/math/dense_batch_kernels.ispc --opt=disable-fma)

        target_include_directories(brutal_mlp PRIVATE "${BRUTAL_MLP_ISPC_GENERATED_DIR}")
        target_compile_definitions(brutal_mlp PRIVATE BRUTAL_MLP_HAS_ISPC=1)
        message(STATUS "brutal_mlp: ISPC SIMD backend enabled (${BRUTAL_MLP_ISPC_TARGET})")
    else()
        if(BRUTAL_MLP_REQUIRE_ISPC)
            message(FATAL_ERROR "BRUTAL_MLP_REQUIRE_ISPC=ON but the ispc compiler was not found")
        else()
            message(STATUS "brutal_mlp: ispc compiler not found; using the C++ kernels")
        endif()
    endif()
elseif(BRUTAL_MLP_ENABLE_ISPC AND BRUTAL_MLP_USE_DOUBLE)
    if(BRUTAL_MLP_REQUIRE_ISPC)
        message(FATAL_ERROR "BRUTAL_MLP_REQUIRE_ISPC=ON is incompatible with BRUTAL_MLP_USE_DOUBLE=ON")
    else()
        message(STATUS "brutal_mlp: ISPC backend is float-only; using scalar backend for double precision")
    endif()
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
