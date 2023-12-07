cmake_minimum_required(VERSION 3.16)

project(Nonlinear_Optimization)

set(LIBRARY_NAME "Optimization")

add_subdirectory(src)

option(BUILD_EXAMPLES "Whether to build examples" OFF)
if (BUILD_EXAMPLES)
    message(STATUS "BUILD_EXAMPLES ON")
    add_subdirectory(examples)
endif()
