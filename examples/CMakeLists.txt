add_executable(matrix matrix.cpp)
target_include_directories(
    matrix
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)